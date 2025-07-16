from collections import deque
from dataclasses import dataclass, field
from typing import Any, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Categorical, Independent, kl_divergence, Normal

from ops import duplicate, split
from utils import knn_with_ids, create_view_proj_matrix
from gsplat.strategy.default import DefaultStrategy
from gsplat.strategy.ops import remove, reset_opa
from tensordict import TensorDict
from torchrl.data import RandomSampler, TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyMemmapStorage
import torch_geometric.nn as gnn

@dataclass
class RSSMState:
    mean: Tensor
    std: Tensor
    stoch: Tensor
    deter: Tensor

class GraphEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.conv1 = gnn.GATv2Conv(input_dim, hidden_dim, heads=4, concat=True, edge_dim=1)
        self.conv2 = gnn.GATv2Conv(hidden_dim * 4, hidden_dim, heads=4, concat=True, edge_dim=1)
        self.output_head = nn.Linear(hidden_dim * 4, output_dim)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        x = F.elu(self.conv1(x, edge_index, edge_attr=edge_attr))
        x = F.elu(self.conv2(x, edge_index, edge_attr=edge_attr))
        return self.output_head(x)

class RSSM(nn.Module):
    def __init__(self, scene_embed_dim, action_dim, stoch_dim=32, deter_dim=256, hidden_dim=256):
        super().__init__()
        self.stoch_dim = stoch_dim
        self.deter_dim = deter_dim
        self.action_dim = action_dim

        self.representation_model = nn.Sequential(
            nn.Linear(scene_embed_dim + deter_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, stoch_dim * 2)
        )

        self.rnn = nn.GRUCell(stoch_dim + action_dim, deter_dim)

        self.transition_model = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, stoch_dim * 2)
        )

        self.reward_predictor = nn.Sequential(
            nn.Linear(stoch_dim + deter_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, 1)
        )
        self.continue_predictor = nn.Sequential(
            nn.Linear(stoch_dim + deter_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, 1)
        )

    def get_initial_state(self, batch_size: int, device: torch.device) -> RSSMState:
        return RSSMState(
            mean=torch.zeros(batch_size, self.stoch_dim, device=device),
            std=torch.ones(batch_size, self.stoch_dim, device=device),
            stoch=torch.zeros(batch_size, self.stoch_dim, device=device),
            deter=torch.zeros(batch_size, self.deter_dim, device=device),
        )

    def observe(self, scene_embed: Tensor, action: Tensor, prev_state: RSSMState) -> Tuple[RSSMState, RSSMState]:
        prior_state = self.imagine_step(prev_state, action)

        x = torch.cat([prev_state.deter, scene_embed], -1)
        mean, std = self.representation_model(x).chunk(2, dim=-1)
        std = F.softplus(std) + 0.1
        stoch = mean + std * torch.randn_like(mean)
        deter = self.rnn(torch.cat([stoch, action], -1), prev_state.deter)
        posterior_state = RSSMState(mean, std, stoch, deter)

        return posterior_state, prior_state

    def imagine_step(self, prev_state: RSSMState, action: Tensor) -> RSSMState:
        x = self.transition_model(prev_state.deter)
        mean, std = x.chunk(2, dim=-1)
        std = F.softplus(std) + 0.1
        stoch = mean + std * torch.randn_like(mean)
        deter = self.rnn(torch.cat([stoch, action], -1), prev_state.deter)
        return RSSMState(mean, std, stoch, deter)

class PerNodeActorCritic(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU()
        )
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, per_gaussian_features: Tensor) -> tuple[Categorical, Tensor]:
        x = self.shared_net(per_gaussian_features)
        action_logits = self.actor_head(x)
        action_dist = Categorical(logits=action_logits)
        value = self.critic_head(x).squeeze(-1)
        return action_dist, value


@dataclass
class AdaptiveStrategy(DefaultStrategy):
    refine_every: int = 400
    learn_every: int = 100
    refine_start_iter: int = 1000

    prune_min_age: int = 1000
    prune_significance_thresh: float = 0.01

    gnn_input_dim: int = 11
    gnn_hidden_dim: int = 64
    gnn_output_dim: int = 128
    ac_hidden_dim: int = 128

    learning_rate: float = 3e-4
    ppo_clip_epsilon: float = 0.2
    entropy_loss_weight: float = 0.01
    critic_loss_weight: float = 0.5

    reward_delay: int = 400
    gauss_count_penalty_factor: float = 0.01

    max_densification_subset: int = 100_000

    graph_encoder: Any = field(default=None, repr=False)
    actor_critic: Any = field(default=None, repr=False)
    optimizer: Any = field(default=None, repr=False)
    writer: Any = field(default=None, repr=False)

    def initialize_state(self, scene_scale: float = 1.0) -> dict[str, Any]:
        state = super().initialize_state(scene_scale)
        state.update({
            "age": None,
            "rssm_state": None,
            "replay_buffer": TensorDictReplayBuffer(
                storage=LazyMemmapStorage(max_size=50_000), sampler=RandomSampler(), batch_size=64,
            ),
            "reward_queue": deque(maxlen=20_000),
            "l1_loss_map": None,
            "detail_error_map": None,
        })

        return state

    def _initialize_learning_components(self, device: torch.device) -> None:
        self.graph_encoder = GraphEncoder(self.gnn_input_dim, self.gnn_hidden_dim, self.gnn_output_dim).to(device)
        self.actor_critic = PerNodeActorCritic(self.gnn_output_dim, self.ac_hidden_dim, 3).to(device)

        all_params = list(self.graph_encoder.parameters()) + list(self.actor_critic.parameters())
        self.optimizer = torch.optim.AdamW(all_params, lr=self.learning_rate)

        if self.verbose:
            print("Initialized densification agent (GNN + World Model).")


    def step_post_backward(
            self,
            params: dict[str, torch.nn.Parameter] | torch.nn.ParameterDict,
            optimizers: dict[str, torch.optim.Optimizer],
            state: dict[str, Any],
            step: int,
            info: dict[str, Any],
            packed: bool = False,
    ) -> None:
        if step >= self.refine_stop_iter:
            return
        state["step"] = step

        if self.actor_critic is None:
            self._initialize_learning_components(params["means"].device)

        if state.get("age") is None:
            state["age"] = torch.zeros(params["means"].shape[0], dtype=torch.int32, device=params["means"].device)

        state["age"] += 1
        state["l1_loss_map"] = info.get("l1_loss_map", None)
        state["detail_error_map"] = info.get("detail_error_map", None)

        self._update_quality_map(params, state, info)
        self._process_rewards(params, state, step)
        self._update_state(params, state, info, packed=packed)

        if step > self.refine_start_iter and step % self.refine_every == 0:
            n_prune = self.prune_gs(params, optimizers, state)
            n_split, n_duplicate = self.grow_gs(params, optimizers, state)
            if self.verbose:
                print(f"Step {step}: Pruned {n_prune}, Split {n_split}, Duplicated {n_duplicate}.")

        if step > self.refine_start_iter and step % self.learn_every == 0:
            self._train_world_model_and_agent(state)

        if step > 0 and step % self.reset_every == 0:
            reset_opa(params, optimizers, state, self.prune_opa * 2.0)


    @torch.no_grad()
    def prune_gs(self, params: dict, optimizers: dict, state: dict) -> int:
        step = state.get("step", 0)

        is_prune_original = torch.sigmoid(params["opacities"].flatten()) < self.prune_opa
        if step > self.reset_every:
            is_too_big = (
                    torch.exp(params["scales"]).max(dim=-1).values
                    > self.prune_scale3d * state["scene_scale"]
            )
            if step < self.refine_scale2d_stop_iter:
                is_too_big |= state["radii"] > self.prune_scale2d
            is_prune_original |= is_too_big

        is_prune_significant = torch.zeros_like(is_prune_original)
        if "significance" in state and state["significance"] is not None and state["significance"].numel() > 0:
            if state["significance"].numel() == is_prune_original.numel():
                is_prune_significant = state["significance"] < self.prune_significance_thresh

        is_prune = is_prune_original | is_prune_significant

        no_prune = state["age"] < self.prune_min_age

        is_prune &= (~no_prune)

        n_prune = is_prune.sum().item()
        if n_prune > 0:
            per_gaussian_state_keys = ["grad2d", "count", "radii", "stagnation_count", "prev_grad2d", "prev_opacity",
                                       "significance", "age", "gaussian_contribution"]
            state_to_prune = {k: v for k, v in state.items() if k in per_gaussian_state_keys and v is not None}

            remove(params=params, optimizers=optimizers, state=state_to_prune, mask=is_prune)

            state.update(state_to_prune)

        return n_prune


    @torch.no_grad()
    def grow_gs(self, params: dict, optimizers: dict, state: dict) -> tuple[int, int]:
        device = params["means"].device
        normalized_grads = state["grad2d"] / state["count"].clamp_min(1.0)
        candidate_mask = normalized_grads > self.grow_grad2d

        num_candidates = candidate_mask.sum().item()
        if num_candidates > self.max_densification_subset:
            candidate_indices = torch.where(candidate_mask)[0]
            rand_indices = torch.randperm(num_candidates, device=device)[:self.max_densification_subset]

            new_mask = torch.zeros_like(candidate_mask)
            new_mask[candidate_indices[rand_indices]] = True
            candidate_mask = new_mask

        if candidate_mask.sum() == 0:
            return 0, 0

        original_indices = torch.where(candidate_mask)[0]

        per_gaussian_features = self._get_features_from_graph(params, state, candidate_mask)
        action_dist, values = self.actor_critic(per_gaussian_features.detach())
        actions = action_dist.sample()

        self._queue_per_node_experience(state, per_gaussian_features, actions, action_dist.log_prob(actions), values, original_indices)

        split_action_mask = (actions == 1)
        duplicate_action_mask = (actions == 2)

        global_split_mask = torch.zeros_like(candidate_mask)
        global_split_mask[original_indices[split_action_mask]] = True

        global_duplicate_mask = torch.zeros_like(candidate_mask)
        global_duplicate_mask[original_indices[duplicate_action_mask]] = True

        state_to_modify = {k: v for k, v in state.items() if k in ["grad2d", "count", "radii", "age"]}
        n_split = global_split_mask.sum().item()
        n_duplicate = global_duplicate_mask.sum().item()

        if n_split > 0:
            split(params, optimizers, state_to_modify, global_split_mask)
        if n_duplicate > 0:
            duplicate(params, optimizers, state_to_modify, global_duplicate_mask)

        if n_split > 0 or n_duplicate > 0:
            num_new = (n_split * 2) + n_duplicate - (n_split + n_duplicate)
            state_to_modify["age"][-num_new:] = 0

        state.update(state_to_modify)

        return n_split, n_duplicate




    def _train_world_model_and_agent(self, state: dict):
        if len(state["replay_buffer"]) < state["replay_buffer"].batch_size: return
        device = self.actor_critic.parameters().__next__().device

        batch = state["replay_buffer"].sample().to(device)
        features = batch.get("features")
        actions = batch.get("action").squeeze(-1)
        rewards = batch.get("reward").squeeze(-1)
        old_log_probs = batch.get("log_prob").squeeze(-1)
        old_values = batch.get("value").squeeze(-1)

        new_dist, new_values = self.actor_critic(features)

        advantage = rewards - old_values
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        new_log_probs = new_dist.log_prob(actions)
        ratio = torch.exp(new_log_probs - old_log_probs)

        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip_epsilon, 1.0 + self.ppo_clip_epsilon) * advantage
        actor_loss = -torch.min(surr1, surr2).mean()

        critic_loss = F.mse_loss(new_values, rewards)
        entropy_loss = -new_dist.entropy().mean()
        loss = actor_loss + self.critic_loss_weight * critic_loss + self.entropy_loss_weight * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.writer.add_scalar("agent/total_loss", loss.item(), state["step"])
        self.writer.add_scalar("agent/actor_loss", actor_loss.item(), state["step"])
        self.writer.add_scalar("agent/critic_loss", critic_loss.item(), state["step"])

    @staticmethod
    def _compute_lambda_returns(rewards: Tensor, values: Tensor, gamma=0.99, lambda_=0.95) -> Tensor:
        returns = torch.zeros_like(rewards)
        last_val = values[-1]
        for t in reversed(range(rewards.shape[0])):
            last_val = rewards[t] + gamma * (1 - lambda_) * values[t] + gamma * lambda_ * last_val
            returns[t] = last_val
        return returns


    @torch.no_grad()
    def _queue_per_node_experience(self, state: dict, features: Tensor, actions: Tensor, log_probs: Tensor, values: Tensor, indices: Tensor):
        initial_error = state["l1_loss_map"].mean()
        initial_gauss_count = state["age"].shape[0]
        for i in range(len(indices)):
            experience = {
                "step": state["step"],
                "features": features[i].detach(),
                "action": actions[i].detach(),
                "log_prob": log_probs[i].detach(),
                "value": values[i].detach(),
                "initial_error": initial_error,
                "initial_gauss_count": initial_gauss_count,
            }
            state["reward_queue"].append(experience)

    def _process_rewards(self, params: dict, state: dict, current_step: int):
        while state["reward_queue"] and (current_step - state["reward_queue"][0]["step"]) >= self.reward_delay:
            exp = state["reward_queue"].popleft()

            current_error = state["l1_loss_map"].mean()
            extrinsic_reward = (exp["initial_error"] - current_error) * 10.0

            gauss_count_now = params["means"].shape[0]
            penalty = self.gauss_count_penalty_factor * max(0, gauss_count_now - exp["initial_gauss_count"])

            reward = (extrinsic_reward - penalty).clamp(-2.0, 2.0)

            if len(state["replay_buffer"]) < state["replay_buffer"]._storage.max_size:
                td = TensorDict({
                    "features": exp["features"],
                    "action": exp["action"],
                    "log_prob": exp["log_prob"],
                    "value": exp["value"],
                    "reward": reward,
                }, batch_size=[])
                state["replay_buffer"].add(td)

    @torch.no_grad()
    def _get_features_from_graph(self, params: dict, state: dict, subset_mask: Tensor) -> Tensor:
        device = params["means"].device
        all_indices = torch.where(subset_mask)[0]
        if all_indices.numel() == 0:
            return torch.zeros(0, self.gnn_output_dim, device=device)

        means = params["means"][subset_mask]
        scales = torch.log(torch.exp(params["scales"][subset_mask]).mean(dim=-1, keepdim=True))
        opacities = params["opacities"][subset_mask].reshape(-1, 1)
        quats = params["quats"][subset_mask]
        ages = state["age"][subset_mask].unsqueeze(-1).float() / 1000.0
        grads2d = (state["grad2d"][subset_mask] / state["count"][subset_mask].clamp_min(1.0)).unsqueeze(-1)
        node_features = torch.cat([means, scales, opacities, quats, ages, grads2d], dim=-1)

        node_features = F.layer_norm(node_features, [node_features.shape[-1]])

        dists, nn_indices = knn_with_ids(means, K=16)
        source_nodes = torch.arange(means.shape[0], device=device).view(-1, 1).repeat(1, 16).flatten()
        target_nodes = nn_indices.flatten()
        edge_index = torch.stack([source_nodes, target_nodes])
        edge_attr = dists.flatten().unsqueeze(-1) / state["scene_scale"]

        encoded_features = self.graph_encoder(node_features, edge_index, edge_attr=edge_attr)

        # scene_encoding = encoded_features.mean(dim=0, keepdim=True)

        return encoded_features


    def _update_quality_map(self, params: dict, state: dict, info: dict):
        if info.get("camtoworlds") is None: return
        width, height = info['width'], info['height']
        view_proj_matrix, _, _ = create_view_proj_matrix(info["camtoworlds"][0], info["Ks"][0], width, height)
        state["view_proj_matrix"] = view_proj_matrix