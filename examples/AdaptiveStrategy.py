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

class LatentActorCritic(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU()
        )
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, latent_state: Tensor) -> tuple[Categorical, Tensor]:
        x = self.shared_net(latent_state)
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
    stoch_dim: int = 32
    deter_dim: int = 256
    wm_hidden_dim: int = 256
    imagine_horizon: int = 15

    wm_learning_rate: float = 1e-4
    ac_learning_rate: float = 3e-5
    kl_loss_weight: float = 0.1
    reconstruction_loss_weight: float = 1.0

    max_densification_subset: int = 100_000

    reward_patch_radius: int = 8
    reward_delay: int = 400
    gauss_count_penalty: float = 0.001
    graph_encoder: Any = field(default=None, repr=False)
    rssm: Any = field(default=None, repr=False)
    actor_critic: Any = field(default=None, repr=False)
    wm_optimizer: Any = field(default=None, repr=False)
    ac_optimizer: Any = field(default=None, repr=False)
    lpips_metric: Any = field(default=None, repr=False)

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
        self.rssm = RSSM(self.gnn_output_dim, 3, self.stoch_dim, self.deter_dim, self.wm_hidden_dim).to(device)
        self.actor_critic = LatentActorCritic(self.stoch_dim + self.deter_dim, self.wm_hidden_dim, 3).to(device)

        wm_params = list(self.graph_encoder.parameters()) + list(self.rssm.parameters())
        self.wm_optimizer = torch.optim.AdamW(wm_params, lr=self.wm_learning_rate)
        self.ac_optimizer = torch.optim.AdamW(self.actor_critic.parameters(), lr=self.ac_learning_rate)

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

        if self.rssm is None:
            self._initialize_learning_components(params["means"].device)

        if state.get("age") is None:
            state["age"] = torch.zeros(params["means"].shape[0], dtype=torch.int32, device=params["means"].device)
        if state.get("rssm_state") is None:
            state["rssm_state"] = self.rssm.get_initial_state(1, params["means"].device)

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

        scene_features, scene_encoding = self._get_features_from_graph(params, state, candidate_mask)

        latent_state_flat = torch.cat([state["rssm_state"].stoch, state["rssm_state"].deter], -1)
        action_dist, _ = self.actor_critic(latent_state_flat.detach())
        action_choice = action_dist.sample().item()

        self._queue_rl_experience(state, scene_encoding, action_choice, action_dist.log_prob(torch.tensor(action_choice).to(device)))

        state_to_modify = {k: v for k, v in state.items() if k in ["grad2d", "count", "radii", "age"]}
        n_split, n_duplicate = 0, 0
        if action_choice == 1: # split
            n_split = candidate_mask.sum().item()
            split(params, optimizers, state_to_modify, candidate_mask)
        elif action_choice == 2: # duplicate
            n_duplicate = candidate_mask.sum().item()
            duplicate(params, optimizers, state_to_modify, candidate_mask)

        if n_split > 0 or n_duplicate > 0:
            num_new = (n_split * 2) + n_duplicate - (n_split + n_duplicate)
            state_to_modify["age"][-num_new:] = 0
        state.update(state_to_modify)

        action_one_hot = F.one_hot(torch.tensor([action_choice]), num_classes=3).float().to(device)
        # action_one_hot = action_one_hot.squeeze(0)

        post_state, _ = self.rssm.observe(scene_encoding.detach(), action_one_hot, state["rssm_state"])
        state["rssm_state"] = post_state
        return n_split, n_duplicate




    def _train_world_model_and_agent(self, state: dict):
        replay_buffer = state["replay_buffer"]
        if len(replay_buffer) < replay_buffer.batch_size:
            return

        device = next(self.actor_critic.parameters()).device
        batch = state["replay_buffer"].sample().to(device)
        scene_embeds = batch["scene_embed"]
        actions = batch["action"]
        rewards = batch["reward"]

        prev_state = self.rssm.get_initial_state(scene_embeds.shape[0], device)
        post_states, prior_states = self.rssm.observe(scene_embeds[:, 0], actions[:, 0], prev_state)

        reward_pred = self.rssm.reward_predictor(torch.cat([post_states.stoch, post_states.deter], -1))
        reward_loss = F.mse_loss(reward_pred.squeeze(), rewards[:, 0])

        kl_loss = kl_divergence(
            Independent(Normal(post_states.mean, post_states.std), 1),
            Independent(Normal(prior_states.mean, prior_states.std), 1)
        ).mean()

        world_model_loss = self.kl_loss_weight * kl_loss + reward_loss

        self.wm_optimizer.zero_grad()
        world_model_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.graph_encoder.parameters()) + list(self.rssm.parameters()), 100.0)
        self.wm_optimizer.step()

        initial_imagine_state = RSSMState(post_states.mean.detach(), post_states.std.detach(), post_states.stoch.detach(), post_states.deter.detach())

        imagined_rewards = []
        imagined_latents = []
        imagined_log_probs = []

        current_state = initial_imagine_state
        for _ in range(self.imagine_horizon):
            latent_flat = torch.cat([current_state.stoch, current_state.deter], -1)
            action_dist, _ = self.actor_critic(latent_flat)
            action = action_dist.sample()
            imagined_log_probs.append(action_dist.log_prob(action))

            # Predict next state and reward in imagination
            current_state = self.rssm.imagine_step(current_state, F.one_hot(action, num_classes=3).float())
            imagined_latents.append(torch.cat([current_state.stoch, current_state.deter], -1))
            imagined_rewards.append(self.rssm.reward_predictor(torch.cat([current_state.stoch, current_state.deter], -1)))

        # Calculate imagined returns (value estimates) using the critic
        imagined_rewards = torch.stack(imagined_rewards).squeeze(-1)
        imagined_latents = torch.stack(imagined_latents)
        imagined_log_probs = torch.stack(imagined_log_probs)

        _, values = self.actor_critic(imagined_latents)
        lambda_returns = self._compute_lambda_returns(imagined_rewards, values)

        # Actor loss (policy gradient)
        actor_loss = -(imagined_log_probs * lambda_returns.detach()).mean()

        # Critic loss
        critic_loss = F.mse_loss(values, lambda_returns)

        ac_loss = actor_loss + critic_loss
        self.ac_optimizer.zero_grad()
        ac_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 100.0)
        self.ac_optimizer.step()

        self.writer.add_scalar("agent/wm_loss", world_model_loss.item(), state["step"])
        self.writer.add_scalar("agent/ac_loss", ac_loss.item(), state["step"])
        self.writer.add_scalar("agent/reward_loss", reward_loss.item(), state["step"])
        self.writer.add_scalar("agent/kl_loss", kl_loss.item(), state["step"])

    @staticmethod
    def _compute_lambda_returns(rewards: Tensor, values: Tensor, gamma=0.99, lambda_=0.95) -> Tensor:
        returns = torch.zeros_like(rewards)
        last_val = values[-1]
        for t in reversed(range(rewards.shape[0])):
            last_val = rewards[t] + gamma * (1 - lambda_) * values[t] + gamma * lambda_ * last_val
            returns[t] = last_val
        return returns


    @torch.no_grad()
    def _queue_rl_experience(self, state: dict, scene_embed: Tensor, action: int, log_prob: Tensor):
        initial_error = state["l1_loss_map"].mean()

        experience = {
            "step": state["step"],
            "scene_embed": scene_embed.detach(),
            "action": torch.tensor(action, device=scene_embed.device),
            "log_prob": log_prob.detach(),
            "initial_error": initial_error,
            "initial_gauss_count": state["age"].shape[0]
        }
        state["reward_queue"].append(experience)


    def _process_rewards(self, params: dict, state: dict, current_step: int):
        while state["reward_queue"] and (current_step - state["reward_queue"][0]["step"]) >= self.reward_delay:
            exp = state["reward_queue"].popleft()

            current_error = state["l1_loss_map"].mean()
            extrinsic_reward = (exp["initial_error"] - current_error) * 10.0

            gauss_count_now = params["means"].shape[0]
            penalty = self.gauss_count_penalty * max(0, gauss_count_now - exp["initial_gauss_count"])

            reward = (extrinsic_reward - penalty).clamp(-1.0, 1.0)

            if len(state["replay_buffer"]) < state["replay_buffer"]._storage.max_size:
                td = TensorDict({
                    "scene_embed": exp["scene_embed"],
                    "action": exp["action"],
                    "reward": reward,
                }, batch_size=[])
                state["replay_buffer"].add(td)

    @torch.no_grad()
    def _get_features_from_graph(self, params: dict, state: dict, subset_mask: Tensor) -> tuple[Tensor, Tensor]:
        device = params["means"].device
        all_indices = torch.where(subset_mask)[0]
        if all_indices.numel() == 0:
            return torch.zeros(0, self.gnn_output_dim, device=device), torch.zeros(self.gnn_output_dim, device=device)

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
        edge_index = torch.stack([source_nodes, target_nodes], dim=0)
        edge_attr = dists.flatten().unsqueeze(-1) / state["scene_scale"]

        encoded_features = self.graph_encoder(node_features, edge_index, edge_attr=edge_attr)

        scene_encoding = encoded_features.mean(dim=0, keepdim=True)

        return encoded_features, scene_encoding




    def _update_quality_map(self, params: dict, state: dict, info: dict):
        if info.get("camtoworlds") is None: return
        width, height = info['width'], info['height']
        view_proj_matrix, _, _ = create_view_proj_matrix(info["camtoworlds"][0], info["Ks"][0], width, height)
        state["view_proj_matrix"] = view_proj_matrix