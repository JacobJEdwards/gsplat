from collections import deque
from dataclasses import dataclass, field
from typing import Any, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Categorical
from torch.optim import Optimizer

from ops import duplicate, split
from utils import knn_with_ids, create_view_proj_matrix
from gsplat.strategy.default import DefaultStrategy
from gsplat.strategy.ops import remove, reset_opa
from tensordict import TensorDict
from torchrl.data import RandomSampler, TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyMemmapStorage


class SimpleActorCritic(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
        )
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        x = self.shared_net(features)
        action_logits = self.actor_head(x)
        value = self.critic_head(x).squeeze(-1)
        return action_logits, value

@dataclass
class AdaptiveStrategy(DefaultStrategy):
    refine_every: int = 400
    learn_every: int = 200
    imitation_steps: int = 1_000

    feature_dim: int = 7
    hidden_dim: int = 64
    learning_rate: float = 1e-4
    ppo_clip_epsilon: float = 0.2
    entropy_loss_weight: float = 0.01

    reward_patch_radius: int = 4
    reward_delay: int = 200
    max_densification_subset: int = 50_000

    prune_ac_net: Any = field(default=None, repr=False)
    prune_ac_optimizer: Any = field(default=None, repr=False)

    grow_ac_net: Any = field(default=None, repr=False)
    grow_ac_optimizer: Any = field(default=None, repr=False)

    writer: Any = field(default=None, repr=False)

    def initialize_state(self, scene_scale: float = 1.0) -> dict[str, Any]:
        state = super().initialize_state(scene_scale)

        state.update({
            "age": None,
            "l1_loss_map": None,
            "detail_error_map": None,
            "view_proj_matrix": None,
            "prune_replay_buffer": TensorDictReplayBuffer(
                storage=LazyMemmapStorage(max_size=30_000), sampler=RandomSampler(), batch_size=512
            ),
            "prune_reward_queue": deque(maxlen=10_000),
            "grow_replay_buffer": TensorDictReplayBuffer(
                storage=LazyMemmapStorage(max_size=30_000), sampler=RandomSampler(), batch_size=512
            ),
            "grow_reward_queue": deque(maxlen=10_000),
        })
        return state

    def _initialize_learning_components(self, device: torch.device) -> None:
        self.prune_ac_net = SimpleActorCritic(self.feature_dim, self.hidden_dim, 2).to(device)
        self.prune_ac_optimizer = torch.optim.AdamW(self.prune_ac_net.parameters(), lr=self.learning_rate)

        self.grow_ac_net = SimpleActorCritic(self.feature_dim, self.hidden_dim, 3).to(device)
        self.grow_ac_optimizer = torch.optim.AdamW(self.grow_ac_net.parameters(), lr=self.learning_rate)

        print("Initialized separate Pruning and Growing Actor-Critic networks.")

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
        is_imitation_phase = step < self.imitation_steps

        if self.prune_ac_net is None:
            self._initialize_learning_components(params["means"].device)
        if state.get("age") is None:
            state["age"] = torch.zeros(params["means"].shape[0], dtype=torch.int32, device=params["means"].device)

        state["age"] += 1
        state["l1_loss_map"] = info.get("l1_loss_map")
        state["pixels"] = info.get("pixels")
        state["gaussian_contribution"] = info.get("gaussian_contribution")

        self._update_quality_map(params, state, info)
        self._update_state(params, state, info, packed=packed)

        if not is_imitation_phase:
            self._process_rewards(state, step)

        if step > self.refine_start_iter and step % self.refine_every == 0:
            n_prune = self.prune_gs(params, optimizers, state, is_imitation_phase)
            n_split, n_duplicate = self.grow_gs(params, optimizers, state, is_imitation_phase)
            if self.verbose:
                print(f"Step {step} ({'Imitation' if is_imitation_phase else 'RL'}): Pruned {n_prune}, Split {n_split}, Duplicated {n_duplicate}.")

        if step > self.refine_start_iter and step % self.learn_every == 0:
            if is_imitation_phase:
                self._train_imitation_agent(state, agent_type="prune")
                self._train_imitation_agent(state, agent_type="grow")
            else:
                self._train_rl_agent(state, agent_type="prune")
                self._train_rl_agent(state, agent_type="grow")

        if step % self.reset_every == 0 and step > 0:
            reset_opa(params, optimizers, state, self.prune_opa * 2.0)

    @torch.no_grad()
    def prune_gs(self, params: dict, optimizers: dict, state: dict, is_imitation_phase: bool) -> int:
        device = params["means"].device
        opacities = torch.sigmoid(params["opacities"].flatten())
        is_too_transparent = opacities < self.prune_opa
        is_too_large = torch.exp(params["scales"]).max(dim=-1).values > self.prune_scale3d * state["scene_scale"]

        candidate_mask = is_too_transparent | is_too_large

        if candidate_mask.sum() == 0:
            return 0

        original_indices = torch.where(candidate_mask)[0]
        features = self._get_simplified_features(params, state, candidate_mask)

        self.prune_ac_net.train(is_imitation_phase)
        action_logits, _ = self.prune_ac_net(features)

        if is_imitation_phase:
            labels = torch.ones(len(original_indices), dtype=torch.long, device=device)
            for i in range(len(original_indices)):
                experience = TensorDict({
                    "features": features[i],
                    "action": labels[i]
                }, batch_size=[])
                state["prune_replay_buffer"].add(experience)
            actions = labels
        else:
            action_dist = Categorical(logits=action_logits)
            actions = action_dist.sample()
            self._queue_rl_experience(state, features, actions, action_dist.log_prob(actions), original_indices, "prune")

        prune_mask_agent = (actions == 1)
        global_prune_mask = torch.zeros_like(candidate_mask)
        global_prune_mask[original_indices[prune_mask_agent]] = True

        n_prune = global_prune_mask.sum().item()
        if n_prune > 0:
            state_to_modify = {k: v for k, v in state.items() if k in ["grad2d", "count", "radii", "age", "gaussian_contribution"]}
            remove(params, optimizers, state_to_modify, global_prune_mask)
            state.update(state_to_modify)

        return n_prune

    @torch.no_grad()
    def grow_gs(self, params: dict, optimizers: dict, state: dict, is_imitation_phase: bool) -> Tuple[int, int]:
        device = params["means"].device
        normalized_grads = state["grad2d"] / state["count"].clamp_min(1.0)
        candidate_mask = normalized_grads > self.grow_grad2d

        if candidate_mask.sum() > self.max_densification_subset:
            candidate_indices = torch.where(candidate_mask)[0]
            rand_indices = torch.randperm(len(candidate_indices), device=device)[:self.max_densification_subset]
            candidate_mask.fill_(False)
            candidate_mask[candidate_indices[rand_indices]] = True

        if candidate_mask.sum() == 0:
            return 0, 0

        original_indices = torch.where(candidate_mask)[0]
        features = self._get_simplified_features(params, state, candidate_mask)

        self.grow_ac_net.train(is_imitation_phase)
        action_logits, _ = self.grow_ac_net(features)

        if is_imitation_phase:
            scales = torch.exp(params["scales"][original_indices])
            is_large_mask = scales.max(dim=-1).values > self.grow_scale3d * state["scene_scale"]
            labels = torch.ones(len(original_indices), dtype=torch.long, device=device)
            labels[~is_large_mask] = 2
            for i in range(len(original_indices)):
                experience = TensorDict({
                    "features": features[i],
                    "action": labels[i]
                }, batch_size=[])
                state["grow_replay_buffer"].add(experience)
            actions = labels
        else:
            action_dist = Categorical(logits=action_logits)
            actions = action_dist.sample()
            self._queue_rl_experience(state, features, actions, action_dist.log_prob(actions), original_indices, "grow")

        state_to_modify = {k: v for k, v in state.items() if k in ["grad2d", "count", "radii", "age", "gaussian_contribution"]}
        split_mask = (actions == 1)
        duplicate_mask = (actions == 2)

        n_split = 0
        if split_mask.any():
            global_split_mask = torch.zeros(params["means"].shape[0], dtype=torch.bool, device=device)
            global_split_mask[original_indices[split_mask]] = True
            n_split = global_split_mask.sum().item()
            split(params, optimizers, state_to_modify, global_split_mask)

        n_duplicate = 0
        if duplicate_mask.any():
            global_duplicate_mask = torch.zeros(params["means"].shape[0], dtype=torch.bool, device=device)
            global_duplicate_mask[original_indices[duplicate_mask]] = True
            n_duplicate = global_duplicate_mask.sum().item()
            duplicate(params, optimizers, state_to_modify, global_duplicate_mask)

        if n_split > 0 or n_duplicate > 0:
            total_new = (n_split * 2) + n_duplicate
            state_to_modify["age"][-total_new:] = 0

        state.update(state_to_modify)
        return n_split, n_duplicate

    def _train_imitation_agent(self, state: dict, agent_type: str):
        replay_buffer = state[f"{agent_type}_replay_buffer"]
        if len(replay_buffer) < replay_buffer.batch_size:
            return

        agent_net = getattr(self, f"{agent_type}_ac_net")
        optimizer = getattr(self, f"{agent_type}_ac_optimizer")
        device = next(agent_net.parameters()).device

        batch = replay_buffer.sample().to(device)
        features = batch.get("features")
        labels = batch.get("action").squeeze(-1)

        action_logits, _ = agent_net(features)
        loss = F.cross_entropy(action_logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.writer.add_scalar(f"agent/{agent_type}_imitation_loss", loss.item(), state.get("step", -1))

    def _train_rl_agent(self, state: dict, agent_type: str):
        replay_buffer = state[f"{agent_type}_replay_buffer"]
        if len(replay_buffer) < replay_buffer.batch_size:
            return

        agent_net = getattr(self, f"{agent_type}_ac_net")
        optimizer = getattr(self, f"{agent_type}_ac_optimizer")
        device = next(agent_net.parameters()).device

        batch = replay_buffer.sample().to(device)
        features = batch.get("features")
        actions = batch.get("action").squeeze(-1)
        rewards = batch.get("reward").squeeze(-1)
        old_log_probs = batch.get("log_prob").squeeze(-1)

        new_logits, values = agent_net(features)
        values = values.squeeze(-1)

        advantage = rewards - values.detach()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        new_dist = Categorical(logits=new_logits)
        new_log_probs = new_dist.log_prob(actions)
        ratio = torch.exp(new_log_probs - old_log_probs)

        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip_epsilon, 1.0 + self.ppo_clip_epsilon) * advantage
        actor_loss = -torch.min(surr1, surr2).mean()

        critic_loss = F.mse_loss(values, rewards)
        entropy_loss = -new_dist.entropy().mean()
        loss = actor_loss + 0.5 * critic_loss + self.entropy_loss_weight * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.writer.add_scalar(f"agent/{agent_type}_rl_loss", loss.item(), state.get("step",-1))
        self.writer.add_scalar(f"agent/{agent_type}_actor_loss", actor_loss.item(), state.get("step",-1))
        self.writer.add_scalar(f"agent/{agent_type}_critic_loss", critic_loss.item(), state.get("step",-1))
        self.writer.add_scalar(f"agent/{agent_type}_mean_reward", rewards.mean().item(), state.get("step",-1))

    @torch.no_grad()
    def _queue_rl_experience(self, state: dict, features: Tensor, actions: Tensor, log_probs: Tensor, indices: Tensor, agent_type: str):
        if state.get("l1_loss_map") is None: return

        l1_loss_map = state["l1_loss_map"].squeeze()
        h, w = l1_loss_map.shape
        means_h = F.pad(state["params_for_features"]["means"][indices], (0, 1), value=1.0)
        p_hom = means_h @ state["view_proj_matrix"]
        p_w = 1.0 / (p_hom[:, 3].clamp_min(1e-6))
        p_proj = p_hom[:, :2] * p_w[:, None]

        pixel_x = torch.clamp(((p_proj[:, 0] * 0.5 + 0.5) * w), 0, w - 1).long()
        pixel_y = torch.clamp(((p_proj[:, 1] * 0.5 + 0.5) * h), 0, h - 1).long()

        r = self.reward_patch_radius
        for i in range(len(indices)):
            y, x = pixel_y[i], pixel_x[i]
            y_min, y_max = max(0, y - r), min(h, y + r + 1)
            x_min, x_max = max(0, x - r), min(w, x + r + 1)

            initial_error_patch = l1_loss_map[y_min:y_max, x_min:x_max]
            initial_patch_error = initial_error_patch.mean() if initial_error_patch.numel() > 0 else torch.tensor(0.0)

            experience = {
                "step": state["step"],
                "features": features[i].detach(),
                "action": actions[i].detach(),
                "log_prob": log_probs[i].detach(),
                "pixel_x": x, "pixel_y": y,
                "initial_patch_error": initial_patch_error,
            }
            state[f"{agent_type}_reward_queue"].append(experience)

    def _process_rewards(self, state: dict, current_step: int):
        for agent_type in ["prune", "grow"]:
            reward_queue = state[f"{agent_type}_reward_queue"]
            replay_buffer = state[f"{agent_type}_replay_buffer"]

            while reward_queue and (current_step - reward_queue[0]["step"]) >= self.reward_delay:
                exp = reward_queue.popleft()
                if state.get("l1_loss_map") is None: continue

                l1_loss_map = state["l1_loss_map"].squeeze()
                h, w = l1_loss_map.shape
                y, x = exp["pixel_y"], exp["pixel_x"]
                r = self.reward_patch_radius
                y_min, y_max = max(0, y - r), min(h, y + r + 1)
                x_min, x_max = max(0, x - r), min(w, x + r + 1)

                new_error_patch = l1_loss_map[y_min:y_max, x_min:x_max]
                new_patch_error = new_error_patch.mean() if new_error_patch.numel() > 0 else torch.tensor(0.0)
                reward = (exp["initial_patch_error"] - new_patch_error).clamp(-1.0, 1.0)

                if len(replay_buffer) < replay_buffer._storage.max_size:
                    td = TensorDict({
                        "features": exp["features"], "action": exp["action"],
                        "log_prob": exp["log_prob"], "reward": reward,
                    }, batch_size=[])
                    replay_buffer.add(td)

    @torch.no_grad()
    def _get_simplified_features(self, params: dict, state: dict, subset_mask: Tensor) -> Tensor:
        n_subset = subset_mask.sum().item()
        if n_subset == 0:
            return torch.zeros(0, self.feature_dim, device=params["means"].device)

        device = params["means"].device
        features = torch.zeros(n_subset, self.feature_dim, device=device)

        state["params_for_features"] = params

        grads2d = state["grad2d"][subset_mask]
        counts = state["count"][subset_mask].clamp_min(1)
        features[:, 0] = (grads2d / counts) / self.grow_grad2d
        scales = params["scales"][subset_mask]
        features[:, 1] = torch.log(scales.max(dim=-1).values / state["scene_scale"])
        features[:, 2] = params["opacities"][subset_mask].flatten()
        features[:, 3] = state["age"][subset_mask] / 1000.0

        if n_subset > 5:
            means3d_subset = params["means"][subset_mask]
            dists, _ = knn_with_ids(means3d_subset, K=5 + 1)
            features[:, 4] = dists[:, 1:].mean(dim=-1) / state["scene_scale"]

        gt_pixels_full = state["pixels"]
        gt_pixels = gt_pixels_full.squeeze(0) if gt_pixels_full is not None and gt_pixels_full.dim() == 4 else gt_pixels_full

        if gt_pixels is not None:
            h, w, _ = gt_pixels.shape
            means_subset = params["means"][subset_mask]
            means_h = F.pad(means_subset, (0, 1), value=1.0)
            p_hom = means_h @ state["view_proj_matrix"]
            p_w = 1.0 / (p_hom[:, 3].clamp_min(1e-6))
            p_proj = p_hom[:, :2] * p_w[:, None]

            pixel_x = torch.clamp(((p_proj[:, 0] * 0.5 + 0.5) * w), 0, w - 1).long()
            pixel_y = torch.clamp(((p_proj[:, 1] * 0.5 + 0.5) * h), 0, h - 1).long()

            r = self.reward_patch_radius
            patch_complexities = torch.zeros(n_subset, device=device)
            for i in range(n_subset):
                y, x = pixel_y[i], pixel_x[i]
                y_min, y_max = max(0, y - r), min(h, y + r + 1)
                x_min, x_max = max(0, x - r), min(w, x + r + 1)
                patch = gt_pixels[y_min:y_max, x_min:x_max]
                if patch.numel() > 0:
                    patch_complexities[i] = patch.mean(dim=-1).std()

            features[:, 5] = patch_complexities

        contribution = state.get("gaussian_contribution")
        if contribution is not None:
            features[:, 6] = contribution[subset_mask]

        return torch.nan_to_num(features, 0.0)
    def _update_quality_map(self, params: dict, state: dict, info: dict):
        if info.get("camtoworlds") is None: return
        width, height = info['width'], info['height']
        view_proj_matrix, _, _ = create_view_proj_matrix(info["camtoworlds"][0], info["Ks"][0], width, height)
        state["view_proj_matrix"] = view_proj_matrix