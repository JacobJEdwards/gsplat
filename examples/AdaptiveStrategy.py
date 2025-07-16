from collections import deque
from dataclasses import dataclass, field
from typing import Any, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Categorical

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

    feature_dim: int = 7
    reward_patch_radius: int = 4
    hidden_dim: int = 64
    action_dim: int = 4  # 0: No-op, 1: Prune, 2: Split, 3: Duplicate
    learning_rate: float = 1e-4
    ppo_clip_epsilon: float = 0.15
    entropy_loss_weight: float = 0.015
    reward_delay: int = 200
    max_densification_subset: int = 100_000

    prune_opacity_threshold: float = 0.005
    split_opacity_reset: float = 0.01

    ac_net: Any = field(default=None, repr=False)
    ac_optimizer: Any = field(default=None, repr=False)
    writer: Any = field(default=None, repr=False)

    def initialize_state(self, scene_scale: float = 1.0) -> dict[str, Any]:
        state = super().initialize_state(scene_scale)
        replay_buffer = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(max_size=60_000),
            sampler=RandomSampler(),
            batch_size=512
        )
        reward_queue = deque(maxlen=20_000)

        state.update({
            "age": None,
            "l1_loss_map": None,
            "view_proj_matrix": None,
            "replay_buffer": replay_buffer,
            "reward_queue": reward_queue,
        })
        return state

    def _initialize_learning_components(self, device: torch.device) -> None:
        self.ac_net = SimpleActorCritic(self.feature_dim, self.hidden_dim, self.action_dim).to(device)
        self.ac_optimizer = torch.optim.AdamW(self.ac_net.parameters(), lr=self.learning_rate)
        print("Initialised Simple Actor-Critic network.")

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

        if self.ac_net is None:
            self._initialize_learning_components(params["means"].device)
        if state.get("age") is None:
            state["age"] = torch.zeros(params["means"].shape[0], dtype=torch.int32, device=params["means"].device)

        state["age"] += 1
        state["l1_loss_map"] = info.get("l1_loss_map")
        state["pixels"] = info.get("pixels")
        state["gaussian_contribution"] = info.get("gaussian_contribution")

        self._update_quality_map(params, state, info)
        self._update_state(params, state, info, packed=packed)
        self._process_rewards(state, step)

        if step > self.refine_start_iter and step % self.refine_every == 0:
            if state.get("pixels") is not None and state.get("gaussian_contribution") is not None:
                n_prune, n_split, n_duplicate = self._update_geometry(params, optimizers, state, step)
                if self.verbose:
                    print(f"Step {step}: Pruned {n_prune}, Split {n_split}, Duplicated {n_duplicate}. Now "
                          f"{len(params['means'])} Gs.")

        if step > self.refine_start_iter and step % self.learn_every == 0:
            self._train_agent(state)

        if step % self.reset_every == 0 and step > 0:
            reset_opa(params, optimizers, state, self.prune_opa * 2.0)


    @torch.no_grad()
    def _get_simplified_features(self, params: dict, state: dict, subset_mask: Tensor) -> Tensor:
        n_subset = subset_mask.sum().item()
        if n_subset == 0:
            return torch.zeros(0, self.feature_dim, device=params["means"].device)

        device = params["means"].device
        features = torch.zeros(n_subset, self.feature_dim, device=device)

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


    @torch.no_grad()
    def _update_geometry(self, params: dict, optimizers: dict, state: dict, step: int) -> tuple[int, int, int]:
        device = params["means"].device

        normalized_grads = state["grad2d"] / state["count"].clamp_min(1.0)
        candidate_mask = normalized_grads > self.grow_grad2d

        if candidate_mask.sum() == 0:
            return 0,0,0

        if candidate_mask.sum() > self.max_densification_subset:
            candidate_indices = torch.where(candidate_mask)[0]
            rand_indices = torch.randperm(len(candidate_indices), device=device)[:self.max_densification_subset]
            candidate_mask.fill_(False)
            candidate_mask[candidate_indices[rand_indices]] = True

        original_indices = torch.where(candidate_mask)[0]

        features = self._get_simplified_features(params, state, candidate_mask)
        self.ac_net.eval()
        action_logits, _ = self.ac_net(features)
        action_dist = Categorical(logits=action_logits)
        actions = action_dist.sample()
        log_probs = action_dist.log_prob(actions)

        l1_loss_map = state["l1_loss_map"].squeeze() if state["l1_loss_map"].dim() > 2 else state["l1_loss_map"]
        h, w = l1_loss_map.shape
        means_h = F.pad(params["means"][original_indices], (0, 1), value=1.0)
        p_hom = means_h @ state["view_proj_matrix"]
        p_w = 1.0 / (p_hom[:, 3].clamp_min(1e-6))
        p_proj = p_hom[:, :2] * p_w[:, None]

        pixel_x = torch.clamp(((p_proj[:, 0] * 0.5 + 0.5) * w), 0, w - 1).long()
        pixel_y = torch.clamp(((p_proj[:, 1] * 0.5 + 0.5) * h), 0, h - 1).long()

        r = self.reward_patch_radius
        for i in range(len(original_indices)):
            y, x = pixel_y[i], pixel_x[i]
            y_min, y_max = max(0, y - r), min(h, y + r + 1)
            x_min, x_max = max(0, x - r), min(w, x + r + 1)

            initial_error_patch = l1_loss_map[y_min:y_max, x_min:x_max]
            initial_patch_error = initial_error_patch.mean() if initial_error_patch.numel() > 0 else torch.tensor(0.0, device=device)

            experience = {
                "step": step,
                "features": features[i].detach(),
                "action": actions[i].detach(),
                "log_prob": log_probs[i].detach(),
                "pixel_x": x,
                "pixel_y": y,
                "initial_patch_error": initial_patch_error,
            }
            state["reward_queue"].append(experience)

        state_to_modify = {k: v for k, v in state.items() if k in ["grad2d", "count", "radii", "age"]}

        prune_agent_mask = (actions == 1)
        prune_opacity_mask = torch.sigmoid(params["opacities"][original_indices].flatten()) < self.prune_opacity_threshold
        prune_mask_subset = prune_agent_mask & prune_opacity_mask

        global_prune_mask = torch.zeros(params["means"].shape[0], dtype=torch.bool, device=device)
        global_prune_mask[original_indices[prune_mask_subset]] = True
        n_prune = global_prune_mask.sum().item()
        if n_prune > 0:
            remove(params, optimizers, state_to_modify, global_prune_mask)

        split_mask_subset = (actions == 2)
        duplicate_mask_subset = (actions == 3)

        prune_map = torch.full((params["means"].shape[0] + global_prune_mask.sum().item(),), -1, dtype=torch.long, device=device)
        prune_map[~global_prune_mask] = torch.arange(params["means"].shape[0], device=device)

        remapped_split_indices = prune_map[original_indices[split_mask_subset]]
        valid_split_mask = remapped_split_indices != -1

        remapped_duplicate_indices = prune_map[original_indices[duplicate_mask_subset]]
        valid_duplicate_mask = remapped_duplicate_indices != -1

        global_split_mask = torch.zeros(params["means"].shape[0], dtype=torch.bool, device=device)
        global_split_mask[remapped_split_indices[valid_split_mask]] = True

        global_duplicate_mask = torch.zeros(params["means"].shape[0], dtype=torch.bool, device=device)
        global_duplicate_mask[remapped_duplicate_indices[valid_duplicate_mask]] = True
        n_split = global_split_mask.sum().item()

        if n_split > 0:
            n_split = global_split_mask.sum().item()
            split(params, optimizers, state_to_modify, global_split_mask)
            state_to_modify["age"][-2*n_split:-n_split] = 0
            state_to_modify["age"][-n_split:] = 0

        n_duplicate = global_duplicate_mask.sum().item()
        if n_duplicate > 0:
            n_duplicate = global_duplicate_mask.sum().item()
            duplicate(params, optimizers, state_to_modify, global_duplicate_mask)
            state_to_modify["age"][-n_duplicate:] = 0

        state.update(state_to_modify)

        return n_prune, n_split, n_duplicate


    def _process_rewards(self, state: dict, current_step: int):
        r = self.reward_patch_radius
        while state["reward_queue"] and (current_step - state["reward_queue"][0]["step"]) >= self.reward_delay:
            exp = state["reward_queue"].popleft()

            if state.get("l1_loss_map") is None:
                continue

            l1_loss_map = state["l1_loss_map"]

            l1_loss_map = l1_loss_map.squeeze() if l1_loss_map.dim() > 2 else l1_loss_map
            h, w = l1_loss_map.shape
            y, x = exp["pixel_y"], exp["pixel_x"]
            y_min, y_max = max(0, y - r), min(h, y + r + 1)
            x_min, x_max = max(0, x - r), min(w, x + r + 1)

            new_error_patch = l1_loss_map[y_min:y_max, x_min:x_max]
            new_patch_error = new_error_patch.mean() if new_error_patch.numel() > 0 else torch.tensor(0.0, device=l1_loss_map.device)

            initial_patch_error = exp["initial_patch_error"]
            reward = initial_patch_error - new_patch_error

            if len(state["replay_buffer"]) < state["replay_buffer"]._storage.max_size:
                experience_td = TensorDict({
                    "features": exp["features"],
                    "action": exp["action"],
                    "log_prob": exp["log_prob"],
                    "reward": torch.as_tensor(reward).clamp(-1.0, 1.0),
                }, batch_size=[])
                state["replay_buffer"].add(experience_td)

    def _train_agent(self, state: dict):
        if len(state["replay_buffer"]) < state["replay_buffer"].batch_size:
            return

        self.ac_net.train()
        device = next(self.ac_net.parameters()).device

        batch = state["replay_buffer"].sample().to(device)
        features = batch.get("features")
        actions = batch.get("action").squeeze(-1)
        rewards = batch.get("reward").squeeze(-1)
        old_log_probs = batch.get("log_prob").squeeze(-1)

        new_logits, values = self.ac_net(features)
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

        self.ac_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ac_net.parameters(), 1.0)
        self.ac_optimizer.step()

        if self.verbose:
            print(f"Agent trained at step {state.get('step', -1)}: Total Loss = {loss.item():.4f}")

        self.writer.add_scalar("agent/loss", loss.item(), state.get("step", -1))
        self.writer.add_scalar("agent/actor_loss", actor_loss.item(), state.get("step", -1))
        self.writer.add_scalar("agent/critic_loss", critic_loss.item(), state.get("step", -1))
        self.writer.add_scalar("agent/entropy_loss", entropy_loss.item(), state.get("step", -1))
        self.writer.add_scalar("agent/mean_reward", rewards.mean().item(), state.get("step", -1))

    @torch.no_grad()
    def _update_quality_map(
            self,
            params: dict[str, torch.nn.Parameter],
            state: dict[str, Any],
            info: dict[str, Any],
    ):
        width, height = info['width'], info['height']

        main_camtoworld = info["camtoworlds"][0]
        main_K = info["Ks"][0]

        view_proj_matrix, view_matrix, proj_matrix = create_view_proj_matrix(
            main_camtoworld, main_K, width, height
        )

        state["view_matrix"] = view_matrix
        state["view_proj_matrix"] = view_proj_matrix
        state["proj_matrix"] = proj_matrix
