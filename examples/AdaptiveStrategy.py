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

class WorldModel(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, action_embedding_dim: int = 16):
        super().__init__()
        self.action_embedding = nn.Embedding(3, action_embedding_dim) # 3 actions: 0=nothing, 1=split, 2=duplicate

        self.net = nn.Sequential(
            nn.Linear(feature_dim + action_embedding_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: Tensor, actions: Tensor) -> Tensor:
        action_embeds = self.action_embedding(actions)
        if action_embeds.dim() == 1:
            action_embeds = action_embeds.unsqueeze(0)
        if features.dim() == 1:
            features = features.unsqueeze(0)

        x = torch.cat([features, action_embeds], dim=-1)
        return self.net(x).squeeze(-1)

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
    initial_exploration_steps: int = 2000

    feature_dim: int = 7
    hidden_dim: int = 64
    learning_rate: float = 3e-4
    ppo_clip_epsilon: float = 0.2
    entropy_loss_weight: float = 0.05
    critic_loss_weight: float = 0.05

    reward_patch_radius: int = 4
    reward_delay: int = 200
    max_densification_subset: int = 100_000

    prune_significance_thresh: float = 0.01
    prune_min_age: int = 800

    es_population_size: int = 50
    es_sigma: float = 0.1
    es_learning_rate: float = 0.01

    grow_ac_net: Any = field(default=None, repr=False)

    writer: Any = field(default=None, repr=False)

    def _get_flat_params(self, model: nn.Module) -> torch.Tensor:
        return torch.cat([p.detach().view(-1) for p in model.parameters()])

    def _set_flat_params(self, model: nn.Module, flat_params: torch.Tensor):
        pointer = 0
        for p in model.parameters():
            num_params = p.numel()
            p.data.copy_(flat_params[pointer:pointer + num_params].view_as(p))
            pointer += num_params

    def initialize_state(self, scene_scale: float = 1.0) -> dict[str, Any]:
        state = super().initialize_state(scene_scale)

        state.update({
            "age": None,
            "l1_loss_map": None,
            "detail_error_map": None,
            "view_proj_matrix": None,
            "es_master_weights": None,
            "es_population_noise": None,
            "es_population_rewards": None,
            "es_current_member_idx": 0,
            "grow_reward_queue": deque(maxlen=10_000),
        })
        return state

    def _initialize_learning_components(self, state: dict, device: torch.device) -> None:
        self.grow_ac_net = SimpleActorCritic(self.feature_dim, self.hidden_dim, 3).to(device)
        master_weights = self._get_flat_params(self.grow_ac_net)

        state["es_master_weights"] = master_weights

        state["es_population_noise"] = torch.randn(
            self.es_population_size,
            len(master_weights),
            device=device
        )

        state["es_population_rewards"] = torch.zeros(self.es_population_size, device=device)

        print("Initialized Actor-Critic network for ES-based growth.")


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

        if self.grow_ac_net is None:
            self._initialize_learning_components(state, params["means"].device)
        if state.get("age") is None:
            state["age"] = torch.zeros(params["means"].shape[0], dtype=torch.int32, device=params["means"].device)

        state["age"] += 1
        state["l1_loss_map"] = info.get("l1_loss_map")
        state["detail_error_map"] = info.get("detail_error_map")
        state["pixels"] = info.get("pixels")
        state["gaussian_contribution"] = info.get("gaussian_contribution")

        if state.get("significance") is None:
            state["significance"] = torch.zeros(params["means"].shape[0], device=params["means"].device)

        if "gaussian_contribution" in info:
            current_significance = info["gaussian_contribution"]

            if state["significance"] is None or state["significance"].shape[0] != current_significance.shape[0]:
                state["significance"] = torch.zeros_like(current_significance)

            if state["significance"].device != current_significance.device:
                state["significance"] = state["significance"].to(current_significance.device)

            state["significance"] = 0.9 * state["significance"] + 0.1 * current_significance


        self._update_quality_map(params, state, info)
        self._update_state(params, state, info, packed=packed)
        self._process_rewards(state, step)

        if step > self.refine_start_iter and step % self.refine_every == 0:
            current_idx = state["es_current_member_idx"]
            next_idx = (current_idx + 1) % self.es_population_size
            state["es_current_member_idx"] = next_idx

            n_prune = self.prune_gs(params, optimizers, state)
            n_split, n_duplicate = self.grow_gs(params, optimizers, state)

            if self.verbose:
                print(f"Step {step}: Evaluated Pop Member {current_idx}. Pruned {n_prune}, Split {n_split}, Duplicated {n_duplicate}.")


        if step > self.refine_start_iter and state["es_current_member_idx"] == 0 and step % self.refine_every == 0:
            self._train_es_agent(state)

        if step % self.reset_every == 0 and step > 0:
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
    def grow_gs(self, params: dict, optimizers: dict, state: dict) -> Tuple[int, int]:
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

        current_idx = state["es_current_member_idx"]
        master_weights = state["es_master_weights"]
        noise = state["es_population_noise"][current_idx]

        perturbed_weights = master_weights + self.es_sigma * noise
        self._set_flat_params(self.grow_ac_net, perturbed_weights)

        self.grow_ac_net.train()
        action_logits, _ = self.grow_ac_net(features)

        action_dist = Categorical(logits=action_logits)
        actions = action_dist.sample()

        self._queue_es_experience(state, features, actions, original_indices, current_idx)

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

    @torch.no_grad()
    def _queue_es_experience(self, state: dict, features: Tensor, actions: Tensor, indices: Tensor, population_idx: int):
        if state.get("detail_error_map") is None: return

        detail_error_map = state["detail_error_map"].squeeze()
        h, w = detail_error_map.shape
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

            initial_error_patch = detail_error_map[y_min:y_max, x_min:x_max]
            initial_patch_error = initial_error_patch.mean() if initial_error_patch.numel() > 0 else torch.tensor(0.0)

            experience = {
                "step": state["step"],
                "population_idx": population_idx,
                "pixel_x": x, "pixel_y": y,
                "initial_patch_error": initial_patch_error,
            }
            state[f"grow_reward_queue"].append(experience)

    def _process_rewards(self, state: dict, current_step: int):
        reward_queue = state["grow_reward_queue"]

        while reward_queue and (current_step - reward_queue[0]["step"]) >= self.reward_delay:
            exp = reward_queue.popleft()
            if state.get("detail_error_map") is None: continue

            detail_error_map = state["detail_error_map"].squeeze()
            h, w = detail_error_map.shape
            y, x = exp["pixel_y"], exp["pixel_x"]
            r = self.reward_patch_radius
            y_min, y_max = max(0, y - r), min(h, y + r + 1)
            x_min, x_max = max(0, x - r), min(w, x + r + 1)

            new_error_patch = detail_error_map[y_min:y_max, x_min:x_max]
            new_patch_error = new_error_patch.mean() if new_error_patch.numel() > 0 else torch.tensor(0.0)

            reward = (exp["initial_patch_error"] - new_patch_error * 10).clamp(-1.0, 1.0)

            pop_idx = exp["population_idx"]
            state["es_population_rewards"][pop_idx] += reward

    def _train_es_agent(self, state: dict):
        step = state.get("step", -1)
        device = state["es_master_weights"].device

        rewards = state["es_population_rewards"]
        if rewards.std() > 1e-6:
            normalized_rewards = (rewards - rewards.mean()) / rewards.std()
        else:
            normalized_rewards = torch.zeros_like(rewards)

        noise_vectors = state["es_population_noise"]
        gradient_estimate = torch.matmul(normalized_rewards, noise_vectors)

        update_step = (self.es_learning_rate / (self.es_population_size * self.es_sigma)) * gradient_estimate
        state["es_master_weights"] += update_step

        if self.writer is not None:
            self.writer.add_scalar(f"agent/es_mean_reward", rewards.mean().item(), step)
            self.writer.add_scalar(f"agent/es_max_reward", rewards.max().item(), step)
            self.writer.add_scalar(f"agent/es_update_norm", torch.linalg.norm(update_step).item(), step)

        state["es_population_noise"] = torch.randn_like(state["es_population_noise"])
        state["es_population_rewards"].zero_()
        state["es_current_member_idx"] = 0

        if self.verbose:
            print(f"Step {step}: ES update complete. Mean reward: {rewards.mean().item():.4f}")


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

        if n_subset > 10:
            means3d_subset = params["means"][subset_mask]
            dists, _ = knn_with_ids(means3d_subset, K=10 + 1)
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