from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn, autocast
from torch.distributions import Categorical

from ops import duplicate, split
from utils import knn_with_ids
from gsplat.strategy.default import DefaultStrategy
from gsplat.strategy.ops import remove, reset_opa
from tensordict import TensorDict
from torchrl.data import RandomSampler, TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyMemmapStorage


class PerNodeActorCritic(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
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
    # More frequent refinement, in line with default gsplat
    refine_every: int = 100
    learn_every: int = 100

    ac_hidden_dim: int = 64
    ac_feature_dim: int = 13
    ac_action_dim: int = 3  # 0:None, 1:Split, 2:Duplicate

    ac_learning_rate: float = 3e-4
    ppo_clip_epsilon: float = 0.2
    entropy_loss_weight: float = 0.05
    critic_loss_weight: float = 0.5
    gamma: float = 0.99 # Discount factor for future rewards (though actions are now immediate)
    local_reward_scale: float = 50.0

    reward_weight_ssim: float = 10.0
    reward_weight_l1: float = 20.0
    gauss_count_penalty_factor: float = 0.005

    max_densification_subset: int = 15_000
    prune_min_age: int = 1000
    prune_significance_thresh: float = 0.01

    # The rasterizer is required for immediate reward calculation
    rasterizer_fn: Any = field(default=None, repr=False)
    actor_critic: Any = field(default=None, repr=False)
    ac_optimizer: Any = field(default=None, repr=False)
    psnr_metric: Any = field(default=None, repr=False)
    ssim_metric: Any = field(default=None, repr=False)
    knn_fn: Any = field(default=None, repr=False)
    writer: Any = field(default=None, repr=False)

    def initialize_state(self, scene_scale: float = 1.0) -> dict[str, Any]:
        state = super().initialize_state(scene_scale)
        state.update({
            "age": None,
            "significance": None,
            "replay_buffer": TensorDictReplayBuffer(
                storage=LazyMemmapStorage(max_size=20_000),
                sampler=RandomSampler(),
                batch_size=512,
            ),
        })
        return state

    def _initialize_learning_components(self, device: torch.device) -> None:
        self.actor_critic = PerNodeActorCritic(self.ac_feature_dim, self.ac_hidden_dim, self.ac_action_dim).to(device)
        self.knn_fn = knn_with_ids
        ac_params = list(self.actor_critic.parameters())
        self.ac_optimizer = torch.optim.AdamW(ac_params, lr=self.ac_learning_rate)
        if self.verbose:
            print("Initialized Densification Agent (AC).")

    def step_post_backward(
            self,
            params: dict[str, torch.nn.Parameter],
            optimizers: dict[str, torch.optim.Optimizer],
            state: dict[str, Any],
            step: int,
            info: dict[str, Any],
            packed: bool = False,
    ) -> None:
        if step >= self.refine_stop_iter: return
        state["step"] = step

        # We must have the rasterizer to calculate immediate rewards
        if self.rasterizer_fn is None:
            raise ValueError("AdaptiveStrategy requires `rasterizer_fn` to be set for reward calculation.")

        if self.actor_critic is None:
            self._initialize_learning_components(params["means"].device)

        if state.get("age") is None:
            state["age"] = torch.zeros(params["means"].shape[0], dtype=torch.int32, device=params["means"].device)

        state["age"] += 1
        state["ssim"] = info["ssim"]
        state["l1_loss"] = info["l1_loss"]

        self._update_state(params, state, info, packed=packed)

        if step > self.refine_start_iter and step % self.refine_every == 0:
            n_prune = self.prune_gs(params, optimizers, state)
            n_split, n_duplicate = self.grow_gs(params, optimizers, state, info)
            if self.verbose:
                print(f"🔄 Step {step}: Pruned {n_prune}, Split {n_split}, Duplicated {n_duplicate}.")

        if step > self.refine_start_iter and step > self.learn_every and step % self.learn_every == 0:
            self._train_models(state)

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
            state_to_modify = {k: v for k, v in state.items() if k in per_gaussian_state_keys}
            remove(params=params, optimizers=optimizers, state=state_to_modify, mask=is_prune)
            state.update(state_to_modify)
        return n_prune

    @torch.no_grad()
    def _calculate_render_loss(self, params: dict, info: dict, step: int) -> Tensor:
        sh_degree_to_use = min(step // 1000, 3)
        colors = torch.cat([params["sh0"], params["shN"]], 1)
        opacities = torch.sigmoid(params["opacities"])
        scales = torch.exp(params["scales"])

        camtoworlds = info["camtoworlds"]
        Ks = info["Ks"]  # [1, 3, 3]
        pixels = info["pixels"]
        height, width = pixels.shape[1:3]

        render_colors, _, _ = self.rasterizer_fn(
            camtoworlds=camtoworlds,
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=sh_degree_to_use,
            near_plane=0.01,
            far_plane=1e10,
            render_mode="RGB",
            means=params["means"], scales=scales, quats=params["quats"],
            opacities=opacities, colors=colors,
        )
        rendered_img_p = render_colors.permute(0, 3, 1, 2)
        gt_img_p = pixels.permute(0, 3, 1, 2)
        ssim_loss = 1.0 - self.ssim_metric(rendered_img_p, gt_img_p, padding="valid")
        l1_loss = F.l1_loss(render_colors, pixels)

        return self.reward_weight_ssim * (1.0 - ssim_loss) + self.reward_weight_l1 * l1_loss

    def grow_gs(self, params: dict, optimizers: dict, state: dict, info: dict) -> tuple[int, int]:
        step = state.get("step", 0)
        device = params["means"].device
        initial_gauss_count = params["means"].shape[0]

        with torch.no_grad():
            candidate_mask = state["grad2d"] / state["count"].clamp_min(1.0) > self.grow_grad2d
            num_candidates = candidate_mask.sum().item()

            if num_candidates > self.max_densification_subset:
                candidate_indices = torch.where(candidate_mask)[0]
                rand_indices = torch.randperm(num_candidates, device=device)[:self.max_densification_subset]
                candidate_mask.fill_(False)
                candidate_mask[candidate_indices[rand_indices]] = True

            if candidate_mask.sum() == 0:
                return 0, 0

            original_indices = torch.where(candidate_mask)[0]

            with autocast(enabled=False, device_type="cuda"):
                per_gaussian_features = self._get_raw_features(params, state, candidate_mask, state["step"])

            action_dist, values = self.actor_critic(per_gaussian_features)
            actions = action_dist.sample()
            log_probs = action_dist.log_prob(actions)

            xy_view_subset = info["xy_view"][original_indices] # Shape: [num_candidates, 2]
            h, w = info["l1_loss_map"].shape
            grid = xy_view_subset.clone()
            grid[:, 0] = (grid[:, 0] / (w - 1)) * 2 - 1
            grid[:, 1] = (grid[:, 1] / (h - 1)) * 2 - 1
            grid = grid.unsqueeze(0).unsqueeze(1) # Shape: [1, 1, num_candidates, 2]

            l1_map_unsqueezed = info["l1_loss_map"].unsqueeze(0).unsqueeze(0)
            initial_local_error = F.grid_sample(l1_map_unsqueezed, grid, align_corners=True).squeeze()

        per_gaussian_state_keys = ["grad2d", "count", "radii", "stagnation_count", "prev_grad2d", "prev_opacity", "significance", "age", "gaussian_contribution"]
        state_to_modify = {k: v for k, v in state.items() if k in per_gaussian_state_keys}

        split_mask_subset = (actions == 1)
        global_split_mask = torch.zeros(initial_gauss_count, dtype=torch.bool, device=device)
        global_split_mask[original_indices[split_mask_subset]] = True
        n_split = global_split_mask.sum().item()
        if n_split > 0:
            split(params, optimizers, state_to_modify, global_split_mask)

        duplicate_mask_subset = (actions == 2)
        global_duplicate_mask = torch.zeros(initial_gauss_count, dtype=torch.bool, device=device)
        global_duplicate_mask[original_indices[duplicate_mask_subset]] = True
        n_duplicate = global_duplicate_mask.sum().item()
        if n_duplicate > 0:
            duplicate(params, optimizers, state_to_modify, global_duplicate_mask)

        if n_split > 0 or n_duplicate > 0:
            num_new = n_split + n_duplicate
            state_to_modify["age"][-num_new:] = 0
        state.update(state_to_modify)


        with torch.no_grad():
            final_rendered_output = self.rasterizer_fn(
                params=params,
                view_matrix=torch.linalg.inv(info["camtoworlds"]),
                proj_matrix=info["Ks"],
                camera_params=(w, h, info["Ks"][0, 0, 0], info["Ks"][0, 1, 1]),
            )["colors"]
            final_l1_map = (final_rendered_output - info["pixels"]).abs().mean(dim=-1).squeeze(0)

            final_l1_map_unsqueezed = final_l1_map.unsqueeze(0).unsqueeze(0)
            final_local_error = F.grid_sample(final_l1_map_unsqueezed, grid, align_corners=True).squeeze()

            local_reward = (initial_local_error - final_local_error) * self.local_reward_scale

            gauss_count_penalty = self.gauss_count_penalty_factor * (actions > 0).float()
            reward = local_reward - gauss_count_penalty

            for i in range(num_candidates):
                if len(state["replay_buffer"]) >= state["replay_buffer"]._storage.max_size:
                    break

                td = TensorDict({
                    "features": per_gaussian_features[i].detach(),
                    "action": actions[i].detach().unsqueeze(0),
                    "log_prob": log_probs[i].detach().unsqueeze(0),
                    "value": values[i].detach().unsqueeze(0),
                    "reward": reward[i].clamp(-5.0, 5.0).detach().unsqueeze(0),
                }, batch_size=[])
                state["replay_buffer"].add(td)

            if self.writer:
                self.writer.add_scalar("reward/immediate_reward_mean", reward.mean().item(), state["step"])

        return n_split, n_duplicate

    def _train_models(self, state: dict):
        if len(state["replay_buffer"]) < state["replay_buffer"].batch_size: return
        device = next(self.actor_critic.parameters()).device
        self.actor_critic.train()

        batch = state["replay_buffer"].sample().to(device)
        features = batch.get("features")
        actions = batch.get("action").squeeze(-1)
        rewards = batch.get("reward").squeeze(-1)
        old_log_probs = batch.get("log_prob").squeeze(-1)
        old_values = batch.get("value").squeeze(-1)

        # Normalize rewards for stable advantage calculation
        rewards_norm = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # Calculate advantages and returns. Since this is a one-step action,
        # the return is simply the immediate reward, and advantage is reward - value.
        with torch.no_grad():
            advantages = rewards_norm - old_values
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            returns = rewards_norm # The critic learns to predict the normalized reward

        with autocast(enabled=False, device_type="cuda"):
            new_dist, new_values = self.actor_critic(features)
            new_log_probs = new_dist.log_prob(actions)

            # PPO Actor Loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip_epsilon, 1.0 + self.ppo_clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic Loss
            critic_loss = F.mse_loss(new_values, returns)

            # Entropy Loss for exploration
            entropy_loss = -new_dist.entropy().mean()

            ac_loss = actor_loss + self.critic_loss_weight * critic_loss + self.entropy_loss_weight * entropy_loss

        self.ac_optimizer.zero_grad()
        ac_loss.backward()
        self.ac_optimizer.step()

        if self.writer:
            self.writer.add_scalar("agent/ac_loss", ac_loss.item(), state["step"])
            self.writer.add_scalar("agent/actor_loss", actor_loss.item(), state["step"])
            self.writer.add_scalar("agent/critic_loss", critic_loss.item(), state["step"])
            self.writer.add_scalar("agent/mean_reward_raw", rewards.mean().item(), state["step"])

    @torch.no_grad()
    def _get_raw_features(self, params: dict, state: dict, subset_mask: Tensor, step: int) -> Tensor:
        num_subset = subset_mask.sum().item()
        if num_subset == 0:
            return None
        device = params["means"].device

        features = torch.zeros(num_subset, self.ac_feature_dim, device=device)

        means3d_subset = params["means"][subset_mask]
        opacities_subset = torch.sigmoid(params["opacities"][subset_mask].flatten())
        features[:, 0] = opacities_subset

        scales = torch.exp(params["scales"][subset_mask])
        features[:, 1] = scales.max(dim=-1).values / state["scene_scale"]
        features[:, 2] = scales.min(dim=-1).values / state["scene_scale"]
        features[:, 3] = scales.mean(dim=-1) / state["scene_scale"]
        features[:, 4] = torch.norm(params["sh0"][subset_mask], dim=(-1, -2))

        all_indices = torch.where(subset_mask)[0]
        chunk_size = 1000

        if self.knn_fn is not None and len(params["means"]) > 5:
            for i in range(0, num_subset, chunk_size):
                chunk_end = min(i + chunk_size, num_subset)
                chunk_indices = all_indices[i:chunk_end]
                means3d_chunk = params["means"][chunk_indices]

                dists, idxs = self.knn_fn(means3d_chunk, K=5 + 1)
                neighbor_idxs = idxs[:, 1:]

                features[i:chunk_end, 5] = dists[:, 1:].mean(dim=-1) / state["scene_scale"]

                neighbor_scales = torch.exp(params["scales"][neighbor_idxs]).max(dim=-1).values
                features[i:chunk_end, 6] = (neighbor_scales.mean(dim=-1) / state["scene_scale"])

                neighbor_opacities = torch.sigmoid(params["opacities"][neighbor_idxs].squeeze(-1))
                features[i:chunk_end, 7] = neighbor_opacities.mean(dim=-1)

                neighbor_sh0 = params["sh0"][neighbor_idxs].squeeze(-2)
                sh0_chunk = params["sh0"][chunk_indices]
                features[i:chunk_end, 8] = torch.norm(neighbor_sh0 - sh0_chunk, dim=-1).mean(dim=-1)

        grad2d_subset = state["grad2d"][subset_mask]
        count_subset = state["count"][subset_mask]
        features[:, 9] = grad2d_subset / count_subset.clamp_min(1.0)

        age_subset = state["age"][subset_mask]
        features[:, 10] = torch.log1p(age_subset.float())

        max_scale = scales.max(dim=-1).values
        min_scale = scales.min(dim=-1).values
        features[:, 11] = max_scale / min_scale.clamp_min(1e-8)

        features[:, 12] = step / self.refine_stop_iter

        return torch.nan_to_num(features, 0.0)