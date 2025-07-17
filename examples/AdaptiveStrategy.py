from collections import deque
from dataclasses import dataclass, field
from typing import Any, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn, autocast, GradScaler
from torch.distributions import Categorical

from torch_geometric.nn import knn_graph, TransformerConv

from ops import duplicate, split
from utils import knn_with_ids, create_view_proj_matrix, scatter_mean
from gsplat.strategy.default import DefaultStrategy
from gsplat.strategy.ops import remove, reset_opa
from tensordict import TensorDict
from torchrl.data import RandomSampler, TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyMemmapStorage

class TemporalGaussianGNN(nn.Module):
    """
    A Graph Neural Network that processes Gaussian features both spatially and temporally.
    It uses a TransformerConv for spatial aggregation and a GRU with attention for temporal context.
    """
    def __init__(self, input_dim: int, hidden_dim: int, edge_dim: int, num_temporal_steps: int = 8):
        super().__init__()
        self.temporal_steps = num_temporal_steps
        self.spatial_gnn = TransformerConv(input_dim, hidden_dim, heads=2, edge_dim=edge_dim)
        self.temporal_gru = nn.GRU(hidden_dim * 2, hidden_dim * 2, batch_first=True)
        self.temporal_attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=4, batch_first=True)
        self.selu = nn.SELU()
        self.norm1 = nn.LayerNorm(hidden_dim * 2)
        self.norm2 = nn.LayerNorm(hidden_dim * 2)

    def forward(self, gaussian_features: Tensor, edge_index: Tensor, edge_attr: Tensor, temporal_history: Tensor) -> tuple[Tensor, Tensor]:
        # Spatial feature extraction
        spatial_features = self.selu(self.spatial_gnn(gaussian_features, edge_index, edge_attr))

        # Get context from historical features
        temporal_features, _ = self.temporal_gru(temporal_history)

        # Attend to temporal features using current spatial features as query
        motion_context, _ = self.temporal_attention(
            spatial_features.unsqueeze(1), temporal_features, temporal_features
        )
        motion_context = self.norm1(motion_context.squeeze(1) + spatial_features)

        # Update the history by rolling the window and adding the new context
        updated_history = torch.cat([temporal_history[:, 1:, :], motion_context.unsqueeze(1)], dim=1)

        return self.norm2(motion_context), updated_history

class PerNodeActorCritic(nn.Module):
    """
    An Actor-Critic network that operates on per-Gaussian features to decide actions.
    Action space: 0 (None), 1 (Split), 2 (Duplicate).
    """
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

class WorldModel(nn.Module):
    """
    A model that predicts the next global scene embedding.
    Used to generate an intrinsic reward based on prediction error (curiosity).
    """
    def __init__(self, scene_embed_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(scene_embed_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, scene_embed_dim)
        )

    def forward(self, scene_embed: Tensor) -> Tensor:
        return self.net(scene_embed)

@dataclass
class AdaptiveStrategy(DefaultStrategy):
    """
    An adaptive strategy using a Temporal GNN for feature extraction and a World Model
    for intrinsic rewards, trained on image quality metrics from a fixed validation set.
    """
    # --- Control Frequencies ---
    refine_every: int = 400
    learn_every: int = 100

    # --- Agent and Model Hyperparameters ---
    num_temporal_steps: int = 8
    gnn_hidden_dim: int = 32
    ac_hidden_dim: int = 64
    wm_hidden_dim: int = 128
    gnn_input_dim: int = 25
    gnn_edge_dim: int = 4
    ac_action_dim: int = 3 # 0:None, 1:Split, 2:Duplicate

    # --- Training Hyperparameters ---
    ac_learning_rate: float = 3e-4
    wm_learning_rate: float = 1e-4
    ppo_clip_epsilon: float = 0.2
    entropy_loss_weight: float = 0.01
    critic_loss_weight: float = 0.5
    intrinsic_reward_factor: float = 0.1
    reward_delay: int = 400

    # --- Reward Metric Weights ---
    reward_weight_psnr: float = 0.1
    reward_weight_ssim: float = 10.0
    reward_weight_l1: float = 20.0
    gauss_count_penalty_factor: float = 0.005
    num_reward_views: int = 4

    # --- Geometry and Pruning ---
    max_densification_subset: int = 75_000
    prune_min_age: int = 1000
    prune_significance_thresh: float = 0.01

    # --- State and Component Fields ---
    reward_validation_set: list[dict] = field(default_factory=list, repr=False)
    rasterizer_fn: Any = field(default=None, repr=False)
    gnn_net: Any = field(default=None, repr=False)
    actor_critic: Any = field(default=None, repr=False)
    world_model: Any = field(default=None, repr=False)
    ac_optimizer: Any = field(default=None, repr=False)
    wm_optimizer: Any = field(default=None, repr=False)
    gnn_optimizer: Any = field(default=None, repr=False)
    psnr_metric: Any = field(default=None, repr=False)
    ssim_metric: Any = field(default=None, repr=False)
    grad_scaler: Any = field(default=None, repr=False)
    knn_fn: Any = field(default=None, repr=False)
    writer: Any = field(default=None, repr=False)

    def setup_validation_set(self, validation_dataset: torch.utils.data.Dataset, device: torch.device) -> None:
        """Creates a fixed set of validation views to calculate reward metrics."""
        if not validation_dataset: return
        indices = torch.randperm(len(validation_dataset))[:self.num_reward_views].tolist()
        for i in indices:
            data = validation_dataset[i]
            self.reward_validation_set.append({
                "camtoworld": data["camtoworld"].to(device),
                "K": data["K"].to(device),
                "pixels": data["image"].to(device) / 255.0,
            })
        if self.verbose:
            print(f"✅ Created a fixed reward validation set with {len(self.reward_validation_set)} views.")

    def initialize_state(self, scene_scale: float = 1.0) -> dict[str, Any]:
        state = super().initialize_state(scene_scale)
        state.update({
            "age": None,
            "significance": None,
            "ac_hidden_states": None,
            "replay_buffer": TensorDictReplayBuffer(
                storage=LazyMemmapStorage(max_size=150_000), sampler=RandomSampler(), batch_size=2048,
            ),
            "reward_queue": deque(maxlen=self.max_densification_subset * 5),
        })
        return state

    def _initialize_learning_components(self, device: torch.device) -> None:
        ac_feature_dim = self.gnn_hidden_dim * 2

        self.gnn_net = TemporalGaussianGNN(self.gnn_input_dim, self.gnn_hidden_dim, self.gnn_edge_dim, self.num_temporal_steps).to(device)
        self.actor_critic = PerNodeActorCritic(ac_feature_dim, self.ac_hidden_dim, self.ac_action_dim).to(device)
        self.world_model = WorldModel(ac_feature_dim, self.wm_hidden_dim).to(device)

        self.grad_scaler = GradScaler()
        self.knn_fn = knn_with_ids

        ac_params = list(self.actor_critic.parameters())
        self.ac_optimizer = torch.optim.AdamW(ac_params, lr=self.ac_learning_rate)
        self.gnn_optimizer = torch.optim.AdamW(self.gnn_net.parameters(), lr=self.ac_learning_rate)
        self.wm_optimizer = torch.optim.AdamW(self.world_model.parameters(), lr=self.wm_learning_rate)

        if self.verbose:
            print("🧠 Initialized Densification Agent (Temporal GNN + AC + World Model).")

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

        if self.actor_critic is None:
            self._initialize_learning_components(params["means"].device)

        if state.get("age") is None:
            state["age"] = torch.zeros(params["means"].shape[0], dtype=torch.int32, device=params["means"].device)

        state["age"] += 1

        self._process_rewards(params, state, step)
        self._update_state(params, state, info, packed=packed)

        if step > self.refine_start_iter and step % self.refine_every == 0:
            n_prune = self.prune_gs(params, optimizers, state)
            n_split, n_duplicate = self.grow_gs(params, optimizers, state, info)
            if self.verbose:
                print(f"🔄 Step {step}: Pruned {n_prune}, Split {n_split}, Duplicated {n_duplicate}.")

        if step > self.refine_start_iter and step % self.learn_every == 0:
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
                                       "significance", "age", "gaussian_contribution", "ac_hidden_states"]
            state_to_modify = {k: v for k, v in state.items() if k in per_gaussian_state_keys}

            remove(params=params, optimizers=optimizers, state=state_to_modify, mask=is_prune)

            state.update(state_to_modify)

        return n_prune


    @torch.no_grad()
    def grow_gs(self, params: dict, optimizers: dict, state: dict, info: dict) -> tuple[int, int]:
        device = params["means"].device
        candidate_mask = state["grad2d"] / state["count"].clamp_min(1.0) > self.grow_grad2d

        num_candidates = candidate_mask.sum().item()
        if num_candidates > self.max_densification_subset:
            candidate_indices = torch.where(candidate_mask)[0]
            rand_indices = torch.randperm(num_candidates, device=device)[:self.max_densification_subset]
            candidate_mask.fill_(False)
            candidate_mask[candidate_indices[rand_indices]] = True

        if candidate_mask.sum() == 0: return 0, 0
        original_indices = torch.where(candidate_mask)[0]

        with autocast(enabled=True, device_type="cuda"):
            per_gaussian_features = self._get_motion_aware_features(params, state, candidate_mask, state["step"])
            if per_gaussian_features is None: return 0, 0

            action_dist, values = self.actor_critic(per_gaussian_features)
            actions = action_dist.sample()

        initial_avg_metrics = self._calculate_avg_metrics(params, step=state["step"])
        self._queue_per_node_experience(state, per_gaussian_features, actions, action_dist.log_prob(actions), values, initial_avg_metrics)

        per_gaussian_state_keys = ["grad2d", "count", "radii", "stagnation_count", "prev_grad2d", "prev_opacity",
                                   "significance", "age", "gaussian_contribution", "ac_hidden_states"]
        state_to_modify = {k: v for k, v in state.items() if k in per_gaussian_state_keys}

        split_mask_subset = (actions == 1)
        global_split_mask = torch.zeros(params["means"].shape[0], dtype=torch.bool, device=device)
        global_split_mask[original_indices[split_mask_subset]] = True
        n_split = global_split_mask.sum().item()
        if n_split > 0:
            split(params, optimizers, state_to_modify, global_split_mask)

        duplicate_mask_subset = (actions == 2)
        global_duplicate_mask = torch.zeros(params["means"].shape[0], dtype=torch.bool, device=device)
        global_duplicate_mask[original_indices[duplicate_mask_subset]] = True
        n_duplicate = global_duplicate_mask.sum().item()
        if n_duplicate > 0:
            duplicate(params, optimizers, state_to_modify, global_duplicate_mask)

        if n_split > 0 or n_duplicate > 0:
            num_new = (n_split * 2) + n_duplicate - (n_split + n_duplicate)
            state_to_modify["age"][-num_new:] = 0

        state.update(state_to_modify)
        return n_split, n_duplicate

    def _train_models(self, state: dict):
        if len(state["replay_buffer"]) < state["replay_buffer"].batch_size: return
        device = next(self.actor_critic.parameters()).device

        self.gnn_net.train()
        self.actor_critic.train()
        self.world_model.train()

        batch = state["replay_buffer"].sample().to(device)
        features = batch.get("features")
        scene_encodings = batch.get("scene_encoding")
        next_scene_encodings = batch.get("next_scene_encoding")
        actions = batch.get("action").squeeze(-1)
        rewards_raw = batch.get("reward").squeeze(-1)
        old_log_probs = batch.get("log_prob").squeeze(-1)
        old_values = batch.get("value").squeeze(-1)

        with autocast(enabled=True, device_type="cuda"):
            next_scene_pred = self.world_model(scene_encodings)
            wm_loss = F.mse_loss(next_scene_pred, next_scene_encodings.detach())

        self.wm_optimizer.zero_grad()
        self.grad_scaler.scale(wm_loss).backward()
        self.grad_scaler.step(self.wm_optimizer)

        with autocast(enabled=True, device_type="cuda"), torch.no_grad():
            rewards = (rewards_raw - rewards_raw.mean()) / (rewards_raw.std() + 1e-8)
            next_values = self.actor_critic(next_scene_encodings)[1]
            delta = rewards + 0.99 * next_values - old_values # GAE
            advantages = (delta - delta.mean()) / (delta.std() + 1e-8)
            returns = advantages + old_values

        with autocast(enabled=True, device_type="cuda"):
            new_dist, new_values = self.actor_critic(features)
            new_log_probs = new_dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip_epsilon, 1.0 + self.ppo_clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(new_values, returns)
            entropy_loss = -new_dist.entropy().mean()
            ac_loss = actor_loss + self.critic_loss_weight * critic_loss + self.entropy_loss_weight * entropy_loss

        self.gnn_optimizer.zero_grad()
        self.ac_optimizer.zero_grad()
        self.grad_scaler.scale(ac_loss).backward()
        self.grad_scaler.step(self.gnn_optimizer)
        self.grad_scaler.step(self.ac_optimizer)

        self.grad_scaler.update()

        if self.writer:
            self.writer.add_scalar("agent/ac_loss", ac_loss.item(), state["step"])
            self.writer.add_scalar("agent/wm_loss", wm_loss.item(), state["step"])
            self.writer.add_scalar("agent/mean_reward", rewards_raw.mean().item(), state["step"])

    @torch.no_grad()
    def _queue_per_node_experience(self, state: dict, features: Tensor, actions: Tensor, log_probs: Tensor, values: Tensor, initial_avg_metrics: dict):
        scene_encoding = features.mean(dim=0).detach()
        for i in range(features.shape[0]):
            experience = {
                "step": state["step"], "features": features[i].detach(),
                "action": actions[i].detach(), "log_prob": log_probs[i].detach(),
                "value": values[i].detach(), "scene_encoding": scene_encoding,
                "initial_avg_metrics": {k: v.detach() for k, v in initial_avg_metrics.items()},
                "initial_gauss_count": state["age"].shape[0]
            }
            state["reward_queue"].append(experience)

    def _process_rewards(self, params: dict, state: dict, current_step: int):
        queue = state["reward_queue"]
        if not queue or (current_step - queue[0]["step"]) < self.reward_delay: return

        with torch.no_grad(), autocast(enabled=True, device_type="cuda"):
            all_node_features = self._get_motion_aware_features(params, state, torch.ones(params["means"].shape[0], dtype=torch.bool, device=params["means"].device), current_step)
            if all_node_features is None or all_node_features.shape[0] == 0: return
            current_scene_encoding = all_node_features.mean(dim=0).detach()

        current_avg_metrics = self._calculate_avg_metrics(params, step=current_step)
        if self.writer:
            for name, val in current_avg_metrics.items():
                self.writer.add_scalar(f"metrics/avg_{name}", val, current_step)

        while queue and (current_step - queue[0]["step"]) >= self.reward_delay:
            exp = queue.popleft()
            initial_metrics = exp["initial_avg_metrics"]

            delta_psnr = current_avg_metrics["psnr"] - initial_metrics["psnr"]
            delta_ssim = current_avg_metrics["ssim"] - initial_metrics["ssim"]
            delta_l1 = initial_metrics["l1"] - current_avg_metrics["l1"]
            extrinsic_reward = self.reward_weight_psnr * delta_psnr + self.reward_weight_ssim * delta_ssim + self.reward_weight_l1 * delta_l1

            with autocast(enabled=True, device_type="cuda"):
                predicted_next_encoding = self.world_model(exp["scene_encoding"])
                intrinsic_reward = F.mse_loss(predicted_next_encoding, current_scene_encoding.detach())

            penalty = self.gauss_count_penalty_factor * max(0, params["means"].shape[0] - exp["initial_gauss_count"])

            reward = extrinsic_reward + self.intrinsic_reward_factor * intrinsic_reward - penalty

            if len(state["replay_buffer"]) < state["replay_buffer"]._storage.max_size:
                td = TensorDict({
                    "features": exp["features"], "scene_encoding": exp["scene_encoding"],
                    "next_scene_encoding": current_scene_encoding, "action": exp["action"],
                    "log_prob": exp["log_prob"], "value": exp["value"], "reward": reward.clamp(-5.0, 5.0).detach(),
                }, batch_size=[])
                state["replay_buffer"].add(td)

    @torch.no_grad()
    def _get_motion_aware_features(self, params: dict, state: dict, subset_mask: Tensor, step: int) -> Tensor:
        if state.get("ac_hidden_states") is None or state["ac_hidden_states"].shape[0] != params["means"].shape[0]:
            state["ac_hidden_states"] = torch.zeros((params["means"].shape[0], self.num_temporal_steps, self.gnn_hidden_dim * 2), device=params["means"].device)

        original_indices = torch.where(subset_mask)[0]
        if original_indices.numel() == 0: return None

        raw_features = self._get_raw_features(params, state, subset_mask, step)
        if raw_features is None: return None

        subset_means = params["means"][subset_mask]
        edge_index = knn_graph(subset_means, k=10, loop=True)

        row, col = edge_index
        rel_dist = torch.norm(subset_means[row] - subset_means[col], dim=-1) / state["scene_scale"]
        scales_i, scales_j = torch.exp(params["scales"][subset_mask][row]).mean(dim=-1), torch.exp(params["scales"][subset_mask][col]).mean(dim=-1)
        scale_diff = torch.abs(scales_i - scales_j)
        opacities_i, opacities_j = torch.sigmoid(params["opacities"][subset_mask][row].flatten()), torch.sigmoid(params["opacities"][subset_mask][col].flatten())
        opacity_diff = torch.abs(opacities_i - opacities_j)
        sh0_i, sh0_j = params["sh0"][subset_mask][row].squeeze(-2), params["sh0"][subset_mask][col].squeeze(-2)
        color_diff = torch.norm(sh0_i - sh0_j, dim=-1)

        edge_attr = torch.stack([rel_dist, scale_diff, opacity_diff, color_diff], dim=-1)
        temporal_history = state['ac_hidden_states'][original_indices]

        self.gnn_net.eval()
        motion_context, updated_history = self.gnn_net(raw_features, edge_index, edge_attr, temporal_history)
        state['ac_hidden_states'][original_indices] = updated_history

        return motion_context

    @torch.no_grad()
    def _get_raw_features(self, params: dict, state: dict, subset_mask: Tensor, step: int) -> Tensor:
        num_subset = subset_mask.sum().item()
        device = params["means"].device

        means3d_subset = params["means"][subset_mask]

        feature_dim = 25
        features = torch.zeros(num_subset, feature_dim, device=device)

        opacities_subset = torch.sigmoid(params["opacities"][subset_mask].flatten())
        features[:, 0] = opacities_subset
        scales = torch.exp(params["scales"][subset_mask])
        features[:, 1] = scales.max(dim=-1).values / state["scene_scale"]
        features[:, 2] = scales.min(dim=-1).values / state["scene_scale"]
        features[:, 3] = scales.mean(dim=-1) / state["scene_scale"]
        features[:, 4] = torch.norm(params["sh0"][subset_mask], dim=(-1, -2))

        if self.knn_fn is not None and len(params["means"]) > 5:
            dists, idxs = self.knn_fn(means3d_subset, K=5 + 1)
            neighbor_idxs = idxs[:, 1:]

            features[:, 9] = dists[:, 1:].mean(dim=-1) / state["scene_scale"]

            neighbor_scales = torch.exp(params["scales"][neighbor_idxs]).max(dim=-1).values
            neighbor_opacities = torch.sigmoid(params["opacities"][neighbor_idxs].squeeze(-1))
            neighbor_sh0 = params["sh0"][neighbor_idxs].squeeze(-2)

            sh0_subset = params["sh0"][subset_mask]

            features[:, 10] = neighbor_scales.mean(dim=-1) / state["scene_scale"]
            features[:, 11] = neighbor_opacities.mean(dim=-1)
            features[:, 12] = torch.norm(neighbor_sh0 - sh0_subset, dim=-1).mean(dim=-1)

        features[:, 19] = step / self.refine_stop_iter

        return torch.nan_to_num(features, 0.0)

    @torch.no_grad()
    def _calculate_avg_metrics(self, params: dict, step: int) -> dict:
        if not self.reward_validation_set: return {"psnr": 0., "ssim": 0., "l1": 0.}

        sh_degree_to_use = min(step // 1000, 3)
        camtoworlds = torch.stack([data["camtoworld"] for data in self.reward_validation_set], dim=0)
        Ks = torch.stack([data["K"] for data in self.reward_validation_set], dim=0)
        gt_pixels = torch.stack([data["pixels"] for data in self.reward_validation_set], dim=0)
        height, width = gt_pixels.shape[1:3]
        colors = torch.cat([params["sh0"], params["shN"]], 1)

        render_colors, _, _ = self.rasterizer_fn(
            camtoworlds=camtoworlds, Ks=Ks, width=width, height=height, sh_degree=sh_degree_to_use,
            means=params["means"], scales=torch.exp(params["scales"]), quats=params["quats"],
            opacities=torch.sigmoid(params["opacities"]), colors=colors,
        )

        rendered_img_p = render_colors.permute(0, 3, 1, 2)
        gt_img_p = gt_pixels.permute(0, 3, 1, 2)

        return {
            "psnr": self.psnr_metric(rendered_img_p, gt_img_p).mean(),
            "ssim": self.ssim_metric(rendered_img_p, gt_img_p).mean(),
            "l1": F.l1_loss(rendered_img_p, gt_img_p),
        }