from collections import deque
from dataclasses import dataclass, field
import time
from typing import Any, Tuple
from approx_topk import topk

import piq
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from torch import Tensor, nn
from torch.distributions import Categorical

from torch_geometric.nn import knn_graph, TransformerConv

from utils import knn_with_ids, scatter_mean
from gsplat.strategy.ops import (
    remove,
    reset_opa,
)
from gsplat.strategy.default import DefaultStrategy
from ops import split, duplicate, merge
from torchrl.data import TensorDictReplayBuffer, RandomSampler
from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from tensordict import TensorDict

class TemporalGaussianGNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, edge_dim: int, num_temporal_steps: int = 5):
        super().__init__()
        self.temporal_steps = num_temporal_steps
        self.spatial_gnn = TransformerConv(input_dim, hidden_dim, heads=2, edge_dim=edge_dim)
        self.temporal_gru = nn.GRU(hidden_dim * 2, hidden_dim * 2, batch_first=True)
        self.temporal_attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=4, batch_first=True)
        self.selu = nn.SELU()
        self.norm1 = nn.LayerNorm(hidden_dim * 2)
        self.norm2 = nn.LayerNorm(hidden_dim * 2)

    def forward(self, gaussian_features: Tensor, edge_index: Tensor, edge_attr: Tensor, temporal_history: Tensor) -> tuple[Tensor, Tensor]:
        spatial_features = self.selu(self.spatial_gnn(gaussian_features, edge_index, edge_attr)) # [N, hidden_dim * 2]

        temporal_features, _ = self.temporal_gru(temporal_history) # [N, T, hidden_dim * 2]

        motion_context, _ = self.temporal_attention(
            spatial_features.unsqueeze(1), temporal_features, temporal_features
        )
        motion_context = self.norm1(motion_context.squeeze(1) + spatial_features)

        updated_history = torch.cat([temporal_history[:, 1:, :], motion_context.unsqueeze(1)], dim=1)

        return self.norm2(motion_context), updated_history

class HierarchicalActorCritic(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, action_dim: int = 6, continuous_dim: int = 9):
        super().__init__()
        region_encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=4, dim_feedforward=hidden_dim, batch_first=True)
        self.region_encoder = nn.TransformerEncoder(region_encoder_layer, num_layers=2)
        self.region_value_head = nn.Linear(feature_dim, 1)
        self.region_actor_head = nn.Linear(feature_dim, 3)  # e.g., 0: stable, 1: refine, 2: prune

        gaussian_encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=4, dim_feedforward=hidden_dim, batch_first=True)
        self.gaussian_encoder = nn.TransformerEncoder(gaussian_encoder_layer, num_layers=2)

        self.cross_scale_attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=4, batch_first=True)
        self.norm_cross = nn.LayerNorm(feature_dim)

        self.gaussian_value_head = nn.Linear(feature_dim, 1)
        self.gaussian_actor_discrete_head = nn.Linear(feature_dim, action_dim)
        self.gaussian_actor_continuous_head = nn.Linear(feature_dim, continuous_dim)
        self.lr_multiplier_head = nn.Linear(feature_dim, 1)

    def forward(self, gaussian_features: Tensor, region_features: Tensor, region_assignments: Tensor) -> Tuple:
        region_features_encoded = self.region_encoder(region_features.unsqueeze(0)).squeeze(0)
        region_values = self.region_value_head(region_features_encoded).squeeze(-1)
        region_action_logits = self.region_actor_head(region_features_encoded)

        region_context = region_features_encoded[region_assignments]

        fused_features, _ = self.cross_scale_attention(
            query=gaussian_features.unsqueeze(1),
            key=region_context.unsqueeze(1),
            value=region_context.unsqueeze(1)
        )
        fused_features = self.norm_cross(fused_features.squeeze(1) + gaussian_features)

        gaussian_features_encoded = self.gaussian_encoder(fused_features.unsqueeze(0)).squeeze(0)

        gaussian_values = self.gaussian_value_head(gaussian_features_encoded).squeeze(-1)
        gaussian_action_logits = self.gaussian_actor_discrete_head(gaussian_features_encoded)

        continuous_params = self.gaussian_actor_continuous_head(gaussian_features_encoded)

        split_ratio = 1.2 + 2.0 * torch.sigmoid(continuous_params[:, 0])
        dupe_offset_mag = torch.sigmoid(continuous_params[:, 1])
        split_dir = F.normalize(continuous_params[:, 2:5], dim=-1)
        dupe_dir = F.normalize(continuous_params[:, 5:8], dim=-1)

        processed_continuous_params = torch.cat([
            split_ratio.unsqueeze(-1), dupe_offset_mag.unsqueeze(-1), split_dir, dupe_dir
        ], dim=-1)
        lr_multiplier = 0.5 + 1.5 * torch.sigmoid(self.lr_multiplier_head(gaussian_features_encoded).squeeze(-1))

        return (gaussian_action_logits, gaussian_values, processed_continuous_params, lr_multiplier,
                region_action_logits, region_values)


class PatchBasedNRQM(nn.Module):
    def __init__(self, model_name: str = "brisque"):
        super().__init__()
        self.model_name = model_name
        if self.model_name == "brisque":
            self.model = piq.BRISQUELoss(reduction='none')
        elif self.model_name == "clipiqa":
            self.model = piq.CLIPIQA()
        else:
            raise ValueError(f"Unknown NRQM model: {self.model_name}. Supported models are 'brisque' and 'clipiqa'.")


    def forward(self, image_patches: torch.Tensor) -> torch.Tensor:
        if self.model_name == "brisque":
            scores = self.model(image_patches)
            return scores
        elif self.model_name == "clipiqa":
            scores = self.model(image_patches)
            return -scores.flatten()

        raise ValueError(f"Unknown NRQM model: {self.model_name}. Supported models are 'brisque' and 'clipiqa'.")


@dataclass
class AdaptiveStrategy(DefaultStrategy):
    """
    An advanced densification strategy that uses a UNIFIED actor-critic network
    to guide the growth and pruning of Gaussians, driven by NRQM feedback.
    """
    refine_every: int = 600
    nrqm_every: int = 500

    num_temporal_steps: int = 8
    num_regions: int = 256
    region_loss_weight: float = 0.5
    region_entropy_weight: float = 0.02

    gnn_knn: int = 10
    gnn_edge_dim: int = 4
    gnn_hidden_dim: int = 32
    gnn_embedding_dim: int = 16
    ac_hidden_dim: int = 64
    num_global_features: int = 5
    use_learned_strategy: bool = True
    bootstrap_steps: int = 0

    learn_every: int = 200
    hindsight_delay: int = 300
    actor_loss_weight: float = 1.0
    entropy_loss_weight: float = 0.3

    start_exploration_epsilon: float = 0.3
    end_exploration_epsilon: float = 0.05
    exploration_decay_steps: int = 5000

    prune_age_threshold: int = 1000
    prune_significance_threshold: float = 0.01

    subset_fraction: float = 1.0
    max_densification_subset: int = 200_000
    action_cost_weight: float = 0.0001

    finetune_lr_multiplier: float = 10.0
    finetune_duration: int = 200

    w_photometric: float = 1.0
    w_detail: float = 1.0
    w_quality: float = -1.0
    w_uncertainty: float = 1.0

    stable_error_threshold: float = 0.01
    stable_quality_threshold: float = 0.005
    stability_reward_bonus: float = 0.05

    ppo_clip_epsilon: float = 0.2

    max_kl_div_threshold: float = 0.05
    gnn_net: Any = field(default=None, repr=False)
    ac_net: Any = field(default=None, repr=False)
    icm_module: Any = field(default=None, repr=False)

    ac_optimizer: Any = field(default=None, repr=False)
    gnn_optimizer: Any = field(default=None, repr=False)
    icm_optimizer: Any = field(default=None, repr=False)

    rasterizer_fn: Any = field(default=None, repr=False)
    nrqm_model: Any = field(default=None, repr=False)

    nrqm_patch_size: int = 32
    nrqm_stagnation_threshold: float = 0.3
    nrqm_ema_decay: float = 0.9

    anisotropic_split: bool = True

    use_geom_uncertainty: bool = False
    num_uncertainty_views: int = 3
    geom_uncertainty_thresh: float = 0.05

    continuous_loss_weight: float = 1.0

    knn_fn: Any = field(default=None, repr=False)

    meta_lr_warmup_steps: int = 300

    novel_poses: Any = field(default=None, repr=False)

    writer: Any = field(default=None, repr=False)

    def initialize_state(self, scene_scale: float = 1.0) -> dict[str, Any]:
        state = super().initialize_state(scene_scale)
        storage = LazyMemmapStorage(max_size=30_000)
        # sample = PrioritizedSampler(max_capacity=30_000, alpha=0.7, beta=0.5)
        sample = RandomSampler()
        replay_buffer = TensorDictReplayBuffer(
            storage=storage,
            sampler=sample,
            batch_size=256
        )
        state.update({
            "prev_means": None,
            "quality_heatmap": None,
            "view_proj_matrix": None,
            "view_matrix": None,
            "detail_error_map": None,
            "geom_uncertainty_map": None,
            "replay_buffer": replay_buffer,
            "hindsight_buffer": deque(maxlen=7_500),
            "significance": None,
            "prev_grad2d": None,
            "prev_opacity": None,
            "age": None,
            "l1_loss_map": None,
            "ac_hidden_states": None,
            "custom_lr_multipliers": None,
            "custom_lr_timers": None,
        })
        return state

    def _initialize_learning_components(self, device) -> None:
        raw_feature_dim = 25
        ac_feature_dim = self.gnn_hidden_dim * 2

        self.gnn_net = TemporalGaussianGNN(
            input_dim=raw_feature_dim,
            hidden_dim=self.gnn_hidden_dim,
            edge_dim=self.gnn_edge_dim,
            num_temporal_steps=self.num_temporal_steps
        ).to(device)

        self.ac_net = HierarchicalActorCritic(
            feature_dim=ac_feature_dim,
            hidden_dim=self.ac_hidden_dim,
        ).to(device)

        ac_params = list(self.ac_net.parameters())
        self.ac_optimizer = torch.optim.AdamW(ac_params, lr=1e-4, weight_decay=1e-5)

        gnn_params = list(self.gnn_net.parameters())
        self.gnn_optimizer = torch.optim.AdamW(gnn_params, lr=1e-4, weight_decay=1e-5)


    @torch.no_grad()
    def _get_raw_features(
            self, params: dict, optimizers: dict, state: dict, subset_mask: Tensor, step: int, campos: Tensor
    ) -> tuple[Tensor, tuple, Tensor]:
        num_subset = subset_mask.sum().item()
        device = params["means"].device

        means3d_subset = params["means"][subset_mask]
        if state.get("l1_loss_map") is None:
            return None, None, None

        h, w = state["l1_loss_map"].shape

        patch_coords_x, patch_coords_y, pixel_coords_x, pixel_coords_y, valid_mask = self._project_to_patch_coords(
            means3d_subset, state["view_proj_matrix"], h, w
        )

        feature_dim = 25
        features = torch.zeros(num_subset, feature_dim, device=device)

        opacities_subset = torch.sigmoid(params["opacities"][subset_mask].flatten())
        features[:, 0] = opacities_subset
        scales = torch.exp(params["scales"][subset_mask])
        features[:, 1] = scales.max(dim=-1).values / state["scene_scale"]
        features[:, 2] = scales.min(dim=-1).values / state["scene_scale"]
        features[:, 3] = scales.mean(dim=-1) / state["scene_scale"]
        features[:, 4] = torch.norm(params["sh0"][subset_mask], dim=(-1, -2))

        valid_indices = torch.where(valid_mask)[0]
        if valid_indices.numel() > 0:
            features[valid_indices, 5] = state["l1_loss_map"][pixel_coords_y[valid_indices], pixel_coords_x[
                valid_indices]]
            # if self.use_geom_uncertainty and state.get("geom_uncertainty_map") is not None:
            #     features[valid_indices, 6] = state["geom_uncertainty_map"][pixel_coords_y[valid_indices], pixel_coords_x[valid_indices]]
            if state.get("quality_heatmap") is not None:
                features[valid_indices, 7] = state["quality_heatmap"][patch_coords_y[valid_indices], patch_coords_x[valid_indices]]
            # if state.get("detail_error_map") is not None:
            #     features[valid_indices, 8] = state["detail_error_map"][pixel_coords_y[valid_indices], pixel_coords_x[valid_indices]]

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

        return torch.nan_to_num(features, 0.0), (pixel_coords_x, pixel_coords_y, patch_coords_x, patch_coords_y), valid_mask

    @torch.no_grad()
    def _get_motion_aware_features(
            self, params: dict, optimizers: dict, state: dict, subset_mask: Tensor, step: int, campos: Tensor
    ) -> tuple[Tensor, tuple, Tensor]:
        original_indices = torch.where(subset_mask)[0]
        raw_features, (px, py, ptx, pty), valid_mask = self._get_raw_features(params, optimizers, state, subset_mask,
                                                                                                  step,
                                                                                         campos)

        if raw_features is None:
            return None, None, None

        subset_means = params["means"][subset_mask]
        edge_index = knn_graph(subset_means, k=self.gnn_knn, loop=True)

        row, col = edge_index
        means_i, means_j = subset_means[row], subset_means[col]
        rel_dist = torch.norm(means_i - means_j, dim=-1) / state["scene_scale"]

        scales_i = torch.exp(params["scales"][subset_mask][row]).mean(dim=-1)
        scales_j = torch.exp(params["scales"][subset_mask][col]).mean(dim=-1)
        scale_diff = torch.abs(scales_i - scales_j) / state["scene_scale"]
        opacities_i = torch.sigmoid(params["opacities"][subset_mask][row].flatten())
        opacities_j = torch.sigmoid(params["opacities"][subset_mask][col].flatten())
        opacity_diff = torch.abs(opacities_i - opacities_j)
        sh0_i = params["sh0"][subset_mask][row].squeeze(-2)
        sh0_j = params["sh0"][subset_mask][col].squeeze(-2)
        color_diff = torch.norm(sh0_i - sh0_j, dim=-1)
        edge_attr = torch.stack([rel_dist, scale_diff, opacity_diff, color_diff], dim=-1)
        edge_attr = torch.nan_to_num(edge_attr, 0.0)

        temporal_history = state['ac_hidden_states'][original_indices]

        self.gnn_net.eval()
        motion_context, updated_history = self.gnn_net(raw_features, edge_index, edge_attr, temporal_history)

        state['ac_hidden_states'][original_indices] = updated_history

        return motion_context, (px, py, ptx, pty, valid_mask), raw_features

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

        if state.get("custom_lr_timers") is not None:
            active_lr_mask = state["custom_lr_timers"] > 0
            if active_lr_mask.any():
                active_indices = torch.where(active_lr_mask)[0]
                multipliers = state["custom_lr_multipliers"][active_indices]

                for p_name, p_val in params.items():
                    if p_val.grad is not None:
                        grad_multipliers = multipliers.view([-1] + [1] * (p_val.grad.dim() - 1))
                        p_val.grad[active_indices] *= grad_multipliers

                state["custom_lr_timers"][active_indices] -= 1

        if self.ac_net is None and self.use_learned_strategy:
            self._initialize_learning_components(params["means"].device)
        if state.get("age") is None:
            state["age"] = torch.zeros(params["means"].shape[0], dtype=torch.int32, device=params["means"].device)
        if state.get("significance") is None:
            state["significance"] = torch.zeros(params["means"].shape[0], device=params["means"].device)
        if self.nrqm_model is None:
            self.nrqm_model = PatchBasedNRQM().to(params["means"].device)
        if self.knn_fn is None:
            self.knn_fn = knn_with_ids

        state["l1_loss_map"] = info.get("l1_loss_map")
        state["detail_error_map"] = info.get("detail_error_map")


        if "gaussian_contribution" in info:
            current_significance = info["gaussian_contribution"]

            if state["significance"] is None or state["significance"].shape[0] != current_significance.shape[0]:
                state["significance"] = torch.zeros_like(current_significance)

            if state["significance"].device != current_significance.device:
                state["significance"] = state["significance"].to(current_significance.device)

            state["significance"] = 0.9 * state["significance"] + 0.1 * current_significance

        if step > self.refine_start_iter:
            state["age"] += 1

        if (state.get("view_proj_matrix") is None and step > 0) or (step % self.nrqm_every == 0):
            self._update_quality_map(params, state, info)
            state["last_nrqm_step"] = step


        self._process_hindsight_buffer(state, step)

        self._update_state(params, state, info, packed=packed)


        if step > self.refine_start_iter:
            if step % self.refine_every == 0:
                t = time.time()
                n_dupli, n_split, n_prune, n_merge, n_finetune = self._update_geometry(params, optimizers, state, step)

                if self.verbose:
                    print(f"Geometry updated at step {step}. Now having {len(params['means'])} GSs. Took {time.time() - t:.2f}s.")
                    print(
                        f"Step {step}: {n_dupli} GSs duplicated, {n_split} GSs split, {n_prune} GSs pruned, "
                        f"{n_merge} GSs merged, {n_finetune} GSs fine-tuned."
                    )

        if step > self.bootstrap_steps:
            if step % self.learn_every == 0:
                self._train_agent(state)

        if step % self.reset_every == 0 and step > 0:
            reset_opa(params=params, optimizers=optimizers, state=state, value=self.prune_opa * 2.0)

    def _update_state(
            self,
            params: dict[str, torch.nn.Parameter] | torch.nn.ParameterDict,
            state: dict[str, Any],
            info: dict[str, Any],
            packed: bool = False,
    ):
        for key in [
            "width",
            "height",
            "n_cameras",
            "radii",
            "gaussian_ids",
            self.key_for_gradient,
        ]:
            assert key in info, f"{key} is required but missing."

        if self.absgrad:
            grads = info[self.key_for_gradient].absgrad.clone()
        else:
            grads = info[self.key_for_gradient].grad.clone()
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

        # initialize state on the first run
        n_gaussian = len(list(params.values())[0])

        if state["grad2d"] is None:
            state["grad2d"] = torch.zeros(n_gaussian, device=grads.device)
        if state["count"] is None:
            state["count"] = torch.zeros(n_gaussian, device=grads.device)
        if self.refine_scale2d_stop_iter > 0 and state["radii"] is None:
            assert "radii" in info, "radii is required but missing."
            state["radii"] = torch.zeros(n_gaussian, device=grads.device)

        # update the running state
        if packed:
            # grads is [nnz, 2]
            gs_ids = info["gaussian_ids"]  # [nnz]
            radii = info["radii"].max(dim=-1).values  # [nnz]
        else:
            # grads is [C, N, 2]
            sel = (info["radii"] > 0.0).all(dim=-1)  # [C, N]
            gs_ids = torch.where(sel)[1]  # [nnz]
            grads = grads[sel]  # [nnz, 2]
            radii = info["radii"][sel].max(dim=-1).values  # [nnz]
        state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
        state["count"].index_add_(
            0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32)
        )
        if self.refine_scale2d_stop_iter > 0:
            # Should be ideally using scatter max
            state["radii"][gs_ids] = torch.maximum(
                state["radii"][gs_ids],
                # normalize radii to [0, 1] screen space
                radii / float(max(info["width"], info["height"])),
                )


    @torch.no_grad()
    def _update_quality_map(
            self,
            params: dict[str, torch.nn.Parameter],
            state: dict[str, Any],
            info: dict[str, Any],
    ):
        step = info["step"]
        width, height = info['width'], info['height']
        sh_degree_to_use = min(step // 1000, 3)

        num_nrqm_poses = min(4, self.novel_poses.shape[0])
        sampled_pose_indices = torch.randperm(self.novel_poses.shape[0])[:num_nrqm_poses]
        novel_camtoworlds = self.novel_poses[sampled_pose_indices]

        Ks = info["Ks"]

        novel_Ks = Ks.repeat(num_nrqm_poses, 1, 1)
        novel_width = width
        novel_height = height

        novel_render_pkg, nrqm_alphas, _ = self.rasterizer_fn(
            means=params["means"],
            quats=params["quats"],
            scales=params["scales"],
            opacities=params["opacities"],
            colors=torch.cat([params["sh0"], params["shN"]], 1),
            camtoworlds=novel_camtoworlds,
            Ks=novel_Ks,
            width=novel_width,
            height=novel_height,
            sh_degree=sh_degree_to_use,
            render_mode="RGB+ED"
        )

        novel_render_for_nrqm = torch.clamp(novel_render_pkg[0, ..., :3].unsqueeze(0).permute(0, 3, 1, 2), 0.0, 1.0)
        all_depth_renders = novel_render_pkg[..., 3]

        p = self.nrqm_patch_size
        patches = novel_render_for_nrqm.unfold(2, p, p).unfold(3, p, p)
        patches = patches.contiguous().view(1, 3, -1, p, p).permute(0, 2, 1, 3, 4).squeeze(0)

        patch_scores = torch.empty(patches.shape[0], device=patches.device)
        std_threshold = 0.01
        is_flat = patches.mean(dim=1).std(dim=[1, 2]) < std_threshold

        if is_flat.any():
            patch_scores[is_flat] = 100.0

        valid_patch_indices = torch.where(~is_flat)[0]
        if valid_patch_indices.numel() > 0:
            valid_patches = patches[valid_patch_indices].float()
            try:
                scores = self.nrqm_model(valid_patches)
                patch_scores[valid_patch_indices] = scores
            except AssertionError:
                patch_scores[valid_patch_indices] = 100.0

        num_patches_h = height // p
        num_patches_w = width // p


        quality_heatmap = patch_scores.view(num_patches_h, num_patches_w)
        if state["quality_heatmap"] is None:
            state["quality_heatmap"] = quality_heatmap
        else:
            state["quality_heatmap"] = torch.lerp(
                quality_heatmap, state["quality_heatmap"], self.nrqm_ema_decay
            )

        if self.use_geom_uncertainty:
            depth_stack = all_depth_renders
            min_d = torch.min(depth_stack[depth_stack > 0])
            max_d = torch.max(depth_stack)
            normalized_depth = (depth_stack - min_d) / (max_d - min_d + 1e-8)
            normalized_depth[depth_stack == 0] = 0
            geom_uncertainty_map = torch.var(normalized_depth, dim=0)

            if state["geom_uncertainty_map"] is None:
                state["geom_uncertainty_map"] = geom_uncertainty_map
            else:
                state["geom_uncertainty_map"] = torch.lerp(
                    geom_uncertainty_map, state["geom_uncertainty_map"], self.nrqm_ema_decay)

        main_novel_camtoworld = novel_camtoworlds[0].unsqueeze(0)
        main_novel_K = novel_Ks[0].unsqueeze(0)

        view_matrix = torch.inverse(main_novel_camtoworld)
        state["view_matrix"] = view_matrix[0]

        fx, fy = main_novel_K[0, 0, 0], main_novel_K[0, 1, 1]
        cx, cy = main_novel_K[0, 0, 2], main_novel_K[0, 1, 2]

        proj_matrix = torch.zeros(4, 4, device=view_matrix.device)
        proj_matrix[0, 0] = 2 * fx / width
        proj_matrix[1, 1] = 2 * fy / height
        proj_matrix[2, 0] = 1.0 - 2 * cx / width
        proj_matrix[2, 1] = 1.0 - 2 * cy / height
        proj_matrix[2, 3] = 1.0
        proj_matrix[3, 2] = 1.0
        state["view_proj_matrix"] = view_matrix[0] @ proj_matrix

        avg_quality = patch_scores.mean()
        normalized_quality = torch.clamp(avg_quality / 50.0, 0.0, 2.0)
        quality_factor = torch.clamp(1.0 + (normalized_quality - 1.0) * 0.5, 0.5, 1.5).item()
        state["dynamic_grow_grad2d"] = self.grow_grad2d * quality_factor

    @torch.no_grad()
    def _project_to_patch_coords(self, means3d: Tensor, view_proj_matrix: Tensor, h: int, w: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        means_h = F.pad(means3d, (0, 1), value=1.0)
        p_hom = means_h @ view_proj_matrix

        w_coord = p_hom[:, 3]
        w_coord_safe = torch.clamp(w_coord, min=1e-6)
        p_w = 1.0 / w_coord_safe
        p_proj = p_hom[:, :2] * p_w[:, None]

        valid_mask = (torch.isfinite(p_proj).all(dim=1) &
                      (torch.abs(p_proj) < 10.0).all(dim=1) &
                      (w_coord > 1e-6))

        coords_x = (p_proj[:, 0] * 0.5 + 0.5) * w
        coords_y = (p_proj[:, 1] * 0.5 + 0.5) * h

        patch_coords_x = torch.clamp(torch.floor(coords_x / self.nrqm_patch_size), 0, (w // self.nrqm_patch_size) - 1).long()
        patch_coords_y = torch.clamp(torch.floor(coords_y / self.nrqm_patch_size), 0, (h // self.nrqm_patch_size) - 1).long()

        pixel_coords_x = torch.clamp(coords_x, 0, w - 1).long()
        pixel_coords_y = torch.clamp(coords_y, 0, h - 1).long()

        return patch_coords_x, patch_coords_y, pixel_coords_x, pixel_coords_y, valid_mask

    def _process_hindsight_buffer(self, state: dict, current_step: int):
        while state["hindsight_buffer"] and \
                (current_step - state["hindsight_buffer"][0]["step"]) >= self.hindsight_delay:

            exp = state["hindsight_buffer"].popleft()

            if state.get("l1_loss_map") is None or \
                    state.get("quality_heatmap") is None or \
                    (state.get("geom_uncertainty_map") is None and self.use_geom_uncertainty) \
                    or state.get("detail_error_map") is None:
                print("Skipping hindsight processing due to missing maps.")
                continue

            px, py = exp["px"], exp["py"]
            ptx, pty = exp["ptx"], exp["pty"]

            current_error = state["l1_loss_map"][max(0, py-2):py+3, max(0, px-2):px+3].mean()
            # reward_photo =(exp["initial_error"] - current_error) / (exp["initial_error"] + 1e-8)

            if not hasattr(state, 'error_baseline'):
                state['error_baseline'] = current_error.detach()
            else:
                state['error_baseline'] = 0.99 * state['error_baseline'] + 0.01 * current_error.detach()

            reward_photo = (exp["initial_error"] - current_error) / (state['error_baseline'] + 1e-6)
            reward_photo = torch.clamp(reward_photo, -2.0, 2.0)

            current_quality = state["quality_heatmap"][pty, ptx]
            reward_quality = exp["initial_quality"] - current_quality

            # current_uncertainty = state["geom_uncertainty_map"][py, px]
            # reward_uncertainty = exp["initial_uncertainty"] - current_uncertainty

            current_detail_error = state["detail_error_map"][max(0, py-2):py+3, max(0, px-2):px+3].mean()
            reward_detail = exp["initial_detail_error"] - current_detail_error

            # base_reward = (self.w_photometric * reward_photo +
            #                self.w_detail * reward_detail +
            #                self.w_quality * reward_quality +
            #                self.w_uncertainty * reward_uncertainty)
            # base_reward = exp["initial_error"] - current_error

            if exp["gaussian_action"] == 0:  # No action
                base_reward = reward_quality
            elif exp["gaussian_action"] in [1, 2]:
                base_reward = reward_quality
            elif exp["gaussian_action"] == 4:
                base_reward = reward_quality + (0.1 if exp["initial_error"] < self.stable_error_threshold else -0.1)
            else:
                base_reward = reward_quality

            shaped_reward = 0.0

            action = exp["gaussian_action"]
            # initial_uncertainty = exp["initial_uncertainty"]

            # if initial_uncertainty > self.geom_uncertainty_thresh:
            #     if action == 1 or action == 2:
            #         shaped_reward += 0.05
            #     elif action == 0:
            #         shaped_reward -= 0.02

            action = exp["gaussian_action"]
            initial_error = exp["initial_error"]
            initial_quality = exp["initial_quality"]

            final_reward = torch.tanh(base_reward * 2.0)

            if action == 0:
                if initial_error < self.stable_error_threshold and \
                        initial_quality < self.stable_quality_threshold:
                    stability_bonus = self.stability_reward_bonus * (1.0 - initial_error / self.stable_error_threshold)
                    final_reward += stability_bonus

            if len(state["replay_buffer"]) < state["replay_buffer"]._storage.max_size:
                experience_tensordict = TensorDict({
                    "motion_features": exp["motion_features"],
                    "region_features": exp["region_features"],
                    "region_assignment": exp["region_assignment"],
                    "gaussian_action": exp["gaussian_action"],
                    "reward": final_reward.clone().detach(),
                    "gaussian_log_prob": exp["gaussian_log_prob"],
                    "continuous_params": exp["continuous_params"],
                }, batch_size=[])
                state["replay_buffer"].add(experience_tensordict)

    def _train_agent(self, state: dict[str, Any]):
        if len(state["replay_buffer"]) < 256:
            if self.verbose:
                print("Not enough samples in replay buffer for training.")
            return

        self.gnn_net.train()
        self.ac_net.train()

        device = next(self.ac_net.parameters()).device

        sampled_td, info = state["replay_buffer"].sample(return_info=True)
        sampled_td = sampled_td.to(device)
        raw_is_weights = sampled_td.get("importance_weights", None)
        if raw_is_weights is None:
            is_weights = torch.ones(sampled_td.shape[0], device=device)
        else:
            is_weights = raw_is_weights

        motion_features = sampled_td.get("motion_features")
        region_features = sampled_td.get("region_features")
        region_assignments = sampled_td.get("region_assignment")
        gauss_actions = sampled_td.get("gaussian_action")
        rewards = sampled_td.get("reward")
        old_gauss_log_probs = sampled_td.get("gaussian_log_prob")

        rewards = torch.nan_to_num(rewards, nan=0.0, posinf=1.0, neginf=-1.0)
        rewards = torch.clamp(rewards, -2.0, 2.0)

        if rewards.std() > 1e-6:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

        (gauss_logits, gauss_values, new_continuous_params, _,
         region_logits, region_values) = self.ac_net(motion_features, region_features, region_assignments)

        gauss_advantage = rewards - gauss_values.detach()
        gauss_advantage = torch.clamp(gauss_advantage, -5.0, 5.0)

        if gauss_advantage.std() > 1e-4:
            gauss_advantage = (gauss_advantage - gauss_advantage.mean()) / (gauss_advantage.std() + 1e-8)

        new_gauss_dist = Categorical(logits=gauss_logits)
        new_gauss_log_probs = new_gauss_dist.log_prob(gauss_actions)

        with torch.no_grad():
            log_ratio = new_gauss_log_probs - old_gauss_log_probs
            approx_kl = torch.exp(log_ratio) - 1 - log_ratio
            mean_kl = approx_kl.mean()

        if mean_kl > self.max_kl_div_threshold:
            if self.verbose:
                print(f"Warning: High KL divergence ({mean_kl.item():.4f} > {self.max_kl_div_threshold}). Skipping training step.")
            self.writer.add_scalar("agent/skipped_updates_kl", 1, state["step"])
            return
        self.writer.add_scalar("agent/skipped_updates_kl", 0, state["step"])

        ratio_gauss = torch.exp(new_gauss_log_probs - old_gauss_log_probs)

        surr1_gauss = ratio_gauss * gauss_advantage
        surr2_gauss = torch.clamp(ratio_gauss, 1.0 - self.ppo_clip_epsilon, 1.0 + self.ppo_clip_epsilon) * gauss_advantage
        actor_loss_gauss = -torch.min(surr1_gauss, surr2_gauss)

        critic_loss_gauss = F.mse_loss(gauss_values, rewards, reduction="none")
        entropy_loss_gauss = -self.entropy_loss_weight * new_gauss_dist.entropy()

        total_loss_unreduced = (actor_loss_gauss + critic_loss_gauss + entropy_loss_gauss)

        loss = (total_loss_unreduced * is_weights).mean()

        if torch.isnan(loss) or torch.isinf(loss):
            print("NaN loss detected, skipping training step to prevent model corruption.")
            return

        self.ac_optimizer.zero_grad()
        self.gnn_optimizer.zero_grad()
        loss.backward()

        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in self.ac_net.parameters() if p.requires_grad],
            max_norm=1.0
        )
        torch.nn.utils.clip_grad_norm_(self.gnn_net.parameters(), 0.5)

        self.ac_optimizer.step()
        self.gnn_optimizer.step()

        td_errors = (rewards - gauss_values).abs().detach()
        td_errors = torch.nan_to_num(td_errors, nan=0.0, posinf=1.0, neginf=0.0)

        sampled_td.set("priority", td_errors)

        state["replay_buffer"].update_priority(info.get("index"), td_errors)

        ratio_mean = ratio_gauss.mean().item()
        ration_std = ratio_gauss.std().item()
        if ration_std > 2.0:
            print(f"Warning: High ratio std: {ration_std:.4f} at step {state['step']}, mean ratio: {ratio_mean:.4f}")

        if self.verbose:
            print(f"Gradient norm: {actor_grad_norm:.4f}")

        if self.verbose:
            print(f"Agent trained: Loss = {loss.item():.4f}, Actor Loss = {actor_loss_gauss.mean().item():.4f}, "
                  f"Critic Loss = {critic_loss_gauss.mean().item():.4f}, "
                  f"Entropy Loss = {entropy_loss_gauss.mean().item():.4f}, ")

        action_counts = torch.bincount(gauss_actions, minlength=6)
        action_probs = action_counts.float() / action_counts.sum().float()
        for i, prob in enumerate(action_probs):
            self.writer.add_scalar(f"action_distribution/action_{i}", prob, state["step"])

        self.writer.add_scalar("agent/loss", loss.item(), state["step"])
        self.writer.add_scalar("agent/actor_loss", actor_loss_gauss.mean().item(), state["step"])
        self.writer.add_scalar("agent/critic_loss", critic_loss_gauss.mean().item(), state["step"])
        self.writer.add_scalar("agent/entropy_loss", entropy_loss_gauss.mean().item(), state["step"])
        self.writer.add_scalar("agent/reward_mean", rewards.mean().item(), state["step"])
        self.writer.add_scalar("agent/reward_std", rewards.std().item(), state["step"])
        self.writer.add_scalar("agent/advantage_mean", gauss_advantage.mean().item(), state["step"])
        self.writer.add_scalar("agent/advantage_std", gauss_advantage.std().item(), state["step"])
        self.writer.add_scalar("agent/mean_kl_div", mean_kl.item(), state["step"])


    @torch.no_grad()
    def _update_geometry(self, params: dict, optimizers: dict, state: dict, step: int) -> Tuple[int, int, int, int, int]:
        initial_num_gaussians = len(params["means"])
        device = params["means"].device

        if state.get("custom_lr_timers") is None or state["custom_lr_timers"].shape[0] != initial_num_gaussians:
            state["custom_lr_timers"] = torch.zeros(initial_num_gaussians, dtype=torch.int32, device=device)
            state["custom_lr_multipliers"] = torch.ones(initial_num_gaussians, device=device)

        if state.get("view_matrix") is None: return 0, 0, 0, 0, 0

        if state.get("ac_hidden_states") is None or state["ac_hidden_states"].shape[0] != initial_num_gaussians:
            hidden_dim = self.gnn_hidden_dim * 2
            state["ac_hidden_states"] = torch.zeros(
                (initial_num_gaussians, self.num_temporal_steps, hidden_dim), device=device
            )

        grad_thresh = self.grow_grad2d
        candidates_by_grad = state["grad2d"] / state["count"].clamp_min(1) > grad_thresh

        h, w = state['l1_loss_map'].shape[0], state['l1_loss_map'].shape[1]

        ptx, pty, px, py, valid_proj = self._project_to_patch_coords(params['means'], state['view_proj_matrix'], h, w)

        low_quality_mask = torch.zeros(initial_num_gaussians, dtype=torch.bool, device=device)
        low_quality_mask[valid_proj] = state["quality_heatmap"][pty[valid_proj], ptx[valid_proj]] < self.nrqm_stagnation_threshold

        high_uncertainty_mask = torch.zeros(initial_num_gaussians, dtype=torch.bool, device=device)
        # high_uncertainty_mask[valid_proj] = state["geom_uncertainty_map"][py[valid_proj], px[valid_proj]] > self.geom_uncertainty_thresh

        if state.get("age") is not None:
            age_mask = state["age"] < 300
        else:
            age_mask = torch.zeros_like(params["means"], dtype=torch.bool, device=device)

        subset_mask = (candidates_by_grad | low_quality_mask | high_uncertainty_mask | age_mask)

        num_candidates = subset_mask.sum().item()
        if num_candidates == 0: return 0, 0, 0, 0, 0

        if num_candidates > self.max_densification_subset:
            candidate_indices = torch.where(subset_mask)[0]
            sampled_indices = candidate_indices[torch.randperm(num_candidates)[:self.max_densification_subset]]
            subset_mask.fill_(False)
            subset_mask[sampled_indices] = True

        original_subset_indices = torch.where(subset_mask)[0]
        camtoworld_matrix = torch.inverse(state['view_matrix'])
        campos = camtoworld_matrix[:3, 3]

        subset_means = params["means"][subset_mask]
        motion_features, (px_sub, py_sub, ptx_sub, pty_sub, valid_mask_subset), _ = self._get_motion_aware_features(
            params, optimizers, state,subset_mask, step,campos)

        if motion_features is None: return 0, 0, 0, 0, 0

        if len(subset_means) < self.num_regions: return 0,0,0,0,0
        kmeans = MiniBatchKMeans(n_clusters=self.num_regions, random_state=step).fit(subset_means.cpu().numpy())
        region_assignments = torch.from_numpy(kmeans.labels_).to(device)
        region_features = scatter_mean(motion_features, region_assignments, self.num_regions)

        self.ac_net.eval()
        (gaussian_logits, _, continuous_params, lr_multipliers,
         region_logits, _) = self.ac_net(motion_features, region_features, region_assignments)


        age_in_steps = state["age"][original_subset_indices]
        significance = state["significance"][original_subset_indices]
        opacities = torch.sigmoid(params["opacities"][original_subset_indices])

        progress = max(0.0, (step - self.bootstrap_steps) / self.exploration_decay_steps)
        epsilon = self.end_exploration_epsilon + (self.start_exploration_epsilon - self.end_exploration_epsilon) * (1 - progress)

        gauss_dist = Categorical(logits=gaussian_logits)

        if torch.rand(1).item() < epsilon:
            final_actions = gauss_dist.sample()
            random_mask = torch.rand(len(final_actions), device=device) < 0.3
            final_actions[random_mask] = torch.randint(0, 6, (random_mask.sum(),), device=device)
        else:
            final_actions = gauss_dist.sample()

        gauss_log_probs = gauss_dist.log_prob(final_actions)

        for i in range(len(original_subset_indices)):
            if not valid_mask_subset[i]:
                continue

            initial_error = state["l1_loss_map"][max(0, py_sub[i]-2):py_sub[i]+3, max(0, px_sub[i]-2):px_sub[i]+3].mean()
            initial_quality = state["quality_heatmap"][pty_sub[i], ptx_sub[i]]
            # initial_uncertainty = state["geom_uncertainty_map"][py_sub[i], px_sub[i]]
            initial_detail_error = state["detail_error_map"][max(0, py_sub[i]-2):py_sub[i]+3, max(0, px_sub[i]-2):px_sub[i]+3].mean()

            region_idx = region_assignments[i]

            exp = {
                "step": step,
                "motion_features": motion_features[i].detach(),
                "region_features": region_features[region_idx].detach(),
                "region_assignment": region_idx.detach(),
                "gaussian_action": final_actions[i].detach(),
                "gaussian_log_prob": gauss_log_probs[i].detach(),
                "initial_error": initial_error.detach(),
                "initial_quality": initial_quality.detach(),
                # "initial_uncertainty": initial_uncertainty.detach(),
                "initial_detail_error": initial_detail_error.detach(),
                "continuous_params": continuous_params[i].detach(),
                "px": px_sub[i], "py": py_sub[i], "ptx": ptx_sub[i], "pty": pty_sub[i]
            }
            state["hindsight_buffer"].append(exp)

        final_finetune_mask_subset = (final_actions == 5) # Finetune action
        final_prune_mask_subset = (final_actions == 4) # Prune action
        final_merge_mask_subset = (final_actions == 3) # Merge action
        final_dupe_mask_subset = (final_actions == 2) # Duplicate action
        final_split_mask_subset = (final_actions == 1) # Split action

        finetune_indices = original_subset_indices[final_finetune_mask_subset]
        n_finetune = finetune_indices.numel()
        if n_finetune > 0:
            state["custom_lr_multipliers"][finetune_indices] = self.finetune_lr_multiplier
            state["custom_lr_timers"][finetune_indices] = self.finetune_duration

        state_to_modify = {k: v for k, v in state.items() if k in ["grad2d", "count", "radii", "age", "significance", "ac_hidden_states", "custom_lr_multipliers", "custom_lr_timers"]}

        prune_orig_indices = original_subset_indices[final_prune_mask_subset]
        global_prune_mask = torch.zeros(initial_num_gaussians, dtype=torch.bool, device=device)
        global_prune_mask[prune_orig_indices] = True
        n_prune = global_prune_mask.sum().item()
        if n_prune > 0:
            remove(params, optimizers, state_to_modify, global_prune_mask)

        prune_map = torch.full((initial_num_gaussians,), -1, dtype=torch.long, device=device)
        prune_map[~global_prune_mask] = torch.arange(initial_num_gaussians - n_prune, device=device)
        num_gaussians_post_prune = len(params["means"])

        # B. MERGE
        merge_orig_indices = original_subset_indices[final_merge_mask_subset]
        global_merge_mask = torch.zeros(num_gaussians_post_prune, dtype=torch.bool, device=device)
        if len(merge_orig_indices) > 0:
            remapped_indices = prune_map[merge_orig_indices]
            valid_mask = remapped_indices != -1
            if valid_mask.any():
                global_merge_mask[remapped_indices[valid_mask]] = True

        removed_by_merge_mask, n_merge_pairs = merge(params, optimizers, state_to_modify, global_merge_mask) if global_merge_mask.any() else (torch.zeros_like(global_merge_mask), 0)
        num_gaussians_post_merge = len(params["means"])

        merge_map = torch.full((num_gaussians_post_prune,), -1, dtype=torch.long, device=device)
        if removed_by_merge_mask.any():
            merge_map[~removed_by_merge_mask] = torch.arange(num_gaussians_post_merge - n_merge_pairs, device=device)
        else:
            valid_prune_indices = (prune_map != -1)
            merge_map[valid_prune_indices] = prune_map[valid_prune_indices]

        # C. SPLIT
        split_orig_indices = original_subset_indices[final_split_mask_subset]
        split_continuous_params = continuous_params[final_split_mask_subset]
        global_split_mask = torch.zeros(num_gaussians_post_merge, dtype=torch.bool, device=device)
        if len(split_orig_indices) > 0:
            remapped_indices = merge_map[prune_map[split_orig_indices]]
            valid_mask = remapped_indices != -1
            if valid_mask.any():
                global_split_mask[remapped_indices[valid_mask]] = True

        n_split = global_split_mask.sum().item()
        if n_split > 0:
            split(params, optimizers, state_to_modify, global_split_mask)
        num_gaussians_post_split = len(params["means"])

        # D. DUPLICATE
        dupe_orig_indices = original_subset_indices[final_dupe_mask_subset]
        dupe_continuous_params = continuous_params[final_dupe_mask_subset]
        n_dupli = 0

        if len(dupe_orig_indices) > 0:
            indices_post_merge = merge_map[prune_map[dupe_orig_indices]]

            valid_mask_after_merge = indices_post_merge != -1
            indices_to_consider = indices_post_merge[valid_mask_after_merge]
            params_to_consider = dupe_continuous_params[valid_mask_after_merge]

            if len(indices_to_consider) > 0:
                was_split_mask = global_split_mask[indices_to_consider]

                final_indices_pre_split = indices_to_consider[~was_split_mask]
                final_dupe_params = params_to_consider[~was_split_mask]

                if len(final_indices_pre_split) > 0:
                    unsplit_map = torch.full((num_gaussians_post_merge,), -1, dtype=torch.long, device=device)
                    unsplit_indices = torch.where(~global_split_mask)[0]
                    unsplit_map[unsplit_indices] = torch.arange(len(unsplit_indices), device=device)

                    final_indices_post_split = unsplit_map[final_indices_pre_split]

                    global_dupe_mask = torch.zeros(num_gaussians_post_split, dtype=torch.bool, device=device)
                    global_dupe_mask[final_indices_post_split] = True
                    n_dupli = global_dupe_mask.sum().item()

                    assert n_dupli == len(final_dupe_params), "Mismatch after robust index filtering"

                    dupe_indices = torch.where(global_dupe_mask)[0]
                    dupe_scales = torch.exp(params["scales"][dupe_indices])
                    offset_mags = final_dupe_params[:, 1]
                    dupe_dirs = final_dupe_params[:, 5:8]
                    offset_magnitudes = dupe_scales.max(dim=-1).values * offset_mags
                    duplicate(params, optimizers, state_to_modify, global_dupe_mask)

        state.update(state_to_modify)
        if n_dupli > 0: state["age"][-n_dupli:] = 0
        if n_split > 0: state["age"][-(n_split*2):-n_split] = 0; state["age"][-n_split:] = 0

        state["prev_means"] = params["means"].clone()

        return n_dupli, n_split, n_prune, n_merge_pairs, n_finetune