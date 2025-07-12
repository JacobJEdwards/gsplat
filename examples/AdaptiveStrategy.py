from collections import deque
from dataclasses import dataclass, field
import time
from typing import Any, Tuple
from approx_topk import topk

import piq
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Categorical, Bernoulli

from torch_geometric.nn import knn_graph, TransformerConv

from gsplat.utils import normalized_quat_to_rotmat
from utils import PrioritizedReplayBuffer
from gsplat.strategy.ops import (
    remove,
    reset_opa,
)
from gsplat.strategy.default import DefaultStrategy
from ops import split, duplicate, merge

class GaussianGraphNetwork(nn.Module):
    def __init__(self, input_dim: int, edge_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(TransformerConv(input_dim, hidden_dim, heads=2, edge_dim=edge_dim))
        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv(hidden_dim * 2, hidden_dim, heads=2, edge_dim=edge_dim))
        self.convs.append(TransformerConv(hidden_dim * 2, output_dim, concat=False, edge_dim=edge_dim))
        self.prelu = nn.PReLU()

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            if i < len(self.convs) - 1:
                x = self.prelu(x)
        return x

class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dim: int = 18, mlp_width: int = 64):
        super().__init__()
        self.base_net = nn.Sequential(
            nn.Linear(input_dim, mlp_width),
            nn.LayerNorm(mlp_width),
            nn.SELU(),
            nn.Linear(mlp_width, mlp_width),
            nn.LayerNorm(mlp_width),
            nn.SELU(),
        )

        self.critic_head = nn.Linear(mlp_width, 1)
        self.actor_discrete_head = nn.Linear(mlp_width, 4)
        self.actor_continuous_head = nn.Linear(mlp_width, 8)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        base_features = self.base_net(x)
        value = self.critic_head(base_features).squeeze(-1)
        action_logits = self.actor_discrete_head(base_features)

        continuous_params = self.actor_continuous_head(base_features)
        split_ratio = 1.2 + 2.0 * torch.sigmoid(continuous_params[:, 0])
        dupe_offset_mag = torch.sigmoid(continuous_params[:, 1])
        split_dir = F.normalize(continuous_params[:, 2:5], dim=-1)
        dupe_dir = F.normalize(continuous_params[:, 5:8], dim=-1)

        processed_continuous_params = torch.cat([
            split_ratio.unsqueeze(-1), dupe_offset_mag.unsqueeze(-1), split_dir, dupe_dir
        ], dim=-1)
        return action_logits, value, processed_continuous_params

class PruningActorCritic(nn.Module):
    """A dedicated agent for the binary decision of pruning a Gaussian."""
    def __init__(self, input_dim: int = 18, mlp_width: int = 64):
        super().__init__()
        self.base_net = nn.Sequential(
            nn.Linear(input_dim, mlp_width),
            nn.LayerNorm(mlp_width), nn.SELU(),
            nn.Linear(mlp_width, mlp_width),
            nn.LayerNorm(mlp_width), nn.SELU(),
        )
        self.critic_head = nn.Linear(mlp_width, 1)
        self.actor_head = nn.Linear(mlp_width, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        base_features = self.base_net(x)
        value = self.critic_head(base_features).squeeze(-1)
        action_logit = self.actor_head(base_features).squeeze(-1)
        return action_logit, value

class ICMModule(nn.Module):
    """Intrinsic Curiosity Module to encourage exploration."""
    def __init__(self, feature_dim: int, action_dim: int, mlp_width: int = 64):
        super().__init__()
        # s_t, s_{t+1} -> a_t
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, mlp_width), nn.SELU(),
            nn.Linear(mlp_width, action_dim)
        )
        # s_t, a_t -> s_{t+1}
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, mlp_width), nn.SELU(),
            nn.Linear(mlp_width, feature_dim)
        )

    def forward(self, state, next_state, action_one_hot):
        pred_next_state = self.forward_model(torch.cat([state, action_one_hot], dim=-1))
        pred_action_logits = self.inverse_model(torch.cat([state, next_state], dim=-1))
        return pred_next_state, pred_action_logits

class PatchBasedNRQM(nn.Module):
    def __init__(self, model_name: str = "clipiqa"):
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
            return scores.view(scores.shape[0])
        elif self.model_name == "clipiqa":
            scores = self.model(image_patches)
            return -scores.view(scores.shape[0])

        raise ValueError(f"Unknown NRQM model: {self.model_name}. Supported models are 'brisque' and 'clipiqa'.")


@dataclass
class AdaptiveStrategy(DefaultStrategy):
    """
    An advanced densification strategy that uses NRQM feedback to guide the
    growth and pruning of Gaussians.

    This strategy introduces three core NRQM-driven mechanics:
    1.  **Spatially-Aware Densification**: Uses a quality heatmap from novel views
        to prioritize densifying Gaussians in low-quality regions.
    2.  **NRQM-Modulated Threshold**: Dynamically adjusts the gradient threshold for
        densification based on the overall rendered quality.
    3.  **Quality-Aware Pruning**: Prunes Gaussians that are "stagnant" - i.e.,
        consistently contribute to low-quality regions without having high enough
        gradients to be densified.

    Args:
        nrqm_every (int): How often to run the NRQM evaluation to update the
          quality map. Default is 250 steps.
        nrqm_patch_size (int): The size of patches to use for the spatial NRQM.
          Default is 32.
        nrqm_stagnation_threshold (float): A score below which a region is
          considered low-quality for stagnation tracking. Default is 0.3.
        nrqm_prune_stagnant_after (int): Prune stagnant Gaussians after they
          have been observed in low-quality regions this many times. Default is 10.
        anisotropic_split (bool): Whether to use anisotropic splitting. Default is True.
        prune_redundant (bool): Whether to prune redundant Gaussians. Default is True.
        redundancy_knn (int): Number of nearest neighbors to consider for redundancy pruning. Default is 5.
        redundancy_overlap_thresh (float): Overlap threshold for redundancy pruning. Default is 0.6.
        redundancy_color_thresh (float): Color similarity threshold for redundancy pruning. Default is 0.1.
    """
    refine_every: int = 400
    prune_every: int = 400
    nrqm_every: int = 1000

    gnn_knn: int = 10
    gnn_edge_dim: int = 4
    gnn_hidden_dim: int = 32
    gnn_embedding_dim: int = 16
    num_global_features: int = 5
    use_learned_strategy: bool = True
    bootstrap_steps: int = 5000

    densify_learn_every: int = 200
    densify_hindsight_delay: int = 400
    actor_loss_weight: float = 1.0
    entropy_loss_weight: float = 0.1
    start_exploration_epsilon: float = 0.3
    end_exploration_epsilon: float = 0.05
    exploration_decay_steps: int = 15000

    use_learned_pruning: bool = True
    pruning_learn_every: int = 500
    pruning_hindsight_delay: int = 800
    prune_threshold: float = 0.6

    use_curiosity: bool = True
    curiosity_weight: float = 0.05
    icm_action_dim: int = 4

    subset_fraction: float = 0.2
    max_densification_subset: int = 1_000_000
    prune_significance_threshold: float = 0.01
    action_cost_weight: float = 0.001
    w_photometric: float = 1.0
    w_quality: float = -1.0
    w_uncertainty: float = 1.0
    ppo_clip_epsilon: float = 0.2

    gnn_net: Any = field(default=None, repr=False)
    ac_net: Any = field(default=None, repr=False)
    pruning_ac_net: Any = field(default=None, repr=False)
    icm_module: Any = field(default=None, repr=False)

    densify_optimizer: Any = field(default=None, repr=False)
    pruning_optimizer: Any = field(default=None, repr=False)
    icm_optimizer: Any = field(default=None, repr=False)

    rasterizer_fn: Any = field(default=None, repr=False)
    nrqm_model: Any = field(default=None, repr=False)


    nrqm_patch_size: int = 32
    nrqm_stagnation_threshold: float = 0.3
    nrqm_prune_stagnant_after: int = 15
    nrqm_ema_decay: float = 0.9

    anisotropic_split: bool = True
    prune_redundant: bool = True
    redundancy_knn: int = 5
    redundancy_overlap_thresh: float = 0.6
    redundancy_color_thresh: float = 0.1

    use_geom_uncertainty: bool = True
    num_uncertainty_views: int = 3
    geom_uncertainty_thresh: float = 0.002

    photometric_error_thresh: float = 0.1

    learn_every: int = 200
    hindsight_delay: int = 400

    continuous_loss_weight: float = 0.5

    pruning_bootstrap_steps: int = 5000
    learned_prune_threshold: float = 0.5

    pruning_net: Any = field(default=None, repr=False)

    knn_fn: Any = field(default=None, repr=False)
    lpips_metric: Any = field(default=None, repr=False)

    combined_optimizer: Any = field(default=None, repr=False)

    start_asc_grad2d: float = 0.0002
    end_asc_grad2d: float = 0.001

    def initialize_state(self, scene_scale: float = 1.0) -> dict[str, Any]:
        state = super().initialize_state(scene_scale)
        state.update({
            "quality_heatmap": None,
            "view_proj_matrix": None,
            "photometric_error_map": None,
            "geom_uncertainty_map": None,
            "densify_replay_buffer": PrioritizedReplayBuffer(capacity=20_000),
            "densify_hindsight_buffer": deque(maxlen=5_000),
            "pruning_replay_buffer": PrioritizedReplayBuffer(capacity=10_000),
            "pruning_hindsight_buffer": deque(maxlen=2_500),
            "significance": None,
            "prev_grad2d": None,
            "prev_opacity": None,
            "age": None,
            "l1_loss_map": None,
        })

        return state

    def _initialize_learning_components(self, device) -> None:
        raw_feature_dim = 18
        ac_input_dim = self.gnn_embedding_dim + self.num_global_features

        self.gnn_net = GaussianGraphNetwork(
            input_dim=raw_feature_dim, edge_dim=self.gnn_edge_dim,
            hidden_dim=self.gnn_hidden_dim, output_dim=self.gnn_embedding_dim,
        ).to(device)

        self.ac_net = ActorCriticNetwork(input_dim=ac_input_dim).to(device)
        densify_params = list(self.gnn_net.parameters()) + list(self.ac_net.parameters())
        self.densify_optimizer = torch.optim.AdamW(densify_params, lr=1e-4, weight_decay=1e-5)

        if self.use_learned_pruning:
            self.pruning_ac_net = PruningActorCritic(input_dim=ac_input_dim).to(device)
            pruning_params = list(self.gnn_net.parameters()) + list(self.pruning_ac_net.parameters())
            self.pruning_optimizer = torch.optim.AdamW(pruning_params, lr=5e-5, weight_decay=1e-5)

        if self.use_curiosity:
            self.icm_module = ICMModule(
                feature_dim=ac_input_dim, action_dim=self.icm_action_dim
            ).to(device)
            self.icm_optimizer = torch.optim.AdamW(self.icm_module.parameters(), lr=1e-4)


    @torch.no_grad()
    def _get_raw_features(
            self, params: dict, state: dict, subset_mask: Tensor, step: int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        num_subset = subset_mask.sum().item()
        device = params["means"].device

        means3d_subset = params["means"][subset_mask]
        if state.get("photometric_error_map") is None and state.get("l1_loss_map") is None:
            return None, None, None, None, None

        if state.get("l1_loss_map") is not None:
            h, w = state["l1_loss_map"].shape
        else:
            h, w = state["photometric_error_map"].shape

        patch_coords_x, patch_coords_y, pixel_coords_x, pixel_coords_y, valid_mask = self._project_to_patch_coords(
            means3d_subset, state["view_proj_matrix"], h, w
        )

        feature_dim = 18
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
            if state.get("l1_loss_map") is not None:
                features[valid_indices, 5] = state["l1_loss_map"][pixel_coords_y[valid_indices], pixel_coords_x[valid_indices]]
            elif state.get("photometric_error_map") is not None:
                features[valid_indices, 5] = state["photometric_error_map"][pixel_coords_y[valid_indices], pixel_coords_x[valid_indices]]

            if self.use_geom_uncertainty and state.get("geom_uncertainty_map") is not None:
                features[valid_indices, 6] = state["geom_uncertainty_map"][pixel_coords_y[valid_indices], pixel_coords_x[valid_indices]]
            if state.get("quality_heatmap") is not None:
                features[valid_indices, 7] = state["quality_heatmap"][patch_coords_y[valid_indices], patch_coords_x[valid_indices]]

        if self.knn_fn is not None and len(params["means"]) > self.redundancy_knn:
            dists, idxs = self.knn_fn(means3d_subset, K=self.redundancy_knn + 1)
            neighbor_idxs = idxs[:, 1:]

            features[:, 8] = dists[:, 1:].mean(dim=-1) / state["scene_scale"]

            neighbor_scales = torch.exp(params["scales"][neighbor_idxs]).max(dim=-1).values
            neighbor_opacities = torch.sigmoid(params["opacities"][neighbor_idxs].squeeze(-1))
            neighbor_sh0 = params["sh0"][neighbor_idxs].squeeze(-2)

            sh0_subset = params["sh0"][subset_mask]

            features[:, 9] = neighbor_scales.mean(dim=-1) / state["scene_scale"]
            features[:, 10] = neighbor_opacities.mean(dim=-1)
            features[:, 11] = torch.norm(neighbor_sh0 - sh0_subset, dim=-1).mean(dim=-1)

        current_grad = state["grad2d"][subset_mask] / state["count"][subset_mask].clamp_min(1)
        features[:, 12] = current_grad

        if state.get("prev_grad2d") is not None and state.get("prev_opacity") is not None:
            time_delta = self.refine_every
            prev_grad_subset = state["prev_grad2d"][subset_mask]
            prev_opacity_subset = state["prev_opacity"][subset_mask]

            features[:, 13] = (current_grad - prev_grad_subset) / time_delta
            features[:, 14] = (opacities_subset - prev_opacity_subset) / time_delta

        if state.get("significance") is not None and state["significance"].numel() == len(subset_mask):
            features[:, 15] = state["significance"][subset_mask]

        features[:, 16] = step / self.refine_stop_iter

        if state.get("age") is not None:
            features[:, 17] = state["age"][subset_mask].float() / self.refine_stop_iter

        return torch.nan_to_num(features, 0.0), pixel_coords_x, pixel_coords_y, patch_coords_x, patch_coords_y, valid_mask


    @torch.no_grad()
    def _get_graph_representation(
            self, params: dict, state: dict, subset_mask: Tensor, step: int
    ) -> tuple[Tensor, tuple, Tensor]:
        raw_features, px, py, ptx, pty, valid_mask = self._get_raw_features(params, state, subset_mask, step)

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

        self.gnn_net.eval()
        graph_embeddings = self.gnn_net(raw_features, edge_index, edge_attr)

        return graph_embeddings, (px, py, ptx, pty, valid_mask), raw_features

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

        if self.ac_net is None and self.use_learned_strategy:
            self._initialize_learning_components(params["means"].device)
        if state.get("age") is None:
            state["age"] = torch.zeros(params["means"].shape[0], dtype=torch.int32, device=params["means"].device)
        if state.get("significance") is None:
            state["significance"] = torch.zeros(params["means"].shape[0], device=params["means"].device)
        if self.nrqm_model is None:
            self.nrqm_model = PatchBasedNRQM().to(params["means"].device)

        state["l1_loss_map"] = info.get("l1_loss_map")

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


        self._process_hindsight_buffers(state, step)

        self._update_state(params, state, info, packed=packed)


        if step > self.refine_start_iter:
            if step % self.refine_every == 0:
                t = time.time()
                n_dupli, n_split, n_prune, n_merge = self._update_geometry(params, optimizers, state, step)

                if self.verbose:
                    print(f"Geometry updated at step {step}. Now having {len(params['means'])} GSs. Took {time.time() - t:.2f}s.")
                    print(
                        f"Step {step}: {n_dupli} GSs duplicated, {n_split} GSs split, {n_prune} GSs pruned, {n_merge} GSs merged."
                    )

        if step > self.bootstrap_steps:
            if step % self.densify_learn_every == 0:
                self._train_densification_agent(state)
            if self.use_learned_pruning and step % self.pruning_learn_every == 0:
                self._train_pruning_agent(state)

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

        # normalize grads to [-1, 1] screen space
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

        cam_indices = torch.randint(0, info['n_cameras'], (self.num_uncertainty_views,))

        novel_camtoworlds = info['camtoworlds'][cam_indices]
        novel_Ks = info['Ks'][cam_indices]

        novel_render_pkg, _, _ = self.rasterizer_fn(
            means=params["means"],
            quats=params["quats"],
            scales=params["scales"],
            opacities=params["opacities"],
            colors=torch.cat([params["sh0"], params["shN"]], 1),
            Ks=novel_Ks,
            width=width,
            height=height,
            sh_degree=sh_degree_to_use,
            camtoworlds=novel_camtoworlds,
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
        quality_heatmap = patch_scores.view(num_patches_h, -1)
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

        gt_image = info["pixels"]
        gt_ids = info["image_ids"]
        gt_ks = info["Ks"]
        camtoworlds_gt = info["camtoworlds"]
        rendered_train_view, _, _ = self.rasterizer_fn(
            means=params["means"], quats=params["quats"], scales=params["scales"],
            opacities=params["opacities"], colors=torch.cat([params["sh0"], params["shN"]], 1),
            Ks=gt_ks, width=width, height=height, sh_degree=sh_degree_to_use,
            camtoworlds=camtoworlds_gt, image_ids=gt_ids,
        )

        if self.lpips_metric is not None:
            rendered_p = rendered_train_view.permute(0, 3, 1, 2)
            gt_p = gt_image.permute(0, 3, 1, 2)

            with torch.no_grad():
                feats_render = self.lpips_metric(rendered_p, gt_p)

            photometric_error_map = feats_render
        else:
            photometric_error_map = torch.abs(rendered_train_view - gt_image).mean(dim=-1).squeeze(0)


        if state["photometric_error_map"] is None:
            state["photometric_error_map"] = photometric_error_map
        else:
            state["photometric_error_map"] = torch.lerp(
                photometric_error_map, state["photometric_error_map"], self.nrqm_ema_decay
            )

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


    def _process_hindsight_buffers(self, state: dict, current_step: int):
        while state["densify_hindsight_buffer"] and \
                (current_step - state["densify_hindsight_buffer"][0]["step"]) >= self.densify_hindsight_delay:

            exp = state["densify_hindsight_buffer"].popleft()

            if state.get("l1_loss_map") is None or \
                    state.get("quality_heatmap") is None or \
                    state.get("geom_uncertainty_map") is None:
                continue

            px, py = exp["px"], exp["py"]
            ptx, pty = exp["ptx"], exp["pty"]
            action = exp["action"]

            current_error = state["l1_loss_map"][max(0, py-2):py+3, max(0, px-2):px+3].mean()
            reward_photo = exp["initial_error"] - current_error # Positive reward if error decreased

            current_quality = state["quality_heatmap"][pty, ptx]
            reward_quality = exp["initial_quality"] - current_quality # Positive if NRQM score improved (i.e., raw score decreased)

            current_uncertainty = state["geom_uncertainty_map"][py, px]
            reward_uncertainty = exp["initial_uncertainty"] - current_uncertainty # Positive if uncertainty decreased

            action_cost = 0.0
            if action == 1 or action == 2:
                action_cost = -self.action_cost_weight
            elif action == 3:
                action_cost = self.action_cost_weight

            final_reward = (self.w_photometric * reward_photo +
                            self.w_quality * reward_quality +
                            self.w_uncertainty * reward_uncertainty +
                            action_cost)

            if len(state["densify_replay_buffer"]) < state["densify_replay_buffer"].capacity:
                state["densify_replay_buffer"].add((
                    exp["features"],
                    exp["action"],
                    final_reward.clone().detach(),
                    exp["log_prob"]
                ))

        if self.use_learned_pruning:
            while state["pruning_hindsight_buffer"] and \
                    (current_step - state["pruning_hindsight_buffer"][0]["step"]) >= self.pruning_hindsight_delay:

                exp = state["pruning_hindsight_buffer"].popleft()

                if state.get("l1_loss_map") is None:
                    continue

                px, py = exp["px"], exp["py"]

                current_error = state["l1_loss_map"][max(0, py-2):py+3, max(0, px-2):px+3].mean()

                pruning_reward = exp["initial_error"] - current_error

                if len(state["pruning_replay_buffer"]) < state["pruning_replay_buffer"].capacity:
                    state["pruning_replay_buffer"].add((
                        exp["features"],
                        exp["action"],
                        pruning_reward.clone().detach(),
                        exp["log_prob"]
                    ))

    def _train_densification_agent(self, state: dict[str, Any]):
        if len(state["densify_replay_buffer"]) < 256: return

        self.gnn_net.train()
        self.ac_net.train()
        device = next(self.ac_net.parameters()).device

        batch, tree_idxs, is_weights = state["densify_replay_buffer"].sample(256)
        is_weights = torch.tensor(is_weights, device=device, dtype=torch.float32)

        ac_inputs = torch.stack([x[0] for x in batch]).to(device)
        actions_taken = torch.tensor([x[1] for x in batch], device=device, dtype=torch.int64)
        rewards = torch.stack([x[2] for x in batch]).to(device)
        old_log_probs = torch.stack([x[3] for x in batch]).to(device)

        action_logits, predicted_values, _ = self.ac_net(ac_inputs)

        advantage = rewards - predicted_values.detach()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        critic_loss_unreduced = F.mse_loss(predicted_values, rewards, reduction="none")

        new_dist = Categorical(logits=action_logits)
        new_log_probs = new_dist.log_prob(actions_taken)
        ratio = torch.exp(new_log_probs - old_log_probs)

        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip_epsilon, 1.0 + self.ppo_clip_epsilon) * advantage
        actor_loss_unreduced = -torch.min(surr1, surr2)

        entropy_loss_unreduced = -self.entropy_loss_weight * new_dist.entropy()

        total_loss_unreduced = (
                critic_loss_unreduced +
                self.actor_loss_weight * actor_loss_unreduced +
                entropy_loss_unreduced
        )

        loss = (total_loss_unreduced * is_weights).mean()

        self.densify_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ac_net.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.gnn_net.parameters(), 1.0)
        self.densify_optimizer.step()

        td_errors = (rewards - predicted_values).abs().detach().cpu().numpy()
        state["densify_replay_buffer"].update_priorities(tree_idxs, td_errors)

    def _train_pruning_agent(self, state: dict[str, Any]):
        if len(state["pruning_replay_buffer"]) < 128: return

        self.gnn_net.train()
        self.pruning_ac_net.train()
        device = next(self.pruning_ac_net.parameters()).device

        batch, tree_idxs, is_weights = state["pruning_replay_buffer"].sample(128)
        is_weights = torch.tensor(is_weights, device=device, dtype=torch.float32)

        ac_inputs = torch.stack([x[0] for x in batch]).to(device)
        actions_taken = torch.tensor([x[1] for x in batch], device=device, dtype=torch.float32) # Bernoulli actions are float
        rewards = torch.stack([x[2] for x in batch]).to(device)
        old_log_probs = torch.stack([x[3] for x in batch]).to(device)

        action_logits, predicted_values = self.pruning_ac_net(ac_inputs)

        advantage = rewards - predicted_values.detach()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        critic_loss_unreduced = F.mse_loss(predicted_values, rewards, reduction="none")

        new_dist = Bernoulli(logits=action_logits)
        new_log_probs = new_dist.log_prob(actions_taken)
        ratio = torch.exp(new_log_probs - old_log_probs)

        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip_epsilon, 1.0 + self.ppo_clip_epsilon) * advantage
        actor_loss_unreduced = -torch.min(surr1, surr2)

        entropy_loss_unreduced = -self.entropy_loss_weight * new_dist.entropy()

        total_loss_unreduced = (
                critic_loss_unreduced +
                self.actor_loss_weight * actor_loss_unreduced +
                entropy_loss_unreduced
        )

        loss = (total_loss_unreduced * is_weights).mean()

        self.densify_optimizer.zero_grad()
        self.pruning_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.pruning_ac_net.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.gnn_net.parameters(), 1.0)
        self.densify_optimizer.step()
        self.pruning_optimizer.step()

        td_errors = (rewards - predicted_values).abs().detach().cpu().numpy()
        state["pruning_replay_buffer"].update_priorities(tree_idxs, td_errors)

    @torch.no_grad()
    def _update_geometry(self, params: dict, optimizers: dict, state: dict, step: int) -> Tuple[int, int, int, int]:
        """
        Robust implementation using a state-independent action resolution strategy.
        Conflicts are resolved before any modifications, ensuring index stability.
        """
        initial_num_gaussians = len(params["means"])
        device = params["means"].device

        # 1. Agent Decision Making
        subset_mask = torch.rand(initial_num_gaussians, device=device) < self.subset_fraction
        if subset_mask.sum() == 0: return 0, 0, 0, 0

        original_subset_indices = torch.where(subset_mask)[0]

        graph_embeddings, coords, _ = self._get_graph_representation(params, state, subset_mask, step)
        if graph_embeddings is None: return 0, 0, 0, 0
        px, py, ptx, pty, valid_mask_subset = coords

        avg_scale = torch.exp(params['scales']).mean() / state['scene_scale']
        std_scale = torch.exp(params['scales']).std() / state['scene_scale']
        global_context = torch.tensor([
            min(initial_num_gaussians / 500_000.0, 1.0),
            state.get("quality_heatmap", torch.zeros(1, device=device)).mean(),
            state.get("significance", torch.zeros(1, device=device)).mean(),
            avg_scale, std_scale
        ], device=device).float()
        expanded_global_context = global_context.unsqueeze(0).expand(graph_embeddings.shape[0], -1)
        ac_input = torch.cat([graph_embeddings, expanded_global_context], dim=-1)

        prune_actions = torch.zeros(len(original_subset_indices), dtype=torch.bool, device=device)
        if self.use_learned_pruning:
            self.pruning_ac_net.eval()
            prune_logits, _ = self.pruning_ac_net(ac_input)
            prune_dist = Bernoulli(logits=prune_logits)
            prune_actions = (prune_dist.sample() == 1)

        self.ac_net.eval()
        action_logits, _, continuous_params = self.ac_net(ac_input)
        progress = max(0.0, (step - self.bootstrap_steps) / self.exploration_decay_steps)
        epsilon = self.end_exploration_epsilon + (self.start_exploration_epsilon - self.end_exploration_epsilon) * (1 - progress)
        densify_dist = Categorical(logits=action_logits)
        if torch.rand(1).item() < epsilon:
            densify_actions = torch.randint(0, 4, (len(original_subset_indices),), device=device)
        else:
            densify_actions = densify_dist.sample()

        # 2. Resolve Action Conflicts and Create Final, Disjoint Masks
        final_actions = torch.zeros_like(densify_actions)
        final_actions[densify_actions == 1] = 1 # Split
        final_actions[densify_actions == 2] = 2 # Duplicate
        final_actions[densify_actions == 3] = 3 # Merge
        final_actions[prune_actions] = 4      # Prune

        age_in_steps = state["age"][original_subset_indices]
        significance = state["significance"][original_subset_indices]
        prune_heuristic_mask = (age_in_steps < 800) | (significance > self.prune_significance_threshold)
        final_actions[(final_actions == 4) & prune_heuristic_mask] = 0 # Veto pruning

        final_prune_mask_subset = (final_actions == 4)
        final_merge_mask_subset = (final_actions == 3)
        final_split_mask_subset = (final_actions == 1)
        final_dupe_mask_subset = (final_actions == 2)

        # 3. Execute Operations Sequentially
        state_to_modify = {k: v for k, v in state.items() if k in ["grad2d", "count", "radii", "age", "significance"]}

        # A. PRUNE
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
        else: # If nothing merged, it's an identity map for prune survivors
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
                split_continuous_params = split_continuous_params[valid_mask]

        n_split = global_split_mask.sum().item()
        if n_split > 0:
            split(params, optimizers, state_to_modify, global_split_mask,
                  split_ratios=split_continuous_params[:, 0],
                  directions=split_continuous_params[:, 2:5])
        num_gaussians_post_split = len(params["means"])

        # D. DUPLICATE
        dupe_orig_indices = original_subset_indices[final_dupe_mask_subset]
        dupe_continuous_params = continuous_params[final_dupe_mask_subset]
        n_dupli = 0

        if len(dupe_orig_indices) > 0:
            # Step-by-step filtering to find final duplication candidates
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
                    duplication_offsets = dupe_dirs * offset_magnitudes.unsqueeze(-1)
                    duplicate(params, optimizers, state_to_modify, global_dupe_mask, offsets=duplication_offsets)

        state.update(state_to_modify)
        if n_dupli > 0: state["age"][-n_dupli:] = 0
        if n_split > 0: state["age"][-(n_split*2):-n_split] = 0; state["age"][-n_split:] = 0

        return n_dupli, n_split, n_prune, n_merge_pairs