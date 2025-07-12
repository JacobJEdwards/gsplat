from collections import deque
from dataclasses import dataclass, field
import time
from typing import Any, Dict, Union
from approx_topk import topk

import piq
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Categorical

from utils import knn_with_ids
from gsplat.strategy.ops import (
    duplicate,
    remove,
    reset_opa,
)
from gsplat.strategy.default import DefaultStrategy
from ops import split

class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dim: int = 18, mlp_width: int = 64):
        super().__init__()
        self.base_net = nn.Sequential(
            nn.Linear(input_dim, mlp_width),
            nn.LayerNorm(mlp_width),
            nn.PReLU(),
            nn.Linear(mlp_width, mlp_width),
            nn.LayerNorm(mlp_width),
            nn.PReLU(),
        )

        self.critic_head = nn.Linear(mlp_width, 1)

        self.actor_discrete_head = nn.Linear(mlp_width, 4)

        self.actor_continuous_head = nn.Linear(mlp_width, 2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        base_features = self.base_net(x)

        value = self.critic_head(base_features).squeeze(-1)

        action_logits = self.actor_discrete_head(base_features)

        continuous_params = self.actor_continuous_head(base_features)

        split_ratio = 1.2 + 2.0 * torch.sigmoid(continuous_params[:, 0])
        dupe_offset_mag = torch.sigmoid(continuous_params[:, 1])

        processed_continuous_params = torch.stack([split_ratio, dupe_offset_mag], dim=-1)

        return action_logits, value, processed_continuous_params

class DensificationNetwork(nn.Module):
    def __init__(self, input_dim: int = 18, mlp_width: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, mlp_width),
            nn.LayerNorm(mlp_width),
            nn.PReLU(),
            nn.Linear(mlp_width, mlp_width),
            nn.LayerNorm(mlp_width),
            nn.PReLU(),
            nn.Linear(mlp_width, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

class PruningNetwork(nn.Module):
    def __init__(self, input_dim: int = 18, mlp_width: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, mlp_width),
            nn.LayerNorm(mlp_width),
            nn.PReLU(),
            nn.Linear(mlp_width, mlp_width),
            nn.LayerNorm(mlp_width),
            nn.PReLU(),
            nn.Linear(mlp_width, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

class PatchBasedNRQM(nn.Module):
    def __init__(self):
        super().__init__()
        self.brisque = piq.BRISQUELoss(reduction='none', data_range=1.)

    def forward(self, image_patches: torch.Tensor) -> torch.Tensor:
        return self.brisque(image_patches)


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

    max_splits_per_step: int = 20000
    max_duplications_per_step: int = 20000

    subset_fraction: float = 0.2
    max_densification_subset: int = 200_000

    nrqm_patch_size: int = 32
    nrqm_stagnation_threshold: float = 0.3
    nrqm_prune_stagnant_after: int = 15
    nrqm_ema_decay: float = 0.9
    prune_significance_threshold: float = 0.01

    anisotropic_split: bool = True
    prune_redundant: bool = True
    redundancy_knn: int = 5
    redundancy_overlap_thresh: float = 0.6
    redundancy_color_thresh: float = 0.1

    use_geom_uncertainty: bool = True
    num_uncertainty_views: int = 3
    geom_uncertainty_thresh: float = 0.002

    photometric_error_thresh: float = 0.1

    use_learned_densification: bool = True
    bootstrap_steps: int = 5000
    learn_every: int = 200
    hindsight_delay: int = 100

    actor_loss_weight: float = 1.0
    entropy_loss_weight: float = 0.01
    continuous_loss_weight: float = 0.5

    use_learned_pruning: bool = True
    pruning_bootstrap_steps: int = 5000
    pruning_learn_every: int = 500
    learned_prune_threshold: float = 0.5

    pruning_net: Any = field(default=None, repr=False)
    pruning_optimizer: Any = field(default=None, repr=False)

    w_photometric: float = 0.6
    w_quality: float = -0.2
    w_uncertainty: float = 0.2

    rasterizer_fn: Any = field(default=None, repr=False)
    nrqm_model: Any = field(default=None, repr=False)
    knn_fn: Any = field(default=None, repr=False)
    lpips_metric: Any = field(default=None, repr=False)

    densification_net: Any = field(default=None, repr=False)
    densification_optimizer: Any = field(default=None, repr=False)

    ac_net: Any = field(default=None, repr=False)
    ac_optimizer: Any = field(default=None, repr=False)

    start_asc_grad2d: float = 0.0002
    end_asc_grad2d: float = 0.001

    def initialize_state(self, scene_scale: float = 1.0) -> dict[str, Any]:
        state = super().initialize_state(scene_scale)
        state.update({
            "quality_heatmap": None,
            "view_proj_matrix": None,
            "stagnation_count": None,
            "last_nrqm_step": -1,
            "last_prune_count": -1,
            "dynamic_grow_grad2d": self.grow_grad2d,
            "photometric_error_map": None,
            "geom_uncertainty_map": None,
            "replay_buffer": deque(maxlen=20_000),
            "hindsight_buffer": deque(maxlen=5_000),
            "pruning_replay_buffer": deque(maxlen=20_000),
            "significance": None,
            "prev_grad2d": None,
            "prev_opacity": None,
            "age": None,

        })

        return state

    def _initialize_learning_components(self, device) -> None:
        if self.use_learned_densification and self.ac_net is None:
            self.ac_net = ActorCriticNetwork(input_dim=18).to(device)
            self.ac_optimizer = torch.optim.AdamW(
                self.ac_net.parameters(), lr=1e-4, weight_decay=1e-5
            )
            if self.verbose:
                print("Initialized Actor-Critic Network.")

    def _initialize_pruning_components(self, device) -> None:
        if self.use_learned_pruning and self.pruning_net is None:
            self.pruning_net = PruningNetwork().to(device)
            self.pruning_optimizer = torch.optim.AdamW(
                self.pruning_net.parameters(), lr=1e-4, weight_decay=1e-5
            )

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

        if state.get("age") is None:
            state["age"] = torch.zeros(params["means"].shape[0], dtype=torch.int32, device=params["means"].device)

        if state.get("significance") is None:
            state["significance"] = torch.zeros(params["means"].shape[0], device=params["means"].device)


        if self.use_learned_densification and self.densification_net is None:
            self._initialize_learning_components(params["means"].device)
        if self.use_learned_pruning and self.pruning_net is None:
            self._initialize_pruning_components(params["means"].device)
        if self.nrqm_model is None:
            self.nrqm_model = PatchBasedNRQM().to(params["means"].device)
        if self.knn_fn is None:
            self.knn_fn = knn_with_ids

        if "gaussian_contribution" in info:
            current_significance = info["gaussian_contribution"]

            if state["significance"] is None or state["significance"].shape[0] != current_significance.shape[0]:
                state["significance"] = torch.zeros_like(current_significance)

            if state["significance"].device != current_significance.device:
                state["significance"] = state["significance"].to(current_significance.device)

            state["significance"] = 0.9 * state["significance"] + 0.1 * current_significance

        should_update_maps = (state["last_nrqm_step"] == -1 and step > 0) or \
                             (step % self.nrqm_every == 0)

        if should_update_maps:
            t = time.time()
            self._update_quality_map(params, state, info)
            if self.verbose:
                print(f"Updated quality maps in {time.time() - t:.2f} seconds.")
            state["last_nrqm_step"] = step

        if self.use_learned_densification:
            self._process_hindsight_buffer(state, step)

        self._update_state(params, state, info, packed=packed)

        if step > self.refine_start_iter:
            state["age"] += 1

        if step > self.refine_start_iter:
            if step % self.refine_every == 0:
                t = time.time()
                n_dupli, n_split, n_prune = self._update_geometry(params, optimizers, state, step)

                if self.verbose:
                    print(f"Geometry updated at step {step}. Now having {len(params['means'])} GSs. Took {time.time() - t:.2f}s.")
                    print(
                        f"Step {step}: {n_dupli} GSs duplicated, {n_split} GSs split, {n_prune} GSs pruned."
                    )
                # n_dupli, n_split = self._grow_gs(params, optimizers, state, step)
                # if self.verbose:
                #     print(
                #         f"Step {step}: {n_dupli} GSs duplicated, {n_split} GSs split. "
                #         f"Now having {len(params['means'])} GSs."
                #         f"Took {time.time() - t:.2f} seconds."
                #     )


        if step % self.reset_every == 0 and step > 0:
            reset_opa(params=params, optimizers=optimizers, state=state, value=self.prune_opa * 2.0)


        if self.use_learned_densification and step > 1000 and step % self.learn_every == 0:
            t = time.time()
            self._train_actor_critic(state)
            if self.verbose:
                print(f"Trained Densification Network at step {step} in {time.time() - t:.2f} seconds.")

    def _update_state(
            self,
            params: Union[dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
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
        main_novel_view_idx = cam_indices[0].item()

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

    @torch.no_grad()
    def _get_gaussian_features(
            self, params: dict, state: dict, subset_mask: Tensor, step: int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        num_subset = subset_mask.sum().item()
        device = params["means"].device

        means3d_subset = params["means"][subset_mask]
        if state.get("photometric_error_map") is None:
            if self.verbose:
                print("Skipping feature extraction: photometric error map is not available.")

            return None, None, None, None, None

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

    def _process_hindsight_buffer(self, state, current_step):
        if state.get("photometric_error_map") is None:
            if self.verbose:
                print("Skipping hindsight processing: photometric error map is not available.")

            return

        while state["hindsight_buffer"] and (current_step - state["hindsight_buffer"][0]["step"]) >= self.hindsight_delay:
            experience = state["hindsight_buffer"].popleft()
            px, py = experience["pixel_coords"]
            patch_x, patch_y = experience["patch_coords"]

            current_error = state["photometric_error_map"][max(0, py-2):py+3, max(0, px-2):px+3].mean()
            reward_photo = experience["initial_error"] - current_error

            current_quality = state["quality_heatmap"][patch_y, patch_x]
            current_uncertainty = state["geom_uncertainty_map"][py, px]
            reward_quality = experience["initial_quality"] - current_quality
            reward_uncertainty = experience["initial_uncertainty"] - current_uncertainty

            final_reward = (self.w_photometric * reward_photo +
                            self.w_quality * reward_quality +
                            self.w_uncertainty * reward_uncertainty)

            if len(state["replay_buffer"]) < state["replay_buffer"].maxlen:
                state["replay_buffer"].append((experience["features"], experience["action"], final_reward.clone().detach()))

    def _train_actor_critic(self, state: dict[str, Any]):
        if len(state["replay_buffer"]) < 256:
            return

        self.ac_net.train()
        device = next(self.ac_net.parameters()).device

        batch_indices = torch.randint(0, len(state["replay_buffer"]), (256,))
        batch = [state["replay_buffer"][i] for i in batch_indices]

        features = torch.stack([x[0] for x in batch]).to(device)
        actions_taken = torch.stack([x[1] for x in batch]).to(device)
        rewards = torch.stack([x[2] for x in batch]).to(device)

        action_logits, predicted_values, _ = self.ac_net(features)

        critic_loss = F.mse_loss(predicted_values, rewards)

        advantage = (rewards - predicted_values).detach()

        log_probs = F.log_softmax(action_logits, dim=-1)
        log_prob_for_action = log_probs.gather(1, actions_taken.unsqueeze(-1)).squeeze(-1)

        policy_gradient_loss = -(log_prob_for_action * advantage).mean()

        probs = F.softmax(action_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        entropy_loss = -self.entropy_loss_weight * entropy

        loss = critic_loss + self.actor_loss_weight * policy_gradient_loss + entropy_loss

        self.ac_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ac_net.parameters(), 1.0)
        self.ac_optimizer.step()

        if self.verbose:
            print(f"AC Loss: {loss.item():.4f} (Critic: {critic_loss.item():.4f}, PG: {policy_gradient_loss.item():.4f})")

    @torch.no_grad()
    def _update_geometry(self, params: dict, optimizers: dict, state: dict, step: int) -> tuple[int, int, int]:
        num_gaussians = len(params["means"])
        device = params["means"].device

        subset_mask = torch.rand(num_gaussians, device=device) < self.subset_fraction
        if subset_mask.sum() == 0:
            return

        subset_indices = torch.where(subset_mask)[0]

        features, px, py, ptx, pty, valid_mask_subset = self._get_gaussian_features(params, state, subset_mask, step)
        if features is None:
            return

        self.ac_net.eval()
        action_logits, _, continuous_params = self.ac_net(features)

        dist = Categorical(logits=action_logits)
        actions = dist.sample()


        subset_is_prune = (actions == 0)
        subset_is_dupe  = (actions == 2)
        subset_is_split = (actions == 3)

        subset_scales = torch.exp(params["scales"][subset_indices])
        is_too_small_to_split = subset_scales.max(dim=-1).values <= self.grow_scale3d * state["scene_scale"]
        is_too_big_to_dupe = ~is_too_small_to_split

        subset_is_split[is_too_small_to_split] = False
        subset_is_dupe[is_too_big_to_dupe] = False

        global_is_prune = torch.zeros(num_gaussians, dtype=torch.bool, device=device)
        global_is_prune[subset_indices[subset_is_prune]] = True

        pruned_mask_on_subset = torch.zeros(len(subset_indices), dtype=torch.bool, device=device)
        pruned_mask_on_subset[subset_is_prune] = True

        final_subset_is_dupe = subset_is_dupe & ~pruned_mask_on_subset
        final_subset_is_split = subset_is_split & ~pruned_mask_on_subset

        final_split_ratios = continuous_params[final_subset_is_split, 0]

        for i, original_idx in enumerate(subset_indices):
            if valid_mask_subset[i]:
                experience = {
                    "step": step, "features": features[i].detach().cpu(),
                    "action": actions[i].detach().cpu(),
                    "continuous_params": continuous_params[i].detach().cpu(),
                    "pixel_coords": (px[i], py[i]), "patch_coords": (ptx[i], pty[i]),
                    "initial_error": state["photometric_error_map"][max(0, py[i]-2):py[i]+3, max(0, px[i]-2):px[i]+3].mean(),
                    "initial_quality": state["quality_heatmap"][pty[i], ptx[i]],
                    "initial_uncertainty": state["geom_uncertainty_map"][py[i], px[i]]
                }
                state["hindsight_buffer"].append(experience)

        per_gaussian_state_keys = ["grad2d", "count", "radii", "stagnation_count", "prev_grad2d", "prev_opacity", "significance", "age"]
        state_to_modify = {k: v for k, v in state.items() if k in per_gaussian_state_keys and v is not None}

        n_prune = global_is_prune.sum().item()

        if global_is_prune.any():
            remove(params, optimizers, state_to_modify, global_is_prune)

        post_prune_indices = torch.arange(num_gaussians, device=device)[~global_is_prune]

        orig_to_post_prune_map = torch.full((num_gaussians,), -1, dtype=torch.long, device=device)
        orig_to_post_prune_map[post_prune_indices] = torch.arange(len(post_prune_indices), device=device)

        post_prune_subset_indices = orig_to_post_prune_map[subset_indices]


        num_after_prune = len(params["means"])
        global_is_dupe_after_prune = torch.zeros(num_after_prune, dtype=torch.bool, device=device)
        global_is_split_after_prune = torch.zeros(num_after_prune, dtype=torch.bool, device=device)

        dupe_subset_indices_post_prune = post_prune_subset_indices[final_subset_is_dupe]
        split_subset_indices_post_prune = post_prune_subset_indices[final_subset_is_split]

        global_is_dupe_after_prune[dupe_subset_indices_post_prune[dupe_subset_indices_post_prune != -1]] = True
        global_is_split_after_prune[split_subset_indices_post_prune[split_subset_indices_post_prune != -1]] = True

        n_dupli = global_is_dupe_after_prune.sum().item()
        if n_dupli > 0:
            duplicate(params, optimizers, state_to_modify, global_is_dupe_after_prune)

        n_split = global_is_split_after_prune.sum().item()
        if n_split > 0:
            is_split_after_dup = torch.cat([global_is_split_after_prune, torch.zeros(n_dupli, dtype=torch.bool, device=device)])
            split(params, optimizers, state_to_modify, is_split_after_dup,
                  anisotropic=self.anisotropic_split, revised_opacity=self.revised_opacity,
                  split_ratios=final_split_ratios)

        if n_dupli > 0:
            state["age"][-n_dupli:] = 0

        state.update(state_to_modify)

        return n_dupli, n_split, n_prune


    @torch.no_grad()
    def _grow_gs(
            self,
            params: dict[str, torch.nn.Parameter] | torch.nn.ParameterDict,
            optimizers: dict[str, torch.optim.Optimizer],
            state: dict[str, Any],
            step: int,
    ) -> tuple[int, int]:
        num_gaussians = len(params["means"])
        device = params["means"].device

        subset_size = min(num_gaussians, self.max_densification_subset)
        selection = torch.randperm(num_gaussians, device=device)[:subset_size]

        subset_mask = torch.rand(num_gaussians, device=device) < self.subset_fraction
        subset_mask[selection] = True
        subset_indices = torch.where(subset_mask)[0]

        if subset_indices.numel() == 0:
            if self.verbose:
                print("Skipping Gaussian growth: no valid subset indices.")

            return 0, 0

        t = time.time()
        features_subset, px, py, ptx, pty, valid_mask_subset = self._get_gaussian_features(params, state, subset_mask, step)
        if self.verbose:
            print(f"Extracted features for {len(subset_indices)} Gaussians in {time.time() - t:.2f} seconds.")

        if features_subset is None:
            if self.verbose:
                print("Skipping Gaussian growth: no valid features extracted.")

            return 0, 0

        if self.use_learned_densification and step >= self.bootstrap_steps:
            self.densification_net.eval()
            with torch.no_grad():
                t = time.time()
                utility_scores = self.densification_net(features_subset).squeeze()
                if self.verbose:
                    print(f"Computed utility scores for {len(subset_indices)} Gaussians in {time.time() - t:.2f} seconds.")
        else:
            utility_scores = features_subset[:, 12]

        if self.use_learned_densification:
            if self.verbose:
                print(f"Appending {len(subset_indices)} Gaussians to hindsight buffer.")
            for i, original_idx in enumerate(subset_indices):
                if valid_mask_subset[i]:
                    initial_error = state["photometric_error_map"][max(0, py[i]-2):py[i]+3, max(0, px[i]-2):px[i]+3].mean()
                    initial_quality = state["quality_heatmap"][pty[i], ptx[i]]
                    initial_uncertainty = state["geom_uncertainty_map"][py[i], px[i]]

                    state["hindsight_buffer"].append({
                        "step": step, "features": features_subset[i].detach().cpu(),
                        "pixel_coords": (px[i], py[i]), "patch_coords": (ptx[i], pty[i]),
                        "initial_error": initial_error, "initial_quality": initial_quality,
                        "initial_uncertainty": initial_uncertainty
                    })

        scales_subset = torch.exp(params["scales"][subset_mask])
        is_small_subset = scales_subset.max(dim=-1).values <= self.grow_scale3d * state["scene_scale"]

        dupli_candidates_mask = is_small_subset
        split_candidates_mask = ~is_small_subset

        is_dupli = torch.zeros(num_gaussians, dtype=torch.bool, device=device)
        dupli_indices_in_subset = torch.where(dupli_candidates_mask)[0]
        n_dupli = min(len(dupli_indices_in_subset), self.max_duplications_per_step)
        if n_dupli > 0:
            t = time.time()
            dupli_scores = utility_scores[dupli_indices_in_subset]
            _, top_indices = topk(dupli_scores, n_dupli, dim=-1)
            final_dupli_indices_in_subset = dupli_indices_in_subset[top_indices]
            is_dupli[subset_indices[final_dupli_indices_in_subset]] = True
            if self.verbose:
                print(f"Selected {n_dupli} Gaussians for duplication in {time.time() - t:.2f} seconds.")

        is_split = torch.zeros(num_gaussians, dtype=torch.bool, device=device)
        split_indices_in_subset = torch.where(split_candidates_mask)[0]
        n_split = min(len(split_indices_in_subset), self.max_splits_per_step)
        if n_split > 0:
            t = time.time()
            split_scores = utility_scores[split_indices_in_subset]
            _, top_indices = topk(split_scores, n_split, dim=-1)
            final_split_indices_in_subset = split_indices_in_subset[top_indices]
            is_split[subset_indices[final_split_indices_in_subset]] = True
            if self.verbose:
                print(f"Selected {n_split} Gaussians for splitting in {time.time() - t:.2f} seconds.")

        per_gaussian_state_keys = ["grad2d", "count", "radii", "stagnation_count", "prev_grad2d", "prev_opacity", "significance"]
        state_to_densify = {k: v for k, v in state.items() if k in per_gaussian_state_keys and v is not None}

        if n_dupli > 0:
            t = time.time()
            duplicate(params, optimizers, state_to_densify, is_dupli)
            if self.verbose:
                print(f"Duplicated {n_dupli} Gaussians in {time.time() - t:.2f} seconds.")
        if n_split > 0:
            t = time.time()
            is_split_after_dup = torch.cat([is_split, torch.zeros(n_dupli, dtype=torch.bool, device=device)])
            split(params, optimizers, state_to_densify, is_split_after_dup, anisotropic=self.anisotropic_split,
                  revised_opacity=self.revised_opacity)
            if self.verbose:
                print(f"Split {n_split} Gaussians in {time.time() - t:.2f} seconds.")


        state.update(state_to_densify)

        if n_dupli > 0 or n_split > 0:
            if n_dupli > 0:
                state["age"][-n_dupli:] = 0

        state["prev_grad2d"] = state["grad2d"] / state["count"].clamp_min(1)
        state["prev_opacity"] = torch.sigmoid(params["opacities"].flatten())

        return n_dupli, n_split

    @torch.no_grad()
    def _prune_gs(
            self,
            params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
            optimizers: Dict[str, torch.optim.Optimizer],
            state: Dict[str, Any],
            step: int,
    ) -> int:
        is_prune_original = torch.sigmoid(params["opacities"].flatten()) < self.prune_opa
        if step > self.reset_every:
            is_too_big = (
                    torch.exp(params["scales"]).max(dim=-1).values
                    > self.prune_scale3d * state["scene_scale"]
            )
            if step < self.refine_scale2d_stop_iter:
                is_too_big |= state["radii"] > self.prune_scale2d
            is_prune_original |= is_too_big

        is_prune_stagnant = torch.zeros_like(is_prune_original)
        if state.get("quality_heatmap") is not None and step > state["last_nrqm_step"] and state["quality_heatmap"].numel() > 0:
            if state.get("stagnation_count") is None:
                state["stagnation_count"] = torch.zeros(params["means"].shape[0], dtype=torch.int, device=params["means"].device)

            grads = state["grad2d"] / state["count"].clamp_min(1)
            is_grad_low = grads < self.grow_grad2d

            means3d = params["means"]
            h = state["quality_heatmap"].shape[0] * self.nrqm_patch_size
            w = state["quality_heatmap"].shape[1] * self.nrqm_patch_size

            patch_coords_x, patch_coords_y, _, _, valid_mask = self._project_to_patch_coords(
                means3d, state["view_proj_matrix"], h, w
            )

            patch_coords_x, patch_coords_y, _, _, valid_mask = self._project_to_patch_coords(params["means"], state["view_proj_matrix"], h, w)

            is_in_low_quality_region = torch.zeros_like(is_prune_original)
            valid_indices = torch.where(valid_mask)[0]
            if valid_indices.numel() > 0:
                patch_scores = state["quality_heatmap"][patch_coords_y[valid_indices], patch_coords_x[valid_indices]]
                is_in_low_quality_region[valid_indices] = patch_scores < self.nrqm_stagnation_threshold
            is_stagnant = is_grad_low & is_in_low_quality_region

            state["stagnation_count"][is_stagnant] += 1
            state["stagnation_count"][~is_stagnant] = (state["stagnation_count"][~is_stagnant] - 1).clamp(min=0)
            is_prune_stagnant = state["stagnation_count"] > self.nrqm_prune_stagnant_after

        is_prune_significant = torch.zeros_like(is_prune_original)
        if "significance" in state and state["significance"] is not None and state["significance"].numel() > 0:
            if state["significance"].numel() == is_prune_original.numel():
                is_prune_significant = state["significance"] < self.prune_significance_threshold


        is_prune_redundant = torch.zeros_like(is_prune_original)
        if self.prune_redundant and self.knn_fn is not None:
            num_gaussians = len(params["means"])
            device = params["means"].device
            subset_mask = torch.rand(num_gaussians, device=device) < self.subset_fraction
            subset_indices_map = torch.where(subset_mask)[0]

            if subset_indices_map.numel() > self.redundancy_knn:
                subset_means = params["means"][subset_mask]
                dists_subset, idxs_subset = self.knn_fn(subset_means, K=self.redundancy_knn)
                original_neighbor_idxs = subset_indices_map[idxs_subset[:, 1:]]
                neighbor_scales = torch.exp(params["scales"][original_neighbor_idxs]).max(dim=-1).values

                neighbor_opacities = torch.sigmoid(params["opacities"][original_neighbor_idxs].squeeze(-1))
                neighbor_sh0 = params["sh0"][original_neighbor_idxs].squeeze(-2)
                scales_subset = torch.exp(params["scales"][subset_mask]).max(dim=-1).values
                opacities_subset = torch.sigmoid(params["opacities"][subset_mask].flatten())
                sh0_subset = params["sh0"][subset_mask].squeeze(1)

                overlap_mask = dists_subset[:, 1:] < (scales_subset.unsqueeze(1) + neighbor_scales) * self.redundancy_overlap_thresh
                color_dist = torch.norm(sh0_subset.unsqueeze(1) - neighbor_sh0, dim=-1)
                color_sim_mask = color_dist < self.redundancy_color_thresh
                is_less_opaque = opacities_subset.unsqueeze(1) < neighbor_opacities
                is_redundant_neighbor = overlap_mask & color_sim_mask & is_less_opaque
                is_prune_redundant[subset_indices_map] = is_redundant_neighbor.any(dim=1)

        if self.use_learned_pruning and "photometric_error_map" in state:
            subset_mask = torch.rand(params["means"].shape[0], device=params["means"].device) < 0.1 # Use a small subset
            t = time.time()
            features, _, _, _, _, _ = self._get_gaussian_features(params, state, subset_mask, step)
            if self.verbose:
                print(f"Extracted features for learned pruning in {time.time() - t:.2f} seconds.")

            if features is not None:
                heuristic_prune_labels = (
                        is_prune_original[subset_mask] |
                        is_prune_stagnant[subset_mask] |
                        is_prune_redundant[subset_mask] |
                        is_prune_significant[subset_mask]
                )

                for i in range(features.shape[0]):
                    if len(state["pruning_replay_buffer"]) < state["pruning_replay_buffer"].maxlen:
                        state["pruning_replay_buffer"].append(
                            (features[i].detach().cpu(), heuristic_prune_labels[i].detach().cpu())
                        )

        is_prune_learned = torch.zeros_like(is_prune_original)
        if self.use_learned_pruning and step > self.pruning_bootstrap_steps and "photometric_error_map" in state:
            self.pruning_net.eval()

            t = time.time()
            all_features, _, _, _, _, _ = self._get_gaussian_features(params, state, torch.ones_like(is_prune_original), step)
            if self.verbose:
                print(f"Extracted features for all Gaussians in {time.time() - t:.2f} seconds.")

            if all_features is not None:
                pruning_logits = self.pruning_net(all_features).squeeze(-1)
                pruning_probs = torch.sigmoid(pruning_logits)
                is_prune_learned = torch.rand_like(pruning_probs) < pruning_probs

        is_prune = is_prune_original | is_prune_stagnant | is_prune_redundant | is_prune_significant | is_prune_learned

        n_prune = is_prune.sum().item()
        if n_prune > 0:
            per_gaussian_state_keys = ["grad2d", "count", "radii", "stagnation_count", "prev_grad2d", "prev_opacity", "significance"]
            state_to_prune = {k: v for k, v in state.items() if k in per_gaussian_state_keys and v is not None}

            remove(params=params, optimizers=optimizers, state=state_to_prune, mask=is_prune)

            state.update(state_to_prune)

        return n_prune