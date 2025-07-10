from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from gsplat.strategy.ops import duplicate, remove, _update_param_with_optimizer
from gsplat.strategy.default import DefaultStrategy
from gsplat.utils import normalized_quat_to_rotmat


@torch.no_grad()
def split(
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Tensor],
        mask: Tensor,
        revised_opacity: bool = False,
        anisotropic: bool = False,
):
    """Inplace split the Gaussian with the given mask.

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        mask: A boolean mask to split the Gaussians.
        revised_opacity: Whether to use revised opacity formulation
          from arXiv:2404.06109. Default: False.
        anisotropic: Whether to split along the largest variance axis. Default: False.
    """
    device = mask.device
    sel = torch.where(mask)[0]
    rest = torch.where(~mask)[0]

    scales = torch.exp(params["scales"][sel])
    quats = F.normalize(params["quats"][sel], dim=-1)
    rotmats = normalized_quat_to_rotmat(quats)  # [N, 3, 3]

    if anisotropic:
        # Find the axis of largest variance (longest scale)
        largest_scale_idx = torch.argmax(scales, dim=1)
        samples = torch.zeros(2, len(scales), 3, device=device)

        # Create displacement vectors along the principal axes
        displacements = torch.zeros_like(scales)
        displacements[torch.arange(len(scales)), largest_scale_idx] = scales[torch.arange(len(scales)), largest_scale_idx] * 0.4

        # Rotate displacements to world coordinates
        rotated_displacements = torch.einsum("nij,nj->ni", rotmats, displacements)

        # Place new Gaussians along the split axis
        samples[0] = rotated_displacements
        samples[1] = -rotated_displacements
    else:
        # Original isotropic split by sampling from the covariance matrix
        samples = torch.einsum(
            "nij,nj,bnj->bni",
            rotmats,
            scales,
            torch.randn(2, len(scales), 3, device=device),
        )  # [2, N, 3]


    def param_fn(name: str, p: Tensor) -> Tensor:
        repeats = [2] + [1] * (p.dim() - 1)
        if name == "means":
            p_split = (p[sel] + samples).reshape(-1, 3)  # [2N, 3]
        elif name == "scales":
            if anisotropic:
                # Reduce the scale along the split axis
                new_scales_val = scales.clone()
                new_scales_val[torch.arange(len(scales)), largest_scale_idx] /= 1.6
                p_split = torch.log(new_scales_val).repeat(2, 1)
            else:
                p_split = torch.log(scales / 1.6).repeat(2, 1)  # [2N, 3]
        elif name == "opacities" and revised_opacity:
            new_opacities = 1.0 - torch.sqrt(1.0 - torch.sigmoid(p[sel]))
            p_split = torch.logit(new_opacities).repeat(repeats)  # [2N]
        else:
            p_split = p[sel].repeat(repeats)
        p_new = torch.cat([p[rest], p_split])
        p_new = torch.nn.Parameter(p_new, requires_grad=p.requires_grad)
        return p_new

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        v_split = torch.zeros((2 * len(sel), *v.shape[1:]), device=device)
        return torch.cat([v[rest], v_split])

    # update the parameters and the state in the optimizers
    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    # update the extra running state
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            repeats = [2] + [1] * (v.dim() - 1)
            v_new = v[sel].repeat(repeats)
            state[k] = torch.cat((v[rest], v_new))


@dataclass
class NRQMStrategy(DefaultStrategy):
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

    nrqm_every: int = 250
    nrqm_patch_size: int = 32
    nrqm_stagnation_threshold: float = 0.3
    nrqm_prune_stagnant_after: int = 15

    nrqm_ema_decay: float = 0.9

    anisotropic_split: bool = True
    prune_redundant: bool = True
    redundancy_knn: int = 5
    redundancy_overlap_thresh: float = 0.6
    redundancy_color_thresh: float = 0.1

    rasterizer_fn: Any = field(default=None, repr=False)
    nrqm_model: Any = field(default=None, repr=False)
    knn_fn: Any = field(default=None, repr=False)

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        state = super().initialize_state(scene_scale)
        state.update({
            "quality_heatmap": None,
            "view_proj_matrix": None,
            "stagnation_count": None,
            "last_nrqm_step": -1,
            "dynamic_grow_grad2d": self.grow_grad2d,
            "photometric_error_map": None,
        })
        return state

    def step_post_backward(
            self,
            params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
            optimizers: Dict[str, torch.optim.Optimizer],
            state: Dict[str, Any],
            step: int,
            info: Dict[str, Any],
            packed: bool = False,
    ):
        if step % self.nrqm_every == 0 and self.rasterizer_fn is not None:
            self._update_quality_map(params, state, info)
            state["last_nrqm_step"] = step

        super().step_post_backward(params, optimizers, state, step, info, packed)

    @torch.no_grad()
    def _update_quality_map(
            self,
            params: Dict[str, torch.nn.Parameter],
            state: Dict[str, Any],
            info: Dict[str, Any],
    ):
        step = info["step"]
        gt_image = info["pixels"]  # [B, H, W, C]
        gt_ids = info["image_ids"]  # [B,]
        gt_ks = info["Ks"]  # [B, 3, 3]
        camtoworlds_gt = info["camtoworlds"] # [B, 4, 4]

        width, height = info['width'], info['height']
        sh_degree_to_use = min(step // 1000, 3)

        # Render the training view to get photometric error
        rendered_train_view, _, _ = self.rasterizer_fn(
            means=params["means"],
            quats=params["quats"],
            scales=params["scales"],
            opacities=params["opacities"],
            colors=torch.cat([params["sh0"], params["shN"]], 1),
            Ks=gt_ks,
            width=width,
            height=height,
            sh_degree=sh_degree_to_use,
            camtoworlds=camtoworlds_gt,
            image_ids=gt_ids,
        )
        # Assuming batch size is 1 for this calculation
        photometric_error_map = torch.abs(rendered_train_view - gt_image).mean(dim=0)  # [H, W, C]

        if state["photometric_error_map"] is None:
            state["photometric_error_map"] = photometric_error_map
        else:
            state["photometric_error_map"] = torch.lerp(
                photometric_error_map,
                state["photometric_error_map"],
                self.nrqm_ema_decay
            )

        # Generate a novel view for NRQM
        cam_idx = torch.randint(0, info['n_cameras'], (1,)).item()
        novel_camtoworld = info['camtoworlds'][cam_idx].unsqueeze(0)
        novel_K = info['Ks'][cam_idx].unsqueeze(0)

        novel_render, _, _ = self.rasterizer_fn(
            means=params["means"],
            quats=params["quats"],
            scales=params["scales"],
            opacities=params["opacities"],
            colors=torch.cat([params["sh0"], params["shN"]], 1),
            Ks=novel_K,
            width=width,
            height=height,
            sh_degree=sh_degree_to_use,
            camtoworlds=novel_camtoworld,
        )
        novel_render = torch.clamp(novel_render.permute(0, 3, 1, 2), 0.0, 1.0) # [1, C, H, W]

        p = self.nrqm_patch_size
        patches = novel_render.unfold(2, p, p).unfold(3, p, p)
        patches = patches.contiguous().view(1, 3, -1, p, p).permute(0, 2, 1, 3, 4).squeeze(0)

        patch_scores = torch.empty(patches.shape[0], device=patches.device)
        std_threshold = 0.01

        is_flat = patches.mean(dim=1).std(dim=[1, 2]) < std_threshold

        if is_flat.any():
            patch_scores[is_flat] = 100.0


        valid_patch_indices = torch.where(~is_flat)[0]
        for idx in valid_patch_indices:
            patch = patches[idx].unsqueeze(0)
            try:
                score = self.nrqm_model(patch.float())
                patch_scores[idx] = score
            except AssertionError:
                patch_scores[idx] = 100.0

        num_patches_h = height // p
        quality_heatmap = patch_scores.view(num_patches_h, -1)
        if state["quality_heatmap"] is None:
            state["quality_heatmap"] = quality_heatmap
        else:
            state["quality_heatmap"] = torch.lerp(
                quality_heatmap,
                state["quality_heatmap"],
                self.nrqm_ema_decay
            )

        # Use the novel view's projection matrix for subsequent checks
        view_matrix = torch.inverse(novel_camtoworld)
        proj_matrix = torch.zeros(4, 4, device=view_matrix.device)
        proj_matrix[0, 0] = 2 * novel_K[0, 0, 0] / width
        proj_matrix[1, 1] = 2 * novel_K[0, 1, 1] / height
        proj_matrix[0, 2] = -2 * novel_K[0, 0, 2] / width + 1
        proj_matrix[1, 2] = -2 * novel_K[0, 1, 2] / height + 1
        proj_matrix[3, 2] = 1.0
        state["view_proj_matrix"] = (proj_matrix.T @ view_matrix[0]).T

        avg_quality = patch_scores.mean()
        normalized_quality = torch.clamp(avg_quality / 50.0, 0.0, 2.0)
        quality_factor = torch.clamp(1.0 + (normalized_quality - 1.0) * 0.5, 0.5, 1.5).item()

        state["dynamic_grow_grad2d"] = self.grow_grad2d * quality_factor

    @torch.no_grad()
    def _project_to_patch_coords(self, means3d, view_proj_matrix, num_patches_h, num_patches_w):
        means_h = F.pad(means3d, (0, 1), value=1.0)
        p_hom = means_h @ view_proj_matrix

        w_coord = p_hom[:, 3]

        w_coord_safe = torch.clamp(w_coord, min=1e-6)
        p_w = 1.0 / w_coord_safe

        p_proj = p_hom[:, :2] * p_w[:, None]

        valid_mask = (
                torch.isfinite(p_proj).all(dim=1) &
                (torch.abs(p_proj) < 10.0).all(dim=1) &
                (w_coord > 1e-6)
        )

        patch_coords_x = (p_proj[:, 0] * 0.5 + 0.5) * num_patches_w
        patch_coords_y = (p_proj[:, 1] * 0.5 + 0.5) * num_patches_h

        patch_coords_x = torch.clamp(torch.floor(patch_coords_x), 0, num_patches_w - 1).long()
        patch_coords_y = torch.clamp(torch.floor(patch_coords_y), 0, num_patches_h - 1).long()

        return patch_coords_x, patch_coords_y, valid_mask

    @torch.no_grad()
    def _grow_gs(
            self,
            params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
            optimizers: Dict[str, torch.optim.Optimizer],
            state: Dict[str, Any],
            step: int,
    ) -> Tuple[int, int]:
        """Override the growth logic to be spatially aware of NRQM quality."""

        current_grow_grad2d = state.get("dynamic_grow_grad2d", self.grow_grad2d)

        count = state["count"]
        grads = state["grad2d"] / count.clamp_min(1)

        is_grad_high = grads > current_grow_grad2d

        # --- NRQM-guided densification ---
        # 1. Prioritize densification in low-quality regions.
        # 2. Add geometric uncertainty check (optional, placeholder added)
        if state.get("quality_heatmap") is not None and state.get("view_proj_matrix") is not None and state["quality_heatmap"].numel() > 0:
            means3d = params["means"]
            num_patches_h = state["quality_heatmap"].shape[0]
            num_patches_w = state["quality_heatmap"].shape[1]

            patch_coords_x, patch_coords_y, valid_mask = self._project_to_patch_coords(
                means3d, state["view_proj_matrix"], num_patches_h, num_patches_w
            )

            is_in_low_quality_region = torch.zeros(len(means3d), dtype=torch.bool, device=means3d.device)

            if valid_mask.any():
                valid_indices = torch.where(valid_mask)[0]
                patch_scores = state["quality_heatmap"][
                    patch_coords_y[valid_indices],
                    patch_coords_x[valid_indices]
                ]
                is_in_low_quality_region[valid_indices] = patch_scores < self.nrqm_stagnation_threshold

                quality_scores_clamped = torch.clamp(patch_scores, 0.0, self.nrqm_stagnation_threshold)

                quality_based_multiplier = 1.0 + (1.0 - quality_scores_clamped / self.nrqm_stagnation_threshold)

                modified_grads = grads.clone()
                modified_grads[valid_indices] *= quality_based_multiplier

                is_grad_high = modified_grads > current_grow_grad2d
                # is_grad_high = is_grad_high | (is_in_low_quality_region & (grads > self.grow_grad2d * 0.5))

        # --- Placeholder for Geometric Uncertainty ---
        # For a full implementation, one would:
        # 1. Compute per-Gaussian normals.
        # 2. Find K-nearest neighbors for each Gaussian.
        # 3. Compute the variance of normals within each neighborhood.
        # 4. Create a `is_geom_uncertain` mask for Gaussians with high normal variance.
        # 5. `is_grad_high = is_grad_high | is_geom_uncertain`
        # This focuses densification on geometrically complex or ambiguous areas.


        is_small = (
                torch.exp(params["scales"]).max(dim=-1).values
                <= self.grow_scale3d * state["scene_scale"]
        )
        is_dupli = is_grad_high & is_small
        n_dupli = is_dupli.sum().item()

        is_large = ~is_small
        is_split = is_grad_high & is_large
        if step < self.refine_scale2d_stop_iter:
            is_split |= state["radii"] > self.grow_scale2d
        n_split = is_split.sum().item()

        per_gaussian_state_keys = ["grad2d", "count", "radii", "stagnation_count"]
        state_to_grow = {k: v for k, v in state.items() if k in per_gaussian_state_keys and v is not None}

        if n_dupli > 0:
            duplicate(params=params, optimizers=optimizers, state=state_to_grow, mask=is_dupli)

        is_split = torch.cat(
            [is_split, torch.zeros(n_dupli, dtype=torch.bool, device=is_split.device)]
        )

        if n_split > 0:
            split(
                params=params,
                optimizers=optimizers,
                state=state_to_grow,
                mask=is_split,
                revised_opacity=self.revised_opacity,
                anisotropic=self.anisotropic_split,
            )

        state.update(state_to_grow)

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
            num_patches_h = state["quality_heatmap"].shape[0]
            num_patches_w = state["quality_heatmap"].shape[1]

            patch_coords_x, patch_coords_y, valid_mask = self._project_to_patch_coords(
                means3d, state["view_proj_matrix"], num_patches_h, num_patches_w
            )

            is_in_low_quality_region = torch.zeros(len(means3d), dtype=torch.bool, device=means3d.device)

            if valid_mask.any():
                valid_indices = torch.where(valid_mask)[0]
                patch_scores = state["quality_heatmap"][
                    patch_coords_y[valid_indices],
                    patch_coords_x[valid_indices]
                ]
                is_in_low_quality_region[valid_indices] = patch_scores < self.nrqm_stagnation_threshold

            is_stagnant = is_grad_low & is_in_low_quality_region

            state["stagnation_count"][is_stagnant] += 1
            state["stagnation_count"][~is_stagnant] = (state["stagnation_count"][~is_stagnant] - 1).clamp(min=0)

            is_prune_stagnant = state["stagnation_count"] > self.nrqm_prune_stagnant_after

        is_prune_redundant = torch.zeros_like(is_prune_original)
        if self.prune_redundant and self.knn_fn is not None:
            means3d = params["means"]
            scales = torch.exp(params["scales"]).max(dim=-1).values
            opacities = torch.sigmoid(params["opacities"].flatten())
            sh0 = params["sh0"].squeeze(1)

            # Find K-nearest neighbors
            # This can be computationally expensive. Consider running it less frequently.
            dists, idxs = self.knn_fn(means3d, K=self.redundancy_knn)

            neighbor_idxs = idxs[:, 1:]
            neighbor_dists = dists[:, 1:]

            # Check for overlap
            overlap_mask = neighbor_dists < (scales.unsqueeze(1) + scales[neighbor_idxs]) * self.redundancy_overlap_thresh

            # Check for color similarity
            color_dist = torch.norm(sh0.unsqueeze(1) - sh0[neighbor_idxs], dim=-1)
            color_sim_mask = color_dist < self.redundancy_color_thresh

            # A Gaussian is redundant if it overlaps with a similar-colored neighbor
            # that is more opaque.
            is_less_opaque = opacities.unsqueeze(1) < opacities[neighbor_idxs]

            is_redundant_neighbor = overlap_mask & color_sim_mask & is_less_opaque
            is_prune_redundant = is_redundant_neighbor.any(dim=1)


        is_prune = is_prune_original | is_prune_stagnant | is_prune_redundant

        n_prune = is_prune.sum().item()
        if n_prune > 0:
            per_gaussian_state_keys = ["grad2d", "count", "radii", "stagnation_count"]
            state_to_prune = {k: v for k, v in state.items() if k in per_gaussian_state_keys and v is not None}

            remove(params=params, optimizers=optimizers, state=state_to_prune, mask=is_prune)

            state.update(state_to_prune)

        return n_prune