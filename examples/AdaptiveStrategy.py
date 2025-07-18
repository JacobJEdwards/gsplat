from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from utils import knn_with_ids
from gsplat.strategy.ops import duplicate, remove, _update_param_with_optimizer
from gsplat.strategy.default import DefaultStrategy
from gsplat.utils import normalized_quat_to_rotmat

class DensificationNetwork(nn.Module):
    """A small MLP to predict densification priority for a Gaussian."""
    def __init__(self, input_dim: int = 10, mlp_width: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, mlp_width),
            nn.LayerNorm(mlp_width),
            nn.ReLU(),
            nn.Linear(mlp_width, mlp_width),
            nn.LayerNorm(mlp_width),
            nn.ReLU(),
            nn.Linear(mlp_width, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

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

    use_geom_uncertainty: bool = True
    num_uncertainty_views: int = 3
    geom_uncertainty_thresh: float = 0.002

    photometric_error_thresh: float = 0.1

    use_learned_densification: bool = True
    bootstrap_steps: int = 4000
    learn_every: int = 500
    hindsight_delay: int = 100

    rasterizer_fn: Any = field(default=None, repr=False)
    nrqm_model: Any = field(default=None, repr=False)
    knn_fn: Any = field(default=None, repr=False)

    densification_net: DensificationNetwork = field(default=None, repr=False)
    densification_optimizer: Any = field(default=None, repr=False)

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
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
        })

        if self.use_learned_densification and self.densification_net is None:
            pass

        return state

    def _initialize_learning_components(self, device) -> None:
        if self.use_learned_densification and self.densification_net is None:
            self.densification_net = DensificationNetwork().to(device)
            self.densification_net_optimizer = torch.optim.AdamW(
                self.densification_net.parameters(), lr=1e-4
            )

            self.knn_fn = knn_with_ids

    def step_post_backward(
            self,
            params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
            optimizers: Dict[str, torch.optim.Optimizer],
            state: Dict[str, Any],
            step: int,
            info: Dict[str, Any],
            packed: bool = False,
    ):
        if self.use_learned_densification and self.densification_net is None:
            self._initialize_learning_components(params["means"].device)

        if step % self.nrqm_every == 0 and self.rasterizer_fn is not None:
            self._update_quality_map(params, state, info)
            state["last_nrqm_step"] = step

        if self.use_learned_densification:
            self._process_hindsight_buffer(state, step)

        super().step_post_backward(params, optimizers, state, step, info, packed)

        if self.use_learned_densification and step > 1000 and step % self.learn_every == 0:
            self._train_densification_network(state)

    @torch.no_grad()
    def _update_quality_map(
            self,
            params: Dict[str, torch.nn.Parameter],
            state: Dict[str, Any],
            info: Dict[str, Any],
    ):
        step = info["step"]
        width, height = info['width'], info['height']
        sh_degree_to_use = min(step // 1000, 3)

        all_depth_renders = []
        main_novel_view_idx = -1

        novel_render_for_nrqm = None
        for i in range(self.num_uncertainty_views):
            cam_idx = torch.randint(0, info['n_cameras'], (1,)).item()
            if i == 0:
                main_novel_view_idx = cam_idx

            novel_camtoworld = info['camtoworlds'][cam_idx].unsqueeze(0)
            novel_K = info['Ks'][cam_idx].unsqueeze(0)

            novel_render_pkg, _, _ = self.rasterizer_fn(
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
                render_mode="RGB+ED"
            )

            novel_color = novel_render_pkg[..., :3]
            novel_depth = novel_render_pkg[..., 3]
            all_depth_renders.append(novel_depth)

            if i == 0:
                novel_render_for_nrqm = torch.clamp(novel_color.permute(0, 3, 1, 2), 0.0, 1.0)

        p = self.nrqm_patch_size
        patches = novel_render_for_nrqm.unfold(2, p, p).unfold(3, p, p)
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

        if self.use_geom_uncertainty:
            depth_stack = torch.stack(all_depth_renders)

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

        main_novel_camtoworld = info['camtoworlds'][main_novel_view_idx].unsqueeze(0)
        main_novel_K = info['Ks'][main_novel_view_idx].unsqueeze(0)
        view_matrix = torch.inverse(main_novel_camtoworld)
        proj_matrix = torch.zeros(4, 4, device=view_matrix.device)
        proj_matrix[0, 0] = 2 * main_novel_K[0, 0, 0] / width
        proj_matrix[1, 1] = 2 * main_novel_K[0, 1, 1] / height
        proj_matrix[0, 2] = -2 * main_novel_K[0, 0, 2] / width + 1
        proj_matrix[1, 2] = -2 * main_novel_K[0, 1, 2] / height + 1
        proj_matrix[3, 2] = 1.0
        state["view_proj_matrix"] = (proj_matrix.T @ view_matrix[0]).T

        avg_quality = patch_scores.mean()
        normalized_quality = torch.clamp(avg_quality / 50.0, 0.0, 2.0)
        quality_factor = torch.clamp(1.0 + (normalized_quality - 1.0) * 0.5, 0.5, 1.5).item()
        state["dynamic_grow_grad2d"] = self.grow_grad2d * quality_factor

        gt_image = info["pixels"]  # [B, H, W, C]
        gt_ids = info["image_ids"]  # [B,]
        gt_ks = info["Ks"]  # [B, 3, 3]
        camtoworlds_gt = info["camtoworlds"] # [B, 4, 4]

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
        photometric_error_map = torch.abs(rendered_train_view - gt_image).mean(dim=-1).squeeze(0) # [H, W]

        if state["photometric_error_map"] is None:
            state["photometric_error_map"] = photometric_error_map
        else:
            state["photometric_error_map"] = torch.lerp(
                photometric_error_map,
                state["photometric_error_map"],
                self.nrqm_ema_decay
            )


    @torch.no_grad()
    def _project_to_patch_coords(self, means3d, view_proj_matrix, h, w):
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
            self, params: Dict, state: Dict, valid_mask: Tensor,
            pixel_coords_x: Tensor, pixel_coords_y: Tensor,
            patch_coords_x: Tensor, patch_coords_y: Tensor
    ) -> Tensor:
        num_gaussians = len(params["means"])
        device = params["means"].device
        feature_dim = 10
        features = torch.zeros(num_gaussians, feature_dim, device=device)

        features[:, 0] = torch.sigmoid(params["opacities"].flatten())
        scales = torch.exp(params["scales"])
        features[:, 1] = scales.max(dim=-1).values / state["scene_scale"]
        features[:, 2] = scales.min(dim=-1).values / state["scene_scale"]
        features[:, 3] = scales.mean(dim=-1) / state["scene_scale"]
        features[:, 4] = torch.norm(params["sh0"], dim=(-1, -2))

        valid_indices = torch.where(valid_mask)[0]
        if valid_indices.numel() > 0:
            if state.get("photometric_error_map") is not None:
                features[valid_indices, 5] = state["photometric_error_map"][pixel_coords_y[valid_indices], pixel_coords_x[valid_indices]]

            if state.get("geom_uncertainty_map") is not None:
                features[valid_indices, 6] = state["geom_uncertainty_map"][pixel_coords_y[valid_indices], pixel_coords_x[valid_indices]]

            if state.get("quality_heatmap") is not None:
                features[valid_indices, 7] = state["quality_heatmap"][patch_coords_y[valid_indices], patch_coords_x[valid_indices]]

        if self.knn_fn is not None and num_gaussians > self.redundancy_knn:
            dists, _ = self.knn_fn(params["means"], K=self.redundancy_knn)
            features[:, 8] = dists[:, 1:].mean(dim=-1) / state["scene_scale"]

        grads = state["grad2d"] / state["count"].clamp_min(1)
        features[:, 9] = grads

        return torch.nan_to_num(features, 0.0)

    def _process_hindsight_buffer(self, state, current_step):
        while state["hindsight_buffer"] and (current_step - state["hindsight_buffer"][0]["step"]) >= self.hindsight_delay:
            experience = state["hindsight_buffer"].popleft()

            px, py = experience["pixel_coords"]
            current_error = state["photometric_error_map"][py, px].mean()

            reward = experience["initial_error"] - current_error
            label = 1.0 if reward > 0.01 else 0.0

            if len(state["replay_buffer"]) < state["replay_buffer"].maxlen:
                state["replay_buffer"].append((experience["features"], torch.tensor(label)))

    def _train_densification_network(self, state):
        if len(state["replay_buffer"]) < 128:
            return

        self.densification_net.train()

        batch_indices = torch.randint(0, len(state["replay_buffer"]), (128,))
        batch = [state["replay_buffer"][i] for i in batch_indices]
        features = torch.stack([x[0] for x in batch]).to(self.densification_net.net[0].weight.device)
        labels = torch.stack([x[1] for x in batch]).to(self.densification_net.net[0].weight.device)

        self.densification_net_optimizer.zero_grad()
        logits = self.densification_net(features).squeeze()
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        self.densification_net_optimizer.step()

        if self.verbose:
            print(f"Trained Densification Network, Loss: {loss.item():.4f}")

    @torch.no_grad()
    def _grow_gs(
            self,
            params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
            optimizers: Dict[str, torch.optim.Optimizer],
            state: Dict[str, Any],
            step: int,
    ) -> Tuple[int, int]:

        if state.get("view_proj_matrix") is not None:
            means3d = params["means"]
            if state.get("photometric_error_map") is not None:
                h, w = state["photometric_error_map"].shape
            else:
                h = state["quality_heatmap"].shape[0] * self.nrqm_patch_size
                w = state["quality_heatmap"].shape[1] * self.nrqm_patch_size

            patch_coords_x, patch_coords_y, pixel_coords_x, pixel_coords_y, valid_mask = self._project_to_patch_coords(
                means3d, state["view_proj_matrix"], h, w
            )
            features = self._get_gaussian_features(
                params, state, valid_mask, pixel_coords_x, pixel_coords_y, patch_coords_x, patch_coords_y
            )
        else:
            return 0, 0

        if self.use_learned_densification and step >= self.bootstrap_steps:
            self.densification_net.eval()
            with torch.no_grad():
                scores = self.densification_net(features).squeeze()
            is_grad_high = scores > 0.5
        else:
            current_grow_grad2d = state.get("dynamic_grow_grad2d", self.grow_grad2d)
            grads = state["grad2d"] / state["count"].clamp_min(1)
            is_grad_high_orig = grads > current_grow_grad2d

            densification_potential = torch.zeros_like(grads)
            if valid_mask.any():
                valid_indices = torch.where(valid_mask)[0]
                error_scores = features[valid_indices, 5]
                nrqm_scores = features[valid_indices, 7]
                uncertainty_scores = features[valid_indices, 6]

                error_potential = torch.clamp(error_scores / self.photometric_error_thresh, 0.0, 1.0)
                nrqm_potential = torch.clamp(1.0 - nrqm_scores / self.nrqm_stagnation_threshold, 0.0, 1.0)
                uncertainty_potential = torch.clamp(uncertainty_scores / self.geom_uncertainty_thresh, 0.0, 1.0)

                densification_potential[valid_indices] = (0.4 * error_potential + 0.3 * nrqm_potential + 0.3 * uncertainty_potential)

            is_high_potential = densification_potential > 0.5
            is_grad_high = is_grad_high_orig | is_high_potential

            if self.use_learned_densification:
                densified_mask = is_grad_high
                if densified_mask.any():
                    densified_indices = torch.where(densified_mask)[0]
                    for idx in densified_indices:
                        if valid_mask[idx]:
                            px, py = pixel_coords_x[idx], pixel_coords_y[idx]
                            initial_error = state["photometric_error_map"][
                                            max(0, py-2):py+3, max(0, px-2):px+3
                                            ].mean()

                            experience = {
                                "step": step,
                                "features": features[idx].detach().cpu(),
                                "pixel_coords": (px, py),
                                "initial_error": initial_error,
                            }
                            state["hindsight_buffer"].append(experience)

        is_small = (torch.exp(params["scales"]).max(dim=-1).values <= self.grow_scale3d * state["scene_scale"])
        is_dupli = is_grad_high & is_small
        n_dupli = is_dupli.sum().item()

        is_large = ~is_small
        is_split = is_grad_high & is_large
        n_split = is_split.sum().item()

        if n_dupli > 0 or n_split > 0:
            per_gaussian_state_keys = ["grad2d", "count", "radii", "stagnation_count"]
            state_to_grow = {k: v for k, v in state.items() if k in per_gaussian_state_keys and v is not None}
            if n_dupli > 0:
                duplicate(params=params, optimizers=optimizers, state=state_to_grow, mask=is_dupli)

            is_split_after_dup = torch.cat([is_split, torch.zeros(n_dupli, dtype=torch.bool, device=is_split.device)])
            if n_split > 0:
                split(params=params, optimizers=optimizers, state=state_to_grow, mask=is_split_after_dup,
                      revised_opacity=self.revised_opacity, anisotropic=self.anisotropic_split)
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
            h = state["quality_heatmap"].shape[0] * self.nrqm_patch_size
            w = state["quality_heatmap"].shape[1] * self.nrqm_patch_size

            patch_coords_x, patch_coords_y, _, _, valid_mask = self._project_to_patch_coords(
                means3d, state["view_proj_matrix"], h, w
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
            num_points = len(means3d)
            last_count = state.get("last_prune_count", 0)

            if last_count == -1 or num_points > last_count * 1.05:
                state["last_prune_count"] = num_points

                if num_points > self.redundancy_knn:
                    scales = torch.exp(params["scales"]).max(dim=-1).values
                    opacities = torch.sigmoid(params["opacities"].flatten())
                    sh0 = params["sh0"].squeeze(1)

                    dists, idxs = self.knn_fn(means3d, K=self.redundancy_knn)

                    neighbor_idxs = idxs[:, 1:]
                    neighbor_dists = dists[:, 1:]

                    overlap_mask = neighbor_dists < (scales.unsqueeze(1) + scales[neighbor_idxs]) * self.redundancy_overlap_thresh
                    color_dist = torch.norm(sh0.unsqueeze(1) - sh0[neighbor_idxs], dim=-1)
                    color_sim_mask = color_dist < self.redundancy_color_thresh
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