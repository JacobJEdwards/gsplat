from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from gsplat.strategy.ops import (
    duplicate,
    remove,
    _update_param_with_optimizer,
    reset_opa,
)
from gsplat.strategy.default import DefaultStrategy
from gsplat.utils import normalized_quat_to_rotmat

class DensificationNetwork(nn.Module):
    def __init__(self, input_dim: int = 10, mlp_width: int = 64):
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

@torch.no_grad()
def split(
        params: dict[str, torch.nn.Parameter] | torch.nn.ParameterDict,
        optimizers: dict[str, torch.optim.Optimizer],
        state: dict[str, Tensor],
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
        largest_scale_idx = torch.argmax(scales, dim=1)
        samples = torch.zeros(2, len(scales), 3, device=device)

        displacements = torch.zeros_like(scales)
        displacements[torch.arange(len(scales)), largest_scale_idx] = scales[torch.arange(len(scales)), largest_scale_idx] * 0.4

        rotated_displacements = torch.einsum("nij,nj->ni", rotmats, displacements)

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
            original_alpha = torch.sigmoid(p[sel])
            original_alpha = torch.clamp(original_alpha, 0.0, 1.0 - 1e-6)
            new_alpha = 1.0 - torch.sqrt(1.0 - original_alpha)
            p_split = torch.logit(new_alpha).repeat(repeats)  # [2N]
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
    refine_every: int = 100
    prune_every: int = 400
    nrqm_every: int = 500

    max_splits_per_step: int = 20000
    max_duplications_per_step: int = 20000
    subset_fraction: float = 0.2

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


    w_photometric: float = 0.6
    w_quality: float = -0.2
    w_uncertainty: float = 0.2

    rasterizer_fn: Any = field(default=None, repr=False)
    nrqm_model: Any = field(default=None, repr=False)
    knn_fn: Any = field(default=None, repr=False)

    densification_net: Any = field(default=None, repr=False)
    densification_optimizer: Any = field(default=None, repr=False)

    start_asc_grad2d: float = 0.0002
    end_asc_grad2d: float = 0.001

    prune_significance_threshold: float = 0.01

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
            "significance": None, # todo
            "prev_grad2d": None,
            "prev_opacity": None

        })

        return state

    def _initialize_learning_components(self, device) -> None:
        if self.use_learned_densification and self.densification_net is None:
            self.densification_net = DensificationNetwork(input_dim=17).to(device)
            self.densification_net_optimizer = torch.optim.AdamW(
                self.densification_net.parameters(), lr=1e-4, weight_decay=1e-5
            )

    def _get_asc_grad2d(self, step: int) -> float:
        if step >= self.refine_stop_iter:
            return self.end_asc_grad2d

        t = step / self.refine_stop_iter
        return self.start_asc_grad2d + (self.end_asc_grad2d - self.start_asc_grad2d) * t


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

        if self.use_learned_densification and self.densification_net is None:
            self._initialize_learning_components(params["means"].device)

        should_update_maps = (state["last_nrqm_step"] == -1 and step > 0) or \
                             (step % self.nrqm_every == 0)

        if should_update_maps:
            self._update_quality_map(params, state, info)
            state["last_nrqm_step"] = step

        if self.use_learned_densification:
            self._process_hindsight_buffer(state, step)

        self._update_state(params, state, info, packed=packed)

        if step > self.refine_start_iter:
            if step % self.refine_every == 0:
                n_dupli, n_split = self._grow_gs(params, optimizers, state, step)
                if self.verbose:
                    print(
                        f"Step {step}: {n_dupli} GSs duplicated, {n_split} GSs split. "
                        f"Now having {len(params['means'])} GSs."
                    )
            if step % self.prune_every == 0:
                n_pruned = self._prune_gs(params, optimizers, state, step)
                if self.verbose:
                    print(
                        f"Step {step}: {n_pruned} GSs pruned. "
                        f"Now having {len(params['means'])} GSs."
                    )
                state["last_prune_count"] = n_pruned

        if step % self.reset_every == 0 and step > 0:
            reset_opa(params=params, optimizers=optimizers, state=state, value=self.prune_opa * 2.0)

        if self.use_learned_densification and step > 1000 and step % self.learn_every == 0:
            self._train_densification_network(state)



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
        """Extracts feature vectors and projection info for a subset of Gaussians."""
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

        feature_dim = 17
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

        return torch.nan_to_num(features, 0.0), pixel_coords_x, pixel_coords_y, patch_coords_x, patch_coords_y, valid_mask

    def _process_hindsight_buffer(self, state, current_step):
        if state.get("photometric_error_map") is None:
            if self.verbose:
                print("Skipping hindsight processing: photometric error map is not available.")

            return

        device = state["photometric_error_map"].device

        while state["hindsight_buffer"] and (current_step - state["hindsight_buffer"][0]["step"]) >= self.hindsight_delay:
            if self.verbose:
                print(f"Processing experience from step {state['hindsight_buffer'][0]['step']}.")

            experience = state["hindsight_buffer"].popleft()

            px, py = experience["pixel_coords"]
            patch_x, patch_y = experience["patch_coords"]

            current_error = state["photometric_error_map"][max(0, py-2):py+3, max(0, px-2):px+3].mean()
            current_quality = state["quality_heatmap"][patch_y, patch_x]
            current_uncertainty = state["geom_uncertainty_map"][py, px]

            reward_photo = experience["initial_error"] - current_error
            reward_quality = experience["initial_quality"] - current_quality
            reward_uncertainty = experience["initial_uncertainty"] - current_uncertainty

            final_reward = (self.w_photometric * reward_photo +
                            self.w_quality * reward_quality +
                            self.w_uncertainty * reward_uncertainty)

            if len(state["replay_buffer"]) < state["replay_buffer"].maxlen:
                if self.verbose:
                    print(f"Adding experience to replay buffer: {len(state['replay_buffer'])} samples before adding.")

                state["replay_buffer"].append((experience["features"], torch.tensor(final_reward, device=device)))

    def _train_densification_network(self, state):
        if len(state["replay_buffer"]) < 256:
            if self.verbose:
                print(f"Not enough data in replay buffer to train the densification network: {len(state['replay_buffer'])} samples available.")

            return
        elif self.verbose:
            print(f"Training Densification Network with {len(state['replay_buffer'])} samples.")

        self.densification_net.train()
        device = self.densification_net.net[0].weight.device

        batch_indices = torch.randint(0, len(state["replay_buffer"]), (256,))
        batch = [state["replay_buffer"][i] for i in batch_indices]
        features = torch.stack([x[0] for x in batch]).to(device)
        rewards = torch.stack([x[1] for x in batch]).to(device)

        self.densification_net_optimizer.zero_grad()
        predicted_utility = self.densification_net(features).squeeze(-1)

        loss = F.mse_loss(predicted_utility, rewards)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.densification_net.parameters(), 1.0)
        self.densification_optimizer.step()

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
        """Performs stochastic, budgeted, and policy-driven densification."""
        num_gaussians = len(params["means"])
        device = params["means"].device

        subset_mask = torch.rand(num_gaussians, device=device) < self.subset_fraction
        subset_indices = torch.where(subset_mask)[0]
        if subset_indices.numel() == 0: 
            if self.verbose:
                print("Skipping Gaussian growth: no valid subset indices.")
                
            return 0, 0

        features_subset, px, py, ptx, pty, valid_mask_subset = self._get_gaussian_features(params, state, subset_mask, step)
        if features_subset is None: 
            if self.verbose:
                print("Skipping Gaussian growth: no valid features extracted.")
                
            return 0, 0

        if self.use_learned_densification and step >= self.bootstrap_steps:
            self.densification_net.eval()
            with torch.no_grad():
                utility_scores = self.densification_net(features_subset).squeeze()
        else:
            utility_scores = features_subset[:, 12]

        if self.use_learned_densification:
            for i, original_idx in enumerate(subset_indices):
                if self.verbose:
                    print(f"Processing Gaussian {original_idx} at step {step}.")
                
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
                    if self.verbose:
                        print(f"Added experience to hindsight buffer at step {step} for Gaussian {original_idx}.")

        scales_subset = torch.exp(params["scales"][subset_mask])
        is_small_subset = scales_subset.max(dim=-1).values <= self.grow_scale3d * state["scene_scale"]

        dupli_candidates_mask = is_small_subset
        split_candidates_mask = ~is_small_subset

        is_dupli = torch.zeros(num_gaussians, dtype=torch.bool, device=device)
        dupli_indices_in_subset = torch.where(dupli_candidates_mask)[0]
        n_dupli = min(len(dupli_indices_in_subset), self.max_duplications_per_step)
        if n_dupli > 0:
            dupli_scores = utility_scores[dupli_indices_in_subset]
            _, top_indices = torch.topk(dupli_scores, n_dupli)
            final_dupli_indices_in_subset = dupli_indices_in_subset[top_indices]
            is_dupli[subset_indices[final_dupli_indices_in_subset]] = True

        is_split = torch.zeros(num_gaussians, dtype=torch.bool, device=device)
        split_indices_in_subset = torch.where(split_candidates_mask)[0]
        n_split = min(len(split_indices_in_subset), self.max_splits_per_step)
        if n_split > 0:
            split_scores = utility_scores[split_indices_in_subset]
            _, top_indices = torch.topk(split_scores, n_split)
            final_split_indices_in_subset = split_indices_in_subset[top_indices]
            is_split[subset_indices[final_split_indices_in_subset]] = True

        per_gaussian_state_keys = ["grad2d", "count", "radii", "stagnation_count", "prev_grad2d", "prev_opacity", "significance"]
        state_to_densify = {k: v for k, v in state.items() if k in per_gaussian_state_keys and v is not None}

        if n_dupli > 0: duplicate(params, optimizers, state_to_densify, is_dupli)
        if n_split > 0:
            is_split_after_dup = torch.cat([is_split, torch.zeros(n_dupli, dtype=torch.bool, device=device)])
            split(params, optimizers, state_to_densify, is_split_after_dup, anisotropic=self.anisotropic_split,
                  revised_opacity=self.revised_opacity)

        state.update(state_to_densify)

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

            is_in_low_quality_region = torch.zeros_like(is_prune_original); valid_indices = torch.where(valid_mask)[0]
            if valid_indices.numel() > 0:
                patch_scores = state["quality_heatmap"][patch_coords_y[valid_indices], patch_coords_x[valid_indices]]
                is_in_low_quality_region[valid_indices] = patch_scores < self.nrqm_stagnation_threshold
            is_stagnant = is_grad_low & is_in_low_quality_region

            state["stagnation_count"][is_stagnant] += 1;
            state["stagnation_count"][~is_stagnant] = (state["stagnation_count"][~is_stagnant] - 1).clamp(min=0)
            is_prune_stagnant = state["stagnation_count"] > self.nrqm_prune_stagnant_after

        is_prune_significant = torch.zeros_like(is_prune_original)
        if "significance" in state and state["significance"] is not None and state["significance"].numel() > 0:
            if state["significance"].numel() == is_prune_original.numel():
                is_prune_significance = state["significance"] < self.prune_significance_threshold


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

        is_prune = is_prune_original | is_prune_stagnant | is_prune_redundant | is_prune_significant

        n_prune = is_prune.sum().item()
        if n_prune > 0:
            per_gaussian_state_keys = ["grad2d", "count", "radii", "stagnation_count", "prev_grad2d", "prev_opacity", "significance"]
            state_to_prune = {k: v for k, v in state.items() if k in per_gaussian_state_keys and v is not None}

            remove(params=params, optimizers=optimizers, state=state_to_prune, mask=is_prune)

            state.update(state_to_prune)

        return n_prune