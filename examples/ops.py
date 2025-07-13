import torch
from torch import Tensor
import torch.nn.functional as F

from utils import knn_with_ids_two_tensor, matrix_to_quaternion
from gsplat.strategy.ops import _update_param_with_optimizer
from gsplat.utils import normalized_quat_to_rotmat


@torch.no_grad()
def split(
        params: dict[str, torch.nn.Parameter] | torch.nn.ParameterDict,
        optimizers: dict[str, torch.optim.Optimizer],
        state: dict[str, Tensor],
        mask: Tensor,
        revised_opacity: bool = False,
        anisotropic: bool = False,
        split_ratios: Tensor | None = None,
        directions: Tensor | None = None,
        lr_multipliers: Tensor | None = None,
        warmup_steps: int = 0,
):
    """Inplace split the Gaussian with the given mask.

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        state: A dictionary of running states.
        mask: A boolean mask to split the Gaussians.
        revised_opacity: Whether to use revised opacity formulation. Default: False.
        anisotropic: Whether to split along the largest variance axis. Default: False.
        split_ratios (Optional[Tensor]): Per-Gaussian split ratios.
        directions (Optional[Tensor]): Per-Gaussian learned split directions. If provided,
          overrides the default split behavior.
    """
    device = mask.device
    sel = torch.where(mask)[0]
    rest = torch.where(~mask)[0]

    scales = torch.exp(params["scales"][sel])

    if directions is not None:
        assert len(directions) == len(scales), "Directions must match the number of Gaussians to split."
        largest_scale_val = scales.max(dim=1).values
        displacements = directions * largest_scale_val.unsqueeze(-1) * 0.4
        samples = torch.stack([displacements, -displacements]) # [2, N_sel, 3]
    else:
        quats = F.normalize(params["quats"][sel], dim=-1)
        rotmats = normalized_quat_to_rotmat(quats)
        if anisotropic:
            largest_scale_idx = torch.argmax(scales, dim=1)
            displacements = torch.zeros_like(scales)
            displacements[torch.arange(len(scales)), largest_scale_idx] = scales[torch.arange(len(scales)), largest_scale_idx] * 0.4
            rotated_displacements = torch.einsum("nij,nj->ni", rotmats, displacements)
            samples = torch.stack([rotated_displacements, -rotated_displacements])
        else:
            samples = torch.einsum(
                "nij,nj,bnj->bni",
                rotmats,
                scales,
                torch.randn(2, len(scales), 3, device=device),
            )

    def param_fn(name: str, p: Tensor) -> Tensor:
        repeats = [2] + [1] * (p.dim() - 1)
        if name == "means":
            p_split = (p[sel] + samples).reshape(-1, 3)
        elif name == "scales":
            current_split_ratios = split_ratios if split_ratios is not None else torch.full((len(scales),), 1.6, device=device)
            if directions is not None:
                p_split = torch.log(scales / current_split_ratios.unsqueeze(-1)).repeat(2,1)
            else:
                largest_scale_idx = torch.argmax(scales, dim=1)
                new_scales_val = scales.clone()
                new_scales_val[torch.arange(len(scales)), largest_scale_idx] /= current_split_ratios
                p_split = torch.log(new_scales_val).repeat(2, 1)
        elif name == "opacities" and revised_opacity:
            original_alpha = torch.sigmoid(p[sel])
            original_alpha = torch.clamp(original_alpha, 0.0, 1.0 - 1e-6)
            new_alpha = 1.0 - torch.sqrt(1.0 - original_alpha)
            p_split = torch.logit(new_alpha).repeat(repeats)
        else:
            p_split = p[sel].repeat(repeats)
        p_new = torch.cat([p[rest], p_split])
        p_new = torch.nn.Parameter(p_new, requires_grad=p.requires_grad)
        return p_new

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        v_split = torch.zeros((2 * len(sel), *v.shape[1:]), device=device)
        return torch.cat([v[rest], v_split])

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)

    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            repeats = [2] + [1] * (v.dim() - 1)
            if k == 'ac_hidden_states':
                v_split = torch.zeros_like(v[sel].repeat(repeats))
                state[k] = torch.cat((v[rest], v_split))
            elif k == 'custom_lr_multipliers' and lr_multipliers is not None:
                v_split = lr_multipliers.repeat_interleave(2)
                state[k] = torch.cat((v[rest], v_split))
            elif k == 'custom_lr_timers' and warmup_steps > 0:
                v_split = torch.full((len(sel) * 2,), warmup_steps, dtype=torch.int32, device=device)
                state[k] = torch.cat((v[rest], v_split))
            else:
                v_new = v[sel].repeat(repeats)
                state[k] = torch.cat((v[rest], v_new))

@torch.no_grad()
def duplicate(
        params: dict[str, torch.nn.Parameter] | torch.nn.ParameterDict,
        optimizers: dict[str, torch.optim.Optimizer],
        state: dict[str, Tensor],
        mask: Tensor,
        offsets: Tensor | None = None,
        lr_multipliers: Tensor | None = None,
        warmup_steps: int = 0,
):
    """
    Inplace duplicate the Gaussian with the given mask, with optional displacement.
    """
    device = mask.device
    sel = torch.where(mask)[0]

    if offsets is not None:
        assert len(sel) == len(offsets), "Number of offsets must match number of Gaussians to duplicate."

    def param_fn(name: str, p: Tensor) -> Tensor:
        if name == "means" and offsets is not None:
            new_params = p[sel] + offsets
        else:
            new_params = p[sel]
        return torch.nn.Parameter(torch.cat([p, new_params]), requires_grad=p.requires_grad)

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        return torch.cat([v, torch.zeros((len(sel), *v.shape[1:]), device=device)])

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            if k == 'ac_hidden_states':
                v_new = torch.zeros_like(v[sel])
                state[k] = torch.cat((v, v_new))
            elif k == 'custom_lr_multipliers' and lr_multipliers is not None:
                state[k] = torch.cat((v, lr_multipliers))
            elif k == 'custom_lr_timers' and warmup_steps > 0:
                v_new = torch.full((len(sel),), warmup_steps, dtype=torch.int32, device=device)
                state[k] = torch.cat((v, v_new))
            else:
                state[k] = torch.cat((v, v[sel]))


@torch.no_grad()
def merge(
        params: dict[str, torch.nn.Parameter] | torch.nn.ParameterDict,
        optimizers: dict[str, torch.optim.Optimizer],
        state: dict[str, Tensor],
        mask: Tensor,
        num_samples_for_pca: int = 100,
) -> tuple[Tensor, int]:
    """
    In-place merges selected Gaussians with their nearest neighbors.
    """
    if not mask.any():
        return torch.zeros_like(mask, dtype=torch.bool), 0

    device = mask.device
    source_indices = torch.where(mask)[0]
    all_means = params["means"]

    _, nn_indices = knn_with_ids_two_tensor(all_means[source_indices], all_means, K=2)
    target_indices = nn_indices[:, 1]

    unique_pairs = set()
    used_indices = torch.zeros(len(all_means), dtype=torch.bool, device=device)
    distances, _ = knn_with_ids_two_tensor(all_means[source_indices], all_means, K=2)
    pair_distances = distances[:, 1]
    sorted_pair_indices = torch.argsort(pair_distances)

    for idx in sorted_pair_indices:
        i = source_indices[idx].item()
        j = target_indices[idx].item()

        if i == j or used_indices[i] or used_indices[j]:
            continue

        pair = tuple(sorted((i, j)))
        unique_pairs.add(pair)
        used_indices[i] = True
        used_indices[j] = True

    if not unique_pairs:
        return torch.zeros_like(mask, dtype=torch.bool), 0

    pairs = torch.tensor(list(unique_pairs), device=device)
    indices_i, indices_j = pairs[:, 0], pairs[:, 1]

    num_pairs = len(indices_i)

    opacities_i = torch.sigmoid(params["opacities"][indices_i])
    opacities_j = torch.sigmoid(params["opacities"][indices_j])
    alpha_sum = opacities_i + opacities_j + 1e-8
    w_i = opacities_i / alpha_sum
    w_j = opacities_j / alpha_sum

    new_opacities_alpha = 1.0 - (1.0 - opacities_i) * (1.0 - opacities_j)
    new_opacities = torch.logit(torch.clamp(new_opacities_alpha, 1e-6, 1.0 - 1e-6))

    colors_i = torch.cat([params["sh0"][indices_i], params["shN"][indices_i]], dim=1)
    colors_j = torch.cat([params["sh0"][indices_j], params["shN"][indices_j]], dim=1)
    new_colors = w_i.unsqueeze(-1).unsqueeze(-1) * colors_i + w_j.unsqueeze(-1).unsqueeze(-1) * colors_j
    new_sh0, new_shN = new_colors[:, :1], new_colors[:, 1:]

    means_i, means_j = params["means"][indices_i], params["means"][indices_j]
    scales_i, scales_j = torch.exp(params["scales"][indices_i]), torch.exp(params["scales"][indices_j])
    quats_i, quats_j = F.normalize(params["quats"][indices_i]), F.normalize(params["quats"][indices_j])
    rotmats_i, rotmats_j = normalized_quat_to_rotmat(quats_i), normalized_quat_to_rotmat(quats_j)

    noise = torch.randn(num_samples_for_pca, num_pairs, 3, device=device) # [Num_Samples, Num_Pairs, 3]

    samples_i = torch.einsum("nij,snj->sni", rotmats_i, noise * scales_i) + means_i
    samples_j = torch.einsum("nij,snj->sni", rotmats_j, noise * scales_j) + means_j

    all_samples = torch.cat([samples_i, samples_j])  # [2 * Num_Samples, Num_Pairs, 3]

    new_means = torch.mean(all_samples, dim=0) # [Num_Pairs, 3]
    centered_samples = all_samples - new_means

    covariances = torch.einsum("sni,snj->nij", centered_samples, centered_samples) / (2 * num_samples_for_pca - 1)

    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(covariances)
    except torch.linalg.LinAlgError:
        identity = torch.eye(3, device=device).expand(num_pairs, -1, -1)
        eigenvalues = torch.ones(num_pairs, 3, device=device) * 1e-8
        eigenvectors = identity

    new_scales = torch.log(torch.sqrt(torch.clamp(eigenvalues, min=1e-8)))
    new_quats = matrix_to_quaternion(eigenvectors)

    new_params_dict = {
        "means": new_means, "scales": new_scales, "opacities": new_opacities,
        "quats": new_quats, "sh0": new_sh0, "shN": new_shN
    }

    prune_mask = torch.zeros(len(mask), dtype=torch.bool, device=device)
    prune_mask[indices_i] = True
    prune_mask[indices_j] = True
    kept_mask = ~prune_mask

    def param_fn(name: str, p: Tensor) -> Tensor:
        p_kept = p[kept_mask]
        p_new = new_params_dict.get(name)
        if p_new is None:
            return torch.nn.Parameter(p_kept, requires_grad=p.requires_grad)
        return torch.nn.Parameter(torch.cat([p_kept, p_new]), requires_grad=p.requires_grad)

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        v_kept = v[kept_mask]
        v_new = torch.zeros((len(new_means), *v.shape[1:]), device=v.device)
        return torch.cat([v_kept, v_new])

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)

    for k, v in state.items():
        if isinstance(v, torch.Tensor) and v.shape[0] == len(mask):
            v_kept = v[kept_mask]
            v_from_i, v_from_j = v[indices_i], v[indices_j]
            w_i_reshaped = w_i.view(-1, *([1]*(v.dim()-1)))
            w_j_reshaped = w_j.view(-1, *([1]*(v.dim()-1)))
            v_new = w_i_reshaped * v_from_i + w_j_reshaped * v_from_j
            state[k] = torch.cat([v_kept, v_new])

    return prune_mask, len(new_means)