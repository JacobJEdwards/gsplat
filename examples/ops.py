import torch
from torch import Tensor
import torch.nn.functional as F

from utils import knn_with_ids, knn_with_ids_two_tensor
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
):
    """Inplace split the Gaussian with the given mask.

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        mask: A boolean mask to split the Gaussians.
        revised_opacity: Whether to use revised opacity formulation
          from arXiv:2404.06109. Default: False.
        anisotropic: Whether to split along the largest variance axis. Default: False.
        split_ratios (Optional[Tensor]): Per-Gaussian split ratios. If None, uses a default.
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
            current_split_ratios = split_ratios if split_ratios is not None else torch.full((len(scales),), 1.6, device=device)
            if anisotropic:
                new_scales_val = scales.clone()
                new_scales_val[torch.arange(len(scales)), largest_scale_idx] /= current_split_ratios
                p_split = torch.log(new_scales_val).repeat(2, 1)
            else:
                p_split = torch.log(scales / current_split_ratios[:, None]).repeat(2, 1)  # [2N, 3]
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

@torch.no_grad()
def duplicate(
        params: dict[str, torch.nn.Parameter] | torch.nn.ParameterDict,
        optimizers: dict[str, torch.optim.Optimizer],
        state: dict[str, Tensor],
        mask: Tensor,
        offsets: Tensor | None = None,
):
    """
    Inplace duplicate the Gaussian with the given mask, with optional displacement.

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        mask: A boolean mask to duplicate the Gaussians.
        offsets (Optional[Tensor]): A tensor of shape [N_dupe, 3] containing the
          displacement vectors for the new Gaussians. If None, duplicates
          are placed at the original location.
    """
    device = mask.device
    sel = torch.where(mask)[0]

    if offsets is not None:
        assert len(sel) == len(offsets), "Number of offsets must match number of Gaussians to duplicate."

    def param_fn(name: str, p: Tensor) -> Tensor:
        # If we are modifying the means and offsets are provided, apply them.
        if name == "means" and offsets is not None:
            new_params = p[sel] + offsets
        # For all other parameters, or if offsets are not provided, just clone.
        else:
            new_params = p[sel]

        return torch.nn.Parameter(torch.cat([p, new_params]), requires_grad=p.requires_grad)

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        # Initialize optimizer state for new params as zeros.
        return torch.cat([v, torch.zeros((len(sel), *v.shape[1:]), device=device)])

    # update the parameters and the state in the optimizers
    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    # update the extra running state
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = torch.cat((v, v[sel]))

@torch.no_grad()
def merge(
        params: dict[str, torch.nn.Parameter] | torch.nn.ParameterDict,
        optimizers: dict[str, torch.optim.Optimizer],
        state: dict[str, Tensor],
        mask: Tensor,
) -> tuple[Tensor, int]:
    """
    In-place merges selected Gaussians with their nearest neighbors, reducing the total count.

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        state: A dictionary of running states (e.g. age, grad2d).
        mask: A boolean mask indicating which Gaussians should initiate a merge.

    Returns:
        A tuple containing:
        - A boolean mask of the same size as the input, indicating which Gaussians were removed.
        - An integer count of the new Gaussians that were created.
    """
    if not mask.any():
        return torch.zeros_like(mask, dtype=torch.bool), 0

    device = mask.device
    source_indices = torch.where(mask)[0]
    all_means = params["means"]

    _, nn_indices = knn_with_ids_two_tensor(all_means[source_indices], all_means, K=2)
    target_indices = nn_indices[:, 1]

    unique_pairs = set()
    for i, j in zip(source_indices.tolist(), target_indices.tolist()):
        if i != j:
            pair = tuple(sorted((i, j)))
            unique_pairs.add(pair)

    if not unique_pairs:
        return torch.zeros_like(mask, dtype=torch.bool), 0

    pairs = torch.tensor(list(unique_pairs), device=device)
    indices_i, indices_j = pairs[:, 0], pairs[:, 1]

    opacities_i = torch.sigmoid(params["opacities"][indices_i])
    opacities_j = torch.sigmoid(params["opacities"][indices_j])

    alpha_sum = opacities_i + opacities_j + 1e-8
    w_i = opacities_i / alpha_sum
    w_j = opacities_j / alpha_sum

    new_opacities_alpha = 1.0 - (1.0 - opacities_i) * (1.0 - opacities_j)
    new_opacities = torch.logit(torch.clamp(new_opacities_alpha, 1e-6, 1.0 - 1e-6))

    new_means = w_i.unsqueeze(-1) * params["means"][indices_i] + w_j.unsqueeze(-1) * params["means"][indices_j]
    new_scales = w_i.unsqueeze(-1) * torch.exp(params["scales"][indices_i]) + w_j.unsqueeze(-1) * torch.exp(params["scales"][indices_j])
    new_scales = torch.log(torch.clamp(new_scales, min=1e-8))

    quat_mask = (opacities_i > opacities_j).unsqueeze(-1)
    new_quats = torch.where(quat_mask, params["quats"][indices_i], params["quats"][indices_j])

    colors_i = torch.cat([params["sh0"][indices_i], params["shN"][indices_i]], dim=1)
    colors_j = torch.cat([params["sh0"][indices_j], params["shN"][indices_j]], dim=1)
    new_colors = w_i.unsqueeze(-1).unsqueeze(-1) * colors_i + w_j.unsqueeze(-1).unsqueeze(-1) * colors_j
    new_sh0, new_shN = new_colors[:, :1], new_colors[:, 1:]

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