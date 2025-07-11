import torch
from torch import Tensor
import torch.nn.functional as F

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
