import random

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch import Tensor, nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import colormaps
# import faiss
# import faiss.contrib.torch_utils


class CameraOptModule(torch.nn.Module):
    """Camera pose optimization module."""

    def __init__(self, n: int):
        super().__init__()
        # Delta positions (3D) + Delta rotations (6D)
        self.embeds = torch.nn.Embedding(n, 9)
        # Identity rotation in 6D representation
        self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def zero_init(self):
        torch.nn.init.zeros_(self.embeds.weight)

    def random_init(self, std: float):
        torch.nn.init.normal_(self.embeds.weight, std=std)

    def forward(self, camtoworlds: Tensor, embed_ids: Tensor) -> Tensor:
        """Adjust camera pose based on deltas.

        Args:
            camtoworlds: (..., 4, 4)
            embed_ids: (...,)

        Returns:
            updated camtoworlds: (..., 4, 4)
        """
        assert camtoworlds.shape[:-2] == embed_ids.shape
        batch_dims = camtoworlds.shape[:-2]
        pose_deltas = self.embeds(embed_ids)  # (..., 9)
        dx, drot = pose_deltas[..., :3], pose_deltas[..., 3:]
        rot = rotation_6d_to_matrix(
            drot + self.identity.expand(*batch_dims, -1)
        )  # (..., 3, 3)
        transform = torch.eye(4, device=pose_deltas.device).repeat((*batch_dims, 1, 1))
        transform[..., :3, :3] = rot
        transform[..., :3, 3] = dx
        return torch.matmul(camtoworlds, transform)


class AppearanceOptModule(torch.nn.Module):
    """Appearance optimization module."""

    def __init__(
            self,
            n: int,
            feature_dim: int,
            embed_dim: int = 16,
            sh_degree: int = 3,
            mlp_width: int = 64,
            mlp_depth: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.sh_degree = sh_degree
        self.embeds = torch.nn.Embedding(n, embed_dim)
        layers = [torch.nn.Linear(embed_dim + feature_dim + (sh_degree + 1) ** 2, mlp_width),
                  torch.nn.PReLU()]
        for _ in range(mlp_depth - 1):
            layers.append(torch.nn.Linear(mlp_width, mlp_width))
            layers.append(torch.nn.PReLU())
        layers.append(torch.nn.Linear(mlp_width, 3))
        self.color_head = torch.nn.Sequential(*layers)

    def forward(
            self, features: Tensor, embed_ids: Tensor, dirs: Tensor, sh_degree: int
    ) -> Tensor:
        """Adjust appearance based on embeddings.

        Args:
            features: (N, feature_dim)
            embed_ids: (C,)
            dirs: (C, N, 3)

        Returns:
            colors: (C, N, 3)
        """
        from gsplat.cuda._torch_impl import _eval_sh_bases_fast

        C, N = dirs.shape[:2]
        # Camera embeddings
        if embed_ids is None:
            embeds = torch.zeros(C, self.embed_dim, device=features.device)
        else:
            embeds = self.embeds(embed_ids)  # [C, D2]
        embeds = embeds[:, None, :].expand(-1, N, -1)  # [C, N, D2]
        # GS features
        features = features[None, :, :].expand(C, -1, -1)  # [C, N, D1]
        # View directions
        dirs = F.normalize(dirs, dim=-1)  # [C, N, 3]
        num_bases_to_use = (sh_degree + 1) ** 2
        num_bases = (self.sh_degree + 1) ** 2
        sh_bases = torch.zeros(C, N, num_bases, device=features.device)  # [C, N, K]
        sh_bases[:, :, :num_bases_to_use] = _eval_sh_bases_fast(num_bases_to_use, dirs)
        # Get colors
        if self.embed_dim > 0:
            h = torch.cat([embeds, features, sh_bases], dim=-1)  # [C, N, D1 + D2 + K]
        else:
            h = torch.cat([features, sh_bases], dim=-1)
        colors = self.color_head(h)
        return colors


def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1]. Adapted from pytorch3d.
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def knn(x: Tensor, K: int = 4, batch_size: int = 1000) -> Tensor:
    return knn_with_ids(x, K, batch_size)[0]

def knn_with_ids_two_tensor(x: Tensor, y: Tensor, K: int = 4, batch_size: int = 2000) -> tuple[Tensor, Tensor]:
    N = x.size(0)
    all_distances = torch.zeros((N, K), device=x.device, dtype=x.dtype)
    all_indices = torch.zeros((N, K), device=x.device, dtype=torch.long)

    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        batch_x = x[i:end]

        dist_matrix = torch.cdist(batch_x, y, p=2)

        distances, indices = torch.topk(dist_matrix, k=K, largest=False, sorted=True)

        all_distances[i:end] = distances
        all_indices[i:end] = indices

        del dist_matrix
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_distances, all_indices

def knn_with_ids(x: Tensor, K: int = 4, batch_size: int = 2000) -> tuple[Tensor, Tensor]:
    N = x.size(0)
    all_distances = torch.zeros((N, K), device=x.device, dtype=x.dtype)
    all_indices = torch.zeros((N, K), device=x.device, dtype=torch.long)

    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        batch_x = x[i:end]

        dist_matrix = torch.cdist(batch_x, x, p=2)

        distances, indices = torch.topk(dist_matrix, k=K, largest=False)

        all_distances[i:end] = distances
        all_indices[i:end] = indices

        del dist_matrix
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_distances, all_indices

# def faiss_knn_with_ids(
#         x: Tensor, K: int = 4
# ) -> tuple[Tensor, Tensor]:
#     x_np = x.cpu().numpy().astype('float32')
#     N, D = x_np.shape
#
#     index_cpu = faiss.IndexFlatL2(D)
#
#     res = faiss.StandardGpuResources()
#     index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)
#
#     index_gpu.add(x_np)
#
#     D_np, I_np = index_gpu.search(x_np, K)
#
#     distances = torch.from_numpy(D_np)
#     indices = torch.from_numpy(I_np).long()
#
#     euclidean_distances = torch.sqrt(distances)
#
#     return euclidean_distances.to(x.device), indices.to(x.device)

def rgb_to_sh(rgb: Tensor) -> Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ref: https://github.com/hbb1/2d-gaussian-splatting/blob/main/utils/general_utils.py#L163
def colormap(img, cmap="jet"):
    W, H = img.shape[:2]
    dpi = 300
    fig, ax = plt.subplots(1, figsize=(H / dpi, W / dpi), dpi=dpi)
    im = ax.imshow(img, cmap=cmap)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = torch.from_numpy(data).float().permute(2, 0, 1)
    plt.close()
    return img


def apply_float_colormap(img: torch.Tensor, colormap: str = "turbo") -> torch.Tensor:
    """Convert single channel to a color img.

    Args:
        img (torch.Tensor): (..., 1) float32 single channel image.
        colormap (str): Colormap for img.

    Returns:
        (..., 3) colored img with colors in [0, 1].
    """
    img = torch.nan_to_num(img, 0)
    if colormap == "gray":
        return img.repeat(1, 1, 3)
    img_long = (img * 255).long()
    img_long_min = torch.min(img_long)
    img_long_max = torch.max(img_long)
    assert img_long_min >= 0, f"the min value is {img_long_min}"
    assert img_long_max <= 255, f"the max value is {img_long_max}"
    return torch.tensor(
        colormaps[colormap].colors,  # type: ignore
        device=img.device,
    )[img_long[..., 0]]


def apply_depth_colormap(
        depth: torch.Tensor,
        acc: torch.Tensor = None,
        near_plane: float = None,
        far_plane: float = None,
) -> torch.Tensor:
    """Converts a depth image to color for easier analysis.

    Args:
        depth (torch.Tensor): (..., 1) float32 depth.
        acc (torch.Tensor | None): (..., 1) optional accumulation mask.
        near_plane: Closest depth to consider. If None, use min image value.
        far_plane: Furthest depth to consider. If None, use max image value.

    Returns:
        (..., 3) colored depth image with colors in [0, 1].
    """
    near_plane = near_plane or float(torch.min(depth))
    far_plane = far_plane or float(torch.max(depth))
    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0.0, 1.0)
    img = apply_float_colormap(depth)
    if acc is not None:
        img = img * acc + (1.0 - acc)
    return img


class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
        self.size = 0

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        return self.tree[0]

    def add(self, p: float, data: object):
        idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(idx, p)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
        if self.size < self.capacity:
            self.size += 1

    def update(self, idx: int, p: float):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s: float) -> tuple[int, float, object]:
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """A Replay Buffer that uses a SumTree to sample experiences based on priority."""
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, beta_increment_per_sampling: float = 0.001):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.epsilon = 1e-5
        self.max_priority = 1.0

    def add(self, experience: object):
        self.tree.add(self.max_priority, experience)

    def sample(self, batch_size: int) -> tuple[list, np.ndarray, np.ndarray]:
        batch, idxs, priorities = [], [], []
        segment = self.tree.total() / batch_size
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            s = np.random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            if data != 0 and data is not None:
                priorities.append(p)
                batch.append(data)
                idxs.append(idx)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.size * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()

        return batch, np.array(idxs), is_weights

    def update_priorities(self, tree_idxs: np.ndarray, td_errors: np.ndarray):
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha
        self.max_priority = max(self.max_priority, np.max(priorities))
        for i, idx in enumerate(tree_idxs):
            self.tree.update(int(idx), priorities[i])

    def __len__(self):
        return self.tree.size

def generate_variational_intrinsics(
        base_K: Tensor,
        num_intrinsics: int,
        focal_perturb_factor: float,
        principal_point_perturb_pixel: int,
) -> Tensor:
    device = base_K.device
    fx_base, fy_base = base_K[0, 0], base_K[1, 1]
    cx_base, cy_base = base_K[0, 2], base_K[1, 2]

    new_Ks = base_K.unsqueeze(0).repeat(num_intrinsics, 1, 1)

    focal_perturb = (
            1.0
            + (torch.rand(num_intrinsics, 2, device=device) * 2 - 1) * focal_perturb_factor
    )
    new_Ks[:, 0, 0] = fx_base * focal_perturb[:, 0]
    new_Ks[:, 1, 1] = fy_base * focal_perturb[:, 1]

    principal_point_perturb = (
                                      torch.rand(num_intrinsics, 2, device=device) * 2 - 1
                              ) * principal_point_perturb_pixel
    new_Ks[:, 0, 2] = cx_base + principal_point_perturb[:, 0]
    new_Ks[:, 1, 2] = cy_base + principal_point_perturb[:, 1]

    return new_Ks

def generate_novel_views(
        base_poses: np.ndarray,
        num_novel_views: int,
        translation_perturbation: float = 0.1,
        rotation_perturbation: float = 5.0,
) -> np.ndarray:
    novel_poses = []
    num_base_poses = base_poses.shape[0]

    for _ in range(num_novel_views):
        base_pose = base_poses[np.random.randint(num_base_poses)]

        translation_offset = np.random.uniform(
            -translation_perturbation, translation_perturbation, size=3
        )
        novel_pose = np.copy(base_pose)
        novel_pose[:3, 3] += translation_offset

        angle_rad = np.deg2rad(
            np.random.uniform(-rotation_perturbation, rotation_perturbation, size=3)
        )
        rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(angle_rad[0]), -np.sin(angle_rad[0])],
                [0, np.sin(angle_rad[0]), np.cos(angle_rad[0])],
            ]
        )
        ry = np.array(
            [
                [np.cos(angle_rad[1]), 0, np.sin(angle_rad[1])],
                [0, 1, 0],
                [-np.sin(angle_rad[1]), 0, np.cos(angle_rad[1])],
            ]
        )
        rz = np.array(
            [
                [np.cos(angle_rad[2]), -np.sin(angle_rad[2]), 0],
                [np.sin(angle_rad[2]), np.cos(angle_rad[2]), 0],
                [0, 0, 1],
            ]
        )
        rotation_offset = rz @ ry @ rx

        novel_pose[:3, :3] = novel_pose[:3, :3] @ rotation_offset

        novel_poses.append(novel_pose)

    return np.array(novel_poses)

@torch.no_grad()
def matrix_to_quaternion(matrix: Tensor) -> Tensor:
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]

    t = m00 + m11 + m22

    q_w = 0.5 * torch.sqrt(torch.clamp(t + 1.0, min=1e-8))
    q_x = 0.5 * torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=1e-8))
    q_y = 0.5 * torch.sqrt(torch.clamp(1.0 - m00 + m11 - m22, min=1e-8))
    q_z = 0.5 * torch.sqrt(torch.clamp(1.0 - m00 - m11 + m22, min=1e-8))

    q_x = torch.copysign(q_x, m21 - m12)
    q_y = torch.copysign(q_y, m02 - m20)
    q_z = torch.copysign(q_z, m10 - m01)

    return torch.stack([q_w, q_x, q_y, q_z], dim=-1)

def scatter_mean(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    out = torch.zeros((dim_size, src.size(1)), device=src.device, dtype=src.dtype)
    index_expanded = index.unsqueeze(1).expand_as(src)
    out.scatter_add_(0, index_expanded, src)

    counts = torch.zeros(dim_size, 1, device=src.device, dtype=src.dtype)
    counts.scatter_add_(0, index.unsqueeze(1), torch.ones_like(src[:, :1]))
    return out / counts.clamp(min=1)