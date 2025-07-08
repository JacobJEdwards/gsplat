import math
import random

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch import Tensor, nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import colormaps


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
                  torch.nn.ReLU(inplace=True)]
        for _ in range(mlp_depth - 1):
            layers.append(torch.nn.Linear(mlp_width, mlp_width))
            layers.append(torch.nn.ReLU(inplace=True))
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


def knn(x: Tensor, K: int = 4) -> Tensor:
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)


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

class PoseGeneratorModule(torch.nn.Module):
    def __init__(self, mlp_width: int = 64, mlp_depth: int = 3,
                 output_dim: int = 9 + 4, ):
        super().__init__()
        input_dim = 32

        layers = [torch.nn.Linear(input_dim, mlp_width), torch.nn.PReLU()]
        for _ in range(mlp_depth - 1):
            layers.append(torch.nn.Linear(mlp_width, mlp_width))
            layers.append(torch.nn.PReLU())
        layers.append(torch.nn.Linear(mlp_width, output_dim))

        self.net = torch.nn.Sequential(*layers)

        self.register_buffer("identity_rot", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        raw_output = self.net(z)
        pose_deltas = raw_output[..., :9]
        intrinsic_deltas = raw_output[..., 9:]

        return pose_deltas, intrinsic_deltas

class ImprovedPoseGeneratorModule(torch.nn.Module):
    def __init__(self,
                 mlp_width: int = 128,
                 mlp_depth: int = 4,
                 noise_dim: int = 32,
                 condition_dim: int = 16,
                 use_attention: bool = True,
                 use_spectral_norm: bool = True):
        super().__init__()

        self.noise_dim = noise_dim
        self.condition_dim = condition_dim
        self.use_attention = use_attention

        self.condition_encoder = nn.Sequential(
            nn.Linear(9, condition_dim),
            nn.LayerNorm(condition_dim),
            nn.ReLU(),
            nn.Linear(condition_dim, condition_dim),
            nn.LayerNorm(condition_dim),
            nn.ReLU()
        )

        input_dim = noise_dim + condition_dim

        layers = [nn.Linear(input_dim, mlp_width), nn.LayerNorm(mlp_width), nn.ReLU()]

        for i in range(mlp_depth - 1):
            layer = nn.Linear(mlp_width, mlp_width)
            if use_spectral_norm:
                layer = nn.utils.spectral_norm(layer)
            layers.append(layer)
            layers.append(nn.LayerNorm(mlp_width))
            layers.append(nn.ReLU())

        if use_attention:
            self.attention = nn.MultiheadAttention(mlp_width, num_heads=8, batch_first=True)

        self.pose_head = nn.Sequential(
            nn.Linear(mlp_width, mlp_width // 2),
            nn.ReLU(),
            nn.Linear(mlp_width // 2, 9)  # 3 trans + 6 rot
        )

        self.intrinsic_head = nn.Sequential(
            nn.Linear(mlp_width, mlp_width // 2),
            nn.ReLU(),
            nn.Linear(mlp_width // 2, 4)  # fx, fy, cx, cy deltas
        )

        self.net = nn.Sequential(*layers)

        self.register_buffer("identity_rot", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor, scene_condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        condition_encoded = self.condition_encoder(scene_condition)

        x = torch.cat([z, condition_encoded], dim=-1)

        x = self.net(x)

        if self.use_attention:
            x_unsqueezed = x.unsqueeze(1)
            x_attended, _ = self.attention(x_unsqueezed, x_unsqueezed, x_unsqueezed)
            x = x_attended.squeeze(1)

        pose_deltas = self.pose_head(x)
        intrinsic_deltas = self.intrinsic_head(x)

        return pose_deltas, intrinsic_deltas

class AdaptiveNovelViewSampler:
    def __init__(self,
                 base_poses: np.ndarray,
                 quality_history_size: int = 1000,
                 temperature: float = 0.1):
        self.base_poses = base_poses
        self.quality_history = []
        self.pose_history = []
        self.quality_history_size = quality_history_size
        self.temperature = temperature

        self.scene_stats = self._compute_scene_stats()

    def _compute_scene_stats(self) -> torch.Tensor:
        positions = self.base_poses[:, :3, 3]
        rotations = self.base_poses[:, :3, :3]

        euler_angles = []
        for rot in rotations:
            sy = np.sqrt(rot[0, 0]**2 + rot[1, 0]**2)
            singular = sy < 1e-6
            if not singular:
                x = np.arctan2(rot[2, 1], rot[2, 2])
                y = np.arctan2(-rot[2, 0], sy)
                z = np.arctan2(rot[1, 0], rot[0, 0])
            else:
                x = np.arctan2(-rot[1, 2], rot[1, 1])
                y = np.arctan2(-rot[2, 0], sy)
                z = 0
            euler_angles.append([x, y, z])

        euler_angles = np.array(euler_angles)

        pos_mean = np.mean(positions, axis=0)
        pos_std = np.std(positions, axis=0)
        rot_mean = np.mean(euler_angles, axis=0)

        return torch.tensor(np.concatenate([pos_mean, pos_std, rot_mean]), dtype=torch.float32)

    def update_quality_history(self, poses: np.ndarray, qualities: np.ndarray):
        qualities_iterable = np.atleast_1d(qualities)
        
        for pose, quality in zip(poses, qualities_iterable):
            self.pose_history.append(pose.copy())
            self.quality_history.append(quality)

            if len(self.quality_history) > self.quality_history_size:
                self.quality_history.pop(0)
                self.pose_history.pop(0)

    def get_difficulty_weighted_poses(self, num_poses: int) -> np.ndarray:
        if len(self.quality_history) < 10:
            return self._generate_random_poses(num_poses)

        qualities = np.array(self.quality_history)
        poses = np.array(self.pose_history)

        probs = F.softmax(torch.tensor(qualities) / self.temperature, dim=0).numpy()

        sampled_indices = np.random.choice(len(poses), size=num_poses//2, p=probs)
        difficult_poses = poses[sampled_indices]

        random_poses = self._generate_random_poses(num_poses - num_poses//2)

        return np.concatenate([difficult_poses, random_poses], axis=0)

    def _generate_random_poses(self, num_poses: int) -> np.ndarray:
        """Generate random poses around base poses."""
        return generate_novel_views(
            self.base_poses,
            num_novel_views=num_poses,
            translation_perturbation=0.15,
            rotation_perturbation=8.0
        )

class HierarchicalPoseGenerator:
    def __init__(self, base_poses: np.ndarray, num_levels: int = 3):
        self.base_poses = base_poses
        self.num_levels = num_levels
        self.level_perturbations = self._compute_level_perturbations()

    def _compute_level_perturbations(self) -> list[tuple[float, float]]:
        base_trans = 0.05
        base_rot = 2.0

        perturbations = []
        for level in range(self.num_levels):
            scale = 2.0 ** level
            perturbations.append((base_trans * scale, base_rot * scale))

        return perturbations

    def generate_hierarchical_poses(self, num_poses_per_level: int) -> np.ndarray:
        """Generate poses at multiple difficulty levels."""
        all_poses = []

        for level, (trans_pert, rot_pert) in enumerate(self.level_perturbations):
            level_poses = generate_novel_views(
                self.base_poses,
                num_novel_views=num_poses_per_level,
                translation_perturbation=trans_pert,
                rotation_perturbation=rot_pert
            )
            all_poses.append(level_poses)

        return np.concatenate(all_poses, axis=0)


class CurriculumPoseScheduler:
    def __init__(self,
                 total_steps: int,
                 initial_difficulty: float = 0.1,
                 final_difficulty: float = 1.0,
                 warmup_steps: int = 1000):
        self.total_steps = total_steps
        self.initial_difficulty = initial_difficulty
        self.final_difficulty = final_difficulty
        self.warmup_steps = warmup_steps

    def get_difficulty(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.initial_difficulty

        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(1.0, progress)

        difficulty = self.initial_difficulty + (self.final_difficulty - self.initial_difficulty) * \
                     (1 - math.cos(progress * math.pi)) / 2

        return difficulty


def generate_quality_aware_poses(base_poses: np.ndarray,
                                 num_poses: int,
                                 quality_scores: np.ndarray | None = None,
                                 diversity_weight: float = 0.3) -> np.ndarray:
    """Generate poses that balance quality and diversity."""
    if quality_scores is None:
        return generate_novel_views(base_poses, num_poses)

    # Use quality scores to guide sampling
    quality_weights = F.softmax(torch.tensor(quality_scores) * 2.0, dim=0).numpy()

    # Sample base poses based on quality
    selected_bases = np.random.choice(
        len(base_poses),
        size=num_poses,
        p=quality_weights
    )

    # Generate variations with adaptive perturbation
    novel_poses = []
    for i, base_idx in enumerate(selected_bases):
        base_pose = base_poses[base_idx]

        # Adaptive perturbation based on quality
        quality = quality_scores[base_idx]
        trans_scale = 0.05 + 0.15 * quality  # Higher quality = more perturbation
        rot_scale = 2.0 + 8.0 * quality

        # Add diversity term
        diversity_factor = 1.0 + diversity_weight * (i / num_poses)

        pose_variation = generate_novel_views(
            base_pose[None, :],
            num_novel_views=1,
            translation_perturbation=trans_scale * diversity_factor,
            rotation_perturbation=rot_scale * diversity_factor
        )[0]

        novel_poses.append(pose_variation)

    return np.array(novel_poses)


def generate_interpolated_challenging_poses(base_poses: np.ndarray,
                                            num_poses: int,
                                            challenge_factor: float = 0.5) -> np.ndarray:
    novel_poses = []

    for _ in range(num_poses):
        idx1, idx2 = np.random.choice(len(base_poses), 2, replace=False)
        pose1, pose2 = base_poses[idx1], base_poses[idx2]

        pos_dist = np.linalg.norm(pose1[:3, 3] - pose2[:3, 3])

        t = np.random.uniform(0.2, 0.8)
        if pos_dist > np.percentile([np.linalg.norm(p1[:3, 3] - p2[:3, 3])
                                     for p1 in base_poses for p2 in base_poses], 70):
            t = t * challenge_factor + (1 - challenge_factor) * 0.5

        interp_pos = (1 - t) * pose1[:3, 3] + t * pose2[:3, 3]

        interp_rot = (1 - t) * pose1[:3, :3] + t * pose2[:3, :3]

        u, s, vh = np.linalg.svd(interp_rot)
        interp_rot = u @ vh

        interp_pose = np.eye(4)
        interp_pose[:3, :3] = interp_rot
        interp_pose[:3, 3] = interp_pos

        perturbation = np.random.normal(0, 0.02, 3)
        interp_pose[:3, 3] += perturbation

        novel_poses.append(interp_pose)

    return np.array(novel_poses)

def integrate_enhanced_pose_generation(runner_instance, step: int):
    cfg = runner_instance.cfg

    if not hasattr(runner_instance, 'adaptive_sampler'):
        runner_instance.adaptive_sampler = AdaptiveNovelViewSampler(
            runner_instance.parser.camtoworlds
        )
        runner_instance.curriculum_scheduler = CurriculumPoseScheduler(
            total_steps=cfg.max_steps
        )
        runner_instance.hierarchical_generator = HierarchicalPoseGenerator(
            runner_instance.parser.camtoworlds
        )

    difficulty = runner_instance.curriculum_scheduler.get_difficulty(step)

    if step % 3 == 0:
        novel_poses = runner_instance.adaptive_sampler.get_difficulty_weighted_poses(
            cfg.num_novel_poses
        )
    elif step % 3 == 1:
        novel_poses = runner_instance.hierarchical_generator.generate_hierarchical_poses(
            cfg.num_novel_poses // 3
        )
    else:
        novel_poses = generate_interpolated_challenging_poses(
            runner_instance.parser.camtoworlds,
            cfg.num_novel_poses,
            challenge_factor=difficulty
        )

    return torch.from_numpy(novel_poses).float().to(runner_instance.device)