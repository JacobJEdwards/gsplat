import json
import math
import os
import time
from collections import defaultdict
from typing import cast, Any

import imageio
import kornia.color
import numpy as np
import piq
import torch
import torch.nn.functional as F
import tqdm
import tyro
import yaml
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ExponentialLR, ChainedScheduler, CosineAnnealingLR
from torch.utils.checkpoint import checkpoint
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never

from datasets.colmap import Dataset, Parser
from datasets.traj import (
    generate_ellipse_path_z,
    generate_interpolated_path,
    generate_spiral_path,
)
from config import Config
from losses import IlluminationFrequencyLoss
from utils import IlluminationOptModule
from losses import FrequencyLoss, EdgeAwareSmoothingLoss
from losses import (
    ColourConsistencyLoss,
    ExposureLoss,
    SpatialLoss,
    AdaptiveCurveLoss,
    TotalVariationLoss,
    LaplacianLoss,
    GradientLoss,
    LocalExposureLoss
    , ExclusionLoss
)
from rendering_double import rasterization_dual
from gsplat import export_splats
from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.optimizers import SelectiveAdam
from gsplat.rendering import rasterization
from gsplat.strategy import MCMCStrategy, DefaultStrategy
from utils import (
    AppearanceOptModule,
    CameraOptModule,
    knn,
    rgb_to_sh,
    set_random_seed,
)
from retinex import RetinexNet, MultiScaleRetinexNet


def create_splats_with_optimizers(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    means_lr: float = 1.6e-4,
    scales_lr: float = 5e-3,
    opacities_lr: float = 5e-2,
    quats_lr: float = 1e-3,
    sh0_lr: float = 2.5e-3,
    shN_lr: float = 2.5e-3 / 20,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    feature_dim: int | None = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
) -> tuple[
    torch.nn.ParameterDict,
    dict[str, torch.optim.Adam | torch.optim.SparseAdam | SelectiveAdam],
]:
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    dist2_avg = (knn(points)[:, 1:] ** 2).mean(dim=-1)
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)

    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.rand((N, 4))
    opacities = torch.logit(torch.full((N,), init_opacity))

    params = [
        ("means", torch.nn.Parameter(points), means_lr * scene_scale),
        ("scales", torch.nn.Parameter(scales), scales_lr),
        ("quats", torch.nn.Parameter(quats), quats_lr),
        ("opacities", torch.nn.Parameter(opacities), opacities_lr),
    ]

    if feature_dim is None:
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), sh0_lr))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), shN_lr))
    else:
        features = torch.rand(N, feature_dim)
        params.append(("features", torch.nn.Parameter(features), sh0_lr))
        colors = torch.logit(rgbs)
        params.append(("colors", torch.nn.Parameter(colors), sh0_lr))

    # lumincance
    adjust_k = torch.nn.Parameter(
        torch.ones_like(colors[:, :1, :]), requires_grad=True
    )  # enhance, for multiply
    adjust_b = torch.nn.Parameter(
        torch.zeros_like(colors[:, :1, :]), requires_grad=True
    )  # bias, for add,

    params.append(("adjust_k", adjust_k, sh0_lr))
    params.append(("adjust_b", adjust_b, sh0_lr))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)

    BS = batch_size * world_size

    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.AdamW

    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }

    return splats, optimizers


class Runner:
    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        self.start_time = time.time()
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        cfg.result_dir.mkdir(exist_ok=True, parents=True)
        self.ckpt_dir = cfg.result_dir / "ckpts"
        self.ckpt_dir.mkdir(exist_ok=True, parents=True)
        self.stats_dir = cfg.result_dir / "stats"
        self.stats_dir.mkdir(exist_ok=True, parents=True)
        self.render_dir = cfg.result_dir / "renders"
        self.render_dir.mkdir(exist_ok=True, parents=True)
        self.ply_dir = cfg.result_dir / "ply"
        self.ply_dir.mkdir(exist_ok=True, parents=True)

        self.writer = SummaryWriter(log_dir=str(cfg.result_dir / "tb"))

        self.parser = Parser(
            data_dir=str(cfg.data_dir),
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
            # is_mip360=True,
        )
        self.trainset = Dataset(
            self.parser, patch_size=cfg.patch_size, load_depths=cfg.depth_loss
        )
        self.valset = Dataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        self.illum_module = IlluminationOptModule(len(self.trainset)).to(self.device)
        self.illum_module.compile()
        if world_size > 1:
            self.illum_module = DDP(self.illum_module, device_ids=[local_rank])

        initial_lr = 1e-3 * math.sqrt(cfg.batch_size)
        self.illum_optimizers = [
            torch.optim.AdamW(
                self.illum_module.parameters(),
                lr=initial_lr,
                weight_decay=1e-4,
            )
        ]


        if cfg.enable_retinex:
            self.loss_color = ColourConsistencyLoss().to(self.device)
            self.loss_color.compile()
            self.loss_exposure = ExposureLoss(patch_size=32).to(self.device)
            self.loss_exposure.compile()
            # self.loss_smooth = SmoothingLoss().to(self.device)
            # self.loss_smooth = LaplacianLoss().to(self.device)
            self.loss_smooth = TotalVariationLoss().to(self.device)
            self.loss_smooth.compile()
            self.loss_spatial = SpatialLoss(learn_contrast=cfg.learn_spatial_contrast, num_images=len(self.trainset)).to(self.device)
            self.loss_spatial.compile()
            self.loss_adaptive_curve = AdaptiveCurveLoss(learn_lambdas=cfg.learn_adaptive_curve_lambdas).to(self.device)
            self.loss_adaptive_curve.compile()
            self.loss_details = LaplacianLoss().to(self.device)
            self.loss_details.compile()
            self.loss_gradient = GradientLoss().to(self.device)
            self.loss_gradient.compile()
            self.loss_frequency = FrequencyLoss().to(self.device)
            self.loss_frequency.compile()
            self.loss_edge_aware_smooth = EdgeAwareSmoothingLoss().to(self.device)
            self.loss_edge_aware_smooth.compile()
            self.loss_exposure_local = LocalExposureLoss(patch_size=64, patch_grid_size=8).to(self.device)
            self.loss_exposure_local.compile()
            self.loss_illum_frequency = IlluminationFrequencyLoss().to(self.device)
            self.loss_illum_frequency.compile()
            self.loss_exclusion = ExclusionLoss().to(self.device)
            self.loss_exclusion.compile()

            retinex_in_channels = 1 if cfg.use_hsv_color_space else 3
            retinex_out_channels = 1 if cfg.use_hsv_color_space else 3

            self.global_mean_val_param = nn.Parameter(torch.tensor([0.5], dtype=torch.float32).to(self.device))

            if cfg.multi_scale_retinex:
                self.retinex_net = MultiScaleRetinexNet(
                    in_channels=retinex_in_channels,
                    out_channels=retinex_out_channels,
                    use_refinement=cfg.use_refinement_net,
                    predictive_adaptive_curve=cfg.predictive_adaptive_curve,
                    spatially_film=cfg.spatial_film,
                    use_dilated_convs=cfg.use_dilated_convs,
                    use_se_blocks=cfg.use_se_blocks,
                    use_spatial_attention=cfg.use_spatial_attention,
                    enable_dynamic_weights=cfg.enable_dynamic_weights,
                    use_stride_conv=cfg.use_stride_conv,
                    use_pixel_shuffle=cfg.use_pixel_shuffle,
                    num_weight_scales=12
                ).to(self.device)
            else:
                self.retinex_net = RetinexNet(
                    in_channels=retinex_in_channels, out_channels=retinex_out_channels
                ).to(self.device)

            # dpp
            self.retinex_net.compile()

            if world_size > 1:
                self.retinex_net = DDP(self.retinex_net, device_ids=[local_rank])

            net_params = list(self.retinex_net.parameters())
            net_params += self.loss_edge_aware_smooth.parameters()
            net_params += self.loss_adaptive_curve.parameters()
            net_params += self.loss_spatial.parameters()
            net_params.append(self.global_mean_val_param)
            # if cfg.use_denoising_net and self.denoising_net is not None:
            #     net_params += list(self.denoising_net.parameters())

            self.retinex_optimizer = torch.optim.AdamW(
                net_params,
                lr=1e-4 * math.sqrt(cfg.batch_size),
                weight_decay=1e-4,
            )
            if cfg.multi_scale_retinex and self.retinex_net.use_refinement:
                self.refinement_optimizer = torch.optim.AdamW(
                    self.retinex_net.refinement_net.parameters(),
                    lr=1e-4 * math.sqrt(cfg.batch_size),
                    weight_decay=1e-4,
                )

            self.retinex_embed_dim = 32
            self.retinex_embeds = nn.Embedding(
                len(self.trainset), self.retinex_embed_dim
            ).to(self.device)
            self.retinex_embeds.compile()
            torch.nn.init.zeros_(self.retinex_embeds.weight)

            if world_size > 1:
                self.retinex_embeds = DDP(self.retinex_embeds, device_ids=[local_rank])

            self.retinex_embed_optimizer = torch.optim.AdamW(
                [{"params": self.retinex_embeds.parameters(), "lr": 1e-3}]
            )


        feature_dim = 32 if cfg.app_opt else None
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            means_lr=cfg.means_lr,
            scales_lr=cfg.scales_lr,
            opacities_lr=cfg.opacities_lr,
            quats_lr=cfg.quats_lr,
            sh0_lr=cfg.sh0_lr,
            shN_lr=cfg.shN_lr,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))

        self.cfg.strategy.check_sanity(self.splats, self.optimizers)
        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.compile()
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.AdamW(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.compile()
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            self.app_module.compile()
            torch.nn.init.zeros_(cast(Tensor, self.app_module.color_head[-1].weight))
            torch.nn.init.zeros_(cast(Tensor, self.app_module.color_head[-1].bias))
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        self.bil_grid_optimizers = []
        if cfg.use_bilateral_grid:
            global BilateralGrid
            assert BilateralGrid is not None
            self.bil_grids = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1],
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)
            self.bil_grid_optimizers = [
                torch.optim.AdamW(
                    self.bil_grids.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                )
            ]

        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(
                self.device
            )
        elif cfg.lpips_net == "vgg":
            self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(
                self.device
            )
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        self.niqe_metric = None
        if cfg.eval_niqe:
            try:
                self.niqe_metric = piq.BRISQUELoss(data_range=1.0).to(self.device)
                print("BRISQUE metric initialized for evaluation.")
            except Exception as e:
                print(
                    f"Error initializing BRISQUE: {e}. BRISQUE evaluation will be skipped."
                )
                cfg.eval_niqe = False

        # Running stats for prunning & growing.
        n_gauss = len(self.splats["means"])
        self.running_stats = {
            "grad2d": torch.zeros(n_gauss, device=self.device),  # norm of the gradient
            "count": torch.zeros(n_gauss, device=self.device, dtype=torch.int),
        }

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Tensor | None = None,
        **kwargs,
    ) -> (
        tuple[Tensor, Tensor, dict[str, Any]]
        | tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Any]]
    ):
        means = self.splats["means"]
        quats = self.splats["quats"]
        scales = torch.exp(self.splats["scales"])
        opacities = torch.sigmoid(self.splats["opacities"])

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors_feat = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors_feat = colors_feat + self.splats["colors"]
            colors = torch.sigmoid(colors_feat)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)

        rasterize_mode: Literal["antialiased", "classic"] = (
            "antialiased" if self.cfg.antialiased else "classic"
        )

        if not self.cfg.enable_retinex:
            render_colors, render_alphas, info = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=torch.linalg.inv(camtoworlds.float()),
                Ks=Ks,
                width=width,
                height=height,
                packed=self.cfg.packed,
                absgrad=(
                    self.cfg.strategy.absgrad
                    if isinstance(self.cfg.strategy, DefaultStrategy)
                    else False
                ),
                sparse_grad=self.cfg.sparse_grad,
                rasterize_mode=rasterize_mode,
                distributed=self.world_size > 1,
                camera_model=self.cfg.camera_model,
                with_ut=self.cfg.with_ut,
                with_eval3d=self.cfg.with_eval3d,
                **kwargs,
            )
            if masks is not None:
                render_colors[~masks] = 0
            return render_colors, render_alphas, info


        if self.cfg.use_illum_opt:

            image_adjust_k, image_adjust_b = self.illum_module(image_ids)
            colors_low = colors * image_adjust_k + image_adjust_b  # least squares: x_enh=a*x+b

        else:
            adjust_k = self.splats["adjust_k"]  # 1090, 1, 3
            adjust_b = self.splats["adjust_b"]  # 1090, 1, 3

            colors_low = colors * adjust_k + adjust_b  # least squares: x_enh=a*x+b

        (
            render_colors_enh,
            render_colors_low,
            render_enh_alphas,
            render_low_alphas,
            info,
        ) = rasterization_dual(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            colors_low=colors_low,
            viewmats=torch.linalg.inv(camtoworlds.float()),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            **kwargs,
        )

        return (
            render_colors_enh,
            render_colors_low,
            render_enh_alphas,
            render_low_alphas,
            info,
        )  # return colors and alphas

    def get_retinex_output(
        self, images_ids: Tensor, pixels: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
        epsilon = torch.finfo(pixels.dtype).eps
        if self.cfg.use_hsv_color_space:
            pixels_nchw = pixels.permute(0, 3, 1, 2)
            pixels_hsv = kornia.color.rgb_to_hsv(pixels_nchw)
            v_channel = pixels_hsv[:, 2:3, :, :]
            input_image_for_net = v_channel
            log_input_image = torch.log(input_image_for_net + epsilon)
        else:
            pixels_hsv = torch.tensor(0.0, device=self.device)
            input_image_for_net = pixels.permute(0, 3, 1, 2)
            log_input_image = torch.log(input_image_for_net + epsilon)

        retinex_embedding = self.retinex_embeds(images_ids)

        log_illumination_map, alpha, beta, local_exposure, weights = checkpoint(
            self.retinex_net,
            input_image_for_net,
            retinex_embedding,
            use_reentrant=False,
        )
        illumination_map = torch.exp(log_illumination_map)
        illumination_map = torch.clamp(illumination_map, min=1e-5)

        log_reflectance_target = log_input_image - log_illumination_map

        if self.cfg.use_hsv_color_space:
            reflectance_v_target = torch.exp(log_reflectance_target)
            h_channel = pixels_hsv[:, 0:1, :, :]
            s_channel_dampened = pixels_hsv[:, 1:2, :, :]
            reflectance_hsv_target = torch.cat(
                [h_channel, s_channel_dampened, reflectance_v_target], dim=1
            )
            reflectance_map = kornia.color.hsv_to_rgb(reflectance_hsv_target)
        else:
            reflectance_map = torch.exp(log_reflectance_target)

        # if self.cfg.use_denoising_net and self.denoising_net is not None:
        # if self.cfg.use_denoising_embedding:
        #     denoising_embedding = self.retinex_embeds(images_ids)
        #     reflectance_map = self.denoising_net(
        #         reflectance_map, denoising_embedding
        #     )
        # else:
        #     reflectance_map = self.denoising_net(reflectance_map)

        reflectance_map = torch.clamp(reflectance_map, 0, 1)

        return input_image_for_net, illumination_map, reflectance_map, alpha, beta, local_exposure, weights

    def retinex_train_step(
            self, images_ids: Tensor, pixels: Tensor, step: int
    ) -> Tensor:
        cfg = self.cfg
        device = self.device


        (input_image_for_net, illumination_map, reflectance_map, alpha, beta, local_exposure_mean,dynamic_weights_from_net) = (
            self.get_retinex_output(images_ids=images_ids, pixels=pixels)
        )
        global_mean_val_target = torch.sigmoid(self.global_mean_val_param)

        loss_color_val = (
            self.loss_color(illumination_map)
            if not cfg.use_hsv_color_space
            else torch.tensor(0.0, device=device)
        )
        loss_smoothing = self.loss_smooth(illumination_map)
        loss_adaptive_curve = self.loss_adaptive_curve(reflectance_map, alpha, beta)
        if cfg.learn_global_exposure:
            loss_exposure_val = self.loss_exposure(reflectance_map, global_mean_val_target)
        else:
            loss_exposure_val = self.loss_exposure(reflectance_map)

        con_degree = (0.5 / torch.mean(pixels)).item()
        loss_reflectance_spa = self.loss_spatial(input_image_for_net, reflectance_map, contrast=con_degree, image_id=images_ids)
        loss_laplacian_val = torch.mean(
            torch.abs(self.loss_details(reflectance_map) - self.loss_details(input_image_for_net))
        )
        loss_gradient = self.loss_gradient(input_image_for_net, reflectance_map)
        loss_frequency_val = self.loss_frequency(input_image_for_net, reflectance_map)
        loss_smooth_edge_aware = self.loss_edge_aware_smooth(illumination_map, input_image_for_net)
        if cfg.learn_local_exposure:
            loss_exposure_local = self.loss_exposure_local(reflectance_map, local_exposure_mean)
        else:
            loss_exposure_local = self.loss_exposure_local(reflectance_map)

        loss_illumination_frequency_penalty = self.loss_illum_frequency(illumination_map)
        loss_exclusion_val = self.loss_exclusion(reflectance_map, illumination_map)

        individual_losses = torch.stack([
            loss_reflectance_spa,                 # 0
            loss_color_val,                       # 1
            loss_exposure_val,                    # 2
            loss_smoothing,                       # 3
            loss_adaptive_curve,                  # 4
            loss_laplacian_val,                   # 5
            loss_gradient,                        # 6
            loss_frequency_val,                   # 7
            loss_smooth_edge_aware,               # 8
            loss_exposure_local,                  # 9
            loss_illumination_frequency_penalty,  # 10
            loss_exclusion_val,                   # 11
        ])

        base_lambdas = torch.tensor([
            cfg.lambda_reflect,
            cfg.lambda_illum_color,
            cfg.lambda_illum_exposure,
            cfg.lambda_smooth,
            cfg.lambda_illum_curve,
            cfg.lambda_laplacian,
            cfg.lambda_gradient,
            cfg.lambda_frequency,
            cfg.lambda_edge_aware_smooth,
            cfg.lambda_illum_exposure_local,
            cfg.lambda_illum_frequency,
            cfg.lambda_exclusion,
        ], device=device)


        if self.cfg.enable_dynamic_weights and cfg.multi_scale_retinex:
            if isinstance(self.retinex_net, DDP):
                log_vars = self.retinex_net.module.log_vars
            else:
                log_vars = self.retinex_net.log_vars

            term1 = (base_lambdas * individual_losses * torch.exp(-log_vars) / 2.0).sum()
            term2 = (0.5 * log_vars).sum()

            total_loss = term1 + term2

            logged_weights = (base_lambdas * torch.exp(-log_vars) / 2.0)
            logged_log_vars = log_vars

        else:
            total_loss = (base_lambdas * individual_losses).sum()
            logged_weights = base_lambdas
            logged_log_vars = torch.zeros_like(individual_losses, device=device)

        if step % self.cfg.tb_every == 0:
            self.writer.add_scalar("retinex_net/total_loss", total_loss.item(), step)

            loss_names = [
                "reflect_spa", "color_val", "exposure_val", "smoothing",
                "adaptive_curve", "laplacian_val", "gradient", "frequency_val",
                "smooth_edge_aware","exposure_local",
                "illumination_frequency_penalty", "exclusion_val"
            ]

            for i, name in enumerate(loss_names):
                self.writer.add_scalar(f"retinex_net/loss_{name}_unweighted", individual_losses[i].item(), step)
                if self.cfg.enable_dynamic_weights:
                    self.writer.add_scalar(f"retinex_net/loss_{name}_weighted_kendall", (individual_losses[i] * logged_weights[i]).item(), step) # Corrected weighted logging
                    self.writer.add_scalar(f"retinex_net/log_var_{name}", logged_log_vars[i].item(), step)
                    self.writer.add_scalar(f"retinex_net/effective_weight_{name}", logged_weights[i].item(), step) # New logging for effective weight
                else:
                    self.writer.add_scalar(f"retinex_net/loss_{name}_fixed_weighted", (individual_losses[i] * base_lambdas[i]).item(), step)


            if not cfg.predictive_adaptive_curve:
                self.writer.add_scalar("retinex_net/learnable_alpha", self.loss_adaptive_curve.alpha.item(), step)
                self.writer.add_scalar("retinex_net/learnable_beta", self.loss_adaptive_curve.beta.item(), step)

            self.writer.add_scalar("retinex_net/edge_aware_gamma", self.loss_edge_aware_smooth.gamma.item(), step)
            self.writer.add_scalar(
                "retinex_net/global_mean_val_param",
                self.global_mean_val_param.item(),
                step,
            )
            if cfg.multi_scale_retinex:
                self.writer.add_scalar(
                    "retinex_net/local_mean_val_param",
                    local_exposure_mean.mean().item(),
                    step,
                )

            # if cfg.learn_spatial_contrast:
            #     self.writer.add_scalar(
            #         "retinex_net/learned_spatial_contrast",
            #         self.loss_spatial.learnable_contrast.item(),
            #     )

            if cfg.learn_adaptive_curve_lambdas:
                self.writer.add_scalar(
                    "retinex_net/learnable_adaptive_curve_lambda1",
                    self.loss_adaptive_curve.lambda1.item(),
                    step,
                )
                self.writer.add_scalar(
                    "retinex_net/learnable_adaptive_curve_lambda2",
                    self.loss_adaptive_curve.lambda2.item(),
                    step,
                )
                self.writer.add_scalar(
                    "retinex_net/learnable_adaptive_curve_lambda3",
                    self.loss_adaptive_curve.lambda3.item(),
                    step,
                )

            if self.cfg.tb_save_image:
                self.writer.add_images(
                    "retinex_net/input_image_for_net",
                    input_image_for_net,
                    step,
                )
                self.writer.add_images(
                    "retinex_net/pixels",
                    pixels.permute(0, 3, 1, 2),
                    step,
                )

                self.writer.add_images(
                    "retinex_net/illumination_map",
                    illumination_map,
                    step,
                )
                self.writer.add_images(
                    "retinex_net/target_reflectance",
                    reflectance_map,
                    step,
                )

        loss_clipping_high = torch.mean(torch.relu(reflectance_map - 0.98)**2)
        loss_clipping_low = torch.mean(torch.relu(0.02 - reflectance_map)**2)
        lambda_clipping = 1.
        total_loss += lambda_clipping * (loss_clipping_high + loss_clipping_low)

        return total_loss

    def pre_train_retinex(self) -> None:
        cfg = self.cfg
        device = self.device

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )

        trainloader_iter = iter(trainloader)

        initial_retinex_lr = self.retinex_optimizer.param_groups[0]["lr"]
        initial_embed_lr = self.retinex_embed_optimizer.param_groups[0]["lr"]
        schedulers = [
            CosineAnnealingLR(
                self.retinex_optimizer,
                T_max=cfg.pretrain_steps + cfg.max_steps,
                eta_min=initial_retinex_lr * 0.01,
            ),
            CosineAnnealingLR(
                self.retinex_embed_optimizer,
                T_max=cfg.pretrain_steps + cfg.max_steps,
                eta_min=initial_embed_lr * 0.01,
            ),
        ]

        if cfg.multi_scale_retinex and self.retinex_net.use_refinement:
            initial_refinement_lr = self.refinement_optimizer.param_groups[0]["lr"]
            schedulers.append(
                CosineAnnealingLR(
                    self.refinement_optimizer,
                    T_max=cfg.pretrain_steps + cfg.max_steps,
                    eta_min=initial_refinement_lr * 0.01,
                )
            )

        for optimizer in self.illum_optimizers:
            schedulers.append(
                CosineAnnealingLR(
                    optimizer,
                    T_max=cfg.pretrain_steps + cfg.max_steps,
                    eta_min=optimizer.param_groups[0]["lr"] * 0.01,
                )
            )

        pbar = tqdm.tqdm(range(self.cfg.pretrain_steps), desc="Pre-training RetinexNet")
        for step in pbar:
            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            with torch.autocast(enabled=False, device_type=device):
                images_ids = data["image_id"].to(device)
                pixels = data["image"].to(device) / 255.0

                loss = self.retinex_train_step(
                    images_ids=images_ids, pixels=pixels, step=step
                )

            loss.backward()

            self.retinex_optimizer.step()
            self.retinex_embed_optimizer.step()
            if cfg.multi_scale_retinex and self.retinex_net.use_refinement:
                self.refinement_optimizer.step()
                self.refinement_optimizer.zero_grad()

            self.retinex_optimizer.zero_grad()
            self.retinex_embed_optimizer.zero_grad()

            for scheduler in schedulers:
                scheduler.step()

            # scaler.update()

            pbar.set_postfix({"loss": loss.item()})

    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        schedulers: list[ExponentialLR | ChainedScheduler | CosineAnnealingLR] = [
            ExponentialLR(self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)),
        ]
        if cfg.enable_retinex:
            initial_retinex_lr = self.retinex_optimizer.param_groups[0]["lr"]
            schedulers.append(
                CosineAnnealingLR(
                    self.retinex_optimizer,
                    T_max=max_steps,
                    eta_min=initial_retinex_lr * 0.01,
                )
            )
            if cfg.multi_scale_retinex and self.retinex_net.use_refinement:
                initial_refinement_lr = self.refinement_optimizer.param_groups[0]["lr"]
                schedulers.append(
                    CosineAnnealingLR(
                        self.refinement_optimizer,
                        T_max=max_steps,
                        eta_min=initial_refinement_lr * 0.01,
                    )
                )

            initial_embed_lr = self.retinex_embed_optimizer.param_groups[0]["lr"]
            schedulers.append(
                CosineAnnealingLR(
                    self.retinex_embed_optimizer,
                    T_max=max_steps,
                    eta_min=initial_embed_lr * 0.01,
                )
            )

        if cfg.pose_opt:
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        if cfg.use_bilateral_grid:
            global total_variation_loss

            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.bil_grid_optimizers[0],
                            start_factor=0.01,
                            total_iters=1000,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.bil_grid_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                        ),
                    ]
                )
            )

        if cfg.pretrain_retinex and cfg.enable_retinex:
            self.pre_train_retinex()

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        global_tic = time.time()

        pbar = tqdm.tqdm(range(init_step, max_steps))

        for step in pbar:
            # if step % 1000 == 0 and step > 0:
            #     self.pre_train_retinex()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            sh_degree_to_use = min(
                step // cfg.sh_degree_interval, cfg.sh_degree
            )  # Defined early

            with torch.autocast(enabled=False, device_type=device):
                camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)
                Ks = data["K"].to(device)

                image_ids = data["image_id"].to(device)
                pixels = data["image"].to(device) / 255.0

                masks = data["mask"].to(device) if "mask" in data else None

                if cfg.depth_loss:
                    points = data["points"].to(device)
                    depths_gt = data["depths"].to(device)
                else:
                    points = None
                    depths_gt = None

                height, width = pixels.shape[1:3]

                if cfg.pose_noise:
                    camtoworlds = self.pose_perturb(camtoworlds, image_ids)
                if cfg.pose_opt:
                    camtoworlds = self.pose_adjust(camtoworlds, image_ids)

                out = self.rasterize_splats(
                    camtoworlds=camtoworlds,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=sh_degree_to_use,
                    near_plane=cfg.near_plane,
                    far_plane=cfg.far_plane,
                    image_ids=image_ids,
                    render_mode="RGB+ED" if cfg.depth_loss else "RGB",
                    masks=masks,
                )

                if len(out) == 5:
                    renders_enh, renders_low, alphas_enh, alphas_low, info = out
                else:
                    renders_low, alphas_low, info = out
                    renders_enh, alphas_enh = renders_low, alphas_low

                if renders_low.shape[-1] == 4:
                    colors_low, depths_low = (
                        renders_low[..., 0:3],
                        renders_low[..., 3:4],
                    )
                    colors_enh, depths_enh = (
                        renders_enh[..., 0:3],
                        renders_enh[..., 3:4],
                    )
                else:
                    colors_low, depths_low = renders_low, None
                    colors_enh, depths_enh = renders_enh, None

                colors_low = torch.clamp(colors_low, 0.0, 1.0)
                colors_enh = torch.clamp(colors_enh, 0.0, 1.0)
                pixels = torch.clamp(pixels, 0.0, 1.0)

                if cfg.use_bilateral_grid:
                    assert slice_func is not None, (
                        "slice_func must be defined for bilateral grid slicing"
                    )

                    grid_y, grid_x = torch.meshgrid(
                        (torch.arange(height, device=self.device) + 0.5) / height,
                        (torch.arange(width, device=self.device) + 0.5) / width,
                        indexing="ij",
                    )
                    grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

                    colors_low = slice_func(
                        self.bil_grids,
                        grid_xy.expand(colors_low.shape[0], -1, -1, -1),
                        colors_low,
                        image_ids.unsqueeze(-1),
                    )["rgb"]

                    colors_enh = slice_func(
                        self.bil_grids,
                        grid_xy.expand(colors_enh.shape[0], -1, -1, -1),
                        colors_enh,
                        image_ids.unsqueeze(-1),
                    )["rgb"]

                if cfg.random_bkgd:
                    bkgd = torch.rand(1, 3, device=device)
                    colors_low += bkgd * (1.0 - alphas_low)
                    colors_enh += bkgd * (1.0 - alphas_enh)

                info["means2d"].retain_grad()

                if cfg.enable_retinex:
                    input_image_for_net, illumination_map, reflectance_target, alpha, beta, local_exposure, _ = (
                        self.get_retinex_output(images_ids=image_ids, pixels=pixels)
                    )
                    global_exposure_mean = torch.sigmoid(self.global_mean_val_param)

                    reflectance_target_permuted = reflectance_target.permute(
                        0, 2, 3, 1
                    )  # [1, H, W, 3]

                    loss_reconstruct_low = F.l1_loss(colors_low, pixels)
                    ssim_loss = 1.0 - self.ssim(
                        colors_low.permute(0, 3, 1, 2),
                        pixels.permute(0, 3, 1, 2),
                    )
                    lpips_loss = self.lpips(
                        colors_low.permute(0, 3, 1, 2),
                        pixels.permute(0, 3, 1, 2),
                    )

                    low_loss = (
                        ssim_loss * cfg.ssim_lambda
                        + loss_reconstruct_low * (1.0 - cfg.ssim_lambda)
                        + lpips_loss * 0.1
                    )

                    loss_reflectance = F.l1_loss(
                        colors_enh, reflectance_target_permuted.detach()
                    )
                    loss_reflectance_ssim = 1.0 - self.ssim(
                        colors_enh.permute(0, 3, 1, 2),
                        reflectance_target_permuted.permute(0, 3, 1, 2),
                    )
                    # loss_lpips_enh = self.lpips(
                    #     colors_enh.permute(0, 3, 1, 2),
                    #     reflectance_target_permuted.permute(0, 3, 1, 2),
                    # )

                    loss_reconstruct_enh = (
                        loss_reflectance * (1.0 - cfg.ssim_lambda)
                        + loss_reflectance_ssim * cfg.ssim_lambda
                        # + 0.1 * loss_lpips_enh
                    )

                    loss_illum_color = (
                        self.loss_color(illumination_map)
                        if not cfg.use_hsv_color_space
                        else torch.tensor(0.0, device=device)
                    )
                    loss_illum_smooth = self.loss_smooth(illumination_map)
                    # loss_illum_variance = torch.var(illumination_map)

                    loss_adaptive_curve = self.loss_adaptive_curve(reflectance_target, alpha, beta)
                    loss_illum_exposure = self.loss_exposure(reflectance_target, global_exposure_mean)

                    con_degree = (0.5 / torch.mean(pixels)).item()
                    loss_illum_contrast = self.loss_spatial(
                        input_image_for_net, reflectance_target, contrast=con_degree, image_id=image_ids
                    )
                    # loss_illum_laplacian = self.loss_details(reflectance_target)
                    loss_illum_laplacian = torch.mean(
                        torch.abs(self.loss_details(reflectance_target) - self.loss_details(input_image_for_net))
                    )
                    loss_illum_gradient = self.loss_gradient(
                        input_image_for_net, reflectance_target
                    )
                    loss_illum_frequency = self.loss_frequency(
                        input_image_for_net, reflectance_target
                    )
                    loss_illum_edge_aware_smooth = self.loss_edge_aware_smooth(
                        illumination_map, input_image_for_net
                    )


                    loss_illumination = (
                        cfg.lambda_reflect * loss_illum_contrast
                        + cfg.lambda_illum_exposure * loss_illum_exposure
                        + cfg.lambda_illum_color * loss_illum_color
                        + cfg.lambda_smooth * loss_illum_smooth
                        # + cfg.lambda_illum_variance * loss_illum_variance
                        + cfg.lambda_illum_curve * loss_adaptive_curve
                        + cfg.lambda_laplacian * loss_illum_laplacian
                        + cfg.lambda_gradient * loss_illum_gradient
                        + cfg.lambda_frequency * loss_illum_frequency
                        + cfg.lambda_edge_aware_smooth * loss_illum_edge_aware_smooth
                    )

                    # loss = cfg.lambda_reflect * (1 - cfg.lambda_low) + low_loss * cfg.lambda_low # + loss_illumination
                    loss = (
                        low_loss * cfg.lambda_low
                        + loss_reconstruct_enh * (1.0 - cfg.lambda_low)
                        + loss_illumination * cfg.lambda_illumination
                    )
                else:
                    f1 = F.l1_loss(colors_low, pixels)
                    ssim_loss = 1.0 - self.ssim(
                        colors_low.permute(0, 3, 1, 2),
                        pixels.permute(0, 3, 1, 2),
                    )

                    loss_reflectance = f1

                    loss = f1 * (1.0 - cfg.ssim_lambda) + ssim_loss * cfg.ssim_lambda

                    low_loss = loss
                    loss_illumination = torch.tensor(0.0, device=device)
                    loss_illum_color = torch.tensor(0.0, device=device)
                    loss_illum_smooth = torch.tensor(0.0, device=device)
                    # loss_illum_variance = torch.tensor(0.0, device=device)
                    loss_adaptive_curve = torch.tensor(0.0, device=device)
                    loss_illum_exposure = torch.tensor(0.0, device=device)
                    loss_illum_contrast = torch.tensor(0.0, device=device)
                    loss_illum_laplacian = torch.tensor(0.0, device=device)
                    illumination_map = torch.tensor(0.0, device=device)
                    reflectance_target = torch.tensor(0.0, device=device)

                # if cfg.enable_retinex:
                #     k_mean = self.splats["adjust_k"].mean(dim=-1, keepdim=True)
                #     loss_k_gray = torch.mean((self.splats["adjust_k"] - k_mean) ** 2)
                #
                #     loss_b_offset = torch.mean(self.splats["adjust_b"] ** 2)
                #
                #     loss += 0.01 * loss_k_gray + 0.01 * loss_b_offset

                self.cfg.strategy.step_pre_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                )

                depthloss_value = torch.tensor(0.0, device=device)

                if cfg.depth_loss:
                    assert depths_gt is not None, (
                        "Depth ground truth is required for depth loss"
                    )
                    assert points is not None, "Points are required for depth loss"
                    assert depths_low is not None, (
                        "Low-resolution depths are required for depth loss"
                    )

                    # query depths from depth map
                    points = torch.stack(
                        [
                            points[:, :, 0] / (width - 1) * 2 - 1,
                            points[:, :, 1] / (height - 1) * 2 - 1,
                        ],
                        dim=-1,
                    )  # normalize to [-1, 1]
                    grid = points.unsqueeze(2)  # [1, M, 1, 2]
                    depths_low = F.grid_sample(
                        depths_low.permute(0, 3, 1, 2), grid, align_corners=True
                    )  # [1, 1, M, 1]
                    depths_low = depths_low.squeeze(3).squeeze(1)  # [1, M]
                    # calculate loss in disparity space
                    disp = torch.where(
                        depths_low > 0.0, 1.0 / depths_low, torch.zeros_like(depths_low)
                    )
                    disp_gt = 1.0 / depths_gt  # [1, M]
                    depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                    loss += depthloss * cfg.depth_lambda

                    if cfg.enable_retinex:
                        assert depths_enh is not None, (
                            "Enhanced depths are required for enhanced depth loss"
                        )

                        # query depths from depth map
                        points = torch.stack(
                            [
                                points[:, :, 0] / (width - 1) * 2 - 1,
                                points[:, :, 1] / (height - 1) * 2 - 1,
                            ],
                            dim=-1,
                        )  # normalize to [-1, 1]
                        grid = points.unsqueeze(2)  # [1, M, 1, 2]
                        depths_enh = F.grid_sample(
                            depths_enh.permute(0, 3, 1, 2), grid, align_corners=True
                        )  # [1, 1, M, 1]
                        depths_enh = depths_enh.squeeze(3).squeeze(1)  # [1, M]
                        # calculate loss in disparity space
                        disp = torch.where(
                            depths_enh > 0.0,
                            1.0 / depths_enh,
                            torch.zeros_like(depths_enh),
                        )
                        disp_gt = 1.0 / depths_gt  # [1, M]
                        depthloss_enh = F.l1_loss(disp, disp_gt) * self.scene_scale
                        loss += depthloss_enh * cfg.depth_lambda

                tvloss_value: Tensor = torch.tensor(0.0, device=device)
                if cfg.use_bilateral_grid:
                    global total_variation_loss

                    assert total_variation_loss is not None

                    tvloss_value = cast(
                        Tensor, 10 * total_variation_loss(self.bil_grids.grids)
                    )
                    loss += tvloss_value

                if cfg.opacity_reg > 0.0:
                    loss += (
                        cfg.opacity_reg
                        * torch.abs(torch.sigmoid(self.splats["opacities"])).mean()
                    )
                if cfg.scale_reg > 0.0:
                    loss += (
                        cfg.scale_reg
                        * torch.abs(torch.exp(self.splats["scales"])).mean()
                    )

            loss.backward()

            desc_parts = [f"loss={loss.item():.3f}"]
            if cfg.enable_retinex:
                desc_parts.append(
                    f"retinex_loss={loss_reflectance.item():.3f} "
                    # f"illum_smooth={loss_illum_contrast.item():.3f} "
                    # f"illum_color={loss_illum_color.item():.3f} "
                    # f"illum_exposure={loss_illum_exposure.item():.3f}"
                    # f"illum_smooth={loss_illum_smooth.item():.3f}"
                    # f"illum_variance={loss_illum_variance.item():.3f}"
                )
            desc_parts.append(f"sh_deg={sh_degree_to_use}")
            if cfg.depth_loss:
                desc_parts.append(f"depth_l={depthloss_value.item():.6f}")
            if cfg.use_bilateral_grid:
                desc_parts.append(f"tv_l={tvloss_value.item():.6f}")
            if cfg.pose_opt and cfg.pose_noise:
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc_parts.append(f"pose_err={pose_err.item():.6f}")
            pbar.set_description("| ".join(desc_parts))

            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", loss_reflectance.item(), step)
                self.writer.add_scalar("train/ssimloss", ssim_loss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.enable_retinex:
                    self.writer.add_scalar(
                        "train/reflectance_loss", loss_reflectance.item(), step
                    )
                    self.writer.add_scalar(
                        "train/illumination_spatial", loss_illum_contrast.item(), step
                    )
                    self.writer.add_scalar(
                        "train/illumination_smoothing", loss_illum_smooth.item(), step
                    )
                    # self.writer.add_scalar(
                    #     "train/illumination_variance", loss_illum_variance.item(), step
                    # )
                    self.writer.add_scalar(
                        "train/illumination_laplacian",
                        loss_illum_laplacian.item(),
                        step,
                    )
                    # self.writer.add_scalar(
                    #     "train/illumination_loss", loss_illumination.item(), step
                    # )
                    self.writer.add_scalar(
                        "train/illumination_color", loss_illum_color.item(), step
                    )
                    self.writer.add_scalar(
                        "train/illumination_exposure", loss_illum_exposure.item(), step
                    )
                    self.writer.add_scalar(
                        "train/loss_reconstruct_low", low_loss.item(), step
                    )
                    self.writer.add_scalar(
                        "train/adaptive_curve_loss", loss_adaptive_curve.item(), step
                    )
                if cfg.depth_loss:
                    self.writer.add_scalar(
                        "train/depthloss", depthloss_value.item(), step
                    )
                if cfg.use_bilateral_grid:
                    self.writer.add_scalar("train/tvloss", tvloss_value.item(), step)
                if cfg.tb_save_image:
                    with torch.no_grad():
                        self.writer.add_images(
                            "train/render_low", colors_low.permute(0, 3, 1, 2), step
                        )
                        self.writer.add_images(
                            "train/pixels",
                            pixels.permute(0, 3, 1, 2),
                            step,
                        )
                        if cfg.enable_retinex:
                            self.writer.add_images(
                                "train/render_enh",
                                colors_enh.permute(0, 3, 1, 2),
                                step,
                            )
                            self.writer.add_images(
                                "train/illumination_map",
                                illumination_map,
                                step,
                            )
                            self.writer.add_images(
                                "train/reflectance_target",
                                reflectance_target,
                                step,
                            )
                            self.writer.add_images(
                                "train/input_image_for_net",
                                input_image_for_net,
                                step,
                            )

                self.writer.flush()

            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats_save = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("Step: ", step, stats_save)
                with open(
                    f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                    "w",
                ) as f:
                    json.dump(stats_save, f)
                data_save = {"step": step, "splats": self.splats.state_dict()}
                if cfg.pose_opt:
                    data_save["pose_adjust"] = (
                        self.pose_adjust.module.state_dict()
                        if world_size > 1
                        else self.pose_adjust.state_dict()
                    )
                if cfg.app_opt:
                    data_save["app_module"] = (
                        self.app_module.module.state_dict()
                        if world_size > 1
                        else self.app_module.state_dict()
                    )
                if cfg.use_bilateral_grid:
                    data_save["bil_grids"] = (
                        self.bil_grids.module.state_dict()
                        if world_size > 1
                        else self.bil_grids.state_dict()
                    )

                if cfg.enable_retinex:
                    data_save["retinex_net"] = (
                        self.retinex_net.module.state_dict()
                        if world_size > 1
                        else self.retinex_net.state_dict()
                    )
                torch.save(
                    data_save, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                )
            if (
                step in [i - 1 for i in cfg.ply_steps] or step == max_steps - 1
            ) and cfg.save_ply:
                if self.cfg.app_opt:
                    rgb_export_feat = self.app_module(
                        features=self.splats["features"],
                        embed_ids=None,
                        dirs=torch.zeros_like(self.splats["means"][None, :, :]),
                        sh_degree=sh_degree_to_use,
                    )
                    rgb_export_feat = rgb_export_feat + self.splats["colors"]
                    rgb_export = torch.sigmoid(rgb_export_feat).squeeze(0).unsqueeze(1)
                    sh0_export = rgb_to_sh(rgb_export)
                    shN_export = torch.empty(
                        [sh0_export.shape[0], 0, 3], device=sh0_export.device
                    )
                else:
                    sh0_export = self.splats["sh0"]
                    shN_export = self.splats["shN"]
                export_splats(
                    means=self.splats["means"],
                    scales=self.splats["scales"],
                    quats=self.splats["quats"],
                    opacities=self.splats["opacities"],
                    sh0=sh0_export,
                    shN=shN_export,
                    save_to=f"{self.ply_dir}/point_cloud_{step}.ply",
                )

            if cfg.sparse_grad:
                assert cfg.packed
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad_val = self.splats[k].grad
                    if grad_val is None or grad_val.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],
                        values=grad_val[gaussian_ids],
                        size=self.splats[k].size(),
                        is_coalesced=len(Ks) == 1,
                    )

            if cfg.visible_adam:
                if cfg.packed:
                    visibility_mask = torch.zeros_like(
                        self.splats["opacities"], dtype=torch.bool
                    )
                    visibility_mask.scatter_(0, info["gaussian_ids"], 1)
                else:
                    visibility_mask = (info["radii"] > 0).all(-1).any(0)
            else:
                visibility_mask = None

            for optimizer in self.optimizers.values():
                if cfg.visible_adam:
                    optimizer.step(visibility_mask)
                else:
                    optimizer.step()
                optimizer.zero_grad()
            for optimizer in self.pose_optimizers:
                # scaler.step(optimizer)
                optimizer.step()
                optimizer.zero_grad()
            for optimizer in self.app_optimizers:
                # scaler.step(optimizer)
                optimizer.step()
                optimizer.zero_grad()
            for optimizer in self.bil_grid_optimizers:
                # scaler.step(optimizer)
                optimizer.step()
                optimizer.zero_grad()
            
            for optimizer in self.illum_optimizers:
                # scaler.step(optimizer)
                optimizer.step()
                optimizer.zero_grad()

            if cfg.enable_retinex:
                # scaler.step(self.retinex_optimizer)
                self.retinex_optimizer.step()
                self.retinex_optimizer.zero_grad()
                # scaler.step(self.retinex_embed_optimizer)
                self.retinex_embed_optimizer.step()
                self.retinex_embed_optimizer.zero_grad()
                if cfg.multi_scale_retinex and self.retinex_net.use_refinement:
                    # scaler.step(self.retinex_refinement_optimizer)
                    self.refinement_optimizer.step()
                    self.refinement_optimizer.zero_grad()

            # scaler.update()

            for scheduler in schedulers:
                scheduler.step()

            if isinstance(self.cfg.strategy, DefaultStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=schedulers[0].get_last_lr()[0],
                )
            else:
                assert_never(self.cfg.strategy)

            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step)
            if step in [i - 1 for i in cfg.eval_steps]:
                self.render_traj(step)
            if cfg.compression is not None and step in [i - 1 for i in cfg.eval_steps]:
                self.run_compression(step=step)

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        print(f"Running evaluation for step {step} on '{stage}' set...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank

        valloader = torch.utils.data.DataLoader(
            self.valset, shuffle=False, num_workers=1
        )
        ellipse_time_total = 0
        metrics = defaultdict(list)
        for i, data in enumerate(tqdm.tqdm(valloader, desc=f"Eval {stage}")):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)

            pixels = data["image"].to(device) / 255.0
            image_id = data["image_id"].to(device)

            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()

            out = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                masks=masks,
                image_ids=image_id,
            )

            if len(out) == 5:
                colors_enh, colors_low, alphas_enh, alphas_low, info = out
            else:
                colors_low, alphas_low, info = out
                colors_enh, alphas_enh = colors_low, colors_low

            torch.cuda.synchronize()
            ellipse_time_total += max(time.time() - tic, 1e-10)

            colors_low = torch.clamp(colors_low, 0.0, 1.0)
            colors_enh = torch.clamp(colors_enh, 0.0, 1.0)

            if world_rank == 0:
                canvas_list_low = [pixels, colors_low]
                canvas_list_enh = [pixels, colors_enh]

                canvas_eval_low = (
                    torch.cat(canvas_list_low, dim=2).squeeze(0).cpu().numpy()
                )
                canvas_eval_low = (canvas_eval_low * 255).astype(np.uint8)

                canvas_eval_enh = (
                    torch.cat(canvas_list_enh, dim=2).squeeze(0).cpu().numpy()
                )
                canvas_eval_enh = (canvas_eval_enh * 255).astype(np.uint8)

                imageio.imwrite(
                    f"{self.render_dir}/{stage}_step{step}_low_{i:04d}.png",
                    canvas_eval_low,
                )

                colors_low_np = colors_low.squeeze(0).cpu().numpy()

                imageio.imwrite(
                    f"{self.render_dir}/{stage}_low_{i:04d}.png",
                    (colors_low_np * 255).astype(np.uint8),
                )

                if cfg.enable_retinex:
                    imageio.imwrite(
                        f"{self.render_dir}/{stage}_step{step}_enh_{i:04d}.png",
                        canvas_eval_enh,
                    )
                    colors_enh_np = colors_enh.squeeze().cpu().numpy()
                    imageio.imwrite(
                        f"{self.render_dir}/{stage}_enh_{i:04d}.png",
                        (colors_enh_np * 255).astype(np.uint8),
                    )

                pixels_p = pixels.permute(0, 3, 1, 2)
                colors_p = colors_low.permute(0, 3, 1, 2)

                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))

                if cfg.eval_niqe and self.niqe_metric is not None:
                    niqe_score = self.niqe_metric(colors_p.contiguous())
                    metrics["niqe"].append(niqe_score)

                    if cfg.enable_retinex:
                        colors_enh_p = colors_enh.permute(0, 3, 1, 2)
                        niqe_score_enh = self.niqe_metric(colors_enh_p.contiguous())
                        metrics["niqe_enh"].append(niqe_score_enh)

                if cfg.use_bilateral_grid:
                    global color_correct
                    assert color_correct is not None
                    cc_colors = color_correct(colors_low, pixels)
                    cc_colors_p = cc_colors.permute(0, 3, 1, 2)
                    metrics["cc_psnr"].append(self.psnr(cc_colors_p, pixels_p))
                    metrics["cc_ssim"].append(self.ssim(cc_colors_p, pixels_p))
                    metrics["cc_lpips"].append(self.lpips(cc_colors_p, pixels_p))

                if cfg.enable_retinex:
                    with torch.no_grad():
                        # _, _, reflectance_target = self.get_retinex_output(image_id, pixels)
                        colors_enh_p = colors_enh.permute(0, 3, 1, 2)
                        # reflectance_target_p = reflectance_target

                        metrics["lpips_enh"].append(self.lpips(colors_enh_p, pixels_p))
                        metrics["ssim_enh"].append(self.ssim(colors_enh_p, pixels_p))
                        metrics["psnr_enh"].append(self.psnr(colors_enh_p, pixels_p))

        if world_rank == 0:
            avg_ellipse_time = (
                ellipse_time_total / len(valloader) if len(valloader) > 0 else 0
            )

            stats_eval = {}
            for k, v_list in metrics.items():
                if v_list:
                    if isinstance(v_list[0], torch.Tensor):
                        stats_eval[k] = torch.stack(v_list).mean().item()
                    else:
                        stats_eval[k] = sum(v_list) / len(v_list)
                else:
                    stats_eval[k] = 0

            stats_eval.update(
                {
                    "ellipse_time": avg_ellipse_time,
                    "num_GS": len(self.splats["means"]),
                    "total_time": time.time() - self.start_time,
                }
            )

            print_parts_eval = [
                f"PSNR: {stats_eval.get('psnr', 0):.3f}",
                f"SSIM: {stats_eval.get('ssim', 0):.4f}",
                f"LPIPS: {stats_eval.get('lpips', 0):.3f}",
            ]
            if cfg.eval_niqe and self.niqe_metric is not None and "niqe" in stats_eval:
                print_parts_eval.append(
                    f"BRISQUE: {stats_eval.get('niqe', 0):.3f} (lower is better)"
                )
            if cfg.use_bilateral_grid:
                print_parts_eval.extend(
                    [
                        f"CC_PSNR: {stats_eval.get('cc_psnr', 0):.3f}",
                        f"CC_SSIM: {stats_eval.get('cc_ssim', 0):.4f}",
                        f"CC_LPIPS: {stats_eval.get('cc_lpips', 0):.3f}",
                    ]
                )
            print_parts_eval.extend(
                [
                    f"Time: {stats_eval.get('ellipse_time', 0):.3f}s/image",
                    f"GS: {stats_eval.get('num_GS', 0)}",
                ]
            )
            print(f"Eval {stage} Step {step}: " + " | ".join(print_parts_eval))

            raw_metrics = {}
            for k, v_list in metrics.items():
                if v_list:
                    if isinstance(v_list[0], torch.Tensor):
                        raw_metrics[k] = [v.item() for v in v_list]
                    else:
                        raw_metrics[k] = v_list
                else:
                    raw_metrics[k] = []

            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats_eval, f)
            with open(f"{self.stats_dir}/{stage}_raw_step{step:04d}.json", "w") as f:
                json.dump(raw_metrics, f)
            for k_stat, v_stat in stats_eval.items():
                self.writer.add_scalar(f"{stage}/{k_stat}", v_stat, step)
            self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int):
        if self.cfg.disable_video:
            return
        if self.world_rank != 0:
            return

        print(f"Running trajectory rendering for step {step}...")
        cfg = self.cfg
        device = self.device

        camtoworlds_all_np = self.parser.camtoworlds
        if not len(camtoworlds_all_np):
            print("No camera poses found for trajectory rendering. Skipping.")
            return

        if cfg.render_traj_path == "interp":
            camtoworlds_all_np = generate_interpolated_path(camtoworlds_all_np, 1)
        elif cfg.render_traj_path == "ellipse":
            height_mean = camtoworlds_all_np[:, 2, 3].mean()
            camtoworlds_all_np = generate_ellipse_path_z(
                camtoworlds_all_np, height=height_mean
            )
        elif cfg.render_traj_path == "spiral":
            camtoworlds_all_np = generate_spiral_path(
                camtoworlds_all_np,
                bounds=self.parser.bounds * self.scene_scale,
                spiral_scale_r=self.parser.extconf.get("spiral_radius_scale", 0.5),
            )
        else:
            raise ValueError(
                f"Render trajectory type not supported: {cfg.render_traj_path}"
            )

        camtoworlds_all_np = np.concatenate(
            [
                camtoworlds_all_np,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all_np), axis=0
                ),
            ],
            axis=1,
        )

        camtoworlds_all_torch = torch.from_numpy(camtoworlds_all_np).float().to(device)

        first_val_cam_key = (
            list(self.parser.Ks_dict.keys())[0] if self.parser.Ks_dict else None
        )
        if not first_val_cam_key:
            print("No camera intrinsics found for trajectory rendering. Skipping.")
            return

        K_traj = (
            torch.from_numpy(self.parser.Ks_dict[first_val_cam_key]).float().to(device)
        )
        width_traj, height_traj = self.parser.imsize_dict[first_val_cam_key]

        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        video_path = f"{video_dir}/traj_{step}.mp4"
        video_writer = imageio.get_writer(video_path, fps=30)

        all_frame_niqe_scores = []

        for i in tqdm.trange(len(camtoworlds_all_torch), desc="Rendering trajectory"):
            cam_c2w = camtoworlds_all_torch[i : i + 1]
            cam_K = K_traj[None]

            out = self.rasterize_splats(
                camtoworlds=cam_c2w,
                Ks=cam_K,
                width=width_traj,
                height=height_traj,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )

            if len(out) == 5:
                renders_traj, _, _, _, _ = out
            else:
                renders_traj, _, _ = out

            colors_traj = torch.clamp(renders_traj[..., 0:3], 0.0, 1.0)
            depths_traj = renders_traj[..., 3:4]
            depths_traj_norm = (depths_traj - depths_traj.min()) / (
                depths_traj.max() - depths_traj.min() + 1e-10
            )

            if cfg.eval_niqe and self.niqe_metric is not None:
                colors_traj_nchw = colors_traj.permute(0, 3, 1, 2).contiguous()
                niqe_score_traj = self.niqe_metric(colors_traj_nchw)
                all_frame_niqe_scores.append(niqe_score_traj.item())

            canvas_traj_list = [colors_traj, depths_traj_norm.repeat(1, 1, 1, 3)]
            canvas_traj = torch.cat(canvas_traj_list, dim=2).squeeze(0).cpu().numpy()
            canvas_traj_uint8 = (canvas_traj * 255).astype(np.uint8)
            video_writer.append_data(canvas_traj_uint8)
        video_writer.close()
        print(f"Video saved to {video_path}")

        if cfg.eval_niqe and self.niqe_metric is not None and all_frame_niqe_scores:
            avg_traj_niqe = sum(all_frame_niqe_scores) / len(all_frame_niqe_scores)
            print(
                f"Average BRISQUE for trajectory video (step {step}): {avg_traj_niqe:.3f} (Lower is better)"
            )
            self.writer.add_scalar(
                f"render_traj/avg_niqe_step_{step}", avg_traj_niqe, step
            )
        self.writer.flush()

    @torch.no_grad()
    def run_compression(self, step: int):
        if self.compression_method is None:
            return
        print("Running compression...")
        world_rank = self.world_rank

        compress_dir = f"{self.cfg.result_dir}/compression/rank{world_rank}"
        os.makedirs(compress_dir, exist_ok=True)

        self.compression_method.compress(compress_dir, self.splats.state_dict())
        splats_c = self.compression_method.decompress(compress_dir)
        for k_splat in splats_c.keys():
            self.splats[k_splat].data = splats_c[k_splat].to(self.device)
        self.eval(step=step, stage="compress")


def main(local_rank: int, world_rank, world_size: int, cfg_param: Config):
    if world_size > 1 and not cfg_param.disable_viewer:
        cfg_param.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg_param)

    if cfg_param.ckpt is not None:
        ckpts_loaded = [
            torch.load(file, map_location=runner.device) for file in cfg_param.ckpt
        ]
        if ckpts_loaded:
            if len(cfg_param.ckpt) > 1 and world_size > 1:
                for k_splat in runner.splats.keys():
                    runner.splats[k_splat].data = torch.cat(
                        [c["splats"][k_splat] for c in ckpts_loaded]
                    )
            else:
                runner.splats.load_state_dict(ckpts_loaded[0]["splats"])

            step_loaded = ckpts_loaded[0]["step"]
            if cfg_param.pose_opt and "pose_adjust" in ckpts_loaded[0]:
                pose_dict = ckpts_loaded[0]["pose_adjust"]
                if world_size > 1:
                    runner.pose_adjust.module.load_state_dict(pose_dict)
                else:
                    runner.pose_adjust.load_state_dict(pose_dict)
            if cfg_param.app_opt and "app_module" in ckpts_loaded[0]:
                app_dict = ckpts_loaded[0]["app_module"]
                if world_size > 1:
                    runner.app_module.module.load_state_dict(app_dict)
                else:
                    runner.app_module.load_state_dict(app_dict)

            print(f"Resuming from checkpoint step {step_loaded}")
            runner.eval(step=step_loaded)
            runner.render_traj(step=step_loaded)
            if cfg_param.compression is not None:
                runner.run_compression(step=step_loaded)
    else:
        runner.train()


BilateralGrid = None
color_correct = None
slice_func = None
total_variation_loss = None

if __name__ == "__main__":
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(strategy=DefaultStrategy(verbose=True, refine_stop_iter=8000)),
        ),
        "mcmc": (
            "Gaussian splatting training using MCMC.",
            Config(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }
    # config = tyro.extras.overridable_config_cli(configs)
    config = tyro.cli(
        Config,
    )

    config.adjust_steps(config.steps_scaler)

    if config.use_bilateral_grid or config.use_fused_bilagrid:
        if config.use_fused_bilagrid:
            config.use_bilateral_grid = True
            try:
                from fused_bilagrid import (
                    BilateralGrid as FusedBilateralGrid,
                    color_correct as fused_color_correct,
                    slice as fused_slice,
                    total_variation_loss as fused_total_variation_loss,
                )

                BilateralGrid = FusedBilateralGrid
                color_correct = fused_color_correct
                slice_func = fused_slice
                total_variation_loss = fused_total_variation_loss
                print("Using Fused Bilateral Grid.")
            except ImportError:
                raise ImportError(
                    "Fused Bilateral Grid components not found. Please ensure it's installed."
                )
        else:
            config.use_bilateral_grid = True
            try:
                from lib_bilagrid import (
                    BilateralGrid as LibBilateralGrid,
                    color_correct as lib_color_correct,
                    bi_slice as lib_slice,
                    total_variation_loss as lib_total_variation_loss,
                )

                BilateralGrid = LibBilateralGrid
                color_correct = lib_color_correct
                slice_func = lib_slice
                total_variation_loss = lib_total_variation_loss
                print("Using Standard Bilateral Grid (lib_bilagrid).")
            except ImportError:
                raise ImportError(
                    "Standard Bilateral Grid (lib_bilagrid) components not found."
                )

    if config.compression == "png":
        try:
            import plas
            import torchpq
        except ImportError:
            raise ImportError(
                "To use PNG compression, you need to install torchpq and plas. "
                "torchpq: https://github.com/DeMoriarty/TorchPQ "
                "plas: pip install git+https://github.com/fraunhoferhhi/PLAS.git"
            )

    if config.with_ut:
        assert config.with_eval3d, (
            "Training with UT requires setting `with_eval3d` flag."
        )

    torch.set_float32_matmul_precision("high")

    cli(main, config, verbose=True)
