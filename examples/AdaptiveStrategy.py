from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn, autocast, GradScaler
from torch.distributions import Categorical

from ops import duplicate, split
from utils import knn_with_ids, create_view_proj_matrix
from gsplat.strategy.default import DefaultStrategy
from gsplat.strategy.ops import remove, reset_opa
from tensordict import TensorDict
from torchrl.data import RandomSampler, TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyMemmapStorage
import torch_geometric.nn as gnn

class GraphEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.conv1 = gnn.GATv2Conv(input_dim, hidden_dim, heads=4, concat=True, edge_dim=1)
        self.conv2 = gnn.GATv2Conv(hidden_dim * 4, hidden_dim, heads=4, concat=True, edge_dim=1)
        self.output_head = nn.Linear(hidden_dim * 4, output_dim)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        x = F.elu(self.conv1(x, edge_index, edge_attr=edge_attr))
        x = F.elu(self.conv2(x, edge_index, edge_attr=edge_attr))
        return self.output_head(x)


class PerNodeActorCritic(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Dropout(0.1),
        )
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, per_gaussian_features: Tensor) -> tuple[Categorical, Tensor]:
        x = self.shared_net(per_gaussian_features)
        action_logits = self.actor_head(x)
        action_dist = Categorical(logits=action_logits)
        value = self.critic_head(x).squeeze(-1)
        return action_dist, value

class WorldModel(nn.Module):
    def __init__(self, scene_embed_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(scene_embed_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, scene_embed_dim)
        )

    def forward(self, scene_embed: Tensor) -> Tensor:
        return self.net(scene_embed)

@dataclass
class AdaptiveStrategy(DefaultStrategy):
    refine_every: int = 400
    learn_every: int = 100
    refine_start_iter: int = 400

    prune_min_age: int = 1000
    prune_significance_thresh: float = 0.01

    gnn_input_dim: int = 11
    gnn_hidden_dim: int = 64
    gnn_output_dim: int = 128
    ac_hidden_dim: int = 128
    wm_hidden_dim: int = 256

    ac_learning_rate: float = 3e-4
    wm_learning_rate: float = 1e-4
    ppo_clip_epsilon: float = 0.2
    entropy_loss_weight: float = 0.01
    critic_loss_weight: float = 0.5
    intrinsic_reward_factor: float = 0.1

    uncertainty_bonus_factor: float = 0.05

    reward_delay: int = 400
    gauss_count_penalty_factor: float = 0.01

    reward_weight_lpips: float = 50.0
    reward_weight_psnr: float = 0.1
    reward_weight_ssim: float = 10.0
    reward_weight_l1: float = 20.0
    reward_weight_mse: float = 20.0

    num_reward_views: int = 4

    reward_validation_set: list[dict] = field(default_factory=list, repr=False)
    rasterize_fn: Any = field(default=None, repr=False)

    max_densification_subset: int = 50_000

    graph_encoder: Any = field(default=None, repr=False)
    actor_critic: Any = field(default=None, repr=False)
    world_model: Any = field(default=None, repr=False)
    ac_optimizer: Any = field(default=None, repr=False)
    wm_optimizer: Any = field(default=None, repr=False)
    lpips_metric: Any = field(default=None, repr=False)
    psnr_metric: Any = field(default=None, repr=False)
    ssim_metric: Any = field(default=None, repr=False)

    grad_scaler: Any = field(default=None, repr=False)

    writer: Any = field(default=None, repr=False)

    def setup_validation_set(self, validation_dataset: torch.utils.data.Dataset, device: torch.device) -> None:
        if not validation_dataset: return

        indices = torch.randperm(len(validation_dataset))[:self.num_reward_views].tolist()

        for i in indices:
            data = validation_dataset[i]
            self.reward_validation_set.append({
                "camtoworld": data["camtoworld"].to(device),
                "K": data["K"].to(device),
                "pixels": data["image"].to(device) / 255.0,
            })
        if self.verbose:
            print(f"Created a fixed reward validation set with {len(self.reward_validation_set)} views.")

    def _calculate_metrics(self, params: dict, camtoworlds: Tensor, Ks: Tensor, pixels: Tensor, step: int) -> dict:
        height, width = pixels.shape[1:3]
        sh_degree_to_use = min(step // 1000, 3)
        colors = torch.cat([params["sh0"], params["shN"]], 1)  # [N, K, 3]
        render_colors, _, _ = self.rasterize_fn(
            camtoworlds=camtoworlds.unsqueeze(0),
            Ks=Ks.unsqueeze(0),
            width=width,
            height=height,
            sh_degree=sh_degree_to_use,
            near_plane=0.01,
            far_plane=1e10,
            means=params["means"], scales=torch.exp(params["scales"]), quats=params["quats"],
            opacities=torch.sigmoid(params["opacities"]), colors=colors,
        )
        rendered_img_p = render_colors.permute(0, 3, 1, 2)
        gt_img_p = pixels.permute(0, 3, 1, 2)

        # lpips_score = self.lpips_metric(rendered_img_p, gt_img_p)
        psnr_score = self.psnr_metric(rendered_img_p, gt_img_p)
        ssim_score = self.ssim_metric(rendered_img_p, gt_img_p)
        l1_loss = F.l1_loss(rendered_img_p, gt_img_p)
        mse_loss = F.mse_loss(rendered_img_p, gt_img_p)

        metrics = {
            # "lpips": lpips_score.item(),
            "psnr": psnr_score.item(),
            "ssim": ssim_score.item(),
            "l1": l1_loss.item(),
            "mse": mse_loss.item(),
        }
        return metrics


    @torch.no_grad()
    def _calculate_avg_metrics(self, params: dict, step: int) -> dict:
        sh_degree_to_use = min(step // 1000, 3)

        camtoworlds = torch.stack([data["camtoworld"] for data in self.reward_validation_set], dim=0)
        Ks = torch.stack([data["K"] for data in self.reward_validation_set], dim=0)
        gt_pixels = torch.stack([data["pixels"] for data in self.reward_validation_set], dim=0)

        height, width = gt_pixels.shape[1:3]
        colors = torch.cat([params["sh0"], params["shN"]], 1)

        render_colors, _, _ = self.rasterize_fn(
            camtoworlds=camtoworlds,
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=sh_degree_to_use,
            near_plane=0.01,
            far_plane=1e10,
            render_mode="RGB",
            means=params["means"],
            scales=torch.exp(params["scales"]),
            quats=params["quats"],
            opacities=torch.sigmoid(params["opacities"]),
            colors=colors,
        )

        rendered_img_p = render_colors.permute(0, 3, 1, 2)
        gt_img_p = gt_pixels.permute(0, 3, 1, 2)

        # total_lpips = self.lpips_metric(rendered_img_p, gt_img_p)
        total_psnr = self.psnr_metric(rendered_img_p, gt_img_p)
        total_ssim = self.ssim_metric(rendered_img_p, gt_img_p)
        total_l1 = F.l1_loss(rendered_img_p, gt_img_p)
        total_mse = F.mse_loss(rendered_img_p, gt_img_p)

        return {
            # "lpips": total_lpips.mean(),
            "psnr": total_psnr.mean(),
            "ssim": total_ssim.mean(),
            "l1": total_l1.mean(),
            "mse": total_mse.mean(),
        }

    def initialize_state(self, scene_scale: float = 1.0) -> dict[str, Any]:
        state = super().initialize_state(scene_scale)
        state.update({
            "age": None,
            "replay_buffer": TensorDictReplayBuffer(
                storage=LazyMemmapStorage(max_size=150_000), sampler=RandomSampler(), batch_size=2048,
            ),
            "reward_queue": deque(maxlen=self.max_densification_subset * 5),
            "l1_loss_map": None,
            "detail_error_map": None,
        })

        return state

    def _initialize_learning_components(self, device: torch.device) -> None:
        self.graph_encoder = GraphEncoder(self.gnn_input_dim, self.gnn_hidden_dim, self.gnn_output_dim).to(device)
        self.actor_critic = PerNodeActorCritic(self.gnn_output_dim, self.ac_hidden_dim, 3).to(device)
        self.world_model = WorldModel(self.gnn_output_dim, self.wm_hidden_dim).to(device)

        self.grad_scaler = GradScaler()

        ac_params = list(self.graph_encoder.parameters()) + list(self.actor_critic.parameters())
        self.ac_optimizer = torch.optim.AdamW(ac_params, lr=self.ac_learning_rate)
        self.wm_optimizer = torch.optim.AdamW(self.world_model.parameters(), lr=self.wm_learning_rate)

        if self.verbose:
            print("Initialized FINAL densification agent (GNN + Per-Node AC + World Model).")

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
        state["step"] = step

        if self.actor_critic is None:
            self._initialize_learning_components(params["means"].device)

        if state.get("age") is None:
            state["age"] = torch.zeros(params["means"].shape[0], dtype=torch.int32, device=params["means"].device)

        state["age"] += 1
        state["l1_loss_map"] = info.get("l1_loss_map", None)
        state["detail_error_map"] = info.get("detail_error_map", None)
        state["lpips_score"] = info.get("lpips_score", None)

        self._update_quality_map(params, state, info)
        self._process_rewards(params, state, step, info)
        self._update_state(params, state, info, packed=packed)

        if step > self.refine_start_iter and step % self.refine_every == 0:
            n_prune = self.prune_gs(params, optimizers, state)
            n_split, n_duplicate = self.grow_gs(params, optimizers, state, info)
            if self.verbose:
                print(f"Step {step}: Pruned {n_prune}, Split {n_split}, Duplicated {n_duplicate}.")

        if step > self.refine_start_iter and step % self.learn_every == 0:
            self._train_models(state)

        if step > 0 and step % self.reset_every == 0:
            reset_opa(params, optimizers, state, self.prune_opa * 2.0)


    @torch.no_grad()
    def prune_gs(self, params: dict, optimizers: dict, state: dict) -> int:
        step = state.get("step", 0)

        is_prune_original = torch.sigmoid(params["opacities"].flatten()) < self.prune_opa
        if step > self.reset_every:
            is_too_big = (
                    torch.exp(params["scales"]).max(dim=-1).values
                    > self.prune_scale3d * state["scene_scale"]
            )
            if step < self.refine_scale2d_stop_iter:
                is_too_big |= state["radii"] > self.prune_scale2d
            is_prune_original |= is_too_big

        is_prune_significant = torch.zeros_like(is_prune_original)
        if "significance" in state and state["significance"] is not None and state["significance"].numel() > 0:
            if state["significance"].numel() == is_prune_original.numel():
                is_prune_significant = state["significance"] < self.prune_significance_thresh

        is_prune = is_prune_original | is_prune_significant

        no_prune = state["age"] < self.prune_min_age

        is_prune &= (~no_prune)

        n_prune = is_prune.sum().item()
        if n_prune > 0:
            per_gaussian_state_keys = ["grad2d", "count", "radii", "stagnation_count", "prev_grad2d", "prev_opacity",
                                       "significance", "age", "gaussian_contribution"]
            state_to_prune = {k: v for k, v in state.items() if k in per_gaussian_state_keys and v is not None}

            remove(params=params, optimizers=optimizers, state=state_to_prune, mask=is_prune)

            state.update(state_to_prune)

        return n_prune


    @torch.no_grad()
    def grow_gs(self, params: dict, optimizers: dict, state: dict, info: dict) -> tuple[int, int]:
        device = params["means"].device
        normalized_grads = state["grad2d"] / state["count"].clamp_min(1.0)
        candidate_mask = normalized_grads > self.grow_grad2d

        num_candidates = candidate_mask.sum().item()
        if num_candidates > self.max_densification_subset:
            candidate_indices = torch.where(candidate_mask)[0]
            rand_indices = torch.randperm(num_candidates, device=device)[:self.max_densification_subset]

            new_mask = torch.zeros_like(candidate_mask)
            new_mask[candidate_indices[rand_indices]] = True
            candidate_mask = new_mask

        if candidate_mask.sum() == 0:
            return 0, 0

        original_indices = torch.where(candidate_mask)[0]

        with autocast(enabled=True, device_type="cuda"):
            per_gaussian_features = self._get_features_from_graph(params, state, candidate_mask)

            self.actor_critic.train()
            num_passes = 5
            all_logits = [self.actor_critic(per_gaussian_features)[0].logits for _ in range(num_passes)]
            all_logits_tensor = torch.stack(all_logits)
            uncertainty = all_logits_tensor.var(dim=0).mean(dim=-1)
            self.actor_critic.eval()

            mean_logits = all_logits_tensor.mean(dim=0)
            action_dist = Categorical(logits=mean_logits)
            actions = action_dist.sample()
            _, values = self.actor_critic(per_gaussian_features)


        initial_avg_metrics = self._calculate_avg_metrics(params, step=state["step"])
        self._queue_per_node_experience(state, info, per_gaussian_features, actions, action_dist.log_prob(actions),
                                        values, uncertainty, initial_avg_metrics)

        split_action_mask = (actions == 1)
        duplicate_action_mask = (actions == 2)

        global_split_mask = torch.zeros_like(candidate_mask)
        global_split_mask[original_indices[split_action_mask]] = True

        global_duplicate_mask = torch.zeros_like(candidate_mask)
        global_duplicate_mask[original_indices[duplicate_action_mask]] = True

        state_to_modify = {k: v for k, v in state.items() if k in ["grad2d", "count", "radii", "age"]}
        n_split = global_split_mask.sum().item()
        n_duplicate = global_duplicate_mask.sum().item()

        if n_split > 0:
            split(params, optimizers, state_to_modify, global_split_mask)
        if n_duplicate > 0:
            duplicate(params, optimizers, state_to_modify, global_duplicate_mask)

        if n_split > 0 or n_duplicate > 0:
            num_new = (n_split * 2) + n_duplicate - (n_split + n_duplicate)
            state_to_modify["age"][-num_new:] = 0

        state.update(state_to_modify)

        return n_split, n_duplicate

    def _train_models(self, state: dict):
        if len(state["replay_buffer"]) < state["replay_buffer"].batch_size: return
        device = self.actor_critic.parameters().__next__().device
        batch = state["replay_buffer"].sample().to(device)

        features = batch.get("features")
        scene_encodings = batch.get("scene_encoding")
        next_scene_encodings = batch.get("next_scene_encoding")
        actions = batch.get("action").squeeze(-1)
        rewards_raw = batch.get("reward").squeeze(-1)
        old_log_probs = batch.get("log_prob").squeeze(-1)
        old_values = batch.get("value").squeeze(-1)

        with torch.autocast(enabled=True, device_type="cuda"):
            next_scene_pred = self.world_model(scene_encodings)
            wm_loss = F.mse_loss(next_scene_pred, next_scene_encodings)

        self.wm_optimizer.zero_grad()
        # wm_loss.backward()
        self.grad_scaler.backward(wm_loss)
        # self.wm_optimizer.step()
        self.grad_scaler.step(self.wm_optimizer)

        with autocast(enabled=True, device_type="cuda"), torch.no_grad():
            rewards = (rewards_raw - rewards_raw.mean()) / (rewards_raw.std() + 1e-8)

            next_values = self.actor_critic(next_scene_encodings)[1]

            delta = rewards + 0.99 * next_values - old_values
            advantages = (delta - delta.mean()) / (delta.std() + 1e-8)
            returns = advantages + old_values

        with autocast(enabled=True, device_type="cuda"):
            new_dist, new_values = self.actor_critic(features)
            new_log_probs = new_dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip_epsilon, 1.0 + self.ppo_clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(new_values, returns)
            entropy_loss = -new_dist.entropy().mean()
            ac_loss = actor_loss + self.critic_loss_weight * critic_loss + self.entropy_loss_weight * entropy_loss

        self.ac_optimizer.zero_grad()
        # ac_loss.backward()
        self.grad_scaler.backward(ac_loss)
        # self.ac_optimizer.step()
        self.grad_scaler.step(self.ac_optimizer)
        self.grad_scaler.update()


        self.writer.add_scalar("agent/ac_loss", ac_loss.item(), state["step"])
        self.writer.add_scalar("agent/actor_loss", actor_loss.item(), state["step"])
        self.writer.add_scalar("agent/critic_loss", critic_loss.item(), state["step"])
        self.writer.add_scalar("agent/entropy_loss", entropy_loss.item(), state["step"])
        self.writer.add_scalar("agent/wm_loss", wm_loss.item(), state["step"])
        self.writer.add_scalar("agent/mean_reward", rewards.mean().item(), state["step"])


    @torch.no_grad()
    def _queue_per_node_experience(self, state: dict, info: dict, features: Tensor, actions: Tensor, log_probs: Tensor,
                                   values: Tensor, uncertainty: Tensor, initial_avg_metrics: dict) -> None:

        scene_encoding = features.mean(dim=0).detach()

        for i in range(features.shape[0]):
            experience = {
                "step": state["step"], "features": features[i].detach(),
                "action": actions[i].detach(), "log_prob": log_probs[i].detach(),
                "value": values[i].detach(), "uncertainty": uncertainty[i].detach(),
                "scene_encoding": scene_encoding,
                "initial_avg_metrics": {k: v.detach() for k, v in initial_avg_metrics.items()},
                "initial_gauss_count": state["age"].shape[0]
            }
            state["reward_queue"].append(experience)


    def _process_rewards(self, params: dict, state: dict, current_step: int, info: dict) -> None:
        queue = state["reward_queue"]
        if not queue or (current_step - queue[0]["step"]) < self.reward_delay: return

        with torch.no_grad(), autocast(enabled=True, device_type="cuda"):
            all_node_features = self._get_features_from_graph(params, state, torch.ones(params["means"].shape[0], dtype=torch.bool, device=params["means"].device))
            if all_node_features.shape[0] == 0: return
            current_scene_encoding = all_node_features.mean(dim=0).detach()

        current_avg_metrics = self._calculate_avg_metrics(params, step=current_step)

        for name, val in current_avg_metrics.items():
            self.writer.add_scalar(f"metrics/avg_{name}", val, current_step)

        while queue and (current_step - queue[0]["step"]) >= self.reward_delay:
            exp = queue.popleft()

            initial_metrics = exp["initial_avg_metrics"]

            # delta_lpips = initial_metrics["lpips"] - current_avg_metrics["lpips"]
            delta_l1 = initial_metrics["l1"] - current_avg_metrics["l1"]
            delta_mse = initial_metrics["mse"] - current_avg_metrics["mse"]

            delta_psnr = current_avg_metrics["psnr"] - initial_metrics["psnr"]
            delta_ssim = current_avg_metrics["ssim"] - initial_metrics["ssim"]

            extrinsic_reward = (
                    # self.reward_weight_lpips * delta_lpips +
                    self.reward_weight_psnr * delta_psnr +
                    self.reward_weight_ssim * delta_ssim +
                    self.reward_weight_l1 * delta_l1 +
                    self.reward_weight_mse * delta_mse
            )

            with autocast(enabled=True, device_type="cuda"):
                predicted_next_encoding = self.world_model(exp["scene_encoding"])
                intrinsic_reward = F.mse_loss(predicted_next_encoding, current_scene_encoding.detach())

            gauss_count_now = params["means"].shape[0]
            penalty = self.gauss_count_penalty_factor * max(0, gauss_count_now - exp["initial_gauss_count"])
            uncertainty_bonus = exp["uncertainty"]
            reward = (extrinsic_reward
                      + self.intrinsic_reward_factor * intrinsic_reward
                      + self.uncertainty_bonus_factor * uncertainty_bonus
                      - penalty)
            reward = reward.clamp(-2.0, 2.0)

            if len(state["replay_buffer"]) < state["replay_buffer"]._storage.max_size:
                td = TensorDict({
                    "features": exp["features"],
                    "scene_encoding": exp["scene_encoding"],
                    "next_scene_encoding": current_scene_encoding,
                    "action": exp["action"], "log_prob": exp["log_prob"],
                    "value": exp["value"], "reward": reward.clamp(-10.0, 10.0).detach(),
                }, batch_size=[])
                state["replay_buffer"].add(td)




    @torch.no_grad()
    def _get_features_from_graph(self, params: dict, state: dict, subset_mask: Tensor) -> Tensor:
        device = params["means"].device
        all_indices = torch.where(subset_mask)[0]
        if all_indices.numel() == 0:
            return torch.zeros(0, self.gnn_output_dim, device=device)

        means = params["means"][subset_mask]
        scales = torch.log(torch.exp(params["scales"][subset_mask]).mean(dim=-1, keepdim=True))
        opacities = params["opacities"][subset_mask].reshape(-1, 1)
        quats = params["quats"][subset_mask]
        ages = state["age"][subset_mask].unsqueeze(-1).float() / 1000.0
        grads2d = (state["grad2d"][subset_mask] / state["count"][subset_mask].clamp_min(1.0)).unsqueeze(-1)
        node_features = torch.cat([means, scales, opacities, quats, ages, grads2d], dim=-1)

        node_features = F.layer_norm(node_features, [node_features.shape[-1]])

        dists, nn_indices = knn_with_ids(means, K=16)
        source_nodes = torch.arange(means.shape[0], device=device).view(-1, 1).repeat(1, 16).flatten()
        target_nodes = nn_indices.flatten()
        edge_index = torch.stack([source_nodes, target_nodes])
        edge_attr = dists.flatten().unsqueeze(-1) / state["scene_scale"]

        encoded_features = self.graph_encoder(node_features, edge_index, edge_attr=edge_attr)

        # scene_encoding = encoded_features.mean(dim=0, keepdim=True)

        return encoded_features


    def _update_quality_map(self, params: dict, state: dict, info: dict):
        if info.get("camtoworlds") is None: return
        width, height = info['width'], info['height']
        view_proj_matrix, _, _ = create_view_proj_matrix(info["camtoworlds"][0], info["Ks"][0], width, height)
        state["view_proj_matrix"] = view_proj_matrix