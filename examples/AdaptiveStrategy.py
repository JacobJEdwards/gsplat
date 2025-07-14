from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from gsplat.strategy.default import DefaultStrategy
from gsplat.strategy.ops import remove, split, reset_opa
from tensordict import TensorDict
from torchrl.data import RandomSampler, TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyMemmapStorage

class ActionValueNetwork(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features)

@dataclass
class AdaptiveStrategy(DefaultStrategy):
    refine_every: int = 400
    learn_every: int = 200

    feature_dim: int = 5
    hidden_dim: int = 64
    action_dim: int = 3  # 0: No-op, 1: Split, 2: Prune
    learning_rate: float = 1e-4
    reward_delay: int = 200
    exploration_epsilon: float = 0.2
    max_densification_subset: int = 100_000

    prune_opacity_threshold: float = 0.005
    split_opacity_reset: float = 0.01

    q_net: Any = field(default=None, repr=False)
    optimizer: Any = field(default=None, repr=False)
    writer: Any = field(default=None, repr=False)

    def initialize_state(self, scene_scale: float = 1.0) -> dict[str, Any]:
        state = super().initialize_state(scene_scale)
        replay_buffer = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(max_size=50_000),
            sampler=RandomSampler(),
            batch_size=256
        )
        reward_queue = deque(maxlen=10_000)

        state.update({
            "age": None,
            "l1_loss_map": None,
            "view_proj_matrix": None,
            "replay_buffer": replay_buffer,
            "reward_queue": reward_queue,
        })
        return state

    def _initialize_learning_components(self, device: torch.device) -> None:
        self.q_net = ActionValueNetwork(self.feature_dim, self.hidden_dim, self.action_dim).to(device)
        self.optimizer = torch.optim.AdamW(self.q_net.parameters(), lr=self.learning_rate)
        print("Initialized Action-Value (Q-Learning) Network.")


    def step_post_backward(
            self, params: dict, optimizers: dict, state: dict, step: int, info: dict, **kwargs
    ) -> None:
        if step >= self.refine_stop_iter: return

        if self.q_net is None:
            self._initialize_learning_components(params["means"].device)
        if state.get("age") is None:
            state["age"] = torch.zeros(params["means"].shape[0], dtype=torch.int32, device=params["means"].device)

        state["age"] += 1
        state["l1_loss_map"] = info.get("l1_loss_map")
        state["view_proj_matrix"] = info.get("view_proj_matrix")

        self._process_rewards(state, step)

        if step > self.refine_start_iter and step % self.refine_every == 0:
            self._update_geometry(params, optimizers, state, step)

        if step > self.refine_start_iter and step % self.learn_every == 0:
            self._train_agent(state)

        if step % self.reset_every == 0 and step > 0:
            reset_opa(params, optimizers, state, self.prune_opa * 2.0)


    @torch.no_grad()
    def _get_simplified_features(self, params: dict, state: dict, subset_mask: Tensor) -> Tensor:
        n_subset = subset_mask.sum()
        device = params["means"].device
        features = torch.zeros(n_subset, self.feature_dim, device=device)

        grads2d = state["grad2d"][subset_mask]
        counts = state["count"][subset_mask].clamp_min(1)
        features[:, 0] = (grads2d / counts) / self.grow_grad2d
        features[:, 1] = torch.log(params["scales"][subset_mask].max(dim=-1).values / state["scene_scale"])
        features[:, 2] = params["opacities"][subset_mask].flatten()
        features[:, 3] = state["age"][subset_mask] / 1000.0

        if n_subset > 5:
            means_subset = params["means"][subset_mask]
            dists, _ = torch.cdist(means_subset, means_subset).kthvalue(2, dim=1)
            features[:, 4] = dists / state["scene_scale"]

        return torch.nan_to_num(features, 0.0)


    @torch.no_grad()
    def _update_geometry(self, params: dict, optimizers: dict, state: dict, step: int) -> None:
        device = params["means"].device
        normalized_grads = state["grad2d"] / state["count"].clamp_min(1.0)
        candidate_mask = normalized_grads > self.grow_grad2d

        if candidate_mask.sum() == 0: return

        if candidate_mask.sum() > self.max_densification_subset:
            candidate_indices = torch.where(candidate_mask)[0]
            rand_indices = torch.randperm(len(candidate_indices), device=device)[:self.max_densification_subset]
            candidate_mask.fill_(False)
            candidate_mask[candidate_indices[rand_indices]] = True

        original_indices = torch.where(candidate_mask)[0]

        features = self._get_simplified_features(params, state, candidate_mask)
        self.q_net.eval()
        predicted_q_values = self.q_net(features)

        use_random_action = torch.rand(1).item() < self.exploration_epsilon
        if use_random_action:
            actions = torch.randint(0, self.action_dim, (predicted_q_values.size(0),), device=device)
        else:
            actions = torch.argmax(predicted_q_values, dim=1)

        h, w = state["l1_loss_map"].shape
        means_h = F.pad(params["means"][original_indices], (0, 1), value=1.0)
        p_hom = means_h @ state["view_proj_matrix"]
        p_w = 1.0 / (p_hom[:, 3].clamp_min(1e-6))
        p_proj = p_hom[:, :2] * p_w[:, None]

        pixel_x = torch.clamp(((p_proj[:, 0] * 0.5 + 0.5) * w), 0, w - 1).long()
        pixel_y = torch.clamp(((p_proj[:, 1] * 0.5 + 0.5) * h), 0, h - 1).long()

        for i in range(len(original_indices)):
            initial_error = state["l1_loss_map"][pixel_y[i], pixel_x[i]]
            state["reward_queue"].append({
                "step": step, "features": features[i].detach(), "action": actions[i].detach(),
                "pixel_x": pixel_x[i], "pixel_y": pixel_y[i], "initial_error": initial_error,
            })

        state_to_modify = {k: v for k, v in state.items() if k in ["grad2d", "count", "radii", "age"]}

        prune_agent_mask = (actions == 2)
        prune_opacity_mask = torch.sigmoid(params["opacities"][original_indices].flatten()) < self.prune_opacity_threshold
        prune_mask_subset = prune_agent_mask & prune_opacity_mask

        global_prune_mask = torch.zeros(params["means"].shape[0], dtype=torch.bool, device=device)
        global_prune_mask[original_indices[prune_mask_subset]] = True
        if global_prune_mask.sum() > 0:
            remove(params, optimizers, state_to_modify, global_prune_mask)

        split_mask_subset = (actions == 1)
        prune_map = torch.full((params["means"].shape[0] + global_prune_mask.sum().item(),), -1, dtype=torch.long, device=device)
        prune_map[~global_prune_mask] = torch.arange(params["means"].shape[0], device=device)

        remapped_split_indices = prune_map[original_indices[split_mask_subset]]
        valid_split_mask = remapped_split_indices != -1

        global_split_mask = torch.zeros(params["means"].shape[0], dtype=torch.bool, device=device)
        global_split_mask[remapped_split_indices[valid_split_mask]] = True

        if global_split_mask.sum() > 0:
            n_split = global_split_mask.sum().item()
            split(params, optimizers, state_to_modify, global_split_mask)
            state_to_modify["age"][-n_split:] = 0


    def _process_rewards(self, state: dict, current_step: int):
        while state["reward_queue"] and (current_step - state["reward_queue"][0]["step"]) >= self.reward_delay:
            exp = state["reward_queue"].popleft()
            if state.get("l1_loss_map") is None: continue

            new_error = state["l1_loss_map"][exp["pixel_y"], exp["pixel_x"]]
            reward = exp["initial_error"] - new_error

            if len(state["replay_buffer"]) < state["replay_buffer"]._storage.max_size:
                experience_td = TensorDict({
                    "features": exp["features"], "action": exp["action"],
                    "reward": torch.as_tensor(reward).clamp(-1.0, 1.0),
                }, batch_size=[])
                state["replay_buffer"].add(experience_td)

    def _train_agent(self, state: dict):
        if len(state["replay_buffer"]) < 256: return

        self.q_net.train()
        device = next(self.q_net.parameters()).device

        batch = state["replay_buffer"].sample().to(device)
        features = batch.get("features")
        actions = batch.get("action")
        rewards = batch.get("reward")

        predicted_q_values = self.q_net(features)

        q_value_for_action_taken = predicted_q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        loss = F.mse_loss(q_value_for_action_taken, rewards)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        if self.verbose:
            print(f"Agent trained at step {state.get('step', -1)}: Reward Prediction Loss (MSE) = {loss.item():.4f}")

        self.writer.add_scalar("train/loss", loss.item(), state["step"])
