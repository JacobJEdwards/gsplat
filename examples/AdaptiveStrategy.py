import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Categorical

from utils import knn_with_ids
from gsplat.strategy.default import DefaultStrategy
from gsplat.strategy.ops import remove, split, reset_opa
from tensordict import TensorDict
from torchrl.data import RandomSampler, TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyMemmapStorage

def create_view_proj_matrix(camtoworld, K, width, height, near=0.1, far=100.0):
    device = camtoworld.device

    view_matrix = torch.inverse(camtoworld)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    proj_matrix = torch.zeros(4, 4, device=device)

    proj_matrix[0, 0] = 2.0 * fx / width
    proj_matrix[1, 1] = 2.0 * fy / height
    proj_matrix[0, 2] = (2.0 * cx - width) / width
    proj_matrix[1, 2] = (2.0 * cy - height) / height
    proj_matrix[2, 2] = -(far + near) / (far - near)
    proj_matrix[2, 3] = -2.0 * far * near / (far - near)
    proj_matrix[3, 2] = -1.0

    view_proj_matrix = proj_matrix @ view_matrix

    return view_proj_matrix, view_matrix, proj_matrix

def project_points(points_3d, view_proj_matrix, width, height):
    points_h = F.pad(points_3d, (0, 1), value=1.0)  # [N, 4]

    projected = points_h @ view_proj_matrix.T  # [N, 4]

    w = projected[:, 3]
    valid_mask = w > 1e-6

    w_safe = torch.where(valid_mask, w, torch.ones_like(w))
    ndc_coords = projected[:, :3] / w_safe.unsqueeze(-1)  # [N, 3]

    screen_x = (ndc_coords[:, 0] + 1.0) * 0.5 * width
    screen_y = (ndc_coords[:, 1] + 1.0) * 0.5 * height

    valid_mask = valid_mask & (ndc_coords[:, 0].abs() <= 1.0) & (ndc_coords[:, 1].abs() <= 1.0)

    screen_coords = torch.stack([screen_x, screen_y], dim=-1)

    return screen_coords, valid_mask

class SimpleActorCritic(nn.Module):
    """A simple MLP-based Actor-Critic network."""
    def __init__(self, feature_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
        )
        self.actor_head = nn.Linear(hidden_dim, action_dim)  # Outputs logits for actions
        self.critic_head = nn.Linear(hidden_dim, 1)        # Outputs a single value

    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass returns action logits and state value."""
        x = self.shared_net(features)
        action_logits = self.actor_head(x)
        value = self.critic_head(x).squeeze(-1)
        return action_logits, value

# ------------------------------------------------------------------------------------
# 2. Main Strategy Class
# - Radically simplified hyperparameters and state management.
# - Implements a clear, back-to-basics reinforcement learning loop.
# ------------------------------------------------------------------------------------
@dataclass
class AdaptiveStrategy(DefaultStrategy):
    """
    A simplified reinforcement learning strategy for Gaussian densification.
    It uses a basic Actor-Critic model to decide whether to split, prune, or
    do nothing to a Gaussian based on simple, direct features.
    """
    # --- Core Scheduling ---
    refine_every: int = 400  # How often to run the densification logic.
    learn_every: int = 200   # How often to train the agent.

    # --- Learning Hyperparameters ---
    feature_dim: int = 5
    hidden_dim: int = 64
    action_dim: int = 3  # 0: No-op, 1: Split, 2: Prune
    learning_rate: float = 3e-4
    ppo_clip_epsilon: float = 0.2
    entropy_loss_weight: float = 0.01
    reward_delay: int = 200 # Steps to wait before calculating reward.
    max_densification_subset: int = 100_000 # Max Gaussians to consider per step.

    # --- Action Thresholds ---
    # Prune gaussians with opacity below this threshold, if the agent decides to.
    prune_opacity_threshold: float = 0.005
    # When splitting, new opacities are reset to this value.
    split_opacity_reset: float = 0.01

    # --- Internal State (fields are managed automatically) ---
    ac_net: Any = field(default=None, repr=False)
    ac_optimizer: Any = field(default=None, repr=False)
    writer: Any = field(default=None, repr=False) # Optional: for Tensorboard logging

    def initialize_state(self, scene_scale: float = 1.0) -> dict[str, Any]:
        """Initializes a much simpler state than the original implementation."""
        state = super().initialize_state(scene_scale)
        # A buffer to store experiences (state, action, reward) for training.
        replay_buffer = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(max_size=50_000),
            sampler=RandomSampler(),
            batch_size=256
        )
        # A temporary queue to hold actions until their outcomes (rewards) are known.
        reward_queue = deque(maxlen=10_000)

        state.update({
            "age": None,
            "l1_loss_map": None,
            "view_proj_matrix": None, # Used to project Gaussians into screen space
            "replay_buffer": replay_buffer,
            "reward_queue": reward_queue,
        })
        return state

    def _initialize_learning_components(self, device: torch.device) -> None:
        """Initializes the simplified network and its optimizer."""
        self.ac_net = SimpleActorCritic(self.feature_dim, self.hidden_dim, self.action_dim).to(device)
        self.ac_optimizer = torch.optim.AdamW(self.ac_net.parameters(), lr=self.learning_rate)
        print("Initialized Simple Actor-Critic network.")

    def step_post_backward(
            self,
            params: dict[str, torch.nn.Parameter] | torch.nn.ParameterDict,
            optimizers: dict[str, torch.optim.Optimizer],
            state: dict[str, Any],
            step: int,
            info: dict[str, Any],
            packed: bool = False,
    ) -> None:
        """The main hook that runs after each backward pass."""
        if step >= self.refine_stop_iter:
            return

        state["step"] = step

        # Basic setup on the first run
        if self.ac_net is None:
            self._initialize_learning_components(params["means"].device)
        if state.get("age") is None:
            state["age"] = torch.zeros(params["means"].shape[0], dtype=torch.int32, device=params["means"].device)

        state["age"] += 1
        state["l1_loss_map"] = info.get("l1_loss_map") # Keep track of the latest loss map
        self._update_quality_map(params, state, info)

        self._update_state(params, state, info, packed=packed)
        # Process the reward queue to see if any actions are old enough to have a reward calculated.
        self._process_rewards(state, step)

        # Decide and apply actions
        if step > self.refine_start_iter and step % self.refine_every == 0:
            self._update_geometry(params, optimizers, state, step)

        # Train the agent
        if step > self.refine_start_iter and step % self.learn_every == 0:
            self._train_agent(state)

        if step % self.reset_every == 0 and step > 0:
            reset_opa(params, optimizers, state, self.prune_opa * 2.0)


    @torch.no_grad()
    def _get_simplified_features(self, params: dict, state: dict, subset_mask: Tensor) -> Tensor:
        """Extracts a minimal set of features for the actor-critic network."""
        n_subset = subset_mask.sum()
        device = params["means"].device
        features = torch.zeros(n_subset, self.feature_dim, device=device)

        # Feature 0: Normalized 2D gradient magnitude
        grads2d = state["grad2d"][subset_mask]
        counts = state["count"][subset_mask].clamp_min(1)
        features[:, 0] = (grads2d / counts) / self.grow_grad2d

        # Feature 1: Max scale (log-space)
        scales = params["scales"][subset_mask]
        features[:, 1] = torch.log(scales.max(dim=-1).values / state["scene_scale"])

        # Feature 2: Opacity (logit-space)
        features[:, 2] = params["opacities"][subset_mask].flatten()

        # Feature 3: Age
        features[:, 3] = state["age"][subset_mask] / 1000.0 # Normalize age

        # Feature 4: Normalized distance to nearest neighbor
        if n_subset > 5:
            dists, _ = knn_with_ids(subset_mask, K=5 + 1)
            features[:, 4] = dists / state["scene_scale"]

        return torch.nan_to_num(features, 0.0)


    @torch.no_grad()
    def _update_geometry(self, params: dict, optimizers: dict, state: dict, step: int) -> None:
        """Selects Gaussians, gets actions from the agent, and applies them."""
        device = params["means"].device

        # 1. Select candidate Gaussians (e.g., those with high gradients).
        normalized_grads = state["grad2d"] / state["count"].clamp_min(1.0)
        candidate_mask = normalized_grads > self.grow_grad2d

        if candidate_mask.sum() == 0:
            return

        # Limit to a random subset if too many candidates
        if candidate_mask.sum() > self.max_densification_subset:
            candidate_indices = torch.where(candidate_mask)[0]
            rand_indices = torch.randperm(len(candidate_indices), device=device)[:self.max_densification_subset]
            candidate_mask.fill_(False)
            candidate_mask[candidate_indices[rand_indices]] = True

        original_indices = torch.where(candidate_mask)[0]

        # 2. Get features and actions from the network.
        features = self._get_simplified_features(params, state, candidate_mask)
        self.ac_net.eval()
        action_logits, _ = self.ac_net(features)
        action_dist = Categorical(logits=action_logits)
        actions = action_dist.sample()
        log_probs = action_dist.log_prob(actions)

        # 3. Project Gaussians to find their location on the L1 loss map.
        h, w = state["l1_loss_map"].shape
        means_h = F.pad(params["means"][original_indices], (0, 1), value=1.0)
        p_hom = means_h @ state["view_proj_matrix"]
        p_w = 1.0 / (p_hom[:, 3].clamp_min(1e-6))
        p_proj = p_hom[:, :2] * p_w[:, None]

        pixel_x = torch.clamp(((p_proj[:, 0] * 0.5 + 0.5) * w), 0, w - 1).long()
        pixel_y = torch.clamp(((p_proj[:, 1] * 0.5 + 0.5) * h), 0, h - 1).long()

        # 4. Queue up the experiences to calculate reward later.
        for i in range(len(original_indices)):
            initial_error = state["l1_loss_map"][pixel_y[i], pixel_x[i]]
            experience = {
                "step": step,
                "features": features[i].detach(),
                "action": actions[i].detach(),
                "log_prob": log_probs[i].detach(),
                "pixel_x": pixel_x[i],
                "pixel_y": pixel_y[i],
                "initial_error": initial_error,
            }
            state["reward_queue"].append(experience)

        # 5. Apply the actions.
        state_to_modify = {k: v for k, v in state.items() if k in ["grad2d", "count", "radii", "age"]}

        # Apply Pruning
        prune_agent_mask = (actions == 2)
        prune_opacity_mask = torch.sigmoid(params["opacities"][original_indices].flatten()) < self.prune_opacity_threshold
        prune_mask_subset = prune_agent_mask & prune_opacity_mask

        global_prune_mask = torch.zeros(params["means"].shape[0], dtype=torch.bool, device=device)
        global_prune_mask[original_indices[prune_mask_subset]] = True
        if global_prune_mask.sum() > 0:
            remove(params, optimizers, state_to_modify, global_prune_mask)

        # Apply Splitting
        split_mask_subset = (actions == 1)

        # Need to map indices after pruning
        prune_map = torch.full((params["means"].shape[0] + global_prune_mask.sum().item(),), -1, dtype=torch.long, device=device)
        prune_map[~global_prune_mask] = torch.arange(params["means"].shape[0], device=device)

        remapped_split_indices = prune_map[original_indices[split_mask_subset]]
        valid_split_mask = remapped_split_indices != -1

        global_split_mask = torch.zeros(params["means"].shape[0], dtype=torch.bool, device=device)
        global_split_mask[remapped_split_indices[valid_split_mask]] = True

        if global_split_mask.sum() > 0:
            n_split = global_split_mask.sum().item()
            split(params, optimizers, state_to_modify, global_split_mask)
            # Reset age for new Gaussians
            state_to_modify["age"][-n_split:] = 0

        state.update(state_to_modify)


    def _process_rewards(self, state: dict, current_step: int):
        """Processes the reward queue for experiences that have matured."""
        # Check from the oldest experience
        while state["reward_queue"] and (current_step - state["reward_queue"][0]["step"]) >= self.reward_delay:
            exp = state["reward_queue"].popleft()

            # If the loss map isn't available, we can't calculate the reward.
            if state.get("l1_loss_map") is None:
                continue

            # Calculate reward: (initial_error - new_error). Positive if error decreased.
            new_error = state["l1_loss_map"][exp["pixel_y"], exp["pixel_x"]]
            reward = exp["initial_error"] - new_error

            # Add the completed experience to the main replay buffer for training.
            if len(state["replay_buffer"]) < state["replay_buffer"]._storage.max_size:
                experience_td = TensorDict({
                    "features": exp["features"],
                    "action": exp["action"],
                    "log_prob": exp["log_prob"],
                    "reward": torch.as_tensor(reward).clamp(-1.0, 1.0), # Clamp reward for stability
                }, batch_size=[])
                state["replay_buffer"].add(experience_td)

    def _train_agent(self, state: dict):
        """Trains the Actor-Critic agent using PPO."""
        if len(state["replay_buffer"]) < 256:
            return # Don't train until we have a minimum number of samples

        self.ac_net.train()
        device = next(self.ac_net.parameters()).device

        # Sample a batch of experiences
        batch = state["replay_buffer"].sample().to(device)
        features = batch.get("features")
        actions = batch.get("action")
        rewards = batch.get("reward")
        old_log_probs = batch.get("log_prob")

        # Get current predictions from the network
        new_logits, values = self.ac_net(features)

        # Calculate Advantage (how much better the reward was than expected)
        advantage = rewards - values.detach()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8) # Normalize advantage

        # Calculate PPO surrogate loss for the actor
        new_dist = Categorical(logits=new_logits)
        new_log_probs = new_dist.log_prob(actions)
        ratio = torch.exp(new_log_probs - old_log_probs)

        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip_epsilon, 1.0 + self.ppo_clip_epsilon) * advantage
        actor_loss = -torch.min(surr1, surr2).mean()

        # Calculate critic loss (MSE between predicted value and actual reward)
        critic_loss = F.mse_loss(values, rewards)

        # Calculate entropy loss to encourage exploration
        entropy_loss = -new_dist.entropy().mean()

        # Total loss
        loss = actor_loss + 0.5 * critic_loss + self.entropy_loss_weight * entropy_loss

        # Optimization step
        self.ac_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ac_net.parameters(), 1.0)
        self.ac_optimizer.step()

        if self.verbose:
            print(f"Agent trained at step {state.get('step', -1)}: Total Loss = {loss.item():.4f}")

        self.writer.add_scalar("agent/loss", loss.item(), state.get("step", -1))

    @torch.no_grad()
    def _update_quality_map(
            self,
            params: dict[str, torch.nn.Parameter],
            state: dict[str, Any],
            info: dict[str, Any],
    ):
        width, height = info['width'], info['height']

        main_camtoworld = info["camtoworlds"][0]
        main_K = info["Ks"][0]

        view_proj_matrix, view_matrix, proj_matrix = create_view_proj_matrix(
            main_camtoworld, main_K, width, height
        )

        state["view_matrix"] = view_matrix
        state["view_proj_matrix"] = view_proj_matrix
        state["proj_matrix"] = proj_matrix
