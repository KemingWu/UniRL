"""Base algorithm interface for embodied RL."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch

from unirl.embodied.types import TrainBatch, Trajectory


class BaseAlgorithm(ABC):
    """Abstract base for RL algorithms.

    Encapsulates advantage computation and policy loss. Implementations include
    GRPO (group-relative), PPO (GAE-based), SAC, etc.
    """

    @abstractmethod
    def compute_advantages(self, trajectory: Trajectory) -> torch.Tensor:
        """Compute per-episode or per-step advantage estimates.

        Args:
            trajectory: collected rollout data [T, B, ...]

        Returns:
            advantages: [B] (episode-level) or [T, B] (step-level)
        """
        ...

    @abstractmethod
    def compute_loss(self, batch: TrainBatch) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute policy loss for one mini-batch.

        Args:
            batch: training mini-batch with observations, actions, old_logprobs, advantages

        Returns:
            (loss, metrics_dict)
        """
        ...


class GRPOAlgorithm(BaseAlgorithm):
    """Group Relative Policy Optimization for embodied RL.

    Groups episodes by initial state and normalizes rewards within each group.
    """

    def __init__(
        self,
        group_size: int = 4,
        clip_ratio_low: float = 0.2,
        clip_ratio_high: float = 0.28,
        reward_aggregation: str = "sum",
    ):
        self.group_size = group_size
        self.clip_ratio_low = clip_ratio_low
        self.clip_ratio_high = clip_ratio_high
        self.reward_aggregation = reward_aggregation

    def compute_advantages(self, trajectory: Trajectory) -> torch.Tensor:
        from unirl.embodied.algorithms.advantages import compute_grpo_advantages

        # Aggregate per-step rewards to episode-level
        mask = trajectory.loss_mask if trajectory.loss_mask is not None else torch.ones_like(trajectory.rewards[:, :, 0])
        if self.reward_aggregation == "sum":
            episode_rewards = (trajectory.rewards.sum(dim=-1) * mask).sum(dim=0)  # [B]
        elif self.reward_aggregation == "final":
            T = int(mask.sum(dim=0).max().item())
            episode_rewards = trajectory.rewards[T - 1, :, -1]
        else:
            episode_rewards = (trajectory.rewards.sum(dim=-1) * mask).sum(dim=0)

        return compute_grpo_advantages(episode_rewards, self.group_size)

    def compute_loss(self, batch: TrainBatch) -> Tuple[torch.Tensor, Dict[str, float]]:
        from unirl.embodied.algorithms.losses import compute_ppo_loss

        return compute_ppo_loss(
            logprobs=batch.old_logprobs,  # will be replaced with new_logprobs from replay
            old_logprobs=batch.old_logprobs,
            advantages=batch.advantages,
            clip_low=self.clip_ratio_low,
            clip_high=self.clip_ratio_high,
            loss_mask=batch.loss_mask,
        )


class PPOAlgorithm(BaseAlgorithm):
    """Proximal Policy Optimization with GAE advantages."""

    def __init__(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio_low: float = 0.2,
        clip_ratio_high: float = 0.2,
        normalize_advantages: bool = True,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio_low = clip_ratio_low
        self.clip_ratio_high = clip_ratio_high
        self.normalize_advantages = normalize_advantages

    def compute_advantages(self, trajectory: Trajectory) -> torch.Tensor:
        from unirl.embodied.algorithms.advantages import compute_gae_advantages

        rewards = trajectory.rewards.sum(dim=-1)  # [T, B] sum over chunk
        values = trajectory.prev_values.squeeze(-1) if trajectory.prev_values is not None else None
        dones = trajectory.dones
        mask = trajectory.loss_mask

        return compute_gae_advantages(
            rewards=rewards,
            values=values,
            dones=dones,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            normalize=self.normalize_advantages,
            loss_mask=mask,
        )

    def compute_loss(self, batch: TrainBatch) -> Tuple[torch.Tensor, Dict[str, float]]:
        from unirl.embodied.algorithms.losses import compute_ppo_loss

        return compute_ppo_loss(
            logprobs=batch.old_logprobs,
            old_logprobs=batch.old_logprobs,
            advantages=batch.advantages,
            clip_low=self.clip_ratio_low,
            clip_high=self.clip_ratio_high,
            loss_mask=batch.loss_mask,
        )
