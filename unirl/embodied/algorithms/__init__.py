"""RL algorithm abstractions for embodied training."""

from unirl.embodied.algorithms.base import BaseAlgorithm
from unirl.embodied.algorithms.advantages import compute_gae_advantages, compute_grpo_advantages
from unirl.embodied.algorithms.losses import compute_ppo_loss

__all__ = [
    "BaseAlgorithm",
    "compute_gae_advantages",
    "compute_grpo_advantages",
    "compute_ppo_loss",
]
