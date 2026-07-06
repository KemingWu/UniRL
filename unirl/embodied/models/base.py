"""Base class for VLA / policy models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch
import torch.nn as nn


class BasePolicy(nn.Module, ABC):
    """Abstract base for Vision-Language-Action policy models.

    Supports both rollout (action prediction with exploration) and training
    (teacher-forced log-prob computation for PPO ratio).
    """

    @abstractmethod
    def predict_action_batch(self, **obs) -> Dict[str, torch.Tensor]:
        """Generate actions from observations (rollout time).

        Args:
            **obs: observation dict from environment
                - main_images: [B, H, W, C] uint8
                - states: Optional[B, D]
                - task_descriptions: Optional[List[str]]

        Returns:
            dict with keys:
                - actions: [B, chunk, action_dim]
                - logprobs: [B, chunk, action_dim] or [B, chunk]
                - values: Optional[B, 1] (if critic head exists)
                - forward_inputs: dict of cached tensors for replay
        """
        ...

    @abstractmethod
    def default_forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Teacher-forced forward pass (training time).

        Args:
            **kwargs: must include observations and ground-truth actions
                - obs: observation dict
                - actions: [B, chunk, action_dim] ground truth actions

        Returns:
            dict with keys:
                - logprobs: [B, chunk, action_dim] or [B, chunk]
                - entropy: Optional[B]
        """
        ...

    @property
    @abstractmethod
    def num_action_chunks(self) -> int:
        """Number of action timesteps predicted per forward pass."""
        ...

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Dimensionality of action space."""
        ...
