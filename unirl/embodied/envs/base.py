"""Base class for embodied environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch
from omegaconf import DictConfig


class BaseEmbodiedEnv(ABC):
    """Abstract base for embodied environments.

    Covers world-model environments (e.g. Wan video model as simulator),
    physics simulators (ManiSkill, IsaacLab, LIBERO), and real-world setups.

    Observation dict contract::

        {
            "main_images": Tensor[B, H, W, C] uint8,
            "wrist_images": Optional[Tensor],
            "states": Optional[Tensor[B, state_dim]] float32,
            "task_descriptions": Optional[List[str]],
        }
    """

    def __init__(self, cfg: DictConfig, num_envs: int, device: torch.device, **kwargs):
        self._cfg = cfg
        self._num_envs = num_envs
        self._device = device

    @abstractmethod
    def reset(
        self,
        *,
        episode_indices: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environments.

        Args:
            episode_indices: [B] dataset indices to initialize from.
            seed: optional RNG seed.

        Returns:
            (obs_dict, infos_dict)
        """
        ...

    @abstractmethod
    def chunk_step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Execute one chunk of actions.

        Args:
            actions: [B, chunk_size, action_dim]

        Returns:
            (obs, rewards[B, chunk], terminations[B, chunk], truncations[B, chunk], infos)
        """
        ...

    def offload(self) -> None:
        """Move heavy models to CPU. Override in world-model envs."""

    def onload(self) -> None:
        """Move models back to GPU. Override in world-model envs."""

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    @abstractmethod
    def chunk_size(self) -> int:
        """Number of action timesteps per chunk."""
        ...

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Dimensionality of the action space."""
        ...

    @property
    def max_episode_steps(self) -> int:
        return int(self._cfg.get("max_episode_steps", 256))
