"""Embodied rollout engine — Remote that returns RolloutResp.

This is the embodied-RL counterpart of ``TrainsideRolloutEngine``. It runs
multi-step episodic interaction (env ↔ policy) and packages the result as
a standard ``RolloutResp`` with ``RolloutTrack`` + ``EmbodiedSegment``.

This makes it pluggable into the same ``train_step`` pattern as diffusion:
    resp = self.rollout.generate(...)
    track.compute_advantages()
    stack.train_track(track)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
from omegaconf import DictConfig

from unirl.distributed.group.remote import Remote, distributed
from unirl.embodied.envs.base import BaseEmbodiedEnv
from unirl.embodied.models.base import BasePolicy
from unirl.embodied.types import EmbodiedSegment
from unirl.types.rollout_resp import RolloutResp, RolloutTrack

logger = logging.getLogger(__name__)


class EmbodiedRolloutEngine(Remote):
    """Multi-step episodic rollout that returns RolloutResp.

    Conforms to the same ``generate()`` → ``RolloutResp`` contract as other
    UniRL rollout engines (trainside, sglang, vllm). TrainStack consumes
    the returned track without knowing it came from episodic interaction.
    """

    def __init__(
        self,
        *,
        env_cfg: DictConfig,
        policy: BasePolicy,
        max_episode_steps: int = 256,
        group_size: int = 4,
        num_groups: int = 8,
        dataset_size: int = 1000,
    ):
        super().__init__()
        self._env_cfg = env_cfg
        self._policy = policy
        self._max_episode_steps = max_episode_steps
        self._group_size = group_size
        self._num_groups = num_groups
        self._dataset_size = dataset_size
        self._env: Optional[BaseEmbodiedEnv] = None

    def _ensure_env(self):
        if self._env is None:
            from hydra.utils import instantiate

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._env = instantiate(self._env_cfg, device=device)

    @distributed()
    def generate(self, *, rollout_id: int = 0) -> RolloutResp:
        """Collect episodes and return a standard RolloutResp."""
        self._ensure_env()
        env = self._env
        policy = self._policy
        device = env.device

        # Sample initial states (repeated for group_size siblings)
        episode_indices = torch.randint(0, self._dataset_size, (self._num_groups,))
        if self._group_size > 1:
            episode_indices = episode_indices.repeat_interleave(self._group_size)
        B = episode_indices.shape[0]

        n_chunk_steps = self._max_episode_steps // env.chunk_size

        # Reset env
        obs, _ = env.reset(episode_indices=episode_indices)

        # Collect trajectory
        all_actions = []
        all_logprobs = []
        all_rewards = []
        all_dones = []
        all_obs = [obs]
        active = torch.ones(B, dtype=torch.bool, device=device)

        for t in range(n_chunk_steps):
            with torch.no_grad():
                result = policy.predict_action_batch(**obs)

            actions = result["actions"]  # [B, chunk, action_dim]
            logprobs = result["logprobs"]  # [B, chunk, action_dim]

            next_obs, rewards, terminations, truncations, infos = env.chunk_step(actions)

            step_dones = terminations.any(dim=-1) | truncations.any(dim=-1)

            all_actions.append(actions)
            all_logprobs.append(logprobs)
            all_rewards.append(rewards)
            all_dones.append(step_dones)

            active = active & ~step_dones
            if not active.any():
                break

            obs = next_obs
            all_obs.append(obs)

        # Build tensors [T, B, ...]
        T = len(all_actions)
        actions_t = torch.stack(all_actions, dim=0)  # [T, B, chunk, action_dim]
        logprobs_t = torch.stack(all_logprobs, dim=0)  # [T, B, chunk, action_dim]
        rewards_t = torch.stack(all_rewards, dim=0)  # [T, B, chunk]
        dones_t = torch.stack(all_dones, dim=0)  # [T, B]

        # Loss mask: 1 for valid steps
        loss_mask = torch.ones(T, B, device=device)
        for t in range(1, T):
            loss_mask[t] = loss_mask[t - 1] * (~dones_t[t - 1]).float()

        # Episode-level rewards for advantage computation
        episode_rewards = (rewards_t.sum(dim=-1) * loss_mask).sum(dim=0)  # [B]

        # Build EmbodiedSegment
        segment = EmbodiedSegment(
            action_log_probs=logprobs_t,
            actions=actions_t,
            observations={"steps": all_obs},
            loss_mask=loss_mask,
        )

        # Build RolloutTrack (same structure as diffusion trainer produces)
        sample_ids = [f"ep_{i}" for i in range(B)]
        group_ids = [f"g_{i // self._group_size}" for i in range(B)]

        track = RolloutTrack(
            sample_ids=sample_ids,
            parent_ids=group_ids,
            parent_track=None,
            conditions={},
            segment=segment,
            decoded=None,
            media_preview=None,
            rewards=episode_rewards,
            advantages=None,
        )

        return RolloutResp(tracks={"embodied": track})

    @distributed()
    def wake_up(self) -> None:
        """No-op; embodied env is always ready."""

    @distributed()
    def sleep(self) -> None:
        """Optionally offload env models."""
        if self._env is not None:
            self._env.offload()
