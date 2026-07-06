"""Embodied rollout engine — Remote that returns RolloutResp.

Collects multi-step episodic trajectories and caches forward_inputs
for PPO replay during training. Returns standard RolloutResp.
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
    """Multi-step episodic rollout engine.

    Collects trajectories by running policy-environment interaction loops.
    Caches forward_inputs (input_ids, attention_mask, pixel_values, action_tokens)
    at each step for PPO replay during training.

    Returns standard RolloutResp so TrainStack can process it.
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
        """Collect episodes and return RolloutResp with cached forward_inputs."""
        self._ensure_env()
        env = self._env
        policy = self._policy
        device = env.device

        # Sample and expand indices for group rollouts
        episode_indices = torch.randint(0, self._dataset_size, (self._num_groups,))
        if self._group_size > 1:
            episode_indices = episode_indices.repeat_interleave(self._group_size)
        B = episode_indices.shape[0]

        n_chunk_steps = self._max_episode_steps // env.chunk_size

        obs, _ = env.reset(episode_indices=episode_indices)

        # Per-step storage
        all_logprobs = []  # [T] of [B, response_len]
        all_forward_inputs = []  # [T] of dicts
        all_rewards = []  # [T] of [B, chunk]
        all_dones = []  # [T] of [B]
        active = torch.ones(B, dtype=torch.bool, device=device)

        for t in range(n_chunk_steps):
            with torch.no_grad():
                result = policy.predict_action_batch(**obs)

            actions = result["actions"]  # [B, num_chunks, action_dim]
            logprobs = result["logprobs"]  # [B, response_len] (action_dim * num_chunks)
            forward_inputs = result["forward_inputs"]  # cached for replay

            next_obs, rewards, terminations, truncations, infos = env.chunk_step(actions)
            step_dones = terminations.any(dim=-1) | truncations.any(dim=-1)

            all_logprobs.append(logprobs)
            all_forward_inputs.append(forward_inputs)
            all_rewards.append(rewards)
            all_dones.append(step_dones)

            active = active & ~step_dones
            if not active.any():
                break

            obs = next_obs

        # Assemble tensors
        T = len(all_logprobs)
        logprobs_t = torch.stack(all_logprobs, dim=0)  # [T, B, response_len]
        rewards_t = torch.stack(all_rewards, dim=0)  # [T, B, chunk]
        dones_t = torch.stack(all_dones, dim=0)  # [T, B]

        # Loss mask: valid steps (before episode ends)
        loss_mask = torch.ones(T, B, device=device)
        for t in range(1, T):
            loss_mask[t] = loss_mask[t - 1] * (~dones_t[t - 1]).float()

        # Episode-level reward for GRPO advantage
        episode_rewards = (rewards_t.sum(dim=-1) * loss_mask).sum(dim=0)  # [B]

        # Concatenate forward_inputs across timesteps for batch replay
        # Shape: [T*B, ...] — the training side replays all steps in one forward
        concat_forward_inputs = self._concat_forward_inputs(all_forward_inputs, loss_mask)

        # Build segment
        segment = EmbodiedSegment(
            action_log_probs=logprobs_t,  # [T, B, response_len]
            actions=None,
            observations=None,
            loss_mask=loss_mask,  # [T, B]
            forward_inputs=concat_forward_inputs,
        )

        # Build track
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

    def _concat_forward_inputs(
        self, all_forward_inputs: List[Dict[str, torch.Tensor]], loss_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Concatenate per-step forward_inputs into a flat batch for training replay.

        Only includes valid steps (loss_mask == 1).
        Returns dict with tensors of shape [N_valid, ...].
        """
        T, B = loss_mask.shape
        valid_mask = loss_mask.bool()  # [T, B]

        result = {}
        keys = all_forward_inputs[0].keys() if all_forward_inputs else []

        for key in keys:
            tensors = []
            for t in range(len(all_forward_inputs)):
                step_tensor = all_forward_inputs[t][key]  # [B, ...]
                # Select only valid envs at this step
                valid_envs = valid_mask[t]
                if valid_envs.any():
                    tensors.append(step_tensor[valid_envs])
            if tensors:
                result[key] = torch.cat(tensors, dim=0)

        # Also store the mask shape for reconstruction
        result["_loss_mask"] = loss_mask
        result["_T"] = torch.tensor(T)
        result["_B"] = torch.tensor(B)

        return result

    @distributed()
    def wake_up(self) -> None:
        pass

    @distributed()
    def sleep(self) -> None:
        if self._env is not None:
            self._env.offload()
