"""Episodic rollout collection engine for embodied RL."""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
from omegaconf import DictConfig

from unirl.embodied.envs.base import BaseEmbodiedEnv
from unirl.embodied.models.base import BasePolicy
from unirl.embodied.types import EnvOutput, RolloutStepResult, Trajectory

logger = logging.getLogger(__name__)


class EmbodiedRolloutEngine:
    """Multi-step episodic rollout collection.

    Runs the policy-environment interaction loop:
        obs → model.predict → env.chunk_step → store → repeat

    Supports grouping: multiple rollouts from the same initial state
    for GRPO-style group-relative advantages.
    """

    def __init__(
        self,
        env: BaseEmbodiedEnv,
        model: BasePolicy,
        cfg: DictConfig,
    ):
        self.env = env
        self.model = model
        self.n_chunk_steps = int(cfg.get("max_episode_steps", 256)) // env.chunk_size
        self.group_size = int(cfg.get("group_size", 1))

    @torch.no_grad()
    def collect_trajectory(
        self,
        episode_indices: torch.Tensor,
        seed: Optional[int] = None,
    ) -> Trajectory:
        """Collect full episodes and return a Trajectory for training.

        Args:
            episode_indices: [num_groups] indices into the episode dataset.
                Each index is repeated group_size times for grouped rollouts.
            seed: optional RNG seed for environment reset.

        Returns:
            Trajectory with shape [T, B, ...] where B = num_groups * group_size.
        """
        # Expand indices for group rollouts
        if self.group_size > 1:
            expanded = episode_indices.repeat_interleave(self.group_size)
        else:
            expanded = episode_indices

        B = expanded.shape[0]
        device = self.env.device

        obs, _ = self.env.reset(episode_indices=expanded, seed=seed)

        steps: List[tuple] = []
        active = torch.ones(B, dtype=torch.bool, device=device)

        for step_idx in range(self.n_chunk_steps):
            # Get actions from policy
            rollout_result = self._policy_forward(obs, active)

            # Step environment
            next_obs, rewards, terminations, truncations, infos = self.env.chunk_step(rollout_result.actions)

            env_output = EnvOutput(
                obs=next_obs,
                rewards=rewards,
                terminations=terminations,
                truncations=truncations,
                dones=terminations.any(dim=-1) | truncations.any(dim=-1),
                infos=infos,
            )

            steps.append((rollout_result, env_output))

            # Update active mask
            active = active & ~env_output.dones
            if not active.any():
                break

            obs = next_obs

        trajectory = Trajectory.from_steps(
            steps=steps,
            max_steps=self.n_chunk_steps,
            num_envs=B,
            chunk_size=self.env.chunk_size,
            action_dim=self.env.action_dim,
            device=device,
        )
        return trajectory

    def _policy_forward(self, obs: dict, active: torch.Tensor) -> RolloutStepResult:
        """Run policy on active environments."""
        result = self.model.predict_action_batch(**obs)
        return RolloutStepResult(
            actions=result["actions"],
            logprobs=result["logprobs"],
            values=result.get("values"),
            forward_inputs=result.get("forward_inputs"),
        )
