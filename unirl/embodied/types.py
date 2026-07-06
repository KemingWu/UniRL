"""Core data structures for embodied RL training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from unirl.types.segments.base import Segment


@dataclass
class EmbodiedSegment(Segment):
    """Segment subclass for embodied RL trajectories.

    Stores action log-probs (π_old anchor) and forward_inputs cache for replay.
    The forward_inputs dict contains: input_ids, attention_mask, pixel_values,
    action_tokens — everything needed to recompute logprobs under updated weights.
    """

    action_log_probs: Optional[Any] = None  # [T, B, response_len] old log-probs
    actions: Optional[Any] = None
    observations: Optional[Any] = None
    loss_mask: Optional[Any] = None  # [T, B]
    task_descriptions: Optional[List[str]] = None
    forward_inputs: Optional[Dict[str, Any]] = None  # cached for PPO replay


@dataclass
class EnvOutput:
    """Single-step environment output.

    Shapes use B = num_envs, chunk = action chunk size.
    """

    obs: Dict[str, Any]
    rewards: Optional[torch.Tensor] = None  # [B] or [B, chunk]
    terminations: Optional[torch.Tensor] = None  # [B] or [B, chunk]
    truncations: Optional[torch.Tensor] = None  # [B] or [B, chunk]
    dones: Optional[torch.Tensor] = None  # [B]
    infos: Optional[Dict[str, Any]] = None

    def all_done(self) -> bool:
        if self.dones is not None:
            return bool(self.dones.all())
        if self.terminations is not None and self.truncations is not None:
            return bool((self.terminations.any(dim=-1) | self.truncations.any(dim=-1)).all())
        return False


@dataclass
class RolloutStepResult:
    """Model output for one chunk step during rollout."""

    actions: torch.Tensor  # [B, chunk, action_dim]
    logprobs: torch.Tensor  # [B, chunk, action_dim] or [B, chunk]
    values: Optional[torch.Tensor] = None  # [B, 1]
    forward_inputs: Optional[Dict[str, torch.Tensor]] = None


@dataclass
class Trajectory:
    """Complete rollout trajectory for training.

    Time-first layout: [T, B, ...] where T = number of chunk steps.
    """

    actions: torch.Tensor  # [T, B, chunk, action_dim]
    rewards: torch.Tensor  # [T, B, chunk]
    terminations: torch.Tensor  # [T, B, chunk]
    truncations: torch.Tensor  # [T, B, chunk]
    dones: torch.Tensor  # [T, B]
    prev_logprobs: torch.Tensor  # [T, B, chunk, action_dim] or [T, B, chunk]
    prev_values: Optional[torch.Tensor] = None  # [T, B, 1]
    observations: Optional[Dict[str, Any]] = None  # per-step obs for replay
    loss_mask: Optional[torch.Tensor] = None  # [T, B]
    task_descriptions: Optional[List[str]] = None

    @classmethod
    def from_steps(
        cls,
        steps: List[tuple],
        max_steps: int,
        num_envs: int,
        chunk_size: int,
        action_dim: int,
        device: torch.device,
    ) -> "Trajectory":
        """Assemble trajectory from collected (RolloutStepResult, EnvOutput) pairs."""
        T = len(steps)
        actions = torch.zeros(max_steps, num_envs, chunk_size, action_dim, device=device)
        rewards = torch.zeros(max_steps, num_envs, chunk_size, device=device)
        terminations = torch.zeros(max_steps, num_envs, chunk_size, dtype=torch.bool, device=device)
        truncations = torch.zeros(max_steps, num_envs, chunk_size, dtype=torch.bool, device=device)
        dones = torch.zeros(max_steps, num_envs, dtype=torch.bool, device=device)
        prev_logprobs = torch.zeros_like(actions)
        prev_values = None
        loss_mask = torch.zeros(max_steps, num_envs, device=device)

        has_values = steps[0][0].values is not None
        if has_values:
            prev_values = torch.zeros(max_steps, num_envs, 1, device=device)

        obs_list = []
        for t, (rollout_result, env_output) in enumerate(steps):
            actions[t] = rollout_result.actions
            prev_logprobs[t] = rollout_result.logprobs
            if has_values and rollout_result.values is not None:
                prev_values[t] = rollout_result.values

            if env_output.rewards is not None:
                r = env_output.rewards
                rewards[t] = r if r.dim() == 2 else r.unsqueeze(-1).expand(-1, chunk_size)
            if env_output.terminations is not None:
                term = env_output.terminations
                terminations[t] = term if term.dim() == 2 else term.unsqueeze(-1)
            if env_output.truncations is not None:
                trunc = env_output.truncations
                truncations[t] = trunc if trunc.dim() == 2 else trunc.unsqueeze(-1)

            step_dones = terminations[t].any(dim=-1) | truncations[t].any(dim=-1)
            dones[t] = step_dones
            loss_mask[t] = 1.0
            obs_list.append(env_output.obs)

        return cls(
            actions=actions,
            rewards=rewards,
            terminations=terminations,
            truncations=truncations,
            dones=dones,
            prev_logprobs=prev_logprobs,
            prev_values=prev_values,
            observations={"steps": obs_list},
            loss_mask=loss_mask,
        )


@dataclass
class TrainBatch:
    """Mini-batch for one training iteration."""

    observations: Dict[str, torch.Tensor]
    actions: torch.Tensor  # [B, chunk, action_dim]
    old_logprobs: torch.Tensor  # [B, chunk, action_dim] or [B, chunk]
    advantages: torch.Tensor  # [B]
    returns: Optional[torch.Tensor] = None  # [B]
    loss_mask: Optional[torch.Tensor] = None  # [B]
