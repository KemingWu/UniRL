"""EmbodiedGRPO — StageAlgorithm-compatible PPO for embodied RL.

This module bridges the embodied RL data (EmbodiedSegment with action
log-probs) into UniRL's StageAlgorithm interface so that TrainStack can
drive training without modification.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from unirl.algorithms.base import AlgorithmStepResult, StageAlgorithm, _grpo_clip_loss
from unirl.embodied.models.base import BasePolicy
from unirl.embodied.types import EmbodiedSegment


class EmbodiedGRPO(StageAlgorithm):
    """PPO clip loss on continuous VLA action log-probs.

    Conforms to UniRL's ``StageAlgorithm`` interface:
    - ``prepare_segment()``: freezes ``action_log_probs`` as π_old anchor
    - ``compute_loss_and_backward()``: replays VLA forward, computes PPO loss

    Unlike FlowGRPO (SDE per-step logp on latents), this operates on
    per-timestep action log-probs across an embodied episode.
    """

    supports_multi_update = True
    anchor_fields = ("action_log_probs",)

    def __init__(
        self,
        *,
        policy: BasePolicy,
        clip_range: float = 0.2,
        clip_range_high: Optional[float] = None,
        clip_schedule: str = "constant",
    ):
        self.policy = policy
        self.clip_range = clip_range
        self.clip_range_high = clip_range_high
        self.clip_schedule = clip_schedule

    def prepare_segment(
        self,
        *,
        conditions: Mapping[str, Any],
        segment: EmbodiedSegment,
    ) -> None:
        """Freeze action_log_probs as the π_old anchor.

        Called once before the multi-update loop. The rollout-time log-probs
        become the frozen anchor for PPO ratio computation.
        """
        # action_log_probs already populated during rollout — just detach
        if segment.action_log_probs is not None:
            segment.action_log_probs = segment.action_log_probs.detach()

    def compute_loss_and_backward(
        self,
        *,
        conditions: Mapping[str, Any],
        segment: EmbodiedSegment,
        advantages: torch.Tensor,
        training_progress: float,
        loss_scale: float,
    ) -> AlgorithmStepResult:
        """Replay VLA forward, compute PPO clip loss, call backward.

        Args:
            conditions: unused for embodied (observations are in segment).
            segment: EmbodiedSegment with action_log_probs (old), actions, observations.
            advantages: [B] per-episode advantages.
            training_progress: [0, 1] for schedule.
            loss_scale: gradient accumulation factor.
        """
        old_logprobs = segment.action_log_probs  # [T, B, chunk, action_dim]
        actions = segment.actions  # [T, B, chunk, action_dim]
        obs = segment.observations  # dict with per-step observations
        mask = segment.loss_mask  # [T, B]

        # Replay: teacher-forced forward to get new log-probs
        new_logprobs = self._replay(obs, actions)

        # Flatten time dimension for loss computation
        T, B = old_logprobs.shape[:2]
        old_flat = old_logprobs.reshape(T * B, -1)  # [T*B, chunk*action_dim]
        new_flat = new_logprobs.reshape(T * B, -1)

        # Broadcast advantages: [B] → [T*B] (each timestep gets episode advantage)
        adv_expanded = advantages.unsqueeze(0).expand(T, B).reshape(T * B)

        # Build per-element mask
        if mask is not None:
            mask_flat = mask.reshape(T * B)
        else:
            mask_flat = torch.ones(T * B, device=old_flat.device)

        # Sum log-probs across action dims for per-step ratio
        old_step_logp = old_flat.sum(dim=-1)  # [T*B]
        new_step_logp = new_flat.sum(dim=-1)  # [T*B]

        # PPO clip loss
        clip_range = self._resolve_clip_range(training_progress)
        clip_high = self.clip_range_high if self.clip_range_high is not None else clip_range

        loss_per_elem, metrics_tensors = _grpo_clip_loss(
            new_logp=new_step_logp,
            old_logp=old_step_logp,
            advantages=adv_expanded,
            clip_range=clip_range,
            clip_range_high=clip_high,
        )

        # Masked mean reduction
        valid_count = mask_flat.sum().clamp(min=1)
        loss = (loss_per_elem * mask_flat).sum() / valid_count

        # Backward
        scaled_loss = loss * loss_scale
        scaled_loss.backward()

        # Metrics
        metrics: Dict[str, Any] = {
            "embodied/policy_loss": float(loss),
            "embodied/ratio_mean": float(metrics_tensors["ratio_mean"]),
            "embodied/clip_fraction": float(metrics_tensors["clip_fraction"]),
            "embodied/approx_kl": float(metrics_tensors["approx_kl"]),
        }

        return AlgorithmStepResult(
            loss=float(loss),
            metrics=metrics,
            num_steps_or_tokens=int(valid_count),
            has_backward=True,
        )

    def _replay(self, observations: Any, actions: torch.Tensor) -> torch.Tensor:
        """Teacher-forced replay through the VLA to get new log-probs."""
        T, B = actions.shape[:2]

        if observations is None or not observations:
            return torch.zeros_like(actions)

        # Replay each timestep through the policy
        all_logprobs = []
        obs_steps = observations.get("steps", []) if isinstance(observations, dict) else []

        for t in range(T):
            if t < len(obs_steps) and obs_steps[t] is not None:
                step_obs = obs_steps[t]
                step_actions = actions[t]  # [B, chunk, action_dim]
                result = self.policy.default_forward(obs=step_obs, actions=step_actions)
                all_logprobs.append(result["logprobs"])
            else:
                all_logprobs.append(torch.zeros_like(actions[t]))

        return torch.stack(all_logprobs, dim=0)  # [T, B, chunk, action_dim]

    def _resolve_clip_range(self, progress: float) -> float:
        if self.clip_schedule == "linear_decay":
            return self.clip_range * (1.0 - 0.5 * progress)
        if self.clip_schedule == "cosine_decay":
            import math

            return self.clip_range * (0.5 * (1.0 + math.cos(math.pi * progress)))
        return self.clip_range
