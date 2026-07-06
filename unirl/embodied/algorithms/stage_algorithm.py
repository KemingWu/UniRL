"""EmbodiedGRPO — StageAlgorithm-compatible PPO for embodied RL.

Uses the cached forward_inputs from rollout to replay the VLA forward pass
with updated weights, computing new log-probs for the PPO ratio. This is
identical in principle to how LLM RLHF works: the action_tokens stay fixed,
but the probability distribution over them changes as the model updates.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from unirl.algorithms.base import AlgorithmStepResult, StageAlgorithm, _grpo_clip_loss
from unirl.embodied.models.base import BasePolicy
from unirl.embodied.types import EmbodiedSegment


class EmbodiedGRPO(StageAlgorithm):
    """PPO clip loss on discretized VLA action token log-probs.

    The VLA predicts action_dim * num_chunks tokens per step. Each token is a
    categorical choice from n_action_bins options. Log-probs are per-token,
    and the PPO ratio is computed on the sum of log-probs across all action
    tokens in a step (equivalent to the joint probability of the action chunk).

    Training replay: uses cached forward_inputs (input_ids, attention_mask,
    pixel_values, action_tokens) to run the VLA forward pass with current
    weights and get new log-probs for the SAME action tokens.
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
        entropy_bonus: float = 0.0,
    ):
        self.policy = policy
        self.clip_range = clip_range
        self.clip_range_high = clip_range_high
        self.clip_schedule = clip_schedule
        self.entropy_bonus = entropy_bonus

    def prepare_segment(
        self,
        *,
        conditions: Mapping[str, Any],
        segment: EmbodiedSegment,
    ) -> None:
        """Freeze action_log_probs as π_old anchor."""
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
        """Replay VLA with cached forward_inputs, compute PPO loss, backward.

        The replay re-runs the model forward on the SAME inputs but with updated
        weights. The action_tokens are fixed (we compute logprobs for the same
        actions), producing new_logprobs. The PPO ratio = exp(new - old).
        """
        old_logprobs = segment.action_log_probs  # [T, B, response_len]
        loss_mask = segment.loss_mask  # [T, B]
        forward_inputs = segment.forward_inputs

        # Replay all valid steps through the policy
        new_logprobs_flat, entropy_flat = self._replay(forward_inputs)

        # Reconstruct [T, B, response_len] from flat [N_valid, response_len]
        T, B = loss_mask.shape
        valid_mask = loss_mask.bool()
        response_len = old_logprobs.shape[-1]

        new_logprobs = torch.zeros(T, B, response_len, device=old_logprobs.device)
        idx = 0
        for t in range(T):
            for b in range(B):
                if valid_mask[t, b]:
                    new_logprobs[t, b] = new_logprobs_flat[idx]
                    idx += 1

        # Sum log-probs across action tokens per step → joint action probability
        old_step_logp = old_logprobs.sum(dim=-1)  # [T, B]
        new_step_logp = new_logprobs.sum(dim=-1)  # [T, B]

        # Flatten valid steps for loss computation
        valid = valid_mask.reshape(-1)  # [T*B]
        old_flat = old_step_logp.reshape(-1)[valid]  # [N_valid]
        new_flat = new_step_logp.reshape(-1)[valid]  # [N_valid]

        # Expand advantages [B] → per-step [T, B] → valid only
        adv_expanded = advantages.unsqueeze(0).expand(T, B).reshape(-1)[valid]

        # PPO clip loss
        clip_range = self._resolve_clip_range(training_progress)
        clip_high = self.clip_range_high if self.clip_range_high is not None else clip_range

        loss_per_elem, metrics_tensors = _grpo_clip_loss(
            new_logp=new_flat,
            old_logp=old_flat,
            advantages=adv_expanded,
            clip_range=clip_range,
            clip_range_high=clip_high,
        )

        loss = loss_per_elem.mean()

        # Entropy bonus
        if self.entropy_bonus > 0 and entropy_flat is not None:
            entropy_mean = entropy_flat.mean()
            loss = loss - self.entropy_bonus * entropy_mean

        # Backward
        (loss * loss_scale).backward()

        metrics: Dict[str, Any] = {
            "embodied/policy_loss": float(loss),
            "embodied/ratio_mean": float(metrics_tensors["ratio_mean"]),
            "embodied/clip_fraction": float(metrics_tensors["clip_fraction"]),
            "embodied/approx_kl": float(metrics_tensors["approx_kl"]),
        }
        if self.entropy_bonus > 0 and entropy_flat is not None:
            metrics["embodied/entropy"] = float(entropy_flat.mean())

        return AlgorithmStepResult(
            loss=float(loss),
            metrics=metrics,
            num_steps_or_tokens=int(valid.sum()),
            has_backward=True,
        )

    def _replay(self, forward_inputs: Dict[str, Any]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Replay VLA forward with cached inputs to get new log-probs.

        Args:
            forward_inputs: dict with input_ids, attention_mask, pixel_values,
                action_tokens concatenated across valid timesteps. Shape [N_valid, ...].

        Returns:
            (new_logprobs[N_valid, response_len], entropy[N_valid] or None)
        """
        # Filter out metadata keys
        model_inputs = {k: v for k, v in forward_inputs.items() if not k.startswith("_")}

        result = self.policy.default_forward(
            forward_inputs=model_inputs,
            compute_entropy=(self.entropy_bonus > 0),
        )

        logprobs = result["logprobs"]  # [N_valid, response_len]
        entropy = result.get("entropy")  # [N_valid] or None

        return logprobs, entropy

    def _resolve_clip_range(self, progress: float) -> float:
        if self.clip_schedule == "linear_decay":
            return self.clip_range * (1.0 - 0.5 * progress)
        if self.clip_schedule == "cosine_decay":
            import math

            return self.clip_range * (0.5 * (1.0 + math.cos(math.pi * progress)))
        return self.clip_range
