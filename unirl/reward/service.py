"""Reward service: score a response :class:`~unirl.types.sample.Sample` in place."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.group.remote import Remote
from unirl.types.primitives import PrimitiveValue, primitive_modality_key
from unirl.types.reward import RewardRequest, RewardResponse
from unirl.types.sample import Sample, _part_with_field
from unirl.types.sampling import ARSamplingParams

from .base import DifferentiableReward, RewardBackend

logger = logging.getLogger(__name__)


def _build_reward_request(sample: Sample, preferred_input_kind: str) -> RewardRequest:
    """Assemble a :class:`RewardRequest` from a response ``Sample``."""
    frontier = sample.parts[-1]
    primitives: Dict[str, PrimitiveValue] = {}
    for prim in sample.conditioning():
        primitives[primitive_modality_key(prim)] = prim

    if preferred_input_kind not in frontier.primitives:
        raise ValueError(
            f"Reward backend consumes {preferred_input_kind!r} but the frontier Part generated "
            f"{sorted(frontier.primitives)!r}; check the recipe's reward/model pairing."
        )

    metadata = sample.root_metadata(-1)
    generated = dict(frontier.primitives)
    audio_sample_rate: Optional[int] = None
    audio_metadata = frontier.primitive_metadata.get("audio", {})
    if "audio" in generated and audio_metadata.get("sample_rate") is not None:
        audio_sample_rate = int(audio_metadata["sample_rate"])
    return RewardRequest(
        primitives=primitives,
        generated=generated,
        audio_sample_rate=audio_sample_rate,
        prompt_ids=[str(sid) for sid in frontier.sample_ids],
        sample_ids=list(frontier.sample_ids),
        group_ids=list(frontier.group_ids),
        metadata=(metadata if any(m is not None for m in metadata) else None),
    )


class RewardService(Remote):
    """Actor-side reward entry: one backend, scores a Sample's frontier Part in place."""

    def __init__(
        self,
        backend: RewardBackend,
        truncated_reward: str = "zero",
        overlong_buffer_len: int = 4096,
        overlong_penalty_factor: float = 1.0,
        max_failure_ratio: float = 0.0,
    ) -> None:
        super().__init__()
        self.backend = backend
        self.truncated_reward = str(truncated_reward)
        self.overlong_buffer_len = int(overlong_buffer_len)
        self.overlong_penalty_factor = float(overlong_penalty_factor)
        # Fraction of per-sample reward failures tolerated per batch before aborting.
        # 0.0 (default) keeps the historical strict behaviour: any failure raises.
        # Generative-judge backends need slack — an LLM judge occasionally emits
        # unparseable output, and killing a multi-thousand-rollout run over one bad
        # sample in a batch is not a useful failure mode. Tolerated samples are
        # neutralized (see score_and_attach), never scored as if they had succeeded.
        self.max_failure_ratio = float(max_failure_ratio)
        if not 0.0 <= self.max_failure_ratio < 1.0:
            raise ValueError(f"max_failure_ratio must be in [0, 1), got {self.max_failure_ratio!r}")
        if self.truncated_reward not in ("zero", "keep", "soft"):
            raise ValueError(f"truncated_reward must be zero|keep|soft, got {self.truncated_reward!r}")
        logger.info(
            "RewardService initialized with backend=%s, truncated_reward=%s, max_failure_ratio=%.3f",
            backend.get_model_name() or type(backend).__name__,
            self.truncated_reward,
            self.max_failure_ratio,
        )

    @property
    def preferred_input_kind(self) -> str:
        """The decoded media kind the backend consumes (image/video/text)."""
        kind = str(getattr(self.backend, "preferred_input_kind", "") or "").strip().lower()
        if kind not in {"image", "video", "text"}:
            raise ValueError(
                f"Reward backend must expose preferred_input_kind as 'image', 'video', or 'text'. Got {kind!r}."
            )
        return kind

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        return self.backend.compute_rewards(request)

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def score_differentiable(
        self,
        media_tensor: torch.Tensor,
        prompts: List[str],
        records: Optional[List[dict]] = None,
    ) -> torch.Tensor:
        """ReFL scoring of grad-carrying ``media_tensor`` against ``prompts``; returns ``[B]`` with grad_fn intact."""
        if not isinstance(self.backend, DifferentiableReward):
            raise TypeError(
                f"RewardService.score_differentiable: backend "
                f"{type(self.backend).__name__} is not a DifferentiableReward — ReFL "
                f"needs a differentiable in-process reward (e.g. pickscore/clip/hpsv2)."
            )
        return self.backend.compute_rewards_differentiable(media_tensor, list(prompts), records=records)

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def score_and_attach(self, sample: Sample) -> Sample:
        """Score the frontier (last) Part's generated media and return the updated Sample."""
        frontier = sample.parts[-1]
        if frontier.rewards is not None:
            raise RuntimeError("Actor-side reward compute does not accept precomputed rewards on the frontier Part.")
        if not frontier.primitives:
            raise ValueError("RewardService.score_and_attach: frontier Part has no generated primitives to score.")

        request = _build_reward_request(sample, self.preferred_input_kind)
        reward_response = self.compute_rewards(request)

        failed = [(i, e) for i, (ok, e) in enumerate(zip(reward_response.successes, reward_response.errors)) if not ok]
        n_total = len(reward_response.successes)
        if failed and (n_total == 0 or len(failed) / n_total > self.max_failure_ratio):
            raise RuntimeError(
                f"Reward computation flagged {len(failed)} of {n_total} "
                f"sample(s) as failure (max_failure_ratio={self.max_failure_ratio}). "
                f"First few: {failed[:3]}"
            )

        rewards = torch.tensor(reward_response.rewards, dtype=torch.float32)

        if failed:
            # Within tolerance: neutralize instead of aborting. Each failed sample
            # takes the mean of the SUCCEEDED rewards, so its group-relative
            # advantage is ~0 and it teaches nothing either way — as opposed to a
            # hard 0.0, which a generative judge's random parse hiccup would turn
            # into a strong "this image was terrible" signal.
            ok_mask = torch.tensor(reward_response.successes, dtype=torch.bool)
            fill = rewards[ok_mask].mean() if bool(ok_mask.any()) else torch.zeros((), dtype=torch.float32)
            rewards[~ok_mask] = fill
            logger.warning(
                "RewardService: neutralized %d/%d failed sample(s) at reward=%.4f "
                "(within max_failure_ratio=%.3f). First few: %s",
                len(failed),
                n_total,
                float(fill),
                self.max_failure_ratio,
                failed[:3],
            )

        sp = frontier.sampling_params
        if self.truncated_reward != "keep" and isinstance(sp, ARSamplingParams) and frontier.segment is not None:
            seg_lengths = getattr(frontier.segment, "lengths", None)
            if seg_lengths is not None and seg_lengths.numel() == rewards.numel():
                seg_lengths = seg_lengths.to(rewards.device).float()
                max_len = float(int(sp.max_new_tokens))
                if self.truncated_reward == "zero":
                    truncated = seg_lengths >= max_len
                    rewards = torch.where(truncated, torch.zeros_like(rewards), rewards)
                else:
                    buf = float(self.overlong_buffer_len)
                    exceed = seg_lengths - (max_len - buf)
                    penalty = torch.clamp(-exceed / buf * self.overlong_penalty_factor, max=0.0)
                    rewards = rewards + penalty

        component_rewards = {
            str(name): torch.tensor(list(values or []), dtype=torch.float32)
            for name, values in dict(reward_response.component_rewards or {}).items()
        }
        scored = _part_with_field(frontier, "rewards", rewards)
        scored = _part_with_field(scored, "component_rewards", component_rewards)
        return sample.with_parts([*sample.parts[:-1], scored])

    def is_available(self) -> bool:
        return self.backend.is_available()

    def offload(self) -> None:
        self.backend.offload()

    def onload(self) -> None:
        self.backend.onload()

    def dispose(self) -> None:
        self.backend.dispose()

    def shutdown(self) -> None:
        """Worker teardown hook: release backend sessions and managed children."""
        self.dispose()


__all__ = [
    "RewardService",
]
