"""Policy loss functions."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


def compute_ppo_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clip_low: float = 0.2,
    clip_high: float = 0.2,
    loss_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """PPO clipped surrogate loss.

    Args:
        logprobs: [B, ...] new log-probabilities from current policy.
        old_logprobs: [B, ...] old log-probabilities from rollout.
        advantages: [B] or broadcastable to logprobs shape.
        clip_low: lower clip epsilon (ratio floor = 1 - clip_low).
        clip_high: upper clip epsilon (ratio ceil = 1 + clip_high).
        loss_mask: optional mask for valid entries.

    Returns:
        (scalar_loss, metrics_dict)
    """
    log_ratio = logprobs - old_logprobs
    ratio = torch.exp(log_ratio)

    # Broadcast advantages to match logprobs shape
    adv = advantages.detach()
    while adv.dim() < ratio.dim():
        adv = adv.unsqueeze(-1)
    adv = adv.expand_as(ratio)

    unclipped = -adv * ratio
    clipped = -adv * torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high)
    loss_per_elem = torch.maximum(unclipped, clipped)

    if loss_mask is not None:
        mask = loss_mask.detach()
        while mask.dim() < loss_per_elem.dim():
            mask = mask.unsqueeze(-1)
        mask = mask.expand_as(loss_per_elem)
        loss = (loss_per_elem * mask).sum() / mask.sum().clamp(min=1)
    else:
        loss = loss_per_elem.mean()

    # Metrics
    with torch.no_grad():
        clip_fraction = ((ratio - 1.0).abs() > clip_low).float().mean()
        approx_kl = (0.5 * log_ratio.pow(2)).mean()

    metrics = {
        "policy_loss": float(loss),
        "ratio_mean": float(ratio.mean()),
        "ratio_std": float(ratio.std()) if ratio.numel() > 1 else 0.0,
        "clip_fraction": float(clip_fraction),
        "approx_kl": float(approx_kl),
    }
    return loss, metrics
