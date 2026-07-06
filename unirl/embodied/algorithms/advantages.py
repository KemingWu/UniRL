"""Advantage estimation functions."""

from __future__ import annotations

from typing import Optional

import torch


def compute_grpo_advantages(
    episode_rewards: torch.Tensor,
    group_size: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Group Relative Policy Optimization advantages.

    Normalizes rewards within groups of episodes from the same initial state.

    Args:
        episode_rewards: [B] scalar rewards per episode.
            B must be divisible by group_size.
        group_size: number of episodes per group.
        eps: numerical stability.

    Returns:
        advantages: [B] normalized within groups.
    """
    B = episode_rewards.shape[0]
    assert B % group_size == 0, f"B={B} not divisible by group_size={group_size}"
    num_groups = B // group_size

    grouped = episode_rewards.reshape(num_groups, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True)
    advantages = (grouped - mean) / (std + eps)
    return advantages.reshape(B)


def compute_gae_advantages(
    rewards: torch.Tensor,
    values: Optional[torch.Tensor],
    dones: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    normalize: bool = True,
    loss_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Generalized Advantage Estimation.

    Args:
        rewards: [T, B] per-step rewards.
        values: [T, B] value estimates (None → zeros).
        dones: [T, B] episode boundary flags.
        gamma: discount factor.
        gae_lambda: GAE lambda.
        normalize: whether to normalize advantages.
        loss_mask: [T, B] valid step mask.
        eps: numerical stability.

    Returns:
        advantages: [T, B]
    """
    T, B = rewards.shape
    device = rewards.device

    if values is None:
        values = torch.zeros(T + 1, B, device=device)
    elif values.shape[0] == T:
        # Append bootstrap value (zero for terminal episodes)
        values = torch.cat([values, torch.zeros(1, B, device=device)], dim=0)

    advantages = torch.zeros(T, B, device=device)
    last_gae = torch.zeros(B, device=device)

    for t in reversed(range(T)):
        not_done = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * values[t + 1] * not_done - values[t]
        last_gae = delta + gamma * gae_lambda * not_done * last_gae
        advantages[t] = last_gae

    if loss_mask is not None:
        advantages = advantages * loss_mask

    if normalize:
        if loss_mask is not None:
            valid = loss_mask.bool()
            mean = advantages[valid].mean()
            std = advantages[valid].std()
        else:
            mean = advantages.mean()
            std = advantages.std()
        advantages = (advantages - mean) / (std + eps)

    return advantages
