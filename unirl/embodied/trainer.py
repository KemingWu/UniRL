"""Embodied RL trainer — orchestrates rollout, advantage, and training."""

from __future__ import annotations

import logging
from typing import Optional

import torch
from omegaconf import DictConfig
from hydra.utils import instantiate

from unirl.embodied.algorithms.base import BaseAlgorithm
from unirl.embodied.engine import EmbodiedRolloutEngine
from unirl.embodied.envs.base import BaseEmbodiedEnv
from unirl.embodied.models.base import BasePolicy
from unirl.embodied.types import TrainBatch, Trajectory

logger = logging.getLogger(__name__)


class EmbodiedTrainer:
    """Top-level training orchestrator for embodied RL.

    Manages the train loop: collect episodes → compute advantages → train policy.

    Config structure:
        env: BaseEmbodiedEnv config
        model: BasePolicy config
        algorithm: BaseAlgorithm config
        rollout: rollout engine config
        training:
            num_rollouts: total training iterations
            micro_batch_size: samples per gradient step
            num_updates_per_rollout: PPO epochs per rollout
            learning_rate: optimizer LR
            max_grad_norm: gradient clipping
            weight_decay: AdamW weight decay
        logging:
            log_interval: steps between metric logs
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

        # Instantiate components via Hydra
        self.env: BaseEmbodiedEnv = instantiate(cfg.env, device=self.device)
        self.model: BasePolicy = instantiate(cfg.model)
        self.model = self.model.to(self.device)
        self.algorithm: BaseAlgorithm = instantiate(cfg.algorithm)
        self.engine = EmbodiedRolloutEngine(self.env, self.model, cfg.rollout)

        # Training setup
        train_cfg = cfg.training
        self.num_rollouts = int(train_cfg.num_rollouts)
        self.micro_batch_size = int(train_cfg.get("micro_batch_size", 8))
        self.num_updates = int(train_cfg.get("num_updates_per_rollout", 1))
        self.max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(train_cfg.learning_rate),
            weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        )
        self.global_step = 0

        # Data source
        self._dataset_size = int(cfg.get("dataset_size", 1000))
        self._num_groups = int(cfg.rollout.get("num_groups", 8))

    def train(self):
        """Main training loop."""
        logger.info(f"Starting embodied RL training for {self.num_rollouts} rollouts")

        for rollout_idx in range(self.num_rollouts):
            # 1. Sample episode indices
            episode_indices = self._sample_episode_indices()

            # 2. Collect trajectories
            trajectory = self.engine.collect_trajectory(episode_indices)

            # 3. Compute advantages
            advantages = self.algorithm.compute_advantages(trajectory)

            # 4. Training updates
            train_metrics = self._train_on_trajectory(trajectory, advantages)

            # 5. Logging
            if rollout_idx % self.cfg.get("logging", {}).get("log_interval", 10) == 0:
                logger.info(
                    f"Step {rollout_idx}/{self.num_rollouts} | "
                    f"loss={train_metrics.get('policy_loss', 0):.4f} | "
                    f"kl={train_metrics.get('approx_kl', 0):.4f}"
                )

            self.global_step += 1

    def _train_on_trajectory(self, trajectory: Trajectory, advantages: torch.Tensor) -> dict:
        """Run multiple PPO epochs over a collected trajectory."""
        all_metrics = {}

        for update_idx in range(self.num_updates):
            batches = self._make_mini_batches(trajectory, advantages)
            for batch in batches:
                self.optimizer.zero_grad()

                # Replay model to get new logprobs
                new_logprobs = self._replay_logprobs(batch)
                batch_with_new = TrainBatch(
                    observations=batch.observations,
                    actions=batch.actions,
                    old_logprobs=batch.old_logprobs,
                    advantages=batch.advantages,
                    returns=batch.returns,
                    loss_mask=batch.loss_mask,
                )

                # Compute loss with new logprobs vs old
                from unirl.embodied.algorithms.losses import compute_ppo_loss

                loss, metrics = compute_ppo_loss(
                    logprobs=new_logprobs,
                    old_logprobs=batch.old_logprobs,
                    advantages=batch.advantages,
                    clip_low=self.algorithm.clip_ratio_low if hasattr(self.algorithm, "clip_ratio_low") else 0.2,
                    clip_high=self.algorithm.clip_ratio_high if hasattr(self.algorithm, "clip_ratio_high") else 0.2,
                    loss_mask=batch.loss_mask,
                )

                loss.backward()
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                all_metrics.update(metrics)

        return all_metrics

    def _replay_logprobs(self, batch: TrainBatch) -> torch.Tensor:
        """Teacher-forced forward to get new log-probs for PPO ratio."""
        result = self.model.default_forward(obs=batch.observations, actions=batch.actions)
        return result["logprobs"]

    def _make_mini_batches(self, trajectory: Trajectory, advantages: torch.Tensor):
        """Split trajectory into mini-batches for training."""
        # Flatten time dimension: gather valid steps per episode
        T, B = trajectory.loss_mask.shape
        valid_mask = trajectory.loss_mask.bool()

        # For now: use episode-level batching (one trajectory slice per episode)
        # Each batch item is the full episode for one env
        batches = []
        for start in range(0, B, self.micro_batch_size):
            end = min(start + self.micro_batch_size, B)
            batch_indices = list(range(start, end))
            mb_size = len(batch_indices)

            # Gather observations for replay (first obs per episode)
            obs_steps = trajectory.observations["steps"] if trajectory.observations else []
            batch_obs = {}
            if obs_steps:
                first_obs = obs_steps[0]
                for key, val in first_obs.items():
                    if isinstance(val, torch.Tensor):
                        batch_obs[key] = val[batch_indices]
                    elif isinstance(val, list):
                        batch_obs[key] = [val[i] for i in batch_indices]

            batches.append(
                TrainBatch(
                    observations=batch_obs,
                    actions=trajectory.actions[:, batch_indices].reshape(mb_size, -1, self.env.action_dim),
                    old_logprobs=trajectory.prev_logprobs[:, batch_indices].reshape(
                        mb_size, -1, self.env.action_dim
                    ),
                    advantages=advantages[batch_indices],
                    loss_mask=trajectory.loss_mask[:, batch_indices].reshape(mb_size, -1)
                    if trajectory.loss_mask is not None
                    else None,
                )
            )
        return batches

    def _sample_episode_indices(self) -> torch.Tensor:
        """Sample random episode indices for rollout."""
        return torch.randint(0, self._dataset_size, (self._num_groups,))
