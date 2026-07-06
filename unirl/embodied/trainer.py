"""Embodied RL trainer — extends BaseTrainer with episodic rollout.

Follows the same architecture as ``DiffusionTrainer``:
- Inherits ``BaseTrainer`` (DevicePool + WandB)
- Uses ``remote_hydra()`` to build all distributed components
- ``train_step()`` pattern: rollout → reward → advantages → stack.train_track
- ``train()`` loop delegates to ``train_step()``

The only structural difference is the rollout engine: instead of a one-shot
generation pipeline, it runs multi-step episodic interaction (env ↔ policy).
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

from omegaconf import DictConfig

from unirl.distributed.group.placement import placement
from unirl.train.stack import TrainStepResult
from unirl.trainer.base import BaseTrainer
from unirl.utils.hydra import remote_hydra

logger = logging.getLogger(__name__)


class EmbodiedTrainer(BaseTrainer):
    """Embodied RL trainer: episodic rollout + UniRL training infrastructure.

    Mirrors ``DiffusionTrainer`` architecture:
    - ``_build_train_side()``: bundle → backend → algorithm → stack (all Remote)
    - ``_build_rollout()``: embodied rollout engine (Remote)
    - ``train_step()``: generate → score → advantages → train_track
    - ``train()``: loop over train_step

    The embodied rollout engine wraps a ``BaseEmbodiedEnv`` and ``BasePolicy``
    to collect multi-step episodes, returning a standard ``RolloutResp`` with
    an ``EmbodiedSegment`` that TrainStack processes.
    """

    def __init__(
        self,
        *,
        cfg: DictConfig,
        batch_size: int,
        bundle_cfg: DictConfig,
        backend_cfg: DictConfig,
        algorithm_cfg: DictConfig,
        stack_cfg: DictConfig,
        rollout_cfg: DictConfig,
        env_cfg: DictConfig,
        reward_cfg: Optional[DictConfig] = None,
        sync_cfg: Optional[DictConfig] = None,
        logging_cfg: Optional[DictConfig] = None,
        layout: str = "colocate",
        train_fraction: float = 0.5,
    ) -> None:
        super().__init__(cfg=cfg, logging_cfg=logging_cfg)
        self.batch_size = batch_size

        self.weight_sync = None

        with placement(self.pool, fraction=1.0, shared_workers=True):
            self._build_train_side(
                bundle_cfg=bundle_cfg,
                backend_cfg=backend_cfg,
                algorithm_cfg=algorithm_cfg,
                stack_cfg=stack_cfg,
            )
            self.rollout = self._build_rollout(rollout_cfg, env_cfg=env_cfg)
            if sync_cfg is not None:
                self.weight_sync = remote_hydra(sync_cfg, backend=self.backend, rollout=self.rollout)

    def _build_train_side(
        self,
        *,
        bundle_cfg: DictConfig,
        backend_cfg: DictConfig,
        algorithm_cfg: DictConfig,
        stack_cfg: DictConfig,
    ) -> None:
        """Build train-side remotes in the active placement scope."""
        self.bundle = remote_hydra(bundle_cfg)
        self.backend = remote_hydra(backend_cfg, bundle=self.bundle)
        self.algorithm = remote_hydra(algorithm_cfg, policy=self.bundle)
        self.stack = remote_hydra(stack_cfg, fsdp_backend=self.backend, algorithm=self.algorithm)

    def _build_rollout(self, rollout_cfg: DictConfig, *, env_cfg: DictConfig):
        """Build the embodied rollout engine in the active placement scope."""
        return remote_hydra(rollout_cfg, env_cfg=env_cfg, policy=self.bundle)

    def train_step(
        self,
        *,
        training_progress: float = 0.0,
        sync_weights: bool = False,
        rollout_id: int = 0,
    ) -> Tuple[TrainStepResult, float]:
        """One rollout → reward → advantage → optimizer step pass.

        Same contract as ``DiffusionTrainer.train_step``.
        """
        t0 = time.perf_counter()

        if sync_weights and self.weight_sync is not None:
            self.weight_sync.sync()

        # Collect episodes via the embodied rollout engine
        resp = self.rollout.generate(rollout_id=rollout_id)

        # Advantage computation (built into the rollout resp)
        mean_reward = 0.0
        for track in resp.tracks.values():
            if track.rewards is not None:
                mean_reward = float(track.rewards.to("cpu").float().mean().item())
                break

        for name, track in list(resp.tracks.items()):
            if track.rewards is not None:
                resp.tracks[name] = track.compute_advantages(normalize=True)

        # Train via TrainStack
        (track,) = resp.tracks.values()
        result = self.stack.train_track(track, training_progress=training_progress)
        self._log_rollout(rollout_id, result, resp, step_time_s=time.perf_counter() - t0)
        return result, mean_reward

    def train(self, *, num_rollouts: int, weight_sync_interval: int = 1) -> None:
        """Training loop: N iterations of train_step."""
        interval = max(1, weight_sync_interval)
        self._init_wandb(num_rollouts=num_rollouts)
        try:
            for rollout_id in range(num_rollouts):
                training_progress = rollout_id / max(1, num_rollouts - 1)
                sync_weights = rollout_id > 0 and rollout_id % interval == 0
                result, mean_reward = self.train_step(
                    training_progress=training_progress,
                    sync_weights=sync_weights,
                    rollout_id=rollout_id,
                )
                logger.info(
                    "rollout %d/%d  reward=%.4f  loss=%.4f  grad_norm=%.4f  lr=%.2e",
                    rollout_id + 1,
                    num_rollouts,
                    mean_reward,
                    result.loss,
                    result.grad_norm,
                    result.lr,
                )
        finally:
            self._finish_wandb()
