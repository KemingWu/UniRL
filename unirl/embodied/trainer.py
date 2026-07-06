"""Embodied RL trainer — integrates with UniRL's training backend.

Uses UniRL's FSDPBackend, TrainStack, and WeightSync infrastructure for
distributed training while orchestrating the embodied-specific episodic
rollout collection loop.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
from omegaconf import DictConfig
from hydra.utils import instantiate

from unirl.embodied.algorithms.base import BaseAlgorithm
from unirl.embodied.engine import EmbodiedRolloutEngine
from unirl.embodied.envs.base import BaseEmbodiedEnv
from unirl.embodied.models.base import BasePolicy
from unirl.embodied.types import Trajectory

logger = logging.getLogger(__name__)


class EmbodiedTrainer:
    """Embodied RL trainer using UniRL's distributed training infrastructure.

    Integrates with:
    - ``unirl.train.backend.fsdp.FSDPBackend`` for FSDP model parallelism + LoRA
    - ``unirl.train.stack.TrainStack`` for micro-batch gradient accumulation
    - ``unirl.distributed.weight_sync`` for rollout ↔ train weight synchronization
    - ``unirl.distributed.group.DevicePool`` for GPU allocation

    The training loop:
        1. Collect episodic rollouts (env ↔ policy interaction)
        2. Compute advantages (GRPO / GAE)
        3. Build RolloutTrack with EmbodiedSegment
        4. Feed into TrainStack.train_track() for PPO update

    Config structure::

        num_devices: int
        batch_size: int
        layout: "colocate" | "separate"
        bundle: VLA model config
        backend: FSDPBackend config
        algorithm: StageAlgorithm config (EmbodiedGRPO)
        stack: TrainStack config
        env: BaseEmbodiedEnv config
        rollout: rollout engine config
        sync: WeightSync config (optional)
        logging: WandB config (optional)
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self._setup_device_pool(cfg)
        self._setup_components(cfg)

    def _setup_device_pool(self, cfg: DictConfig):
        """Initialize Ray-based GPU pool."""
        from unirl.distributed.group.device_pool import DevicePool

        num_devices = int(cfg.get("num_devices", 1))
        transport_kind = cfg.get("transport_kind", "colocate_store")

        self.pool = DevicePool(
            num_devices=num_devices,
            transport_kind=transport_kind,
        )
        self.pool.setup()
        logger.info(f"DevicePool initialized with {num_devices} devices")

    def _setup_components(self, cfg: DictConfig):
        """Instantiate all training components."""
        from unirl.distributed.group.placement import placement
        from unirl.distributed.group.remote import remote

        layout = cfg.get("layout", "colocate")

        with placement(self.pool, fraction=1.0, shared_workers=True):
            # 1. Model bundle (VLA policy)
            self.bundle = instantiate(cfg.bundle)

            # 2. FSDP Backend (wraps model with FSDP + LoRA + optimizer)
            self.backend = remote(
                instantiate,
                cfg.backend,
                bundle=self.bundle,
            )

            # 3. Algorithm (EmbodiedGRPO — PPO clip on action log-probs)
            self.algorithm = remote(instantiate, cfg.algorithm)

            # 4. TrainStack (micro-batch + optimizer step orchestration)
            self.stack = remote(
                instantiate,
                cfg.stack,
                fsdp_backend=self.backend,
                algorithm=self.algorithm,
            )

            # 5. Weight sync (optional, for separate layout)
            self.weight_sync = None
            if cfg.get("sync") is not None:
                self.weight_sync = remote(instantiate, cfg.sync, backend=self.backend)

        # 6. Environment and rollout engine (may run on different devices)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env: BaseEmbodiedEnv = instantiate(cfg.env, device=device)
        self.model: BasePolicy = instantiate(cfg.model)
        self.model = self.model.to(device)

        self.engine = EmbodiedRolloutEngine(self.env, self.model, cfg.rollout)

        # Training config
        self.batch_size = int(cfg.get("batch_size", 8))
        self._dataset_size = int(cfg.get("dataset_size", 1000))
        self._num_groups = int(cfg.rollout.get("num_groups", 8))

        # WandB
        self._logging_cfg = cfg.get("logging")

    def train(self, *, num_rollouts: int, weight_sync_interval: int = 1):
        """Main training loop using UniRL infrastructure.

        Args:
            num_rollouts: total training iterations.
            weight_sync_interval: sync weights to rollout model every N steps.
        """
        self._init_wandb(num_rollouts)

        try:
            for rollout_id in range(num_rollouts):
                training_progress = rollout_id / max(1, num_rollouts - 1)
                sync_weights = rollout_id > 0 and rollout_id % weight_sync_interval == 0

                result, mean_reward = self._train_step(
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

    def _train_step(
        self,
        *,
        training_progress: float,
        sync_weights: bool,
        rollout_id: int,
    ):
        """One full train step: rollout → advantage → train."""
        # Weight sync to rollout model
        if sync_weights and self.weight_sync is not None:
            self.weight_sync.sync()

        # 1. Collect trajectory
        episode_indices = self._sample_episode_indices()
        trajectory = self.engine.collect_trajectory(episode_indices)

        # 2. Compute advantages
        advantages = self._compute_advantages(trajectory)

        # 3. Build RolloutTrack for TrainStack
        track = self._trajectory_to_track(trajectory, advantages)

        # 4. Train via TrainStack (handles micro-batching, FSDP, optimizer)
        result = self.stack.train_track(track, training_progress=training_progress)

        # 5. Compute mean reward for logging
        mean_reward = float(trajectory.rewards.sum(dim=-1).sum(dim=0).mean())

        return result, mean_reward

    def _compute_advantages(self, trajectory: Trajectory) -> torch.Tensor:
        """Compute advantages using the configured algorithm."""
        algorithm_cfg = self.cfg.algorithm
        adv_type = algorithm_cfg.get("adv_type", "grpo")

        if adv_type == "grpo":
            from unirl.embodied.algorithms.advantages import compute_grpo_advantages

            # Episode-level reward aggregation
            mask = trajectory.loss_mask if trajectory.loss_mask is not None else torch.ones_like(
                trajectory.rewards[:, :, 0]
            )
            episode_rewards = (trajectory.rewards.sum(dim=-1) * mask).sum(dim=0)  # [B]
            group_size = int(algorithm_cfg.get("group_size", 4))
            return compute_grpo_advantages(episode_rewards, group_size)
        else:
            from unirl.embodied.algorithms.advantages import compute_gae_advantages

            rewards = trajectory.rewards.sum(dim=-1)  # [T, B]
            values = trajectory.prev_values.squeeze(-1) if trajectory.prev_values is not None else None
            return compute_gae_advantages(
                rewards=rewards,
                values=values,
                dones=trajectory.dones,
                gamma=float(algorithm_cfg.get("gamma", 0.99)),
                gae_lambda=float(algorithm_cfg.get("gae_lambda", 0.95)),
            )

    def _trajectory_to_track(self, trajectory: Trajectory, advantages: torch.Tensor):
        """Convert Trajectory + advantages into a RolloutTrack for TrainStack."""
        from unirl.types.rollout_resp import RolloutTrack
        from unirl.types.segments.base import Segment

        # Build an EmbodiedSegment-compatible structure for the algorithm
        # The algorithm's compute_loss_and_backward receives this segment
        from unirl.embodied.types import EmbodiedSegment

        T, B = trajectory.loss_mask.shape if trajectory.loss_mask is not None else (
            trajectory.actions.shape[0], trajectory.actions.shape[1]
        )

        segment = EmbodiedSegment(
            action_log_probs=trajectory.prev_logprobs,  # [T, B, chunk, action_dim]
            actions=trajectory.actions,  # [T, B, chunk, action_dim]
            observations=trajectory.observations,
            loss_mask=trajectory.loss_mask,  # [T, B]
            task_descriptions=trajectory.task_descriptions,
        )

        track = RolloutTrack(
            sample_ids=[f"ep_{i}" for i in range(B)],
            parent_ids=None,
            parent_track=None,
            conditions={},
            segment=segment,
            decoded=None,
            media_preview=None,
            rewards=advantages,  # [B] - already normalized advantages
            advantages=advantages,
        )
        return track

    def _sample_episode_indices(self) -> torch.Tensor:
        return torch.randint(0, self._dataset_size, (self._num_groups,))

    def _init_wandb(self, num_rollouts: int):
        if self._logging_cfg and self._logging_cfg.get("report_to_wandb", False):
            try:
                import wandb

                wandb.init(
                    project=self._logging_cfg.get("project_name", "unirl-embodied"),
                    name=self._logging_cfg.get("run_name", "embodied-rl"),
                    config=dict(self.cfg),
                )
            except Exception as e:
                logger.warning(f"Failed to init wandb: {e}")

    def _finish_wandb(self):
        try:
            import wandb

            if wandb.run is not None:
                wandb.finish()
        except Exception:
            pass
