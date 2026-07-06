"""Hydra entrypoint for embodied RL training."""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../examples/embodied", config_name="wan_openvla_grpo")
def main(cfg: DictConfig) -> None:
    from unirl.embodied.trainer import EmbodiedTrainer

    trainer = EmbodiedTrainer(
        cfg=cfg,
        batch_size=int(cfg.batch_size),
        bundle_cfg=cfg.bundle,
        backend_cfg=cfg.backend,
        algorithm_cfg=cfg.algorithm,
        stack_cfg=cfg.stack,
        rollout_cfg=cfg.rollout,
        env_cfg=cfg.env,
        reward_cfg=cfg.get("reward"),
        sync_cfg=cfg.get("sync"),
        logging_cfg=cfg.get("logging"),
        layout=cfg.get("layout", "colocate"),
        train_fraction=float(cfg.get("train_fraction", 0.5)),
    )
    trainer.train(
        num_rollouts=int(cfg.training.num_rollouts),
        weight_sync_interval=int(cfg.training.get("weight_sync_interval", 1)),
    )


if __name__ == "__main__":
    main()
