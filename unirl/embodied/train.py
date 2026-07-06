"""Hydra entrypoint for embodied RL training."""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../examples/embodied", config_name="wan_openvla_grpo")
def main(cfg: DictConfig) -> None:
    from unirl.embodied.trainer import EmbodiedTrainer

    trainer = EmbodiedTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
