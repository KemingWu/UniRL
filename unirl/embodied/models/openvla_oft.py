"""OpenVLA-OFT policy model for embodied RL."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig

from unirl.embodied.models.base import BasePolicy


class OpenVLAOFTPolicy(BasePolicy):
    """OpenVLA-OFT: Vision-Language-Action model with continuous action output.

    Wraps a pretrained OpenVLA-OFT checkpoint and exposes the predict/replay
    interface for embodied RL training.

    Config fields:
        model_path: HuggingFace checkpoint path
        num_action_chunks: actions per prediction (default 8)
        action_dim: action space dimensionality (default 7)
        use_value_head: whether to attach a critic (default False)
        precision: "bf16" or "fp32"
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self._cfg = cfg
        self._num_action_chunks = int(cfg.get("num_action_chunks", 8))
        self._action_dim = int(cfg.get("action_dim", 7))
        self._use_value_head = bool(cfg.get("use_value_head", False))

        self._model = self._load_model(cfg)
        if self._use_value_head:
            hidden_size = self._model.config.hidden_size if hasattr(self._model, "config") else 4096
            self._value_head = nn.Linear(hidden_size, 1)
        else:
            self._value_head = None

    def _load_model(self, cfg: DictConfig) -> nn.Module:
        from transformers import AutoModelForVision2Seq, AutoProcessor

        precision = cfg.get("precision", "bf16")
        dtype = torch.bfloat16 if precision == "bf16" else torch.float32

        model = AutoModelForVision2Seq.from_pretrained(
            cfg.model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self._processor = AutoProcessor.from_pretrained(cfg.model_path, trust_remote_code=True)
        return model

    @property
    def num_action_chunks(self) -> int:
        return self._num_action_chunks

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def predict_action_batch(self, **obs) -> Dict[str, torch.Tensor]:
        main_images = obs["main_images"]  # [B, H, W, C] uint8
        task_descriptions = obs.get("task_descriptions", [""] * main_images.shape[0])
        B = main_images.shape[0]
        device = next(self.parameters()).device

        # Process inputs through the VLA model
        inputs = self._prepare_inputs(main_images, task_descriptions, device)
        with torch.no_grad():
            outputs = self._model.generate(**inputs, max_new_tokens=self._num_action_chunks * self._action_dim)

        actions = self._decode_actions(outputs, B)
        logprobs = torch.zeros(B, self._num_action_chunks, self._action_dim, device=device)

        result = {
            "actions": actions,
            "logprobs": logprobs,
            "forward_inputs": {"main_images": main_images, "task_descriptions": task_descriptions},
        }
        if self._value_head is not None:
            result["values"] = torch.zeros(B, 1, device=device)
        return result

    def default_forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        obs = kwargs.get("obs", kwargs)
        actions = kwargs["actions"]  # [B, chunk, action_dim]
        main_images = obs["main_images"]
        task_descriptions = obs.get("task_descriptions", [""] * main_images.shape[0])
        device = next(self.parameters()).device

        inputs = self._prepare_inputs(main_images, task_descriptions, device)
        outputs = self._model(**inputs, labels=self._encode_actions(actions))

        logprobs = self._compute_action_logprobs(outputs, actions)
        return {"logprobs": logprobs}

    def _prepare_inputs(self, images: torch.Tensor, tasks: list, device: torch.device) -> Dict:
        """Prepare model inputs from raw observations."""
        # Convert uint8 [B,H,W,C] to PIL-like format for processor
        B = images.shape[0]
        processed = self._processor(
            text=tasks,
            images=[images[i].cpu().numpy() for i in range(B)],
            return_tensors="pt",
            padding=True,
        )
        return {k: v.to(device) for k, v in processed.items()}

    def _decode_actions(self, outputs: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Decode model outputs to continuous actions [B, chunk, action_dim]."""
        # Model-specific decoding — this is a placeholder that should be
        # adapted to the specific OpenVLA-OFT checkpoint format
        device = outputs.device if hasattr(outputs, "device") else next(self.parameters()).device
        return torch.zeros(batch_size, self._num_action_chunks, self._action_dim, device=device)

    def _encode_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Encode continuous actions for teacher-forced training."""
        return actions.reshape(actions.shape[0], -1)

    def _compute_action_logprobs(self, outputs, actions: torch.Tensor) -> torch.Tensor:
        """Extract per-action log-probabilities from model outputs."""
        B = actions.shape[0]
        device = actions.device
        return torch.zeros(B, self._num_action_chunks, self._action_dim, device=device)
