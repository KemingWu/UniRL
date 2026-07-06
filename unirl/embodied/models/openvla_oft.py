"""OpenVLA-OFT policy model for embodied RL.

Actions are discretized into bins and predicted as token IDs by the VLM.
Log-probabilities are computed per-token (categorical) for PPO training.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from unirl.embodied.models.base import BasePolicy


def compute_logprobs_from_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute per-token log-probabilities: log_softmax(logits)[target].

    Args:
        logits: [*, vocab_size]
        target: [*] token indices

    Returns:
        logprobs: [*] per-token log-probs
    """
    batch_shape = logits.shape[:-1]
    vocab_size = logits.shape[-1]
    logprobs = -F.cross_entropy(
        logits.reshape(-1, vocab_size),
        target.reshape(-1),
        reduction="none",
    )
    return logprobs.view(*batch_shape).float()


def compute_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Compute entropy of the distribution defined by logits."""
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy


class OpenVLAOFTPolicy(BasePolicy):
    """OpenVLA-OFT: Vision-Language-Action model with discretized action output.

    Actions are tokenized into bins (default 256) within the LLM vocabulary.
    The model predicts action_dim * num_action_chunks tokens per forward pass.

    Config fields:
        model_path: HuggingFace checkpoint path
        num_action_chunks: actions per prediction (default 8)
        action_dim: action space dimensionality (default 7)
        n_action_bins: discretization bins (default 256)
        max_prompt_length: max token length for prompt (default 512)
        use_value_head: whether to attach a critic (default False)
        precision: "bf16" or "fp32"
        temperature: sampling temperature (default 1.0)
        top_k: top-k sampling (default -1 = disabled)
        unnorm_key: dataset statistics key for unnormalization
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self._cfg = cfg
        self._num_action_chunks = int(cfg.get("num_action_chunks", 8))
        self._action_dim = int(cfg.get("action_dim", 7))
        self._n_action_bins = int(cfg.get("n_action_bins", 256))
        self._max_prompt_length = int(cfg.get("max_prompt_length", 512))
        self._use_value_head = bool(cfg.get("use_value_head", False))
        self._temperature = float(cfg.get("temperature", 1.0))
        self._top_k = int(cfg.get("top_k", -1))
        self._unnorm_key = cfg.get("unnorm_key", "libero_spatial_no_noops")

        self._response_len = self._action_dim * self._num_action_chunks

        # Load model
        self._model, self._processor, self._config = self._load_model(cfg)

        # Action discretization
        self._vocab_size = self._config.text_config.vocab_size - getattr(self._config, "pad_to_multiple_of", 64)
        bins = np.linspace(-1, 1, self._n_action_bins + 1)
        self._bin_centers = torch.from_numpy((bins[:-1] + bins[1:]) / 2.0).float()

        # Action unnormalization stats
        self._action_stats = self._load_action_stats(cfg)

        # Value head (optional)
        if self._use_value_head:
            hidden_size = self._config.text_config.hidden_size
            self.value_head = nn.Linear(hidden_size, 1)
        else:
            self.value_head = None

    def _load_model(self, cfg: DictConfig):
        from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor

        precision = cfg.get("precision", "bf16")
        dtype = torch.bfloat16 if precision == "bf16" else torch.float32

        config = AutoConfig.from_pretrained(cfg.model_path, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            cfg.model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(cfg.model_path, trust_remote_code=True)
        return model, processor, config

    def _load_action_stats(self, cfg: DictConfig) -> Optional[Dict]:
        """Load dataset statistics for action unnormalization."""
        import json
        from pathlib import Path

        stats_path = Path(cfg.model_path) / "dataset_statistics.json"
        if stats_path.exists():
            with open(stats_path) as f:
                all_stats = json.load(f)
            key = self._unnorm_key
            if key in all_stats:
                return all_stats[key]
        return None

    @property
    def num_action_chunks(self) -> int:
        return self._num_action_chunks

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def predict_action_batch(self, **obs) -> Dict[str, Any]:
        """Generate actions from observations (rollout time).

        Returns actions as continuous values and caches forward_inputs for replay.
        """
        main_images = obs["main_images"]  # [B, H, W, C] uint8
        task_descriptions = obs.get("task_descriptions", [""] * main_images.shape[0])
        B = main_images.shape[0]
        device = next(self.parameters()).device

        # Prepare inputs
        input_ids, attention_mask, pixel_values = self._prepare_inputs(main_images, task_descriptions, device)

        # Forward through model
        with torch.no_grad():
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                output_hidden_states=self._use_value_head,
                use_cache=False,
            )

        # Extract action logits (last response_len positions)
        logits = outputs.logits[:, -self._response_len - 1: -1, :]  # [B, response_len, vocab]

        # Mask to action bins only
        action_logits = logits.clone()
        action_logits[..., : self._vocab_size - self._n_action_bins] = -torch.inf
        action_logits[..., self._vocab_size:] = -torch.inf

        # Sample action tokens
        if self._temperature > 0:
            processed = action_logits / self._temperature
            if self._top_k > 0:
                top_k = min(self._top_k, processed.shape[-1])
                indices_to_remove = processed < torch.topk(processed, top_k, dim=-1).values[..., -1:]
                processed[indices_to_remove] = -torch.inf
            probs = F.softmax(processed, dim=-1)
            action_tokens = torch.multinomial(probs.view(-1, probs.shape[-1]), num_samples=1).view(B, -1)
        else:
            action_tokens = action_logits.argmax(dim=-1)  # [B, response_len]

        # Compute log-probs for sampled tokens
        logprobs = compute_logprobs_from_logits(action_logits, action_tokens)  # [B, response_len]

        # Detokenize to continuous actions
        actions = self._detokenize_actions(action_tokens)  # [B, num_chunks, action_dim]

        # Value head
        values = None
        if self.value_head is not None and outputs.hidden_states is not None:
            hidden = outputs.hidden_states[-1][:, -self._response_len - 1]
            values = self.value_head(hidden)  # [B, 1]

        return {
            "actions": actions,
            "logprobs": logprobs,  # [B, action_dim * num_chunks]
            "values": values,
            "forward_inputs": {
                "input_ids": input_ids.detach(),
                "attention_mask": attention_mask.detach(),
                "pixel_values": pixel_values.detach(),
                "action_tokens": action_tokens.detach(),
            },
        }

    def default_forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Teacher-forced forward for PPO log-prob replay.

        Uses cached forward_inputs to recompute logprobs with current weights.
        """
        forward_inputs = kwargs.get("forward_inputs")
        if forward_inputs is None:
            forward_inputs = kwargs

        input_ids = forward_inputs["input_ids"]
        attention_mask = forward_inputs["attention_mask"]
        pixel_values = forward_inputs["pixel_values"]
        action_tokens = forward_inputs["action_tokens"]  # [B, response_len]

        # Forward pass with current weights
        outputs = self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            output_hidden_states=self._use_value_head,
            use_cache=False,
        )

        # Extract action logits
        logits = outputs.logits[:, -self._response_len - 1: -1, :]

        # Apply temperature + top-k (same as rollout for consistency)
        processed = logits / self._temperature
        if self._top_k > 0:
            top_k = min(self._top_k, processed.shape[-1])
            indices_to_remove = processed < torch.topk(processed, top_k, dim=-1).values[..., -1:]
            processed[indices_to_remove] = -torch.inf

        # Mask to action bins
        action_logits = processed
        action_logits[..., : self._vocab_size - self._n_action_bins] = -torch.inf
        action_logits[..., self._vocab_size:] = -torch.inf

        # Compute logprobs for the SAME action tokens (PPO ratio denominator stays fixed)
        logprobs = compute_logprobs_from_logits(action_logits, action_tokens)

        result = {"logprobs": logprobs}

        # Entropy (optional)
        if kwargs.get("compute_entropy", False):
            result["entropy"] = compute_entropy_from_logits(action_logits).mean(dim=-1)

        # Value head (optional)
        if self.value_head is not None and kwargs.get("compute_values", False):
            hidden = outputs.hidden_states[-1][:, -self._response_len - 1]
            result["values"] = self.value_head(hidden)

        return result

    def _prepare_inputs(self, images: torch.Tensor, tasks: List[str], device: torch.device):
        """Prepare model inputs from raw observations."""
        B = images.shape[0]
        prompts = [f"In: What action should the robot take to {t.lower()}?\nOut: " for t in tasks]

        all_input_ids = []
        all_attention_masks = []
        all_pixel_values = []

        for i in range(B):
            img_np = images[i].cpu().numpy() if images[i].device.type != "cpu" else images[i].numpy()
            from PIL import Image

            pil_img = Image.fromarray(img_np.astype(np.uint8)).convert("RGB")
            processed = self._processor(prompts[i], pil_img, return_tensors="pt")
            all_input_ids.append(processed["input_ids"])
            all_attention_masks.append(processed["attention_mask"])
            all_pixel_values.append(processed["pixel_values"])

        # Pad to same length
        max_len = max(ids.shape[1] for ids in all_input_ids)
        max_len = max(max_len, self._max_prompt_length)
        pad_id = self._processor.tokenizer.pad_token_id or 0

        input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)

        for i in range(B):
            seq_len = all_input_ids[i].shape[1]
            # Left-pad
            input_ids[i, max_len - seq_len:] = all_input_ids[i][0]
            attention_mask[i, max_len - seq_len:] = all_attention_masks[i][0]

        pixel_values = torch.cat(all_pixel_values, dim=0).to(device)

        return input_ids, attention_mask, pixel_values

    def _detokenize_actions(self, action_tokens: torch.Tensor) -> torch.Tensor:
        """Convert action token IDs back to continuous actions [B, num_chunks, action_dim]."""
        B = action_tokens.shape[0]
        # Token IDs → bin indices
        bin_indices = self._vocab_size - action_tokens  # higher token ID = lower bin
        bin_indices = bin_indices.clamp(1, self._n_action_bins) - 1

        # Bin indices → normalized actions [-1, 1]
        bin_centers = self._bin_centers.to(action_tokens.device)
        normalized = bin_centers[bin_indices.long()]  # [B, response_len]

        # Unnormalize using dataset statistics
        actions = self._unnormalize_actions(normalized)

        # Reshape: [B, action_dim * num_chunks] → [B, num_chunks, action_dim]
        return actions.reshape(B, self._num_action_chunks, self._action_dim)

    def _unnormalize_actions(self, normalized: torch.Tensor) -> torch.Tensor:
        """Unnormalize actions from [-1, 1] to real action space."""
        if self._action_stats is None:
            return normalized

        # Use q01/q99 or min/max for unnormalization
        q01 = self._action_stats.get("q01", self._action_stats.get("min"))
        q99 = self._action_stats.get("q99", self._action_stats.get("max"))

        if q01 is None or q99 is None:
            return normalized

        q01 = torch.tensor(q01, device=normalized.device, dtype=normalized.dtype)
        q99 = torch.tensor(q99, device=normalized.device, dtype=normalized.dtype)

        # Scale from [-1, 1] to [q01, q99]
        actions = (normalized + 1) / 2 * (q99 - q01) + q01
        return actions
