"""DiffusionOPD — On-Policy Distillation for diffusion models.

Implements the algorithm from "DiffusionOPD: A Unified Perspective of On-Policy
Distillation in Diffusion Models" (arXiv 2605.15055). The student is trained to
match teacher denoising transitions via a closed-form per-step KL objective:

- **SDE mode** (noise_level > 0): KL = (μ_student - μ_teacher)² / (2σ²)
- **ODE mode** (noise_level = 0): mean-matching = 0.5 * (μ_student - μ_teacher)²

No external reward model is needed — supervision comes entirely from the teacher
adapter(s). Multiple teachers can be distilled via round-robin cycling (one
teacher per rollout batch).

Teacher adapters are loaded as frozen PEFT LoRA adapters on the same backbone;
``set_adapter(teacher_name)`` switches the active adapter before replay.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Mapping, Optional, Type

import torch

from unirl.algorithms.base import AlgorithmStepResult, StageAlgorithm, typed_conditions

logger = logging.getLogger(__name__)


@contextmanager
def _use_adapter(model: Any, adapter_name: str) -> Iterator[None]:
    """Temporarily switch active LoRA adapter at the LoraLayer level, restoring on exit.

    Works with models that use ``peft.inject_adapter_in_model`` (UniRL's pattern)
    rather than the full ``PeftModel`` wrapper.
    """
    from peft.tuners.lora import LoraLayer

    layers = [m for m in model.modules() if isinstance(m, LoraLayer)]
    prev_adapters = [getattr(m, "active_adapter", ["default"]) for m in layers]
    try:
        for m in layers:
            m.set_adapter(adapter_name)
        yield
    finally:
        for m, prev in zip(layers, prev_adapters):
            prev_name = prev[0] if isinstance(prev, list) else prev
            m.set_adapter(prev_name)


def _compute_std_var(sigmas: torch.Tensor, step_idx: int, eta: float) -> torch.Tensor:
    """Compute the SDE transition std_var from the sigma schedule.

    Mirrors FlowSDEStrategy.step:
        dt = sigma_next - sigma
        std_dev_t = sqrt(sigma / (1 - clamp(sigma, max=sigma_max))) * eta
        std_var = std_dev_t * sqrt(-dt)
    """
    sigma = sigmas[step_idx].float()
    sigma_next = sigmas[step_idx + 1].float()
    dt = sigma_next - sigma
    sigma_max = float(sigmas[1].item()) if sigmas.shape[0] > 1 else 0.99
    sigma_clamped = torch.where(sigma == 1.0, torch.tensor(sigma_max), sigma)
    std_dev_t = torch.sqrt(sigma / (1.0 - sigma_clamped)) * eta
    std_var = std_dev_t * torch.sqrt(-dt)
    return std_var


@dataclass
class DiffusionOPDConfig:
    """Typed config for the DiffusionOPD algorithm."""

    stage_attr: str = "diffusion"
    conditions_cls: Optional[str] = None
    params: Any = None
    teachers: List[Dict[str, Any]] = field(default_factory=list)
    noise_level: float = 0.0
    teacher_guidance_scale: float = 4.5


class DiffusionOPD(StageAlgorithm):
    """On-Policy Distillation for diffusion models (multi-teacher).

    Does NOT consume advantages — supervision is purely from teacher transition
    means. The algorithm still satisfies the ``StageAlgorithm`` interface so the
    standard ``DiffusionTrainer`` can host it without modification (it simply
    ignores the ``advantages`` arg in ``compute_loss_and_backward``).
    """

    requires_ema_rollout = False
    supports_multi_update = False
    requires_backend = True
    anchor_fields = ()

    def __init__(
        self,
        *,
        params: Any,
        stage: Any = None,
        pipeline: Any = None,
        stage_attr: str = "diffusion",
        backend: Any = None,
        conditions_cls: Optional[Type[Any]] = None,
        teachers: Optional[List[Dict[str, Any]]] = None,
        noise_level: float = 0.0,
        teacher_guidance_scale: float = 4.5,
    ) -> None:
        super().__init__()
        if stage is None and pipeline is None:
            raise ValueError("DiffusionOPD: either `stage` or `pipeline` must be provided")
        if stage is None:
            stage = getattr(pipeline, stage_attr)
        self.stage = stage
        self.params = params
        self.conditions_cls = conditions_cls
        self.noise_level = float(noise_level)
        self.teacher_guidance_scale = float(teacher_guidance_scale)

        if not teachers:
            raise ValueError("DiffusionOPD requires at least one teacher in `teachers` list.")
        self.teachers = list(teachers)

        # The trainable transformer — used for set_adapter calls.
        # Resolved from the backend (FSDP-wrapped) or the pipeline's bundle.
        self._transformer = None
        if backend is not None and hasattr(backend, "model"):
            self._transformer = backend.model
        elif pipeline is not None:
            bundle = getattr(pipeline, "bundle", None)
            if bundle is not None:
                self._transformer = getattr(bundle, "transformer", None)

        # Load teacher adapters (frozen) onto the transformer.
        # Deferred to first prepare_segment call because at __init__ time the
        # transformer may not yet be wrapped as a PeftModel (LoRA injection
        # happens later in the FSDPBackend lifecycle).
        self._teachers_loaded = False

        # Per-rollout teacher means storage (set in prepare_segment).
        self._teacher_means: Optional[torch.Tensor] = None
        # Round-robin counter: cycles through teachers across rollouts.
        self._rollout_counter: int = 0

    def _load_teacher_adapters(self) -> None:
        """Load each teacher's LoRA adapter onto the shared transformer (frozen).

        UniRL uses ``peft.inject_adapter_in_model`` (not ``get_peft_model``), so
        the model is NOT a PeftModel — it has LoRA layers but not the high-level
        ``.load_adapter()`` / ``.set_adapter()`` API. We inject each teacher as a
        separate named adapter using the same low-level API, then freeze its
        params. Switching adapters is done at the LoraLayer level.
        """
        from peft import LoraConfig as PeftLoraConfig
        from peft import inject_adapter_in_model, set_peft_model_state_dict
        from peft.tuners.lora import LoraLayer
        from safetensors.torch import load_file as safe_load

        model = self._transformer

        for tc in self.teachers:
            adapter_name = f"teacher_{tc['name']}"
            lora_path = tc["lora_path"]

            # Check if adapter already injected (idempotent).
            has_adapter = any(
                adapter_name in getattr(m, "lora_A", {})
                for m in model.modules()
                if isinstance(m, LoraLayer)
            )
            if has_adapter:
                continue

            logger.info("DiffusionOPD: loading teacher adapter %s from %s", adapter_name, lora_path)

            # Load the LoRA config and weights from the HF checkpoint.
            import json
            import os

            from huggingface_hub import hf_hub_download

            # Resolve local or HF path.
            if os.path.isdir(lora_path):
                config_path = os.path.join(lora_path, "adapter_config.json")
                weight_path = os.path.join(lora_path, "adapter_model.safetensors")
                if not os.path.exists(weight_path):
                    weight_path = os.path.join(lora_path, "adapter_model.bin")
            else:
                config_path = hf_hub_download(lora_path, "adapter_config.json")
                try:
                    weight_path = hf_hub_download(lora_path, "adapter_model.safetensors")
                except Exception:
                    weight_path = hf_hub_download(lora_path, "adapter_model.bin")

            with open(config_path) as f:
                adapter_cfg = json.load(f)

            peft_cfg = PeftLoraConfig(
                r=adapter_cfg.get("r", 32),
                lora_alpha=adapter_cfg.get("lora_alpha", 64),
                target_modules=adapter_cfg.get("target_modules", []),
                lora_dropout=adapter_cfg.get("lora_dropout", 0.0),
                bias=adapter_cfg.get("bias", "none"),
            )

            # Inject adapter structure (adds new LoRA A/B matrices under adapter_name).
            inject_adapter_in_model(peft_cfg, model, adapter_name=adapter_name)

            # Load weights.
            if weight_path.endswith(".safetensors"):
                state_dict = safe_load(weight_path)
            else:
                state_dict = torch.load(weight_path, map_location="cpu")
            set_peft_model_state_dict(model, state_dict, adapter_name=adapter_name)

            # Freeze teacher params.
            for m in model.modules():
                if isinstance(m, LoraLayer) and adapter_name in getattr(m, "lora_A", {}):
                    m.lora_A[adapter_name].weight.requires_grad_(False)
                    m.lora_B[adapter_name].weight.requires_grad_(False)

        # Restore student adapter as active.
        self._set_active_adapter(model, "default")

    @staticmethod
    def _set_active_adapter(model: Any, adapter_name: str) -> None:
        """Switch active adapter at the LoraLayer level (no PeftModel needed)."""
        from peft.tuners.lora import LoraLayer

        for m in model.modules():
            if isinstance(m, LoraLayer):
                m.set_adapter(adapter_name)

    @property
    def requires_prepare(self) -> bool:
        return True

    def prepare_segment(
        self,
        *,
        conditions: Mapping[str, Any],
        segment: Any,
    ) -> None:
        """Query the current round-robin teacher on the student's rollout trajectory.

        Following the paper's MOPD protocol: each rollout batch is supervised by
        ONE teacher (round-robin cycling across rollouts). This matches the
        original per-batch teacher assignment where batch_i uses
        teachers[i % K].

        The selected teacher's ``prev_sample_means`` (the per-step denoising
        transition mean under the teacher policy) are stored for
        ``compute_loss_and_backward``. The teacher's own ``guidance_scale`` is
        used during replay (not the student's), matching the original paper where
        each teacher was trained at its own CFG scale.
        """
        import dataclasses

        typed_conds = typed_conditions(conditions, self.conditions_cls)

        # Deferred teacher loading: at __init__ time the transformer is not yet
        # a PeftModel (LoRA injection happens in FSDPBackend after algorithm
        # construction). Load teacher adapters on first prepare_segment call.
        if not self._teachers_loaded and self._transformer is not None:
            self._load_teacher_adapters()
            self._teachers_loaded = True

        sde_indices = segment.sde_indices
        if sde_indices is None:
            self._teacher_means = None
            return
        target_steps = sde_indices.tolist()

        # Round-robin teacher selection (paper: each batch uses one teacher).
        teacher_idx = self._rollout_counter % len(self.teachers)
        self._rollout_counter += 1
        tc = self.teachers[teacher_idx]

        # Use the teacher's own guidance_scale for replay (each teacher was
        # trained at its own CFG scale; e.g. GenEval uses 1.0 while others use 4.5).
        teacher_gs = float(tc.get("guidance_scale", self.teacher_guidance_scale))
        teacher_params = dataclasses.replace(self.params, guidance_scale=teacher_gs)

        model = self._transformer
        adapter_name = f"teacher_{tc['name']}"
        with torch.no_grad(), _use_adapter(model, adapter_name):
            result = self.stage.replay(
                typed_conds,
                segment=segment,
                params=teacher_params,
                step_indices=target_steps,
            )

        if model is not None:
            self._set_active_adapter(model, "default")

        if result.prev_sample_means is not None:
            # Shape: [B, S', 1, *latent] — single teacher, K=1 dim for consistency.
            self._teacher_means = result.prev_sample_means.detach().unsqueeze(2)
        else:
            logger.warning(
                "DiffusionOPD: teacher %s returned None prev_sample_means; "
                "OPD requires the stage's replay to produce means.",
                tc["name"],
            )
            self._teacher_means = None

    def compute_loss_and_backward(
        self,
        *,
        conditions: Mapping[str, Any],
        segment: Any,
        advantages: torch.Tensor,
        training_progress: float,
        loss_scale: float,
    ) -> AlgorithmStepResult:
        """Compute the OPD distillation loss and backward.

        OPD does NOT use ``advantages`` — supervision is purely from the teacher
        transition means stored in ``prepare_segment``.
        """
        if self._teacher_means is None:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)

        typed_conds = typed_conditions(conditions, self.conditions_cls)
        sde_indices = segment.sde_indices
        if sde_indices is None:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)
        target_steps = sde_indices.tolist()

        # Student replay (with grad) to get student_prev_sample_means.
        replay_result = self.stage.replay(
            typed_conds,
            segment=segment,
            params=self.params,
            step_indices=target_steps,
        )
        student_means = replay_result.prev_sample_means  # [B, S', *latent]
        if student_means is None:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)

        # teacher_means: [B, S', K, *latent] (from prepare_segment)
        teacher_means = self._teacher_means.to(device=student_means.device, dtype=student_means.dtype)

        # delta: [B, S', K, *latent]
        # student_means is [B, S', *latent] -> unsqueeze dim=2 for broadcast
        delta = student_means.unsqueeze(2) - teacher_means

        # Compute per-step KL
        if self.noise_level > 0.0:
            # SDE mode: KL = delta² / (2σ²)
            # Compute std_var for each SDE step from the sigma schedule.
            sigmas = segment.sigmas.to(student_means.device)
            eta = float(getattr(self.params, "eta", 1.0))
            std_vars = []
            for step_idx in target_steps:
                sv = _compute_std_var(sigmas, step_idx, eta)
                std_vars.append(sv)
            # std_vars: [S'] scalars -> broadcast to [1, S', 1, 1, 1, 1]
            std_var_t = torch.stack(std_vars).to(student_means.device, student_means.dtype)
            # Reshape for broadcasting: [S'] -> [1, S', 1, ...]
            while std_var_t.dim() < delta.dim():
                std_var_t = std_var_t.unsqueeze(-1)
            std_var_t = std_var_t.unsqueeze(0)  # [1, S', 1, ...]
            sigma_sq = (std_var_t**2).clamp(min=1e-8)
            per_step_kl = (delta**2) / (2.0 * sigma_sq)
        else:
            # ODE mode: mean-matching = 0.5 * delta²
            per_step_kl = 0.5 * (delta**2)

        # Reduce: mean over latent dims, sum over teachers (K), mean over steps (S') and batch (B)
        # [B, S', K, *latent] -> mean over latent dims -> [B, S', K]
        per_step_kl_scalar = per_step_kl.mean(dim=tuple(range(3, per_step_kl.ndim)))
        # Sum over teachers (K dim)
        per_step_kl_per_sample = per_step_kl_scalar.sum(dim=2)  # [B, S']
        # Mean over steps and batch
        distill_loss = per_step_kl_per_sample.mean()

        loss = distill_loss
        (loss * loss_scale).backward()

        metrics: Dict[str, Any] = {
            "distill_loss": float(distill_loss.detach().item()),
            "per_step_kl_mean": float(per_step_kl_per_sample.detach().mean().item()),
        }
        return AlgorithmStepResult(
            loss=float(loss.detach().item()),
            metrics=metrics,
            num_steps_or_tokens=len(target_steps),
            has_backward=True,
        )


__all__ = ["DiffusionOPD", "DiffusionOPDConfig"]
