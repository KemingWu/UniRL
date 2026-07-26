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
    """Temporarily route every LoraLayer through ``adapter_name``.

    PEFT ``LoraLayer.forward`` iterates over ``self.active_adapters`` and applies
    each adapter's A/B matrices. Merely tweaking the ``scaling`` dict does NOT
    switch the adapter — the teacher's A/B are never executed unless it appears
    in ``active_adapter``. We write directly to the ``_active_adapter`` backing
    attribute (same pattern as ``unirl.train.lora.adapters_disabled`` uses
    ``_disable_adapters``). Direct attribute writes avoid PEFT's
    ``LoraLayer.set_adapter()`` side-effects (which flip ``requires_grad`` and
    can misbehave once FSDP has sharded the params).
    """
    from peft.tuners.lora import LoraLayer

    layers = [m for m in model.modules() if isinstance(m, LoraLayer)]
    prev: list = []
    for m in layers:
        aa = getattr(m, "_active_adapter", None)
        if aa is None:
            aa = getattr(m, "active_adapter", "default")
        prev.append(list(aa) if isinstance(aa, list) else [aa])
    try:
        for m in layers:
            m._active_adapter = [adapter_name]
        yield
    finally:
        for m, p in zip(layers, prev):
            m._active_adapter = p


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
    sigma_clamped = torch.where(sigma == 1.0, sigma.new_tensor(sigma_max), sigma)
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
    requires_advantages = False
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

        # Round-robin coupling: the data source (``MultiTeacherRLDataSource``)
        # cycles its per-teacher dataloaders in the same order this list
        # declares. If the two orders drift, batch N's prompts stop matching
        # teacher N's domain — silently. Fail loudly at init if names look
        # off (unique + non-empty).
        _names = [str(t.get("name", "")) for t in self.teachers]
        if any(not n for n in _names):
            raise ValueError(f"DiffusionOPD: every teacher entry needs a 'name', got {_names}")
        if len(set(_names)) != len(_names):
            raise ValueError(f"DiffusionOPD: teacher names must be unique, got {_names}")

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
                adapter_name in getattr(m, "lora_A", {}) for m in model.modules() if isinstance(m, LoraLayer)
            )
            if has_adapter:
                continue

            logger.info("DiffusionOPD: loading teacher adapter %s from %s", adapter_name, lora_path)

            # Load the LoRA config and weights from the HF checkpoint.
            import json
            import os

            from huggingface_hub import hf_hub_download

            # Resolve local or HF path.
            # Supports: local dir, "org/repo" (flat), "org/repo/subfolder" (nested).
            if os.path.isdir(lora_path):
                config_path = os.path.join(lora_path, "adapter_config.json")
                weight_path = os.path.join(lora_path, "adapter_model.safetensors")
                if not os.path.exists(weight_path):
                    weight_path = os.path.join(lora_path, "adapter_model.bin")
            else:
                # Parse "org/repo/subfolder" → repo_id="org/repo", subfolder="subfolder"
                parts = lora_path.split("/")
                if len(parts) > 2:
                    repo_id = "/".join(parts[:2])
                    subfolder = "/".join(parts[2:])
                else:
                    repo_id = lora_path
                    subfolder = None

                dl_kwargs = {"repo_id": repo_id}
                if subfolder:
                    dl_kwargs["subfolder"] = subfolder

                config_path = hf_hub_download(filename="adapter_config.json", **dl_kwargs)
                try:
                    weight_path = hf_hub_download(filename="adapter_model.safetensors", **dl_kwargs)
                except Exception:
                    weight_path = hf_hub_download(filename="adapter_model.bin", **dl_kwargs)

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

        # PEFT's ``inject_adapter_in_model`` invokes ``set_adapter(<new>)`` as a
        # side-effect while wiring up the new adapter, which flips
        # ``requires_grad`` on ALL other adapters (including our student
        # "default") to ``False``. We bypass PEFT's ``set_adapter()`` at
        # runtime, so nothing restores it — student loss ends up with no
        # grad_fn and ``backward()`` raises. Explicitly re-enable trainability
        # on the student adapter after all teachers are loaded.
        n_restored = 0
        for m in model.modules():
            if not isinstance(m, LoraLayer):
                continue
            for attr in ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"):
                d = getattr(m, attr, None)
                if d is None or "default" not in d:
                    continue
                for p in d["default"].parameters():
                    p.requires_grad_(True)
                    n_restored += 1
        logger.info(
            "DiffusionOPD: restored requires_grad=True on %d student ('default') LoRA params after teacher injection.",
            n_restored,
        )

    @staticmethod
    def _set_active_adapter(model: Any, adapter_name: str) -> None:
        """Switch active adapter by directly writing ``_active_adapter``.

        Same rationale as ``_use_adapter``: PEFT's ``LoraLayer.set_adapter()``
        flips ``requires_grad`` on the adapters' weights, which is unsafe once
        FSDP has sharded the params. Writing the backing attribute leaves
        weights alone and only updates the routing metadata.
        """
        from peft.tuners.lora import LoraLayer

        for m in model.modules():
            if isinstance(m, LoraLayer):
                m._active_adapter = [adapter_name]

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
        if sde_indices is None or sde_indices.numel() == 0:
            segment.sde_means = None
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
            # Shape: [B, S', *latent] — DiffusionOPD is per-batch one-teacher
            # (round-robin), so there is no multi-teacher K dim to carry. Keep
            # the means on the segment: TrainStack slices CONCAT fields with
            # each micro-batch, preserving batch alignment during replay.
            segment.sde_means = result.prev_sample_means.detach().cpu()
            # Debug: verify teacher adapter actually changed the output.
            # Do a quick student replay to compare.
            with torch.no_grad():
                student_result = self.stage.replay(
                    typed_conds,
                    segment=segment,
                    params=self.params,
                    step_indices=target_steps[:1],  # just first step for speed
                )
            if student_result.prev_sample_means is not None:
                t_mean = result.prev_sample_means[:, 0].float()
                s_mean = student_result.prev_sample_means[:, 0].float()
                delta_norm = (t_mean - s_mean).norm().item()
                logger.info(
                    "DiffusionOPD[%s]: teacher-student delta norm at step %d = %.6f "
                    "(should be >> 0 if adapter switch works)",
                    tc["name"],
                    target_steps[0],
                    delta_norm,
                )
        else:
            logger.warning(
                "DiffusionOPD: teacher %s returned None prev_sample_means; "
                "OPD requires the stage's replay to produce means.",
                tc["name"],
            )
            segment.sde_means = None

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
        if segment.sde_means is None:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)

        typed_conds = typed_conditions(conditions, self.conditions_cls)
        sde_indices = segment.sde_indices
        if sde_indices is None or sde_indices.numel() == 0:
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

        # Cast BOTH sides to fp32 before the squared-difference. Under autocast
        # the replay returns bf16 means, and squaring in bf16 loses precision
        # fast (the KL is a per-pixel MSE reduced over ~65k elements). Flow-
        # Factory does the same (trainers/opd/trainer.py:339).
        student_means_fp32 = student_means.float()
        teacher_means = segment.sde_means.to(device=student_means.device, dtype=torch.float32)

        # delta: [B, S', *latent]
        delta = student_means_fp32 - teacher_means

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
            # std_vars: [S'] scalars -> broadcast to [1, S', 1, ...]
            std_var_t = torch.stack(std_vars).to(student_means.device, torch.float32)
            std_var_t = std_var_t.unsqueeze(0)
            while std_var_t.dim() < delta.dim():
                std_var_t = std_var_t.unsqueeze(-1)
            sigma_sq = (std_var_t**2).clamp(min=1e-8)
            per_step_kl = (delta**2) / (2.0 * sigma_sq)
        else:
            # ODE mode: mean-matching = 0.5 * delta²
            per_step_kl = 0.5 * (delta**2)

        # Reduce: mean over latent dims → [B, S'], then mean over steps + batch.
        per_step_kl_per_sample = per_step_kl.mean(dim=tuple(range(2, per_step_kl.ndim)))
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
