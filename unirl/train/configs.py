from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

LoraModuleSelection = Union[str, Tuple[str, ...]]


@dataclass
class FrozenAdapterSpec:
    """A frozen, inference-only LoRA adapter loaded at build time.

    ``name`` is the peft adapter name (algorithms address it by this name,
    e.g. DiffusionOPD teachers); ``path`` is a local peft checkpoint dir or an
    HF repo id (``org/repo`` / ``org/repo/subfolder``). Geometry (rank / alpha /
    target_modules) comes from the checkpoint's own ``adapter_config.json``.
    """

    name: str = ""
    path: str = ""


def normalize_frozen_adapters(specs: Any) -> List[FrozenAdapterSpec]:
    """Validate ``LoraConfig.frozen_adapters`` entries into typed specs.

    Accepts ``None`` (→ ``[]``) or a sequence of ``FrozenAdapterSpec`` /
    mappings / attribute objects carrying ``name`` + ``path``. Names must be
    non-empty, unique, and must not shadow the trainable ``"default"`` adapter.
    """
    if not specs:
        return []
    normalized: List[FrozenAdapterSpec] = []
    for entry in specs:
        if isinstance(entry, FrozenAdapterSpec):
            name, path = entry.name, entry.path
        elif hasattr(entry, "get"):
            name, path = entry.get("name"), entry.get("path")
        else:
            name, path = getattr(entry, "name", None), getattr(entry, "path", None)
        if not name or not path:
            raise ValueError(f"frozen_adapters entries need non-empty 'name' and 'path'; got {entry!r}.")
        if str(name) == "default":
            raise ValueError("frozen_adapters: 'default' is the trainable adapter and cannot be frozen.")
        normalized.append(FrozenAdapterSpec(name=str(name), path=str(path)))
    names = [s.name for s in normalized]
    if len(set(names)) != len(names):
        raise ValueError(f"frozen_adapters names must be unique, got {names}.")
    return normalized


@dataclass
class LoraConfig:
    rank: int = 8
    alpha: int = 16
    # A tuple uses PEFT's exact/suffix matching; a string is preserved for
    # regular-expression matching and the special ``all-linear`` shorthand.
    # ``Any`` is intentional: OmegaConf 2.3 rejects unions containing
    # containers. The injection boundary validates this as LoraModuleSelection.
    target_modules: Any = ("q_proj", "k_proj", "v_proj", "o_proj")
    # Modules matched by ``target_modules`` but explicitly excluded from injection.
    # Like ``target_modules``, tuples match exact names/suffixes and strings are
    # regular expressions. This lets a regex exclude an entire frozen sub-tower.
    exclude_modules: Any = None
    # Restrict sequence-style target suffixes to a named model subtree.
    # This cannot be combined with regex or ``all-linear`` target strings.
    module_prefix: str = ""
    dropout: float = 0.0
    bias: str = "none"
    task_type: str = "FEATURE_EXTRACTION"
    # Additional frozen, inference-only adapters injected next to the trainable
    # ``default`` (e.g. DiffusionOPD teachers). Entries carry ``name`` + ``path``
    # (see :class:`FrozenAdapterSpec`; plain dicts are accepted — ``Any`` for the
    # same OmegaConf 2.3 reason as ``target_modules``). Sound only on this
    # plain-LoRA path: the base weights stay frozen, so a teacher's deltas keep
    # referring to the weights it was trained against.
    frozen_adapters: Any = None


@dataclass
class EmaLoraConfig:
    rank: int = 8
    alpha: int = 16
    target_modules: Any = ("q_proj", "k_proj", "v_proj", "o_proj")
    exclude_modules: Any = None
    dropout: float = 0.0
    bias: str = "none"
    task_type: str = "FEATURE_EXTRACTION"
    default_adapter: str = "default"
    shadow_adapter: str = "old"
    ema_decay: float = 0.001
    ema_decay_type: str = "constant"
    ema_flat_steps: int = 0
    ema_uprate: float = 0.001
    ema_uphold: float = 0.5
    timing: str = "rollout_end"


@dataclass
class EmaFullConfig:
    target_decay: float = 0.9999
    timing: str = "optimizer_step"
    shadow_prefix: str = "shadow_"


@dataclass
class FSDPConfig:
    param_dtype: str = "bf16"
    cpu_offload: bool = False
    mixed_precision: bool = True
    fsdp_mode: str = "full"
    reshard_after_forward: bool = True
    activation_checkpointing: bool = False
    use_torch_compile: bool = False
    # Opt-in no-sync gradient accumulation: defer the per-block FSDP2 gradient
    # reduce-scatter to the last micro-batch of an optimizer step, so one
    # reduce-scatter runs per step instead of one per micro-batch (the standard
    # FSDP2 no-sync pattern; a multi-node win, ~no-op over NVLink). Only takes
    # effect under ZeRO-2 (reshard_after_forward=False); ignored otherwise. NOT
    # bit-identical to the per-micro path: the deferred grads accumulate in the
    # unsharded buffers at param dtype (bf16 under mixed precision) across the
    # micro-batches, whereas per-micro sync reduces each in reduce_dtype (fp32) —
    # so reward / grad-norm parity must be confirmed before enabling in a recipe.
    defer_grad_sync: bool = False
    # Opt-in FSDP2 cross-block forward prefetch: overlap each block's all-gather
    # with compute (a multi-node win, ~no-op over NVLink). Chains the default root
    # wrap to prefetch block 0, then block i to prefetch block i+1; needs the root
    # wrap (raises if root_wrap=False). Off keeps the default wrap with no prefetch.
    forward_prefetch: bool = False
    # Optional high-precision master dtype for the TRAINABLE params (e.g. "fp32").
    # When set, fsdp_wrap keeps the trainable (LoRA) sharded master + optimizer
    # states at this dtype while MixedPrecisionPolicy(param_dtype) still casts the
    # all-gathered compute copy to param_dtype (bf16) — the standard "fp32 master +
    # bf16 compute" recipe flow_grpo relies on for stable low-gradient RL. None
    # (default) = master dtype follows param_dtype (the prior all-bf16 behavior).
    master_dtype: Optional[str] = None
    # Root fully_shard (default ON): after the per-block wrap, fully_shard the
    # model root so the leftover params (embed / final norm / lm_head) are
    # sharded + mp_policy'd like everything else instead of staying plain
    # replicated tensors (which need manual grad sync and keep full fp32
    # masters per rank). Set false for models whose stages call submodules of
    # the wrapped object directly outside a root forward (bagel's vendored
    # code) or whose wrapped object carries frozen mixed-dtype sibling
    # sub-models that must not be sharded (hunyuan_image3's VAE/ViT).
    root_wrap: bool = True
    # Checkpoint storage backend. "torch" (default) keeps the legacy path:
    # gather a full state dict to rank 0 and torch.save a single checkpoint.pt.
    # "dcp" uses torch.distributed.checkpoint sharded save/load — each rank
    # reads/writes only its own shard (no rank-0 full-tensor gather), which
    # enables meta-init bundles (80B) to checkpoint, parallelizes I/O across
    # ranks, and resumes under a different world size. load auto-detects the
    # on-disk format, so legacy checkpoint.pt dirs still resume regardless.
    checkpoint_format: str = "torch"
    # dcp only: background (async) save (dcp.async_save) so the train loop is not
    # blocked on the shard I/O — staging is synchronous, the flush runs on a
    # background thread, drained before the next save and after the final one.
    # Ignored under checkpoint_format="torch".
    checkpoint_async: bool = False
    # Ulysses sequence-parallel degree (default 1 = disabled, a true no-op).
    # When >1 the VeOmni backend builds a folded dp_shard x ulysses FSDP mesh
    # (init_parallel_state(ulysses_size=sp_size, dp_size=world//sp_size)) and
    # installs the per-architecture SP patch (slice the sequence across sp_size
    # ranks, all-to-all in attention) — see unirl.train.backend.veomni.sp.
    # Must divide the world size and the model's attention head count. Only the
    # VeOmni backend honors it; FSDPBackend ignores it.
    sp_size: int = 1
    # Expert-parallel degree for MoE models (default 1 = disabled, a true no-op).
    # When >1 the VeOmni backend registers an "ep" extra-parallel submesh in
    # init_parallel_state (ep x ep_fsdp), shards fused experts, and requires the
    # bundle to implement prepare_for_expert_parallel() so the trainable model
    # exposes get_parallel_plan() (Shard(0) on stacked expert weights). Must
    # divide world size and num_experts. VeOmni only; FSDPBackend ignores it.
    ep_size: int = 1


__all__ = [
    "LoraConfig",
    "LoraModuleSelection",
    "EmaLoraConfig",
    "EmaFullConfig",
    "FSDPConfig",
]
