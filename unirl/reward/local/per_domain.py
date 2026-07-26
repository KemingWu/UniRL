"""Per-domain reward scorer for DiffusionOPD (or any multi-teacher setup).

Routes each item to a domain-specific inner scorer based on
``request.metadata[i]["teacher_domain"]`` and returns per-domain component
rewards alongside a scalar-per-item total.

The inner scorers are passed in fully constructed (Hydra ``_target_``
instantiated) via ``PerDomainSpec.scorers``, so any :class:`RewardBackend`
subclass works — local (PickScore, OCR, GenEval) or remote
(:class:`unirl.reward.remote.RemoteRewardBackend`) — without this file
knowing about registry names.

Motivation: for OPD, ``prepare_segment`` cycles through per-teacher batches
(pickscore / ocr / geneval). Different scorers need different metadata (OCR
reads quoted target text from the prompt, GenEval2 wants ``vqa_list``, the
classical GenEval wants ``include``/``exclude``), and a scorer can't score a
batch outside its own domain. This scorer resolves both concerns:

- One inner scorer per domain — each only sees the items it can score.
- Per-domain wandb series via ``component_rewards`` (``pickscore``, ``ocr``,
  ``geneval``, …), so all three teachers' progress is
  visible on separate curves.

Reward is monitoring-only in OPD (the loss comes from teacher transition
means), so the aggregated ``total`` is a bookkeeping value rather than a
training signal.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from unirl.reward.base import BaseRewardComponentSpec, RewardBackend
from unirl.types.primitives import Images, Texts
from unirl.types.reward import RewardRequest, RewardResponse


def _slice_request(request: RewardRequest, indices: List[int]) -> RewardRequest:
    """Return a fresh ``RewardRequest`` containing only the given indices."""
    import torch

    def _slice_primitive(v: Any) -> Any:
        if v is None:
            return v
        if isinstance(v, Texts):
            return Texts(texts=[v.texts[i] for i in indices])
        if isinstance(v, Images):
            return Images(pixels=v.pixels[indices] if isinstance(v.pixels, torch.Tensor) else v.pixels)
        # Fallback: assume slice-able / indexable list-like
        try:
            return type(v)(**{f.name: [getattr(v, f.name)[i] for i in indices] for f in dataclasses.fields(v)})
        except Exception:
            return v

    sub_primitives = {k: _slice_primitive(v) for k, v in request.primitives.items()}
    sub_generated = {k: _slice_primitive(v) for k, v in request.generated.items()}

    def _slice_list(x):
        if x is None:
            return None
        return [x[i] for i in indices]

    return RewardRequest(
        primitives=sub_primitives,
        generated=sub_generated,
        metadata=_slice_list(request.metadata),
        prompt_ids=_slice_list(request.prompt_ids),
        sample_ids=_slice_list(request.sample_ids),
        group_ids=_slice_list(request.group_ids),
        reward_types=list(request.reward_types),
        return_components=request.return_components,
        audio_sample_rate=request.audio_sample_rate,
    )


class PerDomainRewardScorer(RewardBackend):
    """Dispatch per-item scoring to a domain-specific inner scorer.

    ``config.scorers`` maps a teacher-domain tag (as it appears in
    ``metadata[i]["teacher_domain"]``) to a fully constructed
    :class:`RewardBackend` — Hydra builds each entry from its own
    ``_target_``. Items without a domain tag, or with a tag unmapped in
    ``scorers``, are reported as failed so the outer reward service can stop
    rather than silently train or monitor against the wrong scorer.

    Because DiffusionOPD's data source yields single-domain batches, the
    common case runs exactly one inner scorer per call. Mixed-domain
    batches are also handled (via :func:`_slice_request`) for safety.
    """

    input_kind = "image"

    def __init__(self, *, config: "PerDomainSpec", base_device: str) -> None:
        super().__init__(model_name="per_domain", batch_size=config.batch_size)
        if not config.scorers:
            raise ValueError(
                "PerDomainRewardScorer requires a non-empty `scorers` dict (domain -> RewardBackend instance)."
            )
        # Hydra has already recursively instantiated each entry to a real
        # RewardBackend (local or remote). We just hold the references.
        self._scorers: Dict[str, RewardBackend] = dict(config.scorers)
        self._base_device = base_device

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        start = time.time()
        bs = request.batch_size
        try:
            import torch

            # Group indices by teacher_domain tag from per-item metadata.
            groups: Dict[str, List[int]] = {}
            unmapped: List[int] = []
            metadata = request.metadata or [None] * bs
            for i in range(bs):
                md = metadata[i] if i < len(metadata) else None
                domain = (md or {}).get("teacher_domain") if isinstance(md, dict) else None
                if not domain or domain not in self._scorers:
                    unmapped.append(i)
                    continue
                groups.setdefault(str(domain), []).append(i)

            total = torch.zeros(bs, dtype=torch.float32)
            component_rewards: Dict[str, List[float]] = {}
            for domain in self._scorers:
                component_rewards[domain] = [float("nan")] * bs

            errors: List[Optional[str]] = [None] * bs
            successes: List[bool] = [True] * bs

            for domain, indices in groups.items():
                scorer = self._scorers[domain]
                sub_req = _slice_request(request, indices)
                resp = scorer.compute_rewards(sub_req)
                rewards = list(resp.rewards)
                if len(rewards) != len(indices):
                    raise RuntimeError(
                        f"PerDomainRewardScorer: inner scorer for domain {domain!r} returned "
                        f"{len(rewards)} rewards for {len(indices)} items."
                    )
                for local_i, global_i in enumerate(indices):
                    total[global_i] = float(rewards[local_i])
                    component_rewards[domain][global_i] = float(rewards[local_i])
                # Propagate per-item success/error signals if inner provided them.
                inner_succ = getattr(resp, "successes", None)
                inner_err = getattr(resp, "errors", None)
                if inner_succ is not None:
                    for local_i, global_i in enumerate(indices):
                        if local_i < len(inner_succ):
                            successes[global_i] = bool(inner_succ[local_i])
                if inner_err is not None:
                    for local_i, global_i in enumerate(indices):
                        if local_i < len(inner_err):
                            errors[global_i] = inner_err[local_i]

            for i in unmapped:
                successes[i] = False
                errors[i] = "PerDomainRewardScorer: item has no matching teacher_domain in config."

            return RewardResponse(
                rewards=total.tolist(),
                component_rewards=component_rewards,
                successes=successes,
                errors=errors,
                compute_time=time.time() - start,
            )
        except Exception as e:
            return RewardResponse(
                rewards=[0.0] * bs,
                successes=[False] * bs,
                errors=[str(e)] * bs,
                compute_time=time.time() - start,
            )

    @property
    def preferred_input_kind(self) -> str:
        return self.input_kind

    def is_available(self) -> bool:
        return all(s.is_available() for s in self._scorers.values())

    def offload(self) -> None:
        for s in self._scorers.values():
            s.offload()

    def onload(self) -> None:
        for s in self._scorers.values():
            s.onload()

    def dispose(self) -> None:
        for s in self._scorers.values():
            s.dispose()


@dataclass
class PerDomainSpec(BaseRewardComponentSpec):
    """Typed config for :class:`PerDomainRewardScorer`.

    ``scorers`` maps a teacher-domain tag (matching the value that the data
    source attaches to ``metadata["teacher_domain"]``) to a
    :class:`RewardBackend`. In yaml each entry carries its own
    ``_target_`` — Hydra recursively instantiates it before the spec itself
    is constructed, so this field holds real backend instances.

    Example yaml::

        config:
          _target_: unirl.reward.local.per_domain.PerDomainSpec
          batch_size: 8
          scorers:
            pickscore:
              _target_: unirl.reward.local.pickscore.PickScoreRewardScorer
              base_device: cuda
              config: {_target_: unirl.reward.local.pickscore.PickScoreSpec, batch_size: 8}
            geneval:
              _target_: unirl.reward.remote.RemoteRewardBackend
              base_device: cpu
              config:
                _target_: unirl.reward.remote.RemoteRewardSpec
                base_url: ${oc.env:REWARD_SERVICE_URL}
                required_rewards: [geneval]
                reward_weights: {geneval: 1.0}
    """

    batch_size: int = 8
    device: str = "auto"
    scorers: Dict[str, RewardBackend] = field(default_factory=dict)
