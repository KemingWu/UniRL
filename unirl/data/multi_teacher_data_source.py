"""Multi-teacher data source for DiffusionOPD.

The original DiffusionOPD paper uses ONE dataloader per teacher — each teacher
supervises only its own domain's prompts (pickscore→pickscore, ocr→ocr,
geneval→geneval). Merging domains into a single stream breaks this: 2/3 of the
time a teacher gets prompts it was never trained on, and its transition mean
becomes noise that pulls the student off.

This data source composes N inner ``TextPromptDataset``+``DataLoader`` pairs
(one per teacher) and cycles between them per ``get_samples()`` call. Each
returned batch is entirely from one teacher's domain, in the same order as the
``teachers`` list — so the DiffusionOPD algorithm's round-robin counter stays
in sync automatically.

Every item is tagged with ``metadata["teacher_domain"] = <name>`` so downstream
components (currently the per-domain reward scorer) can route by domain.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterator, List, Optional

import torch
from torch.utils.data import DataLoader

from unirl.types.primitives import Texts
from unirl.types.prompts import RolloutInputs

from .datasets import TextPromptDataset, normalize_prompt_example

logger = logging.getLogger(__name__)


class MultiTeacherRLDataSource:
    """Cycles per-teacher prompt streams so each rollout batch is single-domain.

    Config (``args.run``)::

        teachers:
          - name: pickscore
            data_path: datasets/pickscore/train.txt
            eval_data_path: datasets/pickscore/test.txt   # optional
          - name: ocr
            data_path: datasets/ocr/train.txt
            eval_data_path: datasets/ocr/test.txt
          - name: geneval
            data_path: datasets/geneval/train_metadata.jsonl
            eval_data_path: datasets/geneval/test_metadata.jsonl
        seed: 42

    The ``prompts_per_rollout`` from ``args.algorithm`` is the per-batch size —
    identical semantics to :class:`MultimodalRLDataSource`.
    """

    def __init__(self, args):
        self.args = args
        run_cfg = args.run

        teachers = list(run_cfg.teachers)
        if not teachers:
            raise ValueError("MultiTeacherRLDataSource requires at least one teacher in run.teachers.")

        self.teachers: List[Dict[str, Any]] = []
        for t in teachers:
            name = t.get("name") if isinstance(t, dict) else getattr(t, "name", None)
            data_path = t.get("data_path") if isinstance(t, dict) else getattr(t, "data_path", None)
            eval_data_path = (
                t.get("eval_data_path") if isinstance(t, dict) else getattr(t, "eval_data_path", None)
            )
            if not name or not data_path:
                raise ValueError(
                    f"Each teacher entry must define 'name' and 'data_path' (got {dict(t) if hasattr(t, 'items') else t!r})."
                )
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Teacher '{name}' data_path not found: {data_path}")
            if eval_data_path and not os.path.exists(eval_data_path):
                raise FileNotFoundError(f"Teacher '{name}' eval_data_path not found: {eval_data_path}")
            self.teachers.append(
                {"name": str(name), "data_path": str(data_path), "eval_data_path": str(eval_data_path) if eval_data_path else None}
            )

        # Duplicate name detection: algorithm side keys teachers by name.
        names = [t["name"] for t in self.teachers]
        if len(set(names)) != len(names):
            raise ValueError(f"Teacher names must be unique, got {names}")

        self.seed = getattr(run_cfg, "seed", None)
        self.prompts_per_rollout = int(args.algorithm.prompts_per_rollout)
        self.drop_last = True

        # One generator per teacher — independent shuffles, deterministic when
        # seed is set. seed=None means non-reproducible (matches base class).
        self._shuffle_generators: List[torch.Generator] = []
        for i, _ in enumerate(self.teachers):
            g = torch.Generator()
            if self.seed is None:
                g.manual_seed(int.from_bytes(os.urandom(8), "big") & 0x7FFFFFFF)
            else:
                # Different per-teacher seed so pickscore, ocr, geneval don't
                # walk in lockstep.
                g.manual_seed(int(self.seed) + i)
            self._shuffle_generators.append(g)

        self._datasets: List[TextPromptDataset] = []
        self._dataloaders: List[DataLoader] = []
        self._iters: List[Optional[Iterator]] = []
        self._eval_datasets: List[Optional[TextPromptDataset]] = []
        self._eval_ready = False
        self._iter_counter = 0

        self._init_datasets()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _init_datasets(self) -> None:
        for i, tc in enumerate(self.teachers):
            ds = TextPromptDataset(file_path=tc["data_path"])
            if len(ds) < self.prompts_per_rollout:
                raise ValueError(
                    f"Teacher '{tc['name']}' dataset has {len(ds)} prompts < prompts_per_rollout={self.prompts_per_rollout}."
                )
            self._datasets.append(ds)
            loader = DataLoader(
                ds,
                batch_size=self.prompts_per_rollout,
                shuffle=True,
                generator=self._shuffle_generators[i],
                num_workers=0,
                collate_fn=self._make_collate(tc["name"]),
                drop_last=True,
            )
            self._dataloaders.append(loader)
            self._iters.append(iter(loader))
            logger.info(
                "MultiTeacherRLDataSource: teacher '%s' — %d prompts from %s",
                tc["name"],
                len(ds),
                tc["data_path"],
            )

    def _make_collate(self, teacher_name: str):
        """Build a collate fn that tags each item with ``teacher_domain``."""

        def _collate(batch: List[Dict[str, Any]]) -> RolloutInputs:
            prompts = [item["prompt"] for item in batch]
            prompt_ids: List[str] = []
            for idx, item in enumerate(batch):
                pid = item.get("prompt_id")
                prompt_ids.append(str(pid) if pid else f"{teacher_name}:{idx}")
            sample_ids = [f"prompt:{pid}:sample:0" for pid in prompt_ids]

            metadata_list: List[Optional[Dict[str, Any]]] = []
            for item in batch:
                md = dict(item.get("metadata") or {})
                md["teacher_domain"] = teacher_name
                metadata_list.append(md)

            primitives: Dict[str, Any] = {"text": Texts(texts=prompts)}
            return RolloutInputs(
                primitives=primitives,
                sample_ids=sample_ids,
                group_ids=list(prompt_ids),
                metadata=metadata_list,
            )

        return _collate

    # ------------------------------------------------------------------
    # Train sampling — round-robin across teachers
    # ------------------------------------------------------------------

    def get_samples(self, batch_size: int) -> RolloutInputs:
        # batch_size is nominal (mirrors MultimodalRLDataSource — the actual
        # size is prompts_per_rollout, set at construction).
        teacher_idx = self._iter_counter % len(self.teachers)
        self._iter_counter += 1
        try:
            batch = next(self._iters[teacher_idx])
        except StopIteration:
            self._iters[teacher_idx] = iter(self._dataloaders[teacher_idx])
            batch = next(self._iters[teacher_idx])
        return batch

    @property
    def num_prompts(self) -> int:
        return sum(len(ds) for ds in self._datasets)

    # ------------------------------------------------------------------
    # Eval — deterministic pass over each teacher's eval set in order
    # ------------------------------------------------------------------

    def _ensure_eval_datasets(self) -> None:
        if self._eval_ready:
            return
        self._eval_datasets = []
        for tc in self.teachers:
            eval_path = tc["eval_data_path"] or tc["data_path"]
            ds = TextPromptDataset(file_path=eval_path)
            self._eval_datasets.append(ds)
        self._eval_ready = True

    def _example_to_batch(
        self, prompt_examples: List[Dict[str, Any]], teacher_name: str
    ) -> RolloutInputs:
        prompts = [ex["prompt"] for ex in prompt_examples]
        prompt_ids = [str(ex.get("prompt_id") or f"eval:{teacher_name}:{i}") for i, ex in enumerate(prompt_examples)]
        sample_ids = [f"prompt:{pid}:sample:0" for pid in prompt_ids]
        metadata_list: List[Optional[Dict[str, Any]]] = []
        for ex in prompt_examples:
            md = dict(ex.get("metadata") or {})
            md["teacher_domain"] = teacher_name
            metadata_list.append(md)
        primitives: Dict[str, Any] = {"text": Texts(texts=prompts)}
        return RolloutInputs(
            primitives=primitives,
            sample_ids=sample_ids,
            group_ids=list(prompt_ids),
            metadata=metadata_list,
        )

    def iter_eval_batches(
        self,
        batch_size: int,
        *,
        eval_num_prompts: int = -1,
    ) -> Iterator[RolloutInputs]:
        batch_size = int(batch_size)
        eval_num_prompts = int(eval_num_prompts)
        if batch_size <= 0 or eval_num_prompts == 0:
            return
        self._ensure_eval_datasets()

        # Split eval budget evenly across teachers; ``-1`` means "full set per
        # teacher". This mirrors the paper's eval-per-teacher semantics.
        per_teacher_limit = eval_num_prompts if eval_num_prompts < 0 else max(
            1, eval_num_prompts // len(self.teachers)
        )

        for tc, ds in zip(self.teachers, self._eval_datasets):
            if ds is None:
                continue
            total = len(ds)
            limit = total if per_teacher_limit < 0 else min(per_teacher_limit, total)
            for start in range(0, limit, batch_size):
                end = min(start + batch_size, limit)
                prompt_examples = [
                    normalize_prompt_example(
                        ds.get_prompt_example(idx),
                        default_prompt_id=f"eval:{tc['name']}:{idx}",
                    )
                    for idx in range(start, end)
                ]
                yield self._example_to_batch(prompt_examples, tc["name"])

    def get_eval_samples(self, batch_size: int) -> RolloutInputs:
        batch_size = int(batch_size)
        if batch_size <= 0:
            return self._example_to_batch([], self.teachers[0]["name"])
        return next(
            self.iter_eval_batches(batch_size),
            self._example_to_batch([], self.teachers[0]["name"]),
        )
