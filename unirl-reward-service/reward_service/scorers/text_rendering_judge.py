"""TextRenderingJudgeScorer — verbal-feedback → scalar reward via Qwen3-VL 235B.

Wraps a trained **SFT-then-RL error-judge Qwen3-VL-MoE-235B** in vLLM chat form,
scores each (prompt, generated_image) by counting the JSON-structured text-
rendering errors the judge emits, and converts that count into a smooth scalar
via ``exp(-alpha * n_errors)``.

Trainer usage (Flux 2 / SD 3.5 GRPO): each sample's ``item.history[-1] = (prompt,
image)`` — prompt is the ORIGINAL generation prompt (not the judge instruction);
this scorer re-wraps it with the judge instruction internally so the trainer
does not need to know about the judge's prompt format.

Instruction / parsing logic is a **verbatim copy** of the judge's own
training-time pipeline (``error_check_hyimage35.py``); we duplicate it here
(rather than import) so this service is self-contained and does not depend on an
out-of-tree file.
"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any

import torch

from reward_service.scorers._common import (
    build_vllm_llm_kwargs,
    image_to_data_url,
    resolve_model_path,
    split_last_turn,
)
from reward_service.scorers.base import BaseScorer, ScoreItem
from reward_service.scorers.registry import register

logger = logging.getLogger(__name__)


# ── Judge instruct (verbatim from error_check_hyimage35.py:76-90) ──

_JUDGE_INSTRUCTION = (
    "你是一个专业的图像生成文本渲染质检助手。给定一段原始生图 prompt 和一张由生成模型产出的图像，"
    "请只找出 prompt 明确要求但图中没有正确达成的文本渲染错误；不确定或 prompt 未要求的内容不要输出。\n\n"
    "错误类型：\n"
    "- 内容正确性与完整性错误：文字/数字/符号写了什么是否正确完整，包括错字、漏字、多字、乱码、不可读。\n"
    "- 布局与版式质量错误：文字或版式元素在哪里、怎么排，包括位置、对齐、换行、间距、遮挡、分隔线。\n"
    "- 视觉风格忠实度错误：视觉属性是否一致，包括字体、字号、字重、颜色、描边、阴影、材质、水印。\n\n"
    "原子粒度：一条错误 = 一个原始要求 + 一个实际差异 + 一个可定位主体。description 中的错误主体就是 box 要框住的目标；"
    "短文本内部某个字错，也描述完整短文本主体，具体错字放到实际呈现里。\n\n"
    "最终只输出 <answer>...</answer>，其中 <answer> 内必须是合法 JSON，格式如下：\n"
    '{"errors":[{"description":"错误描述","box":[x1,y1,x2,y2]}]}\n\n'
    "description 必须写清错误类型、错误主体、原始要求、实际呈现。box 是错误区域边界框，使用 0-999 归一化坐标，"
    "格式必须是 [x1,y1,x2,y2]，不要写成 [[x1,y1,x2,y2]]。如果没有错误，输出：<answer>{\"errors\":[]}</answer>。\n"
    "---\n"
)
_PROMPT_BLOCK_HEADER = "【原始生图 prompt（生成图像时使用的提示词，作为判定哪些内容被要求出现的依据）】\n"

# Strip a possibly-appended '\n【生成图像】\n<image>' tail from a caller-supplied prompt
# (the training data was pre-formatted this way; user prompts should NOT be but
# stripping is idempotent).
_TAIL_RE = re.compile(r"\n*【生成图像】\s*<image>\s*$")


def _clean_prompt(p: str) -> str:
    return _TAIL_RE.sub("", (p or "")).strip()


def build_instruct_text(original_prompt: str) -> str:
    """Rebuild the exact user-message text the judge was trained on (sans image)."""
    return (
        _JUDGE_INSTRUCTION
        + _PROMPT_BLOCK_HEADER
        + _clean_prompt(original_prompt)
        + "\n\n【生成图像】\n<image>"
    )


# ── Answer parsing (verbatim from error_check_hyimage35.py:130-171) ──

def _coerce_errors(obj: Any) -> list | None:
    if isinstance(obj, dict) and isinstance(obj.get("errors"), list):
        return obj["errors"]
    return None


def parse_errors(text: str) -> tuple[list | None, bool]:
    """Return ``(errors_list_or_None, parse_ok)``.

    Three-tier parser (matches ``verl/utils/reward_score/error_list_judge.py``):
    prefer the last ``<answer>`` block; else strip ``<think>...</think>`` and
    try the whole text; else scan for the last decodable ``{...}`` object with
    an ``errors`` key. Returns ``(None, False)`` if no shape matches.
    """
    if not isinstance(text, str):
        return None, False

    answer_matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if answer_matches:
        try:
            errs = _coerce_errors(json.loads(answer_matches[-1].strip()))
            if errs is not None:
                return errs, True
        except Exception:
            pass

    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    try:
        errs = _coerce_errors(json.loads(stripped))
        if errs is not None:
            return errs, True
    except Exception:
        pass

    starts = [i for i, c in enumerate(stripped) if c == "{"]
    for start in reversed(starts):
        try:
            obj, _ = json.JSONDecoder().raw_decode(stripped[start:])
            errs = _coerce_errors(obj)
            if errs is not None:
                return errs, True
        except Exception:
            continue

    return None, False


# ── Scorer ──

class TextRenderingJudgeScorer(BaseScorer):
    """Text-rendering reward: Qwen3-VL-MoE-235B judge → n_errors → scalar.

    Sub-metrics returned per item:

    - ``text_render_reward``: ``exp(-alpha * n_errors)``, the GRPO training
      signal. Bounded in ``(0, 1]``. Alpha is a config knob (default 0.3).
    - ``n_errors``: raw error count from the judge (monitoring; higher = worse).
    - ``parse_ok``: 1.0 if the JSON parse succeeded, 0.0 otherwise. Watch this
      in wandb — if it drops the reward is uninformative.

    Fails-open on parse errors: ``parse_ok=0``, ``n_errors=0``, and reward is
    set to 0.0 (a safe pessimistic default; GRPO's advantage z-score will treat
    it as a group-relative signal either way).
    """

    name = "text_rendering_judge"
    sub_metric_names = ("text_render_reward", "n_errors", "parse_ok")

    def __init__(
        self,
        model_name: str = "",
        weights_path: str = "",
        tensor_parallel_size: int = 8,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 14336,
        dtype: str = "bfloat16",
        enforce_eager: bool = True,
        swap_space: int = 4,
        max_num_seqs: int = 32,
        trust_remote_code: bool = True,
        limit_mm_per_prompt: dict | None = None,
        extra_llm_kwargs: dict | None = None,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 1.0,
        alpha: float = 0.3,
    ) -> None:
        from vllm import LLM, SamplingParams

        if not (0 < alpha < 10):
            raise ValueError(f"alpha must be in (0, 10) — got {alpha}")
        self._alpha = float(alpha)

        model_path = resolve_model_path(model_name, weights_path)
        if not model_path:
            raise ValueError("TextRenderingJudgeScorer needs `weights_path` or `model_name`.")

        llm_kwargs = build_vllm_llm_kwargs(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
            enforce_eager=enforce_eager,
            swap_space=swap_space,
            max_num_seqs=max_num_seqs,
            trust_remote_code=trust_remote_code,
            limit_mm_per_prompt=limit_mm_per_prompt or {"image": 1},
            extra_llm_kwargs=extra_llm_kwargs,
        )
        logger.info("TextRenderingJudgeScorer: loading vLLM engine from %s (TP=%d)", model_path, tensor_parallel_size)
        self.llm = LLM(**llm_kwargs)
        self.sampling = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens)
        logger.info("TextRenderingJudgeScorer: ready (alpha=%.3f, max_new_tokens=%d)", self._alpha, max_new_tokens)

    @torch.inference_mode()
    def score(self, items: list[ScoreItem]) -> list[dict[str, float]]:
        if not items:
            return []
        texts, images = split_last_turn(items)

        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": build_instruct_text(prompt)},
                        {"type": "image_url", "image_url": {"url": image_to_data_url(image)}},
                    ],
                }
            ]
            for prompt, image in zip(texts, images)
        ]
        outputs = self.llm.chat(conversations, self.sampling)

        results: list[dict[str, float]] = []
        for out in outputs:
            raw_text = out.outputs[0].text if out.outputs else ""
            errs, ok = parse_errors(raw_text)
            n_errors = len(errs) if errs else 0
            if not ok:
                # Judge produced garbled output — safest is 0 reward + parse_ok=0
                # so it stands out in wandb without polluting the advantage.
                results.append({
                    "text_render_reward": 0.0,
                    "n_errors": 0.0,
                    "parse_ok": 0.0,
                })
                continue
            reward = math.exp(-self._alpha * n_errors)
            results.append({
                "text_render_reward": float(reward),
                "n_errors": float(n_errors),
                "parse_ok": 1.0,
            })
        return results


register("text_rendering_judge", TextRenderingJudgeScorer)
