"""Qwen3-VL text-rendering judge: instruction text + response parser (see ../README.md)."""

from __future__ import annotations

import json
import re
from typing import Any

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
    '格式必须是 [x1,y1,x2,y2]，不要写成 [[x1,y1,x2,y2]]。如果没有错误，输出：<answer>{"errors":[]}</answer>。\n'
    "---\n"
)
_PROMPT_BLOCK_HEADER = "【原始生图 prompt（生成图像时使用的提示词，作为判定哪些内容被要求出现的依据）】\n"

_TAIL_RE = re.compile(r"\n*【生成图像】\s*<image>\s*$")


def _clean_prompt(p: str) -> str:
    return _TAIL_RE.sub("", (p or "")).strip()


def build_instruct_text(original_prompt: str) -> str:
    """Rebuild the exact user-message text the judge was trained on (sans image)."""
    return _JUDGE_INSTRUCTION + _PROMPT_BLOCK_HEADER + _clean_prompt(original_prompt) + "\n\n【生成图像】\n<image>"


# ── Answer parsing (verbatim from error_check_hyimage35.py:130-171) ──


def _coerce_errors(obj: Any) -> list | None:
    if isinstance(obj, dict) and isinstance(obj.get("errors"), list):
        return obj["errors"]
    return None


def parse_errors(text: str) -> tuple[list | None, bool]:
    """Three-tier parse of the judge's answer; returns ``(errors_or_None, parse_ok)`` (see ../README.md)."""
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


# ── Severity: how WRONG each error is, not just that it happened ──
#
# The judge's own description contract (see the instruction above) is
# "错误类型：错误主体【X】，原始要求【Y】，实际呈现【Z】。", and for content
# errors Y/Z quote the required and the actually-rendered text. That pair is a
# free severity signal that ``len(errors)`` throws away: "Chain"->"Clain" (one
# letter) and "Chain"->"" (missing entirely) are not equally bad.
#
# Only 内容正确性与完整性 errors expose a comparable pair. Layout and style
# errors describe a *relation* ("位于表头行" vs "实际位于左侧列"), which is not a
# misspelling of anything — for those, severity is undefined and callers fall
# back to a full unit of cost.

_RE_REQUIRED = re.compile(r"原始要求【(.*?)】")
_RE_ACTUAL = re.compile(r"实际呈现【(.*?)】")
# The judge quotes literals with 「」 (or occasionally 『』).
_RE_QUOTED = re.compile(r"[「『]([^」』]*)[」』]")

# What the judge SAW, e.g. 实际呈现为「Clain」 / 实际显示为畸形字符「EIFI」. When this
# fires, the quoted run is the rendered text and an edit distance is meaningful.
# Must be tried before the absence/garble markers below, because descriptions
# routinely carry both ("实际呈现为「Tralt Y36」，未正确显示「Start Gate」") and the
# thing that was actually drawn is the better evidence.
_RE_RENDERED_AS = re.compile(r"(?:呈现|显示|渲染)为[^「『]{0,8}[「『]([^」』]*)[」』]")
# "Nothing was drawn" — 未出现 / 未正确显示 / 缺失 / 不可读 …
_RE_ABSENT = re.compile(r"未(?:正确)?(?:出现|呈现|显示|渲染|识别)|缺失|没有出现|不可读")
# "Something was drawn but it is not legible text" — 乱码 / 畸形 / 无法识别 …
_RE_GARBLED = re.compile(r"乱码|畸形|无法正确识别|无法识别|无法辨认")

# The judge's three fixed categories (see the instruction above). Only the
# content category compares *spellings*; the other two compare a relation, so
# an edit distance over their quoted text is meaningless.
CATEGORY_CONTENT = "内容正确性与完整性错误"
CATEGORY_LAYOUT = "布局与版式质量错误"
CATEGORY_STYLE = "视觉风格忠实度错误"
CATEGORIES = (CATEGORY_CONTENT, CATEGORY_LAYOUT, CATEGORY_STYLE)


def error_category(description: str) -> str | None:
    """Which of the judge's three categories this description declares."""
    if not isinstance(description, str):
        return None
    head = description[:40]
    for c in CATEGORIES:
        if c in head:
            return c
    return None


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def error_severity(description: str) -> float | None:
    """Normalised required-vs-actual edit distance in ``[0, 1]``; ``None`` (not 0) if undefined (see ../README.md)."""
    if error_category(description) != CATEGORY_CONTENT:
        return None
    req_m, act_m = _RE_REQUIRED.search(description), _RE_ACTUAL.search(description)
    if not (req_m and act_m):
        return None
    req_q = _RE_QUOTED.findall(req_m.group(1))
    if not req_q:
        return None
    req = req_q[0]
    actual_side = act_m.group(1)

    rendered = _RE_RENDERED_AS.search(actual_side)
    if rendered:
        act = rendered.group(1)
    elif _RE_ABSENT.search(actual_side) or _RE_GARBLED.search(actual_side):
        # Nothing legible was drawn; the requirement was missed in full.
        return 1.0
    else:
        act_q = _RE_QUOTED.findall(actual_side)
        act = act_q[0] if act_q else ""

    denom = max(len(req), len(act))
    if denom == 0:
        return None
    severity = min(1.0, _levenshtein(req, act) / denom)
    if severity == 0.0:
        # The judge reported an error yet the two strings match, so the defect
        # lives in wording we did not model (e.g. 实际呈现为「$5.80」，字形畸形).
        # Undefined, not free — a reported error must never be invisible to GRPO.
        return None
    return severity


def weighted_error_cost(
    errors: list,
    *,
    undefined_severity_cost: float = 1.0,
    layout_weight: float | None = None,
    style_weight: float | None = None,
    garbled_cost: float | None = None,
) -> float:
    """Sum per-error severity into a continuous cost replacing ``len(errors)``; rescale ``alpha`` (see ../README.md)."""
    w_layout = undefined_severity_cost if layout_weight is None else layout_weight
    w_style = undefined_severity_cost if style_weight is None else style_weight
    total = 0.0
    for e in errors or []:
        if not isinstance(e, dict):
            continue
        desc = e.get("description", "")
        cat = error_category(desc)
        if cat == CATEGORY_LAYOUT:
            total += w_layout
        elif cat == CATEGORY_STYLE:
            total += w_style
        else:
            s = error_severity(desc)
            # Unparseable / uncategorised content errors keep the neutral cost
            # rather than becoming free.
            if s is None:
                total += undefined_severity_cost
                continue
            if garbled_cost is not None and s >= 1.0:
                # Discount ONLY "drew something illegible". Reading the 实际呈现
                # side matters: a description marking BOTH absent and garbled
                # means nothing legible was drawn, which is the case we want to
                # keep expensive.
                act_m = _RE_ACTUAL.search(desc)
                act = act_m.group(1) if act_m else desc
                if _RE_GARBLED.search(act) and not _RE_ABSENT.search(act):
                    s = float(garbled_cost)
            total += s
    return total


__all__ = [
    "build_instruct_text",
    "parse_errors",
    "error_category",
    "error_severity",
    "weighted_error_cost",
    "CATEGORIES",
    "CATEGORY_CONTENT",
    "CATEGORY_LAYOUT",
    "CATEGORY_STYLE",
]
