"""Classical GenEval compositional scorer (Mask2Former + CLIP color classification).

Ports the algorithm from ``unirl-reward-service/reward_service/scorers/geneval.py``
into a :class:`LocalRewardBackend` so the trainer can run it in-process. This
matches the reward used by the original DiffusionOPD paper
(``flow_grpo/gen_eval.py``).

Metadata schema — required in ``request.metadata[i]``::

    {
        "tag": "two_object" | "counting" | "colors" | "color_attr" | "position" | ...,
        "include": [{"class": ..., "count": ..., "color"?: ..., "position"?: [rel, target_idx]}, ...],
        "exclude": [{"class": ..., "count": ...}, ...],       # optional
    }

Dependencies (installed by the launcher before handoff):
- ``mmdet==2.28.2`` + ``mmcv-full==1.7.2`` (Mask2Former)
- ``open_clip_torch`` + ``clip_benchmark`` (zero-shot color classification)
- ``numpy<2``
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image, ImageOps

from unirl.reward.base import BaseRewardComponentSpec
from unirl.reward.local.device import resolve_device
from unirl.types.reward import RewardRequest

from .base import LocalRewardBackend

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────

_COLORS: List[str] = [
    "red", "orange", "yellow", "green", "blue",
    "purple", "pink", "brown", "black", "white",
]

# COCO class names used by Mask2Former (indices match its output layout).
_OBJECT_NAMES: List[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "computer mouse", "tv remote", "computer keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

_DEFAULT_CKPT_URL = (
    "https://download.openmmlab.com/mmdetection/v2.0/mask2former/"
    "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/"
    "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth"
)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _compute_iou(box_a, box_b) -> float:
    def _area(box):
        return max(box[2] - box[0] + 1, 0) * max(box[3] - box[1] + 1, 0)

    i = _area([max(box_a[0], box_b[0]), max(box_a[1], box_b[1]),
               min(box_a[2], box_b[2]), min(box_a[3], box_b[3])])
    u = _area(box_a) + _area(box_b) - i
    return i / u if u else 0.0


def _relative_position(obj_a, obj_b, position_threshold: float = 0.1) -> set:
    boxes = np.array([obj_a[0], obj_b[0]])[:, :4].reshape(2, 2, 2)
    center_a, center_b = boxes.mean(axis=-2)
    dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
    offset = center_a - center_b
    revised_offset = np.maximum(np.abs(offset) - position_threshold * (dim_a + dim_b), 0) * np.sign(offset)
    if np.all(np.abs(revised_offset) < 1e-3):
        return set()
    dx, dy = revised_offset / np.linalg.norm(offset)
    relations: set = set()
    if dx < -0.5:
        relations.add("left of")
    if dx > 0.5:
        relations.add("right of")
    if dy < -0.5:
        relations.add("above")
    if dy > 0.5:
        relations.add("below")
    return relations


class _ImageCrops(torch.utils.data.Dataset):
    def __init__(self, image: Image.Image, objects, transform):
        self._image = image.convert("RGB")
        self._blank = Image.new("RGB", image.size, color="#999999")
        self._objects = objects
        self._transform = transform

    def __len__(self):
        return len(self._objects)

    def __getitem__(self, index):
        box, mask = self._objects[index]
        if mask is not None:
            img = Image.composite(self._image, self._blank, Image.fromarray(mask))
        else:
            img = self._image
        img = img.crop(box[:4])
        return (self._transform(img), 0)


def _resolve_mmdet_config(explicit: str) -> str:
    """Resolve the Mask2Former config path (mmdet 3.x layout).

    If ``explicit`` is a readable file, use it. Otherwise search a few known
    mmdet-shipped config filenames under ``<mmdet>/configs/mask2former/``
    (naming changed between 2.x and 3.x releases).
    """
    if explicit and os.path.isfile(explicit):
        return explicit
    import mmdet  # noqa: F401 — checked so the fallback path is meaningful

    base = os.path.join(os.path.dirname(os.path.dirname(mmdet.__file__)), "configs/mask2former")
    if not os.path.isdir(base):
        # Some installs bundle configs alongside the package rather than one dir up.
        alt = os.path.join(os.path.dirname(mmdet.__file__), ".mim/configs/mask2former")
        if os.path.isdir(alt):
            base = alt

    candidates = [
        # mmdet 3.x naming
        "mask2former_swin-s_8xb2-lsj-50e_coco.py",
        "mask2former_swin-s_8xb2-lsj-50e_coco-panoptic.py",
        # mmdet 2.x naming (in case someone pins the older stack)
        "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py",
    ]
    for name in candidates:
        p = os.path.join(base, name)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"GenEval scorer: cannot find a Mask2Former config under {base!r}. "
        f"Tried {candidates}. Pass ``mmdet_config`` explicitly or ensure "
        f"``mmdet`` was installed with its ``configs/`` shipped."
    )


def _resolve_mmdet_ckpt(explicit: str) -> str:
    """Locate the Mask2Former checkpoint, downloading it if missing.

    Priority:
      1. ``explicit`` if given and readable.
      2. ``$MASK2FORMER_CKPT`` env var.
      3. Cached under ``$MASK2FORMER_CACHE_DIR`` (default ``/dev/shm/geneval``).
      4. Download from openmmlab.
    """
    fname = "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth"
    candidates: List[str] = []
    if explicit:
        candidates.append(explicit)
    env_ckpt = os.environ.get("MASK2FORMER_CKPT")
    if env_ckpt:
        candidates.append(env_ckpt)
    cache_dir = os.environ.get("MASK2FORMER_CACHE_DIR", "/dev/shm/geneval")
    candidates.append(os.path.join(cache_dir, fname))

    for path in candidates:
        if path and os.path.isfile(path):
            return path

    target = candidates[-1]
    os.makedirs(os.path.dirname(target), exist_ok=True)
    logger.info("GenEval: downloading Mask2Former ckpt to %s", target)
    import urllib.request

    tmp = target + ".partial"
    with urllib.request.urlopen(_DEFAULT_CKPT_URL, timeout=600) as resp, open(tmp, "wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.rename(tmp, target)
    return target


# ── Scorer ──────────────────────────────────────────────────────────────────

class GenEvalRewardScorer(LocalRewardBackend):
    """Classical GenEval compositional reward: Mask2Former + CLIP color check.

    ``score_type``:
      - ``"score"`` (default): continuous reward that penalizes count deviation.
      - ``"strict"``: binary 1.0 / 0.0 based on all constraints being exactly met.
    """

    canonical_model_name = "geneval"

    def __init__(self, *, config: "GenEvalSpec", base_device: str) -> None:
        self._mmdet_config = config.mmdet_config
        self._mmdet_ckpt = config.mmdet_ckpt
        self._clip_arch = config.clip_arch
        self._clip_pretrained = config.clip_pretrained
        self._score_type = config.score_type
        self._threshold = float(config.threshold)
        self._counting_threshold = float(config.counting_threshold)
        self._max_objects = int(config.max_objects)
        self._nms_threshold = float(config.nms_threshold)
        self._position_threshold = float(config.position_threshold)
        if self._score_type not in ("strict", "score"):
            raise ValueError(f"score_type must be 'strict' or 'score', got {self._score_type!r}")

        super().__init__(device=resolve_device(config.device, base_device), batch_size=config.batch_size)

        self._object_detector: Any = None
        self._clip_model: Any = None
        self._clip_transform: Any = None
        self._clip_tokenizer: Any = None
        self._color_classifiers: Dict[str, Any] = {}

    def _load_model(self) -> None:
        try:
            from mmdet.apis import init_detector  # noqa: F401
            import open_clip  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "GenEvalRewardScorer requires mmdet, mmcv-full, open_clip_torch, and "
                "clip_benchmark. Install in the launcher before handoff."
            ) from e

        from mmdet.apis import init_detector
        import open_clip

        cfg = _resolve_mmdet_config(self._mmdet_config)
        ckpt = _resolve_mmdet_ckpt(self._mmdet_ckpt)
        logger.info("GenEval: loading Mask2Former (cfg=%s, ckpt=%s, device=%s)", cfg, ckpt, self.device)
        self._object_detector = init_detector(cfg, ckpt, device=str(self.device))

        logger.info("GenEval: loading CLIP (%s, %s)", self._clip_arch, self._clip_pretrained)
        self._clip_model, _, self._clip_transform = open_clip.create_model_and_transforms(
            self._clip_arch, pretrained=self._clip_pretrained, device=str(self.device)
        )
        self._clip_tokenizer = open_clip.get_tokenizer(self._clip_arch)

        # Sentinel for base class' _is_loaded check.
        self.model = self._object_detector

    # -- Color classification ------------------------------------------------

    def _get_color_classifier(self, classname: str):
        if classname not in self._color_classifiers:
            from clip_benchmark.metrics import zeroshot_classification as zsc

            self._color_classifiers[classname] = zsc.zero_shot_classifier(
                self._clip_model,
                self._clip_tokenizer,
                _COLORS,
                [
                    f"a photo of a {{c}} {classname}",
                    f"a photo of a {{c}}-colored {classname}",
                    f"a photo of a {{c}} object",
                ],
                str(self.device),
            )
        return self._color_classifiers[classname]

    def _classify_colors(self, image: Image.Image, bboxes: List[tuple], classname: str) -> List[str]:
        from clip_benchmark.metrics import zeroshot_classification as zsc

        clf = self._get_color_classifier(classname)
        loader = torch.utils.data.DataLoader(
            _ImageCrops(image, bboxes, self._clip_transform),
            batch_size=16,
            num_workers=0,
        )
        with torch.no_grad():
            pred, _ = zsc.run_classification(self._clip_model, clf, loader, str(self.device))
            return [_COLORS[idx.item()] for idx in pred.argmax(1)]

    # -- Evaluation ----------------------------------------------------------

    def _evaluate_strict(self, image, objects, metadata):
        correct = True
        matched_groups: List[Any] = []
        for req in metadata.get("include", []):
            classname = req["class"]
            matched = True
            found = objects.get(classname, [])[: req["count"]]
            if len(found) < req["count"]:
                correct = matched = False
            else:
                if "color" in req:
                    colors = self._classify_colors(image, found, classname)
                    if colors.count(req["color"]) < req["count"]:
                        correct = matched = False
                if "position" in req and matched:
                    expected_rel, target_group = req["position"]
                    if matched_groups[target_group] is None:
                        correct = matched = False
                    else:
                        for obj in found:
                            for target_obj in matched_groups[target_group]:
                                if expected_rel not in _relative_position(obj, target_obj, self._position_threshold):
                                    correct = matched = False
                                    break
                            if not matched:
                                break
            matched_groups.append(found if matched else None)
        for req in metadata.get("exclude", []):
            if len(objects.get(req["class"], [])) >= req["count"]:
                correct = False
        return correct

    def _evaluate_reward(self, image, objects, metadata) -> float:
        matched_groups: List[Any] = []
        rewards: List[float] = []
        for req in metadata.get("include", []):
            classname = req["class"]
            matched = True
            found = objects.get(classname, [])
            rewards.append(1 - abs(req["count"] - len(found)) / req["count"])
            if len(found) != req["count"]:
                matched = False
                if "color" in req or "position" in req:
                    rewards.append(0.0)
            else:
                if "color" in req:
                    colors = self._classify_colors(image, found, classname)
                    rewards.append(1 - abs(req["count"] - colors.count(req["color"])) / req["count"])
                    if colors.count(req["color"]) != req["count"]:
                        matched = False
                if "position" in req and matched:
                    expected_rel, target_group = req["position"]
                    if matched_groups[target_group] is None:
                        matched = False
                        rewards.append(0.0)
                    else:
                        pos_ok = True
                        for obj in found:
                            for target_obj in matched_groups[target_group]:
                                if expected_rel not in _relative_position(obj, target_obj, self._position_threshold):
                                    pos_ok = False
                                    break
                            if not pos_ok:
                                break
                        rewards.append(1.0 if pos_ok else 0.0)
                        matched = matched and pos_ok
            matched_groups.append(found if matched else None)
        return float(sum(rewards) / len(rewards)) if rewards else 0.0

    # -- Detection -----------------------------------------------------------

    def _detect_and_evaluate(self, image: Image.Image, metadata: dict) -> float:
        """Run Mask2Former detection (mmdet 3.x API) and score against metadata.

        mmdet 3.x's ``inference_detector`` returns a ``DetDataSample`` with a
        ``pred_instances`` structure holding tensors for bboxes / scores /
        labels / (optional) masks — different from mmdet 2.x's
        ``(bbox_list_per_class, segm_list_per_class)`` tuple. We re-bucket
        instances by class label to keep the downstream evaluation logic (which
        expects a ``{classname: [(box_with_score_as_5th_col, mask_or_None)]}``
        dict) unchanged.
        """
        from mmdet.apis import inference_detector

        result = inference_detector(self._object_detector, np.array(image))
        pred = result.pred_instances
        bboxes = pred.bboxes.detach().cpu().numpy()   # [N, 4] xyxy
        scores = pred.scores.detach().cpu().numpy()   # [N]
        labels = pred.labels.detach().cpu().numpy()   # [N]
        masks = None
        if hasattr(pred, "masks") and pred.masks is not None:
            masks = pred.masks.detach().cpu().numpy() # [N, H, W] bool

        image = ImageOps.exif_transpose(image)
        tag = metadata.get("tag", "")
        conf_thr = self._counting_threshold if tag == "counting" else self._threshold

        detected: Dict[str, List[tuple]] = {}
        for cls_idx, classname in enumerate(_OBJECT_NAMES):
            cls_sel = np.where(labels == cls_idx)[0]
            if cls_sel.size == 0:
                continue
            cls_boxes = bboxes[cls_sel]
            cls_scores = scores[cls_sel]
            cls_segms = masks[cls_sel] if masks is not None else None
            # Append score as 5th col so downstream helpers (_compute_iou,
            # _relative_position) that only touch box[:4] still work, and
            # _evaluate_* that never read box[4] are unaffected.
            cls_boxes5 = np.concatenate([cls_boxes, cls_scores[:, None]], axis=1)

            ordering = np.argsort(cls_scores)[::-1]
            ordering = ordering[cls_scores[ordering] > conf_thr]
            ordering = ordering[: self._max_objects].tolist()
            entries: List[tuple] = []
            while ordering:
                max_obj = ordering.pop(0)
                entries.append((
                    cls_boxes5[max_obj],
                    None if cls_segms is None else cls_segms[max_obj],
                ))
                ordering = [
                    o for o in ordering
                    if self._nms_threshold == 1.0
                    or _compute_iou(cls_boxes5[max_obj], cls_boxes5[o]) < self._nms_threshold
                ]
            if entries:
                detected[classname] = entries

        if self._score_type == "strict":
            return 1.0 if self._evaluate_strict(image, detected, metadata) else 0.0
        return self._evaluate_reward(image, detected, metadata)

    @torch.inference_mode()
    def _compute_model_rewards(self, request: RewardRequest) -> List[float]:
        images = request.images
        if images is None:
            raise ValueError("GenEvalRewardScorer requires generated images in RewardRequest.generated['image']")

        metadata = request.metadata or []
        rewards: List[float] = []
        for i, image in enumerate(images):
            md = metadata[i] if i < len(metadata) else None
            if not md or not isinstance(md, dict) or "include" not in md:
                logger.warning(
                    "GenEval: item %d has no include metadata; returning 0.0. "
                    "Provide {tag, include, exclude?} via request.metadata.",
                    i,
                )
                rewards.append(0.0)
                continue
            rewards.append(self._detect_and_evaluate(image, md))
        return rewards


@dataclass
class GenEvalSpec(BaseRewardComponentSpec):
    """Typed config for the classical GenEval reward.

    ``mmdet_config`` / ``mmdet_ckpt`` may be left empty — the scorer then falls
    back to the config that ships with the installed ``mmdet`` and downloads
    the ckpt to ``$MASK2FORMER_CACHE_DIR`` (default ``/dev/shm/geneval``) on
    first run.
    """

    batch_size: int = 8
    device: str = "auto"
    mmdet_config: str = ""
    mmdet_ckpt: str = ""
    clip_arch: str = "ViT-L-14"
    clip_pretrained: str = "openai"
    score_type: str = "score"
    threshold: float = 0.3
    counting_threshold: float = 0.9
    max_objects: int = 16
    nms_threshold: float = 1.0
    position_threshold: float = 0.1
