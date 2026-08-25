"""Remote reward backend: an HTTP client for the RewardService server."""

from __future__ import annotations

import base64
import hashlib
import io
import itertools
import json
import logging
import math
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import requests as http_requests
import torch
from PIL import Image
from requests.adapters import HTTPAdapter

from unirl.config.require import require
from unirl.reward._judges.qwen3vl_text_render import (
    build_instruct_text,
    parse_errors,
    weighted_error_cost,
)
from unirl.reward.base import BaseRewardComponentSpec, RewardBackend
from unirl.types.reward import RewardRequest, RewardResponse

logger = logging.getLogger(__name__)


def _pil_from_tensor(tensor: torch.Tensor) -> Image.Image:
    """Convert a CHW float or uint8 tensor to a PIL RGB image."""
    from torchvision.transforms.functional import to_pil_image

    tensor = tensor.detach().cpu()
    if tensor.is_floating_point():
        tensor = tensor.clamp(0.0, 1.0)
    return to_pil_image(tensor)


def _encode_image_b64(
    image: Union[Image.Image, torch.Tensor],
    image_format: str = "JPEG",
    quality: int = 95,
) -> str:
    """Encode an image to a base64 string for the RewardService wire format."""
    if isinstance(image, torch.Tensor):
        image = _pil_from_tensor(image)
    if image.mode != "RGB":
        image = image.convert("RGB")

    buf = io.BytesIO()
    save_kwargs: dict = {"format": image_format}
    if image_format.upper() == "JPEG":
        save_kwargs["quality"] = quality
    image.save(buf, **save_kwargs)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _encode_video_b64(
    video: torch.Tensor,
    fps: int = 8,
) -> str:
    """Encode a video tensor ``(C, T, H, W)`` to a base64 mp4 string."""
    import tempfile

    from diffusers.utils import export_to_video
    from PIL import Image as _PIL_Image

    v = video.detach().cpu()
    if v.dim() == 5:
        v = v.squeeze(0)
    if v.dim() != 4:
        raise ValueError(f"Expected 4D (C, T, H, W) video tensor, got shape {tuple(v.shape)}.")

    if v.is_floating_point():
        v = v.clamp(0.0, 1.0)
    frames = []
    for t in range(v.shape[1]):
        frame = v[:, t, :, :]
        if frame.is_floating_point():
            frame = (frame * 255).byte()
        frame_np = frame.permute(1, 2, 0).numpy()
        frames.append(_PIL_Image.fromarray(frame_np))

    tmp = tempfile.NamedTemporaryFile(prefix="reward_svc_", suffix=".mp4", delete=False)
    tmp.close()
    import os

    try:
        export_to_video(frames, tmp.name, fps=fps)
        with open(tmp.name, "rb") as f:
            video_bytes = f.read()
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    return base64.b64encode(video_bytes).decode("ascii")


def _optional_rank() -> Optional[int]:
    raw = os.environ.get("RANK")
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _payload_fingerprint(*, media_fingerprint: str, prompt: str, metadata: Any) -> str:
    digest = hashlib.sha256()
    digest.update(media_fingerprint.encode("ascii"))
    digest.update(b"\0")
    digest.update(prompt.encode("utf-8"))
    digest.update(b"\0")
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    return digest.hexdigest()


def _wire_identity(
    request: RewardRequest,
    index: int,
    *,
    required_rewards: List[str],
    expected_scorer_version: Optional[str],
    payload_fingerprint: str,
) -> Dict[str, Any]:
    sample_id = request.sample_ids[index] if request.sample_ids and index < len(request.sample_ids) else None
    group_id = request.group_ids[index] if request.group_ids and index < len(request.group_ids) else None
    metadata = request.metadata[index] if request.metadata and index < len(request.metadata) else None
    policy_version = metadata.get("policy_version") if isinstance(metadata, dict) else None
    request_id = str(sample_id or uuid.uuid4())
    digest_input = json.dumps(
        {
            "protocol": "1",
            "request_id": request_id,
            "required_rewards": required_rewards,
            "policy_version": policy_version,
            "scorer_version": expected_scorer_version,
            "payload_fingerprint": payload_fingerprint,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "request_id": request_id,
        "sample_id": sample_id,
        "group_id": group_id,
        "source_rank": _optional_rank(),
        "policy_version": policy_version if isinstance(policy_version, int) else None,
        "scorer_version": expected_scorer_version,
        "idempotency_key": hashlib.sha256(digest_input.encode("utf-8")).hexdigest(),
    }


class RemoteRewardBackend(RewardBackend):
    """HTTP client backend for the remote RewardService ``POST /score`` endpoint."""

    _REDUCE_STRATEGIES = {"first", "mean", "max"}
    _AGGREGATION_METHODS = {"weighted_sum", "mean", "min", "max"}

    def __init__(self, *, config: "RemoteRewardSpec", base_device: str) -> None:
        del base_device
        super().__init__(
            model_name="reward_service",
            batch_size=config.batch_size,
            timeout=config.timeout,
        )
        self.base_url = config.base_url.rstrip("/")
        self.required_rewards = list(config.required_rewards)
        self.reward_weights = dict(config.reward_weights or {})
        self.max_retries = config.max_retries
        self.retry_delay = config.retry_delay
        self.request_batch_size = config.request_batch_size
        self.require_identity_echo = config.require_identity_echo
        self.expected_scorer_version = config.expected_scorer_version
        self.sub_metric_reduce = config.sub_metric_reduce
        self.image_format = config.image_format
        self.image_quality = config.image_quality
        self.raise_on_failure = config.raise_on_failure
        self.aggregation_method = config.aggregation_method
        self.video_fps = config.video_fps
        self.input_kind = config.input_kind

        self._remote_rewards_validated = False

        self._session = http_requests.Session()
        self._session.trust_env = False

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        """Convert a UniRL request, call the remote service, and"""
        start = time.time()
        if request.is_video:
            return self._compute_video_rewards(request, start)

        bs = request.batch_size
        try:
            payload = self._build_score_payload(request)
            raw = self._post_score_requests(payload["requests"])
            return self._parse_score_response(raw, bs, time.time() - start)
        except Exception:
            if self.raise_on_failure:
                raise
            logger.exception("RemoteRewardBackend.compute_rewards failed (degraded mode)")
            return RewardResponse(
                rewards=[0.0] * bs,
                successes=[False] * bs,
                errors=["RemoteRewardBackend failure (see logs)"] * bs,
                compute_time=time.time() - start,
            )

    def is_available(self) -> bool:
        """Ping ``/health``; ``True`` iff the server is reachable."""
        try:
            resp = self._session.get(
                f"{self.base_url}/health",
                timeout=5.0,
            )
        except http_requests.exceptions.RequestException:
            return False
        if resp.status_code != 200:
            return False
        self._validate_required_rewards_once(resp)
        return True

    def _validate_required_rewards_once(self, health_resp: http_requests.Response) -> None:
        """One-shot: ``raise ValueError`` if any required reward is not in the"""
        if self._remote_rewards_validated:
            return

        try:
            body = health_resp.json()
        except ValueError as e:
            raise ValueError(f"RemoteRewardBackend: /health at {self.base_url} returned non-JSON body.") from e

        if not isinstance(body, dict) or not isinstance(body.get("rewards"), dict):
            raise ValueError(
                f"RemoteRewardBackend: /health at {self.base_url} returned unexpected shape: "
                f"{body!r}. Expected {{'status': 'ok', 'rewards': {{<name>: [...]}}}}."
            )

        available = sorted(body["rewards"].keys())
        available_set = set(available)
        missing = [name for name in self.required_rewards if name not in available_set]
        if missing:
            raise ValueError(
                f"RemoteRewardBackend: required_rewards={missing} not served by "
                f"{self.base_url}; server reports available={available}. "
                f"Check REWARD_COMPONENTS for typos "
                f"(e.g. 'unifiedreward' vs 'unified_reward')."
            )

        logger.info(
            "RemoteRewardBackend: %s serves rewards=%s",
            self.base_url,
            available,
        )
        self._remote_rewards_validated = True

    def dispose(self) -> None:
        """Close the HTTP session."""
        self._session.close()

    def _build_score_payload(self, request: RewardRequest) -> Dict[str, Any]:
        """Convert a UniRL ``RewardRequest`` into the RewardService"""
        images = request.images or []
        prompts = request.prompts
        metadata_list = request.metadata
        wire_requests: List[Dict[str, Any]] = []

        condition_images = self._get_condition_images(request)

        for idx in range(len(images)):
            prompt = prompts[idx] if idx < len(prompts) else ""
            image_b64 = _encode_image_b64(
                images[idx],
                image_format=self.image_format,
                quality=self.image_quality,
            )
            sample_metadata = None
            if metadata_list is not None and idx < len(metadata_list):
                sample_metadata = metadata_list[idx]

            if condition_images is not None and idx < len(condition_images):
                condition_b64 = _encode_image_b64(
                    condition_images[idx],
                    image_format=self.image_format,
                    quality=self.image_quality,
                )
                history = [
                    {"text": prompt, "image_b64": condition_b64},
                    {"text": prompt, "image_b64": image_b64},
                ]
                media_fingerprint = hashlib.sha256(f"{condition_b64}:{image_b64}".encode("ascii")).hexdigest()
            else:
                history = [{"text": prompt, "image_b64": image_b64}]
                media_fingerprint = hashlib.sha256(image_b64.encode("ascii")).hexdigest()

            identity = _wire_identity(
                request,
                idx,
                required_rewards=self.required_rewards,
                expected_scorer_version=self.expected_scorer_version,
                payload_fingerprint=_payload_fingerprint(
                    media_fingerprint=media_fingerprint,
                    prompt=prompt,
                    metadata=sample_metadata,
                ),
            )
            wire_requests.append(
                {
                    "history": history,
                    "required_rewards": list(self.required_rewards),
                    "metadata": sample_metadata,
                    **identity,
                }
            )

        return {"protocol_version": "1", "requests": wire_requests}

    def _get_condition_images(self, request: RewardRequest) -> Optional[List[Union[Image.Image, torch.Tensor]]]:
        """Extract per-sample condition images from request primitives."""
        prim_image = request.primitives.get("image")
        if prim_image is None:
            return None
        from unirl.types.primitives import Images

        if not isinstance(prim_image, Images):
            raise TypeError(f"request.primitives['image'] must be Images, got {type(prim_image).__name__}")
        from unirl.utils.media import tensor_frame_to_pil

        return [tensor_frame_to_pil(image.pixels) for image in prim_image.to_list()]

    def _compute_video_rewards(self, request: RewardRequest, start: float) -> RewardResponse:
        """Send video tensors to the remote service and parse the response."""
        bs = request.batch_size
        try:
            payload = self._build_video_score_payload(request)
            raw = self._post_score_requests(payload["requests"])
            return self._parse_score_response(raw, bs, time.time() - start)
        except Exception:
            if self.raise_on_failure:
                raise
            logger.exception("RemoteRewardBackend._compute_video_rewards failed (degraded mode)")
            return RewardResponse(
                rewards=[0.0] * bs,
                successes=[False] * bs,
                errors=["RemoteRewardBackend video failure (see logs)"] * bs,
                compute_time=time.time() - start,
            )

    def _build_video_score_payload(self, request: RewardRequest) -> Dict[str, Any]:
        """Convert a video ``RewardRequest`` into the RewardService ``ScoreRequest``"""
        videos = request.videos or []
        prompts = request.prompts
        metadata_list = request.metadata
        wire_requests: List[Dict[str, Any]] = []

        for idx in range(len(videos)):
            prompt = prompts[idx] if idx < len(prompts) else ""
            video_b64 = _encode_video_b64(videos[idx], fps=self.video_fps)
            sample_metadata = None
            if metadata_list is not None and idx < len(metadata_list):
                sample_metadata = metadata_list[idx]
            identity = _wire_identity(
                request,
                idx,
                required_rewards=self.required_rewards,
                expected_scorer_version=self.expected_scorer_version,
                payload_fingerprint=_payload_fingerprint(
                    media_fingerprint=hashlib.sha256(video_b64.encode("ascii")).hexdigest(),
                    prompt=prompt,
                    metadata=sample_metadata,
                ),
            )
            wire_requests.append(
                {
                    "history": [{"text": prompt, "video_b64": video_b64}],
                    "required_rewards": list(self.required_rewards),
                    "metadata": sample_metadata,
                    **identity,
                }
            )

        return {"protocol_version": "1", "requests": wire_requests}

    def _post_score_requests(self, wire_requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not wire_requests:
            return {"protocol_version": "1", "results": [], "errors": []}
        request_batch_size = self.request_batch_size or len(wire_requests)
        merged_results: List[Dict[str, Dict[str, float]]] = []
        merged_errors: List[Dict[str, str]] = []

        for start in range(0, len(wire_requests), request_batch_size):
            chunk = wire_requests[start : start + request_batch_size]
            raw = self._post_score({"protocol_version": "1", "requests": chunk})
            response_version = raw.get("protocol_version")
            if response_version is not None and response_version != "1":
                raise ValueError(f"RewardService protocol_version={response_version!r}, expected '1'")
            results = list(raw.get("results") or [])
            errors = list(raw.get("errors") or [])
            identities = list(raw.get("identities") or [])
            if len(results) > len(chunk) or len(errors) > len(chunk):
                raise ValueError(
                    f"RewardService returned more rows than requested for chunk {start}: "
                    f"results={len(results)} errors={len(errors)} requested={len(chunk)}"
                )
            results.extend({} for _ in range(len(chunk) - len(results)))
            errors.extend({} for _ in range(len(chunk) - len(errors)))

            if identities:
                if len(identities) != len(chunk):
                    raise ValueError(
                        f"RewardService identity count {len(identities)} != requested chunk size {len(chunk)}"
                    )
                for expected, actual in zip(chunk, identities, strict=True):
                    self._validate_identity_echo(expected, actual)
            elif self.require_identity_echo:
                raise ValueError("RewardService response omitted required item identities")

            merged_results.extend(results)
            merged_errors.extend(errors)

        return {
            "protocol_version": "1",
            "results": merged_results,
            "errors": merged_errors,
        }

    def _validate_identity_echo(self, expected: Dict[str, Any], actual: Dict[str, Any]) -> None:
        for key in ("request_id", "sample_id", "group_id", "source_rank", "policy_version", "idempotency_key"):
            if actual.get(key) != expected.get(key):
                raise ValueError(
                    f"RewardService identity mismatch for {key}: expected {expected.get(key)!r}, "
                    f"got {actual.get(key)!r}"
                )
        if self.expected_scorer_version is not None and actual.get("scorer_version") != self.expected_scorer_version:
            raise ValueError(
                f"RewardService scorer_version={actual.get('scorer_version')!r} "
                f"!= expected {self.expected_scorer_version!r}"
            )

    def _post_score(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST to ``/score`` with retry logic."""
        url = f"{self.base_url}/score"
        last_exc: Optional[BaseException] = None

        for attempt in range(self.max_retries):
            try:
                resp = self._session.post(url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()
            except http_requests.exceptions.Timeout as e:
                last_exc = e
                logger.warning(
                    "RemoteRewardBackend: request timed out (attempt %d/%d)",
                    attempt + 1,
                    self.max_retries,
                )
            except http_requests.exceptions.RequestException as e:
                response = getattr(e, "response", None)
                if response is not None and 400 <= response.status_code < 500 and response.status_code != 429:
                    raise
                last_exc = e
                logger.warning(
                    "RemoteRewardBackend: %s (attempt %d/%d)",
                    e,
                    attempt + 1,
                    self.max_retries,
                )

            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay)

        raise RuntimeError(f"RemoteRewardBackend: failed after {self.max_retries} retries calling {url}") from last_exc

    def _parse_score_response(
        self,
        raw: Dict[str, Any],
        batch_size: int,
        compute_time: float,
    ) -> RewardResponse:
        """Convert the RewardService ``ScoreResponse`` JSON into a UniRL"""
        results: List[Dict[str, Dict[str, float]]] = raw.get("results", [])
        errors_list: List[Dict[str, str]] = raw.get("errors", [])

        while len(results) < batch_size:
            results.append({})
        while len(errors_list) < batch_size:
            errors_list.append({})

        component_rewards: Dict[str, List[float]] = {name: [] for name in self.required_rewards}
        aggregated_rewards: List[float] = []
        successes: List[bool] = []
        sample_errors: List[Optional[str]] = []

        for i in range(batch_size):
            sample_result = results[i]
            sample_errors_dict = errors_list[i]

            scores: List[float] = []
            weights: List[float] = []
            error_parts: List[str] = []

            for reward_name in self.required_rewards:
                if reward_name in sample_result:
                    sub_metrics = sample_result[reward_name]
                    non_finite = self._first_non_finite(sub_metrics)
                    if non_finite is not None:
                        # Reject non-finite scores before they poison group normalization.
                        metric_name, bad_value = non_finite
                        component_rewards[reward_name].append(0.0)
                        error_parts.append(
                            f"{reward_name}: non-finite value {bad_value!r} for sub-metric {metric_name!r}"
                        )
                        continue
                    score = self._reduce_sub_metrics(sub_metrics)
                    component_rewards[reward_name].append(score)
                    scores.append(score)
                    weights.append(self.reward_weights.get(reward_name, 1.0))
                else:
                    component_rewards[reward_name].append(0.0)
                    if reward_name in sample_errors_dict:
                        error_parts.append(f"{reward_name}: {sample_errors_dict[reward_name]}")
                    else:
                        error_parts.append(f"{reward_name}: missing from server response without error")

            if scores:
                aggregated_rewards.append(self._aggregate_scores(scores, weights))
                successes.append(len(error_parts) == 0)
            else:
                aggregated_rewards.append(0.0)
                successes.append(False)

            sample_errors.append("; ".join(error_parts) if error_parts else None)

        return RewardResponse(
            rewards=aggregated_rewards,
            component_rewards=component_rewards,
            successes=successes,
            errors=sample_errors,
            compute_time=compute_time,
        )

    def _aggregate_scores(self, scores: List[float], weights: List[float]) -> float:
        """Aggregate per-reward scores for one sample."""
        if not scores:
            return 0.0
        if self.aggregation_method == "weighted_sum":
            total_w = sum(weights)
            return sum(s * w for s, w in zip(scores, weights)) / total_w if total_w > 0 else 0.0
        if self.aggregation_method == "mean":
            return sum(scores) / len(scores)
        if self.aggregation_method == "min":
            return min(scores)
        return max(scores)

    @staticmethod
    def _first_non_finite(sub_metrics: Dict[str, float]) -> Optional[Tuple[str, Any]]:
        """Return the first ``(name, value)`` whose value is not a finite number."""
        for name, value in sub_metrics.items():
            if (
                value is None
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                return name, value
        return None

    def _reduce_sub_metrics(self, sub_metrics: Dict[str, float]) -> float:
        """Collapse a reward's sub-metric dict into a single float."""
        if not sub_metrics:
            raise ValueError("RewardService returned an empty sub-metric mapping")
        values = list(sub_metrics.values())
        if self.sub_metric_reduce == "first":
            return float(values[0])
        if self.sub_metric_reduce == "mean":
            return float(sum(values) / len(values))
        return float(max(values))


@dataclass
class RemoteRewardSpec(BaseRewardComponentSpec):
    """Typed config for the remote RewardService backend."""

    base_url: str = ""
    required_rewards: Tuple[str, ...] = ()
    reward_weights: Optional[Dict[str, float]] = None
    batch_size: int = 8
    timeout: float = 300.0
    max_retries: int = 3
    retry_delay: float = 1.0
    # Transport chunking is independent from scorer/model micro-batching.
    # None preserves the legacy one-POST-per-DP-shard behavior.
    request_batch_size: Optional[int] = None
    require_identity_echo: bool = False
    # Managed/direct single-scorer servers only; the multi-reward gateway rejects a pin.
    expected_scorer_version: Optional[str] = None
    sub_metric_reduce: str = "first"
    aggregation_method: str = "weighted_sum"
    image_format: str = "JPEG"
    image_quality: int = 95
    video_fps: int = 8
    input_kind: str = "image"
    raise_on_failure: bool = True

    def __post_init__(self) -> None:
        require(
            bool(str(self.base_url).strip()),
            "RemoteRewardSpec.base_url must be non-empty",
        )
        require(
            len(self.required_rewards) > 0,
            "RemoteRewardSpec.required_rewards must be non-empty",
        )
        require(
            self.max_retries >= 1,
            f"RemoteRewardSpec.max_retries must be >= 1; got {self.max_retries!r}",
        )
        require(
            self.retry_delay >= 0,
            f"RemoteRewardSpec.retry_delay must be >= 0; got {self.retry_delay!r}",
        )
        require(
            self.request_batch_size is None or self.request_batch_size >= 1,
            f"RemoteRewardSpec.request_batch_size must be None or >= 1; got {self.request_batch_size!r}",
        )
        require(
            self.sub_metric_reduce in {"first", "mean", "max"},
            f"RemoteRewardSpec.sub_metric_reduce must be one of first/mean/max; got {self.sub_metric_reduce!r}",
        )
        require(
            self.aggregation_method in {"weighted_sum", "mean", "min", "max"},
            f"RemoteRewardSpec.aggregation_method must be one of "
            f"weighted_sum/mean/min/max; got {self.aggregation_method!r}",
        )
        require(
            self.input_kind in {"image", "video"},
            f"RemoteRewardSpec.input_kind must be 'image' or 'video'; got {self.input_kind!r}",
        )


# ---------------------------------------------------------------------------
# OpenAIChatRewardBackend
# ---------------------------------------------------------------------------


class OpenAIChatRewardBackend(RewardBackend):
    """Client-side-scoring HTTP client for a fleet of vLLM ``/v1/chat/completions`` judges (see README.md)."""

    _COMPONENT = "text_rendering_judge"
    _SUB_METRICS = ("text_render_reward", "n_errors", "error_cost", "parse_ok")

    def __init__(self, *, config: "OpenAIChatRewardSpec", base_device: str) -> None:
        del base_device  # HTTP backend, no device dependency
        super().__init__(
            model_name=self._COMPONENT,
            batch_size=config.batch_size,
            timeout=config.timeout,
        )
        self.endpoints: List[str] = self._resolve_endpoints(config)
        self.model = config.model
        self.alpha = float(config.alpha)
        self.max_new_tokens = config.max_new_tokens
        self.stop = list(config.stop or ())
        self.temperature = config.temperature
        self.top_p = config.top_p
        self.request_concurrency = config.request_concurrency
        self.max_retries = config.max_retries
        self.retry_delay = config.retry_delay
        self.parse_retries = int(config.parse_retries)
        self.parse_retry_temperature = float(config.parse_retry_temperature)
        self.image_format = config.image_format
        self.image_quality = config.image_quality
        self.raise_on_failure = config.raise_on_failure

        # ── Severity weighting (L2) ──
        # Off by default, so existing runs are unchanged. When on, each error
        # contributes its normalised edit distance in [0,1] instead of a flat 1,
        # so a one-letter typo costs less than a fully garbled title.
        #
        # ⚠️ SCALE CHANGE: severity <= 1 makes the cost ~0.54x len(errors)
        # (measured on one real 6-error sample; the true ratio depends on how
        # often layout/style errors fire and is still unmeasured). Reusing
        # alpha=0.1 would move the reward range from [0.22,0.45] up to
        # [0.44,0.65] — narrower AND closer to exp()'s flat region, i.e. the
        # opposite of the intent. Rescale alpha ~2x when enabling this.
        self.severity_weighting = bool(getattr(config, "severity_weighting", False))
        self.undefined_severity_cost = float(getattr(config, "undefined_severity_cost", 1.0))
        # Per-category flat costs for the two categories that have NO severity
        # (they compare a relation, not a spelling). None = fall back to
        # undefined_severity_cost.
        _lw = getattr(config, "layout_weight", None)
        _sw = getattr(config, "style_weight", None)
        self.layout_weight = None if _lw is None else float(_lw)
        self.style_weight = None if _sw is None else float(_sw)
        # V4: charge "drew something illegible" less than "drew nothing at all".
        # None = keep them equal (V3 behaviour).
        _gc = getattr(config, "garbled_cost", None)
        self.garbled_cost = None if _gc is None else float(_gc)
        if self.severity_weighting:
            logger.info(
                "OpenAIChatRewardBackend: severity weighting ON (alpha=%.3f, "
                "undefined=%.2f, layout=%s, style=%s, garbled=%s) — cost is a "
                "continuous sum of per-error costs, NOT len(errors)",
                self.alpha,
                self.undefined_severity_cost,
                self.layout_weight if self.layout_weight is not None else "=undefined",
                self.style_weight if self.style_weight is not None else "=undefined",
                self.garbled_cost if self.garbled_cost is not None else "=absent (V3)",
            )

        # Optional observational dump of the full parsed judge output. Empty =
        # off (the default), so existing runs stay byte-identical. The lock
        # serialises appends across the `request_concurrency` worker threads;
        # one line per sample, so a crash mid-run still leaves valid JSONL.
        self.dump_path = str(getattr(config, "dump_path", "") or "").strip()
        self._dump_lock = threading.Lock()
        if self.dump_path:
            _dump_dir = os.path.dirname(os.path.abspath(self.dump_path))
            if _dump_dir:
                os.makedirs(_dump_dir, exist_ok=True)
            logger.info("OpenAIChatRewardBackend: dumping full judge output to %s", self.dump_path)

        # Round-robin endpoint picker. `itertools.cycle` + a Lock is thread-safe
        # under CPython — the lock is cheap and makes the invariant explicit.
        self._endpoint_iter = itertools.cycle(self.endpoints)
        self._endpoint_lock = threading.Lock()

        # Disable proxy env vars — the reward cluster is on the internal
        # network where corporate HTTP proxies would return 503.
        self._session = http_requests.Session()
        self._session.trust_env = False

        # Size the connection pool to the thread pool. urllib3's default
        # (pool_maxsize=10) is smaller than `request_concurrency`, so the extra
        # threads keep evicting each other's keep-alive connections
        # ("Connection pool is full, discarding connection"), forcing a fresh TCP
        # handshake per request and — under a slow judge — read timeouts. One slot
        # per concurrent request per endpoint keeps every connection alive.
        pool_size = max(self.request_concurrency, 10)
        adapter = HTTPAdapter(pool_connections=max(len(self.endpoints), 1), pool_maxsize=pool_size)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

        logger.info(
            "OpenAIChatRewardBackend: %d endpoints, model=%s, alpha=%.3f, parse_retries=%d, pool_maxsize=%d",
            len(self.endpoints),
            self.model,
            self.alpha,
            self.parse_retries,
            pool_size,
        )

    # ------------------------------------------------------------------
    # Public interface (RewardBackend)
    # ------------------------------------------------------------------

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        """Chat-completion each ``(prompt, image)`` sample, parse, and score; image-only."""
        start = time.time()
        if request.is_video:
            raise ValueError("OpenAIChatRewardBackend does not support video rewards; use an image-only recipe.")

        bs = request.batch_size
        images = request.images or []
        prompts = request.prompts

        try:
            per_sample = self._score_all(images, prompts)
            return self._build_response(per_sample, bs, time.time() - start)
        except Exception:
            if self.raise_on_failure:
                raise
            logger.exception("OpenAIChatRewardBackend.compute_rewards failed (degraded mode)")
            return RewardResponse(
                rewards=[0.0] * bs,
                component_rewards={m: [0.0] * bs for m in self._SUB_METRICS},
                successes=[False] * bs,
                errors=["OpenAIChatRewardBackend failure (see logs)"] * bs,
                compute_time=time.time() - start,
            )

    def is_available(self) -> bool:
        """``True`` iff any endpoint's ``/v1/models`` responds 200; gates trainer startup only."""
        for url in self.endpoints:
            try:
                resp = self._session.get(f"{url.rstrip('/')}/v1/models", timeout=5.0)
                if resp.status_code == 200:
                    return True
            except http_requests.exceptions.RequestException:
                continue
        return False

    def dispose(self) -> None:
        """Close the HTTP session."""
        self._session.close()

    # ------------------------------------------------------------------
    # Endpoint resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_endpoints(config: "OpenAIChatRewardSpec") -> List[str]:
        """Combine ``endpoints`` list with ``endpoints_file`` contents, dedup, keep order."""
        seen: set = set()
        out: List[str] = []
        for src in list(config.endpoints):
            u = src.strip().rstrip("/")
            if u and u not in seen:
                seen.add(u)
                out.append(u)
        if config.endpoints_file:
            path = os.path.expanduser(config.endpoints_file)
            try:
                with open(path, "r") as fh:
                    lines = fh.readlines()
            except OSError as e:
                raise ValueError(f"OpenAIChatRewardSpec.endpoints_file={path!r} could not be read: {e}") from e
            for raw in lines:
                line = raw.split("#", 1)[0].strip().rstrip("/")
                if line and line not in seen:
                    seen.add(line)
                    out.append(line)
        if not out:
            raise ValueError("OpenAIChatRewardSpec: no endpoints resolved; set `endpoints` or `endpoints_file`.")
        return out

    def _next_endpoint(self) -> str:
        """Thread-safe round-robin endpoint pick."""
        with self._endpoint_lock:
            return next(self._endpoint_iter)

    # ------------------------------------------------------------------
    # Concurrent scoring
    # ------------------------------------------------------------------

    def _score_all(
        self,
        images: List[Union[Image.Image, torch.Tensor]],
        prompts: List[str],
    ) -> List[Dict[str, float]]:
        """Score every sample concurrently, preserving input order; sequential when n == 1."""
        n = len(images)
        if n == 0:
            return []
        results: List[Optional[Dict[str, float]]] = [None] * n

        def _work(idx: int) -> Tuple[int, Dict[str, float]]:
            prompt = prompts[idx] if idx < len(prompts) else ""
            try:
                return idx, self._score_one(prompt, images[idx])
            except Exception:
                if self.raise_on_failure:
                    raise
                # Degraded mode: downgrade a per-sample transport failure (all
                # `max_retries` exhausted across endpoints) to a per-sample
                # failure MARKER instead of killing the whole batch. Without
                # this, one unlucky sample out of 1024 aborts the run — observed
                # after 15 rollouts against a single saturated judge.
                #
                # This is not "ignore errors": the marker sets parse_ok=0, and
                # RewardService.score_and_attach still raises unless the batch's
                # failure ratio is within `max_failure_ratio`. The abort decision
                # just moves to the layer that can see the whole batch and tell
                # "one flaky request" apart from "the judge is down".
                logger.exception(
                    "OpenAIChatRewardBackend: sample %d failed after all retries; "
                    "marking it failed (see RewardService.max_failure_ratio)",
                    idx,
                )
                return idx, {
                    "text_render_reward": 0.0,
                    "n_errors": 0.0,
                    "error_cost": 0.0,
                    "parse_ok": 0.0,
                }

        if n == 1:
            i, r = _work(0)
            results[i] = r
        else:
            max_workers = min(self.request_concurrency, n)
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="oaichat") as ex:
                for fut in as_completed([ex.submit(_work, i) for i in range(n)]):
                    i, r = fut.result()
                    results[i] = r

        # Every slot filled by now; None here would be a programming bug.
        return [r for r in results if r is not None]

    def _score_one(
        self,
        prompt: str,
        image: Union[Image.Image, torch.Tensor],
    ) -> Dict[str, float]:
        """Score one sample: POST → parse → sub-metric dict, re-asking on unparseable output."""
        image_b64 = _encode_image_b64(image, image_format=self.image_format, quality=self.image_quality)
        data_url = f"data:image/{self.image_format.lower()};base64,{image_b64}"
        base_payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": build_instruct_text(prompt)},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "max_tokens": self.max_new_tokens,
            "top_p": self.top_p,
        }
        if self.stop:
            base_payload["stop"] = list(self.stop)

        for attempt in range(self.parse_retries):
            temperature = self.temperature if attempt == 0 else self.parse_retry_temperature
            raw_text, finish_reason = self._post_one({**base_payload, "temperature": temperature})
            # finish_reason == "length" means the answer was cut off at the token
            # cap, so this score is not trustworthy whether or not it parses. Catch
            # it here rather than inferring from a regex miss: a truncated response
            # can still parse (tier-3 recovers an earlier complete {...}) and would
            # otherwise be silently scored as if the judge had finished, with the
            # errors past the cut simply missing.
            if finish_reason == "length":
                logger.warning(
                    "OpenAIChatRewardBackend: judge output TRUNCATED at max_tokens=%d "
                    "(attempt %d/%d) — discarding and re-asking. Raise max_new_tokens "
                    "if this recurs; raw tail: %r",
                    self.max_new_tokens,
                    attempt + 1,
                    self.parse_retries,
                    (raw_text or "")[-160:],
                )
                continue
            errs, ok = parse_errors(raw_text)
            if ok:
                errs = errs or []
                n_errors = len(errs)
                if self.severity_weighting:
                    cost = weighted_error_cost(
                        errs,
                        undefined_severity_cost=self.undefined_severity_cost,
                        layout_weight=self.layout_weight,
                        style_weight=self.style_weight,
                        garbled_cost=self.garbled_cost,
                    )
                else:
                    cost = float(n_errors)
                self._dump_errors(prompt, errs, n_errors, attempt, raw_text)
                return {
                    "text_render_reward": float(math.exp(-self.alpha * cost)),
                    "n_errors": float(n_errors),
                    # Continuous cost actually fed to exp(). Equals n_errors when
                    # severity weighting is off, so the metric is always readable.
                    "error_cost": float(cost),
                    "parse_ok": 1.0,
                }
            logger.warning(
                "OpenAIChatRewardBackend: unparseable judge output (attempt %d/%d, temperature=%.2f); raw head: %r",
                attempt + 1,
                self.parse_retries,
                temperature,
                (raw_text or "")[:200],
            )

        self._dump_errors(prompt, None, 0, self.parse_retries - 1, raw_text)
        return {
            "text_render_reward": 0.0,
            "n_errors": 0.0,
            "error_cost": 0.0,
            "parse_ok": 0.0,
        }

    def _dump_errors(
        self,
        prompt: str,
        errors: Optional[list],
        n_errors: int,
        attempt: int,
        raw_text: str,
    ) -> None:
        """Append the full parsed judge output to ``dump_path`` as one JSONL line; never raises."""
        if not self.dump_path:
            return
        try:
            record = {
                "prompt": prompt,
                "n_errors": n_errors,
                "parse_ok": errors is not None,
                "attempt": attempt,
                # Each item is the judge's own {"description", "box"} dict, kept
                # verbatim — the error TYPE, the requirement-vs-actual pair and
                # the 0-999 normalised box all live inside `description`/`box`.
                "errors": errors,
            }
            if errors is None:
                record["raw_head"] = (raw_text or "")[:500]
            line = json.dumps(record, ensure_ascii=False)
            with self._dump_lock:
                with open(self.dump_path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except Exception:
            logger.warning("OpenAIChatRewardBackend: judge dump failed (ignored)", exc_info=True)

    def _post_one(self, payload: Dict[str, Any]) -> Tuple[str, Optional[str]]:
        """POST to a round-robin endpoint, hopping endpoints per retry; returns ``(content, finish_reason)``."""
        last_exc: Optional[BaseException] = None
        for attempt in range(self.max_retries):
            endpoint = self._next_endpoint()
            url = f"{endpoint}/v1/chat/completions"
            try:
                resp = self._session.post(url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                body = resp.json()
                choices = body.get("choices") or []
                if not choices:
                    raise ValueError(f"empty `choices` in response from {url}")
                content = (choices[0].get("message") or {}).get("content")
                if not isinstance(content, str):
                    raise ValueError(f"missing/non-string message.content in response from {url}")
                return content, choices[0].get("finish_reason")
            except (http_requests.exceptions.RequestException, ValueError) as e:
                last_exc = e
                logger.warning(
                    "OpenAIChatRewardBackend: %s → %s (attempt %d/%d)",
                    url,
                    e,
                    attempt + 1,
                    self.max_retries,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

        raise RuntimeError(
            f"OpenAIChatRewardBackend: all {self.max_retries} attempts failed across endpoints"
        ) from last_exc

    # ------------------------------------------------------------------
    # Response assembly
    # ------------------------------------------------------------------

    def _build_response(
        self,
        per_sample: List[Dict[str, float]],
        batch_size: int,
        compute_time: float,
    ) -> RewardResponse:
        """Fan per-sample sub-metric dicts back into a ``RewardResponse``."""
        while len(per_sample) < batch_size:
            per_sample.append({m: 0.0 for m in self._SUB_METRICS})

        component_rewards: Dict[str, List[float]] = {m: [] for m in self._SUB_METRICS}
        rewards: List[float] = []
        successes: List[bool] = []
        errors: List[Optional[str]] = []
        for row in per_sample:
            for m in self._SUB_METRICS:
                component_rewards[m].append(float(row.get(m, 0.0)))
            rewards.append(float(row.get("text_render_reward", 0.0)))
            parse_ok = row.get("parse_ok", 0.0) == 1.0
            successes.append(parse_ok)
            errors.append(None if parse_ok else "judge output failed to parse")

        return RewardResponse(
            rewards=rewards,
            component_rewards=component_rewards,
            successes=successes,
            errors=errors,
            compute_time=compute_time,
        )


# Calibration anchors for the severity_weighting alpha guard below, measured on
# a 123k-sample klein-9B judge dump (judge_dump_severity_20260807). The weighted
# cost there runs p50 8.0 / p95 17.0 / max 35.0.
_SEVERITY_HIGH_COST = 17.0
# Below this a reward is close enough to 0 that neighbouring costs are no longer
# resolvable within a group, so those samples stop carrying advantage.
_SEVERITY_MIN_REWARD = 0.02


@dataclass
class OpenAIChatRewardSpec(BaseRewardComponentSpec):
    """Typed config for :class:`OpenAIChatRewardBackend`; needs ``endpoints`` or ``endpoints_file``."""

    endpoints: Tuple[str, ...] = ()
    endpoints_file: str = ""
    model: str = "qwen3vl-judge"
    alpha: float = 0.3
    # 8192, not 4096: at 4096 about 1% of images truncate. A dense infographic
    # with many errors needs 4000+ output tokens, and hitting the cap cuts the
    # JSON mid-string, so the <answer> block never parses. Measured by the judge
    # owner on 193 images (2026-08-23): 2 failures at 4096, 0 at 8192.
    max_new_tokens: int = 8192
    # Stop as soon as the closing tag is emitted instead of letting the model
    # ramble past it — +6.8% throughput (judge owner, same measurement).
    #
    # ⚠️ vLLM STRIPS the stop string from the response, so `message.content` ends
    # at "...}" with NO </answer>. `parse_errors` already handles that: its
    # `<answer>` regex misses, but the third tier scans for the last decodable
    # {...} carrying an `errors` key. Verified against the live endpoint
    # (2026-08-23) — an unterminated tag parses fine, a mid-string truncation
    # correctly reports parse_ok=False. So do NOT re-append the tag here.
    stop: Tuple[str, ...] = ("</answer>",)
    temperature: float = 0.0
    top_p: float = 1.0
    request_concurrency: int = 32
    batch_size: int = 8
    timeout: float = 300.0
    max_retries: int = 3
    retry_delay: float = 1.0
    # Total judge attempts per sample when the answer fails to parse (1 = no
    # re-judge, the historical behaviour). Re-asks after the first use
    # `parse_retry_temperature` — replaying at temperature 0 would just
    # reproduce the same malformed output.
    parse_retries: int = 1
    parse_retry_temperature: float = 0.3
    image_format: str = "JPEG"
    image_quality: int = 95
    raise_on_failure: bool = True
    # Optional JSONL path for the FULL parsed judge output (description + box
    # per error). Empty = off. Observational only: the reward is unchanged.
    # The scalar reward keeps only len(errors), so the error type, the
    # requirement-vs-actual pair and the box are otherwise discarded — this
    # dump is what a finer-grained reward would have to be calibrated on.
    dump_path: str = ""
    # ── L2 severity weighting ──
    # False (default) = cost is len(errors), the historical behaviour.
    # True = cost is the sum of per-error normalised edit distances in [0,1],
    # so "Chain"->"Clain" (0.2) costs less than a fully garbled title.
    #
    # Keep `alpha` at its count-based value when enabling this. The cost scale is
    # nearly unchanged (~0.93x len(errors) on a 123k-sample dump) because most
    # content errors are absent-or-garbled and score a full 1.0, and GRPO
    # normalises advantages per group so alpha's scale cancels anyway.
    # __post_init__ rejects an alpha large enough to saturate exp() instead.
    severity_weighting: bool = False
    # Cost charged for errors with no severity: layout/style errors (which
    # compare a relation, not a spelling) and unparseable descriptions. 1.0 keeps
    # them at full weight — do NOT set 0.0 unless you intend them to be free.
    undefined_severity_cost: float = 1.0
    # Optional per-category overrides for the two no-severity categories, so a
    # recipe can say "a font-weight mismatch matters less than garbled text"
    # without also discounting misplaced-but-correctly-spelled labels.
    # None = use undefined_severity_cost.
    layout_weight: Optional[float] = None
    style_weight: Optional[float] = None
    # ── V4 (2026-08-13): "drew something illegible" < "drew nothing" ──
    # V3 charges a flat 1.0 for both, which makes "render nothing" a strictly
    # safe way to avoid a garble error. Measured on the 263k-sample dump of run
    # klein9b_severity_only_fixed: within a prompt group the top reward quartile
    # carried MORE fully-unrendered text than the bottom quartile (+2.9pp ± 1.4
    # early, +1.6pp ± 0.5 after that run collapsed) — i.e. the reward mildly
    # rewarded not rendering. At 0.85 the measurement flips to -15.9pp ± 1.4.
    # None = keep V3 behaviour, so this is opt-in and existing runs are
    # unaffected. Only affects content errors whose severity came out maximal
    # AND that the judge marked garbled without also marking absent.
    garbled_cost: Optional[float] = None

    def __post_init__(self) -> None:
        require(
            bool(self.endpoints) or bool(str(self.endpoints_file).strip()),
            "OpenAIChatRewardSpec: either endpoints or endpoints_file must be set",
        )
        require(
            0.0 <= float(self.undefined_severity_cost) <= 1.0,
            f"OpenAIChatRewardSpec.undefined_severity_cost must be in [0, 1]; got {self.undefined_severity_cost!r}",
        )
        # garbled_cost must stay strictly below 1.0 (the absent cost) or it does
        # not create the "rendering something beats rendering nothing" gap it
        # exists for; >= 1.0 would silently invert it back or worse.
        require(
            self.garbled_cost is None or 0.0 <= float(self.garbled_cost) < 1.0,
            "OpenAIChatRewardSpec.garbled_cost must be in [0, 1) — it has to be "
            "strictly cheaper than the 1.0 charged for text that never appeared, "
            f"otherwise it does nothing; got {self.garbled_cost!r}",
        )
        require(
            self.garbled_cost is None or bool(self.severity_weighting),
            "OpenAIChatRewardSpec.garbled_cost has no effect without "
            "severity_weighting=True (the count-based cost has no severity to "
            "discount) — set severity_weighting or drop garbled_cost",
        )
        # Severity weighting only changes WHICH cost a sample gets, not the scale
        # it lives on: over a 123k-sample klein-9B dump the weighted cost averages
        # 8.41 vs 9.04 for len(errors) (0.93x), because most content errors turn
        # out to be absent-or-garbled (severity 1.0). So alpha does NOT need
        # rescaling here — keeping the count-based value holds the reward geometry
        # fixed, which is what makes a severity run a single-variable comparison.
        #
        # The real failure mode is alpha too LARGE. Advantages are normalised per
        # group ((r - mean)/std, see RolloutTrack.compute_advantages), so alpha's
        # scale cancels out of the gradient: sweeping 0.05..0.35 leaves the
        # pairwise collision rate at 25.1% and zero_std_ratio at 2.21%, unchanged.
        # What a big alpha does instead is saturate exp() — at alpha=0.25 a p95
        # cost of 17 maps to 0.014, so a group's worst samples collapse to
        # indistinguishable rewards and stop contributing advantage. Guard that.
        require(
            not self.severity_weighting or math.exp(-float(self.alpha) * _SEVERITY_HIGH_COST) >= _SEVERITY_MIN_REWARD,
            "OpenAIChatRewardSpec: alpha is too large for severity_weighting — a "
            f"high-but-ordinary weighted cost of {_SEVERITY_HIGH_COST} maps to reward "
            f"{math.exp(-float(self.alpha) * _SEVERITY_HIGH_COST):.4f} < {_SEVERITY_MIN_REWARD}, "
            "so the worst samples in a group saturate to indistinguishable values and "
            f"stop contributing advantage. Got alpha={self.alpha!r}. The weighted cost "
            "is ~0.93x len(errors), so the count-based alpha (0.1) carries over as-is.",
        )
        require(
            0.0 < float(self.alpha) < 10.0,
            f"OpenAIChatRewardSpec.alpha must be in (0, 10); got {self.alpha!r}",
        )
        require(
            self.request_concurrency >= 1,
            f"OpenAIChatRewardSpec.request_concurrency must be >= 1; got {self.request_concurrency!r}",
        )
        require(
            self.max_retries >= 1,
            f"OpenAIChatRewardSpec.max_retries must be >= 1; got {self.max_retries!r}",
        )
        require(
            self.parse_retries >= 1,
            f"OpenAIChatRewardSpec.parse_retries must be >= 1; got {self.parse_retries!r}",
        )
        require(
            0.0 <= float(self.parse_retry_temperature) <= 2.0,
            f"OpenAIChatRewardSpec.parse_retry_temperature must be in [0, 2]; got {self.parse_retry_temperature!r}",
        )


__all__ = [
    "OpenAIChatRewardBackend",
    "OpenAIChatRewardSpec",
    "RemoteRewardBackend",
    "RemoteRewardSpec",
]
