from __future__ import annotations

import inspect
import types
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any


PRECISION_ALIASES = {
    "auto": "auto",
    "bf16": "bf16",
    "bfloat16": "bf16",
    "fp16": "fp16",
    "float16": "fp16",
    "half": "fp16",
    "fp32": "fp32",
    "float32": "fp32",
    "full": "fp32",
}


@dataclass(frozen=True)
class Sam3RuntimePrecision:
    requested: str
    effective: str
    device_name: str
    compute_capability: tuple[int, int] | None
    bf16_supported: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        capability = payload["compute_capability"]
        payload["compute_capability"] = list(capability) if capability is not None else None
        return payload


def normalize_precision(value: str) -> str:
    normalized = str(value).strip().lower()
    try:
        return PRECISION_ALIASES[normalized]
    except KeyError as exc:
        choices = ", ".join(sorted({"auto", "bf16", "fp16", "fp32"}))
        raise ValueError(f"Unsupported SAM3 inference precision {value!r}; choose one of: {choices}") from exc


def resolve_runtime_precision(requested: str = "auto", *, torch_module: Any | None = None) -> Sam3RuntimePrecision:
    torch = torch_module
    if torch is None:
        import torch as torch_module_imported  # type: ignore

        torch = torch_module_imported

    requested_normalized = normalize_precision(requested)
    cuda_available = bool(torch.cuda.is_available())
    if not cuda_available:
        if requested_normalized in {"bf16", "fp16"}:
            raise RuntimeError(
                f"SAM3 precision {requested_normalized} requires CUDA, but CUDA is unavailable"
            )
        return Sam3RuntimePrecision(
            requested=requested_normalized,
            effective="fp32",
            device_name="cpu",
            compute_capability=None,
            bf16_supported=False,
            reason="CUDA is unavailable; using FP32",
        )

    device_index = int(torch.cuda.current_device())
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(device_index))
    device_name = str(torch.cuda.get_device_name(device_index))
    is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
    bf16_supported = bool(is_bf16_supported()) if callable(is_bf16_supported) else capability[0] >= 8

    if requested_normalized == "auto":
        if bf16_supported:
            effective = "bf16"
            reason = "GPU reports native BF16 support"
        else:
            effective = "fp16"
            reason = "GPU has no native BF16 support; using hardware FP16"
    else:
        effective = requested_normalized
        reason = "precision selected explicitly"

    if effective == "bf16" and not bf16_supported:
        raise RuntimeError(
            f"{device_name} (compute capability {capability[0]}.{capability[1]}) does not provide "
            "native BF16 support; use --inference-precision fp16 or auto"
        )

    return Sam3RuntimePrecision(
        requested=requested_normalized,
        effective=effective,
        device_name=device_name,
        compute_capability=capability,
        bf16_supported=bf16_supported,
        reason=reason,
    )


def configure_sam3_runtime(
    predictor: Any,
    *,
    requested_precision: str = "auto",
    cache_visual_features: bool = False,
    use_fa3: bool = False,
    torch_module: Any | None = None,
) -> Sam3RuntimePrecision:
    torch = torch_module
    if torch is None:
        import torch as torch_module_imported  # type: ignore

        torch = torch_module_imported

    precision = resolve_runtime_precision(requested_precision, torch_module=torch)
    if use_fa3 and precision.effective != "bf16":
        raise RuntimeError(
            "FlashAttention 3 in this SAM3.1 build requires BF16. Disable --use-fa3 on T4, "
            "or run on a BF16-capable Ampere-or-newer GPU."
        )

    _leave_hardcoded_bf16_contexts(predictor)
    _patch_predictor_add_prompt(predictor, precision=precision, torch_module=torch)
    if cache_visual_features:
        enable_single_image_visual_cache(predictor)

    predictor._sam3_runtime_precision = precision
    predictor._sam3_cache_visual_features = bool(cache_visual_features)
    return precision


def _leave_hardcoded_bf16_contexts(predictor: Any) -> None:
    # The multiplex tracker enters its context before the public predictor, so
    # contexts must be left in the opposite order.
    tracker = getattr(getattr(predictor, "model", None), "tracker", None)
    owners = [owner for owner in (tracker, predictor) if owner is not None]
    for owner in reversed(owners):
        context = getattr(owner, "bf16_context", None)
        if context is None:
            continue
        context.__exit__(None, None, None)
        owner.bf16_context = None


def _autocast_context(torch: Any, precision: Sam3RuntimePrecision) -> Any:
    if precision.effective == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision.effective == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _patch_predictor_add_prompt(
    predictor: Any,
    *,
    precision: Sam3RuntimePrecision,
    torch_module: Any,
) -> None:
    torch = torch_module

    def add_prompt_runtime(
        self: Any,
        session_id: str,
        frame_idx: int,
        text: str | None = None,
        points: Any = None,
        point_labels: Any = None,
        clear_old_points: bool = True,
        bounding_boxes: Any = None,
        bounding_box_labels: Any = None,
        clear_old_boxes: bool = True,
        output_prob_thresh: float = 0.5,
        obj_id: int | None = None,
        rel_coordinates: bool = True,
    ) -> dict[str, Any]:
        session = self._get_session(session_id)
        inference_state = session["state"]
        self._extend_expiration_time(session)

        if points is not None and not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32)
        if point_labels is not None and not isinstance(point_labels, torch.Tensor):
            point_labels = torch.tensor(point_labels, dtype=torch.int32)
        if bounding_boxes is not None and not isinstance(bounding_boxes, torch.Tensor):
            bounding_boxes = torch.tensor(bounding_boxes, dtype=torch.float32)
        if bounding_box_labels is not None and not isinstance(bounding_box_labels, torch.Tensor):
            bounding_box_labels = torch.tensor(bounding_box_labels, dtype=torch.int32)

        kwargs = {
            "inference_state": inference_state,
            "frame_idx": frame_idx,
            "text_str": text,
            "points": points,
            "point_labels": point_labels,
            "clear_old_points": clear_old_points,
            "boxes_xywh": bounding_boxes,
            "box_labels": bounding_box_labels,
            "clear_old_boxes": clear_old_boxes,
            "output_prob_thresh": output_prob_thresh,
            "rel_coordinates": rel_coordinates,
        }
        if obj_id is not None:
            kwargs["obj_id"] = obj_id
        valid_params = set(inspect.signature(self.model.add_prompt).parameters)
        filtered_kwargs = {key: value for key, value in kwargs.items() if key in valid_params}

        with _autocast_context(torch, precision):
            output_frame_idx, outputs = self.model.add_prompt(**filtered_kwargs)
        return {"frame_index": output_frame_idx, "outputs": outputs}

    predictor.add_prompt = types.MethodType(add_prompt_runtime, predictor)


def _resolve_visual_backbone(predictor: Any) -> Any:
    model = getattr(predictor, "model", None)
    detector = getattr(model, "detector", None)
    backbone = getattr(detector, "backbone", None)
    if backbone is None or not callable(getattr(backbone, "forward_image", None)):
        raise AttributeError("SAM3 predictor does not expose model.detector.backbone.forward_image")
    return backbone


def enable_single_image_visual_cache(predictor: Any) -> None:
    backbone = _resolve_visual_backbone(predictor)
    if getattr(backbone, "_sam3_visual_cache_enabled", False):
        return

    original_forward_image = backbone.forward_image
    backbone._sam3_original_forward_image = original_forward_image
    backbone._sam3_visual_cache_value = None
    backbone._sam3_visual_cache_hits = 0
    backbone._sam3_visual_cache_misses = 0

    def cached_forward_image(self: Any, samples: Any, *args: Any, **kwargs: Any) -> Any:
        cached = self._sam3_visual_cache_value
        if cached is not None:
            self._sam3_visual_cache_hits += 1
            return cached
        output = self._sam3_original_forward_image(samples, *args, **kwargs)
        self._sam3_visual_cache_value = output
        self._sam3_visual_cache_misses += 1
        return output

    backbone.forward_image = types.MethodType(cached_forward_image, backbone)
    backbone._sam3_visual_cache_enabled = True


def clear_visual_feature_cache(predictor: Any) -> None:
    try:
        backbone = _resolve_visual_backbone(predictor)
    except AttributeError:
        return
    if getattr(backbone, "_sam3_visual_cache_enabled", False):
        backbone._sam3_visual_cache_value = None


def sam3_runtime_summary(predictor: Any) -> dict[str, Any]:
    precision = getattr(predictor, "_sam3_runtime_precision", None)
    summary: dict[str, Any] = {
        "precision": precision.to_dict() if precision is not None else None,
        "visual_feature_cache": bool(
            getattr(predictor, "_sam3_cache_visual_features", False)
        ),
    }
    try:
        backbone = _resolve_visual_backbone(predictor)
    except AttributeError:
        return summary
    if getattr(backbone, "_sam3_visual_cache_enabled", False):
        summary["visual_feature_cache_hits"] = int(backbone._sam3_visual_cache_hits)
        summary["visual_feature_cache_misses"] = int(backbone._sam3_visual_cache_misses)
    return summary
