"""Minimal SAM3.1 Multiplex adapter used by the dataset builder."""

from __future__ import annotations

import inspect
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


MASK_KEYS = ("masks", "pred_masks", "out_binary_masks", "video_res_masks", "mask_logits", "out_mask_logits")
SCORE_KEYS = ("scores", "pred_scores", "object_scores", "ious", "obj_scores", "out_probs")
BOX_KEYS = ("boxes_xyxy", "boxes", "pred_boxes", "out_boxes_xywh")


@dataclass(frozen=True)
class VehicleDetection:
    label: str
    prompt: str
    score: float
    mask: np.ndarray
    box: np.ndarray | None = None


class Sam3VehicleDetector:
    """Load one predictor and reuse it for all prepared images."""

    def __init__(
        self,
        *,
        sam3_root: str | Path,
        model_dir: str | Path,
        min_score: float,
        use_fa3: bool = False,
        cache_visual_features: bool = True,
    ) -> None:
        self.sam3_root = Path(sam3_root).expanduser().resolve()
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.min_score = _validate_probability(min_score)
        _configure_imports(self.sam3_root, self.model_dir)
        self.cache_visual_features = bool(cache_visual_features)
        self.predictor = _build_predictor(
            self.model_dir,
            use_fa3=use_fa3,
            min_score=self.min_score,
            cache_visual_features=self.cache_visual_features,
        )

    def detect(self, image_path: str | Path, *, prompts: Sequence[str]) -> list[VehicleDetection]:
        return self.detect_labeled(
            image_path,
            labeled_prompts=[("vehicle", str(prompt)) for prompt in prompts],
        )

    def detect_labeled(
        self,
        image_path: str | Path,
        *,
        labeled_prompts: Sequence[tuple[str, str]],
    ) -> list[VehicleDetection]:
        from PIL import Image  # noqa: WPS433

        path = Path(image_path).expanduser().resolve()
        width, height = Image.open(path).size
        _clear_visual_feature_cache(self.predictor)
        session_id = _start_session(self.predictor, path)
        detections: list[VehicleDetection] = []
        try:
            for label, prompt in labeled_prompts:
                _reset_session(self.predictor, session_id)
                _set_threshold(self.predictor, self.min_score)
                response = self.predictor.handle_request(
                    request={
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": 0,
                        "text": str(prompt),
                        "output_prob_thresh": self.min_score,
                    }
                )
                masks, scores, boxes = _extract_arrays(response.get("outputs"), height=height, width=width)
                for index, mask in enumerate(masks):
                    score = float(scores[index])
                    if score < self.min_score:
                        continue
                    detections.append(
                        VehicleDetection(
                            label=str(label),
                            prompt=str(prompt),
                            score=score,
                            mask=mask,
                            box=None if boxes is None else boxes[index],
                        )
                    )
        finally:
            try:
                self.predictor.handle_request(
                    request={"type": "close_session", "session_id": session_id}
                )
            finally:
                _clear_visual_feature_cache(self.predictor)
        return detections


def _configure_imports(sam3_root: Path, model_dir: Path) -> None:
    if not (model_dir / "sam3.1_multiplex.pt").is_file():
        raise FileNotFoundError(f"Missing SAM3.1 checkpoint: {model_dir / 'sam3.1_multiplex.pt'}")
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"Missing SAM3.1 config: {model_dir / 'config.json'}")
    sys.path.insert(0, str(sam3_root))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HOME", str(model_dir))


def _build_predictor(
    model_dir: Path,
    *,
    use_fa3: bool,
    min_score: float,
    cache_visual_features: bool = True,
) -> Any:
    import sam3.model_builder as model_builder  # type: ignore # noqa: WPS433

    checkpoint = model_dir / "sam3.1_multiplex.pt"
    model_builder.download_ckpt_from_hf = lambda *args, **kwargs: str(checkpoint)
    builder = model_builder.build_sam3_multiplex_video_predictor
    kwargs = {
        "use_fa3": bool(use_fa3),
        "async_loading_frames": False,
        "default_output_prob_thresh": min_score,
        "checkpoint_path": str(checkpoint),
    }
    parameters = inspect.signature(builder).parameters
    predictor = builder(**{key: value for key, value in kwargs.items() if key in parameters})
    _set_threshold(predictor, min_score)
    if cache_visual_features:
        _enable_single_image_visual_cache(predictor)
    predictor._sam3_cache_visual_features = bool(cache_visual_features)
    return predictor


def _resolve_visual_backbone(predictor: Any) -> Any:
    model = getattr(predictor, "model", None)
    detector = getattr(model, "detector", None)
    backbone = getattr(detector, "backbone", None)
    if backbone is None or not callable(getattr(backbone, "forward_image", None)):
        raise AttributeError("SAM3 predictor does not expose model.detector.backbone.forward_image")
    return backbone


def _enable_single_image_visual_cache(predictor: Any) -> None:
    backbone = _resolve_visual_backbone(predictor)
    if getattr(backbone, "_sam3_visual_cache_enabled", False):
        return

    backbone._sam3_original_forward_image = backbone.forward_image
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


def _clear_visual_feature_cache(predictor: Any) -> None:
    try:
        backbone = _resolve_visual_backbone(predictor)
    except AttributeError:
        return
    if getattr(backbone, "_sam3_visual_cache_enabled", False):
        backbone._sam3_visual_cache_value = None


def _start_session(predictor: Any, image_path: Path) -> str:
    response = predictor.handle_request(request={"type": "start_session", "resource_path": str(image_path)})
    return str(response["session_id"])


def _reset_session(predictor: Any, session_id: str) -> None:
    predictor.handle_request(request={"type": "reset_session", "session_id": session_id})


def _set_threshold(predictor: Any, score: float) -> None:
    predictor.default_output_prob_thresh = float(score)
    for name in ("score_threshold_detection", "image_only_det_thresh", "new_det_thresh"):
        if not hasattr(predictor.model, name):
            raise AttributeError(f"SAM3 predictor model has no threshold attribute {name!r}")
        setattr(predictor.model, name, float(score))


def _extract_arrays(outputs: Any, *, height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if not isinstance(outputs, dict):
        return np.zeros((0, height, width), dtype=bool), np.zeros((0,), dtype=np.float32), None
    masks_value = _first_present(outputs, MASK_KEYS)
    scores_value = _first_present(outputs, SCORE_KEYS)
    boxes_value = _first_present(outputs, BOX_KEYS)
    masks = _to_numpy(masks_value)
    if masks is None:
        masks = np.zeros((0, height, width), dtype=bool)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None]
    if masks.ndim != 3:
        raise ValueError(f"Expected SAM3 masks [N,H,W], got {masks.shape}")
    masks = masks if masks.dtype == np.bool_ else masks > 0
    scores = _to_numpy(scores_value)
    if scores is None or scores.size != masks.shape[0]:
        scores = np.ones((masks.shape[0],), dtype=np.float32)
    else:
        scores = scores.reshape(-1).astype(np.float32, copy=False)
    boxes = _to_numpy(boxes_value)
    if boxes is not None:
        boxes = boxes.astype(np.float32, copy=False)
        if boxes.shape != (masks.shape[0], 4):
            boxes = None
    return masks, scores, boxes


def _first_present(mapping: dict[str, Any], keys: Sequence[str]) -> Any | None:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _validate_probability(value: float) -> float:
    score = float(value)
    if not np.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"min_score must be in [0,1], got {value}")
    return score
