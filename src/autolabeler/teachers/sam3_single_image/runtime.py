"""SAM3 session, model setup, threshold, and output conversion helpers."""

from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from autolabeler.teachers.sam3_runtime_adapter import configure_sam3_runtime

from .types import BOX_KEYS, MASK_KEYS, SCORE_KEYS


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] in {"'", '"'} and value[-1] == value[0]:
        return value[1:-1]
    return value


def load_prompt_config(path: str | Path) -> dict[str, list[str]]:
    prompts: dict[str, list[str]] = {}
    current_key: str | None = None
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if indent == 0 and ":" in stripped:
            key, value = [x.strip() for x in stripped.split(":", 1)]
            current_key = key
            if value:
                prompts[key] = [
                    _strip_quotes(x.strip())
                    for x in value.strip("[]").split(",")
                    if x.strip()
                ]
            else:
                prompts[key] = []
            continue
        if indent >= 2 and stripped.startswith("- ") and current_key is not None:
            prompts[current_key].append(_strip_quotes(stripped[2:].strip()))
    return prompts


def apply_simple_colour_correction_rgb(image_rgb: np.ndarray) -> np.ndarray:
    """Apply the main BGR SimpleWB pipeline and return contiguous RGB."""
    from autolabeler.camera.colour_correction import simple_colour_correction  # noqa: WPS433

    array = np.asarray(image_rgb, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"image_rgb must have shape [H,W,3], got {array.shape}")
    image_bgr = np.ascontiguousarray(array[..., ::-1])
    corrected_bgr = simple_colour_correction(image_bgr)
    corrected_rgb = np.asarray(corrected_bgr, dtype=np.uint8)[..., ::-1]
    return np.ascontiguousarray(corrected_rgb)


def start_session(
    predictor: Any,
    image_path: Path,
    *,
    image_rgb: np.ndarray | None = None,
) -> str:
    if image_rgb is None:
        resource: Any = str(image_path)
    else:
        from PIL import Image  # noqa: WPS433

        array = np.asarray(image_rgb)
        if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
            raise ValueError(
                f"image_rgb must be uint8[H,W,3], got {array.shape} {array.dtype}"
            )
        resource = [Image.fromarray(np.ascontiguousarray(array))]
    response = predictor.handle_request(
        request={
            "type": "start_session",
            "resource_path": resource,
        }
    )
    return str(response["session_id"])


def reset_session(predictor: Any, session_id: str) -> None:
    predictor.handle_request(
        request={"type": "reset_session", "session_id": session_id}
    )


def add_text_prompt(
    predictor: Any, *, session_id: str, prompt: str, min_score: float
) -> dict[str, Any]:
    return predictor.handle_request(
        request={
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "text": prompt,
            "output_prob_thresh": float(min_score),
        }
    )


def close_session(predictor: Any, session_id: str) -> None:
    predictor.handle_request(
        request={"type": "close_session", "session_id": session_id}
    )


def extract_arrays(
    outputs: Any, *, height: int, width: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if outputs is None:
        return empty_arrays(height, width)
    if isinstance(outputs, dict):
        masks_value = first_present(outputs, MASK_KEYS)
        scores_value = first_present(outputs, SCORE_KEYS)
        boxes_value = first_present(outputs, BOX_KEYS)
    elif isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
        masks_value = outputs[1]
        scores_value = outputs[2] if len(outputs) >= 3 else None
        boxes_value = outputs[3] if len(outputs) >= 4 else None
    else:
        return empty_arrays(height, width)

    masks = normalize_masks(masks_value, height=height, width=width)
    scores = normalize_scores(scores_value, masks.shape[0])
    boxes = normalize_boxes(boxes_value, masks.shape[0])
    return masks, scores, boxes


def empty_arrays(height: int, width: int) -> tuple[np.ndarray, np.ndarray, None]:
    return (
        np.zeros((0, height, width), dtype=bool),
        np.zeros((0,), dtype=np.float32),
        None,
    )


def normalize_masks(value: Any, *, height: int, width: int) -> np.ndarray:
    arr = to_numpy(value)
    if arr is None:
        return np.zeros((0, height, width), dtype=bool)
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"Expected masks [N,H,W], got {arr.shape}")
    return arr if arr.dtype == np.bool_ else arr > 0


def normalize_scores(value: Any, count: int) -> np.ndarray:
    arr = to_numpy(value)
    if arr is None:
        return np.ones((count,), dtype=np.float32)
    arr = arr.reshape(-1).astype(np.float32, copy=False)
    if arr.shape != (count,):
        return np.ones((count,), dtype=np.float32)
    return arr


def normalize_boxes(value: Any, count: int) -> np.ndarray | None:
    arr = to_numpy(value)
    if arr is None:
        return None
    arr = arr.astype(np.float32, copy=False)
    if arr.shape == (count, 4):
        return arr
    return None


def first_present(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def configure_sam3_imports(*, sam3_root: Path | None, model_dir: Path | None) -> None:
    if sam3_root is not None:
        sys.path.insert(0, str(sam3_root))
    if model_dir is not None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HOME", str(model_dir))


def resolve_model_dir(
    *, sam3_root: Path | None, sam3_model_path: str | None
) -> Path | None:
    if sam3_model_path:
        model_dir = Path(sam3_model_path).expanduser().resolve()
    elif sam3_root is not None:
        model_dir = sam3_root / "sam3.1"
    else:
        return None
    if not (model_dir / "sam3.1_multiplex.pt").is_file():
        raise FileNotFoundError(
            f"Missing SAM3.1 checkpoint: {model_dir / 'sam3.1_multiplex.pt'}"
        )
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"Missing SAM3.1 config: {model_dir / 'config.json'}")
    return model_dir


def build_predictor(
    *,
    model_dir: Path | None,
    use_fa3: bool,
    min_score: float,
    inference_precision: str = "auto",
    cache_visual_features: bool = False,
) -> Any:
    import sam3.model_builder as sam3_model_builder  # type: ignore # noqa: WPS433

    builder = sam3_model_builder.build_sam3_multiplex_video_predictor
    kwargs: dict[str, Any] = {
        "use_fa3": bool(use_fa3),
        "async_loading_frames": False,
        "default_output_prob_thresh": float(min_score),
    }
    if model_dir is not None:
        checkpoint_path = model_dir / "sam3.1_multiplex.pt"
        patch_hf_checkpoint_download(
            sam3_model_builder, checkpoint_path=checkpoint_path, model_dir=model_dir
        )
        kwargs["checkpoint_path"] = str(checkpoint_path)
    params = inspect.signature(builder).parameters
    predictor = builder(
        **{key: value for key, value in kwargs.items() if key in params}
    )
    set_predictor_detection_threshold(predictor, min_score)
    precision = configure_sam3_runtime(
        predictor,
        requested_precision=inference_precision,
        cache_visual_features=cache_visual_features,
        use_fa3=use_fa3,
    )
    print(
        "SAM3 runtime: "
        f"device={precision.device_name}, precision={precision.effective}, "
        f"visual_cache={'on' if cache_visual_features else 'off'} "
        f"({precision.reason})",
        file=sys.stderr,
        flush=True,
    )
    return predictor


def set_predictor_detection_threshold(predictor: Any, min_score: float) -> None:
    score = validate_probability(min_score, name="min_score")
    if not hasattr(predictor, "default_output_prob_thresh"):
        raise AttributeError(
            "SAM3 predictor has no default_output_prob_thresh attribute"
        )
    if not hasattr(predictor, "model"):
        raise AttributeError("SAM3 predictor has no model attribute")
    model_attributes = (
        "score_threshold_detection",
        "image_only_det_thresh",
        "new_det_thresh",
    )
    missing = [name for name in model_attributes if not hasattr(predictor.model, name)]
    if missing:
        raise AttributeError(
            f"SAM3 predictor model does not expose threshold attributes: {missing}"
        )
    predictor.default_output_prob_thresh = score
    for name in model_attributes:
        setattr(predictor.model, name, score)


def load_label_min_scores(
    path: str | Path | None,
    *,
    known_labels: set[str],
) -> dict[str, float]:
    if path is None:
        return {}
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Label min-score config does not exist: {config_path}")
    if config_path.suffix.lower() == ".json":
        raw_mapping = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw_mapping, dict):
            raise ValueError(
                "Label min-score JSON must contain an object mapping label to score"
            )
    else:
        raw_mapping = parse_simple_score_yaml(config_path)
    scores: dict[str, float] = {}
    for raw_label, raw_score in raw_mapping.items():
        label = str(raw_label).strip()
        if not label:
            raise ValueError(f"Empty label in min-score config: {config_path}")
        if label in scores:
            raise ValueError(
                f"Duplicate label {label!r} in min-score config: {config_path}"
            )
        scores[label] = validate_probability(
            raw_score, name=f"min score for label {label!r}"
        )
    unknown_labels = sorted(set(scores) - known_labels)
    if unknown_labels:
        raise ValueError(
            "Label min-score config contains labels absent from the active prompt/classes configs: "
            f"{unknown_labels}"
        )
    return scores


def parse_simple_score_yaml(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.split("#", 1)[0].strip()
        if not line or line == "{}":
            continue
        if ":" not in line:
            raise ValueError(
                f"Expected 'label: score' in {path}:{line_number}, got {raw_line!r}"
            )
        raw_label, raw_score = line.split(":", 1)
        label = strip_matching_quotes(raw_label.strip())
        score = raw_score.strip()
        if not score:
            raise ValueError(
                f"Missing score for label {label!r} in {path}:{line_number}"
            )
        if label in mapping:
            raise ValueError(f"Duplicate label {label!r} in {path}:{line_number}")
        mapping[label] = score
    return mapping


def strip_matching_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] in {"'", '"'} and value[-1] == value[0]:
        return value[1:-1]
    return value


def validate_probability(value: Any, *, name: str) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number in [0,1], got {value!r}") from exc
    if not np.isfinite(score) or score < 0.0 or score > 1.0:
        raise ValueError(f"{name} must be in [0,1], got {value!r}")
    return score


def patch_hf_checkpoint_download(
    model_builder_module: Any, *, checkpoint_path: Path, model_dir: Path
) -> None:
    def _local_download_ckpt_from_hf(*args: Any, **kwargs: Any) -> str:
        return str(checkpoint_path)

    model_builder_module.download_ckpt_from_hf = _local_download_ckpt_from_hf
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HOME", str(model_dir))
