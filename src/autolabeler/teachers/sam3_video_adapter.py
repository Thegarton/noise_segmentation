from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Sam3VideoFrame:
    frame_id: str
    video_frame_index: int
    image_path: str | None = None
    lidar_timestamp_us: int | None = None
    video_timestamp_us: int | None = None
    delta_ms: float | None = None


@dataclass(frozen=True)
class Sam3VideoResult:
    masks: np.ndarray
    scores: np.ndarray
    labels: np.ndarray
    prompts: np.ndarray
    boxes_xyxy: np.ndarray | None = None
    track_ids: np.ndarray | None = None


@dataclass(frozen=True)
class SemanticCameraMask:
    semantic_mask: np.ndarray
    confidence: np.ndarray
    accepted_instances: int
    ignored_instances: int
    unknown_labels: list[str]
    class_pixel_counts: dict[str, int]


def build_sam3_video_command(
    *,
    video: str | Path,
    prompt_config: str | Path,
    frames_json: str | Path,
    output_dir: str | Path,
    sam3_video_script: str | Path,
    conda_env: str,
    conda_prefix: str | Path | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    env_selector = ["-p", str(conda_prefix)] if conda_prefix else ["-n", conda_env]
    command = [
        "conda",
        "run",
        *env_selector,
        "python",
        str(sam3_video_script),
        "--video",
        str(video),
        "--prompts",
        str(prompt_config),
        "--frames-json",
        str(frames_json),
        "--output-dir",
        str(output_dir),
    ]
    if extra_args:
        command.extend(extra_args)
    return command


def run_sam3_video_teacher(
    *,
    video: str | Path,
    prompt_config: str | Path,
    frames_json: str | Path,
    output_dir: str | Path,
    sam3_video_script: str | Path,
    conda_env: str,
    conda_prefix: str | Path | None = None,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess:
    command = build_sam3_video_command(
        video=video,
        prompt_config=prompt_config,
        frames_json=frames_json,
        output_dir=output_dir,
        sam3_video_script=sam3_video_script,
        conda_env=conda_env,
        conda_prefix=conda_prefix,
        extra_args=extra_args,
    )
    return subprocess.run(command, check=True)


def load_camera_frame_manifest(path: str | Path) -> list[Sam3VideoFrame]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    frames_payload = payload.get("frames")
    if not isinstance(frames_payload, list):
        raise ValueError(f"camera_frame_manifest.json has no frames list: {path}")

    frames: list[Sam3VideoFrame] = []
    for row in frames_payload:
        if not isinstance(row, dict) or not row.get("synced", False):
            continue
        video_frame_index = row.get("video_frame_index")
        frame_id = row.get("frame_id")
        if frame_id is None or video_frame_index is None:
            continue
        frames.append(
            Sam3VideoFrame(
                frame_id=str(frame_id),
                video_frame_index=int(video_frame_index),
                image_path=str(row["image_path"]) if row.get("image_path") is not None else None,
                lidar_timestamp_us=_optional_int(row.get("lidar_timestamp_us")),
                video_timestamp_us=_optional_int(row.get("video_timestamp_us")),
                delta_ms=_optional_float(row.get("delta_ms")),
            )
        )
    return frames


def write_frames_json(path: str | Path, frames: list[Sam3VideoFrame]) -> None:
    payload = {
        "version": 1,
        "frames": [
            _frame_to_json(frame)
            for frame in frames
        ],
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _frame_to_json(frame: Sam3VideoFrame) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "frame_id": frame.frame_id,
        "video_frame_index": frame.video_frame_index,
    }
    if frame.image_path is not None:
        payload["image_path"] = frame.image_path
    return payload


def load_sam3_video_result(path: str | Path) -> Sam3VideoResult:
    with np.load(path, allow_pickle=False) as data:
        masks = np.asarray(data["masks"])
        scores = np.asarray(data["scores"], dtype=np.float32)
        labels = np.asarray(data["labels"]).astype(str)
        prompts = np.asarray(data["prompts"]).astype(str)
        boxes = np.asarray(data["boxes_xyxy"], dtype=np.float32) if "boxes_xyxy" in data.files else None
        track_ids = np.asarray(data["track_ids"], dtype=np.int64) if "track_ids" in data.files else None

    if masks.ndim != 3 or masks.dtype != np.bool_:
        raise ValueError(f"SAM3 video masks must be bool[N,H,W], got {masks.shape} {masks.dtype}")
    if scores.shape != (masks.shape[0],):
        raise ValueError(f"SAM3 video scores shape must be {(masks.shape[0],)}, got {scores.shape}")
    if labels.shape != (masks.shape[0],) or prompts.shape != (masks.shape[0],):
        raise ValueError("SAM3 video labels/prompts must have one value per mask")
    if boxes is not None and boxes.shape != (masks.shape[0], 4):
        raise ValueError(f"SAM3 video boxes_xyxy must have shape {(masks.shape[0], 4)}, got {boxes.shape}")
    if track_ids is not None and track_ids.shape != (masks.shape[0],):
        raise ValueError(f"SAM3 video track_ids must have shape {(masks.shape[0],)}, got {track_ids.shape}")
    return Sam3VideoResult(masks=masks, scores=scores, labels=labels, prompts=prompts, boxes_xyxy=boxes, track_ids=track_ids)


def labels_to_ids_from_prompt_config(project_classes_path: str | Path, prompt_config_path: str | Path) -> dict[str, int]:
    from autolabeler.data.class_config import load_semantic_classes

    classes = load_semantic_classes(str(project_classes_path))
    prompt_config = load_prompt_config(prompt_config_path)
    label_to_id: dict[str, int] = {}
    for label in prompt_config:
        if label not in classes:
            raise ValueError(f"SAM3 prompt label {label!r} is not in project class config")
        label_to_id[label] = int(classes[label])
    return label_to_id


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
                prompts[key] = [_strip_quotes(x.strip()) for x in value.strip("[]").split(",") if x.strip()]
            else:
                prompts[key] = []
            continue
        if indent >= 2 and stripped.startswith("- ") and current_key is not None:
            prompts[current_key].append(_strip_quotes(stripped[2:].strip()))
    return prompts


def class_names_from_mapping(mapping: dict[str, int]) -> list[str]:
    max_id = max(class_id for class_id in mapping.values() if class_id != 255)
    names = [""] * (max_id + 1)
    for name, class_id in mapping.items():
        if class_id == 255:
            continue
        names[class_id] = name
    return names


def sam3_video_to_semantic_mask(
    result: Sam3VideoResult,
    *,
    label_to_id: dict[str, int],
    min_score: float,
) -> SemanticCameraMask:
    height, width = result.masks.shape[1:]
    semantic = np.zeros((height, width), dtype=np.uint16)
    confidence = np.zeros((height, width), dtype=np.float32)
    unknown_labels: list[str] = []
    accepted_instances = 0
    ignored_instances = 0

    order = np.argsort(result.scores, kind="stable")
    for idx in order:
        score = float(result.scores[idx])
        label = str(result.labels[idx])
        if score < min_score:
            ignored_instances += 1
            continue
        class_id = label_to_id.get(label)
        if class_id is None:
            ignored_instances += 1
            if label not in unknown_labels:
                unknown_labels.append(label)
            continue
        mask = result.masks[idx]
        accept = mask & (score >= confidence)
        if not np.any(accept):
            ignored_instances += 1
            continue
        semantic[accept] = np.uint16(class_id)
        confidence[accept] = np.float32(score)
        accepted_instances += 1

    class_pixel_counts: dict[str, int] = {}
    for label, class_id in label_to_id.items():
        count = int(np.count_nonzero(semantic == class_id))
        if count:
            class_pixel_counts[label] = count

    return SemanticCameraMask(
        semantic_mask=semantic,
        confidence=confidence,
        accepted_instances=accepted_instances,
        ignored_instances=ignored_instances,
        unknown_labels=unknown_labels,
        class_pixel_counts=class_pixel_counts,
    )


def save_semantic_overlay(
    image_path: str | Path,
    output_path: str | Path,
    *,
    semantic_mask: np.ndarray,
    alpha: float = 0.45,
) -> bool:
    image_path = Path(image_path)
    if not image_path.is_file():
        return False
    cv2 = _import_cv2()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return False
    mask = np.asarray(semantic_mask)
    if mask.shape[:2] != image.shape[:2]:
        raise ValueError(f"Overlay mask shape {mask.shape} does not match image shape {image.shape[:2]}")

    overlay = image.copy()
    labels = [int(x) for x in np.unique(mask) if int(x) != 0]
    for label in labels:
        color = _stable_bgr_color(label)
        overlay[mask == label] = (np.asarray(color, dtype=np.float32) * alpha + overlay[mask == label] * (1.0 - alpha)).astype(np.uint8)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out), overlay))


def copy_frame_image(image_path: str | None, output_path: str | Path) -> str | None:
    if image_path is None:
        return None
    source = Path(image_path)
    if not source.is_file():
        return None
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, out)
    return str(out)


def _stable_bgr_color(label: int) -> tuple[int, int, int]:
    value = (label * 1103515245 + 12345) & 0xFFFFFF
    return int(value & 0xFF), int((value >> 8) & 0xFF), int((value >> 16) & 0xFF)


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] in {"'", '"'} and value[-1] == value[0]:
        return value[1:-1]
    return value


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _import_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError("SAM3 video overlay requires OpenCV. Install opencv-python or use an env with cv2.") from exc
    return cv2
