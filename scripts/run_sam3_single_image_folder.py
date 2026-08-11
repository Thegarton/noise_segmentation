#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from autolabeler.data.class_config import load_semantic_classes  # noqa: E402
from autolabeler.teachers.sam3_runtime_adapter import (  # noqa: E402
    clear_visual_feature_cache,
    configure_sam3_runtime,
    sam3_runtime_summary,
)

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
                prompts[key] = [_strip_quotes(x.strip()) for x in value.strip("[]").split(",") if x.strip()]
            else:
                prompts[key] = []
            continue
        if indent >= 2 and stripped.startswith("- ") and current_key is not None:
            prompts[current_key].append(_strip_quotes(stripped[2:].strip()))
    return prompts

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MASK_KEYS = ("masks", "pred_masks", "out_binary_masks", "video_res_masks", "mask_logits", "out_mask_logits")
SCORE_KEYS = ("scores", "pred_scores", "object_scores", "ious", "obj_scores", "out_probs")
BOX_KEYS = ("boxes_xyxy", "boxes", "pred_boxes", "out_boxes_xywh")


@dataclass(frozen=True)
class Sam3Instance:
    label: str
    class_id: int
    prompt: str
    score: float
    mask: np.ndarray
    # [x_min, y_min, width, height] in 0-1.0 format
    box: np.ndarray | None = None
    source_label: str | None = None
    orientation_label: str | None = None
    orientation_score: float | None = None
    orientation_margin: float | None = None
    orientation_probabilities: tuple[float, float, float] | None = None
    orientation_fallback: bool = False
    orientation_fallback_reason: str | None = None


@dataclass(frozen=True)
class ImageResult:
    image_path: Path
    output_dir: Path
    image_size: tuple[int, int]
    instances: list[Sam3Instance]
    class_pixel_counts: dict[str, int]
    overlay_instances: tuple[Sam3Instance, ...] = ()
    projection_path: str | None = None
    projection_copy_path: str | None = None
    orientation_predictions: tuple[dict[str, Any], ...] = ()
    mask_projection_path: str | None = None
    projection_error: str | None = None

def process_image(
    *,
    predictor: Any,
    image_path: Path,
    output_dir: Path,
    flat_prompts: list[tuple[str, str]],
    label_to_id: dict[str, int],
    min_score: float,
    label_min_scores: dict[str, float],
    prompt_log: bool,
    projection_dir: Path | None,
    min_mask_size: int,
    require_projection: bool,
    vehicle_orientation_classifier: Any | None = None,
    vehicle_prompt_label: str = "vehicle",
    vehicle_class_mapping: dict[str, tuple[str, int]] | None = None,
    vehicle_orientation_min_confidence: float = 0.70,
    vehicle_orientation_min_margin: float = 0.10,
    vehicle_orientation_nms_iou: float = 0.80,
) -> ImageResult:
    from PIL import Image  # noqa: WPS433

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    clear_visual_feature_cache(predictor)
    session_id = start_session(predictor, image_path)
    instances: list[Sam3Instance] = []
    try:
        for prompt_idx, (label, prompt) in enumerate(flat_prompts, start=1):
            detection_min_score = float(label_min_scores.get(label, min_score))
            if prompt_log:
                print(f"  [{prompt_idx:03d}/{len(flat_prompts):03d}] {label}: {prompt}", file=sys.stderr, flush=True)
            set_predictor_detection_threshold(predictor, detection_min_score)

            response = add_text_prompt(
                predictor,
                session_id=session_id,
                prompt=prompt,
                min_score=detection_min_score
            )
            masks, scores, boxes = extract_arrays(response.get("outputs"), height=height, width=width)
            for mask_index in range(masks.shape[0]):
                score = float(scores[mask_index])
                if score < detection_min_score:
                    continue
                if label in label_to_id:
                    class_id = int(label_to_id[label])
                elif (
                    vehicle_orientation_classifier is not None
                    and vehicle_class_mapping is not None
                    and label == vehicle_prompt_label
                ):
                    class_id = int(vehicle_class_mapping["front"][1])
                else:
                    raise KeyError(f"No semantic or routed class id for prompt label {label!r}")
                instances.append(
                    Sam3Instance(
                        label=label,
                        class_id=class_id,
                        prompt=prompt,
                        score=score,
                        mask=masks[mask_index].astype(bool, copy=False),
                        box=None if boxes is None else boxes[mask_index],
                        source_label=label if label == vehicle_prompt_label else None,
                    )
                )
    finally:
        clear_visual_feature_cache(predictor)
        close_session(predictor, session_id)

    instances = deduplicate_instances_by_label(
            instances,
            iou_threshold=vehicle_orientation_nms_iou,
        )

    if vehicle_orientation_classifier is not None:
        if vehicle_class_mapping is None:
            raise ValueError("vehicle_class_mapping is required when the orientation classifier is enabled")
        instances = apply_vehicle_orientation(
            image_rgb=np.asarray(image, dtype=np.uint8),
            instances=instances,
            classifier=vehicle_orientation_classifier,
            vehicle_prompt_label=vehicle_prompt_label,
            class_mapping=vehicle_class_mapping,
            min_confidence=vehicle_orientation_min_confidence,
            min_margin=vehicle_orientation_min_margin,
            nms_iou=vehicle_orientation_nms_iou,
            min_mask_size=min_mask_size,
        )

    semantic_mask, confidence, class_pixel_counts = build_semantic_outputs(
        instances=instances,
        label_to_id=label_to_id,
        shape=(height, width),
        min_mask_size=min_mask_size
    )
    overlay_instances = collect_overlay_instances(
        instances=instances,
        semantic_mask=semantic_mask,
        confidence=confidence,
        min_mask_size=min_mask_size,
    )
    projection_info = save_image_outputs(
        image=image,
        image_path=image_path,
        output_dir=output_dir,
        instances=instances,
        overlay_instances=list(overlay_instances),
        semantic_mask=semantic_mask,
        confidence=confidence,
        projection_dir=projection_dir,
        require_projection=require_projection,
    )
    return ImageResult(
        image_path=image_path,
        output_dir=output_dir,
        image_size=(width, height),
        instances=instances,
        class_pixel_counts=class_pixel_counts,
        overlay_instances=overlay_instances,
        orientation_predictions=tuple(
            instance_orientation_metadata(item)
            for item in instances
            if item.orientation_label is not None
        ),
        projection_path=projection_info.get("projection_path"),
        projection_copy_path=projection_info.get("projection_copy"),
        mask_projection_path=projection_info.get("mask_projection"),
        projection_error=projection_info.get("projection_error"),
    )


def start_session(predictor: Any, image_path: Path) -> str:
    response = predictor.handle_request(
        request={
            "type": "start_session",
            "resource_path": str(image_path),
        }
    )
    return str(response["session_id"])


def reset_session(predictor: Any, session_id: str) -> None:
    predictor.handle_request(request={"type": "reset_session", "session_id": session_id})


def add_text_prompt(predictor: Any, *, session_id: str, prompt: str, min_score: float) -> dict[str, Any]:
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
    predictor.handle_request(request={"type": "close_session", "session_id": session_id})



CLASS_OVERRIDES = {
    "ground_markings": {"epoxy_floor"},
    "license_plate&taillights": {"vehicle"},
}

def build_semantic_outputs(
    *,
    instances: list[Sam3Instance],
    label_to_id: dict[str, int],
    shape: tuple[int, int],
    min_mask_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    height, width = shape
    if min_mask_size < 0:
        raise ValueError(f"min_mask_size must be non-negative, got {min_mask_size}")

    semantic_mask = np.zeros((height, width), dtype=np.uint16)
    confidence = np.zeros((height, width), dtype=np.float32)

    override_map_ids = {}
    subordinate_map_ids = {}

    for dom_label, sub_labels in CLASS_OVERRIDES.items():
        if dom_label in label_to_id:
            dom_id = label_to_id[dom_label]
            if dom_id not in override_map_ids:
                override_map_ids[dom_id] = set()
            for sub_label in sub_labels:
                if sub_label in label_to_id:
                    sub_id = label_to_id[sub_label]
                    override_map_ids[dom_id].add(sub_id)

                    if sub_id not in subordinate_map_ids:
                        subordinate_map_ids[sub_id] = set()
                    subordinate_map_ids[sub_id].add(dom_id)

    ordered_instances = sorted(
        instances,
        key=lambda item: -float(item.score),
    )

    for item in ordered_instances:
        item_score = float(item.score)
        if not np.isfinite(item_score):
            continue

        current_mask = np.asarray(item.mask, dtype=bool)
        if current_mask.shape != (height, width):
            raise ValueError(
                f"Mask for {item.label!r} has shape {current_mask.shape}, "
                f"expected {(height, width)}"
            )

        if np.count_nonzero(current_mask) < min_mask_size:
            continue

        if reject_instance_by_geometry(item):
            continue
        if reject_instance_by_size(item):
            continue

        item_id = item.class_id


        accept = current_mask & (item_score > confidence)


        if item_id in override_map_ids:
            target_ids = list(override_map_ids[item_id])
            dom_mask = np.isin(semantic_mask, target_ids)

            accept = accept | (current_mask & dom_mask)


        if item_id in subordinate_map_ids:
            target_ids = list(subordinate_map_ids[item_id])
            sub_mask = np.isin(semantic_mask, target_ids)
            accept = accept & ~sub_mask

        if np.count_nonzero(accept) < min_mask_size:
            continue

        semantic_mask[accept] = np.uint16(item_id)
        confidence[accept] = np.float32(item_score)

    class_pixel_counts = {}
    for label, class_id in label_to_id.items():
        count = int(np.count_nonzero(semantic_mask == class_id))
        if count > 0 and count >= min_mask_size:
            class_pixel_counts[label] = count

    return semantic_mask, confidence, class_pixel_counts


def collect_overlay_instances(
    *,
    instances: list[Sam3Instance],
    semantic_mask: np.ndarray,
    confidence: np.ndarray,
    min_mask_size: int,
) -> tuple[Sam3Instance, ...]:
    """Return only instance fragments that are visible in the final overlay."""
    if semantic_mask.shape != confidence.shape:
        raise ValueError(
            "semantic_mask and confidence must have the same shape, got "
            f"{semantic_mask.shape} and {confidence.shape}"
        )

    claimed_pixels = np.zeros(semantic_mask.shape, dtype=bool)
    visible_instances: list[Sam3Instance] = []
    for item in sorted(instances, key=lambda value: float(value.score), reverse=True):
        item_score = float(item.score)
        if not np.isfinite(item_score):
            continue

        current_mask = np.asarray(item.mask, dtype=bool)
        if current_mask.shape != semantic_mask.shape:
            raise ValueError(
                f"Mask for {item.label!r} has shape {current_mask.shape}, "
                f"expected {semantic_mask.shape}"
            )
        if np.count_nonzero(current_mask) < min_mask_size:
            continue
        if reject_instance_by_geometry(item) or reject_instance_by_size(item):
            continue

        visible_mask = (
            current_mask
            & (semantic_mask == int(item.class_id))
            & np.isclose(confidence, np.float32(item_score), atol=1e-6)
            & ~claimed_pixels
        )
        if np.count_nonzero(visible_mask) < min_mask_size:
            continue

        claimed_pixels |= visible_mask
        visible_instances.append(replace(item, mask=visible_mask))

    return tuple(visible_instances)


ARRESTOR_MIN_Y_BOTTOM = 0.4
ARRESTOR_MIN_ASPECT_RATIO = 2.0

GEOMETRY_FILTER_RULES: dict[str, dict[str, float]] = {
    "arrestor": {
        "min_y_bottom": 0.4,
        "min_aspect_ratio": 2.0,
    },
    "wheel_chock": {
        "min_y_bottom": 0.4,
        "min_aspect_ratio": 1.0,
    },
    "underground_parking_sign" : {
        "min_y_bottom": 0.4,
        "min_aspect_ratio": 1.0,
    },
    "ground_markings": {
        "min_y_bottom": 0.4,
        "min_aspect_ratio": 0,
    },
    "induction_sign" : {
        "min_y_bottom": 0.50,
        "min_aspect_ratio": 4.0,
    },
    "parking_barrier_lock": {
        "min_y_bottom": 0.50,
        "min_aspect_ratio": 0,
    },
    "pillar_corner_guard": {
        "min_y_bottom": 0.2,
        "min_aspect_ratio": 0.7,
    }
}


SIZE_FILTER_RULES: dict[str, dict[str, float]] = {
    "arrestor": {
        "max_box_width": 0.4,
        "max_box_height": 0.15,
        "max_box_area": 0.07,
    },
    "wheel_chock": {
        "max_box_width": 0.4,
        "max_box_height": 0.15,
        "max_box_area": 0.07,
    }

}

def reject_instance_by_geometry(
    item: Sam3Instance,
    *,
    rules: Mapping[str, Mapping[str, float]] = GEOMETRY_FILTER_RULES,
) -> bool:
    rule = rules.get(item.label)
    if rule is None:
        return False

    try:
        min_y_bottom = float(rule["min_y_bottom"])
        min_aspect_ratio = float(rule["min_aspect_ratio"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid geometry filter rule for label {item.label!r}: {rule!r}"
        ) from exc
    if (
        not np.isfinite(min_y_bottom)
        or not 0.0 <= min_y_bottom <= 1.0
        or not np.isfinite(min_aspect_ratio)
        or min_aspect_ratio < 0.0
    ):
        raise ValueError(
            f"Invalid geometry filter thresholds for label {item.label!r}: "
            f"min_y_bottom={min_y_bottom}, min_aspect_ratio={min_aspect_ratio}"
        )

    if item.box is None:
        return True

    box = np.asarray(item.box, dtype=np.float32).reshape(-1)
    if box.shape != (4,) or not np.isfinite(box).all():
        return True

    x_min, y_min, box_width, box_height = (float(value) for value in box)
    if box_height <= 1e-6 or box_width < 0.0 or x_min < 0.0 or y_min < 0.0:
        return True
    y_bottom = y_min + box_height
    aspect_ratio = box_width / box_height

    if item.label in { "underground_parking_sign"}:
        return y_bottom > min_y_bottom or aspect_ratio < min_aspect_ratio

    if item.label in {"induction_sign"}:
        return y_bottom > min_y_bottom or aspect_ratio >  min_aspect_ratio

    if item.label == "ground_markings":
        return y_bottom < min_y_bottom

    if item.label in {"parking_barrier_lock"}:
        return y_bottom < min_y_bottom

    if item.label == "pillar_corner_guard":
        return y_bottom < min_y_bottom or aspect_ratio > min_aspect_ratio

    return y_bottom < min_y_bottom or aspect_ratio < min_aspect_ratio


def reject_instance_by_size(
    item: Sam3Instance,
    *,
    rules: Mapping[str, Mapping[str, float]] = SIZE_FILTER_RULES,
) -> bool:
    rule = rules.get(item.label)
    if rule is None:
        return False

    try:
        min_box_width = float(rule.get("min_box_width", 0.0))
        max_box_width = float(rule.get("max_box_width", 1.0))
        min_box_height = float(rule.get("min_box_height", 0.0))
        max_box_height = float(rule.get("max_box_height", 1.0))
        min_box_area = float(rule.get("min_box_area", 0.0))
        max_box_area = float(rule.get("max_box_area", 1.0))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid size filter rule for label {item.label!r}: {rule!r}"
        ) from exc
    if (
        not 0.0 <= min_box_width <= max_box_width <= 1.0
        or not 0.0 <= min_box_height <= max_box_height <= 1.0
        or not 0.0 <= min_box_area <= max_box_area <= 1.0
    ):
        raise ValueError(
            f"Invalid size filter thresholds for label {item.label!r}: "
            f"box_width=[{min_box_width},{max_box_width}], "
            f"box_height=[{min_box_height},{max_box_height}], "
            f"box_area=[{min_box_area},{max_box_area}]"
        )

    if item.box is None:
        return True

    box = np.asarray(item.box, dtype=np.float32).reshape(-1)
    if box.shape != (4,) or not np.isfinite(box).all():
        return True

    x_min, y_min, box_width, box_height = (float(value) for value in box)
    if box_height <= 1e-6 or box_width <= 1e-6 or x_min < 0.0 or y_min < 0.0:
        return True

    box_area = box_width * box_height
    return (
        box_width < min_box_width
        or box_width > max_box_width
        or box_height < min_box_height
        or box_height > max_box_height
        or box_area < min_box_area
        or box_area > max_box_area
    )



def resolve_vehicle_class_mapping(class_to_id: dict[str, int]) -> dict[str, tuple[str, int]]:
    class_names = {
        "front": "front_of_vehicle",
        "rear": "rear_of_vehicle",
        "side": "side_of_vehicle"
    }
    missing = [label for label in class_names.values() if label not in class_to_id]
    if missing:
        raise ValueError(f"Vehicle orientation classes are missing from classes yaml: {missing}")
    mapping = {
        orientation: (label, int(class_to_id[label]))
        for orientation, label in class_names.items()
    }
    class_ids = [class_id for _, class_id in mapping.values()]

    if len(set(class_ids)) != len(class_ids):
        raise ValueError(f"Vehicle orientation classes must have unique ids, got {mapping}")
    reserved = {
        int(class_to_id[label])
        for label in ("background", "ignore")
        if label in class_to_id
    }
    collisions = {
        label: class_id
        for label, class_id in mapping.values()
        if class_id in reserved
    }
    if collisions:
        raise ValueError(
            "Vehicle orientation class ids collide with background/ignore ids: "
            f"{collisions}"
        )
    return mapping



def build_vehicle_orientation_classifier(checkpoint_path: str | Path, *, device: str) -> Any:
    local_src = REPO_ROOT / "vehicle_orientation" / "src"
    if local_src.is_dir() and str(local_src) not in sys.path:
        sys.path.insert(0, str(local_src))
    try:
        from vehicle_orientation.model import VehicleOrientationClassifier  # type: ignore # noqa: WPS433
    except ImportError as exc:
        raise ImportError(
            "Vehicle orientation support is not installed. Run: "
            "pip install -e vehicle_orientation inside the SAM3 environment."
        ) from exc
    return VehicleOrientationClassifier(checkpoint_path, device=device)


def deduplicate_instances_by_label(
    instances: list[Sam3Instance],
    *,
    iou_threshold: float,
) -> list[Sam3Instance]:
    """Suppress duplicate masks from different prompts of the same semantic label."""
    validate_probability(iou_threshold, name="mask_nms_iou")
    if len(instances) < 2:
        return list(instances)

    local_src = REPO_ROOT / "vehicle_orientation" / "src"
    if local_src.is_dir() and str(local_src) not in sys.path:
        sys.path.insert(0, str(local_src))
    from vehicle_orientation.preprocessing import deduplicate_mask_indices  # type: ignore # noqa: WPS433

    indices_by_label: dict[str, list[int]] = {}
    for index, item in enumerate(instances):
        indices_by_label.setdefault(item.label, []).append(index)

    kept_indices: set[int] = set()
    for label_indices in indices_by_label.values():
        if len(label_indices) == 1:
            kept_indices.add(label_indices[0])
            continue
        keep_local = deduplicate_mask_indices(
            [instances[index].mask for index in label_indices],
            [instances[index].score for index in label_indices],
            iou_threshold=iou_threshold,
        )
        kept_indices.update(label_indices[index] for index in keep_local)

    return [item for index, item in enumerate(instances) if index in kept_indices]

def apply_vehicle_orientation(
    *,
    image_rgb: np.ndarray,
    instances: list[Sam3Instance],
    classifier: Any,
    vehicle_prompt_label: str,
    class_mapping: dict[str, tuple[str, int]],
    min_confidence: float,
    min_margin: float,
    nms_iou: float,
    min_mask_size: int,
) -> list[Sam3Instance]:
    local_src = REPO_ROOT / "vehicle_orientation" / "src"
    if local_src.is_dir() and str(local_src) not in sys.path:
        sys.path.insert(0, str(local_src))
    from vehicle_orientation.preprocessing import deduplicate_mask_indices  # type: ignore # noqa: WPS433

    vehicle_indices = [index for index, item in enumerate(instances) if item.label == vehicle_prompt_label]
    eligible_indices = [
        index
        for index in vehicle_indices
        if int(np.count_nonzero(np.asarray(instances[index].mask, dtype=bool))) >= min_mask_size
    ]
    if not eligible_indices:
        return [item for item in instances if item.label != vehicle_prompt_label]
    keep_local = deduplicate_mask_indices(
        [instances[index].mask for index in eligible_indices],
        [instances[index].score for index in eligible_indices],
        iou_threshold=nms_iou,
    )
    kept_indices = [eligible_indices[index] for index in keep_local]
    decisions = classifier.classify(
        image_rgb,
        [instances[index].mask for index in kept_indices],
        min_confidence=min_confidence,
        min_margin=min_margin,
    )
    if len(decisions) != len(kept_indices):
        raise ValueError(
            "Vehicle orientation classifier returned an unexpected number of predictions: "
            f"{len(decisions)} for {len(kept_indices)} masks"
        )

    replacements: dict[int, Sam3Instance] = {}
    for index, decision in zip(kept_indices, decisions):
        semantic_key = str(decision.semantic_label)
        if semantic_key not in class_mapping:
            raise ValueError(f"Orientation classifier returned unsupported semantic label {semantic_key!r}")
        target_label, target_id = class_mapping[semantic_key]
        probabilities = tuple(float(value) for value in decision.probabilities)
        if len(probabilities) != 3:
            raise ValueError(f"Orientation probabilities must contain front/rear/side, got {probabilities}")
        replacements[index] = replace(
            instances[index],
            label=target_label,
            class_id=target_id,
            source_label=vehicle_prompt_label,
            orientation_label=str(decision.predicted_class),
            orientation_score=float(decision.confidence),
            orientation_margin=float(decision.margin),
            orientation_probabilities=probabilities,
            orientation_fallback=bool(decision.fallback),
            orientation_fallback_reason=decision.fallback_reason,
        )

    output: list[Sam3Instance] = []
    for index, item in enumerate(instances):
        if item.label != vehicle_prompt_label:
            output.append(item)
        elif index in replacements:
            output.append(replacements[index])
    return output


def instance_orientation_metadata(item: Sam3Instance) -> dict[str, Any]:
    return {
        "source_label": item.source_label,
        "semantic_label": item.label,
        "class_id": int(item.class_id),
        "predicted_orientation": item.orientation_label,
        "orientation_score": item.orientation_score,
        "orientation_margin": item.orientation_margin,
        "orientation_probabilities": None
        if item.orientation_probabilities is None
        else list(item.orientation_probabilities),
        "fallback": bool(item.orientation_fallback),
        "fallback_reason": item.orientation_fallback_reason,
        "sam3_prompt": item.prompt,
        "sam3_score": float(item.score),
    }




def save_image_outputs(
    *,
    image: Any,
    image_path: Path,
    output_dir: Path,
    instances: list[Sam3Instance],
    overlay_instances: list[Sam3Instance] | None = None,
    semantic_mask: np.ndarray,
    confidence: np.ndarray,
    projection_dir: Path | None = None,
    require_projection: bool = False,
) -> dict[str, str | None]:
    from PIL import Image  # noqa: WPS433

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "semantic_mask.npy", semantic_mask)
    np.save(output_dir / "confidence.npy", confidence)
    save_instances_npz(output_dir / "instances.npz", instances, shape=semantic_mask.shape)

    image_np = np.asarray(image, dtype=np.uint8)
    visible_instances = instances if overlay_instances is None else overlay_instances
    overlay = make_overlay(
        image_np,
        semantic_mask,
        instances=visible_instances,
        confidence=confidence,
    )
    semantic_color = make_semantic_color(semantic_mask)
    Image.fromarray(overlay).save(output_dir / "overlay.jpg", quality=95)
    Image.fromarray(semantic_color).save(output_dir / "semantic_color.png")
    Image.fromarray(make_preview(image_np, semantic_color, overlay)).save(output_dir / "preview.jpg", quality=95)

    projection_info = save_mask_projection_preview(
        image_path=image_path,
        output_dir=output_dir,
        projection_dir=projection_dir,
        image_np=image_np,
        semantic_color=semantic_color,
        overlay=overlay,
        require_projection=require_projection,
    )
    image.save(output_dir / "image.jpg", quality=95)

    instances_json = [
        {
            "label": item.label,
            "class_id": item.class_id,
            "prompt": item.prompt,
            "score": item.score,
            "box": None if item.box is None else np.asarray(item.box, dtype=np.float32).tolist(),
            "source_label": item.source_label,
            "orientation_label": item.orientation_label,
        }
        for item in instances
    ]
    (output_dir / "instances.json").write_text(json.dumps(instances_json, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "source_image.txt").write_text(str(image_path), encoding="utf-8")
    return projection_info


def instance_to_log_json(item: Sam3Instance) -> dict[str, Any]:
    return {
        "label": item.label,
        "class_id": int(item.class_id),
        "prompt": item.prompt,
        "score": float(item.score),
        "box": None
        if item.box is None
        else np.asarray(item.box, dtype=np.float32).tolist(),
        "visible_pixel_count": int(np.count_nonzero(item.mask)),
        "source_label": item.source_label,
        "orientation_label": item.orientation_label,
        "orientation_score": item.orientation_score,
        "orientation_margin": item.orientation_margin,
        "orientation_probabilities": None
        if item.orientation_probabilities is None
        else list(item.orientation_probabilities),
        "orientation_fallback": bool(item.orientation_fallback),
        "orientation_fallback_reason": item.orientation_fallback_reason,
    }


def build_classes_log(
    *,
    result: ImageResult,
    prompt_config: Path,
    min_score: float,
    processing_time_seconds: float,
) -> dict[str, Any]:
    visible_instances = [
        instance_to_log_json(item)
        for item in result.overlay_instances
    ]
    return {
        "version": 2,
        "image_path": str(result.image_path),
        "prompt_config": str(prompt_config),
        "min_score": float(min_score),
        "object_count": len(visible_instances),
        "class_list": make_class_list(result.class_pixel_counts),
        "class_pixel_counts": result.class_pixel_counts,
        "instances": visible_instances,
        "processing_time_seconds": round(processing_time_seconds, 6),
    }


def maybe_save_existing_mask_projection(
    *,
    image_path: Path,
    frame_out: Path,
    projection_dir: Path | None,
    require_projection: bool,
) -> dict[str, str | None] | None:
    if projection_dir is None:
        return None
    mask_projection_path = frame_out / "mask_projection.jpg"
    if mask_projection_path.is_file():
        return None
    semantic_path = frame_out / "semantic_mask.npy"
    local_image_path = frame_out / "image.jpg"
    if not semantic_path.is_file() or not local_image_path.is_file():
        return None

    from PIL import Image  # noqa: WPS433

    semantic_mask = np.load(semantic_path)
    image_np = np.asarray(Image.open(local_image_path).convert("RGB"), dtype=np.uint8)
    semantic_color = make_semantic_color(semantic_mask)
    overlay = make_overlay(image_np, semantic_mask)
    return save_mask_projection_preview(
        image_path=image_path,
        output_dir=frame_out,
        projection_dir=projection_dir,
        image_np=image_np,
        semantic_color=semantic_color,
        overlay=overlay,
        require_projection=require_projection,
    )


def save_mask_projection_preview(
    *,
    image_path: Path,
    output_dir: Path,
    projection_dir: Path | None,
    image_np: np.ndarray,
    semantic_color: np.ndarray,
    overlay: np.ndarray,
    require_projection: bool,
) -> dict[str, str | None]:
    if projection_dir is None:
        return {"projection_path": None, "mask_projection": None, "projection_error": None}

    projection_path = find_projection_for_image(projection_dir, image_path)
    if projection_path is None:
        message = f"Projection image for {image_path.stem!r} was not found in {projection_dir}"
        if require_projection:
            raise FileNotFoundError(message)
        print(f"[run_sam3_single_image_folder] WARNING: {message}", file=sys.stderr, flush=True)
        return {"projection_path": None, "mask_projection": None, "projection_error": message}

    from PIL import Image  # noqa: WPS433

    projection = Image.open(projection_path).convert("RGB")
    target_size = (image_np.shape[1], image_np.shape[0])
    if projection.size != target_size:
        projection = projection.resize(target_size)


    #CHANGES FRO TEST, REPLACE PROJECTION ON TRIPTYCH BY IMAGE
    preview = make_labeled_triptych(
        [
            ("semantic class id", semantic_color),
            ("overlay", overlay),
            ("projection", np.asarray(projection, dtype=np.uint8)),
        ]
    )
    output_path = output_dir / "mask_projection.jpg"
    projection_output_path = output_dir / "projection.jpg"
    Image.fromarray(preview).save(output_path, quality=95)
    # projection.save(projection_output_path, quality=95)
    return {
        "projection_path": str(projection_path),
        "projection_copy": str(projection_output_path),
        "mask_projection": str(output_path),
        "projection_error": None,
    }


def find_projection_for_image(projection_dir: Path, image_path: Path) -> Path | None:
    stem = image_path.stem
    candidates = []
    preferred = projection_dir / f"{stem}{image_path.suffix.lower()}"
    if preferred.is_file():
        return preferred
    for suffix in sorted(IMAGE_SUFFIXES):
        candidate = projection_dir / f"{stem}{suffix}"
        if candidate.is_file():
            candidates.append(candidate)
    if candidates:
        return sorted(candidates)[0]
    recursive_candidates = sorted(path for path in projection_dir.rglob(f"{stem}.*") if path.suffix.lower() in IMAGE_SUFFIXES)
    return recursive_candidates[0] if recursive_candidates else None


def make_labeled_triptych(panels: list[tuple[str, np.ndarray]]) -> np.ndarray:
    from PIL import Image, ImageDraw  # noqa: WPS433

    label_height = 32
    separator_width = 8
    pil_panels = []
    for title, panel in panels:
        image = Image.fromarray(np.asarray(panel, dtype=np.uint8)).convert("RGB")
        canvas = Image.new("RGB", (image.width, image.height + label_height), (255, 255, 255))
        canvas.paste(image, (0, label_height))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), title, fill=(0, 0, 0))
        pil_panels.append(canvas)

    height = max(panel.height for panel in pil_panels)
    width = sum(panel.width for panel in pil_panels) + separator_width * (len(pil_panels) - 1)
    combined = Image.new("RGB", (width, height), (255, 255, 255))
    x = 0
    for panel in pil_panels:
        combined.paste(panel, (x, 0))
        x += panel.width + separator_width
    return np.asarray(combined, dtype=np.uint8)


def update_metadata(metadata_path: Path, values: dict[str, str | None]) -> None:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(values)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def save_instances_npz(path: Path, instances: list[Sam3Instance], *, shape: tuple[int, int]) -> None:
    height, width = shape
    if instances:
        payload: dict[str, np.ndarray] = {
            "masks": np.stack([item.mask for item in instances], axis=0).astype(bool, copy=False),
            "scores": np.asarray([item.score for item in instances], dtype=np.float32),
            "labels": np.asarray([item.label for item in instances]),
            "class_ids": np.asarray([item.class_id for item in instances], dtype=np.uint16),
            "prompts": np.asarray([item.prompt for item in instances]),
            "source_labels": np.asarray([item.source_label or "" for item in instances]),
            "orientation_labels": np.asarray([item.orientation_label or "" for item in instances]),
            "orientation_scores": np.asarray(
                [np.nan if item.orientation_score is None else item.orientation_score for item in instances],
                dtype=np.float32,
            ),
            "orientation_margins": np.asarray(
                [np.nan if item.orientation_margin is None else item.orientation_margin for item in instances],
                dtype=np.float32,
            ),
            "orientation_probabilities": np.asarray(
                [
                    (np.nan, np.nan, np.nan)
                    if item.orientation_probabilities is None
                    else item.orientation_probabilities
                    for item in instances
                ],
                dtype=np.float32,
            ),
            "orientation_fallback": np.asarray(
                [item.orientation_fallback for item in instances],
                dtype=bool,
            ),
            "orientation_fallback_reasons": np.asarray(
                [item.orientation_fallback_reason or "" for item in instances]
            ),
        }
        if all(item.box is not None for item in instances):
            payload["boxes"] = np.stack([np.asarray(item.box, dtype=np.float32) for item in instances], axis=0)
    else:
        payload = {
            "masks": np.zeros((0, height, width), dtype=bool),
            "scores": np.zeros((0,), dtype=np.float32),
            "labels": np.asarray([], dtype=str),
            "class_ids": np.zeros((0,), dtype=np.uint16),
            "prompts": np.asarray([], dtype=str),
            "source_labels": np.asarray([], dtype=str),
            "orientation_labels": np.asarray([], dtype=str),
            "orientation_scores": np.zeros((0,), dtype=np.float32),
            "orientation_margins": np.zeros((0,), dtype=np.float32),
            "orientation_probabilities": np.zeros((0, 3), dtype=np.float32),
            "orientation_fallback": np.zeros((0,), dtype=bool),
            "orientation_fallback_reasons": np.asarray([], dtype=str),

        }
    np.savez_compressed(path, **payload)


def make_overlay(
    image_np: np.ndarray,
    semantic_mask: np.ndarray,
    *,
    alpha: float = 0.45,
    instances: list[Sam3Instance] | None = None,
    confidence: np.ndarray | None = None,
    min_label_area: int = 64,
) -> np.ndarray:
    overlay = image_np.copy()
    for class_id in sorted(int(x) for x in np.unique(semantic_mask) if int(x) != 0):
        color = color_for_class(class_id)
        mask = semantic_mask == class_id
        overlay[mask] = (
            overlay[mask].astype(np.float32) * (1.0 - alpha) + color.astype(np.float32) * alpha
        ).astype(np.uint8)
    if instances:
        overlay = drew_instance_labels(
            overlay=overlay,
            semantic_mask=semantic_mask,
            instances=instances,
            confidence=confidence,
            min_label_area=min_label_area
        )
    return overlay


def drew_instance_labels(
        overlay: np.ndarray,
        *,
        semantic_mask: np.ndarray,
        instances: list[Sam3Instance],
        confidence: np.ndarray | None,
        min_label_area: int
) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont  # noqa: WPS433

    image = Image.fromarray(np.asarray(overlay, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.truetype("DejaVuSans.ttf", size=48)
    height, width = semantic_mask.shape

    for item in sorted(instances, key=lambda value: value.score, reverse=True):
        visible = item.mask & (semantic_mask == item.class_id)
        if confidence is not None:
            visible &= np.isclose(confidence, np.float32(item.score), atol=1e-6)
        if int(np.count_nonzero(visible)) < min_label_area:
            continue

        text = f"{item.label} {item.score:.2f}"
        label_x, label_y = label_anchor(visible, image_size=(width, height), text=text, draw=draw, font=font)
        color = color_for_class(item.class_id)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        padding = 10
        box = [
            label_x,
            label_y,
            label_x + text_width + padding * 2,
            label_y + text_height + padding * 2,
        ]
        draw.rectangle(box, fill=(int(color[0]), int(color[1]), int(color[2]), 220), outline=(0, 0, 0, 220))
        draw.text(
            (label_x + padding, label_y + padding),
            text,
            font=font,
            fill=contrast_text_color(color),
        )
    return np.asarray(image, dtype=np.uint8)


def label_anchor(visible: np.ndarray, *, image_size: tuple[int, int], text: str, draw: Any, font: Any) -> tuple[int, int]:
    width, height = image_size
    ys, xs = np.nonzero(visible)
    center_x = int(np.median(xs))
    center_y = int(np.median(ys))
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0] + 8
    text_height = text_bbox[3] - text_bbox[1] + 8
    label_x = min(max(center_x - text_width // 2, 0), max(width - text_width - 1, 0))
    label_y = min(max(center_y - text_height // 2, 0), max(height - text_height - 1, 0))
    return int(label_x), int(label_y)


def contrast_text_color(color: np.ndarray) -> tuple[int, int, int, int]:
    luminance = 0.2126 * float(color[0]) + 0.7152 * float(color[1]) + 0.0722 * float(color[2])
    return (0, 0, 0, 255) if luminance > 145.0 else (255, 255, 255, 255)


def make_semantic_color(semantic_mask: np.ndarray) -> np.ndarray:
    color = np.zeros((*semantic_mask.shape, 3), dtype=np.uint8)
    for class_id in sorted(int(x) for x in np.unique(semantic_mask) if int(x) != 0):
        color[semantic_mask == class_id] = color_for_class(class_id)
    return color


def make_preview(image_np: np.ndarray, semantic_color: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    separator = np.full((image_np.shape[0], 8, 3), 255, dtype=np.uint8)
    return np.concatenate([image_np, separator, semantic_color, separator, overlay], axis=1)


def color_for_class(class_id: int) -> np.ndarray:
    rng = np.random.default_rng(class_id * 1009 + 17)
    return rng.integers(40, 240, size=3, dtype=np.uint8)


def extract_arrays(outputs: Any, *, height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
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
    return np.zeros((0, height, width), dtype=bool), np.zeros((0,), dtype=np.float32), None


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


def collect_images(image_dir: Path, *, recursive: bool) -> list[Path]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    iterator = image_dir.rglob("*") if recursive else image_dir.iterdir()
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def output_dir_for_image(out_dir: Path, image_dir: Path, image_path: Path, *, recursive: bool) -> Path:
    if not recursive:
        return out_dir / image_path.stem
    rel = image_path.relative_to(image_dir)
    return out_dir / rel.with_suffix("")


def outputs_exist(
    frame_out: Path,
    *,
    projection_enabled: bool = False,
    classes_log_enabled: bool = False,
) -> bool:
    required = [
        frame_out / "semantic_mask.npy",
        frame_out / "confidence.npy",
        frame_out / "instances.npz",
        frame_out / "overlay.jpg",
        frame_out / "metadata.json",
    ]
    if projection_enabled:
        required.extend([frame_out / "projection.jpg", frame_out / "mask_projection.jpg"])
    if classes_log_enabled:
        required.append(frame_out / "classes_log.json")
    return all(path.is_file() for path in required)


def validate_outputs(frame_out: Path) -> None:
    semantic = np.load(frame_out / "semantic_mask.npy")
    confidence = np.load(frame_out / "confidence.npy")
    if semantic.shape != confidence.shape:
        raise ValueError(f"semantic/confidence shape mismatch in {frame_out}: {semantic.shape} vs {confidence.shape}")
    with np.load(frame_out / "instances.npz", allow_pickle=False) as data:
        masks = np.asarray(data["masks"])
        scores = np.asarray(data["scores"])
        labels = np.asarray(data["labels"])
        class_ids = np.asarray(data["class_ids"])
        prompts = np.asarray(data["prompts"])
        orientation_probabilities = np.asarray(data["orientation_probabilities"])
        orientation_scores = np.asarray(data["orientation_scores"])
        orientation_fallback = np.asarray(data["orientation_fallback"])
    if masks.ndim != 3 or masks.dtype != np.bool_:
        raise ValueError(f"instances masks must be bool[N,H,W], got {masks.shape} {masks.dtype}")
    if scores.shape != (masks.shape[0],):
        raise ValueError(f"instances scores mismatch in {frame_out}")
    if labels.shape != (masks.shape[0],) or class_ids.shape != (masks.shape[0],) or prompts.shape != (masks.shape[0],):
        raise ValueError(f"instances metadata mismatch in {frame_out}")
    if orientation_probabilities.shape != (masks.shape[0], 3):
        raise ValueError(f"instances orientation probabilities mismatch in {frame_out}")
    if orientation_scores.shape != (masks.shape[0],) or orientation_fallback.shape != (masks.shape[0],):
        raise ValueError(f"instances orientation metadata mismatch in {frame_out}")


def configure_sam3_imports(*, sam3_root: Path | None, model_dir: Path | None) -> None:
    if sam3_root is not None:
        sys.path.insert(0, str(sam3_root))
    if model_dir is not None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HOME", str(model_dir))


def resolve_model_dir(*, sam3_root: Path | None, sam3_model_path: str | None) -> Path | None:
    if sam3_model_path:
        model_dir = Path(sam3_model_path).expanduser().resolve()
    elif sam3_root is not None:
        model_dir = sam3_root / "sam3.1"
    else:
        return None
    if not (model_dir / "sam3.1_multiplex.pt").is_file():
        raise FileNotFoundError(f"Missing SAM3.1 checkpoint: {model_dir / 'sam3.1_multiplex.pt'}")
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
        patch_hf_checkpoint_download(sam3_model_builder, checkpoint_path=checkpoint_path, model_dir=model_dir)
        kwargs["checkpoint_path"] = str(checkpoint_path)
    params = inspect.signature(builder).parameters
    predictor = builder(**{key: value for key, value in kwargs.items() if key in params})
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
        raise AttributeError("SAM3 predictor has no default_output_prob_thresh attribute")
    if not hasattr(predictor, "model"):
        raise AttributeError("SAM3 predictor has no model attribute")
    model_attributes = (
        "score_threshold_detection",
        "image_only_det_thresh",
        "new_det_thresh",
    )
    missing = [name for name in model_attributes if not hasattr(predictor.model, name)]
    if missing:
        raise AttributeError(f"SAM3 predictor model does not expose threshold attributes: {missing}")
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
            raise ValueError("Label min-score JSON must contain an object mapping label to score")
    else:
        raw_mapping = parse_simple_score_yaml(config_path)
    scores: dict[str, float] = {}
    for raw_label, raw_score in raw_mapping.items():
        label = str(raw_label).strip()
        if not label:
            raise ValueError(f"Empty label in min-score config: {config_path}")
        if label in scores:
            raise ValueError(f"Duplicate label {label!r} in min-score config: {config_path}")
        scores[label] = validate_probability(raw_score, name=f"min score for label {label!r}")
    unknown_labels = sorted(set(scores) - known_labels)
    if unknown_labels:
        raise ValueError(
            "Label min-score config contains labels absent from the active prompt/classes configs: "
            f"{unknown_labels}"
        )
    return scores
def parse_simple_score_yaml(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line or line == "{}":
            continue
        if ":" not in line:
            raise ValueError(f"Expected 'label: score' in {path}:{line_number}, got {raw_line!r}")
        raw_label, raw_score = line.split(":", 1)
        label = strip_matching_quotes(raw_label.strip())
        score = raw_score.strip()
        if not score:
            raise ValueError(f"Missing score for label {label!r} in {path}:{line_number}")
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

def patch_hf_checkpoint_download(model_builder_module: Any, *, checkpoint_path: Path, model_dir: Path) -> None:
    def _local_download_ckpt_from_hf(*args: Any, **kwargs: Any) -> str:
        return str(checkpoint_path)

    model_builder_module.download_ckpt_from_hf = _local_download_ckpt_from_hf
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HOME", str(model_dir))

# may be deleted, but it contain description and help for some fields
# it could be helpful
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SAM3.1 single-image prompt notebook flow over an image folder.")
    parser.add_argument("--image-dir", required=True, help="Directory with camera images.")
    parser.add_argument("--out-dir", required=True, help="Output directory. Each image gets its own subdirectory.")
    parser.add_argument("--prompt-config", default=str(REPO_ROOT / "configs" / "sam3_text_prompts_pointwise_v1.yaml"))
    parser.add_argument("--classes-yaml", default=str(REPO_ROOT / "configs" / "classes_pointwise_v1.yaml"))
    parser.add_argument("--sam3-root", default=None, help="Path to cloned SAM3 repo, e.g. /home/.../git_repo/sam3.")
    parser.add_argument("--sam3-model-path", default=None, help="Path to local facebook/sam3.1 model directory.")
    parser.add_argument("--min-score", type=float, default=0.70)
    parser.add_argument("--max-images", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--max-prompts", type=int, default=None, help="Optional smoke-test prompt limit per image.")
    parser.add_argument("--recursive", action="store_true", help="Read images recursively and mirror the relative output tree.")
    parser.add_argument(
        "--projection-dir",
        default=None,
        help="Optional directory with LiDAR point projection images matched to camera images by file stem.",
    )
    parser.add_argument(
        "--require-projection",
        action="store_true",
        help="Fail if --projection-dir is set and a matching projection image is missing.",
    )
    parser.add_argument("--use-fa3", action="store_true", help="Enable FlashAttention 3. Disabled by default.")
    parser.add_argument("--prompt-log", action="store_true", help="Print every prompt for every image.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--min-mask-size", type=int, default=30)
    return parser.parse_args()

from collections import Counter

def make_class_list(class_pixel_counts):
    return list(class_pixel_counts.keys())


def shard_image_paths(
    image_paths: list[Path],
    *,
    worker_index: int,
    num_workers: int,
) -> list[Path]:
    if num_workers <= 0:
        raise ValueError(f"num_workers must be positive, got {num_workers}")
    if worker_index < 0 or worker_index >= num_workers:
        raise ValueError(
            f"worker_index must be in [0, {num_workers}), got {worker_index}"
        )
    return image_paths[worker_index::num_workers]


def sam3_single_image_folder(image_dir, out_dir, prompt_config, classes_yaml,
                             sam3_root, sam3_model_path, min_score, projection_dir, label_min_scores = None,
                             recursive = None , max_images = None, max_prompts = None,
                             use_fa3 = False, overwrite = False, require_projection = False,
                             validate = False, log_json = False, min_mask_size = 30,
                             vehicle_orientation_checkpoint = None, vehicle_orientation_device = "auto",
                             vehicle_orientation_min_confidence = 0.75, vehicle_orientation_min_margin = 0.10,
                             vehicle_orientation_nms_iou = 0.80, vehicle_prompt_label = "vehicle",
                             sam3_only = False, prompt_log = False,
                             inference_precision = "auto", cache_visual_features = False,
                             worker_index = 0, num_workers = 1
                             ) -> None:


    validate_probability(vehicle_orientation_min_confidence, name="vehicle_orientation_min_confidence")
    validate_probability(vehicle_orientation_min_margin, name="vehicle_orientation_min_margin")
    validate_probability(vehicle_orientation_nms_iou, name="vehicle_orientation_nms_iou")
    image_dir = Path(image_dir).expanduser().resolve()

    out_dir = Path(out_dir).expanduser().resolve()
    projection_dir = Path(projection_dir).expanduser().resolve() if projection_dir else None
    if projection_dir is not None and not projection_dir.is_dir():
        raise FileNotFoundError(f"Projection directory does not exist: {projection_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images(image_dir, recursive=recursive)
    if max_images is not None:
        if max_images <= 0:
            raise ValueError(f"--max-images must be positive, got {max_images}")
        image_paths = image_paths[: max_images]
    image_paths = shard_image_paths(
        image_paths,
        worker_index=int(worker_index),
        num_workers=int(num_workers),
    )
    if not image_paths:
        raise ValueError(
            f"Worker {worker_index}/{num_workers} received no images from {image_dir}"
        )

    prompts_by_label = load_prompt_config(prompt_config)
    class_to_id = load_semantic_classes(classes_yaml)

    vehicle_orientation_classifier = None
    vehicle_class_mapping = None
    routed_prompt_labels: set[str] = set()

    effective_orientation_checkpoint = vehicle_orientation_checkpoint
    if sam3_only:
        if effective_orientation_checkpoint:
            print(
                "--sam3-only is enabled; ignoring --vehicle-orientation-checkpoint",
                file=sys.stderr,
                flush=True,
            )
        effective_orientation_checkpoint = None

    if effective_orientation_checkpoint:
        if vehicle_prompt_label not in prompts_by_label:
            raise ValueError(
                f"Prompt config has no transient label {vehicle_prompt_label!r}"
            )

        vehicle_class_mapping = resolve_vehicle_class_mapping(class_to_id)

        vehicle_orientation_classifier = build_vehicle_orientation_classifier(
            effective_orientation_checkpoint,
            device=vehicle_orientation_device,
        )

        routed_prompt_labels.add(vehicle_prompt_label)


    vehicle_orientation_mode = (
        "sam3_only"
        if sam3_only
        else "efficientnet"
        if vehicle_orientation_classifier is not None
        else "direct_prompts"
    )
    active_prompt_labels = {
        label
        for label in prompts_by_label
        if label in class_to_id or label in routed_prompt_labels
    }

    label_to_id = {
        label: int(class_to_id[label])
        for label in prompts_by_label
        if label in class_to_id
    }

    if vehicle_class_mapping is not None:
        label_to_id.update(
            {
                semantic_label: class_id
                for semantic_label, class_id in vehicle_class_mapping.values()
            }
        )

    flat_prompts = [
        (label, prompt)
        for label, prompts in prompts_by_label.items()
        for prompt in prompts
        if label in active_prompt_labels
    ]

    if max_prompts is not None:
        if max_prompts <= 0:
            raise ValueError(f"--max-prompts must be positive, got {max_prompts}")
        flat_prompts = flat_prompts[: max_prompts]

    print("routed_prompt_labels:", routed_prompt_labels)
    print("active_prompt_labels:", active_prompt_labels)
    print("flat_prompts:", flat_prompts)

    if not flat_prompts:
        raise ValueError(
            "Prompt config produced no active prompts. "
            f"Prompt labels: {sorted(prompts_by_label)}; "
            f"classes yaml labels: {sorted(class_to_id)}; "
            f"routed labels: {sorted(routed_prompt_labels)}; "
            f"checkpoint: {vehicle_orientation_checkpoint!r}"
        )
    sam3_root = Path(sam3_root).expanduser().resolve() if sam3_root else None
    model_dir = resolve_model_dir(sam3_root=sam3_root, sam3_model_path=sam3_model_path)
    configure_sam3_imports(sam3_root=sam3_root, model_dir=model_dir)

    label_min_score_overrides = load_label_min_scores(
        label_min_scores,
        known_labels=active_prompt_labels,
    )
    effective_label_min_scores = {
        label: float(label_min_score_overrides.get(label, min_score))
        for label in active_prompt_labels
    }

    predictor = build_predictor(
        model_dir=model_dir,
        use_fa3=use_fa3,
        min_score=min_score,
        inference_precision=inference_precision,
        cache_visual_features=cache_visual_features,
    )

    results = []
    for index, image_path in enumerate(image_paths, start=1):
        frame_out = output_dir_for_image(out_dir, image_dir, image_path, recursive=recursive)
        if outputs_exist(
            frame_out,
            projection_enabled=projection_dir is not None,
            classes_log_enabled=log_json,
        ) and not overwrite:
            projection_info = maybe_save_existing_mask_projection(
                image_path=image_path,
                frame_out=frame_out,
                projection_dir=projection_dir,
                require_projection=require_projection,
            )
            metadata_path = frame_out / "metadata.json"
            if projection_info and metadata_path.is_file():
                update_metadata(metadata_path, projection_info)
            results.append(
                {
                    "image": str(image_path),
                    "frame_id": frame_out.name,
                    "status": "exists",
                    "output_dir": str(frame_out),
                    "metadata": str(metadata_path),
                }
            )
            continue

        worker_prefix = f"[worker {worker_index + 1}/{num_workers}] " if num_workers > 1 else ""
        print(
            f"{worker_prefix}[{index:04d}/{len(image_paths):04d}] {image_path}",
            file=sys.stderr,
            flush=True,
        )
        frame_started_at =time.perf_counter()
        result = process_image(
            predictor=predictor,
            image_path=image_path,
            output_dir=frame_out,
            flat_prompts=flat_prompts,
            label_to_id=label_to_id,
            label_min_scores=effective_label_min_scores,
            min_score=min_score,
            prompt_log=prompt_log,
            projection_dir=projection_dir,
            min_mask_size=min_mask_size,
            require_projection=require_projection,
            vehicle_orientation_classifier=vehicle_orientation_classifier,
            vehicle_prompt_label=vehicle_prompt_label,
            vehicle_class_mapping=vehicle_class_mapping,
            vehicle_orientation_min_confidence=vehicle_orientation_min_confidence,
            vehicle_orientation_min_margin=vehicle_orientation_min_margin,
            vehicle_orientation_nms_iou=vehicle_orientation_nms_iou,
        )
        processing_time_seconds = time.perf_counter() - frame_started_at
        metadata = {
            "version": 1,
            "image_path": str(result.image_path),
            "image_size": [result.image_size[0], result.image_size[1]],
            "sam3_root": str(sam3_root) if sam3_root is not None else None,
            "sam3_model_path": str(model_dir) if model_dir is not None else None,
            "sam3_checkpoint": str(model_dir / "sam3.1_multiplex.pt") if model_dir is not None else None,
            "prompt_config": str(Path(prompt_config)),
            "classes_yaml": str(Path(classes_yaml)),
            "min_score": float(min_score),
            "label_min_scores_file": str(Path(label_min_scores).expanduser().resolve()) if label_min_scores else None,
            "label_min_score_overrides": label_min_score_overrides,
            "effective_label_min_scores": effective_label_min_scores,
            "instances": len(result.instances),
            "overlay_instances": len(result.overlay_instances),
            "class_pixel_counts": result.class_pixel_counts,
            "projection_dir": str(projection_dir) if projection_dir is not None else None,
            "projection_path": result.projection_path,
            "projection_copy": result.projection_copy_path,
            "mask_projection": result.mask_projection_path,
            "projection_error": result.projection_error,
            "mask_nms_iou": float(vehicle_orientation_nms_iou),
            "processing_time_seconds": round(processing_time_seconds, 6),
            "sam3_runtime": sam3_runtime_summary(predictor),
            "worker": {
                "index": int(worker_index),
                "count": int(num_workers),
                "physical_gpu_id": os.environ.get("SAM3_PHYSICAL_GPU_ID"),
            },
            "vehicle_orientation": {
                "enabled": vehicle_orientation_classifier is not None,
                 "mode": vehicle_orientation_mode,
                "checkpoint": str(Path(effective_orientation_checkpoint).expanduser().resolve())
                if effective_orientation_checkpoint
                else None,
                "prompt_label": vehicle_prompt_label,
                "min_confidence": float(vehicle_orientation_min_confidence),
                "min_margin": float(vehicle_orientation_min_margin),
                "nms_iou": float(vehicle_orientation_nms_iou),
                "predictions": list(result.orientation_predictions),
            },
        }

        if log_json:
            classes_log = build_classes_log(
                result=result,
                prompt_config=Path(prompt_config),
                min_score=min_score,
                processing_time_seconds=processing_time_seconds,
            )
            (frame_out / "classes_log.json").write_text(
                json.dumps(classes_log, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        (frame_out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        if validate:
            validate_outputs(frame_out)
        results.append(
            {
                "image": str(image_path),
                "frame_id": frame_out.name,
                "status": "created",
                "output_dir": str(frame_out),
                "metadata": str(frame_out / "metadata.json"),
                "instances": len(result.instances),
                "processing_time_seconds": round(processing_time_seconds, 6),
            }
        )

    manifest = {
        "version": 1,
        "image_dir": str(image_dir),
        "out_dir": str(out_dir),
        "prompt_config": str(Path(prompt_config)),
        "classes_yaml": str(Path(classes_yaml)),
        "sam3_root": str(sam3_root) if sam3_root is not None else None,
        "sam3_model_path": str(model_dir) if model_dir is not None else None,
        "projection_dir": str(projection_dir) if projection_dir is not None else None,
        "require_projection": bool(require_projection),
        "min_score": float(min_score),
        "label_min_scores_file": str(Path(label_min_scores).expanduser().resolve()) if label_min_scores else None,
        "label_min_score_overrides": label_min_score_overrides,
        "effective_label_min_scores": effective_label_min_scores,
        "images": len(image_paths),
        "prompts": len(flat_prompts),
        "mask_nms_iou": float(vehicle_orientation_nms_iou),
        "labels": label_to_id,
        "frames": results,
        "sam3_runtime": sam3_runtime_summary(predictor),
        "worker": {
            "index": int(worker_index),
            "count": int(num_workers),
            "physical_gpu_id": os.environ.get("SAM3_PHYSICAL_GPU_ID"),
        },
        "vehicle_orientation": {
            "enabled": vehicle_orientation_classifier is not None,
            "mode": vehicle_orientation_mode,
            "checkpoint": str(Path(effective_orientation_checkpoint).expanduser().resolve())
            if effective_orientation_checkpoint
            else None,
            "prompt_label": vehicle_prompt_label,
            "min_confidence": float(vehicle_orientation_min_confidence),
            "min_margin": float(vehicle_orientation_min_margin),
            "nms_iou": float(vehicle_orientation_nms_iou),
            "class_mapping": None
            if vehicle_class_mapping is None
            else {
                key: {"label": label, "class_id": class_id}
                for key, (label, class_id) in vehicle_class_mapping.items()
            },
        },
    }
    manifest_name = (
        "sam3_single_image_folder_manifest.json"
        if num_workers == 1
        else f"sam3_single_image_folder_manifest.worker_{worker_index:03d}.json"
    )
    manifest_path = out_dir / manifest_name
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"images": len(image_paths), "manifest": str(manifest_path)}, indent=2))
