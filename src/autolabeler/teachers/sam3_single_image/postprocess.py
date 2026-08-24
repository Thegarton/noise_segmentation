"""Mask filtering, semantic composition, and auxiliary classifiers."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .runtime import validate_probability
from .types import SIGN_TYPE_LABELS, SIGN_TYPE_REJECT_LABEL, Sam3Instance


def find_project_root(module_path: str | Path = __file__) -> Path:
    """Find the checkout root for both supported package layouts."""
    resolved = Path(module_path).expanduser().resolve()
    for parent in resolved.parents:
        if (parent / "src" / "autolabeler").is_dir():
            return parent
    return Path.cwd().resolve()


REPO_ROOT = find_project_root()

CLASS_OVERRIDES = {
    "ground_markings": {"epoxy_floor"},
    "license_plate&taillights": {"vehicle"},
}
ARRESTOR_MIN_Y_BOTTOM = 0.4
ARRESTOR_MIN_ASPECT_RATIO = 2.0
GEOMETRY_FILTER_RULES: dict[str, dict[str, float]] = {
    "arrestor": {
        "min_y_bottom": 0.4,
        "min_aspect_ratio": 2.0,
    },
    "wheel_chock": {
        "min_y_bottom": 0.4,
        "min_aspect_ratio": 1.5,
    },
    "underground_parking_sign": {
        "min_y_bottom": 0.4,
        "min_aspect_ratio": 1.0,
    },
    "ground_markings": {
        "min_y_bottom": 0.4,
        "min_aspect_ratio": 0,
    },
    "induction_sign": {
        "min_y_bottom": 0.50,
        "min_aspect_ratio": 1.0,
        "max_aspect_ratio": 4.0,
    },
    "parking_barrier_lock": {
        "min_y_bottom": 0.5,
        "min_aspect_ratio": 0,
    },
    "pillar_corner_guard": {
        "min_y_bottom": 0.2,
        "min_aspect_ratio": 0.9,
    },
    "height_restriction_barrel": {
        "min_y_bottom": 0.4,
        "min_aspect_ratio": 0.0,
    },
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
    },
    "blind_spot_mirror": {
        "max_box_width": 0.5,
        "max_box_height": 0.5,
        "max_box_area": 0.25,
    },
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

        if not item.sign_classifier_accepted:
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
        if not item.sign_classifier_accepted:
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
        if item.label == "induction_sign":
            max_aspect_ratio = float(rule["max_aspect_ratio"])
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

    if item.label in {"underground_parking_sign"}:
        return y_bottom > min_y_bottom or aspect_ratio < min_aspect_ratio

    if item.label in {"induction_sign"}:
        return (
            y_bottom > min_y_bottom
            or aspect_ratio > max_aspect_ratio
            or aspect_ratio < min_aspect_ratio
        )

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


def resolve_vehicle_class_mapping(
    class_to_id: dict[str, int],
) -> dict[str, tuple[str, int]]:
    class_names = {
        "front": "front_of_vehicle",
        "rear": "rear_of_vehicle",
        "side": "side_of_vehicle",
    }
    missing = [label for label in class_names.values() if label not in class_to_id]
    if missing:
        raise ValueError(
            f"Vehicle orientation classes are missing from classes yaml: {missing}"
        )
    mapping = {
        orientation: (label, int(class_to_id[label]))
        for orientation, label in class_names.items()
    }
    class_ids = [class_id for _, class_id in mapping.values()]

    if len(set(class_ids)) != len(class_ids):
        raise ValueError(
            f"Vehicle orientation classes must have unique ids, got {mapping}"
        )
    reserved = {
        int(class_to_id[label])
        for label in ("background", "ignore")
        if label in class_to_id
    }
    collisions = {
        label: class_id for label, class_id in mapping.values() if class_id in reserved
    }
    if collisions:
        raise ValueError(
            "Vehicle orientation class ids collide with background/ignore ids: "
            f"{collisions}"
        )
    return mapping


def build_vehicle_orientation_classifier(
    checkpoint_path: str | Path, *, device: str
) -> Any:
    local_src = REPO_ROOT / "vehicle_orientation" / "src"
    if local_src.is_dir() and str(local_src) not in sys.path:
        sys.path.insert(0, str(local_src))
    try:
        from vehicle_orientation.model import VehicleOrientationClassifier  # type: ignore # noqa: WPS433
    except ImportError as exc:
        raise ImportError(
            "Vehicle orientation support could not be imported. "
            f"project_root={REPO_ROOT}, local_src={local_src}. "
            "Install it with 'pip install -e vehicle_orientation' inside the "
            "SAM3 environment if the local source directory is absent."
        ) from exc
    return VehicleOrientationClassifier(checkpoint_path, device=device)


def build_sign_type_classifier(checkpoint_path: str | Path, *, device: str) -> Any:
    local_src = REPO_ROOT / "sign_type_classifier" / "src"
    if local_src.is_dir() and str(local_src) not in sys.path:
        sys.path.insert(0, str(local_src))
    try:
        from sign_type_classifier.features import numeric_feature_names  # type: ignore # noqa: WPS433
        from sign_type_classifier.model import SignTypeEnsembleClassifier  # type: ignore # noqa: WPS433
    except ImportError as exc:
        raise ImportError(
            "Sign-type classifier support could not be imported. "
            f"project_root={REPO_ROOT}, local_src={local_src}. "
            "Install it with 'pip install -e sign_type_classifier' inside the "
            "SAM3 environment if the local source directory is absent."
        ) from exc

    classifier = SignTypeEnsembleClassifier(checkpoint_path, device=device)
    expected_features = tuple(numeric_feature_names(SIGN_TYPE_LABELS))
    if tuple(classifier.feature_names) != expected_features:
        raise ValueError(
            "Sign classifier numeric features do not match the production SAM3 runner. "
            f"checkpoint={tuple(classifier.feature_names)}, expected={expected_features}"
        )
    return classifier


def resolve_sign_type_class_mapping(class_to_id: dict[str, int]) -> dict[str, int]:
    missing = [label for label in SIGN_TYPE_LABELS if label not in class_to_id]
    if missing:
        raise ValueError(
            f"Sign classifier output classes are missing from classes yaml: {missing}"
        )
    mapping = {label: int(class_to_id[label]) for label in SIGN_TYPE_LABELS}
    if len(set(mapping.values())) != len(mapping):
        raise ValueError(f"Sign classifier classes must have unique ids, got {mapping}")
    reserved_ids = {
        int(class_to_id[label])
        for label in ("background", "ignore")
        if label in class_to_id
    }
    collisions = {
        label: class_id
        for label, class_id in mapping.items()
        if class_id in reserved_ids
    }
    if collisions:
        raise ValueError(
            "Sign classifier class ids collide with background/ignore ids: "
            f"{collisions}"
        )
    return mapping


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

    vehicle_indices = [
        index
        for index, item in enumerate(instances)
        if item.label == vehicle_prompt_label
    ]
    eligible_indices = [
        index
        for index in vehicle_indices
        if int(np.count_nonzero(np.asarray(instances[index].mask, dtype=bool)))
        >= min_mask_size
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
            raise ValueError(
                f"Orientation classifier returned unsupported semantic label {semantic_key!r}"
            )
        target_label, target_id = class_mapping[semantic_key]
        probabilities = tuple(float(value) for value in decision.probabilities)
        if len(probabilities) != 3:
            raise ValueError(
                f"Orientation probabilities must contain front/rear/side, got {probabilities}"
            )
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


def apply_sign_type_classification(
    *,
    image_rgb: np.ndarray,
    instances: list[Sam3Instance],
    classifier: Any,
    class_mapping: dict[str, int],
    min_confidence: float,
    min_margin: float,
    min_mask_size: int,
    cluster_iou: float,
    cluster_containment: float,
    context_scale: float,
) -> tuple[list[Sam3Instance], tuple[dict[str, Any], ...]]:
    local_src = REPO_ROOT / "sign_type_classifier" / "src"
    if local_src.is_dir() and str(local_src) not in sys.path:
        sys.path.insert(0, str(local_src))
    from sign_type_classifier.clustering import (  # type: ignore # noqa: WPS433
        SignDetection,
        cluster_sign_detections,
    )
    from sign_type_classifier.features import (  # type: ignore # noqa: WPS433
        build_numeric_features,
        extract_sign_crops,
        numeric_feature_names,
    )

    validate_probability(min_confidence, name="sign_classifier_min_confidence")
    validate_probability(min_margin, name="sign_classifier_min_margin")
    validate_probability(cluster_iou, name="sign_classifier_cluster_iou")
    validate_probability(
        cluster_containment,
        name="sign_classifier_cluster_containment",
    )
    if min_mask_size <= 0:
        raise ValueError(
            f"min_mask_size must be positive for sign classification, got {min_mask_size}"
        )
    if not np.isfinite(context_scale) or context_scale < 1.0:
        raise ValueError(
            f"sign_classifier_context_scale must be finite and >= 1, got {context_scale}"
        )

    missing_mapping = [
        label for label in SIGN_TYPE_LABELS if label not in class_mapping
    ]
    if missing_mapping:
        raise ValueError(f"Missing sign class ids: {missing_mapping}")
    expected_classifier_classes = (*SIGN_TYPE_LABELS, SIGN_TYPE_REJECT_LABEL)
    classifier_classes = tuple(str(value) for value in classifier.class_names)
    if classifier_classes != expected_classifier_classes:
        raise ValueError(
            f"Sign classifier classes must be {expected_classifier_classes}, got {classifier_classes}"
        )

    sign_detections = [
        SignDetection(
            label=item.label,
            prompt=item.prompt,
            score=float(item.score),
            mask=np.asarray(item.mask, dtype=bool),
            box=None if item.box is None else np.asarray(item.box, dtype=np.float32),
        )
        for item in instances
        if item.label in SIGN_TYPE_LABELS
    ]
    retained_instances = [
        item for item in instances if item.label not in SIGN_TYPE_LABELS
    ]
    candidates = cluster_sign_detections(
        sign_detections,
        class_names=SIGN_TYPE_LABELS,
        min_mask_size=min_mask_size,
        iou_threshold=cluster_iou,
        containment_threshold=cluster_containment,
    )
    if not candidates:
        return retained_instances, ()

    feature_names = tuple(numeric_feature_names(SIGN_TYPE_LABELS))
    if tuple(classifier.feature_names) != feature_names:
        raise ValueError(
            "Sign classifier feature schema mismatch: "
            f"checkpoint={tuple(classifier.feature_names)}, expected={feature_names}"
        )

    context_images = []
    feature_rows = []
    for candidate in candidates:
        crops = extract_sign_crops(
            image_rgb,
            candidate.mask,
            context_scale=context_scale,
        )
        features = build_numeric_features(
            image_rgb,
            candidate.mask,
            class_names=SIGN_TYPE_LABELS,
            class_scores=candidate.class_scores,
            context_scale=context_scale,
        )
        context_images.append(crops.context_rgb)
        feature_rows.append([float(features[name]) for name in feature_names])

    decisions = classifier.classify(
        context_images,
        np.asarray(feature_rows, dtype=np.float32),
    )
    if len(decisions) != len(candidates):
        raise ValueError(
            "Sign classifier returned an unexpected number of predictions: "
            f"{len(decisions)} for {len(candidates)} candidates"
        )

    predictions: list[dict[str, Any]] = []
    routed_instances: list[Sam3Instance] = []
    for candidate, decision in zip(candidates, decisions):
        predicted_label = str(decision.label)
        if predicted_label not in expected_classifier_classes:
            raise ValueError(
                f"Sign classifier returned unsupported label {predicted_label!r}"
            )
        confidence = validate_probability(
            decision.confidence,
            name="sign_classifier_prediction_confidence",
        )
        margin = validate_probability(
            decision.margin,
            name="sign_classifier_prediction_margin",
        )
        fallback_reasons = []
        if confidence < float(min_confidence):
            fallback_reasons.append("low_confidence")
        if margin < float(min_margin):
            fallback_reasons.append("low_margin")
        accepted = not fallback_reasons
        assigned_label = predicted_label if accepted else candidate.label
        removed = accepted and assigned_label == SIGN_TYPE_REJECT_LABEL
        fallback_reason = "+".join(fallback_reasons) if fallback_reasons else None

        probabilities = _named_probabilities(classifier_classes, decision.probabilities)
        image_probabilities = _named_probabilities(
            classifier_classes,
            decision.image_probabilities,
        )
        numeric_probabilities = _named_probabilities(
            classifier_classes,
            decision.numeric_probabilities,
        )
        sam3_scores = tuple(
            (label, float(candidate.class_scores[label])) for label in SIGN_TYPE_LABELS
        )
        prediction = {
            "sam3_label": candidate.label,
            "sam3_score": float(candidate.score),
            "sam3_class_scores": dict(sam3_scores),
            "sam3_prompts": [item.prompt for item in candidate.detections],
            "predicted_label": predicted_label,
            "assigned_label": assigned_label,
            "accepted": accepted,
            "removed_as_not_a_sign": removed,
            "fallback_reason": fallback_reason,
            "confidence": confidence,
            "margin": margin,
            "probabilities": dict(probabilities),
            "image_probabilities": dict(image_probabilities),
            "numeric_probabilities": dict(numeric_probabilities),
        }
        predictions.append(prediction)
        if removed:
            continue
        if assigned_label not in class_mapping:
            raise ValueError(f"No semantic class id for sign label {assigned_label!r}")

        canonical = candidate.canonical_detection
        routed_instances.append(
            Sam3Instance(
                label=assigned_label,
                class_id=int(class_mapping[assigned_label]),
                prompt=canonical.prompt,
                score=float(canonical.score),
                mask=np.asarray(candidate.mask, dtype=bool),
                box=None
                if canonical.box is None
                else np.asarray(canonical.box, dtype=np.float32),
                source_label=candidate.label,
                sign_classifier_label=predicted_label,
                sign_classifier_assigned_label=assigned_label,
                sign_classifier_confidence=confidence,
                sign_classifier_margin=margin,
                sign_classifier_probabilities=probabilities,
                sign_classifier_image_probabilities=image_probabilities,
                sign_classifier_numeric_probabilities=numeric_probabilities,
                sign_sam3_class_scores=sam3_scores,
                sign_classifier_accepted=accepted,
                sign_classifier_fallback_reason=fallback_reason,
            )
        )
    return [*retained_instances, *routed_instances], tuple(predictions)


def _named_probabilities(
    class_names: tuple[str, ...],
    values: Any,
) -> tuple[tuple[str, float], ...]:
    probabilities = tuple(float(value) for value in values)
    if len(probabilities) != len(class_names):
        raise ValueError(
            f"Expected {len(class_names)} sign probabilities, got {len(probabilities)}"
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("Sign classifier probabilities contain NaN or infinity")
    return tuple(zip(class_names, probabilities))
