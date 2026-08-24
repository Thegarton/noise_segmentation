"""Per-frame and folder-level orchestration for the SAM3 pipeline."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from autolabeler.data.class_config import load_semantic_classes
from autolabeler.teachers.sam3_runtime_adapter import (
    clear_visual_feature_cache,
    sam3_runtime_summary,
)

from .outputs import (
    build_classes_log,
    classes_from_semantic_mask,
    collect_images,
    output_dir_for_image,
    outputs_exist,
    save_image_outputs,
    validate_outputs,
)
from .postprocess import (
    apply_sign_type_classification,
    apply_vehicle_orientation,
    build_semantic_outputs,
    build_sign_type_classifier,
    build_vehicle_orientation_classifier,
    collect_overlay_instances,
    deduplicate_instances_by_label,
    instance_orientation_metadata,
    resolve_sign_type_class_mapping,
    resolve_vehicle_class_mapping,
)
from .runtime import (
    add_text_prompt,
    apply_simple_colour_correction_rgb,
    build_predictor,
    close_session,
    configure_sam3_imports,
    extract_arrays,
    load_label_min_scores,
    load_prompt_config,
    resolve_model_dir,
    set_predictor_detection_threshold,
    start_session,
    validate_probability,
)
from .types import (
    SIGN_TYPE_LABELS,
    FolderDetectionSummary,
    ImageResult,
    Sam3Instance,
)


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
    min_mask_size: int,
    vehicle_orientation_classifier: Any | None = None,
    vehicle_prompt_label: str = "vehicle",
    vehicle_class_mapping: dict[str, tuple[str, int]] | None = None,
    vehicle_orientation_min_confidence: float = 0.70,
    vehicle_orientation_min_margin: float = 0.10,
    vehicle_orientation_nms_iou: float = 0.80,
    sign_type_classifier: Any | None = None,
    sign_type_class_mapping: dict[str, int] | None = None,
    sign_classifier_min_confidence: float = 0.50,
    sign_classifier_min_margin: float = 0.05,
    sign_classifier_cluster_iou: float = 0.55,
    sign_classifier_cluster_containment: float = 0.80,
    sign_classifier_context_scale: float = 3.0,
    colour_correction: bool = False,
    save_outputs: bool = True,
    _apply_colour_correction: Any | None = None,
    _save_image_outputs: Any | None = None,
) -> ImageResult:
    from PIL import Image  # noqa: WPS433

    with Image.open(image_path) as opened_image:
        source_image = opened_image.convert("RGB")
    width, height = source_image.size
    model_image_rgb = np.asarray(source_image, dtype=np.uint8)
    if colour_correction:
        correction = _apply_colour_correction or apply_simple_colour_correction_rgb
        model_image_rgb = correction(model_image_rgb)
    clear_visual_feature_cache(predictor)
    session_id = start_session(
        predictor,
        image_path,
        image_rgb=model_image_rgb if colour_correction else None,
    )
    instances: list[Sam3Instance] = []
    try:
        for prompt_idx, (label, prompt) in enumerate(flat_prompts, start=1):
            detection_min_score = float(label_min_scores.get(label, min_score))
            if prompt_log:
                print(
                    f"  [{prompt_idx:03d}/{len(flat_prompts):03d}] {label}: {prompt}",
                    file=sys.stderr,
                    flush=True,
                )

            set_predictor_detection_threshold(predictor, detection_min_score)

            response = add_text_prompt(
                predictor,
                session_id=session_id,
                prompt=prompt,
                min_score=detection_min_score,
            )
            masks, scores, boxes = extract_arrays(
                response.get("outputs"), height=height, width=width
            )
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
                    raise KeyError(
                        f"No semantic or routed class id for prompt label {label!r}"
                    )
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
            raise ValueError(
                "vehicle_class_mapping is required when the orientation classifier is enabled"
            )
        instances = apply_vehicle_orientation(
            image_rgb=model_image_rgb,
            instances=instances,
            classifier=vehicle_orientation_classifier,
            vehicle_prompt_label=vehicle_prompt_label,
            class_mapping=vehicle_class_mapping,
            min_confidence=vehicle_orientation_min_confidence,
            min_margin=vehicle_orientation_min_margin,
            nms_iou=vehicle_orientation_nms_iou,
            min_mask_size=min_mask_size,
        )

    sign_predictions: tuple[dict[str, Any], ...] = ()
    if sign_type_classifier is not None:
        if sign_type_class_mapping is None:
            raise ValueError(
                "sign_type_class_mapping is required when the sign classifier is enabled"
            )
        instances, sign_predictions = apply_sign_type_classification(
            image_rgb=model_image_rgb,
            instances=instances,
            classifier=sign_type_classifier,
            class_mapping=sign_type_class_mapping,
            min_confidence=sign_classifier_min_confidence,
            min_margin=sign_classifier_min_margin,
            min_mask_size=min_mask_size,
            cluster_iou=sign_classifier_cluster_iou,
            cluster_containment=sign_classifier_cluster_containment,
            context_scale=sign_classifier_context_scale,
        )

    semantic_mask, confidence, class_pixel_counts = build_semantic_outputs(
        instances=instances,
        label_to_id=label_to_id,
        shape=(height, width),
        min_mask_size=min_mask_size,
    )
    overlay_instances = collect_overlay_instances(
        instances=instances,
        semantic_mask=semantic_mask,
        confidence=confidence,
        min_mask_size=min_mask_size,
    )
    if save_outputs:
        output_writer = _save_image_outputs or save_image_outputs
        output_writer(
            image=source_image,
            image_path=image_path,
            output_dir=output_dir,
            instances=instances,
            overlay_instances=list(overlay_instances),
            semantic_mask=semantic_mask,
            confidence=confidence,
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
        sign_predictions=sign_predictions,
    )


def sam3_single_image_folder(
    image_dir,
    out_dir,
    prompt_config,
    classes_yaml,
    sam3_root,
    sam3_model_path,
    min_score,
    label_min_scores=None,
    recursive=False,
    max_images=None,
    max_prompts=None,
    use_fa3=False,
    overwrite=False,
    validate=False,
    log_json=False,
    min_mask_size=30,
    vehicle_orientation_checkpoint=None,
    vehicle_orientation_device="auto",
    vehicle_orientation_min_confidence=0.75,
    vehicle_orientation_min_margin=0.10,
    vehicle_orientation_nms_iou=0.80,
    vehicle_prompt_label="vehicle",
    sign_classifier_checkpoint=None,
    sign_classifier_device="cpu",
    sign_classifier_min_confidence=0.50,
    sign_classifier_min_margin=0.05,
    sign_classifier_cluster_iou=0.55,
    sign_classifier_cluster_containment=0.80,
    sign_classifier_context_scale=3.0,
    sam3_only=False,
    prompt_log=False,
    inference_precision="auto",
    cache_visual_features=False,
    frame_image_pairs=None,
    colour_correction=False,
    save_outputs=True,
) -> FolderDetectionSummary:

    validate_probability(
        vehicle_orientation_min_confidence, name="vehicle_orientation_min_confidence"
    )
    validate_probability(
        vehicle_orientation_min_margin, name="vehicle_orientation_min_margin"
    )
    validate_probability(
        vehicle_orientation_nms_iou, name="vehicle_orientation_nms_iou"
    )
    validate_probability(
        sign_classifier_min_confidence, name="sign_classifier_min_confidence"
    )
    validate_probability(sign_classifier_min_margin, name="sign_classifier_min_margin")
    validate_probability(
        sign_classifier_cluster_iou, name="sign_classifier_cluster_iou"
    )
    validate_probability(
        sign_classifier_cluster_containment,
        name="sign_classifier_cluster_containment",
    )
    if (
        not np.isfinite(sign_classifier_context_scale)
        or sign_classifier_context_scale < 1.0
    ):
        raise ValueError(
            "sign_classifier_context_scale must be finite and >= 1, got "
            f"{sign_classifier_context_scale}"
        )
    image_dir = Path(image_dir).expanduser().resolve()

    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if frame_image_pairs is None:
        image_records = [
            (None, image_path)
            for image_path in collect_images(image_dir, recursive=recursive)
        ]
    else:
        image_records = []
        seen_frame_ids: set[str] = set()
        for raw_frame_id, raw_image_path in frame_image_pairs:
            frame_id = str(raw_frame_id)
            image_path = Path(raw_image_path).expanduser().resolve()
            if frame_id in seen_frame_ids:
                raise ValueError(f"Duplicate explicit frame id: {frame_id!r}")
            if not image_path.is_file():
                raise FileNotFoundError(f"Matched image does not exist: {image_path}")
            seen_frame_ids.add(frame_id)
            image_records.append((frame_id, image_path))
    if max_images is not None:
        if max_images <= 0:
            raise ValueError(f"--max-images must be positive, got {max_images}")
        image_records = image_records[:max_images]
    if not image_records:
        raise ValueError(f"No images selected from {image_dir}")

    prompts_by_label = load_prompt_config(prompt_config)
    class_to_id = load_semantic_classes(classes_yaml)

    vehicle_orientation_classifier = None
    vehicle_class_mapping = None
    sign_type_classifier = None
    sign_type_class_mapping = None
    routed_prompt_labels: set[str] = set()

    effective_orientation_checkpoint = vehicle_orientation_checkpoint
    effective_sign_checkpoint = sign_classifier_checkpoint
    if sam3_only:
        if effective_orientation_checkpoint:
            print(
                "--sam3-only is enabled; ignoring --vehicle-orientation-checkpoint",
                file=sys.stderr,
                flush=True,
            )
        effective_orientation_checkpoint = None
        if effective_sign_checkpoint:
            print(
                "--sam3-only is enabled; ignoring --sign-classifier-checkpoint",
                file=sys.stderr,
                flush=True,
            )
        effective_sign_checkpoint = None

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

    if effective_sign_checkpoint:
        missing_prompt_labels = [
            label
            for label in SIGN_TYPE_LABELS
            if label not in prompts_by_label or not prompts_by_label[label]
        ]
        if missing_prompt_labels:
            raise ValueError(
                "Sign classifier requires non-empty SAM3 prompts for all sign types; "
                f"missing={missing_prompt_labels}"
            )
        sign_type_class_mapping = resolve_sign_type_class_mapping(class_to_id)
        sign_type_classifier = build_sign_type_classifier(
            effective_sign_checkpoint,
            device=sign_classifier_device,
        )

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
    detected_classes: set[str] = set()
    processed_frame_ids: list[str] = []
    frame_class_presence: list[frozenset[str]] = []
    results = []
    for index, (explicit_frame_id, image_path) in enumerate(image_records, start=1):
        frame_out = (
            out_dir / explicit_frame_id
            if explicit_frame_id is not None
            else output_dir_for_image(
                out_dir, image_dir, image_path, recursive=recursive
            )
        )
        if (
            save_outputs
            and outputs_exist(
                frame_out,
                classes_log_enabled=log_json,
            )
            and not overwrite
        ):
            metadata_path = frame_out / "metadata.json"
            frame_classes = classes_from_semantic_mask(
                frame_out / "semantic_mask.npy",
                label_to_id=label_to_id,
                min_mask_size=min_mask_size,
            )
            detected_classes.update(frame_classes)
            processed_frame_ids.append(frame_out.name)
            frame_class_presence.append(frozenset(frame_classes))
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

        print(
            f"[{index:04d}/{len(image_records):04d}] {image_path}",
            file=sys.stderr,
            flush=True,
        )
        frame_started_at = time.perf_counter()
        result = process_image(
            predictor=predictor,
            image_path=image_path,
            output_dir=frame_out,
            flat_prompts=flat_prompts,
            label_to_id=label_to_id,
            label_min_scores=effective_label_min_scores,
            min_score=min_score,
            prompt_log=prompt_log,
            min_mask_size=min_mask_size,
            vehicle_orientation_classifier=vehicle_orientation_classifier,
            vehicle_prompt_label=vehicle_prompt_label,
            vehicle_class_mapping=vehicle_class_mapping,
            vehicle_orientation_min_confidence=vehicle_orientation_min_confidence,
            vehicle_orientation_min_margin=vehicle_orientation_min_margin,
            vehicle_orientation_nms_iou=vehicle_orientation_nms_iou,
            sign_type_classifier=sign_type_classifier,
            sign_type_class_mapping=sign_type_class_mapping,
            sign_classifier_min_confidence=sign_classifier_min_confidence,
            sign_classifier_min_margin=sign_classifier_min_margin,
            sign_classifier_cluster_iou=sign_classifier_cluster_iou,
            sign_classifier_cluster_containment=sign_classifier_cluster_containment,
            sign_classifier_context_scale=sign_classifier_context_scale,
            colour_correction=colour_correction,
            save_outputs=save_outputs,
        )
        processing_time_seconds = time.perf_counter() - frame_started_at
        frame_classes = frozenset(result.class_pixel_counts)
        detected_classes.update(frame_classes)
        processed_frame_ids.append(frame_out.name)
        frame_class_presence.append(frame_classes)
        metadata = {
            "version": 1,
            "frame_id": frame_out.name,
            "image_path": str(result.image_path),
            "image_size": [result.image_size[0], result.image_size[1]],
            "sam3_root": str(sam3_root) if sam3_root is not None else None,
            "sam3_model_path": str(model_dir) if model_dir is not None else None,
            "sam3_checkpoint": str(model_dir / "sam3.1_multiplex.pt")
            if model_dir is not None
            else None,
            "prompt_config": str(Path(prompt_config)),
            "classes_yaml": str(Path(classes_yaml)),
            "min_score": float(min_score),
            "label_min_scores_file": str(Path(label_min_scores).expanduser().resolve())
            if label_min_scores
            else None,
            "label_min_score_overrides": label_min_score_overrides,
            "effective_label_min_scores": effective_label_min_scores,
            "instances": len(result.instances),
            "overlay_instances": len(result.overlay_instances),
            "class_pixel_counts": result.class_pixel_counts,
            "mask_nms_iou": float(vehicle_orientation_nms_iou),
            "processing_time_seconds": round(processing_time_seconds, 6),
            "sam3_runtime": sam3_runtime_summary(predictor),
            "physical_gpu_id": os.environ.get("SAM3_PHYSICAL_GPU_ID"),
            "colour_correction": bool(colour_correction),
            "vehicle_orientation": {
                "enabled": vehicle_orientation_classifier is not None,
                "mode": vehicle_orientation_mode,
                "checkpoint": str(
                    Path(effective_orientation_checkpoint).expanduser().resolve()
                )
                if effective_orientation_checkpoint
                else None,
                "prompt_label": vehicle_prompt_label,
                "min_confidence": float(vehicle_orientation_min_confidence),
                "min_margin": float(vehicle_orientation_min_margin),
                "nms_iou": float(vehicle_orientation_nms_iou),
                "predictions": list(result.orientation_predictions),
            },
            "sign_classifier": {
                "enabled": sign_type_classifier is not None,
                "checkpoint": str(
                    Path(effective_sign_checkpoint).expanduser().resolve()
                )
                if effective_sign_checkpoint
                else None,
                "device": sign_classifier_device,
                "min_confidence": float(sign_classifier_min_confidence),
                "min_margin": float(sign_classifier_min_margin),
                "cluster_iou": float(sign_classifier_cluster_iou),
                "cluster_containment": float(sign_classifier_cluster_containment),
                "context_scale": float(sign_classifier_context_scale),
                "predictions": list(result.sign_predictions),
            },
        }

        if save_outputs and log_json:
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

        if save_outputs:
            (frame_out / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        if save_outputs and validate:
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
        "min_score": float(min_score),
        "label_min_scores_file": str(Path(label_min_scores).expanduser().resolve())
        if label_min_scores
        else None,
        "label_min_score_overrides": label_min_score_overrides,
        "effective_label_min_scores": effective_label_min_scores,
        "images": len(image_records),
        "prompts": len(flat_prompts),
        "mask_nms_iou": float(vehicle_orientation_nms_iou),
        "labels": label_to_id,
        "frames": results,
        "sam3_runtime": sam3_runtime_summary(predictor),
        "physical_gpu_id": os.environ.get("SAM3_PHYSICAL_GPU_ID"),
        "colour_correction": bool(colour_correction),
        "vehicle_orientation": {
            "enabled": vehicle_orientation_classifier is not None,
            "mode": vehicle_orientation_mode,
            "checkpoint": str(
                Path(effective_orientation_checkpoint).expanduser().resolve()
            )
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
        "sign_classifier": {
            "enabled": sign_type_classifier is not None,
            "checkpoint": str(Path(effective_sign_checkpoint).expanduser().resolve())
            if effective_sign_checkpoint
            else None,
            "device": sign_classifier_device,
            "min_confidence": float(sign_classifier_min_confidence),
            "min_margin": float(sign_classifier_min_margin),
            "cluster_iou": float(sign_classifier_cluster_iou),
            "cluster_containment": float(sign_classifier_cluster_containment),
            "context_scale": float(sign_classifier_context_scale),
            "class_mapping": sign_type_class_mapping,
        },
    }
    if save_outputs:
        manifest_path = out_dir / "sam3_single_image_folder_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {"images": len(image_records), "manifest": str(manifest_path)}, indent=2
            )
        )
    else:
        print(
            json.dumps({"images": len(image_records), "save_outputs": False}, indent=2)
        )
    return FolderDetectionSummary(
        detected_classes=frozenset(detected_classes),
        frame_ids=tuple(processed_frame_ids),
        frame_class_presence=tuple(frame_class_presence),
    )
