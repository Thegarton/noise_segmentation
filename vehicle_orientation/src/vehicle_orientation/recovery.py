"""Recover a training manifest from manually reviewed generator outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .dataset import (
    CLASS_NAMES,
    IMAGE_SUFFIXES,
    collect_mixed_source_images,
    source_id_from_relative_path,
)


@dataclass(frozen=True)
class RecoveryResult:
    records: list[dict[str, Any]]
    source_frames: int
    review_images: int
    ignored_uncompleted_samples: int


def recover_review_records(
    *,
    image_dir: str | Path,
    dataset_dir: str | Path,
    recursive: bool = True,
    completed_source_ids: Iterable[str] | None = None,
) -> RecoveryResult:
    source_root = Path(image_dir).expanduser().resolve()
    output_root = Path(dataset_dir).expanduser().resolve()
    sources = collect_mixed_source_images(source_root, recursive=recursive)
    by_source_id = {
        source_id_from_relative_path(record["source_relative_path"]): record
        for record in sources
    }
    allowed = None if completed_source_ids is None else {str(value) for value in completed_source_ids}
    review_files = scan_review_images(output_root)
    records: list[dict[str, Any]] = []
    ignored_uncompleted = 0
    for sample_id, (label, review_path) in sorted(review_files.items()):
        source_id = source_id_from_sample_id(sample_id)
        if allowed is not None and source_id not in allowed:
            ignored_uncompleted += 1
            continue
        if source_id not in by_source_id:
            raise ValueError(
                f"Cannot map recovered sample {sample_id!r} to an image in {source_root}; "
                f"derived source id is {source_id!r}"
            )
        source = by_source_id[source_id]
        sample_dir = output_root / "samples" / sample_id
        metadata_path = sample_dir / "metadata.json"
        record: dict[str, Any] = {
            "sample_id": sample_id,
            "source_id": source_id,
            "label": label,
            "initial_label": label,
            "source_path": source["source_path"],
            "source_relative_path": source["source_relative_path"],
            "classifier_image": str(review_path.relative_to(output_root)),
            "classifier_label": label,
            "classifier_confidence": None,
            "classifier_margin": None,
            "classifier_probabilities": {},
            "classifier_needs_review": True,
            "auto_label_source": "recovered_partial_output",
            "label_source": "recovered_review_folder",
            "recovered_metadata": True,
            "metadata": str(metadata_path.relative_to(output_root)),
        }
        _add_existing_asset(record, "rgb_crop", sample_dir / "rgb.png", output_root)
        _add_existing_asset(record, "mask", sample_dir / "mask.png", output_root)
        _add_existing_asset(record, "masked_rgb", sample_dir / "masked_rgb.png", output_root)
        _add_existing_asset(record, "preview", sample_dir / "preview.jpg", output_root)
        prepared_path = output_root / "prepared" / f"{source_id}.jpg"
        annotated_path = output_root / "annotated" / f"{source_id}.jpg"
        _add_existing_asset(record, "prepared_image", prepared_path, output_root)
        _add_existing_asset(record, "annotated_image", annotated_path, output_root)
        mask_path = sample_dir / "mask.png"
        if mask_path.is_file():
            record["mask_pixels"] = _count_mask_pixels(mask_path)
        records.append(record)
    return RecoveryResult(
        records=records,
        source_frames=len(sources),
        review_images=len(review_files),
        ignored_uncompleted_samples=ignored_uncompleted,
    )


def scan_review_images(dataset_dir: Path) -> dict[str, tuple[str, Path]]:
    review_root = dataset_dir / "review"
    result: dict[str, tuple[str, Path]] = {}
    for label in CLASS_NAMES:
        class_dir = review_root / label
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing review folder for {label!r}: {class_dir}")
        for path in sorted(class_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            sample_id = path.stem
            if sample_id in result:
                previous_label, previous_path = result[sample_id]
                raise ValueError(
                    f"Duplicate review sample {sample_id!r}: "
                    f"{previous_label}/{previous_path.name} and {label}/{path.name}"
                )
            result[sample_id] = (label, path)
    return result


def source_id_from_sample_id(sample_id: str) -> str:
    try:
        source_id, instance_index = sample_id.rsplit("_", 1)
    except ValueError as exc:
        raise ValueError(f"Cannot parse generated sample id {sample_id!r}") from exc
    if not source_id or len(instance_index) != 3 or not instance_index.isdigit():
        raise ValueError(f"Cannot parse generated sample id {sample_id!r}; expected <source-id>_<NNN>")
    return source_id


def merge_recovered_with_existing(
    recovered: list[dict[str, Any]],
    existing: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    existing_by_id = {str(record["sample_id"]): record for record in existing}
    result = []
    for recovered_record in recovered:
        sample_id = str(recovered_record["sample_id"])
        merged = dict(existing_by_id.get(sample_id, {}))
        merged.update(recovered_record)
        if sample_id in existing_by_id:
            for key in (
                "initial_label",
                "classifier_checkpoint",
                "classifier_confidence",
                "classifier_margin",
                "classifier_probabilities",
                "classifier_needs_review",
                "vehicle_prompt",
                "vehicle_score",
                "vehicle_box_xywh",
                "crop_bbox_xyxy",
                "crop_padding",
            ):
                if key in existing_by_id[sample_id]:
                    merged[key] = existing_by_id[sample_id][key]
            merged["recovered_metadata"] = bool(existing_by_id[sample_id].get("recovered_metadata", False))
        merged["label"] = recovered_record["label"]
        merged["classifier_image"] = recovered_record["classifier_image"]
        merged["label_source"] = "review_folder_on_resume"
        result.append(merged)
    return result


def _add_existing_asset(record: dict[str, Any], key: str, path: Path, root: Path) -> None:
    if path.is_file():
        record[key] = str(path.relative_to(root))


def _count_mask_pixels(path: Path) -> int:
    from PIL import Image  # noqa: WPS433

    return int(np.count_nonzero(np.asarray(Image.open(path).convert("L")) > 0))
