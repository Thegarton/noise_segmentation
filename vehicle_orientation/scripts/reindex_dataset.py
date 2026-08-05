#!/usr/bin/env python3
"""Apply manually reviewed front/rear/side folders to the dataset manifest."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vehicle_orientation.dataset import (  # noqa: E402
    CLASS_NAMES,
    IMAGE_SUFFIXES,
    assign_stratified_splits,
    load_jsonl,
    summarize_manifest,
    write_jsonl,
)


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    manifest_path = dataset_dir / "manifest.jsonl"
    dataset_manifest_path = dataset_dir / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest does not exist: {manifest_path}")

    records = load_jsonl(manifest_path)
    by_sample_id = _index_manifest_records(records)
    review_files = scan_review_folders(dataset_dir)
    unknown = sorted(set(review_files) - set(by_sample_id))
    missing = sorted(set(by_sample_id) - set(review_files))
    if unknown:
        raise ValueError(f"Review folders contain files absent from manifest.jsonl: {unknown[:10]}")
    if missing:
        raise ValueError(
            "Review folders are missing manifest samples. Move files between front/rear/side; "
            f"do not rename or delete them. Missing: {missing[:10]}"
        )

    reviewed_at = datetime.now(timezone.utc).isoformat()
    moved = 0
    updated: list[dict[str, Any]] = []
    for record in records:
        sample_id = str(record["sample_id"])
        label, image_path = review_files[sample_id]
        previous_label = str(record["label"])
        item = dict(record)
        item["label"] = label
        item["classifier_image"] = str(image_path.relative_to(dataset_dir))
        item["label_source"] = "manual_review_folder"
        item["reviewed_at"] = reviewed_at
        if previous_label != label:
            moved += 1
            item["previous_label"] = previous_label
        updated.append(item)

    updated = assign_stratified_splits(
        updated,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    summary = summarize_manifest(updated)
    result = {
        "dataset_dir": str(dataset_dir),
        "samples": len(updated),
        "moved_samples": moved,
        "class_counts": summary["class_counts"],
        "split_counts": summary["split_counts"],
        "dry_run": bool(args.dry_run),
    }
    if args.dry_run:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    _write_manifest_atomic(manifest_path, updated)
    for record in updated:
        metadata_path = dataset_dir / str(record["metadata"])
        metadata_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    dataset_manifest = _read_json_object(dataset_manifest_path)
    dataset_manifest["summary"] = summary
    dataset_manifest["split"] = {
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "test_ratio": float(1.0 - args.train_ratio - args.val_ratio),
        "seed": int(args.seed),
        "grouped_by": "source_id",
    }
    manual_review = dataset_manifest.setdefault("manual_review", {})
    manual_review.update(
        {
            "folders": [f"review/{label}" for label in CLASS_NAMES],
            "last_reindexed_at": reviewed_at,
            "moved_samples": moved,
            "class_counts": summary["class_counts"],
        }
    )
    _write_json_atomic(dataset_manifest_path, dataset_manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def scan_review_folders(dataset_dir: Path) -> dict[str, tuple[str, Path]]:
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
                    f"Duplicate review sample {sample_id!r}: {previous_label}/{previous_path.name} and {label}/{path.name}"
                )
            result[sample_id] = (label, path)
    if not result:
        raise ValueError(f"No review images found under {review_root}")
    return result


def _index_manifest_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        sample_id = str(record.get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError("Every manifest record must have a non-empty sample_id")
        if sample_id in result:
            raise ValueError(f"Duplicate sample_id in manifest.jsonl: {sample_id!r}")
        result[sample_id] = record
    return result


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _write_manifest_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_jsonl(temporary, records)
    temporary.replace(path)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update labels from review/front, review/rear, and review/side after manual file moves."
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
