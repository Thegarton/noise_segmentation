#!/usr/bin/env python3
"""Recover manifest.jsonl after interrupted EfficientNet dataset generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vehicle_orientation.dataset import (  # noqa: E402
    assign_stratified_splits,
    summarize_manifest,
    write_jsonl,
)
from vehicle_orientation.recovery import recover_review_records  # noqa: E402


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir).expanduser().resolve()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    manifest_path = dataset_dir / "manifest.jsonl"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"Manifest already exists: {manifest_path}; use --overwrite to rebuild it")

    recovery = recover_review_records(
        image_dir=image_dir,
        dataset_dir=dataset_dir,
        recursive=not args.non_recursive,
    )
    if not recovery.records:
        raise ValueError(f"No reviewed crops were found under {dataset_dir / 'review'}")
    records = assign_stratified_splits(
        recovery.records,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    _write_metadata(dataset_dir, records)
    write_jsonl(manifest_path, records)
    summary = summarize_manifest(records)
    dataset_manifest_path = dataset_dir / "dataset_manifest.json"
    dataset_manifest = _load_json_object(dataset_manifest_path)
    dataset_manifest.update(
        {
            "version": max(3, int(dataset_manifest.get("version", 0))),
            "mode": "recovered_interrupted_efficientnet_generation",
            "source_root": str(image_dir),
            "out_dir": str(dataset_dir),
            "recovery": {
                "metadata_limitations": (
                    "SAM3 score, classifier probabilities, boxes, and crop coordinates cannot be recovered "
                    "for samples written by the old non-checkpointing generator."
                ),
                "review_images": recovery.review_images,
            },
            "split": {
                "train_ratio": float(args.train_ratio),
                "val_ratio": float(args.val_ratio),
                "test_ratio": float(1.0 - args.train_ratio - args.val_ratio),
                "seed": int(args.seed),
                "grouped_by": "source_id",
            },
            "summary": summary,
            "manifest_jsonl": str(manifest_path),
        }
    )
    dataset_manifest_path.write_text(json.dumps(dataset_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "recovered_samples": len(records),
                **summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _write_metadata(dataset_dir: Path, records: list[dict[str, Any]]) -> None:
    for record in records:
        path = dataset_dir / str(record["metadata"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover a training manifest from review/front|rear|side after an interrupted generation run."
    )
    parser.add_argument("--image-dir", required=True, help="The same source image directory used for generation.")
    parser.add_argument("--dataset-dir", required=True, help="Interrupted generator output directory.")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--non-recursive", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
