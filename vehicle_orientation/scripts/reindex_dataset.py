#!/usr/bin/env python3
"""Apply manual review folders and optionally merge several reviewed datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
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


@dataclass(frozen=True)
class MergePlan:
    record: dict[str, Any]
    source_dataset: Path
    source_sample_dir: Path
    source_review_image: Path
    source_prepared_image: Path | None


def main() -> None:
    args = parse_args()
    dataset_dirs = _resolve_dataset_dirs(args.dataset_dir)
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else None
    merge_mode = len(dataset_dirs) > 1 or out_dir is not None
    if len(dataset_dirs) > 1 and out_dir is None:
        raise ValueError("Multiple --dataset-dir inputs require --out-dir")
    if args.prune_removed and merge_mode:
        raise ValueError("--prune-removed is available only for in-place single-dataset reindexing")

    if merge_mode:
        assert out_dir is not None
        result = merge_datasets(
            dataset_dirs,
            out_dir=out_dir,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
    else:
        result = reindex_in_place(
            dataset_dirs[0],
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            prune_removed=args.prune_removed,
            dry_run=args.dry_run,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def reindex_in_place(
    dataset_dir: Path,
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    prune_removed: bool,
    dry_run: bool,
) -> dict[str, Any]:
    manifest_path = dataset_dir / "manifest.jsonl"
    dataset_manifest_path = dataset_dir / "dataset_manifest.json"
    records = _load_manifest(manifest_path)
    by_sample_id = _index_manifest_records(records)
    review_files = scan_review_folders(dataset_dir)
    _validate_unknown_review_files(review_files, by_sample_id, dataset_dir=dataset_dir)

    removed_ids = sorted(set(by_sample_id) - set(review_files))
    reviewed_at = _utc_now()
    moved = 0
    updated: list[dict[str, Any]] = []
    for record in records:
        sample_id = str(record["sample_id"])
        if sample_id not in review_files:
            continue
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
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    summary = summarize_manifest(updated)
    result = {
        "mode": "in_place",
        "dataset_dir": str(dataset_dir),
        "samples": len(updated),
        "moved_samples": moved,
        "removed_samples": len(removed_ids),
        "removed_sample_ids": removed_ids,
        "class_counts": summary["class_counts"],
        "split_counts": summary["split_counts"],
        "dry_run": bool(dry_run),
    }
    if dry_run:
        return result

    _write_manifest_atomic(manifest_path, updated)
    _write_sample_metadata(dataset_dir, updated)
    dataset_manifest = _read_json_object(dataset_manifest_path)
    dataset_manifest["summary"] = summary
    dataset_manifest["split"] = _split_metadata(seed=seed, train_ratio=train_ratio, val_ratio=val_ratio)
    manual_review = dataset_manifest.setdefault("manual_review", {})
    previous_removed = {str(value) for value in manual_review.get("removed_sample_ids", [])}
    manual_review.update(
        {
            "folders": [f"review/{label}" for label in CLASS_NAMES],
            "last_reindexed_at": reviewed_at,
            "moved_samples": moved,
            "removed_samples": len(previous_removed | set(removed_ids)),
            "removed_sample_ids": sorted(previous_removed | set(removed_ids)),
            "class_counts": summary["class_counts"],
        }
    )
    _write_json_atomic(dataset_manifest_path, dataset_manifest)
    if prune_removed:
        for sample_id in removed_ids:
            sample_dir = _sample_dir(dataset_dir, by_sample_id[sample_id])
            if sample_dir.is_dir():
                shutil.rmtree(sample_dir)
    return result


def merge_datasets(
    dataset_dirs: list[Path],
    *,
    out_dir: Path,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    overwrite: bool,
    dry_run: bool,
) -> dict[str, Any]:
    _validate_merge_output(out_dir, dataset_dirs)
    plans: list[MergePlan] = []
    input_summaries: list[dict[str, Any]] = []
    for dataset_dir in dataset_dirs:
        records = _load_manifest(dataset_dir / "manifest.jsonl")
        by_sample_id = _index_manifest_records(records)
        review_files = scan_review_folders(dataset_dir)
        _validate_unknown_review_files(review_files, by_sample_id, dataset_dir=dataset_dir)
        namespace = _dataset_namespace(dataset_dir)
        removed_ids = sorted(set(by_sample_id) - set(review_files))
        kept = 0
        for record in records:
            old_sample_id = str(record["sample_id"])
            if old_sample_id not in review_files:
                continue
            label, review_image = review_files[old_sample_id]
            plans.append(
                _make_merge_plan(
                    dataset_dir=dataset_dir,
                    namespace=namespace,
                    record=record,
                    label=label,
                    review_image=review_image,
                )
            )
            kept += 1
        input_summaries.append(
            {
                "dataset_dir": str(dataset_dir),
                "namespace": namespace,
                "manifest_samples": len(records),
                "kept_samples": kept,
                "removed_samples": len(removed_ids),
            }
        )

    records = assign_stratified_splits(
        [plan.record for plan in plans],
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    by_sample_id = {str(record["sample_id"]): record for record in records}
    summary = summarize_manifest(records)
    result = {
        "mode": "merge",
        "out_dir": str(out_dir),
        "input_datasets": len(dataset_dirs),
        "samples": len(records),
        "removed_samples": sum(int(item["removed_samples"]) for item in input_summaries),
        "class_counts": summary["class_counts"],
        "split_counts": summary["split_counts"],
        "dry_run": bool(dry_run),
    }
    if dry_run:
        return result

    _prepare_merge_output(out_dir, overwrite=overwrite)
    for label in CLASS_NAMES:
        (out_dir / "review" / label).mkdir(parents=True, exist_ok=True)
    for plan in plans:
        record = by_sample_id[str(plan.record["sample_id"])]
        _copy_merge_plan(plan, record=record, out_dir=out_dir)
    _write_sample_metadata(out_dir, records)
    _write_manifest_atomic(out_dir / "manifest.jsonl", records)
    merged_manifest = {
        "version": 3,
        "mode": "merged_reviewed_vehicle_orientation",
        "out_dir": str(out_dir),
        "created_at": _utc_now(),
        "inputs": input_summaries,
        "split": _split_metadata(seed=seed, train_ratio=train_ratio, val_ratio=val_ratio),
        "manual_review": {
            "folders": [f"review/{label}" for label in CLASS_NAMES],
            "instructions": "Move or delete PNG files, then re-run reindex_dataset.py on this merged dataset.",
        },
        "summary": summary,
        "manifest_jsonl": str(out_dir / "manifest.jsonl"),
    }
    _write_json_atomic(out_dir / "dataset_manifest.json", merged_manifest)
    return result


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
    return result


def _make_merge_plan(
    *,
    dataset_dir: Path,
    namespace: str,
    record: dict[str, Any],
    label: str,
    review_image: Path,
) -> MergePlan:
    old_sample_id = str(record["sample_id"])
    old_source_id = str(record["source_id"])
    sample_id = f"{namespace}__{old_sample_id}"
    source_id = f"{namespace}__{old_source_id}"
    source_sample_dir = _sample_dir(dataset_dir, record)
    if not source_sample_dir.is_dir():
        raise FileNotFoundError(f"Sample assets do not exist: {source_sample_dir}")

    item = dict(record)
    item.update(
        {
            "sample_id": sample_id,
            "source_id": source_id,
            "label": label,
            "classifier_image": f"review/{label}/{sample_id}{review_image.suffix.lower()}",
            "metadata": f"samples/{sample_id}/metadata.json",
            "label_source": "manual_review_folder",
            "reviewed_at": _utc_now(),
            "merge_origin": {
                "dataset_dir": str(dataset_dir),
                "sample_id": old_sample_id,
                "source_id": old_source_id,
            },
        }
    )
    for key in ("rgb_crop", "mask", "masked_rgb", "preview"):
        value = record.get(key)
        if value:
            item[key] = f"samples/{sample_id}/{Path(str(value)).name}"

    source_prepared = None
    prepared_value = record.get("prepared_image")
    if prepared_value:
        candidate = dataset_dir / str(prepared_value)
        if candidate.is_file():
            source_prepared = candidate
            item["prepared_image"] = f"prepared/{source_id}{candidate.suffix.lower()}"
    return MergePlan(
        record=item,
        source_dataset=dataset_dir,
        source_sample_dir=source_sample_dir,
        source_review_image=review_image,
        source_prepared_image=source_prepared,
    )


def _copy_merge_plan(plan: MergePlan, *, record: dict[str, Any], out_dir: Path) -> None:
    target_sample_dir = out_dir / "samples" / str(record["sample_id"])
    shutil.copytree(plan.source_sample_dir, target_sample_dir)
    review_target = out_dir / str(record["classifier_image"])
    review_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(plan.source_review_image, review_target)
    if plan.source_prepared_image is not None:
        prepared_target = out_dir / str(record["prepared_image"])
        if not prepared_target.exists():
            prepared_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(plan.source_prepared_image, prepared_target)


def _sample_dir(dataset_dir: Path, record: dict[str, Any]) -> Path:
    metadata_value = record.get("metadata")
    if metadata_value:
        return (dataset_dir / str(metadata_value)).parent
    return dataset_dir / "samples" / str(record["sample_id"])


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset manifest does not exist: {path}")
    return load_jsonl(path)


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


def _validate_unknown_review_files(
    review_files: dict[str, tuple[str, Path]],
    manifest_records: dict[str, dict[str, Any]],
    *,
    dataset_dir: Path,
) -> None:
    unknown = sorted(set(review_files) - set(manifest_records))
    if unknown:
        raise ValueError(
            f"Review folders in {dataset_dir} contain renamed or unknown files absent from manifest.jsonl: {unknown[:10]}"
        )


def _dataset_namespace(dataset_dir: Path) -> str:
    safe_name = "".join(character if character.isalnum() or character in "-_" else "_" for character in dataset_dir.name)
    digest = hashlib.sha1(str(dataset_dir).encode("utf-8")).hexdigest()[:8]
    return f"{safe_name or 'dataset'}_{digest}"


def _resolve_dataset_dirs(values: list[str]) -> list[Path]:
    result: list[Path] = []
    seen: set[Path] = set()
    for value in values:
        path = Path(value).expanduser().resolve()
        if path in seen:
            raise ValueError(f"Duplicate --dataset-dir: {path}")
        if not path.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {path}")
        result.append(path)
        seen.add(path)
    return result


def _validate_merge_output(out_dir: Path, dataset_dirs: list[Path]) -> None:
    for dataset_dir in dataset_dirs:
        if out_dir == dataset_dir or out_dir in dataset_dir.parents or dataset_dir in out_dir.parents:
            raise ValueError(f"--out-dir must be disjoint from input dataset directory: {dataset_dir}")


def _prepare_merge_output(path: Path, *, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {path}; use --overwrite")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _write_sample_metadata(dataset_dir: Path, records: list[dict[str, Any]]) -> None:
    for record in records:
        metadata_path = dataset_dir / str(record["metadata"])
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


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


def _split_metadata(*, seed: int, train_ratio: float, val_ratio: float) -> dict[str, Any]:
    return {
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "test_ratio": float(1.0 - train_ratio - val_ratio),
        "seed": int(seed),
        "grouped_by": "source_id",
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Update labels after moving/deleting review images, or merge several reviewed datasets into one dataset."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        action="append",
        required=True,
        help="Input dataset directory. Repeat to merge multiple datasets.",
    )
    parser.add_argument("--out-dir", help="Required for multiple inputs; writes a namespaced merged dataset.")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prune-removed",
        action="store_true",
        help="In single-dataset mode, also delete stable samples/<id> assets for deleted review images.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace a non-empty merge --out-dir.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
