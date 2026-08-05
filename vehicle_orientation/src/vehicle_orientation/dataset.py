"""Dataset manifests and source-image-aware stratified splitting."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


CLASS_NAMES = ("front", "rear", "other")
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


def parse_folder_mappings(values: Iterable[str]) -> dict[str, str]:
    mapping = {name: name for name in CLASS_NAMES}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Folder mapping must be LABEL=RELATIVE_PATH, got {value!r}")
        label, relative = (token.strip() for token in value.split("=", 1))
        if label not in CLASS_NAMES:
            raise ValueError(f"Unknown orientation label {label!r}; expected one of {CLASS_NAMES}")
        if not relative:
            raise ValueError(f"Folder mapping for {label!r} has an empty path")
        mapping[label] = relative
    return mapping


def collect_source_images(source_root: Path, folder_mapping: dict[str, str]) -> list[dict[str, str]]:
    root = source_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Source root does not exist: {root}")
    records: list[dict[str, str]] = []
    for label in CLASS_NAMES:
        class_dir = (root / folder_mapping[label]).resolve()
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing source folder for {label!r}: {class_dir}")
        paths = sorted(path for path in class_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        if not paths:
            raise ValueError(f"Source folder for {label!r} contains no supported images: {class_dir}")
        for path in paths:
            records.append(
                {
                    "label": label,
                    "source_path": str(path),
                    "source_relative_path": str(path.relative_to(root)),
                }
            )
    return records


def limit_sources_balanced(records: list[dict[str, str]], max_images: int | None) -> list[dict[str, str]]:
    if max_images is None:
        return records
    if max_images <= 0:
        raise ValueError(f"max_images must be positive, got {max_images}")
    queues = {label: [record for record in records if record["label"] == label] for label in CLASS_NAMES}
    selected: list[dict[str, str]] = []
    while len(selected) < max_images and any(queues.values()):
        for label in CLASS_NAMES:
            if queues[label] and len(selected) < max_images:
                selected.append(queues[label].pop(0))
    return selected


def assign_stratified_splits(
    samples: list[dict[str, Any]],
    *,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> list[dict[str, Any]]:
    if not 0.0 < train_ratio < 1.0 or not 0.0 <= val_ratio < 1.0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("Split ratios must satisfy train>0, val>=0, train+val<1")
    source_labels: dict[str, str] = {}
    for sample in samples:
        source_id = str(sample["source_id"])
        label = str(sample["label"])
        previous = source_labels.setdefault(source_id, label)
        if previous != label:
            raise ValueError(f"Source {source_id!r} has conflicting labels: {previous!r} and {label!r}")

    by_label: dict[str, list[str]] = defaultdict(list)
    for source_id, label in source_labels.items():
        by_label[label].append(source_id)
    source_to_split: dict[str, str] = {}
    for label in CLASS_NAMES:
        source_ids = sorted(by_label.get(label, []))
        random.Random(f"{seed}:{label}").shuffle(source_ids)
        train_count, val_count = _split_counts(len(source_ids), train_ratio=train_ratio, val_ratio=val_ratio)
        for index, source_id in enumerate(source_ids):
            if index < train_count:
                split = "train"
            elif index < train_count + val_count:
                split = "val"
            else:
                split = "test"
            source_to_split[source_id] = split

    result = []
    for sample in samples:
        updated = dict(sample)
        updated["split"] = source_to_split[str(sample["source_id"])]
        result.append(updated)
    return result


def summarize_manifest(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "samples": len(samples),
        "class_counts": dict(sorted(Counter(str(item["label"]) for item in samples).items())),
        "split_counts": dict(sorted(Counter(str(item.get("split", "unassigned")) for item in samples).items())),
        "split_class_counts": {
            split: dict(sorted(Counter(str(item["label"]) for item in samples if item.get("split") == split).items()))
            for split in ("train", "val", "test")
        },
    }


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        value = json.loads(raw_line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object in {path}:{line_number}")
        records.append(value)
    return records


def _split_counts(total: int, *, train_ratio: float, val_ratio: float) -> tuple[int, int]:
    if total <= 0:
        return 0, 0
    if total == 1:
        return 1, 0
    if total == 2:
        return 1, 1
    val_count = max(1, int(round(total * val_ratio))) if val_ratio > 0.0 else 0
    test_count = max(1, int(round(total * (1.0 - train_ratio - val_ratio))))
    train_count = total - val_count - test_count
    if train_count < 1:
        train_count = 1
        if val_count >= test_count and val_count > 1:
            val_count -= 1
        else:
            test_count -= 1
    return train_count, val_count

