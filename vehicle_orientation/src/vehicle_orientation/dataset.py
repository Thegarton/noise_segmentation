"""Dataset manifests and source-image-aware stratified splitting."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


CLASS_NAMES = ("front", "rear", "side")
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


def collect_mixed_source_images(source_root: Path, *, recursive: bool = True) -> list[dict[str, str]]:
    root = source_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Source root does not exist: {root}")
    iterator = root.rglob("*") if recursive else root.iterdir()
    paths = sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise ValueError(f"Source folder contains no supported images: {root}")
    return [
        {
            "source_path": str(path),
            "source_relative_path": str(path.relative_to(root)),
        }
        for path in paths
    ]

def assign_stratified_splits(
    samples: list[dict[str, Any]],
    *,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> list[dict[str, Any]]:
    if not 0.0 < train_ratio < 1.0 or not 0.0 <= val_ratio < 1.0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("Split ratios must satisfy train>0, val>=0, train+val<1")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        source_id = str(sample["source_id"])
        label = str(sample["label"])
        if label not in CLASS_NAMES:
            raise ValueError(f"Unknown orientation label {label!r}; expected one of {CLASS_NAMES}")
        groups[source_id].append(sample)

    source_to_split = _assign_group_splits(
        groups,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

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


def _assign_group_splits(
    groups: dict[str, list[dict[str, Any]]],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, str]:
    split_names = ("train", "val", "test")
    ratios = np.asarray([train_ratio, val_ratio, 1.0 - train_ratio - val_ratio], dtype=np.float64)
    class_to_index = {name: index for index, name in enumerate(CLASS_NAMES)}
    group_vectors: dict[str, np.ndarray] = {}
    for source_id, records in groups.items():
        vector = np.zeros((len(CLASS_NAMES),), dtype=np.float64)
        for record in records:
            vector[class_to_index[str(record["label"])]] += 1.0
        group_vectors[source_id] = vector

    total_vector = sum(group_vectors.values(), start=np.zeros((len(CLASS_NAMES),), dtype=np.float64))
    target_vectors = ratios[:, None] * total_vector[None, :]
    target_totals = ratios * float(total_vector.sum())
    train_groups, val_groups = _split_counts(len(groups), train_ratio=train_ratio, val_ratio=val_ratio)
    group_quotas = np.asarray([train_groups, val_groups, len(groups) - train_groups - val_groups], dtype=np.int64)

    randomizer = random.Random(seed)
    tie_breakers = {source_id: randomizer.random() for source_id in sorted(groups)}
    rarity = 1.0 / np.maximum(total_vector, 1.0)
    ordered_sources = sorted(
        groups,
        key=lambda source_id: (
            -float(np.sum(group_vectors[source_id] * rarity)),
            -float(np.sum(group_vectors[source_id])),
            tie_breakers[source_id],
            source_id,
        ),
    )

    current_vectors = np.zeros_like(target_vectors)
    current_totals = np.zeros((len(split_names),), dtype=np.float64)
    current_groups = np.zeros((len(split_names),), dtype=np.int64)
    assignments: dict[str, str] = {}
    for source_id in ordered_sources:
        vector = group_vectors[source_id]
        candidates = [index for index in range(len(split_names)) if current_groups[index] < group_quotas[index]]
        if not candidates:
            raise RuntimeError("Internal split error: all group quotas are full")
        scored_candidates = []
        for split_index in candidates:
            proposed_vectors = current_vectors.copy()
            proposed_totals = current_totals.copy()
            proposed_vectors[split_index] += vector
            proposed_totals[split_index] += float(vector.sum())
            class_cost = np.sum((proposed_vectors - target_vectors) ** 2 / (target_vectors + 1.0))
            total_cost = np.sum((proposed_totals - target_totals) ** 2 / (target_totals + 1.0))
            scored_candidates.append((float(class_cost + 0.25 * total_cost), split_index))
        _, chosen = min(scored_candidates, key=lambda item: (item[0], item[1]))
        current_vectors[chosen] += vector
        current_totals[chosen] += float(vector.sum())
        current_groups[chosen] += 1
        assignments[source_id] = split_names[chosen]
    return assignments
