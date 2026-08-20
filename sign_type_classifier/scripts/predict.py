#!/usr/bin/env python3
"""Run a trained sign-type ensemble on an existing sign dataset manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sign_type_classifier.dataset import CLASSIFIER_CLASS_NAMES, load_jsonl, write_jsonl  # noqa: E402
from sign_type_classifier.model import SignTypeEnsembleClassifier  # noqa: E402


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    output_path = Path(args.output_jsonl).expanduser().resolve()
    records = load_jsonl(manifest_path)
    if args.split != "all":
        records = [record for record in records if record.get("split") == args.split]
    if not records:
        raise ValueError(f"No records selected from {manifest_path}")

    classifier = SignTypeEnsembleClassifier(args.checkpoint, device=args.device)
    dataset_root = manifest_path.parent
    results: list[dict[str, Any]] = []
    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        images = [_read_rgb(dataset_root / str(record["context_rgb"])) for record in batch]
        numeric = np.stack(
            [_ordered_numeric_vector(record, classifier.feature_names) for record in batch],
            axis=0,
        )
        decisions = classifier.classify(images, numeric)
        for record, decision in zip(batch, decisions):
            results.append(
                {
                    "sample_id": str(record["sample_id"]),
                    "source_id": str(record["source_id"]),
                    "split": record.get("split"),
                    "target": record.get("label"),
                    "prediction": decision.label,
                    "confidence": decision.confidence,
                    "margin": decision.margin,
                    "probabilities": dict(zip(classifier.class_names, decision.probabilities)),
                    "image_probabilities": dict(zip(classifier.class_names, decision.image_probabilities)),
                    "numeric_probabilities": dict(zip(classifier.class_names, decision.numeric_probabilities)),
                }
            )
        print(f"Processed {min(start + len(batch), len(records))}/{len(records)}", flush=True)

    write_jsonl(output_path, results)
    metrics = _metrics(results)
    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"predictions": str(output_path), "metrics": str(metrics_path), **metrics}, ensure_ascii=False, indent=2))


def _ordered_numeric_vector(record: dict[str, Any], feature_names: tuple[str, ...]) -> np.ndarray:
    feature_map = record.get("numeric_features")
    if isinstance(feature_map, dict) and all(name in feature_map for name in feature_names):
        values = np.asarray([feature_map[name] for name in feature_names], dtype=np.float32)
    else:
        values = np.asarray(record.get("numeric_feature_vector", ()), dtype=np.float32).reshape(-1)
    if values.shape != (len(feature_names),) or not np.isfinite(values).all():
        raise ValueError(
            f"Sample {record.get('sample_id')!r} has invalid numeric features {values.shape}; "
            f"expected {(len(feature_names),)} finite values"
        )
    return values


def _read_rgb(path: Path) -> np.ndarray:
    from PIL import Image  # noqa: WPS433

    if not path.is_file():
        raise FileNotFoundError(f"Context image does not exist: {path}")
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    labeled = [record for record in records if record.get("target") in CLASSIFIER_CLASS_NAMES]
    if not labeled:
        return {"samples": len(records), "labeled_samples": 0}
    class_to_id = {name: index for index, name in enumerate(CLASSIFIER_CLASS_NAMES)}
    confusion = np.zeros((len(CLASSIFIER_CLASS_NAMES), len(CLASSIFIER_CLASS_NAMES)), dtype=np.int64)
    for record in labeled:
        confusion[class_to_id[str(record["target"])], class_to_id[str(record["prediction"])]] += 1
    per_class = {}
    f1_values = []
    for index, name in enumerate(CLASSIFIER_CLASS_NAMES):
        true_positive = int(confusion[index, index])
        false_positive = int(confusion[:, index].sum() - true_positive)
        false_negative = int(confusion[index, :].sum() - true_positive)
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        f1_values.append(f1)
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(confusion[index].sum()),
        }
    return {
        "samples": len(records),
        "labeled_samples": len(labeled),
        "accuracy": float(np.trace(confusion)) / max(int(confusion.sum()), 1),
        "macro_f1": float(np.mean(f1_values)),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify reviewed sign samples with the trained ensemble.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--split", choices=("all", "train", "val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


if __name__ == "__main__":
    main()
