#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")

    result = summarize_classes_logs(output_dir, strict=bool(args.strict))
    print_summary(result)

    if args.out_json:
        write_json(Path(args.out_json).expanduser().resolve(), result)
    if args.out_csv:
        write_class_csv(Path(args.out_csv).expanduser().resolve(), result["classes"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize final SAM3 overlay instances from recursively discovered "
            "classes_log.json files."
        )
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="SAM3 output directory containing one subdirectory per frame.",
    )
    parser.add_argument("--out-json", default=None, help="Optional path for the full JSON summary.")
    parser.add_argument("--out-csv", default=None, help="Optional path for the per-class CSV table.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop on a malformed log instead of reporting and skipping it.",
    )
    return parser.parse_args()


def summarize_classes_logs(output_dir: Path, *, strict: bool = False) -> dict[str, Any]:
    log_paths = sorted(output_dir.rglob("classes_log.json"))
    if not log_paths:
        raise FileNotFoundError(f"No classes_log.json files found under: {output_dir}")

    scores_by_label: dict[str, list[float]] = defaultdict(list)
    frames_by_label: dict[str, set[str]] = defaultdict(set)
    class_ids_by_label: dict[str, set[int]] = defaultdict(set)
    processing_times: list[float] = []
    errors: list[dict[str, str]] = []
    valid_logs = 0
    total_instances = 0

    for log_path in log_paths:
        try:
            payload = json.loads(log_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("root value must be a JSON object")

            frame_id = log_path.parent.name
            instances = payload.get("instances", [])
            if not isinstance(instances, list):
                raise ValueError("'instances' must be a list")

            processing_time = finite_float(payload.get("processing_time_seconds"))
            frame_instances: list[tuple[str, float, int | None]] = []

            for index, instance in enumerate(instances):
                if not isinstance(instance, dict):
                    raise ValueError(f"instances[{index}] must be a JSON object")
                label = instance.get("label")
                if not isinstance(label, str) or not label.strip():
                    raise ValueError(f"instances[{index}].label must be a non-empty string")

                score = finite_float(instance.get("score"))
                if score is None:
                    raise ValueError(f"instances[{index}].score must be a finite number")

                label = label.strip()
                class_id = instance.get("class_id")
                parsed_class_id = None
                if class_id is not None:
                    try:
                        parsed_class_id = int(class_id)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"instances[{index}].class_id must be an integer"
                        ) from exc
                frame_instances.append((label, score, parsed_class_id))

            if processing_time is not None and processing_time >= 0.0:
                processing_times.append(processing_time)
            for label, score, class_id in frame_instances:
                scores_by_label[label].append(score)
                frames_by_label[label].add(frame_id)
                total_instances += 1
                if class_id is not None:
                    class_ids_by_label[label].add(class_id)

            valid_logs += 1
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            if strict:
                raise ValueError(f"Invalid classes log {log_path}: {exc}") from exc
            errors.append({"path": str(log_path), "error": str(exc)})

    classes = []
    for label in sorted(scores_by_label, key=str.casefold):
        scores = scores_by_label[label]
        classes.append(
            {
                "label": label,
                "class_ids": sorted(class_ids_by_label[label]),
                "instance_count": len(scores),
                "frame_count": len(frames_by_label[label]),
                "mean_confidence": statistics.fmean(scores),
                "min_confidence": min(scores),
                "max_confidence": max(scores),
            }
        )

    return {
        "version": 1,
        "output_dir": str(output_dir),
        "logs_found": len(log_paths),
        "logs_read": valid_logs,
        "logs_skipped": len(errors),
        "total_instances": total_instances,
        "processing_time": summarize_values(processing_times),
        "classes": classes,
        "errors": errors,
    }


def finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def summarize_values(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "frame_count": 0,
            "mean_seconds": None,
            "median_seconds": None,
            "min_seconds": None,
            "max_seconds": None,
            "total_seconds": 0.0,
        }
    return {
        "frame_count": len(values),
        "mean_seconds": statistics.fmean(values),
        "median_seconds": statistics.median(values),
        "min_seconds": min(values),
        "max_seconds": max(values),
        "total_seconds": sum(values),
    }


def print_summary(result: dict[str, Any]) -> None:
    timing = result["processing_time"]
    print(
        f"Logs: {result['logs_read']}/{result['logs_found']} read, "
        f"instances: {result['total_instances']}"
    )
    if timing["mean_seconds"] is None:
        print("Average processing time: unavailable")
    else:
        print(
            f"Average processing time: {timing['mean_seconds']:.3f} s/frame "
            f"({timing['frame_count']} timed frames)"
        )

    print()
    print(f"{'class':32} {'objects':>8} {'frames':>8} {'mean':>9} {'min':>9} {'max':>9}")
    print("-" * 81)
    for item in result["classes"]:
        print(
            f"{item['label'][:32]:32} "
            f"{item['instance_count']:8d} "
            f"{item['frame_count']:8d} "
            f"{item['mean_confidence']:9.4f} "
            f"{item['min_confidence']:9.4f} "
            f"{item['max_confidence']:9.4f}"
        )

    if result["logs_skipped"]:
        print(f"\nSkipped malformed logs: {result['logs_skipped']}")
        for error in result["errors"][:10]:
            print(f"  {error['path']}: {error['error']}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_class_csv(path: Path, classes: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "label",
                "class_ids",
                "instance_count",
                "frame_count",
                "mean_confidence",
                "min_confidence",
                "max_confidence",
            ],
        )
        writer.writeheader()
        for item in classes:
            row = dict(item)
            row["class_ids"] = " ".join(str(value) for value in item["class_ids"])
            writer.writerow(row)


if __name__ == "__main__":
    main()
