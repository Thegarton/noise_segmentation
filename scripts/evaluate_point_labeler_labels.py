#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from autolabeler.data.class_config import invert_class_mapping, load_semantic_classes  # noqa: E402


DEFAULT_IGNORE_ID = 255


@dataclass(frozen=True)
class LabelFrame:
    frame_id: str
    path: Path
    kind: str


def main() -> None:
    args = parse_args()
    pred_dir = Path(args.pred_dir).expanduser().resolve()
    gt_dir = Path(args.gt_dir).expanduser().resolve()
    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory does not exist: {pred_dir}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground-truth directory does not exist: {gt_dir}")

    ignore_ids = set(int(value) for value in args.ignore_id)
    id_to_name = load_class_names(
        labels_xml=Path(args.labels_xml).expanduser().resolve() if args.labels_xml else None,
        classes_yaml=Path(args.classes_yaml).expanduser().resolve() if args.classes_yaml else None,
        search_dirs=[gt_dir, pred_dir],
    )
    requested_class_ids = parse_class_ids(args.class_ids)
    result = evaluate_label_dirs(
        pred_dir=pred_dir,
        gt_dir=gt_dir,
        ignore_ids=ignore_ids,
        id_to_name=id_to_name,
        class_ids=requested_class_ids,
        strict=bool(args.strict),
        max_frames=args.max_frames,
    )

    if args.out_json:
        write_json(Path(args.out_json).expanduser().resolve(), result)
    if args.out_csv:
        write_per_class_csv(Path(args.out_csv).expanduser().resolve(), result["per_class"])
    if args.frame_csv:
        write_frame_csv(Path(args.frame_csv).expanduser().resolve(), result["frames"])
    if args.confusion_csv:
        write_confusion_csv(Path(args.confusion_csv).expanduser().resolve(), result)

    print_summary(result, top_k=args.top_k)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate point_labeler labels: prediction folder vs manually cleaned ground truth. "
            "Supports labels/<frame>.label, direct *.label, and export <frame>/semantic_mask.npy layouts."
        )
    )
    parser.add_argument("--pred-dir", required=True, help="Predicted point_labeler labels directory or export directory.")
    parser.add_argument("--gt-dir", required=True, help="Ground-truth point_labeler labels directory or export directory.")
    parser.add_argument(
        "--ignore-id",
        type=int,
        action="append",
        default=[DEFAULT_IGNORE_ID],
        help="GT class id ignored during evaluation. Can be repeated. Default: 255.",
    )
    parser.add_argument("--labels-xml", default=None, help="Optional labels.xml for class names.")
    parser.add_argument("--classes-yaml", default=None, help="Optional classes.yaml with semantic_classes for class names.")
    parser.add_argument("--class-ids", default=None, help="Optional comma/space separated class ids to report.")
    parser.add_argument("--strict", action="store_true", help="Fail if either side has missing frames.")
    parser.add_argument("--max-frames", type=int, default=None, help="Evaluate only the first N common frames.")
    parser.add_argument("--out-json", default=None, help="Optional JSON metrics output path.")
    parser.add_argument("--out-csv", default=None, help="Optional per-class CSV metrics output path.")
    parser.add_argument("--frame-csv", default=None, help="Optional per-frame CSV metrics output path.")
    parser.add_argument("--confusion-csv", default=None, help="Optional confusion matrix CSV output path.")
    parser.add_argument("--top-k", type=int, default=40, help="How many classes to print in the terminal table.")
    return parser.parse_args()


def evaluate_label_dirs(
    *,
    pred_dir: Path,
    gt_dir: Path,
    ignore_ids: set[int],
    id_to_name: dict[int, str],
    class_ids: list[int] | None,
    strict: bool,
    max_frames: int | None,
) -> dict[str, Any]:
    pred_frames = discover_label_frames(pred_dir)
    gt_frames = discover_label_frames(gt_dir)
    if not pred_frames:
        raise FileNotFoundError(f"No label frames found in prediction directory: {pred_dir}")
    if not gt_frames:
        raise FileNotFoundError(f"No label frames found in ground-truth directory: {gt_dir}")

    pred_ids = set(pred_frames)
    gt_ids = set(gt_frames)
    missing_pred = sorted(gt_ids - pred_ids)
    missing_gt = sorted(pred_ids - gt_ids)
    if strict and (missing_pred or missing_gt):
        raise FileNotFoundError(
            f"Frame mismatch: missing in pred={missing_pred[:10]} missing in gt={missing_gt[:10]}"
        )

    frame_ids = sorted(pred_ids & gt_ids)
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError(f"--max-frames must be positive, got {max_frames}")
        frame_ids = frame_ids[:max_frames]
    if not frame_ids:
        raise ValueError("No common frames to evaluate")

    confusion: dict[tuple[int, int], int] = {}
    frames: list[dict[str, Any]] = []
    gt_unique_ids: set[int] = set()
    pred_unique_ids: set[int] = set()

    for frame_id in frame_ids:
        gt = load_label_array(gt_frames[frame_id]).reshape(-1)
        pred = load_label_array(pred_frames[frame_id]).reshape(-1)
        if gt.shape != pred.shape:
            raise ValueError(
                f"Frame {frame_id} shape mismatch: pred {pred.shape} from {pred_frames[frame_id].path}, "
                f"gt {gt.shape} from {gt_frames[frame_id].path}"
            )
        valid = ~np.isin(gt, np.asarray(sorted(ignore_ids), dtype=np.int64))
        valid_gt = gt[valid].astype(np.int64, copy=False)
        valid_pred = pred[valid].astype(np.int64, copy=False)
        update_confusion(confusion, valid_gt, valid_pred)
        gt_unique_ids.update(int(value) for value in np.unique(valid_gt))
        pred_unique_ids.update(int(value) for value in np.unique(valid_pred))
        frames.append(frame_metrics(frame_id, gt, pred, valid, ignore_ids=ignore_ids))

    if class_ids is None:
        report_ids = sorted((gt_unique_ids | pred_unique_ids) - ignore_ids)
    else:
        report_ids = [int(value) for value in class_ids if int(value) not in ignore_ids]

    per_class = per_class_metrics(confusion, report_ids, id_to_name=id_to_name)
    overall = overall_metrics(confusion, per_class)
    return {
        "version": 1,
        "pred_dir": str(pred_dir),
        "gt_dir": str(gt_dir),
        "ignore_ids": sorted(ignore_ids),
        "frame_count": len(frame_ids),
        "frames_compared": frame_ids,
        "missing_pred_frames": missing_pred,
        "missing_gt_frames": missing_gt,
        "overall": overall,
        "per_class": per_class,
        "frames": frames,
        "confusion": confusion_to_json(confusion),
    }


def discover_label_frames(root: Path) -> dict[str, LabelFrame]:
    frames: dict[str, LabelFrame] = {}

    labels_dir = root / "labels"
    if labels_dir.is_dir():
        for path in sorted(labels_dir.glob("*.label")):
            frames[path.stem] = LabelFrame(frame_id=path.stem, path=path, kind="point_labeler_label")

    for path in sorted(root.glob("*.label")):
        frames.setdefault(path.stem, LabelFrame(frame_id=path.stem, path=path, kind="point_labeler_label"))

    for path in sorted(root.iterdir()) if root.is_dir() else []:
        if not path.is_dir():
            continue
        semantic_path = path / "semantic_mask.npy"
        if semantic_path.is_file():
            frames.setdefault(path.name, LabelFrame(frame_id=path.name, path=semantic_path, kind="semantic_mask_npy"))

    for path in sorted(root.glob("*.npy")):
        frames.setdefault(path.stem, LabelFrame(frame_id=path.stem, path=path, kind="flat_npy"))

    return frames


def load_label_array(frame: LabelFrame) -> np.ndarray:
    if frame.path.suffix == ".label":
        arr = np.fromfile(frame.path, dtype=np.uint32)
    elif frame.path.suffix == ".npy":
        arr = np.load(frame.path, allow_pickle=False)
    else:
        raise ValueError(f"Unsupported label file: {frame.path}")
    arr = np.asarray(arr)
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"Label array must be integer, got {arr.dtype}: {frame.path}")
    return arr.astype(np.int64, copy=False).reshape(-1)


def update_confusion(confusion: dict[tuple[int, int], int], gt: np.ndarray, pred: np.ndarray) -> None:
    if gt.size == 0:
        return
    pairs = np.stack([gt, pred], axis=1)
    unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
    for pair, count in zip(unique_pairs, counts):
        key = (int(pair[0]), int(pair[1]))
        confusion[key] = int(confusion.get(key, 0)) + int(count)


def frame_metrics(
    frame_id: str,
    gt: np.ndarray,
    pred: np.ndarray,
    valid: np.ndarray,
    *,
    ignore_ids: set[int],
) -> dict[str, Any]:
    valid_points = int(np.count_nonzero(valid))
    ignored_points = int(valid.size - valid_points)
    correct = int(np.count_nonzero((gt == pred) & valid))
    return {
        "frame_id": frame_id,
        "points": int(gt.size),
        "valid_points": valid_points,
        "ignored_gt_points": ignored_points,
        "correct_points": correct,
        "accuracy": safe_div(correct, valid_points),
        "gt_counts": counts_dict(gt[valid]),
        "pred_counts_on_valid_gt": counts_dict(pred[valid]),
        "ignore_ids": sorted(ignore_ids),
    }


def per_class_metrics(
    confusion: dict[tuple[int, int], int],
    class_ids: list[int],
    *,
    id_to_name: dict[int, str],
) -> list[dict[str, Any]]:
    rows = []
    for class_id in class_ids:
        tp = int(confusion.get((class_id, class_id), 0))
        gt_count = int(sum(count for (gt_id, _), count in confusion.items() if gt_id == class_id))
        pred_count = int(sum(count for (_, pred_id), count in confusion.items() if pred_id == class_id))
        fp = int(pred_count - tp)
        fn = int(gt_count - tp)
        union = int(tp + fp + fn)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        rows.append(
            {
                "id": int(class_id),
                "name": id_to_name.get(int(class_id), f"class_{class_id}"),
                "support": gt_count,
                "predicted": pred_count,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "iou": safe_div(tp, union),
                "precision": precision,
                "recall": recall,
                "f1": safe_div(2.0 * precision * recall, precision + recall),
            }
        )
    return rows


def overall_metrics(confusion: dict[tuple[int, int], int], per_class: list[dict[str, Any]]) -> dict[str, Any]:
    valid_points = int(sum(confusion.values()))
    correct = int(sum(count for (gt_id, pred_id), count in confusion.items() if gt_id == pred_id))
    active = [row for row in per_class if int(row["support"]) > 0 or int(row["predicted"]) > 0]
    supported = [row for row in per_class if int(row["support"]) > 0]
    fw_iou_num = sum(float(row["iou"]) * int(row["support"]) for row in supported)
    return {
        "valid_points": valid_points,
        "correct_points": correct,
        "accuracy": safe_div(correct, valid_points),
        "mean_iou": mean_metric(active, "iou"),
        "mean_iou_supported": mean_metric(supported, "iou"),
        "frequency_weighted_iou": safe_div(fw_iou_num, sum(int(row["support"]) for row in supported)),
        "mean_precision": mean_metric(active, "precision"),
        "mean_recall": mean_metric(active, "recall"),
        "mean_f1": mean_metric(active, "f1"),
        "classes_evaluated": len(active),
        "classes_with_gt": len(supported),
    }


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return float(sum(float(row[key]) for row in rows) / len(rows))


def counts_dict(values: np.ndarray) -> dict[str, int]:
    if values.size == 0:
        return {}
    unique, counts = np.unique(values.astype(np.int64, copy=False), return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(unique, counts)}


def confusion_to_json(confusion: dict[tuple[int, int], int]) -> list[dict[str, int]]:
    return [
        {"gt": int(gt_id), "pred": int(pred_id), "count": int(count)}
        for (gt_id, pred_id), count in sorted(confusion.items())
    ]


def load_class_names(
    *,
    labels_xml: Path | None,
    classes_yaml: Path | None,
    search_dirs: list[Path],
) -> dict[int, str]:
    if labels_xml is not None:
        return read_labels_xml(labels_xml)
    if classes_yaml is not None:
        return invert_class_mapping(load_semantic_classes(str(classes_yaml)))
    for directory in search_dirs:
        candidate = directory / "labels.xml"
        if candidate.is_file():
            return read_labels_xml(candidate)
    return {0: "background", DEFAULT_IGNORE_ID: "ignore"}


def read_labels_xml(path: Path) -> dict[int, str]:
    if not path.is_file():
        raise FileNotFoundError(f"labels.xml does not exist: {path}")
    root = ET.parse(path).getroot()
    id_to_name: dict[int, str] = {}
    for label in root.findall("label"):
        raw_id = (label.findtext("id") or "").strip()
        name = (label.findtext("name") or "").strip()
        if raw_id and name:
            id_to_name[int(raw_id)] = name
    id_to_name.setdefault(0, "background")
    id_to_name.setdefault(DEFAULT_IGNORE_ID, "ignore")
    return id_to_name


def parse_class_ids(value: str | None) -> list[int] | None:
    if value is None:
        return None
    tokens = value.replace(",", " ").split()
    if not tokens:
        raise ValueError("--class-ids was provided but no ids were parsed")
    return [int(token) for token in tokens]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_per_class_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "name", "support", "predicted", "tp", "fp", "fn", "iou", "precision", "recall", "f1"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_frame_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["frame_id", "points", "valid_points", "ignored_gt_points", "correct_points", "accuracy"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def write_confusion_csv(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = sorted(
        {
            int(item["gt"])
            for item in result["confusion"]
        }
        | {int(item["pred"]) for item in result["confusion"]}
    )
    counts = {(int(item["gt"]), int(item["pred"])): int(item["count"]) for item in result["confusion"]}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gt\\pred", *ids])
        for gt_id in ids:
            writer.writerow([gt_id, *[counts.get((gt_id, pred_id), 0) for pred_id in ids]])


def print_summary(result: dict[str, Any], *, top_k: int) -> None:
    overall = result["overall"]
    print(f"Frames: {result['frame_count']}")
    print(f"Valid GT points: {overall['valid_points']}")
    print(f"Accuracy: {overall['accuracy']:.6f}")
    print(f"mIoU: {overall['mean_iou']:.6f}")
    print(f"mIoU supported: {overall['mean_iou_supported']:.6f}")
    print(f"FWIoU: {overall['frequency_weighted_iou']:.6f}")
    if result["missing_pred_frames"] or result["missing_gt_frames"]:
        print(
            f"Missing frames: pred={len(result['missing_pred_frames'])}, gt={len(result['missing_gt_frames'])}"
        )

    rows = sorted(result["per_class"], key=lambda row: (int(row["support"]) == 0, -int(row["support"]), int(row["id"])))
    if top_k > 0:
        rows = rows[:top_k]
    if not rows:
        return
    print()
    print("class_id  class_name                    support  pred      iou     prec    recall  f1")
    for row in rows:
        print(
            f"{int(row['id']):>8}  {str(row['name'])[:28]:<28} "
            f"{int(row['support']):>8} {int(row['predicted']):>6} "
            f"{float(row['iou']):>8.4f} {float(row['precision']):>7.4f} "
            f"{float(row['recall']):>7.4f} {float(row['f1']):>7.4f}"
        )


if __name__ == "__main__":
    main()
