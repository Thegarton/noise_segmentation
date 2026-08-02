#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from autolabeler.data.class_config import load_semantic_classes  # noqa: E402
from autolabeler.hl320.csv_points import load_hl320_csv, primary_return_mask  # noqa: E402
from autolabeler.hl320.manual_annotations import (  # noqa: E402
    IGNORE_ID,
    build_manual_label_array,
    discover_manual_annotation_files,
    load_class_alias_file,
    parse_inline_class_aliases,
    resolve_annotation_path_for_frame,
)


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
DEFAULT_TILE_SIZE = 100.0
TILE_MARGIN = 20.0
DEFAULT_MAX_RANGE = 200.0
RANGE_MARGIN = 10.0


def main() -> None:
    args = parse_args()
    csv_dir = Path(args.csv_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    classes_yaml = Path(args.classes_yaml).expanduser().resolve()
    image_dir = Path(args.image_dir).expanduser().resolve() if args.image_dir else None

    require_dir(csv_dir, "CSV directory")
    if image_dir is not None:
        require_dir(image_dir, "image directory")
    if not classes_yaml.is_file():
        raise FileNotFoundError(f"Classes YAML does not exist: {classes_yaml}")

    if args.annotation_file and args.annotation_dir:
        raise SystemExit("Use either --annotation-file or --annotation-dir, not both.")
    if not args.annotation_file and not args.annotation_dir:
        raise SystemExit("One of --annotation-file or --annotation-dir is required.")
    if args.annotation_file and not args.frame_id:
        raise SystemExit("--frame-id is required when --annotation-file is used.")

    prepare_output_dir(out_dir, overwrite=args.overwrite)
    labels_root = out_dir / "manual_labels"
    labeler_root = out_dir / "point_labeler"
    labeler_velodyne = labeler_root / "velodyne"
    labeler_labels = labeler_root / "labels"
    labeler_rgb = labeler_root / "point_rgb"
    labeler_images = labeler_root / "image_2"
    labels_root.mkdir(parents=True)
    labeler_velodyne.mkdir(parents=True)
    labeler_labels.mkdir(parents=True)
    if image_dir is not None:
        labeler_rgb.mkdir(parents=True)
        labeler_images.mkdir(parents=True)

    class_to_id = load_semantic_classes(str(classes_yaml))
    ignore_id = resolve_default_id(args.ignore_id, class_to_id=class_to_id, fallback=IGNORE_ID)
    default_label_id = resolve_default_id(args.default_label_id, class_to_id=class_to_id, fallback=ignore_id)
    class_aliases = load_class_alias_file(args.class_map_file)
    class_aliases.update(parse_inline_class_aliases(args.class_map))

    csv_paths = sorted(csv_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No .csv files found in {csv_dir}")

    if args.annotation_file:
        annotation_files = {str(args.frame_id): Path(args.annotation_file).expanduser().resolve()}
    else:
        annotation_files = discover_manual_annotation_files(args.annotation_dir)
        if not annotation_files:
            raise FileNotFoundError(f"No .json/.txt annotation files found in {args.annotation_dir}")

    manifest: dict[str, Any] = {
        "version": 1,
        "format": "hl320_manual_json_import_v1",
        "csv_dir": str(csv_dir),
        "annotation_dir": str(Path(args.annotation_dir).expanduser().resolve()) if args.annotation_dir else None,
        "annotation_file": str(Path(args.annotation_file).expanduser().resolve()) if args.annotation_file else None,
        "out_dir": str(out_dir),
        "manual_labels_dir": str(labels_root),
        "point_labeler_dir": str(labeler_root),
        "classes_yaml": str(classes_yaml),
        "image_dir": str(image_dir) if image_dir else None,
        "index_base": int(args.index_base),
        "ignore_id": int(ignore_id),
        "default_label_id": int(default_label_id),
        "conflict_policy": args.conflict_policy,
        "unknown_policy": args.unknown_policy,
        "out_of_range_policy": args.out_of_range_policy,
        "missing_policy": args.missing_policy,
        "label_layout": "flat_points_all_echo_rows",
        "point_index_definition": "label index == original CSV row number",
        "classes": [
            {"name": name, "id": int(class_id)}
            for name, class_id in sorted(class_to_id.items(), key=lambda item: int(item[1]))
        ],
        "frames": [],
        "skipped_frames": [],
    }

    max_abs_xy = 0.0
    max_range = 0.0
    processed = 0
    total_labeled = 0
    total_points = 0

    for csv_path in csv_paths:
        if args.annotation_file and csv_path.stem != str(args.frame_id):
            continue
        annotation_path = resolve_annotation_path_for_frame(frame_id=csv_path.stem, annotation_files=annotation_files)
        if annotation_path is None:
            handle_missing_frame(csv_path.stem, manifest=manifest, policy=args.missing_policy)
            continue

        frame = load_hl320_csv(csv_path)
        labels = build_manual_label_array(
            annotation_path=annotation_path,
            frame_id=frame.frame_id,
            point_count=frame.point_count,
            class_to_id=class_to_id,
            class_aliases=class_aliases,
            ignore_id=ignore_id,
            default_label_id=default_label_id,
            index_base=int(args.index_base),
            conflict_policy=args.conflict_policy,
            unknown_policy=args.unknown_policy,
            out_of_range_policy=args.out_of_range_policy,
        )
        primary_mask = primary_return_mask(frame)
        primary_labels = labels.labels[primary_mask]

        frame_labels_dir = labels_root / frame.frame_id
        frame_labels_dir.mkdir(parents=True)
        np.save(frame_labels_dir / "semantic_mask.npy", labels.labels.astype(np.int32, copy=False))
        np.save(frame_labels_dir / "primary_semantic_mask.npy", primary_labels.astype(np.int32, copy=False))

        labeler_bin_path = labeler_velodyne / f"{frame.frame_id}.bin"
        labeler_label_path = labeler_labels / f"{frame.frame_id}.label"
        frame.points.astype(np.float32, copy=False).tofile(labeler_bin_path)
        labels.labels.astype(np.uint32, copy=False).tofile(labeler_label_path)

        image_path = find_frame_image(image_dir, frame.frame_id) if image_dir is not None else None
        copied_image_path = copy_frame_image(image_path=image_path, out_image_dir=labeler_images, frame_id=frame.frame_id)
        point_rgb_path = None
        if image_path is not None and "cxd" in frame.fields and "cyd" in frame.fields:
            point_rgb = sample_point_rgb(image_path=image_path, cxd=frame.fields["cxd"], cyd=frame.fields["cyd"])
            point_rgb_path = labeler_rgb / f"{frame.frame_id}.rgb"
            point_rgb.tofile(point_rgb_path)

        frame_max_abs_xy, frame_max_range = frame_extent(frame.points)
        max_abs_xy = max(max_abs_xy, frame_max_abs_xy)
        max_range = max(max_range, frame_max_range)

        frame_metadata = {
            "frame_id": frame.frame_id,
            "source_csv": str(csv_path),
            "source_annotation": str(annotation_path),
            "point_count": int(frame.point_count),
            "primary_point_count": int(np.count_nonzero(primary_mask)),
            "label_layout": "flat_points_all_echo_rows",
            "primary_label_layout": "blockID == 0 selected from all_echo labels",
            "semantic_mask": str(frame_labels_dir / "semantic_mask.npy"),
            "primary_semantic_mask": str(frame_labels_dir / "primary_semantic_mask.npy"),
            "point_labeler_velodyne": str(labeler_bin_path),
            "point_labeler_label": str(labeler_label_path),
            "point_rgb": str(point_rgb_path) if point_rgb_path else None,
            "image": str(copied_image_path) if copied_image_path else None,
            "csv_columns": frame.columns,
            "stats": labels.stats,
        }
        (frame_labels_dir / "metadata.json").write_text(
            json.dumps(frame_metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        manifest["frames"].append(frame_metadata)
        processed += 1
        total_points += frame.point_count
        total_labeled += int(labels.stats["labeled_points"])

    if processed == 0:
        raise RuntimeError("No frames were imported. Check --annotation-dir/--frame-id and --missing-policy.")

    write_labels_xml(labeler_root / "labels.xml", class_to_id)
    write_identity_calib(labeler_root / "calib.txt")
    write_identity_poses(labeler_root / "poses.txt", processed)
    (labeler_root / "settings.cfg").write_text(
        settings_content(labeler_root, max_abs_xy=max_abs_xy, max_range=max_range),
        encoding="utf-8",
    )
    write_bridge_manifest(
        labeler_root / "bridge_manifest.json",
        manifest=manifest,
        labeler_root=labeler_root,
        classes=class_to_id,
    )

    manifest["summary"] = {
        "processed_frames": int(processed),
        "total_points": int(total_points),
        "total_labeled_points": int(total_labeled),
        "total_default_or_ignore_points": int(total_points - total_labeled),
    }
    manifest_path = out_dir / "hl320_manual_json_import_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "processed_frames": processed,
                "manual_labels_dir": str(labels_root),
                "point_labeler_dir": str(labeler_root),
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import the new HL320 manual JSON format. Each annotation index is treated as the original CSV row index, "
            "including all echo-channel rows."
        )
    )
    parser.add_argument("--csv-dir", required=True, help="Directory with all-echo HL320 CSV files.")
    parser.add_argument("--annotation-dir", default=None, help="Directory with per-frame .json/.txt annotation files.")
    parser.add_argument("--annotation-file", default=None, help="Single annotation JSON/TXT file. Requires --frame-id.")
    parser.add_argument("--frame-id", default=None, help="Frame id for --annotation-file, e.g. 000091.")
    parser.add_argument("--out-dir", required=True, help="Output root. Writes manual_labels/ and point_labeler/.")
    parser.add_argument("--classes-yaml", default=str(REPO_ROOT / "configs" / "classes.yaml"))
    parser.add_argument("--image-dir", default=None, help="Optional matching image folder for image_2 and point_rgb.")
    parser.add_argument("--class-map", action="append", default=[], help="Extra label mapping, e.g. '串扰=crosstalk_noise_1'.")
    parser.add_argument("--class-map-file", default=None, help="Optional JSON or text file with source_label=target_class mappings.")
    parser.add_argument("--index-base", type=int, choices=(0, 1), default=0, help="Annotation index base. Current format is 0.")
    parser.add_argument("--ignore-id", default="ignore", help="Ignore class name or numeric id. Default: ignore.")
    parser.add_argument("--default-label-id", default="ignore", help="Initial label for all unlabeled points. Default: ignore.")
    parser.add_argument("--missing-policy", choices=("skip", "error"), default="skip")
    parser.add_argument("--unknown-policy", choices=("ignore", "error"), default="ignore")
    parser.add_argument("--out-of-range-policy", choices=("ignore", "error"), default="error")
    parser.add_argument("--conflict-policy", choices=("priority", "last", "first", "error"), default="priority")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_dir(path: Path, description: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{description} does not exist: {path}")


def prepare_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}. Use --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def resolve_default_id(value: str | int, *, class_to_id: dict[str, int], fallback: int) -> int:
    if isinstance(value, int):
        return int(value)
    raw = str(value).strip()
    if raw == "":
        return int(fallback)
    if raw.lstrip("-").isdigit():
        return int(raw)
    normalized = raw.casefold()
    for name, class_id in class_to_id.items():
        if str(name).casefold() == normalized:
            return int(class_id)
    return int(fallback)


def handle_missing_frame(frame_id: str, *, manifest: dict[str, Any], policy: str) -> None:
    if policy == "error":
        raise FileNotFoundError(f"No manual annotation file found for frame {frame_id!r}")
    manifest["skipped_frames"].append({"frame_id": frame_id, "reason": "missing_annotation"})


def find_frame_image(image_dir: Path | None, frame_id: str) -> Path | None:
    if image_dir is None:
        return None
    for suffix in IMAGE_SUFFIXES:
        candidate = image_dir / f"{frame_id}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def copy_frame_image(*, image_path: Path | None, out_image_dir: Path, frame_id: str) -> Path | None:
    if image_path is None:
        return None
    target = out_image_dir / f"{frame_id}{image_path.suffix.lower()}"
    shutil.copy2(image_path, target)
    return target


def sample_point_rgb(*, image_path: Path, cxd: np.ndarray, cyd: np.ndarray) -> np.ndarray:
    from PIL import Image  # noqa: WPS433

    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    height, width = image.shape[:2]
    rgb = np.zeros((cxd.size, 3), dtype=np.uint8)
    finite_projection = np.isfinite(cxd) & np.isfinite(cyd)
    u = np.full(cxd.size, -1, dtype=np.int64)
    v = np.full(cyd.size, -1, dtype=np.int64)
    u[finite_projection] = round_like_point_labeler(cxd[finite_projection])
    v[finite_projection] = round_like_point_labeler(cyd[finite_projection])
    inside = finite_projection & (u >= 0) & (v >= 0) & (u < width) & (v < height)
    rgb[inside] = image[v[inside], u[inside]]
    return rgb


def round_like_point_labeler(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    rounded = np.where(arr >= 0.0, np.floor(arr + 0.5), np.ceil(arr - 0.5))
    return rounded.astype(np.int64)


def frame_extent(points: np.ndarray) -> tuple[float, float]:
    xyz = np.asarray(points[:, :3], dtype=np.float64)
    finite = np.all(np.isfinite(xyz), axis=1)
    if not np.any(finite):
        return 0.0, 0.0
    xyz = xyz[finite]
    return float(np.max(np.abs(xyz[:, :2]))), float(np.max(np.linalg.norm(xyz, axis=1)))


def write_labels_xml(path: Path, class_to_id: dict[str, int]) -> None:
    root = ET.Element("config")
    for name, label_id in sorted(class_to_id.items(), key=lambda item: int(item[1])):
        label = ET.SubElement(root, "label")
        ET.SubElement(label, "id").text = str(int(label_id))
        ET.SubElement(label, "name").text = str(name)
        ET.SubElement(label, "description").text = ""
        ET.SubElement(label, "color").text = " ".join(str(value) for value in label_color(str(name), int(label_id)))
        ET.SubElement(label, "root").text = "manual"
        ET.SubElement(label, "macro").text = ""
        ET.SubElement(label, "category").text = str(name)
    indent_xml(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def label_color(name: str, label_id: int) -> tuple[int, int, int]:
    explicit = {
        0: (32, 32, 32),
        1: (255, 85, 127),
        2: (255, 85, 0),
        5: (0, 85, 0),
        9: (120, 219, 148),
        10: (191, 171, 235),
        12: (76, 68, 117),
        14: (66, 165, 42),
        15: (0, 0, 255),
        20: (255, 0, 0),
        22: (255, 170, 0),
        23: (255, 170, 255),
        24: (0, 0, 127),
        255: (120, 120, 120),
    }
    if int(label_id) in explicit:
        return explicit[int(label_id)]
    seed = (int(label_id) * 1009 + sum(ord(char) for char in name) * 17) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    return tuple(int(value) for value in rng.integers(40, 240, size=3))


def indent_xml(element: ET.Element, level: int = 0) -> None:
    indentation = "\n" + level * "  "
    if len(element):
        if not element.text or not element.text.strip():
            element.text = indentation + "  "
        for child in element:
            indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indentation
    if level and (not element.tail or not element.tail.strip()):
        element.tail = indentation


def write_identity_calib(path: Path) -> None:
    path.write_text(
        "P2: 1 0 0 0 0 1 0 0 0 0 1 0\n"
        "Tr: 1 0 0 0 0 1 0 0 0 0 1 0\n",
        encoding="utf-8",
    )


def write_identity_poses(path: Path, frame_count: int) -> None:
    identity = "1 0 0 0 0 1 0 0 0 0 1 0\n"
    path.write_text(identity * int(frame_count), encoding="utf-8")


def settings_content(labeler_root: Path, *, max_abs_xy: float, max_range: float) -> str:
    labels_xml = labeler_root / "labels.xml"
    tile_size = max(DEFAULT_TILE_SIZE, 2.0 * max_abs_xy + TILE_MARGIN)
    max_range_value = max(DEFAULT_MAX_RANGE, max_range + RANGE_MARGIN)
    return "\n".join(
        [
            f"labels file: {labels_xml}",
            "point size: 4",
            "render points as spheres: true",
            "shade point spheres: true",
            "show intensity: false",
            "allow velodyne only: true",
            "point cloud source: velodyne",
            "min range: 0.001",
            f"max range: {max_range_value:.3f}",
            f"tile size: {tile_size:.3f}",
            "max scans: 500",
            "",
        ]
    )


def write_bridge_manifest(path: Path, *, manifest: dict[str, Any], labeler_root: Path, classes: dict[str, int]) -> None:
    payload = {
        "version": 1,
        "label_layout": "flat_points",
        "source_format": "hl320_manual_json_all_echo",
        "point_index_definition": "label index == original CSV row number, including echo rows",
        "labeler_dir": str(labeler_root),
        "manual_import_manifest": str(Path(manifest["out_dir"]) / "hl320_manual_json_import_manifest.json"),
        "classes": [{"name": name, "id": int(class_id)} for name, class_id in sorted(classes.items(), key=lambda item: int(item[1]))],
        "frames": [
            {
                "frame_id": frame["frame_id"],
                "label_layout": "flat_points",
                "point_count": frame["point_count"],
                "source_csv": frame["source_csv"],
                "source_annotation": frame["source_annotation"],
                "velodyne": frame["point_labeler_velodyne"],
                "label": frame["point_labeler_label"],
                "point_rgb": frame["point_rgb"],
                "image": frame["image"],
                "source_metadata_payload": {
                    "semantic_classes": {name: int(class_id) for name, class_id in classes.items()},
                    "ignore_index": int(manifest["ignore_id"]),
                    "label_layout": "flat_points",
                    "point_count": int(frame["point_count"]),
                    "source_csv": frame["source_csv"],
                    "source_annotation": frame["source_annotation"],
                },
            }
            for frame in manifest["frames"]
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
