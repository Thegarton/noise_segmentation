#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
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

from autolabeler.data.class_config import load_semantic_classes  # noqa: E402


IGNORE_ID = 255
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
DEFAULT_TILE_SIZE = 100.0
TILE_MARGIN = 20.0
DEFAULT_MAX_RANGE = 200.0
RANGE_MARGIN = 10.0


@dataclass(frozen=True)
class CsvPointFrame:
    frame_id: str
    path: Path
    points: np.ndarray
    cxd: np.ndarray
    cyd: np.ndarray
    columns: list[str]


@dataclass(frozen=True)
class LiftedLabels:
    labels: np.ndarray
    confidence: np.ndarray
    stats: dict[str, Any]


def main() -> None:
    args = parse_args()
    csv_dir = Path(args.csv_dir).expanduser().resolve()
    sam3_dir = Path(args.sam3_dir).expanduser().resolve()
    out_dir = Path(args.out_labeler_dir).expanduser().resolve()
    image_dir = Path(args.image_dir).expanduser().resolve() if args.image_dir else None
    classes_yaml = Path(args.classes_yaml).expanduser().resolve()

    require_dir(csv_dir, "CSV directory")
    require_dir(sam3_dir, "SAM3 output directory")
    if image_dir is not None:
        require_dir(image_dir, "image directory")
    if not classes_yaml.is_file():
        raise FileNotFoundError(f"Classes YAML does not exist: {classes_yaml}")
    if not 0.0 <= args.min_confidence <= 1.0:
        raise ValueError(f"--min-confidence must be in [0, 1], got {args.min_confidence}")

    prepare_output_dir(out_dir, overwrite=args.overwrite)
    class_to_id = load_semantic_classes(str(classes_yaml))
    known_ids = {int(class_id) for class_id in class_to_id.values()}
    class_definitions = [(str(name), int(class_id)) for name, class_id in class_to_id.items()]

    csv_paths = sorted(csv_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")

    velodyne_dir = out_dir / "velodyne"
    labels_dir = out_dir / "labels"
    point_rgb_dir = out_dir / "point_rgb"
    out_image_dir = out_dir / "image_2"
    velodyne_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    if image_dir is not None:
        point_rgb_dir.mkdir(parents=True)
        out_image_dir.mkdir(parents=True)

    manifest: dict[str, Any] = {
        "version": 1,
        "label_layout": "flat_points",
        "source_format": "csv_sam3_projection",
        "csv_dir": str(csv_dir),
        "sam3_dir": str(sam3_dir),
        "image_dir": str(image_dir) if image_dir is not None else None,
        "classes_yaml": str(classes_yaml),
        "min_confidence": float(args.min_confidence),
        "ignore_index": IGNORE_ID,
        "classes": [{"name": name, "id": class_id} for name, class_id in class_definitions],
        "frames": [],
    }

    total_stats = init_total_stats()
    max_abs_xy = 0.0
    max_range = 0.0
    for csv_path in csv_paths:
        frame = load_csv_point_frame(csv_path)
        frame_max_abs_xy, frame_max_range = frame_extent(frame.points)
        max_abs_xy = max(max_abs_xy, frame_max_abs_xy)
        max_range = max(max_range, frame_max_range)

        semantic_path = sam3_dir / frame.frame_id / "semantic_mask.npy"
        confidence_path = sam3_dir / frame.frame_id / "confidence.npy"
        semantic = load_semantic_mask(semantic_path)
        confidence = load_confidence_mask(confidence_path, expected_shape=semantic.shape)
        lifted = lift_mask_to_points(
            semantic=semantic,
            confidence=confidence,
            cxd=frame.cxd,
            cyd=frame.cyd,
            known_ids=known_ids,
            min_confidence=float(args.min_confidence),
        )

        velodyne_path = velodyne_dir / f"{frame.frame_id}.bin"
        label_path = labels_dir / f"{frame.frame_id}.label"
        frame.points.astype(np.float32, copy=False).tofile(velodyne_path)
        lifted.labels.astype(np.uint32, copy=False).tofile(label_path)

        image_path = find_frame_image(image_dir, frame.frame_id) if image_dir is not None else None
        copied_image_path = copy_frame_image(image_path=image_path, out_image_dir=out_image_dir, frame_id=frame.frame_id)
        point_rgb_path = None
        if image_path is not None:
            point_rgb = sample_point_rgb(image_path=image_path, cxd=frame.cxd, cyd=frame.cyd)
            point_rgb_path = point_rgb_dir / f"{frame.frame_id}.rgb"
            point_rgb.tofile(point_rgb_path)

        frame_manifest = {
            "frame_id": frame.frame_id,
            "label_layout": "flat_points",
            "source_csv": str(csv_path),
            "source_sam3_mask": str(semantic_path),
            "source_sam3_confidence": str(confidence_path) if confidence_path.is_file() else None,
            "source_image": str(image_path) if image_path is not None else None,
            "image": str(copied_image_path) if copied_image_path is not None else None,
            "velodyne": str(velodyne_path),
            "label": str(label_path),
            "point_rgb": str(point_rgb_path) if point_rgb_path is not None else None,
            "point_count": int(frame.points.shape[0]),
            "shape": [int(frame.points.shape[0])],
            "sam3_mask_shape": [int(value) for value in semantic.shape],
            "csv_columns": frame.columns,
            "projection_stats": lifted.stats,
            "source_metadata_payload": {
                "semantic_classes": {name: int(class_id) for name, class_id in class_definitions},
                "ignore_index": IGNORE_ID,
                "label_layout": "flat_points",
                "point_count": int(frame.points.shape[0]),
                "source_csv": str(csv_path),
                "source_sam3_mask": str(semantic_path),
            },
        }
        manifest["frames"].append(frame_manifest)
        accumulate_stats(total_stats, lifted.stats)

    manifest["summary"] = total_stats
    write_labels_xml(out_dir / "labels.xml", class_definitions)
    write_identity_calib(out_dir / "calib.txt")
    write_identity_poses(out_dir / "poses.txt", len(manifest["frames"]))
    (out_dir / "settings.cfg").write_text(
        settings_content(out_dir, max_abs_xy=max_abs_xy, max_range=max_range),
        encoding="utf-8",
    )
    (out_dir / "bridge_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "frames": len(manifest["frames"]),
                "out_dir": str(out_dir),
                "accepted_points": total_stats["accepted"],
                "ignored_points": total_stats["ignored"],
                "manifest": str(out_dir / "bridge_manifest.json"),
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project SAM3 image masks to LiDAR points and build a point_labeler dataset.")
    parser.add_argument("--csv-dir", required=True, help="Directory with per-frame CSV point clouds.")
    parser.add_argument("--sam3-dir", required=True, help="SAM3 output directory with <frame>/semantic_mask.npy.")
    parser.add_argument("--out-labeler-dir", required=True, help="Output point_labeler dataset directory.")
    parser.add_argument("--classes-yaml", required=True, help="Class taxonomy YAML with semantic_classes.")
    parser.add_argument("--image-dir", default=None, help="Optional image directory used for image_2 copies and point_rgb.")
    parser.add_argument("--min-confidence", type=float, default=0.7, help="Minimum SAM3 pixel confidence accepted for point labels.")
    parser.add_argument("--overwrite", action="store_true", help="Delete and recreate an existing output directory.")
    return parser.parse_args()


def require_dir(path: Path, description: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{description} does not exist: {path}")


def prepare_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if any(path.iterdir()):
            if not overwrite:
                raise FileExistsError(f"Output directory is not empty: {path}. Use --overwrite to replace it.")
            shutil.rmtree(path)
        else:
            path.rmdir()
    path.mkdir(parents=True)


def load_csv_point_frame(path: Path) -> CsvPointFrame:
    with path.open("r", encoding="utf-8") as handle:
        header_line = read_next_data_line(handle, source=path)
        columns = split_table_row(header_line)
        column_map = {name.strip().casefold(): index for index, name in enumerate(columns)}
        required = {name: require_column(column_map, name, path) for name in ("x", "y", "z", "intensity", "cxd", "cyd")}
        max_column = max(required.values())

        points: list[list[float]] = []
        cxd: list[float] = []
        cyd: list[float] = []
        for line_number, raw_line in enumerate(handle, start=2):
            stripped = raw_line.strip()
            if not stripped:
                continue
            tokens = split_table_row(stripped)
            if len(tokens) <= max_column:
                raise ValueError(f"{path}:{line_number}: expected at least {max_column + 1} columns, got {len(tokens)}")
            x = parse_float(tokens[required["x"]], path, line_number, "x")
            y = parse_float(tokens[required["y"]], path, line_number, "y")
            z = parse_float(tokens[required["z"]], path, line_number, "z")
            intensity = parse_float(tokens[required["intensity"]], path, line_number, "intensity")
            points.append([x, y, z, intensity])
            cxd.append(parse_float(tokens[required["cxd"]], path, line_number, "Cxd"))
            cyd.append(parse_float(tokens[required["cyd"]], path, line_number, "Cyd"))

    if not points:
        raise ValueError(f"CSV point cloud contains no points: {path}")
    return CsvPointFrame(
        frame_id=path.stem,
        path=path,
        points=np.asarray(points, dtype=np.float32),
        cxd=np.asarray(cxd, dtype=np.float64),
        cyd=np.asarray(cyd, dtype=np.float64),
        columns=columns,
    )


def read_next_data_line(handle: Any, *, source: Path) -> str:
    for raw_line in handle:
        stripped = raw_line.strip()
        if stripped:
            return stripped
    raise ValueError(f"CSV point cloud is empty: {source}")


def split_table_row(line: str) -> list[str]:
    if "," in line:
        return [value.strip() for value in line.split(",")]
    return line.replace("\t", " ").split()


def require_column(column_map: dict[str, int], name: str, source: Path) -> int:
    key = name.casefold()
    if key not in column_map:
        raise ValueError(f"CSV {source} is missing required column {name!r}")
    return int(column_map[key])


def parse_float(value: str, source: Path, line_number: int, column: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{source}:{line_number}: cannot parse {column}={value!r} as float") from exc


def load_semantic_mask(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing SAM3 semantic mask: {path}")
    arr = np.load(path, allow_pickle=False)
    if arr.ndim != 2:
        raise ValueError(f"SAM3 semantic mask {path} must be 2D, got {arr.shape}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"SAM3 semantic mask {path} must have integer dtype, got {arr.dtype}")
    return np.asarray(arr)


def load_confidence_mask(path: Path, *, expected_shape: tuple[int, int]) -> np.ndarray | None:
    if not path.is_file():
        return None
    arr = np.load(path, allow_pickle=False)
    if arr.shape != expected_shape:
        raise ValueError(f"SAM3 confidence mask {path} has shape {arr.shape}, expected {expected_shape}")
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"SAM3 confidence mask {path} must be numeric, got {arr.dtype}")
    return np.asarray(arr, dtype=np.float32)


def lift_mask_to_points(
    *,
    semantic: np.ndarray,
    confidence: np.ndarray | None,
    cxd: np.ndarray,
    cyd: np.ndarray,
    known_ids: set[int],
    min_confidence: float,
) -> LiftedLabels:
    if cxd.shape != cyd.shape:
        raise ValueError(f"Cxd/Cyd shapes differ: {cxd.shape} vs {cyd.shape}")

    point_count = cxd.size
    labels = np.full(point_count, IGNORE_ID, dtype=np.uint32)
    point_confidence = np.zeros(point_count, dtype=np.float32)
    height, width = semantic.shape

    finite_projection = np.isfinite(cxd) & np.isfinite(cyd)
    u = np.full(point_count, -1, dtype=np.int64)
    v = np.full(point_count, -1, dtype=np.int64)
    u[finite_projection] = round_like_point_labeler(cxd[finite_projection])
    v[finite_projection] = round_like_point_labeler(cyd[finite_projection])
    inside = finite_projection & (u >= 0) & (v >= 0) & (u < width) & (v < height)

    stats: dict[str, Any] = {
        "points": int(point_count),
        "inside_image": int(np.count_nonzero(inside)),
        "invalid_projection": int(np.count_nonzero(~finite_projection)),
        "out_of_image": int(np.count_nonzero(finite_projection & ~inside)),
        "missing_confidence": 0,
        "low_confidence": 0,
        "unknown_class": 0,
        "ignored_class": 0,
        "accepted": 0,
        "ignored": 0,
        "class_point_counts": {},
    }

    if confidence is None:
        stats["missing_confidence"] = int(np.count_nonzero(inside))
        stats["ignored"] = int(point_count)
        return LiftedLabels(labels=labels, confidence=point_confidence, stats=stats)

    projected_labels = np.full(point_count, IGNORE_ID, dtype=np.int64)
    projected_confidence = np.zeros(point_count, dtype=np.float32)
    projected_labels[inside] = semantic[v[inside], u[inside]].astype(np.int64, copy=False)
    projected_confidence[inside] = confidence[v[inside], u[inside]].astype(np.float32, copy=False)
    point_confidence[inside] = projected_confidence[inside]

    finite_confidence = np.isfinite(projected_confidence)
    high_confidence = inside & finite_confidence & (projected_confidence >= min_confidence)
    known = np.isin(projected_labels, np.asarray(sorted(known_ids), dtype=np.int64))
    source_ignore = projected_labels == IGNORE_ID
    accepted = high_confidence & known & ~source_ignore
    labels[accepted] = projected_labels[accepted].astype(np.uint32, copy=False)

    stats["low_confidence"] = int(np.count_nonzero(inside & ~high_confidence))
    stats["unknown_class"] = int(np.count_nonzero(high_confidence & ~known))
    stats["ignored_class"] = int(np.count_nonzero(high_confidence & known & source_ignore))
    stats["accepted"] = int(np.count_nonzero(accepted))
    stats["ignored"] = int(point_count - stats["accepted"])
    unique_ids, counts = np.unique(labels[accepted], return_counts=True)
    stats["class_point_counts"] = {str(int(label_id)): int(count) for label_id, count in zip(unique_ids, counts)}
    return LiftedLabels(labels=labels, confidence=point_confidence, stats=stats)


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
    v = np.full(cxd.size, -1, dtype=np.int64)
    u[finite_projection] = round_like_point_labeler(cxd[finite_projection])
    v[finite_projection] = round_like_point_labeler(cyd[finite_projection])
    inside = finite_projection & (u >= 0) & (v >= 0) & (u < width) & (v < height)
    rgb[inside] = image[v[inside], u[inside]]
    return rgb


def round_like_point_labeler(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    rounded = np.where(arr >= 0.0, np.floor(arr + 0.5), np.ceil(arr - 0.5))
    return rounded.astype(np.int64)


def write_labels_xml(path: Path, classes: list[tuple[str, int]]) -> None:
    root = ET.Element("config")
    for name, label_id in classes:
        label = ET.SubElement(root, "label")
        ET.SubElement(label, "id").text = str(label_id)
        ET.SubElement(label, "name").text = name
        ET.SubElement(label, "description").text = ""
        ET.SubElement(label, "color").text = " ".join(str(value) for value in label_color(name, label_id))
        ET.SubElement(label, "root").text = "manual"
        ET.SubElement(label, "macro").text = ""
        ET.SubElement(label, "category").text = name
    indent_xml(root)
    tree = ET.ElementTree(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def label_color(name: str, label_id: int) -> tuple[int, int, int]:
    if label_id == IGNORE_ID or name.casefold() in {"ignore", "ignored"}:
        return (120, 120, 120)
    if label_id == 0:
        return (32, 32, 32)
    seed = (label_id * 1009 + sum(ord(char) for char in name) * 17) & 0xFFFFFFFF
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
    path.write_text(identity * frame_count, encoding="utf-8")


def frame_extent(points: np.ndarray) -> tuple[float, float]:
    xyz = np.asarray(points[:, :3], dtype=np.float64)
    finite = np.all(np.isfinite(xyz), axis=1)
    if not np.any(finite):
        return 0.0, 0.0
    xyz = xyz[finite]
    return float(np.max(np.abs(xyz[:, :2]))), float(np.max(np.linalg.norm(xyz, axis=1)))


def settings_content(out_dir: Path, *, max_abs_xy: float, max_range: float) -> str:
    labels_xml = out_dir / "labels.xml"
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
            "csv image width: 1920",
            "csv image height: 1536",
            f"tile size: {tile_size:.3f}",
            "max scans: 500",
            "min range: 0.0",
            f"max range: {max_range_value:.3f}",
            "",
        ]
    )


def init_total_stats() -> dict[str, Any]:
    return {
        "points": 0,
        "inside_image": 0,
        "invalid_projection": 0,
        "out_of_image": 0,
        "missing_confidence": 0,
        "low_confidence": 0,
        "unknown_class": 0,
        "ignored_class": 0,
        "accepted": 0,
        "ignored": 0,
        "class_point_counts": {},
    }


def accumulate_stats(total: dict[str, Any], frame: dict[str, Any]) -> None:
    for key in (
        "points",
        "inside_image",
        "invalid_projection",
        "out_of_image",
        "missing_confidence",
        "low_confidence",
        "unknown_class",
        "ignored_class",
        "accepted",
        "ignored",
    ):
        total[key] += int(frame.get(key, 0))
    class_counts = total["class_point_counts"]
    for label_id, count in frame.get("class_point_counts", {}).items():
        class_counts[str(label_id)] = int(class_counts.get(str(label_id), 0)) + int(count)


if __name__ == "__main__":
    main()
