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


DEFAULT_IGNORE_ID = 255


def main() -> None:
    args = parse_args()
    export_dir = Path(args.export_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    labeler_dir = Path(args.labeler_dir).expanduser().resolve() if args.labeler_dir else None
    if not export_dir.is_dir():
        raise FileNotFoundError(f"Export dir does not exist: {export_dir}")
    if labeler_dir is not None and not labeler_dir.is_dir():
        raise FileNotFoundError(f"Labeler dir does not exist: {labeler_dir}")

    from_id = int(args.from_id)
    to_id = resolve_target_id(
        explicit_to_id=args.to_id,
        target_name=args.to_name,
        export_dir=export_dir,
        labeler_dir=labeler_dir,
    )
    if from_id == to_id:
        raise ValueError(f"Source and target ids are equal: {from_id}")

    prepare_output_dir(out_dir, overwrite=args.overwrite)
    frame_dirs = discover_frame_dirs(export_dir)
    if not frame_dirs:
        raise FileNotFoundError(f"No frame semantic_mask.npy files found in {export_dir}")

    summary: dict[str, Any] = {
        "version": 1,
        "source_export_dir": str(export_dir),
        "out_dir": str(out_dir),
        "labeler_dir": str(labeler_dir) if labeler_dir is not None else None,
        "from_id": from_id,
        "to_id": to_id,
        "to_name": args.to_name,
        "frames": [],
        "total_replaced": 0,
        "total_points": 0,
    }

    for frame_dir in frame_dirs:
        frame_id = frame_dir.name
        out_frame_dir = out_dir / frame_id
        shutil.copytree(frame_dir, out_frame_dir)
        mask_path = out_frame_dir / "semantic_mask.npy"
        mask = np.load(mask_path, allow_pickle=False)
        if not np.issubdtype(mask.dtype, np.integer):
            raise ValueError(f"Mask {frame_dir / 'semantic_mask.npy'} must have integer dtype, got {mask.dtype}")
        remapped = np.asarray(mask).copy()
        replace_mask = remapped == from_id
        replaced = int(np.count_nonzero(replace_mask))
        remapped[replace_mask] = np.asarray(to_id, dtype=remapped.dtype)
        np.save(mask_path, remapped.astype(mask.dtype, copy=False))

        metadata_path = out_frame_dir / "metadata.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        else:
            metadata = {}
        remaps = list(metadata.get("label_remaps") or [])
        remaps.append(
            {
                "from_id": from_id,
                "to_id": to_id,
                "to_name": args.to_name,
                "replaced_points": replaced,
                "source_export_dir": str(export_dir),
                "reason": args.reason,
            }
        )
        metadata["label_remaps"] = remaps
        metadata["semantic_mask"] = str(mask_path)
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        frame_summary = {
            "frame_id": frame_id,
            "mask_shape": [int(value) for value in remapped.shape],
            "points": int(remapped.size),
            "replaced_points": replaced,
            "source_mask": str(frame_dir / "semantic_mask.npy"),
            "out_mask": str(mask_path),
        }
        summary["frames"].append(frame_summary)
        summary["total_replaced"] += replaced
        summary["total_points"] += int(remapped.size)

    (out_dir / "remap_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy a point_labeler export and remap one semantic id to another, e.g. ignore 255 -> background 0."
    )
    parser.add_argument("--export-dir", required=True, help="Original export_from_point_labeler.py output.")
    parser.add_argument("--out-dir", required=True, help="New export directory to write.")
    parser.add_argument("--labeler-dir", default=None, help="Optional point_labeler dataset used to read labels.xml.")
    parser.add_argument("--from-id", type=int, default=DEFAULT_IGNORE_ID, help="Source semantic id to replace.")
    parser.add_argument("--to-id", type=int, default=None, help="Target semantic id. If omitted, --to-name is used.")
    parser.add_argument("--to-name", default="background", help="Target class name looked up in labels.xml/metadata.")
    parser.add_argument("--reason", default="convert_ignore_to_background_for_finetune")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output dir already exists: {path}. Use --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def discover_frame_dirs(export_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in export_dir.iterdir()
        if path.is_dir() and (path / "semantic_mask.npy").is_file()
    )


def resolve_target_id(
    *,
    explicit_to_id: int | None,
    target_name: str,
    export_dir: Path,
    labeler_dir: Path | None,
) -> int:
    if explicit_to_id is not None:
        return int(explicit_to_id)

    target_key = target_name.casefold()
    candidates: list[dict[str, int]] = []
    if labeler_dir is not None:
        labels_xml = labeler_dir / "labels.xml"
        if labels_xml.is_file():
            candidates.append(read_labels_xml(labels_xml))
    candidates.extend(read_semantic_classes_from_metadata(export_dir))

    for mapping in candidates:
        for name, label_id in mapping.items():
            if name.casefold() == target_key:
                return int(label_id)
    if target_key == "background":
        return 0
    raise ValueError(
        f"Cannot resolve target class {target_name!r}. Pass --to-id explicitly or provide --labeler-dir with labels.xml."
    )


def read_labels_xml(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    classes: dict[str, int] = {}
    for label in root.findall("label"):
        name = (label.findtext("name") or "").strip()
        raw_id = (label.findtext("id") or "").strip()
        if name and raw_id:
            classes[name] = int(raw_id)
    return classes


def read_semantic_classes_from_metadata(export_dir: Path) -> list[dict[str, int]]:
    mappings: list[dict[str, int]] = []
    for metadata_path in sorted(export_dir.glob("*/metadata.json"))[:20]:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        semantic_classes = metadata.get("semantic_classes")
        if isinstance(semantic_classes, dict):
            mappings.append({str(name): int(label_id) for name, label_id in semantic_classes.items()})
    return mappings


if __name__ == "__main__":
    main()
