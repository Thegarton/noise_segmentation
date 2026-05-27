#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from autolabeler.data.class_config import load_semantic_classes  # noqa: E402
from autolabeler.export.semantic_mask_exporter import export_semantic_segmentation_result  # noqa: E402
from autolabeler.data.schemas import SemanticSegmentationResult  # noqa: E402
from autolabeler.fusion.pointwise_teacher_fusion import (  # noqa: E402
    class_names_to_project_ids,
    fuse_litept_and_sam3,
    labels_to_ids_from_prompt_config,
    lift_sam3_candidates_to_points,
    load_name_mapping,
    read_class_names_from_metadata,
)
from autolabeler.teachers.sam3_text_adapter import load_prompt_config, load_sam3_text_result  # noqa: E402


def main() -> None:
    args = parse_args()
    litept_dir = Path(args.litept_output_dir)
    sam3_dir = Path(args.sam3_dir)
    projection_dir = Path(args.projection_dir)
    out_dir = Path(args.out_dir)

    project_classes = load_semantic_classes(args.classes_yaml)
    source_to_target = load_name_mapping(args.litept_mapping_yaml)
    prompt_config = load_prompt_config(args.prompt_config)
    sam3_label_to_id = labels_to_ids_from_prompt_config(args.classes_yaml, prompt_config)

    frame_dirs = sorted([p for p in litept_dir.iterdir() if p.is_dir()])
    exported = []
    for frame_dir in frame_dirs:
        frame_id = frame_dir.name
        metadata_path = frame_dir / "metadata.json"
        litept_mask = np.load(frame_dir / "semantic_mask.npy", allow_pickle=False)
        confidence_path = frame_dir / "confidence.npy"
        litept_confidence = np.load(confidence_path, allow_pickle=False) if confidence_path.exists() else np.ones_like(litept_mask, dtype=np.float32)
        class_names = read_class_names_from_metadata(metadata_path)
        litept_id_to_project_id = class_names_to_project_ids(
            class_names,
            source_to_target=source_to_target,
            project_classes=project_classes,
        )

        point_to_pixel = np.load(projection_dir / frame_id / "point_to_pixel.npy", allow_pickle=False)
        sam3 = load_sam3_text_result(sam3_dir / f"{frame_id}.npz")
        sam3_candidate, sam3_candidate_conf = lift_sam3_candidates_to_points(
            sam3,
            point_to_pixel=point_to_pixel,
            label_to_id=sam3_label_to_id,
            min_points_per_mask=args.min_sam3_points,
            min_score=args.min_sam3_score,
        )
        fused = fuse_litept_and_sam3(
            litept_mask=litept_mask,
            litept_confidence=litept_confidence,
            litept_id_to_project_id=litept_id_to_project_id,
            sam3_candidate_mask=sam3_candidate,
            sam3_candidate_confidence=sam3_candidate_conf,
            litept_accept_threshold=args.litept_accept_threshold,
            sam3_accept_threshold=args.sam3_accept_threshold,
        )

        result = SemanticSegmentationResult(
            frame_id=frame_id,
            semantic_mask=fused.semantic_mask,
            confidence_mask=fused.confidence_mask,
            pseudo_label_version="pointwise_teacher_v1",
            provenance="ensemble_agreed",
            metadata={
                **fused.metadata,
                "class_names": class_names_from_mapping(project_classes),
                "semantic_classes": project_classes,
                "raw_litept": {
                    "metadata": str(metadata_path),
                    "semantic_mask": str(frame_dir / "semantic_mask.npy"),
                    "confidence_mask": str(confidence_path) if confidence_path.exists() else None,
                },
                "sam3_text": {
                    "npz": str(sam3_dir / f"{frame_id}.npz"),
                    "prompt_config": str(args.prompt_config),
                },
                "projection": {
                    "point_to_pixel": str(projection_dir / frame_id / "point_to_pixel.npy"),
                },
                "manual_reviewed": False,
            },
        )
        metadata = export_semantic_segmentation_result(str(out_dir), result)
        frame_out = out_dir / frame_id
        np.save(frame_out / "sam3_candidate_mask.npy", fused.sam3_candidate_mask)
        metadata_path_out = frame_out / "metadata.json"
        payload = json.loads(metadata_path_out.read_text(encoding="utf-8"))
        payload["sam3_text"]["candidate_mask"] = str(frame_out / "sam3_candidate_mask.npy")
        metadata_path_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        exported.append(metadata)

    print(json.dumps({"frames": len(exported), "out_dir": str(out_dir)}, indent=2))


def class_names_from_mapping(mapping: dict[str, int]) -> list[str]:
    max_id = max(class_id for class_id in mapping.values() if class_id != 255)
    names = [""] * (max_id + 1)
    for name, class_id in mapping.items():
        if class_id == 255:
            continue
        names[class_id] = name
    return names


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fuse Waymo LitePT masks and SAM3 text-prompt camera candidates.")
    p.add_argument("--litept-output-dir", required=True)
    p.add_argument("--sam3-dir", required=True)
    p.add_argument("--projection-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--classes-yaml", default=str(REPO_ROOT / "configs" / "classes_pointwise_v1.yaml"))
    p.add_argument("--litept-mapping-yaml", default=str(REPO_ROOT / "configs" / "litept_waymo_to_pointwise_v1.yaml"))
    p.add_argument("--prompt-config", default=str(REPO_ROOT / "configs" / "sam3_text_prompts_pointwise_v1.yaml"))
    p.add_argument("--litept-accept-threshold", type=float, default=0.65)
    p.add_argument("--sam3-accept-threshold", type=float, default=0.70)
    p.add_argument("--min-sam3-score", type=float, default=0.70)
    p.add_argument("--min-sam3-points", type=int, default=8)
    return p.parse_args()


if __name__ == "__main__":
    main()

