from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..data.ins_pose import write_kitti_pose_values
from ..data.schemas import SemanticSegmentationResult
from ..data.semantic_mask import save_semantic_mask, validate_semantic_mask


def export_semantic_segmentation_result(output_dir: str, result: SemanticSegmentationResult) -> dict[str, str]:
    out_dir = Path(output_dir) / result.frame_id
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_path = out_dir / "semantic_mask.npy"
    save_semantic_mask(str(mask_path), result.semantic_mask)

    metadata = {
        "frame_id": result.frame_id,
        "semantic_mask": str(mask_path),
        "confidence_mask": None,
        "pose": None,
        "pseudo_label_version": result.pseudo_label_version,
        "provenance": result.provenance,
    }
    metadata.update(result.metadata)

    if result.confidence_mask is not None:
        confidence = np.asarray(result.confidence_mask, dtype=np.float32)
        validate_semantic_mask(result.semantic_mask)
        if confidence.shape != result.semantic_mask.shape:
            raise ValueError(f"Confidence mask shape must match semantic mask, got {confidence.shape}")
        confidence_path = out_dir / "confidence.npy"
        np.save(confidence_path, confidence)
        metadata["confidence_mask"] = str(confidence_path)

    if metadata.get("ego_pose") is not None:
        pose_path = out_dir / "pose.txt"
        write_kitti_pose_values(str(pose_path), metadata["ego_pose"]["kitti_pose"])
        metadata["pose"] = str(pose_path)

    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata["metadata"] = str(metadata_path)
    return metadata
