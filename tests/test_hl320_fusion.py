from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from autolabeler.hl320.fusion import (
    PROVENANCE_CODES,
    FusionPolicy,
    PointPrediction,
    fuse_point_predictions,
    lift_sam3_to_points,
)


def test_fuse_point_predictions_protects_noise_and_fills_weak_litept():
    policy = FusionPolicy(
        min_sam3_confidence=0.7,
        litept_keep_threshold=0.6,
        noise_protect_threshold=0.4,
        sam3_override_margin=0.2,
    )
    litept = PointPrediction(
        labels=np.asarray([20, 255, 2, 2], dtype=np.int32),
        confidence=np.asarray([0.8, 0.0, 0.2, 0.9], dtype=np.float32),
    )
    sam3 = PointPrediction(
        labels=np.asarray([2, 5, 5, 5], dtype=np.int32),
        confidence=np.asarray([0.95, 0.8, 0.8, 0.95], dtype=np.float32),
    )

    result = fuse_point_predictions(litept=litept, sam3=sam3, noise_ids={20}, policy=policy)

    assert result.labels.tolist() == [20, 5, 5, 2]
    assert result.provenance.tolist() == [
        PROVENANCE_CODES["noise_protected"],
        PROVENANCE_CODES["sam3_fill"],
        PROVENANCE_CODES["sam3_fill"],
        PROVENANCE_CODES["litept"],
    ]
    assert result.stats["noise_protected_points"] == 1
    assert result.stats["provenance_counts"]["sam3_fill"] == 2


def test_lift_sam3_to_points_uses_cxd_cyd_rounding_and_filters_invalid(tmp_path: Path):
    frame_dir = tmp_path / "000000"
    frame_dir.mkdir()
    np.save(frame_dir / "semantic_mask.npy", np.asarray([[2, 42], [5, 0]], dtype=np.int32))
    np.save(frame_dir / "confidence.npy", np.asarray([[0.9, 0.9], [0.8, 0.95]], dtype=np.float32))

    lifted = lift_sam3_to_points(
        sam3_frame_dir=frame_dir,
        cxd=np.asarray([0.49, 0.1, 1.0, 1.0, 5.0, np.nan], dtype=np.float32),
        cyd=np.asarray([0.49, 0.6, 0.0, 1.0, 5.0, 0.0], dtype=np.float32),
        known_ids={0, 2, 5, 255},
        policy=FusionPolicy(min_sam3_confidence=0.7),
    )

    assert lifted.labels.tolist() == [2, 5, 255, 255, 255, 255]
    np.testing.assert_allclose(lifted.confidence[:4], np.asarray([0.9, 0.8, 0.9, 0.95], dtype=np.float32))


def test_fuse_hl320_point_predictions_cli_smoke(tmp_path: Path):
    csv_dir = tmp_path / "csv"
    litept_dir = tmp_path / "litept"
    sam3_dir = tmp_path / "sam3"
    out_dir = tmp_path / "fused"
    csv_dir.mkdir()
    (litept_dir / "000000").mkdir(parents=True)
    (sam3_dir / "000000").mkdir(parents=True)
    classes_yaml = tmp_path / "classes.yaml"
    classes_yaml.write_text(
        "semantic_classes:\n"
        "  background: 0\n"
        "  CAR: 2\n"
        "  PEDESTRIAN: 5\n"
        "  dust_noise: 20\n"
        "  ignore: 255\n"
        "noise_groups:\n"
        "  dust_noise:\n"
        "    - dust_noise\n",
        encoding="utf-8",
    )
    (csv_dir / "000000.csv").write_text(
        "x y z intensity slot pixel Cxd Cyd\n"
        "1 0 0 4 2 0 0 0\n"
        "2 0 0 5 2 1 1 0\n"
        "3 0 0 6 2 2 0 1\n",
        encoding="utf-8",
    )
    np.save(litept_dir / "000000" / "semantic_mask.npy", np.asarray([20, 255, 2], dtype=np.int32))
    np.save(litept_dir / "000000" / "confidence.npy", np.asarray([0.8, 0.0, 0.1], dtype=np.float32))
    np.save(sam3_dir / "000000" / "semantic_mask.npy", np.asarray([[2, 5], [5, 0]], dtype=np.int32))
    np.save(sam3_dir / "000000" / "confidence.npy", np.asarray([[0.95, 0.9], [0.8, 0.0]], dtype=np.float32))

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "fuse_hl320_point_predictions.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--csv-dir",
            str(csv_dir),
            "--litept-dir",
            str(litept_dir),
            "--sam3-dir",
            str(sam3_dir),
            "--classes-yaml",
            str(classes_yaml),
            "--out-dir",
            str(out_dir),
            "--overwrite",
        ],
        check=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["frame_count"] == 1
    assert (out_dir / "hl320_fusion_manifest.json").is_file()
    assert np.load(out_dir / "000000" / "semantic_mask.npy").tolist() == [20, 5, 5]
    assert np.load(out_dir / "000000" / "provenance.npy").tolist() == [
        PROVENANCE_CODES["noise_protected"],
        PROVENANCE_CODES["sam3_fill"],
        PROVENANCE_CODES["sam3_fill"],
    ]
