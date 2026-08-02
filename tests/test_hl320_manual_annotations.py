from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from autolabeler.hl320.manual_annotations import build_manual_label_array


def test_manual_annotation_indices_are_csv_row_indices(tmp_path: Path):
    annotation = tmp_path / "000001.json"
    annotation.write_text(
        json.dumps(
            [
                {"objectType": "车", "indices": [0, 2]},
                {"objectType": "串扰", "indices": [1]},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    labels = build_manual_label_array(
        annotation_path=annotation,
        frame_id="000001",
        point_count=4,
        class_to_id={"CAR": 2, "crosstalk_noise_1": 15, "ignore": 255},
    )

    assert labels.labels.tolist() == [2, 15, 2, 255]
    assert labels.stats["class_counts"]["2"]["count"] == 2
    assert labels.stats["class_counts"]["15"]["count"] == 1


def test_manual_json_import_writes_labels_and_point_labeler_dataset(tmp_path: Path):
    csv_dir = tmp_path / "csv"
    ann_dir = tmp_path / "annotations"
    out_dir = tmp_path / "out"
    csv_dir.mkdir()
    ann_dir.mkdir()
    classes_yaml = tmp_path / "classes.yaml"
    classes_yaml.write_text(
        "semantic_classes:\n"
        "  background: 0\n"
        "  CAR: 2\n"
        "  crosstalk_noise_1: 15\n"
        "  ignore: 255\n",
        encoding="utf-8",
    )
    (csv_dir / "000001.csv").write_text(
        "x y z intensity slot pixel blockID Cxd Cyd\n"
        "1 0 0 10 2 0 0 10 20\n"
        "2 0 0 20 2 0 1 10 20\n"
        "3 0 0 30 2 1 0 11 21\n"
        "4 0 0 40 2 1 1 11 21\n",
        encoding="utf-8",
    )
    (ann_dir / "000001.json").write_text(
        json.dumps([{"objectType": "车", "indices": [0, 2]}, {"objectType": "串扰", "indices": [3]}], ensure_ascii=False),
        encoding="utf-8",
    )

    script = Path(__file__).resolve().parents[1] / "scripts" / "import_hl320_manual_json_labels.py"
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--csv-dir",
            str(csv_dir),
            "--annotation-dir",
            str(ann_dir),
            "--out-dir",
            str(out_dir),
            "--classes-yaml",
            str(classes_yaml),
        ],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )

    labels = np.load(out_dir / "manual_labels" / "000001" / "semantic_mask.npy")
    primary = np.load(out_dir / "manual_labels" / "000001" / "primary_semantic_mask.npy")
    point_labeler_labels = np.fromfile(out_dir / "point_labeler" / "labels" / "000001.label", dtype=np.uint32)
    point_labeler_points = np.fromfile(out_dir / "point_labeler" / "velodyne" / "000001.bin", dtype=np.float32).reshape(-1, 4)

    assert labels.tolist() == [2, 255, 2, 15]
    assert primary.tolist() == [2, 2]
    assert point_labeler_labels.tolist() == [2, 255, 2, 15]
    assert point_labeler_points.shape == (4, 4)
    assert (out_dir / "point_labeler" / "bridge_manifest.json").is_file()
