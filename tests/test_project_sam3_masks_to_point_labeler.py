from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def test_project_sam3_masks_to_point_labeler_preserves_csv_row_order(tmp_path: Path):
    csv_dir = tmp_path / "csv_1090_1245"
    sam3_dir = tmp_path / "sam3"
    image_dir = tmp_path / "img2"
    out_dir = tmp_path / "labeler"
    csv_dir.mkdir()
    (sam3_dir / "000001").mkdir(parents=True)
    image_dir.mkdir()
    classes_yaml = tmp_path / "classes.yaml"
    classes_yaml.write_text(
        "semantic_classes:\n"
        "  background: 0\n"
        "  CAR: 2\n"
        "  thin_vertical_bar: 9\n"
        "  ignore: 255\n",
        encoding="utf-8",
    )
    (csv_dir / "000001.csv").write_text(
        "\t".join(["x", "y", "z", "azimuth", "vertical", "intensity", "slot", "pixel", "hcell", "vcell", "Cxd", "Cyd"])
        + "\n"
        + "\n".join(
            [
                "1 0 0 0 0 10 2 0 0 0 0 0",
                "2 0 0 0 0 20 2 1 0 0 0 1",
                "3 0 0 0 0 30 2 2 0 0 1 0",
                "4 0 0 0 0 40 2 3 0 0 1 1",
                "5 0 0 0 0 50 2 4 0 0 5 5",
                "6 0 0 0 0 60 2 5 0 0 nan 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    semantic = np.asarray([[2, 42], [9, 2]], dtype=np.uint16)
    confidence = np.asarray([[0.9, 0.95], [0.8, 0.6]], dtype=np.float32)
    np.save(sam3_dir / "000001" / "semantic_mask.npy", semantic)
    np.save(sam3_dir / "000001" / "confidence.npy", confidence)
    image = np.asarray(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )
    Image.fromarray(image).save(image_dir / "000001.jpg")

    run_script(
        "--csv-dir",
        str(csv_dir),
        "--sam3-dir",
        str(sam3_dir),
        "--out-labeler-dir",
        str(out_dir),
        "--classes-yaml",
        str(classes_yaml),
        "--image-dir",
        str(image_dir),
        "--min-confidence",
        "0.7",
    )

    points = np.fromfile(out_dir / "velodyne" / "000001.bin", dtype=np.float32).reshape(-1, 4)
    labels = np.fromfile(out_dir / "labels" / "000001.label", dtype=np.uint32)
    point_rgb = np.fromfile(out_dir / "point_rgb" / "000001.rgb", dtype=np.uint8).reshape(-1, 3)
    np.testing.assert_allclose(points[:, 0], np.asarray([1, 2, 3, 4, 5, 6], dtype=np.float32))
    np.testing.assert_allclose(points[:, 3], np.asarray([10, 20, 30, 40, 50, 60], dtype=np.float32))
    assert labels.tolist() == [2, 9, 255, 255, 255, 255]
    assert point_rgb.tolist() == [
        [255, 0, 0],
        [0, 0, 255],
        [0, 255, 0],
        [255, 255, 255],
        [0, 0, 0],
        [0, 0, 0],
    ]
    assert (out_dir / "image_2" / "000001.jpg").is_file()
    settings = (out_dir / "settings.cfg").read_text(encoding="utf-8")
    assert "allow velodyne only: true" in settings
    assert "point cloud source: velodyne" in settings

    manifest = json.loads((out_dir / "bridge_manifest.json").read_text(encoding="utf-8"))
    frame = manifest["frames"][0]
    assert manifest["label_layout"] == "flat_points"
    assert frame["frame_id"] == "000001"
    assert frame["point_count"] == 6
    assert frame["projection_stats"]["accepted"] == 2
    assert frame["projection_stats"]["unknown_class"] == 1
    assert frame["projection_stats"]["low_confidence"] == 1
    assert frame["projection_stats"]["out_of_image"] == 1
    assert frame["projection_stats"]["invalid_projection"] == 1
    assert frame["projection_stats"]["class_point_counts"] == {"2": 1, "9": 1}


def test_missing_confidence_ignores_inside_points():
    script = load_script()
    semantic = np.asarray([[2]], dtype=np.uint16)

    lifted = script.lift_mask_to_points(
        semantic=semantic,
        confidence=None,
        cxd=np.asarray([0.0]),
        cyd=np.asarray([0.0]),
        known_ids={0, 2, 255},
        min_confidence=0.7,
    )

    assert lifted.labels.tolist() == [255]
    assert lifted.stats["missing_confidence"] == 1
    assert lifted.stats["accepted"] == 0
    assert lifted.stats["ignored"] == 1


def test_projection_rounding_matches_point_labeler():
    script = load_script()

    rounded = script.round_like_point_labeler(np.asarray([0.49, 0.5, 1.5, -0.5, -1.5]))

    assert rounded.tolist() == [0, 1, 2, -1, -2]


def run_script(*args: str) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "project_sam3_masks_to_point_labeler.py"
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    subprocess.run([sys.executable, str(script), *args], check=True, env=env, capture_output=True, text=True)


def load_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "project_sam3_masks_to_point_labeler.py"
    spec = importlib.util.spec_from_file_location("project_sam3_masks_to_point_labeler", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
