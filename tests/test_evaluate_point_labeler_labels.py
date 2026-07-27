from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def test_evaluate_point_labeler_label_dirs_computes_iou_and_ignore(tmp_path: Path):
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    (gt_dir / "labels").mkdir(parents=True)
    (pred_dir / "labels").mkdir(parents=True)
    write_labels_xml(gt_dir / "labels.xml")

    np.asarray([2, 2, 5, 255], dtype=np.uint32).tofile(gt_dir / "labels" / "000001.label")
    np.asarray([2, 5, 5, 2], dtype=np.uint32).tofile(pred_dir / "labels" / "000001.label")

    script = load_script()
    result = script.evaluate_label_dirs(
        pred_dir=pred_dir,
        gt_dir=gt_dir,
        ignore_ids={255},
        id_to_name={2: "CAR", 5: "PEDESTRIAN", 255: "ignore"},
        class_ids=None,
        strict=True,
        max_frames=None,
    )

    assert result["overall"]["valid_points"] == 3
    assert result["overall"]["correct_points"] == 2
    assert result["overall"]["accuracy"] == 2 / 3
    assert result["overall"]["mean_iou"] == 0.5
    by_id = {row["id"]: row for row in result["per_class"]}
    assert by_id[2]["name"] == "CAR"
    assert by_id[2]["support"] == 2
    assert by_id[2]["predicted"] == 1
    assert by_id[2]["iou"] == 0.5
    assert by_id[5]["precision"] == 0.5
    assert by_id[5]["recall"] == 1.0


def test_evaluate_export_masks_cli_writes_outputs(tmp_path: Path):
    gt_dir = tmp_path / "gt_export"
    pred_dir = tmp_path / "pred_export"
    out_json = tmp_path / "metrics" / "summary.json"
    out_csv = tmp_path / "metrics" / "per_class.csv"
    frame_csv = tmp_path / "metrics" / "frames.csv"
    confusion_csv = tmp_path / "metrics" / "confusion.csv"
    (gt_dir / "000001").mkdir(parents=True)
    (pred_dir / "000001").mkdir(parents=True)

    np.save(gt_dir / "000001" / "semantic_mask.npy", np.asarray([[2, 2], [5, 255]], dtype=np.uint16))
    np.save(pred_dir / "000001" / "semantic_mask.npy", np.asarray([[2, 5], [5, 2]], dtype=np.uint16))
    classes_yaml = tmp_path / "classes.yaml"
    classes_yaml.write_text(
        "semantic_classes:\n"
        "  background: 0\n"
        "  CAR: 2\n"
        "  PEDESTRIAN: 5\n"
        "  ignore: 255\n",
        encoding="utf-8",
    )

    script = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_point_labeler_labels.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--pred-dir",
            str(pred_dir),
            "--gt-dir",
            str(gt_dir),
            "--classes-yaml",
            str(classes_yaml),
            "--out-json",
            str(out_json),
            "--out-csv",
            str(out_csv),
            "--frame-csv",
            str(frame_csv),
            "--confusion-csv",
            str(confusion_csv),
        ],
        check=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
        capture_output=True,
        text=True,
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert "Accuracy: 0.666667" in completed.stdout
    assert payload["frame_count"] == 1
    assert payload["overall"]["mean_iou"] == 0.5
    assert out_csv.read_text(encoding="utf-8").splitlines()[0].startswith("id,name,support")
    assert frame_csv.read_text(encoding="utf-8").splitlines()[1].startswith("000001,4,3,1,2,")
    assert "gt\\pred" in confusion_csv.read_text(encoding="utf-8")


def test_evaluate_raises_on_shape_mismatch(tmp_path: Path):
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    (gt_dir / "labels").mkdir(parents=True)
    (pred_dir / "labels").mkdir(parents=True)
    np.asarray([2, 2], dtype=np.uint32).tofile(gt_dir / "labels" / "000001.label")
    np.asarray([2], dtype=np.uint32).tofile(pred_dir / "labels" / "000001.label")

    script = load_script()
    try:
        script.evaluate_label_dirs(
            pred_dir=pred_dir,
            gt_dir=gt_dir,
            ignore_ids={255},
            id_to_name={2: "CAR"},
            class_ids=None,
            strict=True,
            max_frames=None,
        )
    except ValueError as exc:
        assert "shape mismatch" in str(exc)
    else:
        raise AssertionError("shape mismatch did not raise")


def write_labels_xml(path: Path) -> None:
    path.write_text(
        "<?xml version='1.0' encoding='utf-8'?>\n"
        "<config>\n"
        "  <label><id>2</id><name>CAR</name><color>1 2 3</color></label>\n"
        "  <label><id>5</id><name>PEDESTRIAN</name><color>4 5 6</color></label>\n"
        "  <label><id>255</id><name>ignore</name><color>7 8 9</color></label>\n"
        "</config>\n",
        encoding="utf-8",
    )


def load_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_point_labeler_labels.py"
    spec = importlib.util.spec_from_file_location("evaluate_point_labeler_labels", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
