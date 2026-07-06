from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def test_collect_images_non_recursive_sorted(tmp_path: Path):
    script = load_script()
    (tmp_path / "b.png").write_bytes(b"")
    (tmp_path / "a.jpg").write_bytes(b"")
    (tmp_path / "ignore.txt").write_bytes(b"")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "c.jpg").write_bytes(b"")

    images = script.collect_images(tmp_path, recursive=False)

    assert [path.name for path in images] == ["a.jpg", "b.png"]


def test_extract_arrays_accepts_sam3_31_output_keys():
    script = load_script()
    outputs = {
        "masks": None,
        "out_binary_masks": np.asarray([[[True, False], [False, True]]], dtype=bool),
        "out_probs": np.asarray([0.81], dtype=np.float32),
        "out_boxes_xywh": np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
    }

    masks, scores, boxes = script.extract_arrays(outputs, height=2, width=2)

    assert masks.shape == (1, 2, 2)
    np.testing.assert_allclose(scores, np.asarray([0.81], dtype=np.float32))
    np.testing.assert_allclose(boxes, np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32))


def test_build_semantic_outputs_keeps_highest_score():
    script = load_script()
    mask_a = np.zeros((2, 2), dtype=bool)
    mask_b = np.zeros((2, 2), dtype=bool)
    mask_a[0, 0] = True
    mask_a[0, 1] = True
    mask_b[0, 1] = True
    instances = [
        script.Sam3Instance(label="CAR", class_id=2, prompt="car", score=0.8, mask=mask_a),
        script.Sam3Instance(label="TRUCK_BUS", class_id=1, prompt="truck", score=0.9, mask=mask_b),
    ]

    semantic, confidence, counts = script.build_semantic_outputs(
        instances=instances,
        label_to_id={"TRUCK_BUS": 1, "CAR": 2},
        shape=(2, 2),
    )

    assert semantic.tolist() == [[2, 1], [0, 0]]
    assert confidence[0, 0] == np.float32(0.8)
    assert confidence[0, 1] == np.float32(0.9)
    assert counts == {"TRUCK_BUS": 1, "CAR": 1}


def load_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_sam3_single_image_folder.py"
    spec = importlib.util.spec_from_file_location("run_sam3_single_image_folder", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
