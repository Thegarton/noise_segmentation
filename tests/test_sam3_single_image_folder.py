from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from PIL import Image


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


def test_projection_is_matched_by_image_stem(tmp_path: Path):
    script = load_script()
    image_path = tmp_path / "images" / "000001.jpg"
    projection_dir = tmp_path / "projection"
    image_path.parent.mkdir()
    projection_dir.mkdir()
    image_path.write_bytes(b"")
    expected = projection_dir / "000001.png"
    expected.write_bytes(b"")

    assert script.find_projection_for_image(projection_dir, image_path) == expected


def test_projection_strips_original_suffix_when_matching(tmp_path: Path):
    script = load_script()
    image_path = tmp_path / "images" / "000001_original.jpg"
    projection_dir = tmp_path / "projection"
    image_path.parent.mkdir()
    projection_dir.mkdir()
    image_path.write_bytes(b"")
    expected = projection_dir / "000001.jpg"
    expected.write_bytes(b"")

    assert script.projection_stems_for_image("000001_original", stem_suffixes=("_original",)) == [
        "000001_original",
        "000001",
    ]
    assert script.find_projection_for_image(projection_dir, image_path) == expected


def test_save_mask_projection_preview(tmp_path: Path):
    script = load_script()
    image_path = tmp_path / "images" / "000000.jpg"
    projection_dir = tmp_path / "projection"
    output_dir = tmp_path / "out"
    image_path.parent.mkdir()
    projection_dir.mkdir()
    output_dir.mkdir()
    Image.fromarray(np.full((2, 3, 3), 10, dtype=np.uint8)).save(image_path)
    Image.fromarray(np.full((2, 3, 3), 200, dtype=np.uint8)).save(projection_dir / "000000.jpg")

    info = script.save_mask_projection_preview(
        image_path=image_path,
        output_dir=output_dir,
        projection_dir=projection_dir,
        image_np=np.full((2, 3, 3), 10, dtype=np.uint8),
        semantic_color=np.full((2, 3, 3), 30, dtype=np.uint8),
        overlay=np.full((2, 3, 3), 50, dtype=np.uint8),
        require_projection=True,
    )

    assert Path(info["projection_path"]).name == "000000.jpg"
    assert Path(info["projection_copy"]).is_file()
    assert Path(info["mask_projection"]).is_file()
    assert info["projection_error"] is None


def test_outputs_exist_requires_projection_outputs_when_enabled(tmp_path: Path):
    script = load_script()
    frame_out = tmp_path / "frame"
    frame_out.mkdir()
    for filename in ["semantic_mask.npy", "confidence.npy", "instances.npz", "overlay.jpg", "metadata.json"]:
        (frame_out / filename).write_bytes(b"")

    assert script.outputs_exist(frame_out)
    assert not script.outputs_exist(frame_out, projection_enabled=True)

    (frame_out / "projection.jpg").write_bytes(b"")
    (frame_out / "mask_projection.jpg").write_bytes(b"")
    assert script.outputs_exist(frame_out, projection_enabled=True)


def load_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_sam3_single_image_folder.py"
    spec = importlib.util.spec_from_file_location("run_sam3_single_image_folder", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
