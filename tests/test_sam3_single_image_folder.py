from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace
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
        min_mask_size=1,
    )

    assert semantic.tolist() == [[2, 1], [0, 0]]
    assert confidence[0, 0] == np.float32(0.8)
    assert confidence[0, 1] == np.float32(0.9)
    assert counts == {"TRUCK_BUS": 1, "CAR": 1}


def test_priority_mask_overrides_higher_score_normal_mask():
    script = load_script()
    mask = np.ones((2, 2), dtype=bool)
    instances = [
        script.Sam3Instance(label="CAR", class_id=2, prompt="car", score=0.95, mask=mask),
        script.Sam3Instance(
            label="ground_markings",
            class_id=7,
            prompt="road marking",
            score=0.60,
            mask=mask,
        ),
    ]

    semantic, confidence, counts = script.build_semantic_outputs(
        instances=instances,
        label_to_id={"CAR": 2, "ground_markings": 7},
        shape=(2, 2),
        min_mask_size=1,
    )

    assert np.all(semantic == 7)
    assert np.all(confidence == np.float32(0.60))
    assert counts == {"ground_markings": 4}


def test_highest_score_wins_between_priority_masks():
    script = load_script()
    mask = np.ones((2, 2), dtype=bool)
    instances = [
        script.Sam3Instance(
            label="ground_markings",
            class_id=7,
            prompt="marking",
            score=0.55,
            mask=mask,
        ),
        script.Sam3Instance(
            label="license_plate&taillights",
            class_id=8,
            prompt="plate",
            score=0.80,
            mask=mask,
        ),
    ]

    semantic, _, _ = script.build_semantic_outputs(
        instances=instances,
        label_to_id={"ground_markings": 7, "license_plate&taillights": 8},
        shape=(2, 2),
        min_mask_size=1,
    )

    assert np.all(semantic == 8)


def test_canonical_license_plate_label_is_also_priority():
    script = load_script()
    mask = np.ones((2, 2), dtype=bool)
    semantic, _, _ = script.build_semantic_outputs(
        instances=[
            script.Sam3Instance(label="front_of_vehicle", class_id=9, prompt="vehicle", score=0.99, mask=mask),
            script.Sam3Instance(
                label="license_plate_and_taillights",
                class_id=18,
                prompt="license plate",
                score=0.60,
                mask=mask,
            ),
        ],
        label_to_id={"front_of_vehicle": 9, "license_plate_and_taillights": 18},
        shape=(2, 2),
        min_mask_size=1,
    )

    assert np.all(semantic == 18)


def test_vehicle_orientation_deduplicates_and_routes_instances():
    script = load_script()
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    instances = [
        script.Sam3Instance(label="vehicle", class_id=9, prompt="car", score=0.95, mask=mask),
        script.Sam3Instance(label="vehicle", class_id=9, prompt="vehicle", score=0.80, mask=mask.copy()),
        script.Sam3Instance(
            label="ground_markings",
            class_id=25,
            prompt="ground markings",
            score=0.70,
            mask=np.eye(4, dtype=bool),
        ),
    ]

    class FakeClassifier:
        def classify(self, image_rgb, masks, *, min_confidence, min_margin):
            assert len(masks) == 1
            return [
                SimpleNamespace(
                    predicted_class="rear",
                    semantic_label="rear",
                    confidence=0.91,
                    margin=0.70,
                    probabilities=(0.04, 0.91, 0.05),
                    fallback=False,
                    fallback_reason=None,
                )
            ]

    output = script.apply_vehicle_orientation(
        image_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        instances=instances,
        classifier=FakeClassifier(),
        vehicle_prompt_label="vehicle",
        class_mapping={"front": ("front_of_vehicle", 9), "rear": ("rear_of_vehicle", 10)},
        min_confidence=0.7,
        min_margin=0.1,
        nms_iou=0.8,
        min_mask_size=1,
    )

    oriented = [item for item in output if item.orientation_label is not None]
    assert len(oriented) == 1
    assert oriented[0].label == "rear_of_vehicle"
    assert oriented[0].class_id == 10
    assert oriented[0].source_label == "vehicle"
    assert oriented[0].orientation_probabilities == (0.04, 0.91, 0.05)
    assert any(item.label == "ground_markings" for item in output)


def test_other_orientation_fallback_is_recorded_as_front():
    script = load_script()
    mask = np.ones((2, 2), dtype=bool)

    class FakeClassifier:
        def classify(self, image_rgb, masks, *, min_confidence, min_margin):
            return [
                SimpleNamespace(
                    predicted_class="other",
                    semantic_label="front",
                    confidence=0.75,
                    margin=0.50,
                    probabilities=(0.10, 0.15, 0.75),
                    fallback=True,
                    fallback_reason="other",
                )
            ]

    output = script.apply_vehicle_orientation(
        image_rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        instances=[script.Sam3Instance(label="vehicle", class_id=9, prompt="vehicle", score=0.9, mask=mask)],
        classifier=FakeClassifier(),
        vehicle_prompt_label="vehicle",
        class_mapping={"front": ("front_of_vehicle", 9), "rear": ("rear_of_vehicle", 10)},
        min_confidence=0.7,
        min_margin=0.1,
        nms_iou=0.8,
        min_mask_size=1,
    )

    assert output[0].label == "front_of_vehicle"
    assert output[0].orientation_label == "other"
    assert output[0].orientation_fallback
    assert output[0].orientation_fallback_reason == "other"


def test_vehicle_class_mapping_validates_stable_ids():
    import pytest

    script = load_script()
    mapping = script.resolve_vehicle_class_mapping({"front_of_vehicle": 9, "rear_of_vehicle": 10})
    assert mapping == {"front": ("front_of_vehicle", 9), "rear": ("rear_of_vehicle", 10)}

    with pytest.raises(ValueError, match="stable taxonomy"):
        script.resolve_vehicle_class_mapping({"front_of_vehicle": 10, "rear_of_vehicle": 9})


def test_arrestor_geometry_uses_normalized_xywh_and_handles_missing_box():
    script = load_script()
    mask = np.ones((2, 2), dtype=bool)
    rejected_high = script.Sam3Instance(
        label="arrestor",
        class_id=4,
        prompt="arrestor",
        score=0.9,
        mask=mask,
        box=np.asarray([0.1, 0.2, 0.6, 0.2], dtype=np.float32),
    )
    rejected_narrow = script.Sam3Instance(
        label="arrestor",
        class_id=4,
        prompt="arrestor",
        score=0.9,
        mask=mask,
        box=np.asarray([0.1, 0.6, 0.1, 0.2], dtype=np.float32),
    )
    accepted = script.Sam3Instance(
        label="arrestor",
        class_id=4,
        prompt="arrestor",
        score=0.9,
        mask=mask,
        box=np.asarray([0.1, 0.6, 0.6, 0.2], dtype=np.float32),
    )

    assert script.reject_instance_by_geometry(rejected_high)
    assert script.reject_instance_by_geometry(rejected_narrow)
    assert script.reject_instance_by_geometry(
        script.Sam3Instance(label="arrestor", class_id=4, prompt="arrestor", score=0.9, mask=mask)
    )
    assert not script.reject_instance_by_geometry(accepted)


def test_arrestor_example_in_upper_image_is_rejected_by_bottom_edge():
    script = load_script()
    instance = script.Sam3Instance(
        label="arrestor",
        class_id=4,
        prompt="arrestor",
        score=0.9,
        mask=np.ones((2, 2), dtype=bool),
        box=np.asarray(
            [
                0.11432292312383652,
                0.23006536066532135,
                0.6955729722976685,
                0.1545751690864563,
            ],
            dtype=np.float32,
        ),
    )

    assert float(instance.box[1] + instance.box[3]) < script.ARRESTOR_MIN_Y_BOTTOM
    assert script.reject_instance_by_geometry(instance)


def test_make_overlay_can_draw_class_name_and_score():
    script = load_script()
    image = np.full((40, 60, 3), 220, dtype=np.uint8)
    semantic = np.zeros((40, 60), dtype=np.uint16)
    semantic[10:30, 15:45] = 2
    confidence = np.zeros((40, 60), dtype=np.float32)
    confidence[10:30, 15:45] = np.float32(0.87)
    mask = semantic == 2
    instance = script.Sam3Instance(label="CAR", class_id=2, prompt="car", score=0.87, mask=mask)

    plain = script.make_overlay(image, semantic)
    annotated = script.make_overlay(image, semantic, instances=[instance], confidence=confidence)

    assert annotated.shape == image.shape
    assert not np.array_equal(annotated, plain)


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


def test_load_label_min_scores_uses_exact_prompt_labels(tmp_path: Path):
    script = load_script()
    config = tmp_path / "scores.yaml"
    config.write_text("traffic_cone: 0.30\n'roadblock': 0.45\n", encoding="utf-8")

    scores = script.load_label_min_scores(
        config,
        known_labels={"traffic_cone", "roadblock", "CAR"},
    )

    assert scores == {"traffic_cone": 0.30, "roadblock": 0.45}


def test_load_label_min_scores_rejects_unknown_label(tmp_path: Path):
    import pytest

    script = load_script()
    config = tmp_path / "scores.yaml"
    config.write_text("traffic_cone_typo: 0.30\n", encoding="utf-8")

    with pytest.raises(ValueError, match="absent from the active"):
        script.load_label_min_scores(config, known_labels={"traffic_cone"})


def test_set_predictor_detection_threshold_changes_internal_image_threshold():
    script = load_script()
    predictor = SimpleNamespace(
        default_output_prob_thresh=0.5,
        model=SimpleNamespace(
            score_threshold_detection=0.4,
            image_only_det_thresh=0.5,
            new_det_thresh=0.65,
        ),
    )

    script.set_predictor_detection_threshold(predictor, 0.3)

    assert predictor.default_output_prob_thresh == 0.3
    assert predictor.model.score_threshold_detection == 0.3
    assert predictor.model.image_only_det_thresh == 0.3
    assert predictor.model.new_det_thresh == 0.3
    assert not hasattr(predictor.model, "image_only_det_threshold")


def load_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_sam3_single_image_folder.py"
    spec = importlib.util.spec_from_file_location("run_sam3_single_image_folder", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
