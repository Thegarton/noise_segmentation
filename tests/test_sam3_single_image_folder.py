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


def test_classes_from_semantic_mask_does_not_require_json_logs(tmp_path: Path):
    script = load_script()
    semantic = np.asarray([[0, 2, 2], [5, 5, 5]], dtype=np.uint16)
    path = tmp_path / "semantic_mask.npy"
    np.save(path, semantic)

    classes = script.classes_from_semantic_mask(
        path,
        label_to_id={"CAR": 2, "PEDESTRIAN": 5, "traffic_sign": 10},
        min_mask_size=2,
    )

    assert classes == {"CAR", "PEDESTRIAN"}


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


def test_overlay_instances_exclude_filtered_and_fully_covered_objects():
    script = load_script()
    mask = np.ones((2, 2), dtype=bool)
    instances = [
        script.Sam3Instance(label="CAR", class_id=2, prompt="car", score=0.90, mask=mask),
        script.Sam3Instance(label="TRUCK_BUS", class_id=1, prompt="truck", score=0.80, mask=mask),
        script.Sam3Instance(
            label="arrestor",
            class_id=4,
            prompt="arrestor too high",
            score=0.99,
            mask=mask,
            box=np.asarray([0.1, 0.1, 0.6, 0.2], dtype=np.float32),
        ),
        script.Sam3Instance(
            label="arrestor",
            class_id=4,
            prompt="arrestor too large",
            score=0.98,
            mask=mask,
            box=np.asarray([0.0, 0.5, 0.9, 0.22], dtype=np.float32),
        ),
    ]

    semantic, confidence, _ = script.build_semantic_outputs(
        instances=instances,
        label_to_id={"TRUCK_BUS": 1, "CAR": 2, "arrestor": 4},
        shape=(2, 2),
        min_mask_size=1,
    )
    visible = script.collect_overlay_instances(
        instances=instances,
        semantic_mask=semantic,
        confidence=confidence,
        min_mask_size=1,
    )

    assert np.all(semantic == 2)
    assert [item.label for item in visible] == ["CAR"]
    assert np.count_nonzero(visible[0].mask) == 4


def test_overlay_instances_keep_only_pixels_owned_after_overlap_resolution():
    script = load_script()
    high_mask = np.asarray([[True, True], [False, False]], dtype=bool)
    low_mask = np.asarray([[False, True], [False, True]], dtype=bool)
    instances = [
        script.Sam3Instance(label="CAR", class_id=2, prompt="car", score=0.9, mask=high_mask),
        script.Sam3Instance(label="TRUCK_BUS", class_id=1, prompt="truck", score=0.8, mask=low_mask),
    ]
    semantic, confidence, _ = script.build_semantic_outputs(
        instances=instances,
        label_to_id={"TRUCK_BUS": 1, "CAR": 2},
        shape=(2, 2),
        min_mask_size=1,
    )
    visible = script.collect_overlay_instances(
        instances=instances,
        semantic_mask=semantic,
        confidence=confidence,
        min_mask_size=1,
    )

    assert len(visible) == 2
    np.testing.assert_array_equal(
        visible[0].mask,
        np.asarray([[True, True], [False, False]], dtype=bool),
    )
    np.testing.assert_array_equal(
        visible[1].mask,
        np.asarray([[False, False], [False, True]], dtype=bool),
    )


def test_classes_log_contains_only_final_overlay_instances(tmp_path: Path):
    script = load_script()
    visible = script.Sam3Instance(
        label="CAR",
        class_id=2,
        prompt="car",
        score=0.9,
        mask=np.asarray([[True, False], [True, False]], dtype=bool),
    )
    result = script.ImageResult(
        image_path=tmp_path / "000001.jpg",
        output_dir=tmp_path / "000001",
        image_size=(2, 2),
        instances=[],
        class_pixel_counts={"CAR": 2},
        overlay_instances=(visible,),
    )

    payload = script.build_classes_log(
        result=result,
        prompt_config=tmp_path / "prompts.yaml",
        min_score=0.45,
        processing_time_seconds=1.23456789,
    )

    assert payload["object_count"] == 1
    assert payload["class_list"] == ["CAR"]
    assert payload["instances"][0]["label"] == "CAR"
    assert payload["instances"][0]["visible_pixel_count"] == 2


def test_priority_mask_overrides_higher_score_normal_mask():
    script = load_script()
    mask = np.ones((2, 2), dtype=bool)
    instances = [
        script.Sam3Instance(label="epoxy_floor", class_id=11, prompt="floor", score=0.95, mask=mask),
        script.Sam3Instance(
            label="ground_markings",
            class_id=25,
            prompt="road marking",
            score=0.60,
            mask=mask,
            box=np.asarray([0.0, 0.4, 0.8, 0.2], dtype=np.float32),
        ),
    ]

    semantic, confidence, counts = script.build_semantic_outputs(
        instances=instances,
        label_to_id={"epoxy_floor": 11, "ground_markings": 25},
        shape=(2, 2),
        min_mask_size=1,
    )

    assert np.all(semantic == 25)
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


def test_server_license_plate_alias_overrides_vehicle():
    script = load_script()
    mask = np.ones((2, 2), dtype=bool)
    semantic, _, _ = script.build_semantic_outputs(
        instances=[
            script.Sam3Instance(label="vehicle", class_id=9, prompt="vehicle", score=0.99, mask=mask),
            script.Sam3Instance(
                label="license_plate&taillights",
                class_id=18,
                prompt="license plate",
                score=0.60,
                mask=mask,
            ),
        ],
        label_to_id={"vehicle": 9, "license_plate&taillights": 18},
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


def test_deduplicate_instances_applies_per_label_and_preserves_other_classes():
    script = load_script()
    object_mask = np.zeros((5, 5), dtype=bool)
    object_mask[1:4, 1:4] = True
    separate_mask = np.zeros((5, 5), dtype=bool)
    separate_mask[0, 0] = True
    instances = [
        script.Sam3Instance(
            label="traffic_cone",
            class_id=15,
            prompt="traffic cone",
            score=0.93,
            mask=object_mask,
        ),
        script.Sam3Instance(
            label="traffic_cone",
            class_id=15,
            prompt="road cone",
            score=0.81,
            mask=object_mask.copy(),
        ),
        script.Sam3Instance(
            label="roadblock",
            class_id=11,
            prompt="roadblock",
            score=0.75,
            mask=object_mask.copy(),
        ),
        script.Sam3Instance(
            label="traffic_cone",
            class_id=15,
            prompt="traffic cone",
            score=0.70,
            mask=separate_mask,
        ),
    ]

    output = script.deduplicate_instances_by_label(instances, iou_threshold=0.8)

    assert [(item.label, item.prompt) for item in output] == [
        ("traffic_cone", "traffic cone"),
        ("roadblock", "roadblock"),
        ("traffic_cone", "traffic cone"),
    ]
    assert [item.score for item in output] == [0.93, 0.75, 0.70]


def test_side_orientation_is_recorded_as_its_own_semantic_class():
    script = load_script()
    mask = np.ones((2, 2), dtype=bool)

    class FakeClassifier:
        def classify(self, image_rgb, masks, *, min_confidence, min_margin):
            return [
                SimpleNamespace(
                    predicted_class="side",
                    semantic_label="side",
                    confidence=0.75,
                    margin=0.50,
                    probabilities=(0.10, 0.15, 0.75),
                    fallback=False,
                    fallback_reason=None,
                )
            ]

    output = script.apply_vehicle_orientation(
        image_rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        instances=[script.Sam3Instance(label="vehicle", class_id=9, prompt="vehicle", score=0.9, mask=mask)],
        classifier=FakeClassifier(),
        vehicle_prompt_label="vehicle",
        class_mapping={
            "front": ("front_of_vehicle", 9),
            "rear": ("rear_of_vehicle", 10),
            "side": ("side_of_vehicle", 33),
        },
        min_confidence=0.7,
        min_margin=0.1,
        nms_iou=0.8,
        min_mask_size=1,
    )

    assert output[0].label == "side_of_vehicle"
    assert output[0].class_id == 33
    assert output[0].orientation_label == "side"
    assert not output[0].orientation_fallback
    assert output[0].orientation_fallback_reason is None


def test_vehicle_class_mapping_uses_active_taxonomy_ids():
    import pytest

    script = load_script()
    mapping = script.resolve_vehicle_class_mapping(
        {
            "background": 0,
            "front_of_vehicle": 1,
            "rear_of_vehicle": 2,
            "side_of_vehicle": 3,
            "ignore": 255,
        }
    )
    assert mapping == {
        "front": ("front_of_vehicle", 1),
        "rear": ("rear_of_vehicle", 2),
        "side": ("side_of_vehicle", 3),
    }

    with pytest.raises(ValueError, match="unique ids"):
        script.resolve_vehicle_class_mapping(
            {"front_of_vehicle": 1, "rear_of_vehicle": 1, "side_of_vehicle": 3}
        )

    with pytest.raises(ValueError, match="background/ignore"):
        script.resolve_vehicle_class_mapping(
            {
                "background": 0,
                "front_of_vehicle": 0,
                "rear_of_vehicle": 2,
                "side_of_vehicle": 3,
                "ignore": 255,
            }
        )


def test_arrestor_geometry_uses_normalized_xywh_and_handles_missing_box():
    script = load_script()
    mask = np.ones((2, 2), dtype=bool)
    rejected_high = script.Sam3Instance(
        label="arrestor",
        class_id=4,
        prompt="arrestor",
        score=0.9,
        mask=mask,
        box=np.asarray([0.1, 0.1, 0.6, 0.2], dtype=np.float32),
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


def test_arrestor_rejects_oversized_box_and_keeps_real_example():
    script = load_script()
    mask = np.ones((2, 2), dtype=bool)
    oversized = script.Sam3Instance(
        label="arrestor",
        class_id=4,
        prompt="yellow and black horizontal parking barrier mounted on short floor supports",
        score=0.6604774594306946,
        mask=mask,
        box=np.asarray(
            [
                0.01848958432674408,
                0.37254902720451355,
                0.9380208849906921,
                0.22385621070861816,
            ],
            dtype=np.float32,
        ),
    )
    correct = script.Sam3Instance(
        label="arrestor",
        class_id=4,
        prompt="yellow and black horizontal parking barrier mounted on short floor supports",
        score=0.7620818018913269,
        mask=mask,
        box=np.asarray(
            [
                0.6333333849906921,
                0.5127451419830322,
                0.06302084028720856,
                0.028104575350880623,
            ],
            dtype=np.float32,
        ),
    )

    assert (
        float(oversized.box[2] * oversized.box[3])
        > script.SIZE_FILTER_RULES["arrestor"]["max_box_area"]
    )
    assert not script.reject_instance_by_geometry(oversized)
    assert script.reject_instance_by_size(oversized)
    assert not script.reject_instance_by_geometry(correct)
    assert not script.reject_instance_by_size(correct)


def test_arrestor_size_rules_check_width_height_and_area_independently():
    script = load_script()
    mask = np.ones((2, 2), dtype=bool)

    def instance(box):
        return script.Sam3Instance(
            label="arrestor",
            class_id=4,
            prompt="arrestor",
            score=0.9,
            mask=mask,
            box=np.asarray(box, dtype=np.float32),
        )

    assert script.reject_instance_by_size(instance([0.0, 0.55, 0.90, 0.05]))
    assert script.reject_instance_by_size(instance([0.1, 0.40, 0.50, 0.22]))
    assert script.reject_instance_by_size(instance([0.1, 0.40, 0.80, 0.20]))


def test_size_filter_rules_support_multiple_labels():
    script = load_script()
    mask = np.ones((2, 2), dtype=bool)
    rules = {
        "arrestor": {"max_box_area": 0.15},
        "traffic_sign": {"min_box_height": 0.05, "max_box_height": 0.4},
    }
    rejected = script.Sam3Instance(
        label="traffic_sign",
        class_id=17,
        prompt="traffic sign",
        score=0.9,
        mask=mask,
        box=np.asarray([0.1, 0.2, 0.1, 0.02], dtype=np.float32),
    )
    accepted = script.Sam3Instance(
        label="traffic_sign",
        class_id=17,
        prompt="traffic sign",
        score=0.9,
        mask=mask,
        box=np.asarray([0.1, 0.2, 0.1, 0.1], dtype=np.float32),
    )
    unchecked = script.Sam3Instance(
        label="ground_markings",
        class_id=25,
        prompt="ground markings",
        score=0.9,
        mask=mask,
        box=None,
    )

    assert script.reject_instance_by_size(rejected, rules=rules)
    assert not script.reject_instance_by_size(accepted, rules=rules)
    assert not script.reject_instance_by_size(unchecked, rules=rules)


def test_geometry_filter_rules_support_multiple_labels():
    script = load_script()
    mask = np.ones((2, 2), dtype=bool)
    rules = {
        "arrestor": {"min_y_bottom": 0.6, "min_aspect_ratio": 2.0},
        "height_restriction_barrel": {
            "min_y_bottom": 0.4,
            "min_aspect_ratio": 0.5,
        },
    }
    rejected = script.Sam3Instance(
        label="height_restriction_barrel",
        class_id=15,
        prompt="height restriction barrel",
        score=0.9,
        mask=mask,
        box=np.asarray([0.1, 0.1, 0.2, 0.2], dtype=np.float32),
    )
    accepted = script.Sam3Instance(
        label="height_restriction_barrel",
        class_id=15,
        prompt="height restriction barrel",
        score=0.9,
        mask=mask,
        box=np.asarray([0.1, 0.3, 0.2, 0.2], dtype=np.float32),
    )
    unchecked = script.Sam3Instance(
        label="traffic_sign",
        class_id=17,
        prompt="traffic sign",
        score=0.9,
        mask=mask,
        box=None,
    )

    assert script.reject_instance_by_geometry(rejected, rules=rules)
    assert not script.reject_instance_by_geometry(accepted, rules=rules)
    assert not script.reject_instance_by_geometry(unchecked, rules=rules)


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
    assert Path(info["projection_copy"]).name == "projection.jpg"
    assert not Path(info["projection_copy"]).is_file()
    assert Path(info["mask_projection"]).is_file()
    assert info["projection_error"] is None


def test_outputs_exist_requires_projection_outputs_when_enabled(tmp_path: Path):
    script = load_script()
    frame_out = tmp_path / "frame"
    frame_out.mkdir()
    for filename in ["semantic_mask.npy", "confidence.npy", "instances.npz", "overlay.jpg", "metadata.json"]:
        (frame_out / filename).write_bytes(b"")

    assert script.outputs_exist(frame_out)
    assert not script.outputs_exist(frame_out, classes_log_enabled=True)
    assert not script.outputs_exist(frame_out, projection_enabled=True)

    (frame_out / "classes_log.json").write_bytes(b"")
    assert script.outputs_exist(frame_out, classes_log_enabled=True)

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
