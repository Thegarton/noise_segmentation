from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SRC = PROJECT_ROOT / "sign_type_classifier" / "src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from sign_type_classifier.clustering import SignDetection, cluster_sign_detections  # noqa: E402
from sign_type_classifier import colour_correction as sign_colour_correction  # noqa: E402
from sign_type_classifier.dataset import (  # noqa: E402
    CLASSIFIER_CLASS_NAMES,
    CLASS_NAMES,
    NOT_A_SIGN_CLASS,
    assign_grouped_splits,
    load_jsonl,
    write_jsonl,
)
from sign_type_classifier.model import (  # noqa: E402
    SignTypeDecision,
    blend_probabilities,
    compute_numeric_normalization,
    decisions_from_probabilities,
    standardize_numeric_features,
)
from sign_type_classifier.detection_filters import (  # noqa: E402
    filter_detections,
    reject_instance_by_geometry,
    reject_instance_by_size,
)
from sign_type_classifier import sam3_adapter as sign_sam3_adapter  # noqa: E402


def test_sign_sam3_visual_features_are_cached_only_within_one_image():
    class FakeBackbone:
        def __init__(self) -> None:
            self.calls = 0

        def forward_image(self, samples):
            self.calls += 1
            return {"features_for": samples}

    backbone = FakeBackbone()
    predictor = SimpleNamespace(
        model=SimpleNamespace(detector=SimpleNamespace(backbone=backbone))
    )
    sign_sam3_adapter._enable_single_image_visual_cache(predictor)

    first = backbone.forward_image("frame-a")
    cached = backbone.forward_image("frame-a-again")

    assert cached is first
    assert backbone.calls == 1
    assert backbone._sam3_visual_cache_hits == 1
    assert backbone._sam3_visual_cache_misses == 1

    sign_sam3_adapter._clear_visual_feature_cache(predictor)
    assert backbone.forward_image("frame-b") == {"features_for": "frame-b"}
    assert backbone.calls == 2


def test_sign_colour_correction_matches_simple_wb_and_gray_world(monkeypatch: pytest.MonkeyPatch):
    class FakeWhiteBalance:
        percentile = None

        def setP(self, value):
            self.percentile = float(value)

        def balanceWhite(self, image):
            return image

    white_balance = FakeWhiteBalance()
    fake_cv2 = SimpleNamespace(
        xphoto=SimpleNamespace(createSimpleWB=lambda: white_balance),
        split=lambda image: tuple(image[..., index] for index in range(3)),
        merge=lambda channels: np.stack(channels, axis=-1),
    )
    monkeypatch.setattr(sign_colour_correction, "_import_cv2", lambda: fake_cv2)

    image_rgb = np.zeros((3, 4, 3), dtype=np.uint8)
    image_rgb[..., 0] = 10
    image_rgb[..., 1] = 20
    image_rgb[..., 2] = 40
    corrected = sign_colour_correction.simple_colour_correction_rgb(image_rgb)

    assert white_balance.percentile == pytest.approx(0.5)
    assert corrected.shape == image_rgb.shape
    assert corrected.dtype == np.uint8
    assert np.ptp(corrected.mean(axis=(0, 1))) <= 1.0


def test_sign_sam3_adapter_passes_corrected_rgb_in_memory(tmp_path: Path):
    pil = pytest.importorskip("PIL.Image")

    class FakePredictor:
        def __init__(self):
            self.model = SimpleNamespace()
            self.default_output_prob_thresh = 0.0
            self.requests = []

        def handle_request(self, *, request):
            self.requests.append(request)
            if request["type"] == "start_session":
                return {"session_id": "test-session"}
            if request["type"] == "add_prompt":
                return {
                    "outputs": {
                        "masks": np.ones((1, 3, 5), dtype=bool),
                        "scores": np.asarray([0.8], dtype=np.float32),
                    }
                }
            return {}

    detector = object.__new__(sign_sam3_adapter.Sam3SignDetector)
    detector.min_score = 0.45
    detector.predictor = FakePredictor()
    corrected_rgb = np.full((3, 5, 3), (11, 22, 33), dtype=np.uint8)

    detections = detector.detect_labeled(
        tmp_path / "does-not-need-to-exist.jpg",
        labeled_prompts=[("side_plate", "roadside sign")],
        image_rgb=corrected_rgb,
    )

    resource = detector.predictor.requests[0]["resource_path"]
    assert isinstance(resource, list) and len(resource) == 1
    assert isinstance(resource[0], pil.Image)
    np.testing.assert_array_equal(np.asarray(resource[0]), corrected_rgb)
    assert len(detections) == 1
    assert detections[0].label == "side_plate"


def _sign_detection(label: str, box: list[float] | None) -> SignDetection:
    return SignDetection(
        label=label,
        prompt=f"{label} prompt",
        score=0.8,
        mask=np.ones((4, 6), dtype=bool),
        box=None if box is None else np.asarray(box, dtype=np.float32),
    )


def test_sign_geometry_filters_use_normalized_xywh_rules():
    parking_valid = _sign_detection("underground_parking_sign", [0.2, 0.20, 0.20, 0.10])
    parking_too_low = _sign_detection("underground_parking_sign", [0.2, 0.35, 0.20, 0.10])
    induction_valid = _sign_detection("induction_sign", [0.2, 0.20, 0.30, 0.10])
    induction_too_wide = _sign_detection("induction_sign", [0.2, 0.20, 0.50, 0.10])
    barrel_too_high = _sign_detection("height_restriction_barrel", [0.2, 0.20, 0.30, 0.10])
    barrel_valid = _sign_detection("height_restriction_barrel", [0.2, 0.35, 0.30, 0.10])

    assert reject_instance_by_geometry(parking_valid) is False
    assert reject_instance_by_geometry(parking_too_low) is True
    assert reject_instance_by_geometry(induction_valid) is False
    assert reject_instance_by_geometry(induction_too_wide) is True
    assert reject_instance_by_geometry(barrel_too_high) is True
    assert reject_instance_by_geometry(barrel_valid) is False
    assert reject_instance_by_geometry(_sign_detection("side_plate", None)) is False


def test_sign_filters_reject_invalid_boxes_and_keep_rejection_reason():
    missing_box = _sign_detection("induction_sign", None)
    oversized = _sign_detection("arrestor", [0.1, 0.50, 0.50, 0.20])
    accepted, rejected = filter_detections([missing_box, oversized])

    assert accepted == []
    assert [item.filter_name for item in rejected] == ["geometry", "size"]
    assert rejected[0].box_metrics is None
    assert reject_instance_by_size(oversized) is True


def test_prompt_config_requires_all_six_sign_classes(tmp_path: Path):
    builder = load_build_script()
    prompts = builder.load_prompt_config(
        PROJECT_ROOT / "sign_type_classifier" / "configs" / "sign_type_prompts.yaml"
    )

    assert tuple(prompts) == CLASS_NAMES
    assert prompts["height_restriction_barrel"] == [
        "overhead height restriction bar",
        "parking clearance bar",
        "height limit barrier beam",
    ]

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("side_plate:\n  - roadside sign\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing="):
        builder.load_prompt_config(invalid)


def test_cross_prompt_clustering_uses_top_class_score_and_keeps_neighbor():
    first = np.zeros((12, 16), dtype=bool)
    first[2:8, 2:8] = True
    contained = np.zeros_like(first)
    contained[3:7, 3:7] = True
    neighbor = np.zeros_like(first)
    neighbor[3:9, 11:15] = True
    detections = [
        SignDetection(
            label="underground_parking_sign",
            prompt="parking sign",
            score=0.82,
            mask=first,
        ),
        SignDetection(
            label="overhead_traffic_sign",
            prompt="overhead sign",
            score=0.91,
            mask=contained,
        ),
        SignDetection(
            label="side_plate",
            prompt="roadside sign",
            score=0.75,
            mask=neighbor,
        ),
    ]

    candidates = cluster_sign_detections(
        detections,
        class_names=CLASS_NAMES,
        min_mask_size=1,
        iou_threshold=0.55,
        containment_threshold=0.80,
    )

    assert len(candidates) == 2
    clustered = candidates[0]
    assert clustered.label == "overhead_traffic_sign"
    assert clustered.score == pytest.approx(0.91)
    assert clustered.margin == pytest.approx(0.09)
    assert clustered.class_scores["underground_parking_sign"] == pytest.approx(0.82)
    assert clustered.class_scores["side_plate"] == 0.0
    np.testing.assert_array_equal(clustered.mask, contained)
    assert candidates[1].label == "side_plate"


def test_equal_scores_use_fixed_class_order():
    mask = np.ones((4, 5), dtype=bool)
    candidates = cluster_sign_detections(
        [
            SignDetection(
                label="height_restriction_barrel",
                prompt="bar",
                score=0.8,
                mask=mask,
            ),
            SignDetection(
                label="height_restriction_sign_at_underground",
                prompt="sign",
                score=0.8,
                mask=mask.copy(),
            ),
        ],
        class_names=CLASS_NAMES,
        min_mask_size=1,
    )

    assert len(candidates) == 1
    assert candidates[0].label == "height_restriction_sign_at_underground"


def test_grouped_split_keeps_all_signs_from_source_frame_together():
    samples = []
    for source_index in range(12):
        for class_index, label in enumerate(CLASS_NAMES):
            samples.append(
                {
                    "sample_id": f"frame-{source_index}-{class_index}",
                    "source_id": f"frame-{source_index}",
                    "label": label,
                }
            )

    result = assign_grouped_splits(samples, seed=42)
    source_splits: dict[str, set[str]] = {}
    for record in result:
        source_splits.setdefault(record["source_id"], set()).add(record["split"])

    assert all(len(values) == 1 for values in source_splits.values())
    assert {record["split"] for record in result} == {"train", "val", "test"}
    assert assign_grouped_splits(samples, seed=42) == result


def test_context_crop_and_numeric_features_preserve_position():
    pytest.importorskip("cv2")
    from sign_type_classifier.features import build_numeric_features, extract_sign_crops

    image = np.zeros((20, 30, 3), dtype=np.uint8)
    image[..., 0] = np.arange(30, dtype=np.uint8)[None, :]
    mask = np.zeros((20, 30), dtype=bool)
    mask[1:5, 0:4] = True
    scores = {label: 0.0 for label in CLASS_NAMES}
    scores["side_plate"] = 0.8

    crops = extract_sign_crops(image, mask, crop_padding=0.12, context_scale=3.0)
    features = build_numeric_features(
        image,
        mask,
        class_names=CLASS_NAMES,
        class_scores=scores,
        context_scale=3.0,
    )

    assert crops.mask_bbox_xyxy == (0, 1, 4, 5)
    assert crops.tight_bbox_xyxy[0] == 0
    assert crops.context_bbox_xyxy[0] == 0
    assert crops.context_rgb.shape[1] == 12
    assert features["bbox_x0"] == 0.0
    assert features["bbox_y0"] == pytest.approx(0.05)
    assert features["touches_left"] == 1.0
    assert features["sam_top1_score"] == pytest.approx(0.8)
    assert features["sam_top1_margin"] == pytest.approx(0.8)
    assert "object_hue_hist_0" in features
    assert "context_edge_density" in features


def test_builder_checkpoints_and_resumes_with_fake_detector(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    builder = load_build_script()
    source_root = tmp_path / "images"
    out_dir = tmp_path / "dataset"
    source_root.mkdir()
    for index in range(2):
        (source_root / f"{index:06d}.jpg").write_bytes(b"prepared image")

    class FakeDetector:
        calls = 0
        corrected_means = []

        def __init__(self, **kwargs):
            pass

        def detect_labeled(self, image_path, *, labeled_prompts, image_rgb=None):
            FakeDetector.calls += 1
            FakeDetector.corrected_means.append(float(np.mean(image_rgb)))
            mask = np.zeros((8, 10), dtype=bool)
            mask[2:6, 3:8] = True
            return [
                SimpleNamespace(
                    label="side_plate",
                    prompt="roadside traffic sign",
                    score=0.8,
                    mask=mask,
                    box=None,
                )
            ]

    fake_crops = SimpleNamespace(
        mask_bbox_xyxy=(3, 2, 8, 6),
        tight_bbox_xyxy=(2, 1, 9, 7),
        context_bbox_xyxy=(0, 0, 10, 8),
    )

    def fake_copy(source_path, *, out_dir, source_id):
        target = out_dir / "images" / f"{source_id}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"source")
        return target

    def fake_save(*, sample_dir, review_path, **kwargs):
        sample_dir.mkdir(parents=True, exist_ok=True)
        review_path.parent.mkdir(parents=True, exist_ok=True)
        paths = {
            "rgb": sample_dir / "rgb.png",
            "mask": sample_dir / "mask.png",
            "masked": sample_dir / "masked_rgb.png",
            "context": sample_dir / "context_rgb.png",
            "context_mask": sample_dir / "context_mask.png",
            "preview": sample_dir / "preview.jpg",
            "review": review_path,
        }
        for path in paths.values():
            path.write_bytes(b"asset")
        return paths

    monkeypatch.setattr(builder, "Sam3SignDetector", FakeDetector)
    monkeypatch.setattr(builder, "read_rgb", lambda path: np.zeros((8, 10, 3), dtype=np.uint8))
    monkeypatch.setattr(builder, "simple_colour_correction_rgb", lambda image: image + 7)
    monkeypatch.setattr(builder, "copy_source_image", fake_copy)
    monkeypatch.setattr(builder, "extract_sign_crops", lambda *args, **kwargs: fake_crops)
    monkeypatch.setattr(builder, "numeric_feature_names", lambda names: ("feature",))
    monkeypatch.setattr(builder, "build_numeric_features", lambda *args, **kwargs: {"feature": 1.0})
    monkeypatch.setattr(builder, "save_sample_assets", fake_save)
    argv = [
        "build_dataset.py",
        "--source-root",
        str(source_root),
        "--out-dir",
        str(out_dir),
        "--sam3-root",
        str(tmp_path / "sam3"),
        "--sam3-model-path",
        str(tmp_path / "sam3.1"),
        "--min-mask-size",
        "1",
        "--colour-correction",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    builder.main()

    records = load_jsonl(out_dir / "manifest.jsonl")
    state = json.loads((out_dir / "generation_state.json").read_text(encoding="utf-8"))
    assert len(records) == 2
    assert FakeDetector.calls == 2
    assert FakeDetector.corrected_means == [7.0, 7.0]
    assert state["status"] == "complete"
    assert state["completed_frames"] == 2
    assert all(record["numeric_feature_vector"] == [1.0] for record in records)
    dataset_manifest = json.loads((out_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert dataset_manifest["colour_correction"] is True

    class MustNotLoadDetector:
        def __init__(self, **kwargs):
            raise AssertionError("SAM3 must not load when all selected frames are complete")

    monkeypatch.setattr(builder, "Sam3SignDetector", MustNotLoadDetector)
    monkeypatch.setattr(sys, "argv", [*argv, "--resume"])
    builder.main()
    assert len(load_jsonl(out_dir / "manifest.jsonl")) == 2


def test_builder_uses_trained_classifier_to_sort_sam3_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    builder = load_build_script()
    source_root = tmp_path / "images"
    out_dir = tmp_path / "dataset"
    source_root.mkdir()
    (source_root / "000000.jpg").write_bytes(b"prepared image")
    checkpoint = tmp_path / "model_best.pth"
    checkpoint.write_bytes(b"fake checkpoint")

    class FakeDetector:
        def __init__(self, **kwargs):
            pass

        def detect_labeled(self, image_path, *, labeled_prompts):
            mask = np.zeros((8, 10), dtype=bool)
            mask[2:6, 3:8] = True
            return [
                SimpleNamespace(
                    label="side_plate",
                    prompt="roadside traffic sign",
                    score=0.8,
                    mask=mask,
                    box=None,
                )
            ]

    class FakeClassifier:
        calls = 0
        feature_names = ("feature",)

        def __init__(self, checkpoint_path, *, device):
            assert Path(checkpoint_path) == checkpoint
            assert device == "cpu"

        def classify(self, context_images_rgb, numeric_features):
            FakeClassifier.calls += 1
            assert len(context_images_rgb) == 1
            assert numeric_features.tolist() == [[3.0]]
            probabilities = [0.01] * len(CLASSIFIER_CLASS_NAMES)
            probabilities[CLASSIFIER_CLASS_NAMES.index("not_a_sign")] = 0.94
            probabilities = tuple(probabilities)
            return [
                SignTypeDecision(
                    label="not_a_sign",
                    confidence=0.94,
                    margin=0.80,
                    probabilities=probabilities,
                    image_probabilities=probabilities,
                    numeric_probabilities=probabilities,
                )
            ]

    mask = np.zeros((8, 10), dtype=bool)
    mask[2:6, 3:8] = True
    fake_crops = SimpleNamespace(
        mask_bbox_xyxy=(3, 2, 8, 6),
        tight_bbox_xyxy=(2, 1, 9, 7),
        context_bbox_xyxy=(0, 0, 10, 8),
        rgb=np.zeros((6, 7, 3), dtype=np.uint8),
        mask=mask[1:7, 2:9],
        masked_rgb=np.zeros((6, 7, 3), dtype=np.uint8),
        context_rgb=np.zeros((8, 10, 3), dtype=np.uint8),
        context_mask=mask,
    )

    def fake_copy(source_path, *, out_dir, source_id):
        target = out_dir / "images" / f"{source_id}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"source")
        return target

    def fake_save(*, sample_dir, review_path, **kwargs):
        sample_dir.mkdir(parents=True, exist_ok=True)
        review_path.parent.mkdir(parents=True, exist_ok=True)
        paths = {
            "rgb": sample_dir / "rgb.png",
            "mask": sample_dir / "mask.png",
            "masked": sample_dir / "masked_rgb.png",
            "context": sample_dir / "context_rgb.png",
            "context_mask": sample_dir / "context_mask.png",
            "preview": sample_dir / "preview.jpg",
            "review": review_path,
        }
        for path in paths.values():
            path.write_bytes(b"asset")
        return paths

    monkeypatch.setattr(builder, "Sam3SignDetector", FakeDetector)
    monkeypatch.setattr(builder, "SignTypeEnsembleClassifier", FakeClassifier)
    monkeypatch.setattr(builder, "read_rgb", lambda path: np.zeros((8, 10, 3), dtype=np.uint8))
    monkeypatch.setattr(builder, "copy_source_image", fake_copy)
    monkeypatch.setattr(builder, "extract_sign_crops", lambda *args, **kwargs: fake_crops)
    monkeypatch.setattr(builder, "numeric_feature_names", lambda names: ("feature",))
    monkeypatch.setattr(builder, "build_numeric_features", lambda *args, **kwargs: {"feature": 3.0})
    monkeypatch.setattr(builder, "save_sample_assets", fake_save)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_dataset.py",
            "--source-root",
            str(source_root),
            "--out-dir",
            str(out_dir),
            "--sam3-root",
            str(tmp_path / "sam3"),
            "--sam3-model-path",
            str(tmp_path / "sam3.1"),
            "--classifier-checkpoint",
            str(checkpoint),
            "--classifier-device",
            "cpu",
            "--min-mask-size",
            "1",
        ],
    )

    builder.main()

    records = load_jsonl(out_dir / "manifest.jsonl")
    assert FakeClassifier.calls == 1
    assert len(records) == 1
    assert records[0]["label"] == "not_a_sign"
    assert records[0]["initial_label"] == "side_plate"
    assert records[0]["classifier_accepted"] is True
    assert records[0]["classifier_prediction"]["predicted_label"] == "not_a_sign"
    assert records[0]["review_image"].startswith("review/not_a_sign/")
    manifest = json.loads((out_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["classifier"]["enabled"] is True


def test_builder_smoke_and_resume_without_cuda(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("cv2")
    pil = pytest.importorskip("PIL.Image")
    builder = load_build_script()
    source_root = tmp_path / "images"
    out_dir = tmp_path / "dataset"
    source_root.mkdir()
    for index in range(3):
        image = np.full((24, 32, 3), 40 + index, dtype=np.uint8)
        pil.fromarray(image).save(source_root / f"{index:06d}.jpg")

    class FakeDetector:
        calls = 0

        def __init__(self, **kwargs):
            pass

        def detect_labeled(self, image_path, *, labeled_prompts):
            FakeDetector.calls += 1
            assert {label for label, _ in labeled_prompts} == set(CLASS_NAMES)
            full = np.zeros((24, 32), dtype=bool)
            full[4:18, 5:21] = True
            detail = np.zeros_like(full)
            detail[6:16, 7:19] = True
            return [
                SimpleNamespace(
                    label="underground_parking_sign",
                    prompt="parking sign",
                    score=0.85,
                    mask=full,
                    box=np.asarray([0.1, 0.2, 0.5, 0.4], dtype=np.float32),
                ),
                SimpleNamespace(
                    label="overhead_traffic_sign",
                    prompt="overhead sign",
                    score=0.75,
                    mask=detail,
                    box=None,
                ),
            ]

    monkeypatch.setattr(builder, "Sam3SignDetector", FakeDetector)
    argv = [
        "build_dataset.py",
        "--source-root",
        str(source_root),
        "--out-dir",
        str(out_dir),
        "--sam3-root",
        str(tmp_path / "sam3"),
        "--sam3-model-path",
        str(tmp_path / "sam3.1"),
        "--min-mask-size",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    builder.main()

    records = load_jsonl(out_dir / "manifest.jsonl")
    assert len(records) == 3
    assert FakeDetector.calls == 3
    assert {record["label"] for record in records} == {"underground_parking_sign"}
    assert all((out_dir / record["context_rgb"]).is_file() for record in records)
    assert all((out_dir / record["review_image"]).is_file() for record in records)
    assert all(len(record["numeric_feature_vector"]) == len(record["numeric_features"]) for record in records)
    raw_crop = np.asarray(pil.open(out_dir / records[0]["rgb_crop"]).convert("RGB"))
    assert np.all(raw_crop == 40)
    state = json.loads((out_dir / "generation_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "complete"
    assert state["completed_frames"] == 3

    class MustNotLoadDetector:
        def __init__(self, **kwargs):
            raise AssertionError("SAM3 must not be initialized when every selected frame is complete")

    monkeypatch.setattr(builder, "Sam3SignDetector", MustNotLoadDetector)
    monkeypatch.setattr(sys, "argv", [*argv, "--resume"])
    builder.main()
    assert len(load_jsonl(out_dir / "manifest.jsonl")) == 3


def test_reindex_moves_and_deletes_review_samples(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    reindex = load_reindex_script()
    dataset_dir = tmp_path / "dataset"
    records = create_review_dataset(dataset_dir, labels=("side_plate", "induction_sign", "side_plate"))
    moved = dataset_dir / "review" / "induction_sign" / f"{records[0]['sample_id']}.jpg"
    (dataset_dir / records[0]["review_image"]).replace(moved)
    (dataset_dir / records[2]["review_image"]).unlink()
    monkeypatch.setattr(sys, "argv", ["reindex_dataset.py", "--dataset-dir", str(dataset_dir)])

    reindex.main()

    updated = load_jsonl(dataset_dir / "manifest.jsonl")
    assert len(updated) == 2
    first = next(item for item in updated if item["sample_id"] == records[0]["sample_id"])
    assert first["label"] == "induction_sign"
    assert first["previous_label"] == "side_plate"
    assert all(item["sample_id"] != records[2]["sample_id"] for item in updated)
    manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["manual_review"]["removed_sample_ids"] == [records[2]["sample_id"]]


def test_reindex_keeps_not_a_sign_as_classifier_class(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    reindex = load_reindex_script()
    dataset_dir = tmp_path / "dataset"
    records = create_review_dataset(dataset_dir, labels=("side_plate", "induction_sign"))
    not_a_sign_dir = dataset_dir / "review" / NOT_A_SIGN_CLASS
    not_a_sign_dir.mkdir(parents=True)
    moved = not_a_sign_dir / f"{records[0]['sample_id']}.jpg"
    (dataset_dir / records[0]["review_image"]).replace(moved)
    monkeypatch.setattr(sys, "argv", ["reindex_dataset.py", "--dataset-dir", str(dataset_dir)])

    reindex.main()

    updated = load_jsonl(dataset_dir / "manifest.jsonl")
    assert len(updated) == 2
    negative = next(item for item in updated if item["sample_id"] == records[0]["sample_id"])
    assert negative["label"] == NOT_A_SIGN_CLASS
    manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert tuple(manifest["class_names"]) == CLASSIFIER_CLASS_NAMES


def test_numeric_normalization_and_probability_fusion():
    values = np.asarray([[1.0, 2.0, 5.0], [3.0, 2.0, 9.0]], dtype=np.float32)
    mean, std = compute_numeric_normalization(values)
    normalized = standardize_numeric_features(values, mean=mean, std=std)

    assert np.allclose(normalized.mean(axis=0), 0.0)
    assert np.allclose(normalized[:, 1], 0.0)

    image = np.zeros((1, len(CLASSIFIER_CLASS_NAMES)), dtype=np.float32)
    numeric = np.zeros_like(image)
    image[0, 0] = 1.0
    numeric[0, 1] = 1.0
    fused = blend_probabilities(image, numeric, image_weight=0.75)
    decisions = decisions_from_probabilities(
        fused,
        image_probabilities=image,
        numeric_probabilities=numeric,
    )
    assert decisions[0].label == CLASSIFIER_CLASS_NAMES[0]
    assert decisions[0].confidence == pytest.approx(0.75)


def test_classifier_assignment_falls_back_to_sam3_below_thresholds():
    probabilities = tuple([1.0 / len(CLASSIFIER_CLASS_NAMES)] * len(CLASSIFIER_CLASS_NAMES))
    prediction = SignTypeDecision(
        label="overhead_traffic_sign",
        confidence=0.45,
        margin=0.02,
        probabilities=probabilities,
        image_probabilities=probabilities,
        numeric_probabilities=probabilities,
    )
    builder = load_build_script()

    assignment = builder.resolve_auto_label_assignment(
        sam3_label="side_plate",
        classifier_prediction=prediction,
        min_confidence=0.50,
        min_margin=0.05,
    )

    assert assignment.assigned_label == "side_plate"
    assert assignment.classifier_accepted is False
    assert assignment.classifier_fallback_reason == "low_confidence+low_margin"


def test_reindex_merges_datasets_with_colliding_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    reindex = load_reindex_script()
    first = tmp_path / "first"
    second = tmp_path / "second"
    merged = tmp_path / "merged"
    create_review_dataset(first, labels=("side_plate",))
    create_review_dataset(second, labels=("induction_sign",))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reindex_dataset.py",
            "--dataset-dir",
            str(first),
            "--dataset-dir",
            str(second),
            "--out-dir",
            str(merged),
        ],
    )

    reindex.main()

    records = load_jsonl(merged / "manifest.jsonl")
    assert len(records) == 2
    assert {item["label"] for item in records} == {"side_plate", "induction_sign"}
    assert len({item["sample_id"] for item in records}) == 2
    assert all((merged / item["review_image"]).is_file() for item in records)
    assert all((merged / item["metadata"]).is_file() for item in records)


def create_review_dataset(dataset_dir: Path, *, labels: tuple[str, ...]) -> list[dict]:
    for class_name in CLASS_NAMES:
        (dataset_dir / "review" / class_name).mkdir(parents=True, exist_ok=True)
    records = []
    for index, label in enumerate(labels):
        sample_id = f"sample_{index}"
        source_id = f"frame_{index // 2}"
        sample_dir = dataset_dir / "samples" / sample_id
        sample_dir.mkdir(parents=True)
        (sample_dir / "context_rgb.png").write_bytes(b"context")
        (dataset_dir / "review" / label / f"{sample_id}.jpg").write_bytes(b"review")
        record = {
            "sample_id": sample_id,
            "source_id": source_id,
            "label": label,
            "initial_label": label,
            "context_rgb": f"samples/{sample_id}/context_rgb.png",
            "review_image": f"review/{label}/{sample_id}.jpg",
            "metadata": f"samples/{sample_id}/metadata.json",
            "numeric_feature_vector": [0.0],
        }
        (sample_dir / "metadata.json").write_text(json.dumps(record), encoding="utf-8")
        records.append(record)
    write_jsonl(dataset_dir / "manifest.jsonl", records)
    (dataset_dir / "dataset_manifest.json").write_text(
        json.dumps({"class_names": list(CLASS_NAMES), "numeric_feature_names": ["feature"]}),
        encoding="utf-8",
    )
    return records


def load_build_script():
    return load_script(
        "sign_type_build_dataset",
        PROJECT_ROOT / "sign_type_classifier" / "scripts" / "build_dataset.py",
    )


def load_reindex_script():
    return load_script(
        "sign_type_reindex_dataset",
        PROJECT_ROOT / "sign_type_classifier" / "scripts" / "reindex_dataset.py",
    )


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
