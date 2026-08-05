from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SRC = PROJECT_ROOT / "vehicle_orientation" / "src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from vehicle_orientation.dataset import (  # noqa: E402
    assign_stratified_splits,
    collect_mixed_source_images,
    load_jsonl,
    source_id_from_relative_path,
)
from vehicle_orientation.model import decide_orientation  # noqa: E402
from vehicle_orientation.preprocessing import (  # noqa: E402
    FisheyeConfig,
    FisheyePreprocessor,
    deduplicate_mask_indices,
    extract_mask_crop,
)


def test_extract_mask_crop_uses_padding_and_neutral_background():
    image = np.full((8, 10, 3), [10, 20, 30], dtype=np.uint8)
    mask = np.zeros((8, 10), dtype=bool)
    mask[2:6, 0:3] = True

    crop = extract_mask_crop(image, mask, padding=0.25, background=127)

    assert crop.bbox_xyxy == (0, 1, 4, 7)
    assert crop.rgb.shape == (6, 4, 3)
    assert np.all(crop.masked_rgb[crop.mask] == [10, 20, 30])
    assert np.all(crop.masked_rgb[~crop.mask] == 127)


def test_mask_nms_keeps_highest_score_and_distinct_instance():
    first = np.zeros((5, 5), dtype=bool)
    duplicate = np.zeros((5, 5), dtype=bool)
    distinct = np.zeros((5, 5), dtype=bool)
    first[1:4, 1:4] = True
    duplicate[1:4, 1:4] = True
    distinct[0, 0] = True

    keep = deduplicate_mask_indices(
        [first, duplicate, distinct],
        [0.9, 0.8, 0.7],
        iou_threshold=0.8,
    )

    assert keep == [0, 2]


def test_group_aware_split_keeps_mixed_cars_from_one_frame_together():
    samples = []
    for source_index in range(10):
        for crop_index, label in enumerate(("front", "rear", "side")):
            samples.append(
                {
                    "sample_id": f"frame-{source_index}-{crop_index}",
                    "source_id": f"frame-{source_index}",
                    "label": label,
                }
            )

    split = assign_stratified_splits(samples, seed=42)
    source_splits: dict[str, set[str]] = {}
    for item in split:
        source_splits.setdefault(item["source_id"], set()).add(item["split"])

    assert all(len(values) == 1 for values in source_splits.values())
    assert {item["split"] for item in split} == {"train", "val", "test"}
    assert assign_stratified_splits(samples, seed=42) == split


def test_collect_mixed_source_images_needs_no_class_folders(tmp_path: Path):
    (tmp_path / "000002.jpg").write_bytes(b"fake")
    (tmp_path / "000001.png").write_bytes(b"fake")
    nested = tmp_path / "camera_a"
    nested.mkdir()
    (nested / "000003.jpeg").write_bytes(b"fake")
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")

    records = collect_mixed_source_images(tmp_path)

    assert [item["source_relative_path"] for item in records] == [
        "000001.png",
        "000002.jpg",
        "camera_a/000003.jpeg",
    ]
    assert all("label" not in item for item in records)


def test_orientation_policy_preserves_side_and_marks_uncertainty_without_relabeling():
    rear = decide_orientation([0.05, 0.90, 0.05], min_confidence=0.7, min_margin=0.1)
    side = decide_orientation([0.10, 0.20, 0.70], min_confidence=0.7, min_margin=0.1)
    uncertain = decide_orientation([0.45, 0.40, 0.15], min_confidence=0.7, min_margin=0.1)

    assert rear.semantic_label == "rear" and not rear.fallback
    assert side.predicted_class == "side" and side.semantic_label == "side" and not side.fallback
    assert uncertain.semantic_label == "front" and uncertain.fallback_reason == "low_confidence"


def test_fisheye_preprocessor_reuses_remap(monkeypatch: pytest.MonkeyPatch):
    import vehicle_orientation.preprocessing as preprocessing

    class FakeCV2:
        INTER_LINEAR = 1
        INTER_CUBIC = 2
        INTER_LANCZOS4 = 4
        BORDER_CONSTANT = 0

        def __init__(self) -> None:
            self.calls = 0

        def remap(self, image, map_x, map_y, **kwargs):
            self.calls += 1
            return np.zeros((*map_x.shape, 3), dtype=np.uint8)

    fake = FakeCV2()
    monkeypatch.setattr(preprocessing, "_import_cv2", lambda: fake)
    config = FisheyeConfig(
        output_width=7,
        output_height=5,
        fov=180.0,
        pfov=90.0,
        xcenter=4.0,
        ycenter=3.0,
        radius=3.0,
        mask_radius=4.0,
        mask_center_y_offset=0.0,
        color_correction=False,
    )
    processor = FisheyePreprocessor(config)
    image = np.ones((8, 10, 3), dtype=np.uint8)

    first = processor.prepare_bgr(image)
    second = processor.prepare_bgr(image)

    assert first.shape == (5, 7, 3)
    assert second.shape == first.shape
    assert processor.cache_size == 1
    assert fake.calls == 2


def test_auto_label_uses_full_vehicle_mask_and_orientation_prompt_only_for_label():
    builder = load_build_script()
    full = np.zeros((8, 12), dtype=bool)
    full[1:7, 1:6] = True
    duplicate = full.copy()
    rear_detail = np.zeros_like(full)
    rear_detail[2:6, 1:3] = True
    second_vehicle = np.zeros_like(full)
    second_vehicle[2:7, 8:11] = True
    detections = [
        SimpleNamespace(label="generic", prompt="vehicle", score=0.95, mask=full, box=None),
        SimpleNamespace(label="generic", prompt="car", score=0.80, mask=duplicate, box=None),
        SimpleNamespace(label="rear", prompt="back of a car", score=0.75, mask=rear_detail, box=None),
        SimpleNamespace(label="generic", prompt="vehicle", score=0.90, mask=second_vehicle, box=None),
    ]

    vehicles = builder.auto_label_vehicle_instances(
        detections,
        min_mask_size=1,
        nms_iou=0.8,
        orientation_match_overlap=0.3,
    )

    rear = next(item for item in vehicles if item.label == "rear")
    side = next(item for item in vehicles if item.label == "side")
    assert np.array_equal(rear.mask, full)
    assert rear.orientation_prompt == "back of a car"
    assert side.label_source == "unmatched_generic_fallback_side"
    assert len(vehicles) == 2


def test_auto_label_rejects_orientation_detection_without_whitelisted_vehicle_mask():
    builder = load_build_script()
    motorcycle_like_mask = np.ones((5, 5), dtype=bool)

    vehicles = builder.auto_label_vehicle_instances(
        [
            SimpleNamespace(
                label="front",
                prompt="front view of a vehicle",
                score=0.95,
                mask=motorcycle_like_mask,
                box=None,
            )
        ],
        min_mask_size=1,
        nms_iou=0.8,
        orientation_match_overlap=0.3,
    )

    assert vehicles == []


def test_efficientnet_cpu_forward_smoke():
    pytest.importorskip("torchvision")
    import torch

    from vehicle_orientation.model import build_efficientnet_b0

    model = build_efficientnet_b0(pretrained=False)
    model.eval()
    with torch.inference_mode():
        output = model(torch.zeros((1, 3, 224, 224), dtype=torch.float32))
    assert output.shape == (1, 3)


def test_training_epoch_and_checkpoint_cpu_smoke(tmp_path: Path):
    torch = pytest.importorskip("torch")
    train_script = load_train_script()
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 4 * 4, 3))
    images = torch.randn((9, 3, 4, 4), dtype=torch.float32)
    targets = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=torch.long)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(images, targets), batch_size=3)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    metrics = train_script._run_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        device="cpu",
        torch=torch,
        optimizer=optimizer,
    )
    payload = train_script._checkpoint_payload(
        model=model,
        optimizer=optimizer,
        epoch=1,
        input_size=224,
        crop_padding=0.12,
        class_weights=[1.0, 1.0, 1.0],
        metrics={"train": metrics},
    )
    checkpoint = tmp_path / "model_last.pth"
    torch.save(payload, checkpoint)
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)

    assert metrics["samples"] == 9
    assert np.asarray(metrics["confusion_matrix"]).shape == (3, 3)
    assert loaded["class_names"] == ("front", "rear", "side")
    assert loaded["model_state_dict"].keys() == model.state_dict().keys()


def test_dataset_builder_auto_sorts_one_mixed_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    builder = load_build_script()
    source_root = tmp_path / "mixed_images"
    out_dir = tmp_path / "dataset"
    source_root.mkdir()
    for index in range(9):
        (source_root / f"{index:06d}.jpg").write_bytes(b"fake")

    class FakePreprocessor:
        def __init__(self, config):
            self.config = config
            self.cache_size = 1

        def prepare_bgr(self, image):
            return np.zeros((10, 12, 3), dtype=np.uint8)

    class FakeDetector:
        def __init__(self, **kwargs):
            self.calls = 0

        def detect_labeled(self, image_path, *, labeled_prompts):
            label = ("front", "rear", "side")[self.calls % 3]
            self.calls += 1
            full = np.zeros((10, 12), dtype=bool)
            full[1:9, 1:11] = True
            detail = np.zeros_like(full)
            detail[2:8, 2:6] = True
            return [
                SimpleNamespace(label="generic", prompt="vehicle", score=0.9, mask=full, box=None),
                SimpleNamespace(label=label, prompt=f"{label} of car", score=0.8, mask=detail, box=None),
            ]

    monkeypatch.setattr(builder, "FisheyePreprocessor", FakePreprocessor)
    monkeypatch.setattr(builder, "Sam3VehicleDetector", FakeDetector)
    monkeypatch.setattr(builder, "_read_bgr", lambda path: np.zeros((8, 10, 3), dtype=np.uint8))

    def fake_write(path, image):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"prepared")

    monkeypatch.setattr(builder, "_write_bgr", fake_write)
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
            "--min-mask-size",
            "1",
        ],
    )

    builder.main()

    records = load_jsonl(out_dir / "manifest.jsonl")
    summary = json.loads((out_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert len(records) == 9
    assert summary["summary"]["class_counts"] == {"front": 3, "rear": 3, "side": 3}
    assert summary["summary"]["split_counts"] == {"test": 1, "train": 7, "val": 1}
    assert len(list((out_dir / "samples").glob("*/rgb.png"))) == 9
    assert len(list((out_dir / "review" / "front").glob("*.png"))) == 3
    assert len(list((out_dir / "review" / "rear").glob("*.png"))) == 3
    assert len(list((out_dir / "review" / "side").glob("*.png"))) == 3


def test_efficientnet_bootstrap_keeps_side_prediction():
    generator = load_efficientnet_generator_script()
    mask = np.ones((4, 4), dtype=bool)

    class FakeClassifier:
        def classify(self, image_rgb, masks, *, min_confidence, min_margin):
            return [
                SimpleNamespace(
                    predicted_class="side",
                    semantic_label="side",
                    confidence=0.80,
                    margin=0.60,
                    probabilities=(0.10, 0.10, 0.80),
                    fallback=False,
                    fallback_reason=None,
                )
            ]

    classified = generator.classify_vehicle_detections(
        classifier=FakeClassifier(),
        image_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        detections=[SimpleNamespace(prompt="vehicle", score=0.9, mask=mask, box=None)],
        review_min_confidence=0.70,
        review_min_margin=0.10,
    )

    assert classified[0].label == "side"
    assert not classified[0].needs_review


def test_efficientnet_bootstrap_detects_only_generic_vehicles_and_builds_review_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    generator = load_efficientnet_generator_script()
    image_dir = tmp_path / "mixed_images"
    out_dir = tmp_path / "generated_dataset"
    image_dir.mkdir()
    for index in range(6):
        (image_dir / f"{index:06d}.jpg").write_bytes(b"fake")

    class FakePreprocessor:
        def __init__(self, config):
            self.config = config
            self.cache_size = 1

        def prepare_bgr(self, image):
            return np.zeros((10, 12, 3), dtype=np.uint8)

    class FakeDetector:
        def __init__(self, **kwargs):
            pass

        def detect(self, image_path, *, prompts):
            assert tuple(prompts) == (
                "passenger car",
                "sport utility vehicle",
                "van",
                "pickup truck",
                "truck",
                "bus",
            )
            assert all("front" not in prompt and "rear" not in prompt and "side" not in prompt for prompt in prompts)
            mask = np.zeros((10, 12), dtype=bool)
            mask[1:9, 1:11] = True
            return [SimpleNamespace(label="vehicle", prompt="vehicle", score=0.9, mask=mask, box=None)]

    class FakeClassifier:
        def __init__(self, checkpoint, *, device):
            self.calls = 0
            self.crop_padding = 0.12
            self.input_size = 224
            self.device = "cpu"

        def classify(self, image_rgb, masks, *, min_confidence, min_margin):
            label_index = self.calls % 3
            self.calls += 1
            probabilities = [0.05, 0.05, 0.05]
            probabilities[label_index] = 0.90
            return [
                SimpleNamespace(
                    predicted_class=("front", "rear", "side")[label_index],
                    semantic_label=("front", "rear", "side")[label_index],
                    confidence=0.90,
                    margin=0.85,
                    probabilities=tuple(probabilities),
                    fallback=False,
                    fallback_reason=None,
                )
            ]

    monkeypatch.setattr(generator, "FisheyePreprocessor", FakePreprocessor)
    monkeypatch.setattr(generator, "Sam3VehicleDetector", FakeDetector)
    monkeypatch.setattr(generator, "VehicleOrientationClassifier", FakeClassifier)
    monkeypatch.setattr(generator, "_read_bgr", lambda path: np.zeros((8, 10, 3), dtype=np.uint8))

    def fake_write(path, image):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"prepared")

    monkeypatch.setattr(generator, "_write_bgr", fake_write)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_dataset_with_efficientnet.py",
            "--image-dir",
            str(image_dir),
            "--out-dir",
            str(out_dir),
            "--checkpoint",
            str(tmp_path / "model_best.pth"),
            "--sam3-root",
            str(tmp_path / "sam3"),
            "--sam3-model-path",
            str(tmp_path / "sam3.1"),
            "--min-mask-size",
            "1",
        ],
    )

    generator.main()

    records = load_jsonl(out_dir / "manifest.jsonl")
    dataset_manifest = json.loads((out_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert len(records) == 6
    assert dataset_manifest["mode"] == "sam3_vehicle_efficientnet_bootstrap"
    assert dataset_manifest["summary"]["class_counts"] == {"front": 2, "rear": 2, "side": 2}
    assert len(list((out_dir / "annotated").glob("*.jpg"))) == 6
    assert len(list((out_dir / "review" / "front").glob("*.png"))) == 2
    assert len(list((out_dir / "review" / "rear").glob("*.png"))) == 2
    assert len(list((out_dir / "review" / "side").glob("*.png"))) == 2
    assert all(item["label_source"] == "efficientnet_b0" for item in records)


def test_generator_checkpoints_manifest_after_each_completed_frame(tmp_path: Path):
    generator = load_efficientnet_generator_script()
    out_dir = tmp_path / "dataset"
    out_dir.mkdir()
    samples = [
        {"sample_id": "frame_0_000", "source_id": "frame_0", "label": "front"},
        {"sample_id": "frame_1_000", "source_id": "frame_1", "label": "rear"},
    ]

    generator._checkpoint_progress(
        out_dir=out_dir,
        samples=samples,
        source_results=[{"source_id": "frame_0"}, {"source_id": "frame_1"}],
        completed_source_ids={"frame_0", "frame_1"},
        seed=42,
        train_ratio=0.70,
        val_ratio=0.15,
        status="running",
    )

    manifest = load_jsonl(out_dir / "manifest.jsonl")
    state = json.loads((out_dir / "generation_state.json").read_text(encoding="utf-8"))
    assert len(manifest) == 2
    assert all("split" in item for item in manifest)
    assert state["status"] == "running"
    assert state["completed_source_ids"] == ["frame_0", "frame_1"]


def test_recover_interrupted_generator_uses_current_review_folders_and_can_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    recover_script = load_recover_generator_script()
    generator = load_efficientnet_generator_script()
    image_dir = tmp_path / "images"
    dataset_dir = tmp_path / "interrupted"
    image_dir.mkdir()
    (image_dir / "000001.jpg").write_bytes(b"source image is not decoded during recovery")
    source_id = source_id_from_relative_path("000001.jpg")
    sample_id = f"{source_id}_000"
    for label in ("front", "rear", "side"):
        (dataset_dir / "review" / label).mkdir(parents=True, exist_ok=True)
    sample_dir = dataset_dir / "samples" / sample_id
    sample_dir.mkdir(parents=True)
    mask = np.zeros((4, 5), dtype=np.uint8)
    mask[1:3, 1:4] = 255
    Image.fromarray(mask).save(sample_dir / "mask.png")
    Image.fromarray(np.full((4, 5, 3), 127, dtype=np.uint8)).save(sample_dir / "masked_rgb.png")
    Image.fromarray(np.full((4, 5, 3), 127, dtype=np.uint8)).save(
        dataset_dir / "review" / "rear" / f"{sample_id}.png"
    )
    annotated_path = dataset_dir / "annotated" / f"{source_id}.jpg"
    annotated_path.parent.mkdir(parents=True)
    annotated_path.write_bytes(b"completed frame marker")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "recover_generated_dataset.py",
            "--image-dir",
            str(image_dir),
            "--dataset-dir",
            str(dataset_dir),
        ],
    )

    recover_script.main()

    records = load_jsonl(dataset_dir / "manifest.jsonl")
    assert len(records) == 1
    assert records[0]["label"] == "rear"
    assert records[0]["mask_pixels"] == 6
    assert records[0]["recovered_metadata"]
    source_records = collect_mixed_source_images(image_dir)
    resumed, _, completed = generator._load_resume_progress(
        image_dir=image_dir,
        out_dir=dataset_dir,
        source_records=source_records,
        recursive=True,
    )
    assert [item["sample_id"] for item in resumed] == [sample_id]
    assert completed == {source_id}


def test_reindex_uses_manually_moved_review_file_and_preserves_source_groups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    reindex = load_reindex_script()
    dataset_dir = tmp_path / "dataset"
    records = []
    for index, label in enumerate(("front", "front", "rear", "rear", "side", "side")):
        sample_id = f"sample_{index}"
        source_id = f"frame_{index // 2}"
        review_path = dataset_dir / "review" / label / f"{sample_id}.png"
        review_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((4, 4, 3), index, dtype=np.uint8)).save(review_path)
        metadata_path = dataset_dir / "samples" / sample_id / "metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text("{}", encoding="utf-8")
        records.append(
            {
                "sample_id": sample_id,
                "source_id": source_id,
                "label": label,
                "initial_label": label,
                "classifier_image": str(review_path.relative_to(dataset_dir)),
                "masked_rgb": f"samples/{sample_id}/masked_rgb.png",
                "metadata": str(metadata_path.relative_to(dataset_dir)),
            }
        )
    from vehicle_orientation.dataset import write_jsonl

    write_jsonl(dataset_dir / "manifest.jsonl", records)
    (dataset_dir / "dataset_manifest.json").write_text("{}", encoding="utf-8")
    moved_path = dataset_dir / "review" / "rear" / "sample_0.png"
    (dataset_dir / "review" / "front" / "sample_0.png").replace(moved_path)
    (dataset_dir / "review" / "side" / "sample_5.png").unlink()
    monkeypatch.setattr(
        sys,
        "argv",
        ["reindex_dataset.py", "--dataset-dir", str(dataset_dir), "--seed", "42"],
    )

    reindex.main()

    updated = load_jsonl(dataset_dir / "manifest.jsonl")
    moved = next(item for item in updated if item["sample_id"] == "sample_0")
    assert moved["label"] == "rear"
    assert moved["classifier_image"] == "review/rear/sample_0.png"
    assert moved["initial_label"] == "front"
    assert len(updated) == 5
    assert all(item["sample_id"] != "sample_5" for item in updated)
    dataset_manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert dataset_manifest["manual_review"]["removed_sample_ids"] == ["sample_5"]
    source_splits: dict[str, set[str]] = {}
    for item in updated:
        source_splits.setdefault(item["source_id"], set()).add(item["split"])
    assert all(len(splits) == 1 for splits in source_splits.values())


def test_reindex_merges_multiple_datasets_with_colliding_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    reindex = load_reindex_script()
    first = tmp_path / "first"
    second = tmp_path / "second"
    merged = tmp_path / "merged"
    _create_review_dataset(first, label="front")
    _create_review_dataset(second, label="rear")
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
            "--seed",
            "42",
        ],
    )

    reindex.main()

    records = load_jsonl(merged / "manifest.jsonl")
    assert len(records) == 2
    assert {item["label"] for item in records} == {"front", "rear"}
    assert len({item["sample_id"] for item in records}) == 2
    assert len({item["source_id"] for item in records}) == 2
    for record in records:
        assert (merged / record["classifier_image"]).is_file()
        assert (merged / record["metadata"]).is_file()
        assert record["merge_origin"]["sample_id"] == "sample_0"


def _create_review_dataset(dataset_dir: Path, *, label: str) -> None:
    from vehicle_orientation.dataset import CLASS_NAMES, write_jsonl

    for class_name in CLASS_NAMES:
        (dataset_dir / "review" / class_name).mkdir(parents=True, exist_ok=True)
    sample_id = "sample_0"
    sample_dir = dataset_dir / "samples" / sample_id
    sample_dir.mkdir(parents=True)
    image = np.full((4, 4, 3), 127, dtype=np.uint8)
    Image.fromarray(image).save(sample_dir / "masked_rgb.png")
    Image.fromarray(image).save(dataset_dir / "review" / label / f"{sample_id}.png")
    record = {
        "sample_id": sample_id,
        "source_id": "frame_0",
        "label": label,
        "initial_label": label,
        "classifier_image": f"review/{label}/{sample_id}.png",
        "masked_rgb": f"samples/{sample_id}/masked_rgb.png",
        "metadata": f"samples/{sample_id}/metadata.json",
    }
    (sample_dir / "metadata.json").write_text(json.dumps(record), encoding="utf-8")
    write_jsonl(dataset_dir / "manifest.jsonl", [record])
    (dataset_dir / "dataset_manifest.json").write_text("{}", encoding="utf-8")


def load_train_script():
    return load_script("vehicle_orientation_train", PROJECT_ROOT / "vehicle_orientation" / "scripts" / "train.py")


def load_build_script():
    return load_script(
        "vehicle_orientation_build_dataset",
        PROJECT_ROOT / "vehicle_orientation" / "scripts" / "build_dataset.py",
    )


def load_reindex_script():
    return load_script(
        "vehicle_orientation_reindex_dataset",
        PROJECT_ROOT / "vehicle_orientation" / "scripts" / "reindex_dataset.py",
    )


def load_efficientnet_generator_script():
    return load_script(
        "vehicle_orientation_generate_dataset_with_efficientnet",
        PROJECT_ROOT / "vehicle_orientation" / "scripts" / "generate_dataset_with_efficientnet.py",
    )


def load_recover_generator_script():
    return load_script(
        "vehicle_orientation_recover_generated_dataset",
        PROJECT_ROOT / "vehicle_orientation" / "scripts" / "recover_generated_dataset.py",
    )


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
