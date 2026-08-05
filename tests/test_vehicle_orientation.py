from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SRC = PROJECT_ROOT / "vehicle_orientation" / "src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from vehicle_orientation.dataset import (  # noqa: E402
    assign_stratified_splits,
    collect_source_images,
    parse_folder_mappings,
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


def test_stratified_split_never_separates_crops_from_one_image():
    samples = []
    for label in ("front", "rear", "other"):
        for source_index in range(10):
            for crop_index in range(2):
                samples.append(
                    {
                        "sample_id": f"{label}-{source_index}-{crop_index}",
                        "source_id": f"{label}-{source_index}",
                        "label": label,
                    }
                )

    split = assign_stratified_splits(samples, seed=42)
    source_splits = {}
    for item in split:
        source_splits.setdefault(item["source_id"], set()).add(item["split"])

    assert all(len(values) == 1 for values in source_splits.values())
    assert {item["split"] for item in split} == {"train", "val", "test"}
    assert assign_stratified_splits(samples, seed=42) == split


def test_folder_mapping_supports_existing_rear_folder_name(tmp_path: Path):
    for folder in ("front", "backlights_and_licence_plate", "other"):
        path = tmp_path / folder
        path.mkdir()
        (path / "frame.jpg").write_bytes(b"not-decoded-in-this-test")
    mapping = parse_folder_mappings(["rear=backlights_and_licence_plate"])

    records = collect_source_images(tmp_path, mapping)

    assert {record["label"] for record in records} == {"front", "rear", "other"}
    rear = next(record for record in records if record["label"] == "rear")
    assert rear["source_relative_path"] == "backlights_and_licence_plate/frame.jpg"


def test_orientation_policy_maps_other_and_uncertain_to_front():
    rear = decide_orientation([0.05, 0.90, 0.05], min_confidence=0.7, min_margin=0.1)
    other = decide_orientation([0.10, 0.20, 0.70], min_confidence=0.7, min_margin=0.1)
    uncertain = decide_orientation([0.45, 0.40, 0.15], min_confidence=0.7, min_margin=0.1)

    assert rear.semantic_label == "rear" and not rear.fallback
    assert other.predicted_class == "other" and other.semantic_label == "front" and other.fallback
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
    assert loaded["class_names"] == ("front", "rear", "other")
    assert loaded["model_state_dict"].keys() == model.state_dict().keys()


def test_dataset_builder_smoke_with_fake_sam3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    builder = load_build_script()
    source_root = tmp_path / "sources"
    out_dir = tmp_path / "dataset"
    for label in ("front", "rear", "other"):
        class_dir = source_root / label
        class_dir.mkdir(parents=True)
        for index in range(3):
            (class_dir / f"{index:06d}.jpg").write_bytes(b"fake")

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
            mask = np.zeros((10, 12), dtype=bool)
            mask[2:8, 3:10] = True
            return [SimpleNamespace(prompt=prompts[0], score=0.9, mask=mask, box=None)]

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

    manifest = (out_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    summary = json.loads((out_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest) == 9
    assert summary["summary"]["class_counts"] == {"front": 3, "other": 3, "rear": 3}
    assert summary["summary"]["split_counts"] == {"test": 3, "train": 3, "val": 3}
    assert len(list((out_dir / "samples").glob("*/*/rgb.png"))) == 9


def load_train_script():
    script_path = PROJECT_ROOT / "vehicle_orientation" / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("vehicle_orientation_train", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_build_script():
    script_path = PROJECT_ROOT / "vehicle_orientation" / "scripts" / "build_dataset.py"
    spec = importlib.util.spec_from_file_location("vehicle_orientation_build_dataset", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
