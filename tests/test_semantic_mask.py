import json

import numpy as np
import pytest

from autolabeler.data.bin_loader import H, W
from autolabeler.data.schemas import SemanticSegmentationResult
from autolabeler.data.semantic_mask import flatten_mask, load_semantic_mask, save_semantic_mask, unflatten_mask
from autolabeler.export.semantic_mask_exporter import export_semantic_segmentation_result


def test_semantic_mask_flatten_unflatten_preserves_range_order():
    mask = np.arange(H * W, dtype=np.uint16).reshape(H, W)

    flat = flatten_mask(mask)
    restored = unflatten_mask(flat)

    assert flat.shape == (H * W,)
    assert restored.shape == (H, W)
    assert np.array_equal(restored, mask)


def test_semantic_mask_save_load_lossless(tmp_path):
    mask = np.zeros((H, W), dtype=np.uint16)
    mask[10, 20] = 7
    path = tmp_path / "semantic_mask.npy"

    save_semantic_mask(str(path), mask)
    loaded = load_semantic_mask(str(path))

    assert loaded.dtype == np.uint16
    assert np.array_equal(loaded, mask)


def test_semantic_mask_rejects_invalid_shape():
    with pytest.raises(ValueError, match="Semantic mask must have shape"):
        flatten_mask(np.zeros((10, 10), dtype=np.uint16))

    with pytest.raises(ValueError, match="Semantic labels must have shape"):
        unflatten_mask(np.zeros((10,), dtype=np.uint16))


def test_export_semantic_segmentation_result(tmp_path):
    mask = np.zeros((H, W), dtype=np.uint16)
    confidence = np.ones((H, W), dtype=np.float32) * 0.75
    result = SemanticSegmentationResult(
        frame_id="frame_000",
        semantic_mask=mask,
        confidence_mask=confidence,
        pseudo_label_version="litept_v0",
        provenance="student_predicted",
    )

    metadata = export_semantic_segmentation_result(str(tmp_path), result)

    loaded_mask = np.load(metadata["semantic_mask"])
    loaded_confidence = np.load(metadata["confidence_mask"])
    loaded_metadata = json.loads((tmp_path / "frame_000" / "metadata.json").read_text(encoding="utf-8"))

    assert np.array_equal(loaded_mask, mask)
    assert np.allclose(loaded_confidence, confidence)
    assert loaded_metadata["pseudo_label_version"] == "litept_v0"
