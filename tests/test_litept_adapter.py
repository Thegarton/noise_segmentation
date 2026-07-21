from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest

from autolabeler.teachers.litept_adapter import (
    _normalize_state_dict_keys,
    _resolve_checkpoint_path,
    _resolve_litept_config,
    normalize_litept_strength,
    remap_training_predictions,
)


def test_custom_litept_requires_explicit_config(tmp_path: Path):
    with pytest.raises(ValueError, match="--litept-config is required"):
        _resolve_litept_config(tmp_path, "custom", None)


def test_litept_config_resolves_relative_to_root(tmp_path: Path):
    config = tmp_path / "configs" / "custom.py"
    config.parent.mkdir()
    config.write_text("model = dict()\n", encoding="utf-8")

    assert _resolve_litept_config(tmp_path, "custom", "configs/custom.py") == config.resolve()


def test_checkpoint_directory_must_contain_single_checkpoint(tmp_path: Path):
    checkpoint = tmp_path / "model_best.pth"
    checkpoint.write_bytes(b"checkpoint")

    assert _resolve_checkpoint_path(tmp_path) == checkpoint.resolve()


def test_custom_predictions_are_remapped_to_source_taxonomy_ids():
    labels = np.asarray([0, 1, 0, 2], dtype=np.int64)
    assert remap_training_predictions(labels, [2, 10, 42]).tolist() == [2, 10, 2, 42]
    with pytest.raises(ValueError, match="outside"):
        remap_training_predictions(np.asarray([3]), [2, 10, 42])


def test_litept_strength_is_normalized_like_inference():
    values = normalize_litept_strength(np.asarray([0.0, 1.0, 255.0], dtype=np.float32))
    np.testing.assert_allclose(values, np.asarray([0.0, 1.0 / 255.0, 1.0], dtype=np.float32))


def test_state_dict_module_prefix_is_matched_to_model_keys():
    state = OrderedDict([("module.backbone.weight", 1), ("module.seg_head.bias", 2)])
    normalized = _normalize_state_dict_keys(state, {"backbone.weight", "seg_head.bias"})

    assert list(normalized) == ["backbone.weight", "seg_head.bias"]

    state = OrderedDict([("backbone.weight", 1)])
    normalized = _normalize_state_dict_keys(state, {"module.backbone.weight"})

    assert list(normalized) == ["module.backbone.weight"]
