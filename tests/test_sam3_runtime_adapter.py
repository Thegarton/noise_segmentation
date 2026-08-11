from __future__ import annotations

from types import SimpleNamespace

import pytest

from autolabeler.teachers.sam3_runtime_adapter import (
    clear_visual_feature_cache,
    enable_single_image_visual_cache,
    resolve_runtime_precision,
    sam3_runtime_summary,
)


class FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def current_device() -> int:
        return 0

    @staticmethod
    def get_device_capability(device_index: int) -> tuple[int, int]:
        assert device_index == 0
        return 7, 5

    @staticmethod
    def get_device_name(device_index: int) -> str:
        assert device_index == 0
        return "Tesla T4"

    @staticmethod
    def is_bf16_supported() -> bool:
        return False


class FakeTorch:
    cuda = FakeCuda()


def test_auto_precision_selects_fp16_on_t4() -> None:
    precision = resolve_runtime_precision("auto", torch_module=FakeTorch())

    assert precision.effective == "fp16"
    assert precision.compute_capability == (7, 5)
    assert precision.device_name == "Tesla T4"


def test_explicit_bf16_is_rejected_on_t4() -> None:
    with pytest.raises(RuntimeError, match="does not provide native BF16"):
        resolve_runtime_precision("bf16", torch_module=FakeTorch())


def test_single_image_visual_cache_reuses_and_clears_features() -> None:
    calls: list[object] = []

    class Backbone:
        def forward_image(self, samples: object) -> dict[str, object]:
            calls.append(samples)
            return {"features": object()}

    backbone = Backbone()
    predictor = SimpleNamespace(
        model=SimpleNamespace(detector=SimpleNamespace(backbone=backbone)),
        _sam3_cache_visual_features=True,
    )

    enable_single_image_visual_cache(predictor)
    first = backbone.forward_image(object())
    second = backbone.forward_image(object())

    assert first is second
    assert len(calls) == 1
    assert sam3_runtime_summary(predictor)["visual_feature_cache_hits"] == 1

    clear_visual_feature_cache(predictor)
    third = backbone.forward_image(object())

    assert third is not first
    assert len(calls) == 2

