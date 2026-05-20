from pathlib import Path
import importlib.util

import numpy as np


def test_remap_legacy_raw_csv_mask_order():
    module = _load_visualize_point_segmentation()
    height = 3
    width = 4
    legacy = np.arange(height * width, dtype=np.int32)

    remapped = module.remap_legacy_raw_csv_mask_order(legacy, height=height, width=width)

    expected = legacy.reshape(width, height).T.reshape(-1)
    assert np.array_equal(remapped, expected)


def _load_visualize_point_segmentation():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "visualize_point_segmentation.py"
    spec = importlib.util.spec_from_file_location("visualize_point_segmentation", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
