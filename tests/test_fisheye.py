from __future__ import annotations

import numpy as np
import pytest

from autolabeler.camera.fisheye import build_fisheye_remap, fisheye_radius_fraction, resolve_output_fov


def test_perspective_remap_preserves_center_and_requested_shape():
    remap = build_fisheye_remap(
        input_shape=(1536, 1920),
        output_shape=(1081, 1921),
        xcenter=960.0,
        ycenter=768.0,
        radius=760.0,
        fov=180.0,
        pfov=110.0,
    )

    assert remap.map_x.shape == (1081, 1921)
    assert remap.map_y.shape == (1081, 1921)
    assert remap.map_x[540, 960] == pytest.approx(960.0)
    assert remap.map_y[540, 960] == pytest.approx(768.0)
    assert remap.horizontal_fov == pytest.approx(110.0)
    assert 0.0 < remap.vertical_fov < remap.horizontal_fov


def test_output_aspect_ratio_changes_vertical_fov_without_stretching_pixels():
    horizontal, vertical, focal = resolve_output_fov(
        output_shape=(2160, 3840),
        pfov=120.0,
        pfov_axis="horizontal",
    )

    expected_vertical = np.rad2deg(2.0 * np.arctan(((2160 - 1) / 2.0) / focal))
    assert horizontal == pytest.approx(120.0)
    assert vertical == pytest.approx(expected_vertical)
    assert vertical < horizontal


def test_all_supported_fisheye_models_map_edge_to_source_radius():
    theta_max = np.deg2rad(90.0)
    theta = np.asarray([0.0, theta_max])

    for dtype in ("linear", "equalarea", "orthographic", "stereographic"):
        fraction = fisheye_radius_fraction(theta, theta_max=theta_max, dtype=dtype)
        assert fraction[0] == pytest.approx(0.0)
        assert fraction[1] == pytest.approx(1.0)


def test_cylindrical_remap_is_finite_for_wide_output():
    remap = build_fisheye_remap(
        input_shape=(1536, 1920),
        output_shape=(1080, 3840),
        xcenter=960.0,
        ycenter=768.0,
        radius=760.0,
        fov=200.0,
        pfov=150.0,
        projection="cylindrical",
    )

    assert np.isfinite(remap.map_x).all()
    assert np.isfinite(remap.map_y).all()
    assert np.count_nonzero(remap.valid) > 0


def test_yaw_moves_output_center_inside_source_circle():
    remap = build_fisheye_remap(
        input_shape=(1001, 1001),
        output_shape=(101, 101),
        xcenter=500.0,
        ycenter=500.0,
        radius=500.0,
        fov=180.0,
        pfov=60.0,
        yaw=30.0,
    )

    assert remap.map_x[50, 50] > 500.0
    assert remap.map_y[50, 50] == pytest.approx(500.0)
