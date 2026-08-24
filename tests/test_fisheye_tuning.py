from __future__ import annotations

import numpy as np
import pytest

from autolabeler.camera.fisheye_tuning import (
    build_tunable_fisheye_remap,
    crop_and_stitch_three_views,
    rotation_matrix,
)


def default_remap(**overrides):
    parameters = {
        "input_shape": (1001, 1001),
        "output_shape": (101, 201),
        "source_cx": 500.0,
        "source_cy": 500.0,
        "source_radius_x": 500.0,
        "source_radius_y": 500.0,
        "source_fov": 200.0,
        "fisheye_model": "linear",
        "projection": "cylindrical",
        "horizontal_fov": 150.0,
        "vertical_fov": 90.0,
    }
    parameters.update(overrides)
    return build_tunable_fisheye_remap(**parameters)


def test_unrotated_output_center_maps_to_fisheye_center():
    remap = default_remap()

    assert remap.map_x[50, 100] == pytest.approx(500.0)
    assert remap.map_y[50, 100] == pytest.approx(500.0)
    assert remap.valid[50, 100]


def test_cylindrical_horizontal_edges_have_expected_source_order():
    remap = default_remap(horizontal_fov=120.0)

    center_y = remap.map_y.shape[0] // 2
    assert remap.map_x[center_y, 0] < remap.map_x[center_y, 100]
    assert remap.map_x[center_y, -1] > remap.map_x[center_y, 100]


def test_extrinsic_yaw_moves_virtual_center_in_source_image():
    left = default_remap(yaw=-30.0)
    center = default_remap(yaw=0.0)
    right = default_remap(yaw=30.0)

    index = (50, 100)
    assert left.map_x[index] < center.map_x[index] < right.map_x[index]


def test_roll_rotation_matrix_is_orthonormal():
    matrix = rotation_matrix(yaw=17.0, pitch=-12.0, roll=4.0)

    assert matrix @ matrix.T == pytest.approx(np.eye(3), abs=1e-12)
    assert np.linalg.det(matrix) == pytest.approx(1.0)


def test_radial_polynomial_changes_mapping_away_from_center():
    base = default_remap(k1=0.0)
    distorted = default_remap(k1=0.2)
    index = (50, 150)

    assert distorted.map_x[index] > base.map_x[index]
    assert distorted.map_y[50, 100] == pytest.approx(base.map_y[50, 100])


def test_crop_and_stitch_removes_two_overlap_regions():
    views = tuple(np.full((4, 10, 3), value, dtype=np.uint8) for value in (1, 2, 3))

    stitched = crop_and_stitch_three_views(views, overlap_pixels=4)

    assert stitched.shape == (4, 22, 3)
    assert np.all(stitched[:, :8] == 1)
    assert np.all(stitched[:, 8:14] == 2)
    assert np.all(stitched[:, 14:] == 3)
