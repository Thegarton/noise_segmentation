from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from autolabeler.hl320.csv_points import (
    HL320_FEATURE_NAMES,
    build_hl320_features,
    group_echo_returns,
    load_hl320_csv,
    primary_returns_frame,
)
from autolabeler.hl320.dataset import build_hl320_dataset, class_aware_split


def test_hl320_csv_preserves_row_order_and_groups_echoes(tmp_path: Path):
    csv_path = tmp_path / "000001.csv"
    csv_path.write_text(
        "x\ty\tz\tazimuth\tvertical\tintensity\treflectivity\tslot\tpixel\tblockID\tCxd\tCyd\n"
        "1\t0\t0\t10\t1\t20\t30\t2\t7\t1\t100.1\t200.2\n"
        "3\t0\t0\t10\t1\t80\t90\t2\t7\t0\t100.1\t200.2\n"
        "0\t2\t0\t11\t2\t10\t11\t2\t8\t0\t101.0\t201.0\n",
        encoding="utf-8",
    )

    frame = load_hl320_csv(csv_path)
    groups = group_echo_returns(frame)
    features = build_hl320_features(frame)

    assert frame.frame_id == "000001"
    np.testing.assert_allclose(frame.points[:, :3], np.asarray([[1, 0, 0], [3, 0, 0], [0, 2, 0]], dtype=np.float32))
    assert groups.row_indices.shape == (2, 2)
    assert groups.return_count_by_row.tolist() == [2, 2, 1]
    assert groups.echo_rank_by_distance.tolist() == [0, 1, 0]
    assert features.shape == (3, len(HL320_FEATURE_NAMES))
    assert features[0, HL320_FEATURE_NAMES.index("has_camera_projection")] == 1.0


def test_zero_xyz_echo_rows_are_not_counted_as_real_returns(tmp_path: Path):
    csv_path = tmp_path / "000002.csv"
    csv_path.write_text(
        "x y z azimuth vertical intensity reflectivity slot pixel blockID Cxd Cyd\n"
        "0.008 0.800 0.634 89.35 38.38 0 3 2 0 0 294.9 173.2\n"
        "0.000 0.000 0.000 89.35 38.38 0 0 2 0 1 294.9 173.2\n"
        "0.000 0.000 0.000 89.35 38.38 0 0 2 0 2 294.9 173.2\n",
        encoding="utf-8",
    )

    echo_frame = load_hl320_csv(csv_path)
    primary = primary_returns_frame(echo_frame)
    groups = group_echo_returns(echo_frame)
    features = build_hl320_features(primary, echo_frame=echo_frame)

    assert groups.return_count_by_row.tolist() == [1, 0, 0]
    assert features.shape == (1, len(HL320_FEATURE_NAMES))
    assert features[0, HL320_FEATURE_NAMES.index("return_count")] == 1.0
    assert features[0, HL320_FEATURE_NAMES.index("nearest_echo_distance_delta")] == 0.0
    assert features[0, HL320_FEATURE_NAMES.index("strongest_echo_intensity_delta")] == 0.0


def test_class_aware_split_keeps_single_frame_class_in_train():
    train, val = class_aware_split(
        [
            ("000000", {2: 10}),
            ("000001", {5: 20}),
            ("000002", {5: 5}),
            ("000003", {2: 4}),
        ],
        val_ratio=0.5,
        seed=3,
    )

    assert train
    assert val
    assert {"000000", "000003"} & set(train)
    assert {"000001", "000002"} & set(train)


def test_build_hl320_dataset_writes_coord_features_segment_and_manifest(tmp_path: Path):
    csv_dir = tmp_path / "csv"
    labels_dir = tmp_path / "labels"
    classes_yaml = tmp_path / "classes.yaml"
    csv_dir.mkdir()
    labels_dir.mkdir()
    classes_yaml.write_text(
        "semantic_classes:\n"
        "  background: 0\n"
        "  CAR: 2\n"
        "  PEDESTRIAN: 5\n"
        "  ignore: 255\n",
        encoding="utf-8",
    )
    for frame_id, class_id in (("000000", 2), ("000001", 5), ("000002", 5)):
        (csv_dir / f"{frame_id}.csv").write_text(
            "x y z intensity slot pixel blockID Cxd Cyd\n"
            "0.1 0 0 0 2 0 0 10 20\n"
            f"1 2 3 4 2 1 0 11 21\n"
            f"2 3 4 8 2 1 1 11 21\n",
            encoding="utf-8",
        )
        frame_label_dir = labels_dir / frame_id
        frame_label_dir.mkdir()
        np.save(frame_label_dir / "semantic_mask.npy", np.asarray([class_id, class_id], dtype=np.uint16))

    result = build_hl320_dataset(
        csv_dir=csv_dir,
        labels_dir=labels_dir,
        output_dir=tmp_path / "out",
        classes_yaml=classes_yaml,
        val_ratio=0.34,
        seed=1,
        overwrite=True,
    )

    assert result.frame_count == 3
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["format"] == "hl320_pointwise_v1"
    assert manifest["feature_names"] == HL320_FEATURE_NAMES
    assert manifest["taxonomy"]["training_id_to_source_id"] == [2, 5]
    assert manifest["litept_in_channels"] == 3 + len(HL320_FEATURE_NAMES)
    written_frames = list((result.output_dir / "dataset").glob("*/*"))
    assert len(written_frames) == 3
    for frame_dir in written_frames:
        assert np.load(frame_dir / "coord.npy").shape == (2, 3)
        assert np.load(frame_dir / "features.npy").shape == (2, len(HL320_FEATURE_NAMES))
        assert np.load(frame_dir / "strength.npy").shape == (2, len(HL320_FEATURE_NAMES))
        segment = np.load(frame_dir / "segment.npy")
        assert segment.shape == (2,)
        assert set(segment.tolist()) <= {0, 1}
        assert (frame_dir / "metadata.json").is_file()
