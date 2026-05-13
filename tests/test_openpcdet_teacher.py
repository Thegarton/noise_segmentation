import json
from pathlib import Path

import numpy as np

from autolabeler.actors.actor_autolabeler import ActorAutoLabeler
from autolabeler.actors.openpcdet_teacher import OpenPCDetPredictionStore, remap_openpcdet_class
from autolabeler.data.bin_loader import H, W, C
from autolabeler.data.kitti_mask_loader import build_manual_actor_labels
from autolabeler.export.kitti_xml_exporter import export_openpcdet_records_as_kitti_xml
from autolabeler.data.schemas import OrganizedLiDARFrame, SequenceSample
from autolabeler.teachers.openpcdet_adapter import _match_point_feature_dim, prepare_openpcdet_points


def test_openpcdet_class_remap():
    assert remap_openpcdet_class("car") == "CAR"
    assert remap_openpcdet_class("truck") == "TRUCK_BUS"
    assert remap_openpcdet_class("bus") == "TRUCK_BUS"
    assert remap_openpcdet_class("bicycle") == "CYCLIST"
    assert remap_openpcdet_class("traffic_cone") is None


def test_prepare_openpcdet_points_from_csv(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    csv_path = data_dir / "frame_000.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("x,y,z,intensity\n")
        for i in range(H * W):
            if i == 10:
                f.write("1.0,2.0,3.0,0.5\n")
            else:
                f.write("0.0,0.0,0.0,0.0\n")

    prepared = prepare_openpcdet_points(str(data_dir), str(tmp_path / "openpcdet"), input_format="csv")

    assert len(prepared) == 1
    points = np.load(prepared[0].points_path)
    assert points.shape == (1, 4)
    assert np.allclose(points[0], [1.0, 2.0, 3.0, 0.5])
    assert Path(prepared[0].points_path).is_absolute()
    assert Path(prepared[0].source_path).is_absolute()


def test_match_point_feature_dim_pads_nuscenes_timestamp_feature():
    points = np.asarray([[1.0, 2.0, 3.0, 0.5]], dtype=np.float32)
    dataset_cfg = {"POINT_FEATURE_ENCODING": {"src_feature_list": ["x", "y", "z", "intensity", "timestamp"]}}

    matched = _match_point_feature_dim(points, dataset_cfg)

    assert matched.shape == (1, 5)
    assert np.allclose(matched[0], [1.0, 2.0, 3.0, 0.5, 0.0])


def test_match_point_feature_dim_truncates_extra_features():
    points = np.asarray([[1.0, 2.0, 3.0, 0.5, 0.1, 0.2]], dtype=np.float32)
    dataset_cfg = {"POINT_FEATURE_ENCODING": {"src_feature_list": ["x", "y", "z", "intensity"]}}

    matched = _match_point_feature_dim(points, dataset_cfg)

    assert matched.shape == (1, 4)
    assert np.allclose(matched[0], [1.0, 2.0, 3.0, 0.5])


def test_actor_autolabeler_consumes_openpcdet_predictions(tmp_path: Path):
    arr = np.zeros((H, W, C), dtype=np.float32)
    arr[90:96, 230:238, 0] = 12.0
    arr[90:96, 230:238, 1] = 0.2
    arr[90:96, 230:238, 2] = 0.6
    arr[90:96, 230:238, 3] = 1.0
    frame = OrganizedLiDARFrame(frame_id="frame_000", points_range=arr, points_flat=arr.reshape(-1, C))

    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps(
            {
                "frame_id": "frame_000",
                "source": "openpcdet_centerpoint_pointpillar_nuscenes",
                "predictions": [
                    {
                        "class_name": "truck",
                        "score": 0.9,
                        "box_3d": {
                            "center": [12.0, 0.2, 1.6],
                            "size": [8.0, 2.5, 3.0],
                            "yaw": 0.1,
                        },
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    store = OpenPCDetPredictionStore.from_jsonl(str(predictions_path))
    assert store.get("frame_000")[0].semantic_class == "TRUCK_BUS"

    labels = ActorAutoLabeler(openpcdet_predictions_path=str(predictions_path), use_heuristic_fallback=False).run(
        SequenceSample(current=frame)
    )

    assert len(labels) == 1
    label = labels[0]
    assert label.semantic_class == "TRUCK_BUS"
    assert label.provenance == "lidar_teacher"
    assert label.teacher_sources == ["openpcdet_centerpoint_pointpillar_nuscenes"]
    assert label.box_3d is not None
    assert label.box_3d.size == [10.0, 2.6, 3.2]


def test_teacher_predictions_export_to_kitti_tracklets_and_confidence_log(tmp_path: Path):
    records = [
        {
            "frame_id": "frame_000",
            "source_path": "/data/frame_000.csv",
            "source": "openpcdet_centerpoint_pointpillar_nuscenes",
            "predictions": [
                {
                    "class_name": "car",
                    "score": 0.91,
                    "box_3d": {
                        "center": [10.0, 2.0, 0.3],
                        "size": [4.2, 1.8, 1.6],
                        "yaw": 0.2,
                    },
                }
            ],
        }
    ]

    out_dir = tmp_path / "openpcdet_kitti_mask"
    export_openpcdet_records_as_kitti_xml(str(out_dir), records)

    labels = build_manual_actor_labels(str(out_dir))
    assert (out_dir / "frame_list.txt").read_text(encoding="utf-8") == "frame_000\n"
    assert (out_dir / "detection_confidence_log.csv").exists()
    assert len(labels["frame_000"]) == 1
    label = labels["frame_000"][0]
    assert label.semantic_class == "CAR"
    assert label.box_3d is not None
    assert label.box_3d.center == [10.0, 2.0, 0.3]
    assert label.box_3d.size == [1.8, 1.6, 4.2]

    log_text = (out_dir / "detection_confidence_log.csv").read_text(encoding="utf-8")
    assert "frame_000,1,1,car,0.91,0.91,0.91" in log_text
