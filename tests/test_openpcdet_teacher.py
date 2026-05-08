import json
from pathlib import Path

import numpy as np

from autolabeler.actors.actor_autolabeler import ActorAutoLabeler
from autolabeler.actors.openpcdet_teacher import OpenPCDetPredictionStore, remap_openpcdet_class
from autolabeler.data.bin_loader import H, W, C
from autolabeler.data.schemas import OrganizedLiDARFrame, SequenceSample
from autolabeler.teachers.openpcdet_adapter import prepare_openpcdet_points


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
