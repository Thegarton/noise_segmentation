import numpy as np

from autolabeler.actors.actor_autolabeler import ActorAutoLabeler
from autolabeler.actors.fixed_box_decoder import load_fixed_actor_sizes
from autolabeler.data.bin_loader import H, W, C
from autolabeler.data.schemas import OrganizedLiDARFrame, SequenceSample


def _synthetic_frame(frame_id: str, x_offset: float = 0.0) -> OrganizedLiDARFrame:
    arr = np.zeros((H, W, C), dtype=np.float32)
    rows = np.arange(82, 96)
    cols = np.arange(220, 250)
    for r in rows:
        for c in cols:
            arr[r, c, 0] = 12.0 + x_offset + (c - cols[0]) * 0.11
            arr[r, c, 1] = -0.7 + (r - rows[0]) * 0.09
            arr[r, c, 2] = 0.25 + ((r - rows[0]) / max(1, len(rows) - 1)) * 1.15
            arr[r, c, 3] = 1.0
    return OrganizedLiDARFrame(frame_id=frame_id, points_range=arr, points_flat=arr.reshape(-1, C))


def test_actor_autolabeler_detects_fixed_size_car():
    sample = SequenceSample(
        current=_synthetic_frame("frame_001", x_offset=0.0),
        past=[_synthetic_frame("frame_000", x_offset=-0.15)],
        future=[_synthetic_frame("frame_002", x_offset=0.15)],
    )

    labels = ActorAutoLabeler().run(sample)

    assert len(labels) == 1
    label = labels[0]
    assert label.semantic_class == "CAR"
    assert label.branch_name == "actor"
    assert label.provenance == "student_predicted"
    assert label.teacher_sources == ["range_pillar_centerpoint_lite"]
    assert label.box_3d is not None
    assert label.box_3d.box_type == "fixed_actor"
    assert label.box_3d.size == [4.2, 1.8, 1.6]
    assert len(label.point_indices) > 0
    assert len(label.range_image_indices) == len(label.point_indices)
    assert label.track_id == 1


def test_fixed_actor_sizes_are_loaded_from_config_as_whl(tmp_path):
    config = tmp_path / "classes.yaml"
    config.write_text(
        """fixed_actor_sizes:
  CAR: [1.8, 1.6, 4.2]
""",
        encoding="utf-8",
    )

    sizes = load_fixed_actor_sizes(str(config))

    assert sizes["CAR"] == [1.8, 1.6, 4.2]
