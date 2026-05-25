from pathlib import Path

from autolabeler.data.ins_pose import load_ins_pose_index, write_kitti_pose


def test_load_ins_pose_index_matches_nearest_timestamp_and_writes_kitti_pose(tmp_path: Path):
    ins_path = tmp_path / "ins"
    ins_path.write_text(
        "\t".join(
            [
                "secs",
                "nsecs",
                "attitude_X",
                "attitude_Y",
                "attitude_Z",
                "latitude",
                "longitude",
                "elevation",
                "utmPosition_X",
                "utmPosition_Y",
                "utmPosition_Z",
            ]
        )
        + "\n"
        + "\t".join(["10", "100000000", "0", "0", "0", "55.0", "37.0", "150.0", "1", "2", "3"])
        + "\n"
        + "\t".join(["10", "300000000", "0", "0", "1.57079632679", "55.0", "37.0", "150.0", "4", "5", "6"])
        + "\n",
        encoding="utf-8",
    )

    index = load_ins_pose_index(str(ins_path))
    pose = index.nearest(10_260_000)

    assert pose.source_timestamp_us == 10_300_000
    assert pose.delta_us == 40_000
    assert pose.translation == [4.0, 5.0, 6.0]
    assert len(pose.kitti_pose) == 12

    pose_path = tmp_path / "pose.txt"
    write_kitti_pose(str(pose_path), pose)

    values = pose_path.read_text(encoding="utf-8").strip().split()
    assert len(values) == 12


def test_ins_pose_uses_local_enu_when_utm_is_zero(tmp_path: Path):
    ins_path = tmp_path / "ins"
    ins_path.write_text(
        "secs\tnsecs\tattitude_X\tattitude_Y\tattitude_Z\tlatitude\tlongitude\televation\tutmPosition_X\tutmPosition_Y\tutmPosition_Z\n"
        "10\t0\t0\t0\t0\t55.0\t37.0\t150.0\t0\t0\t0\n"
        "10\t100000000\t0\t0\t0\t55.00001\t37.00001\t151.0\t0\t0\t0\n",
        encoding="utf-8",
    )

    pose = load_ins_pose_index(str(ins_path)).nearest(10_100_000)

    assert abs(pose.translation[0]) > 0.1
    assert abs(pose.translation[1]) > 0.1
    assert pose.translation[2] == 1.0
