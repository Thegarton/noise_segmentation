from pathlib import Path

from autolabeler.data.kitti_mask_loader import build_manual_actor_labels, build_manual_labels
from autolabeler.data.schemas import AutoLabelingResult, Box3D, LabelInstance
from autolabeler.export.kitti_xml_exporter import export_kitti_xml


def test_export_results_to_kitti_xml_roundtrip(tmp_path: Path):
    results = [
        AutoLabelingResult(
            frame_id="frame_000",
            labels=[
                LabelInstance(
                    frame_id="frame_000",
                    semantic_class="CAR",
                    instance_id=7,
                    track_id=3,
                    point_indices=[1, 2],
                    range_image_indices=[[0, 1], [0, 2]],
                    box_3d=Box3D(center=[10.0, 2.0, 0.3], size=[4.2, 1.8, 1.6], yaw=0.25, box_type="fixed_actor"),
                    mask_confidence=0.8,
                    class_confidence=0.9,
                    box_confidence=0.85,
                    final_confidence=0.88,
                    provenance="lidar_teacher",
                    branch_name="actor",
                    teacher_sources=["test_teacher"],
                    review_status="auto_accepted",
                    pseudo_label_version="test",
                )
            ],
        )
    ]

    out_dir = tmp_path / "kitti_mask"
    export_kitti_xml(str(out_dir), results)

    by_frame = build_manual_actor_labels(str(out_dir))
    assert (out_dir / "frame_list.txt").read_text(encoding="utf-8") == "frame_000\n"
    assert (out_dir / "detection_confidence_log.csv").exists()
    assert len(by_frame["frame_000"]) == 1
    label = by_frame["frame_000"][0]
    assert label.semantic_class == "CAR"
    assert label.box_3d is not None
    assert label.box_3d.center == [10.0, 2.0, 0.3]
    assert label.box_3d.size == [4.2, 1.8, 1.6]
    assert label.box_3d.yaw == 0.25


def test_export_noise_label_to_kitti_xml_roundtrip(tmp_path: Path):
    results = [
        AutoLabelingResult(
            frame_id="frame_000",
            labels=[
                LabelInstance(
                    frame_id="frame_000",
                    semantic_class="dust_noise",
                    instance_id=11,
                    track_id=None,
                    point_indices=[7, 8],
                    range_image_indices=[[0, 7], [0, 8]],
                    box_3d=Box3D(center=[5.0, 0.0, 0.2], size=[1.5, 0.8, 0.6], yaw=0.0, box_type="noise_adaptive_aabb"),
                    mask_confidence=0.7,
                    class_confidence=0.8,
                    box_confidence=0.75,
                    final_confidence=0.76,
                    provenance="rule_labeled",
                    branch_name="noise",
                    teacher_sources=["range_view_noise_rules_v0"],
                    review_status="auto_accepted",
                    pseudo_label_version="test",
                )
            ],
        )
    ]

    out_dir = tmp_path / "kitti_noise"
    export_kitti_xml(str(out_dir), results, actor_only=False)

    by_frame = build_manual_labels(str(out_dir), branch_name="noise")
    assert len(by_frame["frame_000"]) == 1
    label = by_frame["frame_000"][0]
    assert label.semantic_class == "dust_noise"
    assert label.box_3d is not None
    assert label.box_3d.center == [5.0, 0.0, 0.2]

    log_text = (out_dir / "detection_confidence_log.csv").read_text(encoding="utf-8")
    assert "frame_id,instance_id,track_id,object_type,final_confidence,class_confidence,box_confidence,mask_confidence" in log_text
    assert "frame_000,11,,dust_noise,0.76,0.8,0.75,0.7" in log_text
