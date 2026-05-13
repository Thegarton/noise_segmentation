from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

from ..data.schemas import AutoLabelingResult, LabelInstance


CLASS_TO_KITTI_OBJECT_TYPE = {
    "TRUCK_BUS": "truck",
    "CAR": "car",
    "CYCLIST": "cyclist",
    "MOTORCYCLE": "motorcycle",
    "PEDESTRIAN": "pedestrian",
}


@dataclass(frozen=True)
class _PoseRow:
    frame_id: str
    frame_index: int
    label: LabelInstance


def export_kitti_xml(
    output_dir: str,
    results: list[AutoLabelingResult],
    *,
    confidence_log_path: str | None = None,
    actor_only: bool = True,
) -> str:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_ids = [result.frame_id for result in results]
    (out_dir / "frame_list.txt").write_text("\n".join(frame_ids) + ("\n" if frame_ids else ""), encoding="utf-8")

    root, tracklets = _build_xml_tree()
    tracklet_items = _build_tracklet_items(results, actor_only=actor_only)
    for item in tracklet_items:
        tracklets.append(item)

    _text(tracklets, "count", str(len(tracklet_items)), index=0)
    _text(tracklets, "item_version", "1", index=1)

    xml_path = out_dir / "tracklet_labels.xml"
    ET.indent(root, space="  ")
    xml_body = ET.tostring(root, encoding="unicode")
    xml_path.write_text(
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n"
        "<!DOCTYPE boost_serialization>\n"
        f"{xml_body}\n",
        encoding="utf-8",
    )

    _write_confidence_log(confidence_log_path or str(out_dir / "detection_confidence_log.csv"), results, actor_only=actor_only)
    return str(xml_path)


def export_openpcdet_records_as_kitti_xml(
    output_dir: str,
    records: list[dict],
    *,
    confidence_log_path: str | None = None,
) -> str:
    from ..data.schemas import Box3D

    results: list[AutoLabelingResult] = []
    instance_id = 1
    for record in records:
        labels = []
        for pred in record.get("predictions", []):
            box = pred["box_3d"]
            labels.append(
                LabelInstance(
                    frame_id=str(record["frame_id"]),
                    semantic_class=_teacher_class_to_semantic(str(pred["class_name"])),
                    instance_id=instance_id,
                    track_id=instance_id,
                    point_indices=[],
                    range_image_indices=[],
                    box_3d=Box3D(
                        center=[float(x) for x in box["center"]],
                        size=_teacher_size_to_whl([float(x) for x in box["size"]]),
                        yaw=float(box["yaw"]),
                        box_type="fixed_actor",
                    ),
                    mask_confidence=0.0,
                    class_confidence=float(pred["score"]),
                    box_confidence=float(pred["score"]),
                    final_confidence=float(pred["score"]),
                    provenance="lidar_teacher",
                    branch_name="actor",
                    teacher_sources=[str(record.get("source", ""))],
                    review_status="auto_accepted",
                    pseudo_label_version="openpcdet_teacher_export",
                )
            )
            instance_id += 1
        results.append(AutoLabelingResult(frame_id=str(record["frame_id"]), labels=labels))

    return export_kitti_xml(output_dir, results, confidence_log_path=confidence_log_path, actor_only=True)


def _build_xml_tree() -> tuple[ET.Element, ET.Element]:
    root = ET.Element("boost_serialization", {"version": "9", "signature": "serialization::archive"})
    tracklets = ET.SubElement(root, "tracklets", {"version": "0", "tracking_level": "0", "class_id": "0"})
    return root, tracklets


def _build_tracklet_items(results: list[AutoLabelingResult], *, actor_only: bool) -> list[ET.Element]:
    frame_index = {result.frame_id: i for i, result in enumerate(results)}
    rows = [
        _PoseRow(frame_id=result.frame_id, frame_index=frame_index[result.frame_id], label=label)
        for result in results
        for label in result.labels
        if _exportable(label, actor_only=actor_only)
    ]

    items: list[ET.Element] = []
    for row in rows:
        items.append(_single_pose_tracklet(row))
    return items


def _single_pose_tracklet(row: _PoseRow) -> ET.Element:
    label = row.label
    box = label.box_3d
    if box is None:
        raise ValueError("Cannot export KITTI XML label without box_3d")

    item = ET.Element("item", {"version": "1", "tracking_level": "0", "class_id": "1"})
    _text(item, "objectType", _object_type(label.semantic_class))
    _text(item, "h", _fmt(box.size[1]))
    _text(item, "w", _fmt(box.size[0]))
    _text(item, "l", _fmt(box.size[2]))
    _text(item, "first_frame", str(row.frame_index))
    poses = ET.SubElement(item, "poses", {"version": "0", "tracking_level": "0", "class_id": "2"})
    _text(poses, "count", "1")
    _text(poses, "item_version", "0")
    pose = ET.SubElement(poses, "item", {"version": "1", "tracking_level": "0", "class_id": "3"})
    _text(pose, "tx", _fmt(box.center[0]))
    _text(pose, "ty", _fmt(box.center[1]))
    _text(pose, "tz", _fmt(box.center[2]))
    _text(pose, "rx", "0.0")
    _text(pose, "ry", "0.0")
    _text(pose, "rz", _fmt(box.yaw))
    _text(pose, "state", "2")
    _text(pose, "occlusion", "0")
    _text(pose, "occlusion_kf", "1")
    _text(pose, "truncation", "0")
    _text(pose, "amt_occlusion", "-1")
    _text(pose, "amt_border_l", "-1")
    _text(pose, "amt_border_r", "-1")
    _text(pose, "amt_occlusion_kf", "-1")
    _text(pose, "amt_border_kf", "-1")
    _text(item, "finished", "1")
    return item


def _write_confidence_log(path: str, results: list[AutoLabelingResult], *, actor_only: bool) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_id",
                "instance_id",
                "track_id",
                "object_type",
                "final_confidence",
                "class_confidence",
                "box_confidence",
                "mask_confidence",
                "provenance",
                "review_status",
            ],
        )
        writer.writeheader()
        for result in results:
            for label in result.labels:
                if not _exportable(label, actor_only=actor_only):
                    continue
                writer.writerow(
                    {
                        "frame_id": result.frame_id,
                        "instance_id": label.instance_id,
                        "track_id": "" if label.track_id is None else label.track_id,
                        "object_type": _object_type(label.semantic_class),
                        "final_confidence": _fmt(label.final_confidence),
                        "class_confidence": _fmt(label.class_confidence),
                        "box_confidence": _fmt(label.box_confidence),
                        "mask_confidence": _fmt(label.mask_confidence),
                        "provenance": label.provenance,
                        "review_status": label.review_status,
                    }
                )


def _exportable(label: LabelInstance, *, actor_only: bool) -> bool:
    if label.box_3d is None:
        return False
    if actor_only and label.branch_name != "actor":
        return False
    return True


def _object_type(semantic_class: str) -> str:
    return CLASS_TO_KITTI_OBJECT_TYPE.get(semantic_class, semantic_class.lower())


def _teacher_class_to_semantic(class_name: str) -> str:
    mapping = {
        "car": "CAR",
        "truck": "TRUCK_BUS",
        "bus": "TRUCK_BUS",
        "motorcycle": "MOTORCYCLE",
        "bicycle": "CYCLIST",
        "pedestrian": "PEDESTRIAN",
        "cyclist": "CYCLIST",
    }
    return mapping.get(class_name.strip().lower(), class_name.strip().upper())


def _teacher_size_to_whl(size_lwh: list[float]) -> list[float]:
    length, width, height = size_lwh
    return [float(width), float(height), float(length)]


def _text(parent: ET.Element, tag: str, value: str, *, index: int | None = None) -> ET.Element:
    child = ET.Element(tag)
    child.text = value
    if index is None:
        parent.append(child)
    else:
        parent.insert(index, child)
    return child


def _fmt(value: float) -> str:
    return f"{float(value):.8g}"
