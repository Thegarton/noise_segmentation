from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

from .schemas import Box3D, LabelInstance


OBJECT_TYPE_MAP = {
    "truck": "TRUCK_BUS",
    "bus": "TRUCK_BUS",
    "car": "CAR",
    "cyclist": "CYCLIST",
    "motorcycle": "MOTORCYCLE",
    "pedestrian": "PEDESTRIAN",
}


@dataclass(frozen=True)
class KittiTrackPose:
    frame_index: int
    tx: float
    ty: float
    tz: float
    rz: float


@dataclass(frozen=True)
class KittiTracklet:
    track_id: int
    object_type: str
    h: float
    w: float
    l: float
    first_frame: int
    poses: list[KittiTrackPose]


def load_frame_list(frame_list_path: str) -> list[str]:
    lines = Path(frame_list_path).read_text(encoding="utf-8").splitlines()
    return [x.strip() for x in lines if x.strip()]


def _safe_float(item: ET.Element, tag: str, default: float = 0.0) -> float:
    node = item.find(tag)
    if node is None or node.text is None:
        return default
    return float(node.text)


def _safe_int(item: ET.Element, tag: str, default: int = 0) -> int:
    node = item.find(tag)
    if node is None or node.text is None:
        return default
    return int(float(node.text))


def load_tracklets_xml(xml_path: str) -> list[KittiTracklet]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    tracklets_root = root.find("tracklets")
    if tracklets_root is None:
        raise ValueError("No <tracklets> section found in XML")

    out: list[KittiTracklet] = []
    tid = 0
    for item in tracklets_root.findall("item"):
        obj_type_node = item.find("objectType")
        if obj_type_node is None or obj_type_node.text is None:
            continue

        object_type = obj_type_node.text.strip().lower()
        h = _safe_float(item, "h")
        w = _safe_float(item, "w")
        l = _safe_float(item, "l")
        first_frame = _safe_int(item, "first_frame", 0)

        poses_container = item.find("poses")
        poses: list[KittiTrackPose] = []
        if poses_container is not None:
            for pose_i, p in enumerate(poses_container.findall("item")):
                poses.append(
                    KittiTrackPose(
                        frame_index=first_frame + pose_i,
                        tx=_safe_float(p, "tx"),
                        ty=_safe_float(p, "ty"),
                        tz=_safe_float(p, "tz"),
                        rz=_safe_float(p, "rz"),
                    )
                )

        out.append(KittiTracklet(track_id=tid, object_type=object_type, h=h, w=w, l=l, first_frame=first_frame, poses=poses))
        tid += 1

    return out


def build_manual_actor_labels(mask_dir: str) -> dict[str, list[LabelInstance]]:
    frame_names = load_frame_list(str(Path(mask_dir) / "frame_list.txt"))
    tracklets = load_tracklets_xml(str(Path(mask_dir) / "tracklet_labels.xml"))

    frame_to_labels: dict[str, list[LabelInstance]] = {f: [] for f in frame_names}
    instance_id = 1

    for tr in tracklets:
        semantic_class = OBJECT_TYPE_MAP.get(tr.object_type, tr.object_type.upper())
        for pose in tr.poses:
            if pose.frame_index < 0 or pose.frame_index >= len(frame_names):
                continue
            frame_id = frame_names[pose.frame_index]
            label = LabelInstance(
                frame_id=frame_id,
                semantic_class=semantic_class,
                instance_id=instance_id,
                track_id=tr.track_id,
                point_indices=[],
                range_image_indices=[],
                box_3d=Box3D(center=[pose.tx, pose.ty, pose.tz], size=[tr.l, tr.w, tr.h], yaw=pose.rz, box_type="fixed_actor"),
                mask_confidence=1.0,
                class_confidence=1.0,
                box_confidence=1.0,
                final_confidence=1.0,
                provenance="manual",
                branch_name="actor",
                teacher_sources=[],
                review_status="reviewed_accepted",
                pseudo_label_version="manual_seed",
            )
            frame_to_labels[frame_id].append(label)
            instance_id += 1

    return frame_to_labels
