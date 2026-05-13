from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .bin_loader import W
from .box_masking import points_inside_box
from .schemas import Box3D, LabelInstance


OBJECT_TYPE_MAP = {
    "truck": "TRUCK_BUS",
    "bus": "TRUCK_BUS",
    "car": "CAR",
    "cyclist": "CYCLIST",
    "motorcycle": "MOTORCYCLE",
    "pedestrian": "PEDESTRIAN",
}

NOISE_CLASS_NAMES = {
    "crosstalk_noise_1",
    "crosstalk_noise_2",
    "underground_mirror_noise",
    "multiple_range_noise",
    "multipath_noise",
    "multi_machine_interference_noise",
    "exhaust_gas_noise",
    "horizontal_crosstalk_noise",
    "dust_noise",
    "vertical_crosstalk_noise",
    "lane_line_crosstalk_noise",
    "near_range_layered_noise",
    "long_distance_noise",
    "adhesive_noise",
    "unknown_artifact",
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
    return build_manual_labels(mask_dir, branch_name="actor")


def build_manual_labels(mask_dir: str, *, branch_name: str) -> dict[str, list[LabelInstance]]:
    frame_names = load_frame_list(str(Path(mask_dir) / "frame_list.txt"))
    tracklets = load_tracklets_xml(str(Path(mask_dir) / "tracklet_labels.xml"))

    frame_to_labels: dict[str, list[LabelInstance]] = {f: [] for f in frame_names}
    instance_id = 1

    for tr in tracklets:
        semantic_class = _semantic_class(tr.object_type, branch_name)
        box_type = _box_type(branch_name)
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
                box_3d=Box3D(center=[pose.tx, pose.ty, pose.tz], size=[tr.l, tr.w, tr.h], yaw=pose.rz, box_type=box_type),
                mask_confidence=1.0,
                class_confidence=1.0,
                box_confidence=1.0,
                final_confidence=1.0,
                provenance="manual",
                branch_name=branch_name,
                teacher_sources=[],
                review_status="reviewed_accepted",
                pseudo_label_version="manual_seed",
            )
            frame_to_labels[frame_id].append(label)
            instance_id += 1

    return frame_to_labels


def densify_actor_label_masks(labels: list[LabelInstance], points_flat: np.ndarray) -> list[LabelInstance]:
    return densify_label_masks(labels, points_flat)


def densify_label_masks(
    labels: list[LabelInstance],
    points_flat: np.ndarray,
    *,
    allowed_mask: list[bool] | np.ndarray | None = None,
) -> list[LabelInstance]:
    allowed = np.asarray(allowed_mask, dtype=bool) if allowed_mask is not None else None
    for label in labels:
        if label.box_3d is None:
            continue
        point_indices, range_indices, mask_conf = points_inside_box(
            points_flat,
            label.box_3d.center,
            label.box_3d.size,
            label.box_3d.yaw,
        )
        if allowed is not None:
            point_indices = [idx for idx in point_indices if bool(allowed[idx])]
            range_indices = [[int(idx // W), int(idx % W)] for idx in point_indices]
            mask_conf = min(mask_conf, 1.0 if point_indices else 0.0)
        label.point_indices = point_indices
        label.range_image_indices = range_indices
        label.mask_confidence = mask_conf
    return labels


def _semantic_class(object_type: str, branch_name: str) -> str:
    normalized = _normalize_object_type(object_type)
    if branch_name == "actor":
        return OBJECT_TYPE_MAP.get(normalized, normalized.upper())
    if branch_name == "noise":
        return normalized
    return OBJECT_TYPE_MAP.get(normalized, normalized)


def _box_type(branch_name: str) -> str:
    if branch_name == "actor":
        return "fixed_actor"
    if branch_name == "noise":
        return "manual_noise_box"
    return "adaptive_obb"


def _normalize_object_type(object_type: str) -> str:
    return object_type.strip().lower().replace(" ", "_").replace("-", "_")
