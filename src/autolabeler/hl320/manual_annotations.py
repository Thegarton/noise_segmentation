from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


IGNORE_ID = 255
DEFAULT_OBJECT_FIELDS = (
    "objectType",
    "obj_type",
    "label",
    "name",
    "category",
    "class",
    "type",
)
DEFAULT_ANNOTATION_SUFFIXES = (".json", ".txt")
FRAME_DIRECTORY_LABEL_FILES = (
    "instances.json",
    "instances.txt",
    "labels.json",
    "labels.txt",
    "annotations.json",
    "annotations.txt",
)


@dataclass(frozen=True)
class ManualAnnotationObject:
    object_index: int
    raw_label: str
    target_name: str | None
    target_id: int
    indices: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ManualFrameLabels:
    frame_id: str
    annotation_path: Path
    point_count: int
    labels: np.ndarray
    objects: list[ManualAnnotationObject]
    stats: dict[str, Any]


def load_manual_annotation_payload(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Manual annotation file does not exist: {source}")
    text = source.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = _load_json_lines(text, source=source)

    objects = _extract_annotation_objects(payload)
    return [obj for obj in objects if isinstance(obj, dict)]


def build_manual_label_array(
    *,
    annotation_path: str | Path,
    frame_id: str,
    point_count: int,
    class_to_id: dict[str, int],
    class_aliases: dict[str, str | int] | None = None,
    ignore_id: int = IGNORE_ID,
    default_label_id: int | None = None,
    index_base: int = 0,
    conflict_policy: str = "priority",
    unknown_policy: str = "ignore",
    out_of_range_policy: str = "error",
) -> ManualFrameLabels:
    if index_base not in {0, 1}:
        raise ValueError(f"index_base must be 0 or 1, got {index_base}")
    if conflict_policy not in {"priority", "last", "first", "error"}:
        raise ValueError(f"Unsupported conflict policy: {conflict_policy!r}")
    if unknown_policy not in {"ignore", "error"}:
        raise ValueError(f"Unsupported unknown policy: {unknown_policy!r}")
    if out_of_range_policy not in {"ignore", "error"}:
        raise ValueError(f"Unsupported out-of-range policy: {out_of_range_policy!r}")
    if point_count < 0:
        raise ValueError(f"point_count must be non-negative, got {point_count}")

    annotation_source = Path(annotation_path).expanduser().resolve()
    default_id = int(ignore_id if default_label_id is None else default_label_id)
    labels = np.full((point_count,), default_id, dtype=np.int32)
    priorities = np.full((point_count,), _label_priority(default_id, _name_for_id(class_to_id, default_id)), dtype=np.int32)
    payload_objects = load_manual_annotation_payload(annotation_source)
    resolver = ClassResolver(class_to_id=class_to_id, class_aliases=class_aliases or {}, ignore_id=ignore_id)
    parsed_objects: list[ManualAnnotationObject] = []
    unknown_objects: list[dict[str, Any]] = []
    out_of_range_indices: list[int] = []
    duplicate_assignments = 0
    assigned_points = 0

    for object_index, raw_object in enumerate(payload_objects):
        raw_label = read_object_label(raw_object)
        resolved = resolver.resolve(raw_label, raw_object=raw_object)
        raw_indices = parse_indices(raw_object)
        if index_base == 1 and raw_indices.size:
            raw_indices = raw_indices - 1

        valid_range = (raw_indices >= 0) & (raw_indices < point_count)
        if not np.all(valid_range):
            invalid_values = raw_indices[~valid_range].astype(np.int64).tolist()
            out_of_range_indices.extend(invalid_values[:100])
            if out_of_range_policy == "error":
                raise ValueError(
                    f"{annotation_source}: object #{object_index} label={raw_label!r} has "
                    f"{len(invalid_values)} indices outside [0, {point_count - 1}], examples={invalid_values[:10]}"
                )
            raw_indices = raw_indices[valid_range]

        if resolved is None:
            unknown_objects.append(
                {
                    "object_index": object_index,
                    "raw_label": raw_label,
                    "indices": int(raw_indices.size),
                }
            )
            if unknown_policy == "error":
                raise ValueError(f"{annotation_source}: cannot map object label {raw_label!r} to classes YAML")
            target_name = None
            target_id = int(ignore_id)
        else:
            target_name, target_id = resolved

        unique_indices = np.unique(raw_indices.astype(np.int64, copy=False))
        metadata = summarize_object_metadata(raw_object)
        parsed = ManualAnnotationObject(
            object_index=object_index,
            raw_label=raw_label,
            target_name=target_name,
            target_id=int(target_id),
            indices=unique_indices,
            metadata=metadata,
        )
        parsed_objects.append(parsed)

        if unique_indices.size == 0:
            continue
        already_assigned = labels[unique_indices] != default_id
        duplicate_assignments += int(np.count_nonzero(already_assigned))
        should_assign = choose_assignment_mask(
            current_labels=labels[unique_indices],
            current_priorities=priorities[unique_indices],
            target_id=int(target_id),
            target_name=target_name,
            conflict_policy=conflict_policy,
            class_to_id=class_to_id,
        )
        labels[unique_indices[should_assign]] = int(target_id)
        priorities[unique_indices[should_assign]] = _label_priority(int(target_id), target_name)
        assigned_points += int(np.count_nonzero(should_assign))

    stats = build_manual_stats(
        labels=labels,
        objects=parsed_objects,
        class_to_id=class_to_id,
        ignore_id=ignore_id,
        default_id=default_id,
        assigned_points=assigned_points,
        duplicate_assignments=duplicate_assignments,
        unknown_objects=unknown_objects,
        out_of_range_indices=out_of_range_indices,
    )
    return ManualFrameLabels(
        frame_id=frame_id,
        annotation_path=annotation_source,
        point_count=int(point_count),
        labels=labels,
        objects=parsed_objects,
        stats=stats,
    )


def read_object_label(raw_object: dict[str, Any]) -> str:
    for field in DEFAULT_OBJECT_FIELDS:
        value = raw_object.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    class_id = raw_object.get("class_id", raw_object.get("label_id"))
    if class_id is not None and str(class_id).strip():
        return str(class_id).strip()
    return ""


def parse_indices(raw_object: dict[str, Any]) -> np.ndarray:
    value = None
    for key in ("indices", "point_indices", "pointIndexes", "point_indices_list"):
        if key in raw_object:
            value = raw_object[key]
            break
    if value is None:
        return np.asarray([], dtype=np.int64)
    if isinstance(value, np.ndarray):
        return value.astype(np.int64, copy=False).reshape(-1)
    if isinstance(value, str):
        tokens = re.findall(r"-?\d+", value)
        return np.asarray([int(token) for token in tokens], dtype=np.int64)
    if isinstance(value, Iterable):
        parsed = []
        for item in value:
            if item is None or str(item).strip() == "":
                continue
            parsed.append(int(float(item)))
        return np.asarray(parsed, dtype=np.int64)
    raise ValueError(f"Unsupported indices payload type: {type(value).__name__}")


def discover_manual_annotation_files(
    annotation_dir: str | Path,
    *,
    suffixes: tuple[str, ...] = DEFAULT_ANNOTATION_SUFFIXES,
) -> dict[str, Path]:
    root = Path(annotation_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Manual annotation directory does not exist: {root}")
    suffix_set = {suffix.casefold() for suffix in suffixes}
    result: dict[str, Path] = {}

    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix.casefold() in suffix_set:
            result[path.stem] = path.resolve()
        elif path.is_dir():
            label_file = next((path / name for name in FRAME_DIRECTORY_LABEL_FILES if (path / name).is_file()), None)
            if label_file is not None:
                result[path.name] = label_file.resolve()
    return result


def resolve_annotation_path_for_frame(
    *,
    frame_id: str,
    annotation_files: dict[str, Path],
) -> Path | None:
    if frame_id in annotation_files:
        return annotation_files[frame_id]
    suffixes = ("_instances", "_labels", "_annotations", "_label")
    for suffix in suffixes:
        candidate = annotation_files.get(f"{frame_id}{suffix}")
        if candidate is not None:
            return candidate
    return None


def load_class_alias_file(path: str | Path | None) -> dict[str, str | int]:
    if path is None:
        return {}
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Class alias file does not exist: {source}")
    text = source.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    if text.startswith("{"):
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError(f"Class alias JSON must be an object: {source}")
        return {str(key): value for key, value in payload.items()}
    aliases: dict[str, str | int] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" in line:
            key, value = [item.strip() for item in line.split("=", 1)]
        elif ":" in line:
            key, value = [item.strip() for item in line.split(":", 1)]
        else:
            raise ValueError(f"{source}:{line_number}: expected 'source=target' or 'source: target'")
        aliases[key] = int(value) if re.fullmatch(r"-?\d+", value) else value
    return aliases


def parse_inline_class_aliases(values: list[str] | None) -> dict[str, str | int]:
    aliases: dict[str, str | int] = {}
    for raw_value in values or []:
        if "=" not in raw_value:
            raise ValueError(f"--class-map must use source=target format, got {raw_value!r}")
        source, target = [item.strip() for item in raw_value.split("=", 1)]
        if not source or not target:
            raise ValueError(f"--class-map must use non-empty source=target, got {raw_value!r}")
        aliases[source] = int(target) if re.fullmatch(r"-?\d+", target) else target
    return aliases


class ClassResolver:
    def __init__(self, *, class_to_id: dict[str, int], class_aliases: dict[str, str | int], ignore_id: int = IGNORE_ID):
        self.class_to_id = {str(name): int(class_id) for name, class_id in class_to_id.items()}
        self.ignore_id = int(ignore_id)
        self.name_by_normalized = {_normalize_label(name): name for name in self.class_to_id}
        self.id_to_name = {int(class_id): str(name) for name, class_id in self.class_to_id.items()}
        merged_aliases: dict[str, str | int] = {}
        merged_aliases.update(default_class_aliases())
        merged_aliases.update(class_aliases)
        self.aliases = {_normalize_label(source): target for source, target in merged_aliases.items()}

    def resolve(self, raw_label: str, *, raw_object: dict[str, Any]) -> tuple[str, int] | None:
        explicit_id = raw_object.get("class_id", raw_object.get("label_id"))
        if explicit_id is not None and str(explicit_id).strip():
            label_id = int(float(explicit_id))
            if label_id in self.id_to_name:
                return self.id_to_name[label_id], label_id

        normalized = _normalize_label(raw_label)
        direct_name = self.name_by_normalized.get(normalized)
        if direct_name is not None:
            return direct_name, self.class_to_id[direct_name]
        if re.fullmatch(r"-?\d+", raw_label):
            label_id = int(raw_label)
            if label_id in self.id_to_name:
                return self.id_to_name[label_id], label_id

        target = self.aliases.get(normalized)
        if target is None:
            return None
        return self._resolve_alias_target(target)

    def _resolve_alias_target(self, target: str | int) -> tuple[str, int] | None:
        if isinstance(target, int):
            name = self.id_to_name.get(int(target))
            return None if name is None else (name, int(target))
        if re.fullmatch(r"-?\d+", str(target).strip()):
            label_id = int(str(target).strip())
            name = self.id_to_name.get(label_id)
            return None if name is None else (name, label_id)
        candidates = [item.strip() for item in str(target).split("|") if item.strip()]
        for candidate in candidates:
            direct = self.name_by_normalized.get(_normalize_label(candidate))
            if direct is not None:
                return direct, self.class_to_id[direct]
        return None


def default_class_aliases() -> dict[str, str]:
    return {
        "车": "CAR|Car",
        "汽车": "CAR|Car",
        "车辆": "CAR|Car",
        "小车": "CAR|Car",
        "轿车": "CAR|Car",
        "car": "CAR|Car",
        "truck": "TRUCK_BUS|Truck",
        "bus": "TRUCK_BUS|Bus",
        "货车": "TRUCK_BUS|Truck",
        "卡车": "TRUCK_BUS|Truck",
        "公交车": "TRUCK_BUS|Bus",
        "行人": "PEDESTRIAN|Pedestrian",
        "人": "PEDESTRIAN|Pedestrian",
        "pedestrian": "PEDESTRIAN|Pedestrian",
        "骑行者": "CYCLIST|Cyclist",
        "自行车": "CYCLIST|Cyclist",
        "摩托车": "MOTORCYCLE|motorcycle",
        "路牌": "traffic_sign|Sign",
        "标志牌": "traffic_sign|Sign",
        "交通标志": "traffic_sign|Sign",
        "交通灯": "Traffic Light|traffic_light",
        "红绿灯": "Traffic Light|traffic_light",
        "路面": "road|ground|background",
        "道路": "road|ground|background",
        "地面": "ground|road|background",
        "护栏": "roadblock|unknown_object",
        "路障": "roadblock|unknown_object",
        "防撞桶": "roadblock|traffic_cone|unknown_object",
        "锥桶": "traffic_cone|roadblock|unknown_object",
        "交通锥": "traffic_cone|Construction Cone",
        "轮胎": "tire|unknown_object",
        "三角牌": "triangular_traffic_sign|traffic_sign",
        "串扰": "crosstalk_noise_1|crosstalk noise|noise|unknown_noise",
        "水平串扰": "horizontal_crosstalk_noise|crosstalk_noise_1|unknown_noise",
        "垂直串扰": "vertical_crosstalk_noise|crosstalk_noise_1|unknown_noise",
        "尘土": "dust_noise|dust noise|unknown_noise",
        "扬尘": "dust_noise|dust noise|unknown_noise",
        "尾气": "exhaust_gas_noise|dust_noise|unknown_noise",
        "远距噪声": "long_distance_noise|Long-distance noise|unknown_noise",
        "远距离噪声": "long_distance_noise|Long-distance noise|unknown_noise",
        "粘连噪声": "adhesive_noise|adhesive noise|unknown_noise",
        "附着噪声": "adhesive_noise|adhesive noise|unknown_noise",
        "地下镜像": "underground_mirror_noise|Mirror noise|unknown_noise",
        "多机干扰": "multi_machine_interference_noise|unknown_noise",
        "多径": "multipath_noise|multipath noise|unknown_noise",
        "多距离": "multiple_range_noise|unknown_noise",
        "近距分层": "near_range_layered_noise|unknown_noise",
        "噪声": "unknown_noise|noise|Noise",
    }


def choose_assignment_mask(
    *,
    current_labels: np.ndarray,
    current_priorities: np.ndarray,
    target_id: int,
    target_name: str | None,
    conflict_policy: str,
    class_to_id: dict[str, int],
) -> np.ndarray:
    if conflict_policy == "last":
        return np.ones(current_labels.shape, dtype=bool)
    if conflict_policy == "first":
        ignore_id = _ignore_id_from_classes(class_to_id)
        return current_labels == ignore_id
    if conflict_policy == "error":
        ignore_id = _ignore_id_from_classes(class_to_id)
        occupied = current_labels != ignore_id
        if np.any(occupied):
            raise ValueError(f"Manual annotation conflict for target class {target_name or target_id!r}")
        return np.ones(current_labels.shape, dtype=bool)
    priority = _label_priority(target_id, target_name)
    return priority >= current_priorities


def build_manual_stats(
    *,
    labels: np.ndarray,
    objects: list[ManualAnnotationObject],
    class_to_id: dict[str, int],
    ignore_id: int,
    default_id: int,
    assigned_points: int,
    duplicate_assignments: int,
    unknown_objects: list[dict[str, Any]],
    out_of_range_indices: list[int],
) -> dict[str, Any]:
    id_to_name = {int(class_id): str(name) for name, class_id in class_to_id.items()}
    unique_ids, counts = np.unique(labels, return_counts=True)
    class_counts = {
        str(int(label_id)): {
            "name": id_to_name.get(int(label_id), "unknown"),
            "count": int(count),
        }
        for label_id, count in zip(unique_ids, counts)
    }
    return {
        "point_count": int(labels.size),
        "objects": len(objects),
        "objects_with_points": int(sum(1 for obj in objects if obj.indices.size > 0)),
        "assigned_points_attempted": int(assigned_points),
        "labeled_points": int(np.count_nonzero(labels != default_id)),
        "ignored_or_default_points": int(np.count_nonzero(labels == default_id)),
        "duplicate_assignments": int(duplicate_assignments),
        "unknown_objects": unknown_objects,
        "out_of_range_indices_sample": out_of_range_indices[:100],
        "ignore_id": int(ignore_id),
        "default_id": int(default_id),
        "class_counts": class_counts,
        "object_summaries": [
            {
                "object_index": obj.object_index,
                "raw_label": obj.raw_label,
                "target_name": obj.target_name,
                "target_id": obj.target_id,
                "indices": int(obj.indices.size),
                **obj.metadata,
            }
            for obj in objects
        ],
    }


def summarize_object_metadata(raw_object: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("markType", "annotateType", "annotationId", "trackId", "true_pointcloud_num", "real_pointcloud_num"):
        if key in raw_object:
            result[key] = raw_object[key]
    psr = raw_object.get("psr")
    if isinstance(psr, dict):
        result["psr"] = psr
    return result


def _extract_annotation_objects(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("objects", "annotations", "labels", "instances", "data", "result", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
            if isinstance(value, dict):
                nested = _extract_annotation_objects(value)
                if nested:
                    return nested
        if "indices" in payload:
            return [payload]
    raise ValueError("Manual annotation payload must be a JSON list or contain an objects/annotations list")


def _load_json_lines(text: str, *, source: Path) -> list[Any]:
    result = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}:{line_number}: invalid JSON line: {exc}") from exc
    return result


def _normalize_label(value: str) -> str:
    return re.sub(r"[\s_\-]+", "", str(value).strip().casefold())


def _name_for_id(class_to_id: dict[str, int], label_id: int) -> str | None:
    for name, class_id in class_to_id.items():
        if int(class_id) == int(label_id):
            return str(name)
    return None


def _ignore_id_from_classes(class_to_id: dict[str, int]) -> int:
    for name, class_id in class_to_id.items():
        if str(name).casefold() in {"ignore", "ignored"}:
            return int(class_id)
    return IGNORE_ID


def _label_priority(label_id: int, label_name: str | None) -> int:
    name = "" if label_name is None else str(label_name).casefold()
    if int(label_id) == IGNORE_ID or name in {"ignore", "ignored"}:
        return 0
    if name in {"background", "road", "ground", "other ground"}:
        return 10
    if "noise" in name or "crosstalk" in name or "multipath" in name:
        return 100
    if name in {
        "traffic_cone",
        "roadblock",
        "tire",
        "wheel_chock",
        "traffic_sign",
        "triangular_traffic_sign",
        "thin_horizontal_bar",
        "thin_vertical_bar",
        "road_stud",
    }:
        return 80
    if name in {"car", "truck_bus", "pedestrian", "cyclist", "motorcycle"}:
        return 70
    return 50
