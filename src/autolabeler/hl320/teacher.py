from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from autolabeler.data.class_config import load_noise_groups, load_semantic_classes
from autolabeler.hl320.csv_points import HL320Frame, group_echo_returns, load_hl320_csv
from autolabeler.hl320.fusion import IGNORE_ID, round_like_point_labeler


@dataclass(frozen=True)
class TeacherPolicy:
    ignore_id: int = IGNORE_ID
    min_candidate_confidence: float = 0.35
    angular_bin_deg: float = 0.5
    isolated_density_max: int = 3
    stable_distance_tolerance_m: float = 0.75
    low_intensity_threshold: float = 0.08
    very_low_intensity_threshold: float = 0.03
    long_distance_min_m: float = 80.0
    far_distance_quantile: float = 98.0
    adhesive_near_m: float = 2.5
    multi_echo_delta_m: float = 0.3
    strong_multi_echo_delta_m: float = 1.0
    sam3_min_confidence: float = 0.7
    sam3_boundary_penalty: float = 0.08
    sam3_object_penalty: float = 0.06
    litept_noise_boost: float = 0.08
    litept_non_noise_penalty: float = 0.05


@dataclass(frozen=True)
class TeacherBuildResult:
    output_dir: Path
    manifest_path: Path
    frame_count: int


@dataclass(frozen=True)
class TeacherFrameResult:
    labels: np.ndarray
    confidence: np.ndarray
    reasons: dict[str, Any]
    features_debug: dict[str, np.ndarray]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PointContext:
    labels: np.ndarray
    confidence: np.ndarray
    boundary: np.ndarray
    source_path: str | None


class CandidateAccumulator:
    def __init__(self, point_count: int, *, ignore_id: int, min_confidence: float) -> None:
        self.labels = np.full(point_count, int(ignore_id), dtype=np.int32)
        self.confidence = np.zeros(point_count, dtype=np.float32)
        self._ignore_id = int(ignore_id)
        self._min_confidence = float(min_confidence)
        self._entries: dict[str, list[dict[str, Any]]] = {}
        self._rule_counts: dict[str, int] = {}

    def add(
        self,
        *,
        label_id: int | None,
        label_name: str,
        rule: str,
        score: np.ndarray,
        reason_masks: list[tuple[str, np.ndarray]],
        allow_overwrite_noise: bool = True,
    ) -> None:
        if label_id is None:
            self._rule_counts[rule] = 0
            return
        scores = np.nan_to_num(np.asarray(score, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        candidate = scores >= self._min_confidence
        if not allow_overwrite_noise:
            candidate &= self.labels == self._ignore_id
        indices = np.flatnonzero(candidate)
        self._rule_counts[rule] = int(indices.size)
        for index in indices.tolist():
            tags = [tag for tag, mask in reason_masks if bool(np.asarray(mask, dtype=bool)[index])]
            payload = {
                "label_id": int(label_id),
                "label_name": str(label_name),
                "rule": str(rule),
                "confidence": round(float(scores[index]), 6),
                "reasons": tags or [rule],
            }
            self._entries.setdefault(str(index), []).append(payload)
        better = candidate & (scores > self.confidence)
        self.labels[better] = int(label_id)
        self.confidence[better] = scores[better]

    @property
    def point_reasons(self) -> dict[str, list[dict[str, Any]]]:
        return self._entries

    @property
    def rule_counts(self) -> dict[str, int]:
        return dict(self._rule_counts)


def build_hl320_teacher_candidates(
    *,
    csv_dir: str | Path,
    classes_yaml: str | Path,
    out_dir: str | Path,
    sam3_dir: str | Path | None = None,
    litept_dir: str | Path | None = None,
    image_dir: str | Path | None = None,
    overwrite: bool = False,
    policy: TeacherPolicy = TeacherPolicy(),
) -> TeacherBuildResult:
    csv_root = Path(csv_dir).expanduser().resolve()
    classes_path = Path(classes_yaml).expanduser().resolve()
    output_root = Path(out_dir).expanduser().resolve()
    sam3_root = Path(sam3_dir).expanduser().resolve() if sam3_dir else None
    litept_root = Path(litept_dir).expanduser().resolve() if litept_dir else None
    image_root = Path(image_dir).expanduser().resolve() if image_dir else None
    _require_dir(csv_root, "CSV directory")
    _require_file(classes_path, "Classes YAML")
    if sam3_root is not None:
        _require_dir(sam3_root, "SAM3 directory")
    if litept_root is not None:
        _require_dir(litept_root, "LitePT directory")
    if image_root is not None:
        _require_dir(image_root, "Image directory")
    _prepare_output_dir(output_root, overwrite=overwrite)

    csv_paths = sorted(csv_root.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No .csv files found in {csv_root}")

    class_to_id = load_semantic_classes(str(classes_path))
    noise_groups = load_noise_groups(str(classes_path))
    frames = [load_hl320_csv(path) for path in csv_paths]
    manifest: dict[str, Any] = {
        "version": 1,
        "mode": "hl320_physics_teacher_v1",
        "csv_dir": str(csv_root),
        "classes_yaml": str(classes_path),
        "sam3_dir": str(sam3_root) if sam3_root is not None else None,
        "litept_dir": str(litept_root) if litept_root is not None else None,
        "image_dir": str(image_root) if image_root is not None else None,
        "out_dir": str(output_root),
        "policy": asdict(policy),
        "frames": [],
        "summary": _empty_summary(),
    }
    top_candidates = output_root / "teacher_candidates"
    top_candidates.mkdir(parents=True, exist_ok=True)

    for index, frame in enumerate(frames):
        prev_frame = frames[index - 1] if index > 0 else None
        next_frame = frames[index + 1] if index + 1 < len(frames) else None
        result = evaluate_hl320_teacher_frame(
            frame=frame,
            prev_frame=prev_frame,
            next_frame=next_frame,
            class_to_id=class_to_id,
            noise_groups=noise_groups,
            sam3_dir=sam3_root,
            litept_dir=litept_root,
            image_dir=image_root,
            policy=policy,
        )
        frame_out = output_root / frame.frame_id
        frame_out.mkdir(parents=True, exist_ok=True)
        np.save(frame_out / "candidate_mask.npy", result.labels)
        np.save(frame_out / "candidate_confidence.npy", result.confidence)
        np.savez_compressed(frame_out / "features_debug.npz", **result.features_debug)
        np.save(top_candidates / f"{frame.frame_id}.npy", result.labels)
        (frame_out / "candidate_reasons.json").write_text(
            json.dumps(result.reasons, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (frame_out / "metadata.json").write_text(
            json.dumps(result.metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        manifest_frame = {
            "frame_id": frame.frame_id,
            "path": str(frame_out),
            "point_count": int(frame.point_count),
            "candidate_points": int(np.count_nonzero(result.labels != policy.ignore_id)),
            "candidate_mask": str(frame_out / "candidate_mask.npy"),
            "candidate_confidence": str(frame_out / "candidate_confidence.npy"),
            "candidate_reasons": str(frame_out / "candidate_reasons.json"),
            "features_debug": str(frame_out / "features_debug.npz"),
            "metadata": str(frame_out / "metadata.json"),
            "rule_counts": result.metadata["rule_counts"],
            "class_counts": result.metadata["class_counts"],
        }
        manifest["frames"].append(manifest_frame)
        _accumulate_summary(manifest["summary"], manifest_frame)

    manifest_path = output_root / "hl320_teacher_candidates_manifest.json"
    manifest["manifest"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return TeacherBuildResult(output_dir=output_root, manifest_path=manifest_path, frame_count=len(frames))


def evaluate_hl320_teacher_frame(
    *,
    frame: HL320Frame,
    prev_frame: HL320Frame | None,
    next_frame: HL320Frame | None,
    class_to_id: dict[str, int],
    noise_groups: dict[str, list[str]] | None = None,
    sam3_dir: Path | None = None,
    litept_dir: Path | None = None,
    image_dir: Path | None = None,
    policy: TeacherPolicy = TeacherPolicy(),
) -> TeacherFrameResult:
    features = build_teacher_features(
        frame=frame,
        prev_frame=prev_frame,
        next_frame=next_frame,
        class_to_id=class_to_id,
        noise_groups=noise_groups or {},
        sam3_dir=sam3_dir,
        litept_dir=litept_dir,
        policy=policy,
    )
    accumulator = CandidateAccumulator(
        frame.point_count,
        ignore_id=policy.ignore_id,
        min_confidence=policy.min_candidate_confidence,
    )
    rules = build_rule_scores(features, class_to_id, noise_groups or {}, policy)
    for rule in rules:
        accumulator.add(
            label_id=rule["label_id"],
            label_name=rule["label_name"],
            rule=rule["rule"],
            score=rule["score"],
            reason_masks=rule["reason_masks"],
        )

    add_sam3_object_context_candidates(
        accumulator=accumulator,
        features=features,
        class_to_id=class_to_id,
        policy=policy,
    )

    final_labels = accumulator.labels.astype(np.int32, copy=False)
    final_confidence = accumulator.confidence.astype(np.float32, copy=False)
    class_counts = _class_counts(final_labels, ignore_id=policy.ignore_id)
    image_path = resolve_image_path(image_dir, frame.frame_id) if image_dir is not None else None
    reasons = {
        "version": 1,
        "frame_id": frame.frame_id,
        "min_candidate_confidence": float(policy.min_candidate_confidence),
        "rule_counts": accumulator.rule_counts,
        "points": accumulator.point_reasons,
    }
    metadata = {
        "version": 1,
        "frame_id": frame.frame_id,
        "source_csv": str(frame.path),
        "image_path": str(image_path) if image_path is not None else None,
        "point_count": int(frame.point_count),
        "candidate_points": int(np.count_nonzero(final_labels != policy.ignore_id)),
        "class_counts": {str(key): int(value) for key, value in class_counts.items()},
        "rule_counts": accumulator.rule_counts,
        "sam3_source": features["sam3_source_path"],
        "litept_source": features["litept_source_path"],
        "policy": asdict(policy),
    }
    return TeacherFrameResult(
        labels=final_labels,
        confidence=final_confidence,
        reasons=reasons,
        features_debug=debug_feature_arrays(features),
        metadata=metadata,
    )


def build_rule_scores(
    features: dict[str, Any],
    class_to_id: dict[str, int],
    noise_groups: dict[str, list[str]],
    policy: TeacherPolicy,
) -> list[dict[str, Any]]:
    distance = features["distance"]
    valid_xyz = features["valid_xyz"]
    intensity = features["intensity_norm"]
    reflectivity = features["reflectivity_norm"]
    return_count = features["return_count"]
    echo_rank = features["echo_rank_by_distance"]
    nearest_delta = features["nearest_echo_distance_delta"]
    strongest_delta = features["strongest_echo_intensity_delta_norm"]
    angular_density = features["angular_density"]
    angular_row_count = features["angular_row_count"]
    angular_col_count = features["angular_col_count"]
    isolated = features["isolated"]
    has_projection = features["has_camera_projection"]
    transient = features["temporal_transient"]
    stable = features["temporal_stable"]
    below_ground = features["below_ground"]
    sam3_boundary = features["sam3_boundary"]
    sam3_context = features["sam3_context_valid"]
    litept_noise = features["litept_noise_support"]
    litept_non_noise = features["litept_non_noise_context"]
    low_intensity = intensity <= policy.low_intensity_threshold
    very_low_intensity = intensity <= policy.very_low_intensity_threshold
    low_reflectivity = reflectivity <= policy.low_intensity_threshold
    single_return = return_count <= 1
    multi_echo = return_count >= 2
    secondary_echo = echo_rank > 0
    large_delta = nearest_delta >= policy.multi_echo_delta_m
    strong_delta = nearest_delta >= policy.strong_multi_echo_delta_m
    far_threshold = max(policy.long_distance_min_m, _safe_percentile(distance, policy.far_distance_quantile, default=policy.long_distance_min_m))
    far = distance >= far_threshold
    near = distance <= policy.adhesive_near_m
    line_threshold = max(12.0, _safe_percentile(np.maximum(angular_row_count, angular_col_count), 90.0, default=12.0))
    line_artifact = (angular_row_count >= line_threshold) | (angular_col_count >= line_threshold)

    rules = []

    dust_score = (
        0.18
        + 0.18 * isolated.astype(np.float32)
        + 0.17 * low_intensity.astype(np.float32)
        + 0.15 * transient.astype(np.float32)
        + 0.08 * (~has_projection).astype(np.float32)
        + 0.07 * single_return.astype(np.float32)
    )
    dust_score = adjust_noise_score(dust_score, features, policy, suppress_when_stable=True)
    rules.append(
        _rule(
            "dust_noise",
            _resolve_class(class_to_id, "dust_noise", *noise_groups.get("dust_noise", [])),
            dust_score,
            valid_xyz & isolated & low_intensity & single_return & (transient | (angular_density <= 1)),
            [
                ("isolated_angular_neighbor", isolated),
                ("low_intensity", low_intensity),
                ("temporal_transient", transient),
                ("single_return", single_return),
                ("no_camera_projection", ~has_projection),
                ("sam3_context_penalty", sam3_context),
                ("sam3_boundary_needs_review", sam3_boundary),
                ("litept_noise_support", litept_noise),
                ("litept_non_noise_context", litept_non_noise),
            ],
        )
    )

    long_score = (
        0.24
        + 0.24 * far.astype(np.float32)
        + 0.12 * low_intensity.astype(np.float32)
        + 0.10 * isolated.astype(np.float32)
        + 0.08 * transient.astype(np.float32)
        + 0.08 * (~has_projection).astype(np.float32)
    )
    long_score = adjust_noise_score(long_score, features, policy)
    rules.append(
        _rule(
            "long_distance_noise",
            _resolve_class(class_to_id, "long_distance_noise", "Long_distance_noise", *noise_groups.get("long_distance_noise", [])),
            long_score,
            valid_xyz & far & (low_intensity | isolated | (~has_projection)),
            [
                ("far_range", far),
                ("low_intensity", low_intensity),
                ("isolated_angular_neighbor", isolated),
                ("temporal_transient", transient),
                ("no_camera_projection", ~has_projection),
                ("sam3_context_penalty", sam3_context),
                ("litept_noise_support", litept_noise),
            ],
        )
    )

    multipath_score = (
        0.22
        + 0.18 * multi_echo.astype(np.float32)
        + 0.16 * secondary_echo.astype(np.float32)
        + 0.14 * large_delta.astype(np.float32)
        + 0.08 * (strongest_delta <= -0.05).astype(np.float32)
        + 0.06 * stable.astype(np.float32)
    )
    multipath_score = adjust_noise_score(multipath_score, features, policy)
    rules.append(
        _rule(
            "multipath_noise",
            _resolve_class(class_to_id, "multipath_noise", *noise_groups.get("multipath_noise", [])),
            multipath_score,
            valid_xyz & multi_echo & secondary_echo & large_delta,
            [
                ("multi_echo_return", multi_echo),
                ("secondary_echo", secondary_echo),
                ("large_echo_distance_delta", large_delta),
                ("weaker_than_strongest_echo", strongest_delta <= -0.05),
                ("temporal_stable", stable),
                ("sam3_boundary_needs_review", sam3_boundary),
                ("litept_noise_support", litept_noise),
            ],
        )
    )

    multiple_range_score = (
        0.20
        + 0.20 * multi_echo.astype(np.float32)
        + 0.16 * (return_count >= 3).astype(np.float32)
        + 0.14 * strong_delta.astype(np.float32)
        + 0.08 * stable.astype(np.float32)
    )
    multiple_range_score = adjust_noise_score(multiple_range_score, features, policy)
    rules.append(
        _rule(
            "multiple_range_noise",
            _resolve_class(class_to_id, "multiple_range_noise", *noise_groups.get("multiple_range_noise", [])),
            multiple_range_score,
            valid_xyz & multi_echo & ((return_count >= 3) | (secondary_echo & strong_delta)),
            [
                ("multi_echo_return", multi_echo),
                ("three_or_more_returns", return_count >= 3),
                ("strong_echo_distance_delta", strong_delta),
                ("secondary_echo", secondary_echo),
                ("temporal_stable", stable),
                ("litept_noise_support", litept_noise),
            ],
        )
    )

    underground_score = (
        0.24
        + 0.24 * below_ground.astype(np.float32)
        + 0.10 * secondary_echo.astype(np.float32)
        + 0.08 * (~has_projection).astype(np.float32)
        + 0.08 * low_reflectivity.astype(np.float32)
    )
    underground_score = adjust_noise_score(underground_score, features, policy)
    rules.append(
        _rule(
            "underground_mirror_noise",
            _resolve_class(class_to_id, "underground_mirror_noise", "underground_noise", *noise_groups.get("underground_mirror_noise", [])),
            underground_score,
            valid_xyz & below_ground & ((~has_projection) | low_reflectivity | secondary_echo),
            [
                ("below_ground_z_outlier", below_ground),
                ("secondary_echo", secondary_echo),
                ("low_reflectivity", low_reflectivity),
                ("no_camera_projection", ~has_projection),
                ("sam3_boundary_needs_review", sam3_boundary),
                ("litept_noise_support", litept_noise),
            ],
        )
    )

    crosstalk_score = (
        0.18
        + 0.16 * line_artifact.astype(np.float32)
        + 0.15 * transient.astype(np.float32)
        + 0.13 * low_intensity.astype(np.float32)
        + 0.09 * isolated.astype(np.float32)
        + 0.08 * (~has_projection).astype(np.float32)
    )
    crosstalk_score = adjust_noise_score(crosstalk_score, features, policy, suppress_when_stable=True)
    rules.append(
        _rule(
            "crosstalk_noise",
            _resolve_class(
                class_to_id,
                "crosstalk_noise",
                "crosstalk_noise_1",
                "horizontal_crosstalk_noise",
                "vertical_crosstalk_noise",
                *noise_groups.get("crosstalk_noise", []),
            ),
            crosstalk_score,
            valid_xyz & line_artifact & transient & low_intensity,
            [
                ("angular_line_artifact", line_artifact),
                ("temporal_transient", transient),
                ("low_intensity", low_intensity),
                ("isolated_angular_neighbor", isolated),
                ("no_camera_projection", ~has_projection),
                ("sam3_context_penalty", sam3_context),
                ("litept_noise_support", litept_noise),
            ],
        )
    )

    adhesive_score = (
        0.20
        + 0.18 * near.astype(np.float32)
        + 0.15 * stable.astype(np.float32)
        + 0.11 * very_low_intensity.astype(np.float32)
        + 0.08 * low_reflectivity.astype(np.float32)
        + 0.05 * single_return.astype(np.float32)
    )
    adhesive_score = adjust_noise_score(adhesive_score, features, policy)
    rules.append(
        _rule(
            "adhesive_noise",
            _resolve_class(class_to_id, "adhesive_noise", *noise_groups.get("adhesive_noise", [])),
            adhesive_score,
            valid_xyz & near & stable & single_return & (very_low_intensity | low_reflectivity),
            [
                ("near_range", near),
                ("temporal_stable", stable),
                ("single_return", single_return),
                ("very_low_intensity", very_low_intensity),
                ("low_reflectivity", low_reflectivity),
                ("litept_noise_support", litept_noise),
            ],
        )
    )

    return rules


def build_teacher_features(
    *,
    frame: HL320Frame,
    prev_frame: HL320Frame | None,
    next_frame: HL320Frame | None,
    class_to_id: dict[str, int],
    noise_groups: dict[str, list[str]],
    sam3_dir: Path | None,
    litept_dir: Path | None,
    policy: TeacherPolicy,
) -> dict[str, Any]:
    groups = group_echo_returns(frame)
    point_count = frame.point_count
    xyz = np.asarray(frame.points[:, :3], dtype=np.float32)
    valid_xyz = np.all(np.isfinite(xyz), axis=1) & (np.linalg.norm(np.nan_to_num(xyz, nan=0.0), axis=1) > 1e-8)
    distance = _field(frame, "distance", fill=np.nan)
    xyz_distance = np.linalg.norm(np.nan_to_num(xyz, nan=0.0), axis=1).astype(np.float32)
    distance = np.where(np.isfinite(distance), distance, xyz_distance).astype(np.float32, copy=False)
    intensity = _normalize_u8(frame.points[:, 3])
    reflectivity = _normalize_u8(_field(frame, "reflectivity", fill=np.nan))
    azimuth = _field(frame, "azimuth", fill=np.nan)
    vertical = _field(frame, "vertical", fill=np.nan)
    cxd = _field(frame, "cxd", fill=np.nan)
    cyd = _field(frame, "cyd", fill=np.nan)
    has_projection = np.isfinite(cxd) & np.isfinite(cyd)
    angular_density, row_count, col_count = angular_density_features(azimuth, vertical, bin_deg=policy.angular_bin_deg)
    temporal_support, temporal_neighbors = temporal_support_features(
        frame,
        prev_frame=prev_frame,
        next_frame=next_frame,
        tolerance_m=policy.stable_distance_tolerance_m,
    )
    temporal_stable = temporal_support > 0
    temporal_transient = (temporal_neighbors > 0) & (temporal_support == 0)
    z = np.asarray(frame.points[:, 2], dtype=np.float32)
    z_threshold = min(-0.25, _safe_percentile(z, 5.0, default=-0.25) - 0.15)
    below_ground = np.isfinite(z) & (z <= z_threshold)
    sam3_context = load_sam3_context(
        frame=frame,
        class_to_id=class_to_id,
        sam3_dir=sam3_dir,
        policy=policy,
    )
    litept_context = load_litept_context(frame=frame, litept_dir=litept_dir, policy=policy)
    noise_ids = _noise_source_ids(class_to_id, noise_groups)
    litept_noise = np.isin(litept_context.labels, np.asarray(sorted(noise_ids), dtype=np.int32)) & (litept_context.confidence >= 0.5)
    litept_non_noise = (
        (litept_context.labels != policy.ignore_id)
        & ~np.isin(litept_context.labels, np.asarray(sorted(noise_ids), dtype=np.int32))
        & (litept_context.confidence >= 0.7)
    )
    strongest_delta = groups.strongest_echo_intensity_delta.astype(np.float32, copy=False)
    if np.nanmax(np.abs(strongest_delta)) > 1.0:
        strongest_delta = strongest_delta / 255.0
    return {
        "distance": np.nan_to_num(distance.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0),
        "valid_xyz": valid_xyz,
        "intensity_norm": intensity,
        "reflectivity_norm": np.nan_to_num(reflectivity, nan=0.0, posinf=0.0, neginf=0.0),
        "azimuth": azimuth.astype(np.float32, copy=False),
        "vertical": vertical.astype(np.float32, copy=False),
        "return_count": groups.return_count_by_row.astype(np.int32, copy=False),
        "echo_rank_by_distance": groups.echo_rank_by_distance.astype(np.int32, copy=False),
        "nearest_echo_distance_delta": groups.nearest_echo_distance_delta.astype(np.float32, copy=False),
        "strongest_echo_intensity_delta_norm": strongest_delta.astype(np.float32, copy=False),
        "angular_density": angular_density.astype(np.int32, copy=False),
        "angular_row_count": row_count.astype(np.int32, copy=False),
        "angular_col_count": col_count.astype(np.int32, copy=False),
        "isolated": angular_density <= policy.isolated_density_max,
        "has_camera_projection": has_projection,
        "temporal_support": temporal_support.astype(np.int32, copy=False),
        "temporal_neighbors": temporal_neighbors.astype(np.int32, copy=False),
        "temporal_stable": temporal_stable,
        "temporal_transient": temporal_transient,
        "below_ground": below_ground,
        "sam3_label": sam3_context.labels,
        "sam3_confidence": sam3_context.confidence,
        "sam3_boundary": sam3_context.boundary,
        "sam3_context_valid": sam3_context.labels != policy.ignore_id,
        "sam3_source_path": sam3_context.source_path,
        "litept_label": litept_context.labels,
        "litept_confidence": litept_context.confidence,
        "litept_noise_support": litept_noise,
        "litept_non_noise_context": litept_non_noise,
        "litept_source_path": litept_context.source_path,
        "point_count": np.asarray([point_count], dtype=np.int32),
    }


def add_sam3_object_context_candidates(
    *,
    accumulator: CandidateAccumulator,
    features: dict[str, Any],
    class_to_id: dict[str, int],
    policy: TeacherPolicy,
) -> None:
    context_ids = {
        class_id
        for name, class_id in class_to_id.items()
        if name.casefold()
        in {
            "car",
            "truck_bus",
            "pedestrian",
            "traffic_sign",
            "triangular_traffic_sign",
            "roadblock",
            "wheel_chock",
            "tire",
            "traffic_cone",
            "cyclist",
            "motorcycle",
            "road",
        }
    }
    if not context_ids:
        return
    sam3_label = np.asarray(features["sam3_label"], dtype=np.int32)
    sam3_conf = np.asarray(features["sam3_confidence"], dtype=np.float32)
    boundary = np.asarray(features["sam3_boundary"], dtype=bool)
    for class_id in sorted(context_ids):
        mask = sam3_label == int(class_id)
        if not np.any(mask):
            continue
        score = sam3_conf * 0.65
        score[boundary] *= 0.5
        label_name = _name_for_id(class_to_id, int(class_id))
        accumulator.add(
            label_id=int(class_id),
            label_name=label_name,
            rule="sam3_object_context",
            score=np.where(mask, score, 0.0),
            reason_masks=[
                ("sam3_object_context", mask),
                ("sam3_boundary_needs_review", boundary),
            ],
            allow_overwrite_noise=False,
        )


def adjust_noise_score(
    score: np.ndarray,
    features: dict[str, Any],
    policy: TeacherPolicy,
    *,
    suppress_when_stable: bool = False,
) -> np.ndarray:
    adjusted = np.asarray(score, dtype=np.float32).copy()
    adjusted[features["sam3_context_valid"]] -= float(policy.sam3_object_penalty)
    adjusted[features["sam3_boundary"]] -= float(policy.sam3_boundary_penalty)
    adjusted[features["litept_noise_support"]] += float(policy.litept_noise_boost)
    adjusted[features["litept_non_noise_context"]] -= float(policy.litept_non_noise_penalty)
    if suppress_when_stable:
        adjusted[features["temporal_stable"]] -= 0.10
    return np.clip(adjusted, 0.0, 0.98)


def angular_density_features(azimuth: np.ndarray, vertical: np.ndarray, *, bin_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    point_count = int(np.asarray(azimuth).size)
    valid = np.isfinite(azimuth) & np.isfinite(vertical)
    density = np.zeros(point_count, dtype=np.int32)
    row_count = np.zeros(point_count, dtype=np.int32)
    col_count = np.zeros(point_count, dtype=np.int32)
    if point_count == 0 or not np.any(valid):
        return density, row_count, col_count
    az_bin = np.full(point_count, 0, dtype=np.int32)
    v_bin = np.full(point_count, 0, dtype=np.int32)
    az_bin[valid] = np.floor(np.asarray(azimuth)[valid] / float(bin_deg)).astype(np.int32)
    v_bin[valid] = np.floor(np.asarray(vertical)[valid] / float(bin_deg)).astype(np.int32)
    cell_counts: dict[tuple[int, int], int] = {}
    rows: dict[int, int] = {}
    cols: dict[int, int] = {}
    for az, vv in zip(az_bin[valid], v_bin[valid], strict=False):
        cell_counts[(int(az), int(vv))] = cell_counts.get((int(az), int(vv)), 0) + 1
        cols[int(az)] = cols.get(int(az), 0) + 1
        rows[int(vv)] = rows.get(int(vv), 0) + 1
    valid_indices = np.flatnonzero(valid)
    for index in valid_indices.tolist():
        az = int(az_bin[index])
        vv = int(v_bin[index])
        density[index] = sum(cell_counts.get((az + da, vv + dv), 0) for da in (-1, 0, 1) for dv in (-1, 0, 1))
        row_count[index] = rows.get(vv, 0)
        col_count[index] = cols.get(az, 0)
    return density, row_count, col_count


def temporal_support_features(
    frame: HL320Frame,
    *,
    prev_frame: HL320Frame | None,
    next_frame: HL320Frame | None,
    tolerance_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    slot = _field(frame, "slot", fill=np.nan)
    pixel = _field(frame, "pixel", fill=np.nan)
    distance = _field(frame, "distance", fill=np.nan)
    if not np.any(np.isfinite(distance)):
        distance = np.linalg.norm(frame.points[:, :3], axis=1).astype(np.float32)
    support = np.zeros(frame.point_count, dtype=np.int32)
    checked = np.zeros(frame.point_count, dtype=np.int32)
    neighbors = [item for item in (prev_frame, next_frame) if item is not None]
    if not neighbors:
        return support, checked
    neighbor_maps = [_distance_map_by_slot_pixel(item) for item in neighbors]
    for row in range(frame.point_count):
        if not (np.isfinite(slot[row]) and np.isfinite(pixel[row]) and np.isfinite(distance[row])):
            continue
        key = (int(slot[row]), int(pixel[row]))
        for neighbor in neighbor_maps:
            values = neighbor.get(key)
            if values is None:
                continue
            checked[row] += 1
            if np.any(np.abs(values - float(distance[row])) <= float(tolerance_m)):
                support[row] += 1
    return support, checked


def load_sam3_context(
    *,
    frame: HL320Frame,
    class_to_id: dict[str, int],
    sam3_dir: Path | None,
    policy: TeacherPolicy,
) -> PointContext:
    empty = PointContext(
        labels=np.full(frame.point_count, policy.ignore_id, dtype=np.int32),
        confidence=np.zeros(frame.point_count, dtype=np.float32),
        boundary=np.zeros(frame.point_count, dtype=bool),
        source_path=None,
    )
    if sam3_dir is None:
        return empty
    frame_dir = sam3_dir / frame.frame_id
    semantic_path = frame_dir / "semantic_mask.npy"
    if not semantic_path.is_file():
        return empty
    semantic = np.load(semantic_path, allow_pickle=False)
    if semantic.ndim != 2 or not np.issubdtype(semantic.dtype, np.integer):
        return empty
    confidence_path = frame_dir / "confidence.npy"
    if confidence_path.is_file():
        confidence_image = np.load(confidence_path, allow_pickle=False)
        if confidence_image.shape != semantic.shape or not np.issubdtype(confidence_image.dtype, np.number):
            confidence_image = np.zeros(semantic.shape, dtype=np.float32)
        else:
            confidence_image = np.nan_to_num(confidence_image.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    else:
        confidence_image = np.ones(semantic.shape, dtype=np.float32)
    cxd = _field(frame, "cxd", fill=np.nan)
    cyd = _field(frame, "cyd", fill=np.nan)
    labels = np.full(frame.point_count, policy.ignore_id, dtype=np.int32)
    confidence = np.zeros(frame.point_count, dtype=np.float32)
    boundary = np.zeros(frame.point_count, dtype=bool)
    finite = np.isfinite(cxd) & np.isfinite(cyd)
    u = np.full(frame.point_count, -1, dtype=np.int64)
    v = np.full(frame.point_count, -1, dtype=np.int64)
    u[finite] = round_like_point_labeler(cxd[finite])
    v[finite] = round_like_point_labeler(cyd[finite])
    height, width = semantic.shape
    inside = finite & (u >= 0) & (v >= 0) & (u < width) & (v < height)
    known_ids = {int(value) for value in class_to_id.values()}
    sampled = np.full(frame.point_count, policy.ignore_id, dtype=np.int32)
    sampled_conf = np.zeros(frame.point_count, dtype=np.float32)
    sampled[inside] = semantic[v[inside], u[inside]].astype(np.int32, copy=False)
    sampled_conf[inside] = confidence_image[v[inside], u[inside]]
    accepted = (
        inside
        & (sampled_conf >= policy.sam3_min_confidence)
        & np.isin(sampled, np.asarray(sorted(known_ids), dtype=np.int32))
        & (sampled != 0)
        & (sampled != policy.ignore_id)
    )
    labels[accepted] = sampled[accepted]
    confidence[inside] = sampled_conf[inside]
    boundary_image = semantic_boundary(semantic)
    boundary[inside] = boundary_image[v[inside], u[inside]]
    return PointContext(labels=labels, confidence=confidence, boundary=boundary, source_path=str(semantic_path))


def load_litept_context(*, frame: HL320Frame, litept_dir: Path | None, policy: TeacherPolicy) -> PointContext:
    empty = PointContext(
        labels=np.full(frame.point_count, policy.ignore_id, dtype=np.int32),
        confidence=np.zeros(frame.point_count, dtype=np.float32),
        boundary=np.zeros(frame.point_count, dtype=bool),
        source_path=None,
    )
    if litept_dir is None:
        return empty
    semantic_path = litept_dir / frame.frame_id / "semantic_mask.npy"
    if not semantic_path.is_file():
        return empty
    labels = np.load(semantic_path, allow_pickle=False)
    if labels.shape != (frame.point_count,) or not np.issubdtype(labels.dtype, np.integer):
        return empty
    confidence_path = litept_dir / frame.frame_id / "confidence.npy"
    if confidence_path.is_file():
        confidence = np.load(confidence_path, allow_pickle=False)
        if confidence.shape != labels.shape or not np.issubdtype(confidence.dtype, np.number):
            confidence = np.zeros(frame.point_count, dtype=np.float32)
        else:
            confidence = np.nan_to_num(confidence.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    else:
        confidence = np.zeros(frame.point_count, dtype=np.float32)
    return PointContext(labels=labels.astype(np.int32, copy=False), confidence=confidence, boundary=empty.boundary, source_path=str(semantic_path))


def semantic_boundary(semantic: np.ndarray) -> np.ndarray:
    arr = np.asarray(semantic)
    boundary = np.zeros(arr.shape, dtype=bool)
    if arr.ndim != 2 or arr.size == 0:
        return boundary
    boundary[1:, :] |= arr[1:, :] != arr[:-1, :]
    boundary[:-1, :] |= arr[:-1, :] != arr[1:, :]
    boundary[:, 1:] |= arr[:, 1:] != arr[:, :-1]
    boundary[:, :-1] |= arr[:, :-1] != arr[:, 1:]
    return boundary & (arr != 0)


def debug_feature_arrays(features: dict[str, Any]) -> dict[str, np.ndarray]:
    keys = [
        "distance",
        "intensity_norm",
        "reflectivity_norm",
        "return_count",
        "echo_rank_by_distance",
        "nearest_echo_distance_delta",
        "strongest_echo_intensity_delta_norm",
        "angular_density",
        "angular_row_count",
        "angular_col_count",
        "isolated",
        "has_camera_projection",
        "temporal_support",
        "temporal_neighbors",
        "temporal_stable",
        "temporal_transient",
        "below_ground",
        "valid_xyz",
        "sam3_label",
        "sam3_confidence",
        "sam3_boundary",
        "litept_label",
        "litept_confidence",
        "litept_noise_support",
        "litept_non_noise_context",
    ]
    return {key: np.asarray(features[key]) for key in keys}


def resolve_image_path(image_dir: Path | None, frame_id: str) -> Path | None:
    if image_dir is None:
        return None
    for suffix in (".jpg", ".jpeg", ".png", ".bmp"):
        candidate = image_dir / f"{frame_id}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _rule(
    rule_name: str,
    label: tuple[str, int] | None,
    score: np.ndarray,
    mask: np.ndarray,
    reason_masks: list[tuple[str, np.ndarray]],
) -> dict[str, Any]:
    label_name, label_id = label if label is not None else (rule_name, None)
    return {
        "rule": rule_name,
        "label_name": label_name,
        "label_id": label_id,
        "score": np.where(mask, score, 0.0).astype(np.float32, copy=False),
        "reason_masks": reason_masks,
    }


def _resolve_class(class_to_id: dict[str, int], *names: str) -> tuple[str, int] | None:
    by_casefold = {name.casefold(): (name, int(class_id)) for name, class_id in class_to_id.items()}
    for name in names:
        if name.casefold() in by_casefold:
            return by_casefold[name.casefold()]
    return None


def _name_for_id(class_to_id: dict[str, int], class_id: int) -> str:
    for name, current_id in class_to_id.items():
        if int(current_id) == int(class_id):
            return name
    return str(class_id)


def _noise_source_ids(class_to_id: dict[str, int], noise_groups: dict[str, list[str]]) -> set[int]:
    names = {name for members in noise_groups.values() for name in members}
    names.update(name for name in class_to_id if "noise" in name.casefold())
    return {int(class_to_id[name]) for name in names if name in class_to_id}


def _distance_map_by_slot_pixel(frame: HL320Frame) -> dict[tuple[int, int], np.ndarray]:
    slot = _field(frame, "slot", fill=np.nan)
    pixel = _field(frame, "pixel", fill=np.nan)
    distance = _field(frame, "distance", fill=np.nan)
    if not np.any(np.isfinite(distance)):
        distance = np.linalg.norm(frame.points[:, :3], axis=1).astype(np.float32)
    values: dict[tuple[int, int], list[float]] = {}
    for row in range(frame.point_count):
        if not (np.isfinite(slot[row]) and np.isfinite(pixel[row]) and np.isfinite(distance[row])):
            continue
        values.setdefault((int(slot[row]), int(pixel[row])), []).append(float(distance[row]))
    return {key: np.asarray(items, dtype=np.float32) for key, items in values.items()}


def _field(frame: HL320Frame, name: str, *, fill: float) -> np.ndarray:
    values = frame.fields.get(name)
    if values is None:
        return np.full(frame.point_count, fill, dtype=np.float32)
    return values.astype(np.float32, copy=False)


def _normalize_u8(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float32).copy()
    finite = out[np.isfinite(out)]
    if finite.size and np.max(finite) > 1.0:
        out /= 255.0
    out[~np.isfinite(out)] = 0.0
    return np.clip(out, 0.0, 1.0)


def _safe_percentile(values: np.ndarray, q: float, *, default: float) -> float:
    arr = np.asarray(values, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float(default)
    return float(np.percentile(finite, q))


def _class_counts(labels: np.ndarray, *, ignore_id: int) -> dict[int, int]:
    valid = np.asarray(labels) != int(ignore_id)
    if not np.any(valid):
        return {}
    ids, counts = np.unique(np.asarray(labels)[valid], return_counts=True)
    return {int(label_id): int(count) for label_id, count in zip(ids, counts)}


def _empty_summary() -> dict[str, Any]:
    return {
        "frames": 0,
        "points": 0,
        "candidate_points": 0,
        "class_counts": {},
        "rule_counts": {},
    }


def _accumulate_summary(summary: dict[str, Any], frame: dict[str, Any]) -> None:
    summary["frames"] += 1
    summary["points"] += int(frame["point_count"])
    summary["candidate_points"] += int(frame["candidate_points"])
    for class_id, count in frame.get("class_counts", {}).items():
        summary["class_counts"][str(class_id)] = int(summary["class_counts"].get(str(class_id), 0)) + int(count)
    for rule, count in frame.get("rule_counts", {}).items():
        summary["rule_counts"][str(rule)] = int(summary["rule_counts"].get(str(rule), 0)) + int(count)


def _require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def _prepare_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if any(path.iterdir()):
            if not overwrite:
                raise FileExistsError(f"Output directory is not empty: {path}. Use --overwrite to replace it.")
            shutil.rmtree(path)
        else:
            path.rmdir()
    path.mkdir(parents=True)
