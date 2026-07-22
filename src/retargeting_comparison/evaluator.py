"""Unified canonical quality, temporal, constraint, and artifact evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .calibration import (
    EVALUATOR_SCHEMA_VERSION,
    build_evaluator_protocol,
    human_heading_yaw,
)
from .robot_model import CanonicalRobotModel
from .rotations import quaternion_wxyz_to_matrix, yaw_from_matrix
from .schemas import CanonicalG1, CanonicalHuman

HUMAN_SEMANTIC_JOINTS = {
    "root": "Hips",
    "torso": "Spine2",
    "head": "Head",
    "left_shoulder": "LeftArm",
    "right_shoulder": "RightArm",
    "left_elbow": "LeftForeArm",
    "right_elbow": "RightForeArm",
    "left_wrist": "LeftHand",
    "right_wrist": "RightHand",
    "left_hip": "LeftUpLeg",
    "right_hip": "RightUpLeg",
    "left_knee": "LeftLeg",
    "right_knee": "RightLeg",
    "left_ankle": "LeftFoot",
    "right_ankle": "RightFoot",
    "left_toe": "LeftToe",
    "right_toe": "RightToe",
}

TARGETED = ("left_wrist", "right_wrist", "left_ankle", "right_ankle")
UNTRACKED = tuple(
    name for name in HUMAN_SEMANTIC_JOINTS if name not in TARGETED and name != "root"
)

BONES = (
    ("root", "torso"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
)

BEND_CHAINS = (
    ("left_shoulder", "left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow", "right_wrist"),
    ("left_hip", "left_knee", "left_ankle"),
    ("right_hip", "right_knee", "right_ankle"),
)


def _percentile(value: np.ndarray, q: float) -> float:
    finite = np.asarray(value, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.percentile(finite, q)) if len(finite) else float("nan")


def _mean_or_zero(value: np.ndarray) -> float:
    finite = np.asarray(value, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if len(finite) else 0.0


def _human_semantics(human: CanonicalHuman, frame_indices: np.ndarray) -> dict[str, np.ndarray]:
    index = {name: i for i, name in enumerate(human.joint_names.astype(str))}
    missing = [joint for joint in HUMAN_SEMANTIC_JOINTS.values() if joint not in index]
    if missing:
        raise ValueError(f"Canonical human source is missing semantic joints: {missing}")
    return {
        semantic: human.world_positions[frame_indices, index[joint]]
        for semantic, joint in HUMAN_SEMANTIC_JOINTS.items()
    }


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    return vector / np.maximum(norm, 1e-12)


def _angle_between(left: np.ndarray, right: np.ndarray, *, unsigned: bool = False) -> np.ndarray:
    dot = np.sum(_unit(left) * _unit(right), axis=-1)
    if unsigned:
        dot = np.abs(dot)
    return np.arccos(np.clip(dot, -1.0, 1.0))


def _root_frame_vectors(
    points: dict[str, np.ndarray], root: np.ndarray, yaw: np.ndarray
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    cosine = np.cos(-yaw)
    sine = np.sin(-yaw)
    for name, point in points.items():
        local = point - root
        result[name] = np.column_stack(
            [
                cosine * local[:, 0] - sine * local[:, 1],
                sine * local[:, 0] + cosine * local[:, 1],
                local[:, 2],
            ]
        )
    return result


def evaluate_motion(
    human: CanonicalHuman,
    motion: CanonicalG1,
    robot: CanonicalRobotModel,
    protocol: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    human.validate()
    motion.validate(source_frame_count=len(human.timestamps))
    frame_indices = np.asarray(motion.source_frame_idx, dtype=np.int64)
    if frame_indices.max(initial=-1) >= len(human.timestamps):
        raise ValueError("G1 source_frame_idx exceeds canonical human frames")
    human_points = _human_semantics(human, frame_indices)
    robot_frames = [robot.semantic_positions(qpos) for qpos in motion.qpos]
    robot_points = {
        semantic: np.stack([frame[semantic] for frame in robot_frames])
        for semantic in HUMAN_SEMANTIC_JOINTS
    }
    if protocol is None:
        protocol = build_evaluator_protocol(
            human, robot, source_path="in-memory-canonical-source"
        )
    if int(protocol.get("schema_version", 0)) != EVALUATOR_SCHEMA_VERSION:
        raise ValueError(
            f"Evaluator protocol v{EVALUATOR_SCHEMA_VERSION} is required"
        )
    method = str(motion.metadata.get("method", "unknown"))
    timeline = protocol.get("timeline", {})
    relative_fps_tolerance = float(
        timeline.get("accepted_declared_fps_relative_tolerance", 2e-5)
    )
    if (
        not np.isfinite(motion.fps)
        or abs(float(motion.fps) - float(human.fps)) / float(human.fps)
        > relative_fps_tolerance
    ):
        raise ValueError(
            "G1 declared fps is incompatible with the canonical source timeline: "
            f"motion={motion.fps}, source={human.fps}"
        )
    declared_source_hash = motion.metadata.get("canonical_source_sha256")
    if declared_source_hash is not None and declared_source_hash != human.source_sha256:
        raise ValueError("G1 metadata canonical_source_sha256 does not match the source")
    if (
        declared_source_hash is None
        and motion.metadata.get("method_family") != "external_precomputed_reference"
        and method != "synthetic"
    ):
        raise ValueError("G1 metadata is missing canonical_source_sha256")
    local_scale = float(protocol["scale"]["common_local_body_scale"])
    root_scale = float(protocol["scale"]["common_root_displacement_scale"])
    native_scale_entry = protocol.get("native_root_scales", {}).get(method)
    if native_scale_entry is None:
        native_scale = root_scale
        native_scale_xyz = np.full(3, root_scale, dtype=np.float64)
        native_scale_policy = "undeclared; common scale fallback"
    elif "value_xyz" in native_scale_entry:
        native_scale_xyz = np.asarray(native_scale_entry["value_xyz"], dtype=np.float64)
        if native_scale_xyz.shape != (3,) or not np.isfinite(native_scale_xyz).all():
            raise ValueError("native_root_scales.value_xyz must be three finite values")
        native_scale = float(np.mean(native_scale_xyz[:2]))
        native_scale_policy = str(native_scale_entry["policy"])
    else:
        native_scale = float(native_scale_entry["value"])
        native_scale_xyz = np.full(3, native_scale, dtype=np.float64)
        native_scale_policy = str(native_scale_entry["policy"])

    human_yaw = human_heading_yaw(human, frame_indices)
    robot_rotation = quaternion_wxyz_to_matrix(motion.qpos[:, 3:7])
    robot_yaw = np.unwrap(yaw_from_matrix(robot_rotation))
    human_root = human_points["root"]
    robot_root = robot_points["root"]

    human_rf = _root_frame_vectors(human_points, human_root, human_yaw)
    robot_rf = _root_frame_vectors(robot_points, robot_root, robot_yaw)
    errors: dict[str, np.ndarray] = {}
    for semantic in HUMAN_SEMANTIC_JOINTS:
        if semantic == "root":
            continue
        errors[semantic] = np.linalg.norm(
            robot_rf[semantic] - human_rf[semantic] * local_scale, axis=1
        )

    bone_angles = np.stack(
        [
            _angle_between(
                human_rf[child] - human_rf[parent],
                robot_rf[child] - robot_rf[parent],
            )
            for parent, child in BONES
        ],
        axis=1,
    )
    bend_angles = []
    for proximal, middle, distal in BEND_CHAINS:
        human_normal = np.cross(
            human_rf[middle] - human_rf[proximal],
            human_rf[distal] - human_rf[middle],
        )
        robot_normal = np.cross(
            robot_rf[middle] - robot_rf[proximal],
            robot_rf[distal] - robot_rf[middle],
        )
        bend_angles.append(_angle_between(human_normal, robot_normal, unsigned=True))
    bend_plane = np.stack(bend_angles, axis=1)

    targeted = np.mean(np.stack([errors[name] for name in TARGETED]), axis=0)
    untracked = np.mean(np.stack([errors[name] for name in UNTRACKED]), axis=0)
    all_kpe = np.mean(np.stack(list(errors.values())), axis=0)
    source_root_delta = (human_root - human_root[0]) * root_scale
    robot_root_delta = robot_root - robot_root[0]
    root_translation = np.linalg.norm(robot_root_delta - source_root_delta, axis=1)
    native_root_translation = np.linalg.norm(
        robot_root_delta - (human_root - human_root[0]) * native_scale_xyz,
        axis=1,
    )
    human_root_xy = human_root[:, :2] - human_root[0, :2]
    robot_root_xy = robot_root[:, :2] - robot_root[0, :2]
    root_xy_energy = float(np.sum(human_root_xy**2))
    effective_root_scale = (
        float(np.sum(human_root_xy * robot_root_xy) / root_xy_energy)
        if root_xy_energy > 1e-12
        else float("nan")
    )
    # Fit XY path scale only to XY and keep vertical morphology referenced to
    # the frozen common scale.  Applying an XY fit to Z conflates path shape
    # with ground and height errors.
    scale_invariant_root_xy = np.linalg.norm(
        robot_root_xy - human_root_xy * effective_root_scale,
        axis=1,
    )
    root_vertical_common_scale = np.abs(
        robot_root_delta[:, 2]
        - (human_root[:, 2] - human_root[0, 2]) * root_scale
    )
    scale_invariant_root_translation = np.sqrt(
        scale_invariant_root_xy**2 + root_vertical_common_scale**2
    )
    yaw_delta = robot_yaw - human_yaw
    yaw_error = np.abs(
        np.arctan2(np.sin(yaw_delta), np.cos(yaw_delta))
    )

    # Every method is scored on the canonical source clock.  Rounded nominal
    # 30 Hz declarations remain provenance only and cannot change a metric.
    dt = 1.0 / human.fps
    joint_velocity = np.zeros(len(motion.qpos))
    joint_acceleration = np.zeros(len(motion.qpos))
    joint_jerk = np.zeros(len(motion.qpos))
    if len(motion.qpos) > 1:
        joint_delta = np.arctan2(
            np.sin(np.diff(motion.qpos[:, 7:], axis=0)),
            np.cos(np.diff(motion.qpos[:, 7:], axis=0)),
        )
        velocity = joint_delta / dt
        joint_velocity[1:] = np.sqrt(np.mean(velocity**2, axis=1))
        if len(velocity) > 1:
            acceleration = np.diff(velocity, axis=0) / dt
            joint_acceleration[2:] = np.sqrt(np.mean(acceleration**2, axis=1))
            if len(acceleration) > 1:
                jerk = np.diff(acceleration, axis=0) / dt
                joint_jerk[3:] = np.sqrt(np.mean(jerk**2, axis=1))

    foot_speed = np.zeros((len(motion.qpos), 2), dtype=np.float64)
    for side, semantic in enumerate(("left_toe", "right_toe")):
        if len(motion.qpos) > 1:
            foot_speed[1:, side] = (
                np.linalg.norm(np.diff(robot_points[semantic][:, :2], axis=0), axis=1)
                / dt
            )
    source_contacts = human.foot_contact_labels[frame_indices]
    skating_threshold = float(protocol["thresholds"]["foot_skating_speed_m_s"])
    penetration_threshold = float(protocol["thresholds"]["ground_penetration_m"])
    skating = source_contacts & (foot_speed > skating_threshold)

    penetration = np.zeros(len(motion.qpos))
    self_contact_count = np.zeros(len(motion.qpos), dtype=np.int64)
    limit_violation = np.zeros(len(motion.qpos))
    for frame, qpos in enumerate(motion.qpos):
        penetration[frame], self_contact_count[frame] = robot.contact_diagnostics(qpos)
        limit_violation[frame] = robot.joint_limit_violation(
            qpos,
            tolerance=float(protocol["thresholds"]["joint_limit_tolerance_rad"]),
        )
    invalid = ~np.asarray(motion.valid, dtype=bool)
    semantic_stack = np.stack(
        [robot_points[name] for name in HUMAN_SEMANTIC_JOINTS if name != "root"], axis=1
    )
    pose_jump = np.zeros(len(motion.qpos), dtype=np.float64)
    if len(motion.qpos) > 1:
        pose_jump[1:] = np.sqrt(
            np.mean(np.sum(np.diff(semantic_stack, axis=0) ** 2, axis=2), axis=1)
        )
    penetration_artifact = penetration > penetration_threshold
    limit_artifact = limit_violation > 0.0
    skating_artifact = np.any(skating, axis=1)
    artifact = invalid | penetration_artifact | limit_artifact | skating_artifact
    artifact_causes = np.asarray(
        [
            "+".join(
                cause
                for cause, active in (
                    ("invalid", bool(invalid[frame])),
                    ("penetration", bool(penetration_artifact[frame])),
                    ("joint-limit", bool(limit_artifact[frame])),
                    ("left-skating", bool(skating[frame, 0])),
                    ("right-skating", bool(skating[frame, 1])),
                )
                if active
            )
            or "none"
            for frame in range(len(motion.qpos))
        ],
        dtype=object,
    )

    columns: dict[str, Any] = {
            "source_frame_idx": frame_indices,
            "valid": motion.valid,
            "rf_kpe_all_m": all_kpe,
            "rf_kpe_targeted_m": targeted,
            "rf_kpe_untracked_m": untracked,
            "bone_direction_error_rad": np.mean(bone_angles, axis=1),
            "bend_plane_error_rad": np.mean(bend_plane, axis=1),
            "root_translation_error_m": root_translation,
            "root_translation_common_scale_error_m": root_translation,
            "root_translation_native_scale_error_m": native_root_translation,
            "root_translation_scale_invariant_error_m": scale_invariant_root_translation,
            "root_translation_scale_invariant_xy_error_m": scale_invariant_root_xy,
            "root_vertical_common_scale_error_m": root_vertical_common_scale,
            "effective_root_xy_scale": np.full(len(motion.qpos), effective_root_scale),
            "root_scale_bias_fraction": np.full(
                len(motion.qpos), effective_root_scale / root_scale - 1.0
            ),
            "root_yaw_error_rad": yaw_error,
            "joint_velocity_rms_rad_s": joint_velocity,
            "joint_acceleration_rms_rad_s2": joint_acceleration,
            "joint_jerk_rms_rad_s3": joint_jerk,
            "pose_jump_rms_m": pose_jump,
            "left_foot_speed_m_s": foot_speed[:, 0],
            "right_foot_speed_m_s": foot_speed[:, 1],
            "left_source_stance": source_contacts[:, 0],
            "right_source_stance": source_contacts[:, 1],
            "left_foot_skating": skating[:, 0],
            "right_foot_skating": skating[:, 1],
            "foot_skating": skating_artifact,
            "ground_penetration_depth_m": penetration,
            "ground_penetration_artifact": penetration_artifact,
            "joint_limit_violation_rad": limit_violation,
            "joint_limit_artifact": limit_artifact,
            "invalid_artifact": invalid,
            "self_contact_count_diagnostic": self_contact_count,
            "artifact": artifact,
            "artifact_causes": artifact_causes,
            "solve_time_s": motion.per_frame_solve_time_s,
    }
    for semantic, values in errors.items():
        columns[f"rf_kpe_{semantic}_m"] = values
    table = pd.DataFrame(columns)
    stance_speeds = foot_speed[source_contacts]
    stance_heights = np.concatenate(
        [
            robot_points[semantic][source_contacts[:, side], 2]
            for side, semantic in enumerate(("left_toe", "right_toe"))
        ]
    )
    summary: dict[str, Any] = {
        "method": method,
        "evaluator_schema_version": EVALUATOR_SCHEMA_VERSION,
        "evaluator_protocol_sha256": protocol.get("manifest_sha256"),
        "frames": int(len(table)),
        "source_frames": int(len(human.timestamps)),
        "completion_ratio": float(len(table) / len(human.timestamps)),
        "completion_status": motion.metadata.get("completion_status"),
        "canonical_source_sha256": human.source_sha256,
        "source_fps": float(human.fps),
        "method_declared_fps": float(motion.fps),
        "temporal_metric_fps": float(human.fps),
        "timeline_policy": timeline.get(
            "definition", "canonical source timestamps; nominal 30 Hz tolerated"
        ),
        "static_height_scale": local_scale,
        "common_static_scale": local_scale,
        "common_local_body_scale": local_scale,
        "common_root_displacement_scale": root_scale,
        "native_root_scale": native_scale,
        "native_root_scale_x": float(native_scale_xyz[0]),
        "native_root_scale_y": float(native_scale_xyz[1]),
        "native_root_scale_z": float(native_scale_xyz[2]),
        "native_root_scale_policy": native_scale_policy,
        "effective_root_xy_scale": effective_root_scale,
        "root_scale_bias_fraction": effective_root_scale / root_scale - 1.0,
        "rf_kpe_all_mean_m": float(table.rf_kpe_all_m.mean()),
        "rf_kpe_all_p95_m": _percentile(table.rf_kpe_all_m.to_numpy(), 95),
        "rf_kpe_targeted_mean_m": float(table.rf_kpe_targeted_m.mean()),
        "rf_kpe_targeted_p95_m": _percentile(
            table.rf_kpe_targeted_m.to_numpy(), 95
        ),
        "rf_kpe_untracked_mean_m": float(table.rf_kpe_untracked_m.mean()),
        "rf_kpe_untracked_p95_m": _percentile(
            table.rf_kpe_untracked_m.to_numpy(), 95
        ),
        "bone_direction_mean_rad": float(table.bone_direction_error_rad.mean()),
        "bone_direction_p95_rad": _percentile(
            table.bone_direction_error_rad.to_numpy(), 95
        ),
        "bend_plane_mean_rad": float(table.bend_plane_error_rad.mean()),
        "bend_plane_p95_rad": _percentile(
            table.bend_plane_error_rad.to_numpy(), 95
        ),
        "root_translation_mean_m": float(table.root_translation_error_m.mean()),
        "root_translation_common_scale_mean_m": float(
            table.root_translation_common_scale_error_m.mean()
        ),
        "root_translation_common_scale_p95_m": _percentile(
            table.root_translation_common_scale_error_m.to_numpy(), 95
        ),
        "root_translation_native_scale_mean_m": float(
            table.root_translation_native_scale_error_m.mean()
        ),
        "root_translation_native_scale_p95_m": _percentile(
            table.root_translation_native_scale_error_m.to_numpy(), 95
        ),
        "root_translation_scale_invariant_mean_m": float(
            table.root_translation_scale_invariant_error_m.mean()
        ),
        "root_translation_scale_invariant_p95_m": _percentile(
            table.root_translation_scale_invariant_error_m.to_numpy(), 95
        ),
        "root_translation_scale_invariant_xy_mean_m": float(
            table.root_translation_scale_invariant_xy_error_m.mean()
        ),
        "root_vertical_common_scale_mean_m": float(
            table.root_vertical_common_scale_error_m.mean()
        ),
        "root_yaw_mean_rad": float(table.root_yaw_error_rad.mean()),
        "root_yaw_p95_rad": _percentile(table.root_yaw_error_rad.to_numpy(), 95),
        "joint_velocity_rms_mean_rad_s": float(table.joint_velocity_rms_rad_s.mean()),
        "joint_velocity_rms_p95_rad_s": _percentile(
            table.joint_velocity_rms_rad_s.to_numpy(), 95
        ),
        "joint_velocity_rms_max_rad_s": float(table.joint_velocity_rms_rad_s.max()),
        "joint_acceleration_rms_mean_rad_s2": float(
            table.joint_acceleration_rms_rad_s2.mean()
        ),
        "joint_acceleration_rms_p95_rad_s2": _percentile(
            table.joint_acceleration_rms_rad_s2.to_numpy(), 95
        ),
        "joint_acceleration_rms_max_rad_s2": float(
            table.joint_acceleration_rms_rad_s2.max()
        ),
        "joint_jerk_rms_p95_rad_s3": _percentile(
            table.joint_jerk_rms_rad_s3.to_numpy(), 95
        ),
        "joint_jerk_rms_mean_rad_s3": float(table.joint_jerk_rms_rad_s3.mean()),
        "joint_jerk_rms_max_rad_s3": float(table.joint_jerk_rms_rad_s3.max()),
        "pose_jump_mean_m": float(table.pose_jump_rms_m.mean()),
        "pose_jump_p95_m": _percentile(table.pose_jump_rms_m.to_numpy(), 95),
        "pose_jump_max_m": float(table.pose_jump_rms_m.max()),
        "foot_skating_frame_rate": float(table.foot_skating.mean()),
        "foot_skating_duration_s": float(table.foot_skating.sum() * dt),
        "stance_foot_speed_p95_m_s": (
            _percentile(stance_speeds, 95) if len(stance_speeds) else 0.0
        ),
        "stance_foot_speed_max_m_s": float(np.max(stance_speeds, initial=0.0)),
        "stance_foot_height_mean_m": _mean_or_zero(stance_heights),
        "stance_foot_height_p95_m": (
            _percentile(stance_heights, 95) if len(stance_heights) else 0.0
        ),
        "ground_penetration_frame_rate": float(
            table.ground_penetration_artifact.mean()
        ),
        "ground_penetration_p95_m": _percentile(
            table.ground_penetration_depth_m.to_numpy(), 95
        ),
        "ground_penetration_max_m": float(table.ground_penetration_depth_m.max()),
        "ground_penetration_max_frame": int(
            table.ground_penetration_depth_m.to_numpy().argmax()
        ),
        "joint_limit_violation_frame_rate": float(
            table.joint_limit_artifact.mean()
        ),
        "joint_limit_violation_p95_rad": _percentile(
            table.joint_limit_violation_rad.to_numpy(), 95
        ),
        "invalid_frame_rate": float(table.invalid_artifact.mean()),
        "artifact_rate": float(table.artifact.mean()),
        "solve_time_median_s": float(table.solve_time_s.median()),
        "solve_time_p95_s": _percentile(table.solve_time_s.to_numpy(), 95),
        "self_collision_in_primary_metric": False,
        "robot_model_sha256": robot.sha256,
    }
    native_limit = (
        motion.metadata.get("joint_limit_diagnostics", {})
        .get("protomotions_native")
    )
    if isinstance(native_limit, dict):
        summary.update(
            {
                "native_joint_limit_diagnostic_available": True,
                "native_joint_limit_asset_sha256": native_limit.get("urdf_sha256"),
                "native_joint_limit_violation_frame_rate": float(
                    native_limit.get("violating_frame_count", 0)
                )
                / len(table),
                "native_joint_limit_max_violation_rad": float(
                    native_limit.get("maximum_violation_rad", 0.0)
                ),
            }
        )
    else:
        summary.update(
            {
                "native_joint_limit_diagnostic_available": False,
                "native_joint_limit_asset_sha256": None,
                "native_joint_limit_violation_frame_rate": None,
                "native_joint_limit_max_violation_rad": None,
            }
        )
    return table, summary


def save_evaluation(
    table: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: str | Path,
    run_id: str,
) -> tuple[Path, Path, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / f"{run_id}_per_frame.csv"
    parquet_path = root / f"{run_id}_per_frame.parquet"
    summary_path = root / f"{run_id}_summary.json"
    table.to_csv(csv_path, index=False)
    table.to_parquet(parquet_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return csv_path, parquet_path, summary_path
