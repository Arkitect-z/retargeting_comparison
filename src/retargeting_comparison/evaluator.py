"""Unified canonical quality, temporal, constraint, and artifact evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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


def _human_semantics(human: CanonicalHuman, frame_indices: np.ndarray) -> dict[str, np.ndarray]:
    index = {name: i for i, name in enumerate(human.joint_names.astype(str))}
    missing = [joint for joint in HUMAN_SEMANTIC_JOINTS.values() if joint not in index]
    if missing:
        raise ValueError(f"Canonical human source is missing semantic joints: {missing}")
    return {
        semantic: human.world_positions[frame_indices, index[joint]]
        for semantic, joint in HUMAN_SEMANTIC_JOINTS.items()
    }


def _static_height_scale(
    human_positions: dict[str, np.ndarray], robot_positions: dict[str, np.ndarray]
) -> float:
    human_foot = 0.5 * (human_positions["left_toe"][0] + human_positions["right_toe"][0])
    robot_foot = 0.5 * (robot_positions["left_toe"] + robot_positions["right_toe"])
    human_height = float(np.linalg.norm(human_positions["head"][0] - human_foot))
    robot_height = float(np.linalg.norm(robot_positions["head"] - robot_foot))
    if human_height < 0.5 or robot_height < 0.5:
        raise ValueError("Implausible source or robot height for static scaling")
    return robot_height / human_height


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
    scale = _static_height_scale(human_points, robot_frames[0])

    human_root_rotation = quaternion_wxyz_to_matrix(
        human.world_rotations[frame_indices, 0]
    )
    human_yaw = np.unwrap(yaw_from_matrix(human_root_rotation))
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
            robot_rf[semantic] - human_rf[semantic] * scale, axis=1
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
    source_root_delta = (human_root - human_root[0]) * scale
    robot_root_delta = robot_root - robot_root[0]
    root_translation = np.linalg.norm(robot_root_delta - source_root_delta, axis=1)
    yaw_error = np.abs(
        np.arctan2(
            np.sin((robot_yaw - robot_yaw[0]) - (human_yaw - human_yaw[0])),
            np.cos((robot_yaw - robot_yaw[0]) - (human_yaw - human_yaw[0])),
        )
    )

    dt = 1.0 / motion.fps
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
            foot_speed[0, side] = foot_speed[1, side]
    source_contacts = human.foot_contact_labels[frame_indices]
    skating = source_contacts & (foot_speed > 0.01)

    penetration = np.zeros(len(motion.qpos))
    self_contact_count = np.zeros(len(motion.qpos), dtype=np.int64)
    limit_violation = np.zeros(len(motion.qpos))
    for frame, qpos in enumerate(motion.qpos):
        penetration[frame], self_contact_count[frame] = robot.contact_diagnostics(qpos)
        limit_violation[frame] = robot.joint_limit_violation(qpos)
    invalid = ~np.asarray(motion.valid, dtype=bool)
    semantic_stack = np.stack(
        [robot_points[name] for name in HUMAN_SEMANTIC_JOINTS if name != "root"], axis=1
    )
    pose_jump = np.zeros(len(motion.qpos), dtype=np.float64)
    if len(motion.qpos) > 1:
        pose_jump[1:] = np.sqrt(
            np.mean(np.sum(np.diff(semantic_stack, axis=0) ** 2, axis=2), axis=1)
        )
    artifact = (
        invalid
        | (penetration > 0.01)
        | (limit_violation > 0.0)
        | np.any(skating, axis=1)
    )

    table = pd.DataFrame(
        {
            "source_frame_idx": frame_indices,
            "valid": motion.valid,
            "rf_kpe_all_m": all_kpe,
            "rf_kpe_targeted_m": targeted,
            "rf_kpe_untracked_m": untracked,
            "bone_direction_error_rad": np.mean(bone_angles, axis=1),
            "bend_plane_error_rad": np.mean(bend_plane, axis=1),
            "root_translation_error_m": root_translation,
            "root_yaw_error_rad": yaw_error,
            "joint_velocity_rms_rad_s": joint_velocity,
            "joint_acceleration_rms_rad_s2": joint_acceleration,
            "joint_jerk_rms_rad_s3": joint_jerk,
            "pose_jump_rms_m": pose_jump,
            "left_foot_speed_m_s": foot_speed[:, 0],
            "right_foot_speed_m_s": foot_speed[:, 1],
            "left_source_stance": source_contacts[:, 0],
            "right_source_stance": source_contacts[:, 1],
            "foot_skating": np.any(skating, axis=1),
            "ground_penetration_depth_m": penetration,
            "joint_limit_violation_rad": limit_violation,
            "self_contact_count_diagnostic": self_contact_count,
            "artifact": artifact,
            "solve_time_s": motion.per_frame_solve_time_s,
        }
    )
    summary: dict[str, Any] = {
        "method": motion.metadata.get("method", "unknown"),
        "frames": int(len(table)),
        "source_frames": int(len(human.timestamps)),
        "completion_ratio": float(len(table) / len(human.timestamps)),
        "completion_status": motion.metadata.get("completion_status"),
        "static_height_scale": scale,
        "rf_kpe_all_mean_m": float(table.rf_kpe_all_m.mean()),
        "rf_kpe_all_p95_m": _percentile(table.rf_kpe_all_m.to_numpy(), 95),
        "rf_kpe_targeted_mean_m": float(table.rf_kpe_targeted_m.mean()),
        "rf_kpe_untracked_mean_m": float(table.rf_kpe_untracked_m.mean()),
        "bone_direction_mean_rad": float(table.bone_direction_error_rad.mean()),
        "bend_plane_mean_rad": float(table.bend_plane_error_rad.mean()),
        "root_translation_mean_m": float(table.root_translation_error_m.mean()),
        "root_yaw_mean_rad": float(table.root_yaw_error_rad.mean()),
        "joint_velocity_rms_mean_rad_s": float(table.joint_velocity_rms_rad_s.mean()),
        "joint_acceleration_rms_mean_rad_s2": float(
            table.joint_acceleration_rms_rad_s2.mean()
        ),
        "joint_jerk_rms_p95_rad_s3": _percentile(
            table.joint_jerk_rms_rad_s3.to_numpy(), 95
        ),
        "pose_jump_p95_m": _percentile(table.pose_jump_rms_m.to_numpy(), 95),
        "foot_skating_frame_rate": float(table.foot_skating.mean()),
        "ground_penetration_frame_rate": float(
            (table.ground_penetration_depth_m > 0.01).mean()
        ),
        "ground_penetration_max_m": float(table.ground_penetration_depth_m.max()),
        "joint_limit_violation_frame_rate": float(
            (table.joint_limit_violation_rad > 0).mean()
        ),
        "artifact_rate": float(table.artifact.mean()),
        "solve_time_median_s": float(table.solve_time_s.median()),
        "solve_time_p95_s": _percentile(table.solve_time_s.to_numpy(), 95),
        "self_collision_in_primary_metric": False,
        "robot_model_sha256": robot.sha256,
    }
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
