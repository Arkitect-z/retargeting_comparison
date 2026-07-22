"""Method-independent Stage 1 scale and heading calibration.

The public pipelines intentionally use different native scaling policies.  A
benchmark must not infer its reference scale from a method output, because that
makes the reference method-dependent.  This module freezes one common scale
from homologous source/robot landmarks and records native method scales
separately so scale choice and tracking error can be reported independently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import atomic_write_yaml, load_yaml, sha256_file
from .robot_model import CanonicalRobotModel
from .schemas import CanonicalHuman


EVALUATOR_SCHEMA_VERSION = 2
HEADING_DEFINITION = "z_up_cross_mean_left_to_right_hips_shoulders"
SCALE_DEFINITION = "neutral_g1_head_to_mean_toe_over_source_frame0_head_to_mean_toe"


def _human_indices(human: CanonicalHuman) -> dict[str, int]:
    return {name: index for index, name in enumerate(human.joint_names.astype(str))}


def human_heading_yaw(
    human: CanonicalHuman, frame_indices: np.ndarray | None = None
) -> np.ndarray:
    """Return a geometry-defined LAFAN heading in the canonical Z-up frame.

    BVH root quaternion axes are not the same as the G1 pelvis axes.  Heading is
    therefore derived from the body lateral axis rather than from a format-
    specific root quaternion.  With left/right joints in their declared order,
    ``up x lateral`` points forward in the frozen canonical convention.
    """

    indices = _human_indices(human)
    required = ("LeftUpLeg", "RightUpLeg", "LeftArm", "RightArm")
    missing = [name for name in required if name not in indices]
    if missing:
        raise ValueError(f"Cannot derive source heading; missing joints: {missing}")
    frames = (
        np.arange(len(human.timestamps), dtype=np.int64)
        if frame_indices is None
        else np.asarray(frame_indices, dtype=np.int64)
    )
    positions = human.world_positions[frames]
    hip_lateral = (
        positions[:, indices["RightUpLeg"]] - positions[:, indices["LeftUpLeg"]]
    )
    shoulder_lateral = (
        positions[:, indices["RightArm"]] - positions[:, indices["LeftArm"]]
    )
    lateral = 0.5 * (hip_lateral + shoulder_lateral)
    lateral_xy_norm = np.linalg.norm(lateral[:, :2], axis=1)
    if np.any(lateral_xy_norm < 1e-5):
        raise ValueError("Degenerate hip/shoulder lateral axis in source motion")
    up = np.asarray([0.0, 0.0, 1.0])
    forward = np.cross(np.broadcast_to(up, lateral.shape), lateral)
    return np.unwrap(np.arctan2(forward[:, 1], forward[:, 0]))


def common_landmark_scale(
    human: CanonicalHuman, robot: CanonicalRobotModel
) -> tuple[float, float, float]:
    """Return common scale, source landmark height, and neutral G1 height."""

    indices = _human_indices(human)
    required = ("Head", "LeftToe", "RightToe")
    missing = [name for name in required if name not in indices]
    if missing:
        raise ValueError(f"Cannot calibrate scale; missing source joints: {missing}")
    human_foot = 0.5 * (
        human.world_positions[0, indices["LeftToe"]]
        + human.world_positions[0, indices["RightToe"]]
    )
    human_height = float(
        np.linalg.norm(human.world_positions[0, indices["Head"]] - human_foot)
    )
    neutral = robot.semantic_positions(robot.model.qpos0.copy())
    robot_foot = 0.5 * (neutral["left_toe"] + neutral["right_toe"])
    robot_height = float(np.linalg.norm(neutral["head"] - robot_foot))
    if human_height < 0.5 or robot_height < 0.5:
        raise ValueError("Implausible source or neutral-robot landmark height")
    return robot_height / human_height, human_height, robot_height


def build_evaluator_protocol(
    human: CanonicalHuman,
    robot: CanonicalRobotModel,
    *,
    source_path: str | Path,
) -> dict[str, Any]:
    """Build the frozen evaluator-v2 protocol for the selected Pilot."""

    common_scale, human_height, robot_height = common_landmark_scale(human, robot)
    return {
        "schema_version": EVALUATOR_SCHEMA_VERSION,
        "name": "stage1-pilot-evaluator-v2",
        "source_sha256": human.source_sha256,
        "source_path": str(source_path),
        "robot_xml": str(robot.xml_path),
        "robot_xml_sha256": robot.sha256,
        "heading": {
            "definition": HEADING_DEFINITION,
            "uses_bvh_root_quaternion": False,
            "robot_definition": "canonical_qpos_root_x_axis_yaw",
        },
        "scale": {
            "definition": SCALE_DEFINITION,
            "method_independent": True,
            "source_frame": 0,
            "source_landmark_height_m": human_height,
            "neutral_robot_landmark_height_m": robot_height,
            "common_static_scale": common_scale,
            "source_landmarks": ["Head", "mean(LeftToe,RightToe)"],
            "robot_landmarks": ["head", "mean(left_toe,right_toe)"],
        },
        "native_root_scales": {
            "sparse-neutral": {
                "value": common_scale,
                "policy": "controlled-v2 common benchmark scale",
            },
            "sparse-a": {
                "value": common_scale,
                "policy": "controlled-v2 common benchmark scale",
            },
            "sparse-b": {
                "value": common_scale,
                "policy": "controlled-v2 common benchmark scale",
            },
            "dense": {
                "value": common_scale,
                "policy": "controlled-v2 common benchmark scale",
            },
            "gmr": {
                "value": 0.9 * (1.75 / 1.8),
                "policy": "official GMR Hips scale 0.9 times hard-coded LAFAN height 1.75 / assumption 1.8",
            },
            "omniretarget": {
                "value": 1.27 / 1.7,
                "policy": "official Holosoma LAFAN default_scale_factor",
            },
        },
        "thresholds": {
            "foot_skating_speed_m_s": 0.01,
            "ground_penetration_m": 0.01,
            "joint_limit_tolerance_rad": 1e-4,
            "minimum_completion_ratio": 0.95,
        },
        "reporting": {
            "common_scale_error": "primary morphology-referenced root metric",
            "native_scale_error": "solver tracking at each public pipeline's declared scale",
            "scale_invariant_error": "root path shape after one sequence-level scalar fit",
            "native_scale_is_not_used_for_rf_kpe": True,
        },
        "full_lafan_authorized": False,
    }


def freeze_evaluator_protocol(
    human: CanonicalHuman,
    robot: CanonicalRobotModel,
    output_path: str | Path,
    *,
    source_path: str | Path,
) -> dict[str, Any]:
    value = build_evaluator_protocol(human, robot, source_path=source_path)
    output = Path(output_path).resolve()
    repo_root = output.parent.parent
    try:
        value["robot_xml"] = robot.xml_path.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        value["robot_xml"] = str(robot.xml_path)
    atomic_write_yaml(output_path, value)
    return value


def load_evaluator_protocol(path: str | Path) -> dict[str, Any]:
    value = load_yaml(path)
    if int(value.get("schema_version", 0)) != EVALUATOR_SCHEMA_VERSION:
        raise ValueError("Stage 1 requires evaluator schema version 2")
    scale = value.get("scale", {})
    if scale.get("method_independent") is not True:
        raise ValueError("Evaluator common scale must be method-independent")
    common = float(scale.get("common_static_scale", float("nan")))
    if not np.isfinite(common) or not 0.3 < common < 1.5:
        raise ValueError("Evaluator common scale is missing or implausible")
    return value


def protocol_hash(path: str | Path) -> str:
    return sha256_file(path)
