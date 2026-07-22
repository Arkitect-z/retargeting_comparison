"""Method-independent Stage 1 scale and heading calibration.

The public pipelines intentionally use different native scaling policies.  A
benchmark must not infer its reference scale from a method output, because that
makes the reference method-dependent.  This module freezes one common scale
from homologous source/robot landmarks and records native method scales
separately so scale choice and tracking error can be reported independently.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .io_utils import atomic_write_yaml, load_yaml, sha256_file
from .schemas import CanonicalHuman

if TYPE_CHECKING:
    from .robot_model import CanonicalRobotModel


EVALUATOR_SCHEMA_VERSION = 3
HEADING_DEFINITION = "z_up_cross_mean_left_to_right_hips_shoulders"
SCALE_DEFINITION = (
    "frame0_root_heading_aligned_shared_semantic_landmark_weighted_scalar_"
    "least_squares"
)
HEAD_TO_TOE_DIAGNOSTIC = (
    "neutral_g1_head_to_mean_toe_over_source_frame0_head_to_mean_toe"
)

# These are joint centres represented by a real body/site in the canonical
# Holosoma G1 scene and by an unambiguous LAFAN joint.  Torso is excluded
# because the LAFAN Spine2 joint and G1 torso-link origin are not homologous;
# elbows/wrists are excluded because frame-0 human and neutral-G1 arm poses are
# different.  The registered set was frozen before the corrected experiment.
COMMON_SCALE_LANDMARKS: tuple[tuple[str, str, float], ...] = (
    ("Head", "head", 1.0),
    ("LeftArm", "left_shoulder", 1.0),
    ("RightArm", "right_shoulder", 1.0),
    ("LeftUpLeg", "left_hip", 1.0),
    ("RightUpLeg", "right_hip", 1.0),
    ("LeftLeg", "left_knee", 1.0),
    ("RightLeg", "right_knee", 1.0),
    ("LeftFoot", "left_ankle", 1.0),
    ("RightFoot", "right_ankle", 1.0),
    ("LeftToe", "left_toe", 1.0),
    ("RightToe", "right_toe", 1.0),
)


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


def _head_to_toe_diagnostic(
    human: CanonicalHuman, robot: CanonicalRobotModel
) -> tuple[float, float, float]:
    """Return the legacy head/toe span ratio as a diagnostic only."""

    indices = _human_indices(human)
    required = ("Head", "LeftToe", "RightToe")
    missing = [name for name in required if name not in indices]
    if missing:
        raise ValueError(f"Cannot diagnose head/toe span; missing source joints: {missing}")
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


def shared_landmark_least_squares_calibration(
    human: CanonicalHuman, robot: CanonicalRobotModel
) -> dict[str, Any]:
    """Fit the registered method-independent scalar common-scale policy.

    Source and robot landmarks are expressed relative to their respective
    roots.  The source vectors are rotated by the geometry-defined frame-0
    heading so both point sets use canonical +X forward.  The scalar is then
    the closed-form no-intercept weighted least-squares solution required by
    the registered protocol::

        s = sum_k w_k <h_k-h_root, r_k-r_root>
            / sum_k w_k ||h_k-h_root||^2.

    This function does not choose the root-path gain or world anchor.  Stage 1
    freezes those as distinct policy fields, even though the registered root
    gain deliberately has the same numeric value as the local/body gain.
    """

    human.validate()
    indices = _human_indices(human)
    missing = (["Hips"] if "Hips" not in indices else []) + [
        source for source, _, _ in COMMON_SCALE_LANDMARKS if source not in indices
    ]
    if missing:
        raise ValueError(
            f"Cannot fit shared-landmark scale; missing source joints: {missing}"
        )
    neutral = robot.semantic_positions(robot.model.qpos0.copy())
    missing_robot = [
        semantic
        for _, semantic, _ in COMMON_SCALE_LANDMARKS
        if semantic not in neutral
    ]
    if missing_robot:
        raise ValueError(
            f"Cannot fit shared-landmark scale; missing robot frames: {missing_robot}"
        )
    source_root = np.asarray(human.world_positions[0, indices["Hips"]], dtype=float)
    robot_root = np.asarray(neutral["root"], dtype=float)
    source_yaw = float(human_heading_yaw(human, np.asarray([0], dtype=np.int64))[0])
    cosine = float(np.cos(-source_yaw))
    sine = float(np.sin(-source_yaw))
    source_to_canonical = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    source_vectors = np.stack(
        [
            source_to_canonical
            @ (human.world_positions[0, indices[source]] - source_root)
            for source, _, _ in COMMON_SCALE_LANDMARKS
        ]
    )
    robot_vectors = np.stack(
        [
            np.asarray(neutral[semantic], dtype=float) - robot_root
            for _, semantic, _ in COMMON_SCALE_LANDMARKS
        ]
    )
    weights = np.asarray(
        [weight for _, _, weight in COMMON_SCALE_LANDMARKS], dtype=np.float64
    )
    numerator = float(np.sum(weights[:, None] * source_vectors * robot_vectors))
    denominator = float(np.sum(weights[:, None] * source_vectors * source_vectors))
    if not np.isfinite(denominator) or denominator <= 1e-12:
        raise ValueError("Degenerate shared-landmark least-squares denominator")
    scale = numerator / denominator
    if not np.isfinite(scale) or not 0.3 < scale < 1.5:
        raise ValueError(f"Implausible shared-landmark least-squares scale: {scale}")
    residual = scale * source_vectors - robot_vectors
    per_landmark = np.linalg.norm(residual, axis=1)
    diagnostic, source_span, robot_span = _head_to_toe_diagnostic(human, robot)
    return {
        "value": scale,
        "source_heading_yaw_rad": source_yaw,
        "source_heading_alignment_matrix": source_to_canonical.tolist(),
        "source_vectors_m": source_vectors.tolist(),
        "robot_vectors_m": robot_vectors.tolist(),
        "weights": weights.tolist(),
        "numerator_m2": numerator,
        "denominator_m2": denominator,
        "weighted_residual_rmse_m": float(
            np.sqrt(np.sum(weights * per_landmark**2) / np.sum(weights))
        ),
        "maximum_residual_m": float(np.max(per_landmark)),
        "per_landmark_residual_m": per_landmark.tolist(),
        "source_landmarks": [source for source, _, _ in COMMON_SCALE_LANDMARKS],
        "robot_landmarks": [semantic for _, semantic, _ in COMMON_SCALE_LANDMARKS],
        "head_to_toe_diagnostic_scale": diagnostic,
        "source_head_to_toe_span_m": source_span,
        "robot_head_to_toe_span_m": robot_span,
    }


def common_landmark_scale(
    human: CanonicalHuman, robot: CanonicalRobotModel
) -> tuple[float, float, float]:
    """Compatibility tuple for the registered LS scale plus span diagnostics.

    New code should consume :func:`shared_landmark_least_squares_calibration`
    so it cannot accidentally relabel the diagnostic head/toe spans as the
    estimator.
    """

    value = shared_landmark_least_squares_calibration(human, robot)
    return (
        float(value["value"]),
        float(value["source_head_to_toe_span_m"]),
        float(value["robot_head_to_toe_span_m"]),
    )


def build_evaluator_protocol(
    human: CanonicalHuman,
    robot: CanonicalRobotModel,
    *,
    source_path: str | Path,
) -> dict[str, Any]:
    """Build the corrected evaluator-v3 protocol for the selected Pilot."""

    calibration = shared_landmark_least_squares_calibration(human, robot)
    local_scale = float(calibration["value"])
    # Registered as a separate intervention even though the numerical value is
    # shared.  Scale-response experiments perturb these two fields separately.
    root_displacement_scale = local_scale
    neutral = robot.semantic_positions(robot.model.qpos0.copy())
    root_alignment = (
        neutral["root"]
        - root_displacement_scale * human.world_positions[0, 0]
    )
    return {
        "schema_version": EVALUATOR_SCHEMA_VERSION,
        "name": "stage1-pilot-evaluator-v3",
        "source_sha256": human.source_sha256,
        "source_path": str(source_path),
        "robot_xml": str(robot.xml_path),
        "robot_xml_sha256": robot.sha256,
        "robot_joint_order": list(robot.joint_names),
        "robot_joint_order_sha256": robot.joint_order_sha256,
        "heading": {
            "definition": HEADING_DEFINITION,
            "uses_bvh_root_quaternion": False,
            "robot_definition": "canonical_qpos_root_x_axis_yaw",
        },
        "timeline": {
            "definition": (
                "canonical source timestamps; method declarations rounded to nominal "
                "30 Hz are accepted within a frozen relative tolerance"
            ),
            "source_fps": float(human.fps),
            "nominal_fps": 30.0,
            "accepted_declared_fps_relative_tolerance": 2e-5,
            "temporal_metrics_use_source_fps": True,
        },
        "scale": {
            "definition": SCALE_DEFINITION,
            "method_independent": True,
            "source_frame": 0,
            "estimator_formula": (
                "argmin_s sum_k w_k ||s(h_k-h_root)-(r_k-r_root)||^2"
            ),
            "source_root_frame": (
                "frame0 Hips origin; geometry heading removed; canonical +X forward"
            ),
            "robot_root_frame": "neutral canonical Holosoma G1 pelvis; +X forward",
            "landmark_weights_policy": "registered_equal_weight_per_landmark",
            "source_landmarks": calibration["source_landmarks"],
            "robot_landmarks": calibration["robot_landmarks"],
            "landmark_weights": calibration["weights"],
            "excluded_landmarks": {
                "torso": (
                    "LAFAN Spine2 and canonical G1 torso-link origins are not homologous"
                ),
                "elbows_and_wrists": (
                    "source frame-0 and neutral-G1 distal-arm poses differ"
                ),
            },
            "least_squares_numerator_m2": calibration["numerator_m2"],
            "least_squares_denominator_m2": calibration["denominator_m2"],
            "weighted_residual_rmse_m": calibration["weighted_residual_rmse_m"],
            "maximum_residual_m": calibration["maximum_residual_m"],
            "per_landmark_residual_m": calibration["per_landmark_residual_m"],
            "source_heading_yaw_rad": calibration["source_heading_yaw_rad"],
            "source_heading_alignment_matrix": calibration[
                "source_heading_alignment_matrix"
            ],
            "common_local_body_scale": local_scale,
            "local_body_scale_definition": "shared_semantic_landmark_ls_value",
            "common_root_displacement_scale": root_displacement_scale,
            "root_displacement_scale_definition": (
                "separately frozen scalar; numerically initialized from shared LS"
            ),
            # Backward-compatible evaluator column.  It is explicitly the
            # local/body value, not an independent estimator.
            "common_static_scale": local_scale,
            "common_root_alignment_translation_m": root_alignment.tolist(),
            "root_alignment_definition": (
                "neutral_g1_root_minus_root_displacement_scale_times_source_frame0_root"
            ),
            "root_anchor_policy": "neutral_g1_pelvis_at_source_frame0",
            "root_local_and_anchor_frozen_separately": True,
            "diagnostics": {
                "head_to_toe_definition": HEAD_TO_TOE_DIAGNOSTIC,
                "head_to_toe_role": "diagnostic_only_not_common_policy",
                "source_head_to_toe_span_m": calibration[
                    "source_head_to_toe_span_m"
                ],
                "robot_head_to_toe_span_m": calibration[
                    "robot_head_to_toe_span_m"
                ],
                "head_to_toe_scale": calibration[
                    "head_to_toe_diagnostic_scale"
                ],
            },
        },
        "native_root_scales": {
            "sparse-neutral": {
                "value": root_displacement_scale,
                "policy": "controlled-v6 registered shared-landmark LS root gain",
            },
            "sparse-a": {
                "value": root_displacement_scale,
                "policy": "controlled-v6 registered shared-landmark LS root gain",
            },
            "sparse-b": {
                "value": root_displacement_scale,
                "policy": "controlled-v6 registered shared-landmark LS root gain",
            },
            "dense": {
                "value": root_displacement_scale,
                "policy": "controlled-v6 registered shared-landmark LS root gain",
            },
            "gmr": {
                "value": 0.9 * (1.75 / 1.8),
                "policy": "official GMR Hips scale 0.9 times hard-coded LAFAN height 1.75 / assumption 1.8",
            },
            "omniretarget": {
                "value": 1.27 / 1.7,
                "policy": "official Holosoma LAFAN default_scale_factor",
            },
            "protomotions_v2_3_mink": {
                "value_xyz": [0.75, 1.0, 0.8],
                "policy": "official ProtoMotions v2.3 world-axis position scaling",
            },
            "protomotions_v3": {
                "value_xyz": [0.9, 0.9, 0.85],
                "policy": "official ProtoMotions v3 root/lower-body axis scaling",
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
        raise ValueError("Stage 1 requires evaluator schema version 3")
    scale = value.get("scale", {})
    if scale.get("method_independent") is not True:
        raise ValueError("Evaluator common scale must be method-independent")
    if scale.get("definition") != SCALE_DEFINITION:
        raise ValueError("Evaluator does not use the registered shared-landmark LS scale")
    local = float(scale.get("common_local_body_scale", float("nan")))
    root = float(scale.get("common_root_displacement_scale", float("nan")))
    common = float(scale.get("common_static_scale", float("nan")))
    if not all(np.isfinite(item) and 0.3 < item < 1.5 for item in (local, root, common)):
        raise ValueError("Evaluator root/local common scales are missing or implausible")
    if not np.isclose(common, local, atol=1e-12, rtol=0.0):
        raise ValueError("common_static_scale must be the local/body compatibility alias")
    if scale.get("root_local_and_anchor_frozen_separately") is not True:
        raise ValueError("Root gain, local gain, and root anchor must be frozen separately")
    numerator = float(scale.get("least_squares_numerator_m2", float("nan")))
    denominator = float(scale.get("least_squares_denominator_m2", float("nan")))
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("Evaluator least-squares sufficient statistics are invalid")
    if not np.isclose(local, numerator / denominator, atol=1e-12, rtol=0.0):
        raise ValueError("Evaluator common scale does not equal its LS solution")
    expected_source = [source for source, _, _ in COMMON_SCALE_LANDMARKS]
    expected_robot = [semantic for _, semantic, _ in COMMON_SCALE_LANDMARKS]
    if scale.get("source_landmarks") != expected_source:
        raise ValueError("Evaluator source landmark order differs from the registration")
    if scale.get("robot_landmarks") != expected_robot:
        raise ValueError("Evaluator robot landmark order differs from the registration")
    diagnostics = scale.get("diagnostics", {})
    if diagnostics.get("head_to_toe_role") != "diagnostic_only_not_common_policy":
        raise ValueError("Head-to-toe span must remain diagnostic-only")
    alignment = np.asarray(
        scale.get("common_root_alignment_translation_m", []), dtype=np.float64
    )
    if alignment.shape != (3,) or not np.isfinite(alignment).all():
        raise ValueError("Evaluator common root alignment must be a finite xyz vector")
    joint_order = value.get("robot_joint_order", [])
    if len(joint_order) != 29 or len(set(joint_order)) != 29:
        raise ValueError("Evaluator canonical robot joint order is not 29-DoF")
    joint_hash = hashlib.sha256(
        json.dumps(tuple(joint_order), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if str(value.get("robot_joint_order_sha256", "")) != joint_hash:
        raise ValueError("Evaluator canonical robot joint-order hash is invalid")
    return value


def protocol_hash(path: str | Path) -> str:
    return sha256_file(path)
