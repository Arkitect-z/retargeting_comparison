from __future__ import annotations

import numpy as np
import pytest

from retargeting_comparison.evaluator import HUMAN_SEMANTIC_JOINTS, evaluate_motion
from retargeting_comparison.robot_model import CanonicalRobotModel, default_robot_scene
from retargeting_comparison.schemas import CanonicalG1, CanonicalHuman


pytestmark = pytest.mark.skipif(
    not default_robot_scene().is_file(), reason="frozen Holosoma checkout is not present"
)


def _matched_pair(robot: CanonicalRobotModel) -> tuple[CanonicalHuman, CanonicalG1]:
    qpos = robot.model.qpos0.copy()
    points = robot.semantic_positions(qpos)
    joint_names = np.asarray(list(HUMAN_SEMANTIC_JOINTS.values()))
    semantic_for_joint = {joint: semantic for semantic, joint in HUMAN_SEMANTIC_JOINTS.items()}
    one_frame = np.stack([points[semantic_for_joint[joint]] for joint in joint_names])
    frames = 4
    rotations = np.zeros((frames, len(joint_names), 4), dtype=np.float64)
    rotations[..., 0] = 1.0
    human = CanonicalHuman(
        joint_names=joint_names,
        parent_indices=np.asarray([-1] + [0] * (len(joint_names) - 1)),
        local_rotations=rotations,
        world_rotations=rotations.copy(),
        world_positions=np.tile(one_frame, (frames, 1, 1)),
        root_translation=np.tile(points["root"], (frames, 1)),
        fps=30.0,
        timestamps=np.arange(frames) / 30.0,
        foot_contact_labels=np.zeros((frames, 2), dtype=bool),
        source_sha256="b" * 64,
    )
    motion = CanonicalG1(
        qpos=np.tile(qpos, (frames, 1)),
        fps=30.0,
        source_frame_idx=np.arange(frames),
        valid=np.ones(frames, dtype=bool),
        per_frame_solve_time_s=np.zeros(frames),
        metadata={"method": "synthetic", "completion_status": "succeeded"},
    )
    return human, motion


def test_canonical_robot_contract() -> None:
    robot = CanonicalRobotModel(default_robot_scene())
    assert robot.model.nq == 36
    assert robot.model.njnt == 30
    assert robot.sha256


def test_matched_motion_has_zero_fidelity_and_temporal_error() -> None:
    robot = CanonicalRobotModel(default_robot_scene())
    human, motion = _matched_pair(robot)
    table, summary = evaluate_motion(human, motion, robot)
    assert np.allclose(table.rf_kpe_all_m, 0.0, atol=1e-12)
    assert np.allclose(table.rf_kpe_targeted_m, 0.0, atol=1e-12)
    assert np.allclose(table.rf_kpe_untracked_m, 0.0, atol=1e-12)
    assert np.allclose(table.bone_direction_error_rad, 0.0, atol=1e-7)
    assert np.allclose(table.bend_plane_error_rad, 0.0, atol=1e-7)
    assert np.allclose(table.joint_velocity_rms_rad_s, 0.0)
    assert np.allclose(table.pose_jump_rms_m, 0.0)
    assert summary["completion_ratio"] == 1.0


def test_synthetic_joint_limit_and_invalid_frame_are_artifacts() -> None:
    robot = CanonicalRobotModel(default_robot_scene())
    human, motion = _matched_pair(robot)
    limited_joint = next(
        index for index in range(1, robot.model.njnt) if robot.model.jnt_limited[index]
    )
    address = robot.model.jnt_qposadr[limited_joint]
    motion.qpos[1, address] = robot.model.jnt_range[limited_joint, 1] + 0.1
    motion.valid[2] = False
    table, _ = evaluate_motion(human, motion, robot)
    assert table.loc[1, "joint_limit_violation_rad"] > 0.09
    assert bool(table.loc[1, "artifact"])
    assert bool(table.loc[2, "artifact"])
