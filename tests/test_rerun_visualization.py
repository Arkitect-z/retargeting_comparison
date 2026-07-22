from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from retargeting_comparison.rerun_visualization import (
    METHOD_STYLES,
    UrdfSemanticKinematics,
    load_stage1_visualization,
    root_frame_points,
)
from retargeting_comparison.robot_model import CanonicalRobotModel, default_robot_scene


URDF = Path(
    "external/holosoma/src/holosoma/holosoma/data/robots/g1/g1_29dof.urdf"
)


def test_all_operating_points_and_sparse_diagnostics_are_declared() -> None:
    assert [style.key for style in METHOD_STYLES] == [
        "sparse-neutral",
        "dense",
        "gmr",
        "omniretarget",
        "sparse-a",
        "sparse-b",
    ]
    assert [style.key for style in METHOD_STYLES if style.operating_point] == [
        "sparse-neutral",
        "dense",
        "gmr",
        "omniretarget",
    ]


def test_root_frame_points_remove_translation_and_yaw() -> None:
    roots = np.asarray([[1.0, 2.0, 0.5], [4.0, -2.0, 0.7]])
    yaw = np.asarray([0.0, np.pi / 2.0])
    local = np.asarray([[1.0, 0.0, 0.2], [0.0, 1.0, -0.1]])
    points = np.empty((2, 2, 3), dtype=np.float64)
    points[0] = roots[0] + local
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    points[1] = roots[1] + local @ rotation.T
    restored = root_frame_points(points, roots, yaw)
    assert np.allclose(restored[0], local)
    assert np.allclose(restored[1], local)


@pytest.mark.skipif(
    not URDF.is_file() or not default_robot_scene().is_file(),
    reason="frozen Holosoma checkout is not present",
)
def test_urdf_visualization_fk_matches_canonical_mujoco_fk() -> None:
    urdf = UrdfSemanticKinematics(URDF)
    robot = CanonicalRobotModel(default_robot_scene())
    rng = np.random.default_rng(20260722)
    for _ in range(4):
        qpos = robot.model.qpos0.copy()
        qpos[:3] = rng.uniform([-0.5, -0.5, 0.7], [0.5, 0.5, 1.0])
        for joint_id in range(1, robot.model.njnt):
            if not robot.model.jnt_limited[joint_id]:
                continue
            address = robot.model.jnt_qposadr[joint_id]
            lower, upper = robot.model.jnt_range[joint_id]
            qpos[address] = rng.uniform(max(lower, -0.6), min(upper, 0.6))
        expected = robot.semantic_positions(qpos)
        actual = urdf.semantic_positions(qpos)
        assert actual.keys() == expected.keys()
        assert max(np.linalg.norm(actual[key] - expected[key]) for key in expected) < 1e-6


def test_local_stage1_visualization_contract_when_artifacts_exist() -> None:
    sequence_id = "dance1_subject1_f000000_000600"
    paths = [Path("source/canonical_human") / f"{sequence_id}.npz"]
    paths.extend(
        Path("runs") / sequence_id / style.key / "canonical_g1.npz"
        for style in METHOD_STYLES
    )
    if not all(path.is_file() for path in paths):
        pytest.skip("licensed/generated local Stage 1 artifacts are not present")
    data = load_stage1_visualization(".")
    assert data.frame_count == 600
    assert set(data.methods) == {style.key for style in METHOD_STYLES}
    assert all(method.positions.shape == (600, 17, 3) for method in data.methods.values())
    assert all(
        np.isfinite(value).all()
        for method in data.methods.values()
        for value in method.metrics.values()
    )
