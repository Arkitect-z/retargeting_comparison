from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import mujoco

from retargeting_comparison.controlled_mink import (
    ControlledMinkRetargeter,
    common_baseline_config,
    seed_qpos,
)
from retargeting_comparison.io_utils import load_yaml
from retargeting_comparison.constants import G1_JOINT_NAMES
from retargeting_comparison.robot_model import (
    CanonicalRobotModel,
    SEMANTIC_FRAMES,
    default_robot_scene,
)
from retargeting_comparison.schemas import CanonicalHuman


pytestmark = pytest.mark.skipif(
    not default_robot_scene().is_file(),
    reason="frozen Holosoma checkout is not present",
)


def test_sparse_dense_only_vary_declared_target_set() -> None:
    config = load_yaml("configs/controlled_mink.yaml")
    common = common_baseline_config(config)
    assert common == config["common"]
    assert "target_sets" not in common
    sparse = config["target_sets"]["sparse"]
    dense = config["target_sets"]["dense"]
    assert sparse == [target for target in dense if target["semantic"] in {
        "root", "left_wrist", "right_wrist", "left_ankle", "right_ankle"
    }]
    assert common["position_scale_root_torso_legs"] == common["position_scale_arms"]
    assert common["root_displacement_scale"] == common["local_body_scale"]
    assert common["position_scale_arms"] == common["local_body_scale"]
    assert common["temporal_smoothness_cost"] > 0.0
    assert common["temporal_smoothness_cost"] < min(
        target["position_cost"] for target in dense
    )


def test_sparse_seeds_are_deterministic_and_distinct() -> None:
    retargeter = ControlledMinkRetargeter(".", "sparse")
    neutral = seed_qpos(retargeter.model, retargeter.config, "neutral")
    a_first = seed_qpos(retargeter.model, retargeter.config, "A")
    a_second = seed_qpos(retargeter.model, retargeter.config, "A")
    b = seed_qpos(retargeter.model, retargeter.config, "B")
    assert np.array_equal(a_first, a_second)
    assert not np.array_equal(neutral, a_first)
    assert not np.array_equal(a_first, b)
    assert np.allclose(a_first[:7], neutral[:7])


def test_controlled_baseline_has_distinct_posture_and_temporal_tasks() -> None:
    retargeter = ControlledMinkRetargeter(".", "sparse")
    assert retargeter.posture is not retargeter.temporal
    assert retargeter.tasks[-2:] == [retargeter.posture, retargeter.temporal]
    source = CanonicalHuman.load(
        "source/canonical_human/dance1_subject1_f000000_000600.npz"
    )
    target = retargeter._target_position(source, 0, 0, "root_torso_legs")
    neutral_root = retargeter.model.qpos0[:3]
    assert np.allclose(target, neutral_root, atol=1e-12, rtol=0.0)


def test_controlled_solver_uses_canonical_holosoma_joint_order_and_fk_frames() -> None:
    retargeter = ControlledMinkRetargeter(".", "dense")
    config = load_yaml("configs/controlled_mink.yaml")
    assert Path(config["robot_xml"]) == default_robot_scene()
    assert retargeter.robot.joint_names == G1_JOINT_NAMES
    assert (
        retargeter.robot.joint_order_sha256
        == retargeter.evaluator["robot_joint_order_sha256"]
    )
    neutral = retargeter.robot.semantic_positions(retargeter.model.qpos0.copy())
    for spec in config["target_sets"]["dense"]:
        assert (spec["robot_frame_type"], spec["robot_frame"]) == SEMANTIC_FRAMES[
            spec["semantic"]
        ]
        assert np.isfinite(neutral[spec["semantic"]]).all()
        object_type = (
            mujoco.mjtObj.mjOBJ_BODY
            if spec["robot_frame_type"] == "body"
            else mujoco.mjtObj.mjOBJ_SITE
        )
        frame_id = mujoco.mj_name2id(
            retargeter.model, object_type, spec["robot_frame"]
        )
        raw_position = (
            retargeter.robot.data.xpos[frame_id]
            if spec["robot_frame_type"] == "body"
            else retargeter.robot.data.site_xpos[frame_id]
        )
        assert np.allclose(raw_position, neutral[spec["semantic"]])

    evaluator_robot = CanonicalRobotModel(default_robot_scene())
    random_valid_qpos = seed_qpos(retargeter.model, retargeter.config, "A")
    controlled_fk = retargeter.robot.semantic_positions(random_valid_qpos)
    evaluator_fk = evaluator_robot.semantic_positions(random_valid_qpos)
    assert controlled_fk.keys() == evaluator_fk.keys()
    for semantic in controlled_fk:
        assert np.allclose(controlled_fk[semantic], evaluator_fk[semantic])


def test_controlled_timing_metadata_has_explicit_native_core_boundary() -> None:
    retargeter = ControlledMinkRetargeter(".", "sparse")
    source = CanonicalHuman.load(
        "source/canonical_human/dance1_subject1_f000000_000600.npz"
    )
    motion = retargeter.run(source, max_frames=1)
    assert motion.metadata["native_core_total_s"] >= motion.per_frame_solve_time_s.sum()
    assert "target construction start" in motion.metadata["native_core_timing_boundary"]
    assert motion.metadata["canonical_joint_order"] == list(G1_JOINT_NAMES)
