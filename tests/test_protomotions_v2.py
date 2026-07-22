from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from retargeting_comparison.constants import G1_JOINT_NAMES
from retargeting_comparison.protomotions_v2 import (
    ProtoMotionsV2Retargeter,
    audit_robot_compatibility,
    audit_smplx_pose_contract,
    run_scale_variant,
    smplx_165d_to_mink_52x3,
    verify_frozen_installation,
)
from retargeting_comparison.schemas import CanonicalHuman


WORKTREE = Path("external/ProtoMotions-v2.3")
pytestmark = pytest.mark.skipif(
    not WORKTREE.is_dir(), reason="frozen ProtoMotions v2.3 worktree is not present"
)


def test_frozen_installation_and_canonical_joint_order() -> None:
    audit = verify_frozen_installation(".")
    assert audit["upstream_commit"] == "4a905b998101333a2fb91f2de8e2cab4bd0db68e"
    assert audit["dimensions"] == {"nq": 36, "nv": 35, "nu": 29}
    assert tuple(audit["joint_order"]) == G1_JOINT_NAMES
    assert audit["dependency_mismatches"] == {}
    assert audit["smplx_pose_contract"]["input_shape"] == ["T", 165]


def test_upstream_165d_smplx_pose_contract_is_exact_and_source_bound() -> None:
    poses = np.arange(2 * 165, dtype=np.float64).reshape(2, 165)
    converted = smplx_165d_to_mink_52x3(poses)
    expected = np.concatenate((poses[:, :66], poses[:, 75:]), axis=-1)
    assert converted.shape == (2, 52, 3)
    assert converted.dtype == np.float64
    assert np.array_equal(converted.reshape(2, 156), expected)
    assert not np.isin(poses[:, 66:75], converted).any()
    with pytest.raises(ValueError, match=r"\[T,165\]"):
        smplx_165d_to_mink_52x3(np.zeros((2, 164)))

    evidence = audit_smplx_pose_contract(".")
    assert evidence["contract"] == (
        "165 -> concatenate [:66] and [75:] -> 156 -> [T,52,3]"
    )
    assert evidence["upstream_sha256"] == (
        "356087a34d5e68088db0bfa0c15561f506b158efcf013083b19a7bdc011e5f31"
    )
    assert evidence["upstream_commit"] == (
        "4a905b998101333a2fb91f2de8e2cab4bd0db68e"
    )
    assert evidence["line_evidence"]["amass_pose_field"]["line"] == 611
    assert evidence["line_evidence"]["slice_and_reshape"]["line_start"] == 637
    assert evidence["line_evidence"]["slice_and_reshape"]["line_end"] == 644
    assert len(evidence["line_evidence_sha256"]) == 64


def test_native_and_canonical_robot_mismatch_is_measured() -> None:
    audit = audit_robot_compatibility(".", random_samples=2)
    assert audit["joint_order_matches"] is True
    assert audit["fk_sample_count"] == 3
    assert len(audit["fk_common_bodies"]) == 13
    assert np.isfinite(audit["fk_root_aligned_mean_error_m"])
    assert np.isfinite(audit["fk_root_aligned_max_error_m"])
    assert audit["joint_limit_exact_match_count"] + audit["joint_limit_mismatch_count"] == 29


def test_two_frame_run_has_real_canonical_qpos_and_is_incomplete() -> None:
    human = CanonicalHuman.load(
        "source/canonical_human/dance1_subject1_f000000_000600.npz"
    )
    retargeter = ProtoMotionsV2Retargeter(".")
    indices = {name: index for index, name in enumerate(human.joint_names.astype(str))}
    retargeter._initialize_from_source(human, indices)
    assert np.allclose(
        retargeter.configuration.data.qpos[3:7],
        human.world_rotations[0, indices["Hips"]],
    )
    result = retargeter.run(human, max_frames=2)
    result.validate(source_frame_count=len(human.timestamps))
    assert result.qpos.shape == (2, 36)
    assert result.qpos.dtype == np.float64
    assert np.isfinite(result.qpos).all()
    assert np.allclose(np.linalg.norm(result.qpos[:, 3:7], axis=1), 1.0)
    assert np.array_equal(result.source_frame_idx, np.arange(2))
    assert result.valid.all()
    assert result.metadata["completion_status"] == "incomplete"
    assert result.metadata["is_phc_result"] is False
    assert "Mink task/solver" in result.metadata["experiment_identity"]
    assert result.metadata["upstream_smplx_pose_contract"]["input_shape"] == [
        "T",
        165,
    ]
    assert result.metadata["quaternion_order"] == "wxyz"
    assert result.metadata["position_scale_xyz"] == [0.75, 1.0, 0.8]
    assert result.metadata["warmup_frames"] == 100
    lower = retargeter.model.jnt_range[1:, 0]
    upper = retargeter.model.jnt_range[1:, 1]
    assert np.all(result.qpos[:, 7:] >= lower)
    assert np.all(result.qpos[:, 7:] <= upper)


def test_scale_variant_perturbs_root_and_local_before_official_axis_scale(
    tmp_path: Path,
) -> None:
    human = CanonicalHuman.load(
        "source/canonical_human/dance1_subject1_f000000_000600.npz"
    )
    indices = {name: index for index, name in enumerate(human.joint_names.astype(str))}
    retargeter = ProtoMotionsV2Retargeter(
        ".", root_scale_multiplier=0.95, local_scale_multiplier=1.05
    )
    frame = 10
    root = human.world_positions[frame, indices["Hips"]]
    anchor = human.world_positions[0, indices["Hips"]]
    hand = human.world_positions[frame, indices["LeftHand"]]
    expected = (
        anchor + (root - anchor) * 0.95 + (hand - root) * 1.05
    ) * np.asarray([0.75, 1.0, 0.8])
    actual = retargeter._target_position(
        human, frame, indices["LeftHand"], indices["Hips"]
    )
    assert np.allclose(actual, expected)
    native = ProtoMotionsV2Retargeter(".")
    assert np.array_equal(
        native._target_position(
            human, frame, indices["LeftHand"], indices["Hips"]
        ),
        hand * np.asarray([0.75, 1.0, 0.8]),
    )

    motion = run_scale_variant(
        repo_root=".",
        canonical_source="source/canonical_human/dance1_subject1_f000000_000600.npz",
        output_dir=tmp_path,
        root_scale_multiplier=0.95,
        local_scale_multiplier=1.05,
    )
    assert motion.qpos.shape == (600, 36)
    assert motion.metadata["root_scale_multiplier"] == 0.95
    assert motion.metadata["local_scale_multiplier"] == 1.05
    assert motion.metadata["position_scale_xyz"] == [0.75, 1.0, 0.8]
    assert (tmp_path / "robot_compatibility_audit.json").is_file()
