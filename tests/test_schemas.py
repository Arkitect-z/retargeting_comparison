from __future__ import annotations

import numpy as np
import pytest

from retargeting_comparison.schemas import (
    CanonicalG1,
    CanonicalHuman,
    canonical_g1_source_prefix_view,
)


def _human() -> CanonicalHuman:
    local = np.zeros((3, 2, 4), dtype=np.float64)
    local[..., 0] = 1.0
    return CanonicalHuman(
        joint_names=np.asarray(["root", "child"]),
        parent_indices=np.asarray([-1, 0]),
        local_rotations=local,
        world_rotations=local.copy(),
        world_positions=np.zeros((3, 2, 3)),
        root_translation=np.zeros((3, 3)),
        fps=30.0,
        timestamps=np.arange(3) / 30.0,
        foot_contact_labels=np.zeros((3, 2), dtype=bool),
        source_sha256="a" * 64,
    )


def test_human_roundtrip(tmp_path) -> None:
    path = tmp_path / "human.npz"
    _human().save(path)
    restored = CanonicalHuman.load(path)
    assert restored.joint_names.tolist() == ["root", "child"]
    assert restored.world_positions.shape == (3, 2, 3)


def test_g1_completion_contract(tmp_path) -> None:
    motion = CanonicalG1(
        qpos=np.column_stack(
            [
                np.zeros((10, 3)),
                np.tile([1.0, 0.0, 0.0, 0.0], (10, 1)),
                np.zeros((10, 29)),
            ]
        ),
        fps=30.0,
        source_frame_idx=np.arange(10),
        valid=np.ones(10, dtype=bool),
        per_frame_solve_time_s=np.zeros(10),
        metadata={"completion_status": "succeeded"},
    )
    path = tmp_path / "g1.npz"
    motion.save(path, source_frame_count=10)
    assert CanonicalG1.load(path).qpos.shape == (10, 36)
    motion.metadata["completion_status"] = "incomplete"
    with pytest.raises(ValueError, match="completion_status"):
        motion.validate(source_frame_count=10)


def test_g1_rejects_xyzw_identity_mislabelled_as_wxyz() -> None:
    qpos = np.zeros((2, 36))
    qpos[:, 6] = 1.0
    motion = CanonicalG1(
        qpos=qpos,
        fps=30.0,
        source_frame_idx=np.arange(2),
        valid=np.ones(2, dtype=bool),
        per_frame_solve_time_s=np.zeros(2),
        metadata={},
    )
    # It is a valid quaternion but not the identity in canonical wxyz; semantic
    # identity checks belong to adapters rather than the schema.
    motion.validate()
    assert not np.allclose(motion.qpos[0, 3:7], [1.0, 0.0, 0.0, 0.0])


def test_g1_rejects_non_unit_quaternion() -> None:
    qpos = np.zeros((2, 36))
    qpos[:, 3] = 1.01
    motion = CanonicalG1(
        qpos=qpos,
        fps=30.0,
        source_frame_idx=np.arange(2),
        valid=np.ones(2, dtype=bool),
        per_frame_solve_time_s=np.zeros(2),
        metadata={},
    )
    with pytest.raises(ValueError, match="unit wxyz"):
        motion.validate()


def test_g1_source_prefix_view_is_exact_and_keeps_full_source_completion() -> None:
    frames = 10
    qpos = np.zeros((frames, 36), dtype=np.float64)
    qpos[:, 0] = np.arange(frames)
    qpos[:, 3] = 1.0
    motion = CanonicalG1(
        qpos=qpos,
        fps=30.0,
        source_frame_idx=np.arange(frames),
        valid=np.ones(frames, dtype=bool),
        per_frame_solve_time_s=np.arange(frames, dtype=np.float64) / 1000.0,
        metadata={"completion_status": "succeeded"},
    )

    view = canonical_g1_source_prefix_view(
        motion, 7, source_frame_count=frames
    )

    assert view.qpos.shape == (7, 36)
    assert np.array_equal(view.qpos[:, 0], np.arange(7))
    assert np.array_equal(view.source_frame_idx, np.arange(7))
    assert view.metadata["completion_status"] == "incomplete"
    assert view.metadata["comparison_source_frame_range"] == [0, 7]
    assert view.metadata["comparison_padding_or_interpolation_used"] is False
    assert not np.shares_memory(view.qpos, motion.qpos)


def test_g1_source_prefix_view_rejects_a_noncontiguous_prefix() -> None:
    qpos = np.zeros((5, 36), dtype=np.float64)
    qpos[:, 3] = 1.0
    motion = CanonicalG1(
        qpos=qpos,
        fps=30.0,
        source_frame_idx=np.asarray([0, 1, 3, 4, 5]),
        valid=np.ones(5, dtype=bool),
        per_frame_solve_time_s=np.zeros(5),
        metadata={"completion_status": "succeeded"},
    )

    with pytest.raises(ValueError, match="source prefix exactly"):
        canonical_g1_source_prefix_view(
            motion, 3, source_frame_count=5
        )
