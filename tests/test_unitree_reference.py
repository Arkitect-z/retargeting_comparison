from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from retargeting_comparison.constants import G1_JOINT_NAMES
from retargeting_comparison.unitree_reference import (
    AS_PUBLISHED_VIEW,
    CANONICAL_COORDINATE_VIEW,
    REFERENCE_TO_CANONICAL_YAW_RAD,
    UNITREE_REFERENCE_LABEL,
    UNITREE_REFERENCE_PILOT_AS_PUBLISHED_QPOS_SHA256,
    UNITREE_REFERENCE_PILOT_CANONICAL_QPOS_SHA256,
    UNITREE_REFERENCE_PILOT_SHA256,
    UPSTREAM_G1_COLUMNS,
    compare_urdf_kinematics,
    compare_urdf_to_mujoco_fk,
    fit_root_xy_similarity,
    load_unitree_g1_csv,
    qpos_content_sha256,
    transform_reference_coordinate_view,
)


def _write_csv(path: Path, rows: np.ndarray) -> str:
    np.savetxt(path, rows, delimiter=",", fmt="%.9f")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_column_contract_matches_canonical_g1_order() -> None:
    assert len(UPSTREAM_G1_COLUMNS) == 36
    assert UPSTREAM_G1_COLUMNS[7:] == G1_JOINT_NAMES


def test_csv_adapter_slices_and_reorders_xyzw_to_wxyz(tmp_path: Path) -> None:
    raw = np.zeros((3, 36), dtype=np.float64)
    raw[:, :3] = [[1.0, 2.0, 0.8], [1.1, 2.2, 0.81], [1.3, 2.4, 0.79]]
    raw[0, 3:7] = [0.0, 0.0, 0.0, 1.0]
    raw[1, 3:7] = [0.0, 0.0, 1.0, 1.0]
    raw[2, 3:7] = [0.5, 0.5, 0.5, 0.5]
    raw[:, 7:] = np.arange(3 * 29).reshape(3, 29)
    path = tmp_path / "motion.csv"
    digest = _write_csv(path, raw)

    motion = load_unitree_g1_csv(
        path,
        frame_start=1,
        frame_end=3,
        expected_sha256=digest,
        expected_source_frames=2,
    )

    assert motion.qpos.shape == (2, 36)
    assert np.allclose(motion.qpos[0, 3:7], [2**-0.5, 0.0, 0.0, 2**-0.5])
    assert np.allclose(motion.qpos[1, 3:7], [0.5, 0.5, 0.5, 0.5])
    assert np.array_equal(motion.qpos[:, 7:], raw[1:, 7:])
    assert np.array_equal(motion.source_frame_idx, [1, 2])
    assert motion.fps == 30.0
    assert motion.metadata["label"] == UNITREE_REFERENCE_LABEL
    assert motion.metadata["verified_official_ground_truth"] is False
    assert motion.metadata["timing_available"] is False
    assert motion.metadata["adapter_translation_scale_ground_resampling"] == "none"
    assert motion.metadata["coordinate_view"] == AS_PUBLISHED_VIEW
    assert motion.metadata["qpos_content_sha256"] == qpos_content_sha256(motion.qpos)


def test_fixed_coordinate_view_is_scale_free_and_invertible() -> None:
    raw = np.zeros((2, 36), dtype=np.float64)
    raw[:, :3] = [[2.0, 3.0, 0.8], [-4.0, 5.0, 0.7]]
    raw[:, 3] = 1.0
    raw[:, 7:] = np.arange(58, dtype=np.float64).reshape(2, 29)

    canonical = transform_reference_coordinate_view(raw)

    assert REFERENCE_TO_CANONICAL_YAW_RAD == -np.pi / 2.0
    assert np.allclose(canonical[:, :3], [[3.0, -2.0, 0.8], [5.0, 4.0, 0.7]])
    assert np.allclose(
        canonical[:, 3:7], [[2**-0.5, 0.0, 0.0, -(2**-0.5)]] * 2
    )
    assert np.linalg.norm(canonical[:, :2], axis=1) == pytest.approx(
        np.linalg.norm(raw[:, :2], axis=1)
    )
    canonical_step = np.linalg.norm(np.diff(canonical[:, :3], axis=0), axis=1)
    raw_step = np.linalg.norm(np.diff(raw[:, :3], axis=0), axis=1)
    assert canonical_step == pytest.approx(raw_step)
    assert np.array_equal(canonical[:, 7:], raw[:, 7:])
    restored = transform_reference_coordinate_view(canonical, inverse=True)
    assert restored == pytest.approx(raw)
    assert qpos_content_sha256(canonical) != qpos_content_sha256(raw)


def test_csv_adapter_exposes_both_coordinate_views(tmp_path: Path) -> None:
    raw = np.zeros((1, 36), dtype=np.float64)
    raw[0, :3] = [2.0, 3.0, 0.8]
    raw[0, 6] = 1.0  # published xyzw identity
    path = tmp_path / "motion.csv"
    digest = _write_csv(path, raw)

    published = load_unitree_g1_csv(
        path, expected_sha256=digest, coordinate_view=AS_PUBLISHED_VIEW
    )
    canonical = load_unitree_g1_csv(
        path, expected_sha256=digest, coordinate_view=CANONICAL_COORDINATE_VIEW
    )

    assert published.qpos[0, :3] == pytest.approx([2.0, 3.0, 0.8])
    assert canonical.qpos[0, :3] == pytest.approx([3.0, -2.0, 0.8])
    assert canonical.metadata["coordinate_rotation_yaw_rad"] == pytest.approx(
        -np.pi / 2
    )
    assert canonical.metadata["coordinate_rotation_fitted_from_results"] is False
    assert canonical.metadata["coordinate_transform_scale"] == 1.0
    assert canonical.metadata["coordinate_transform_translation_m"] == [
        0.0,
        0.0,
        0.0,
    ]
    assert published.metadata["qpos_content_sha256"] != canonical.metadata[
        "qpos_content_sha256"
    ]


def test_csv_adapter_rejects_hash_width_and_interval_errors(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    _write_csv(path, np.zeros((2, 35)))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_unitree_g1_csv(path, expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="36 columns"):
        load_unitree_g1_csv(path)

    valid = np.zeros((2, 36))
    valid[:, 6] = 1.0
    _write_csv(path, valid)
    with pytest.raises(ValueError, match="Invalid frame interval"):
        load_unitree_g1_csv(path, frame_start=1, frame_end=3)


def test_root_similarity_recovers_rotation_scale_and_anchor() -> None:
    source = np.asarray(
        [[2.0, -3.0, 0.8], [3.0, -3.0, 0.9], [3.0, -1.0, 1.0], [1.0, 0.0, 0.7]]
    )
    rotation = np.asarray([[0.0, 1.0], [-1.0, 0.0]])
    scale = 0.75
    source_delta = source[:, :2] - source[0, :2]
    reference = np.zeros_like(source)
    reference[:, :2] = np.asarray([0.2, -0.1]) + scale * source_delta @ rotation
    reference[:, 2] = 0.7 * source[:, 2] + 0.15

    audit = fit_root_xy_similarity(source, reference)

    assert audit.scale == pytest.approx(scale)
    assert np.asarray(audit.rotation_row_major) == pytest.approx(rotation)
    assert audit.rotation_det == pytest.approx(1.0)
    assert audit.rmse_m < 1e-12
    assert audit.z_slope == pytest.approx(0.7)
    assert audit.z_intercept_m == pytest.approx(0.15)


def test_urdf_kinematic_comparison_scope(tmp_path: Path) -> None:
    template = """<robot name="r">
      <link name="root"/><link name="child"/>
      <joint name="joint" type="revolute">
        <origin xyz="0 0 -0.2" rpy="0 0 0"/>
        <parent link="root"/><child link="child"/><axis xyz="0 1 0"/>
        <limit lower="-1" upper="2" effort="3" velocity="4"/>
      </joint>
    </robot>"""
    left = tmp_path / "left.urdf"
    right = tmp_path / "right.urdf"
    left.write_text(template)
    right.write_text(template.replace('effort="3"', 'effort="99"'))

    audit = compare_urdf_kinematics(left, right)

    assert audit["joint_order_match"] is True
    assert audit["kinematic_contract_equivalent"] is True
    assert "excludes visual" in audit["comparison_scope"]


def test_revision_pinned_pilot_file_when_local_data_is_available() -> None:
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "data"
        / "external"
        / "unitree_lafan1_reference"
        / "ce1572906efe6157840e8474d5a0d7aa87481e74"
        / "g1"
        / "dance1_subject1.csv"
    )
    if not path.exists():
        pytest.skip("Revision-pinned external reference CSV is not downloaded")
    published = load_unitree_g1_csv(
        path,
        frame_start=0,
        frame_end=600,
        expected_sha256=UNITREE_REFERENCE_PILOT_SHA256,
        expected_source_frames=600,
        coordinate_view=AS_PUBLISHED_VIEW,
    )
    motion = load_unitree_g1_csv(
        path,
        frame_start=0,
        frame_end=600,
        expected_sha256=UNITREE_REFERENCE_PILOT_SHA256,
        expected_source_frames=600,
        coordinate_view=CANONICAL_COORDINATE_VIEW,
    )
    assert motion.qpos.shape == (600, 36)
    assert motion.metadata["source_csv_rows"] == 3945
    assert motion.metadata["upstream_quaternion_norm_max_abs_error"] < 1e-5
    assert motion.metadata["coordinate_view"] == CANONICAL_COORDINATE_VIEW
    assert motion.metadata["byte_identical_human_source_verified"] is False
    assert motion.metadata["exact_timestamp_identity_claimed"] is False
    assert motion.metadata["absolute_root_anchor_preserved"] is True
    assert "same sequence basename and frame indices" in motion.metadata[
        "alignment_basis"
    ]
    assert "after required root quaternion" in motion.metadata[
        "as_published_semantics"
    ]
    assert (
        published.metadata["qpos_content_sha256"]
        == UNITREE_REFERENCE_PILOT_AS_PUBLISHED_QPOS_SHA256
    )
    assert (
        motion.metadata["qpos_content_sha256"]
        == UNITREE_REFERENCE_PILOT_CANONICAL_QPOS_SHA256
    )
    restored = transform_reference_coordinate_view(motion.qpos, inverse=True)
    assert restored == pytest.approx(published.qpos)


def test_reference_and_canonical_urdf_contract_when_assets_are_available() -> None:
    root = Path(__file__).resolve().parents[1]
    reference = (
        root
        / "data"
        / "external"
        / "unitree_lafan1_reference"
        / "ce1572906efe6157840e8474d5a0d7aa87481e74"
        / "robot_description"
        / "g1"
        / "g1_29dof_rev_1_0.urdf"
    )
    canonical = (
        root
        / "external"
        / "holosoma"
        / "src"
        / "holosoma"
        / "holosoma"
        / "data"
        / "robots"
        / "g1"
        / "g1_29dof.urdf"
    )
    if not reference.exists() or not canonical.exists():
        pytest.skip("External reference/canonical URDF assets are not downloaded")

    audit = compare_urdf_kinematics(reference, canonical)

    assert audit["joint_order_match"] is True
    assert audit["canonical_g1_order_match"] is True
    assert audit["kinematic_contract_equivalent"] is True


def test_cross_engine_asset_fk_when_audit_dependencies_are_available() -> None:
    pytest.importorskip("mujoco")
    root = Path(__file__).resolve().parents[1]
    reference = (
        root
        / "data"
        / "external"
        / "unitree_lafan1_reference"
        / "ce1572906efe6157840e8474d5a0d7aa87481e74"
        / "robot_description"
        / "g1"
        / "g1_29dof_rev_1_0.urdf"
    )
    canonical = (
        root
        / "external"
        / "holosoma"
        / "src"
        / "holosoma"
        / "holosoma"
        / "data"
        / "robots"
        / "g1"
        / "scenes"
        / "scene_g1_29dof_wbt_plane.xml"
    )
    if not reference.exists() or not canonical.exists():
        pytest.skip("External reference/canonical assets are not downloaded")

    audit = compare_urdf_to_mujoco_fk(
        reference,
        canonical,
        random_sample_count=10,
        seed=1947,
    )

    assert audit["kinematic_fk_equivalent"] is True
    assert audit["max_position_error_m"] < 1e-5
    assert audit["max_rotation_error_rad"] < 1e-5
    assert "excludes visual" in audit["comparison_scope"]
