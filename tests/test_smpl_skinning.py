from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from retargeting_comparison.smpl_skinning import (
    SMPL_JOINT_MAP,
    SmplSkinMotion,
    default_skin_cache,
    smpl_mesh_sequence,
)


FINAL_MODE = os.environ.get("RTCMP_FINAL_TESTS") == "1"


def _motion(frames: int = 2) -> SmplSkinMotion:
    return SmplSkinMotion(
        pose_aa_zup=np.zeros((frames, 72)),
        betas=np.zeros(10),
        body_scale=1.0,
        root_positions_zup=np.zeros((frames, 3)),
        transl_zup=np.zeros((frames, 3)),
        source_frame_idx=np.arange(frames),
        fps=30.0,
        metadata={
            "source_representation": "LAFAN1_BVH",
            "dataset_native_smpl": False,
        },
    )


def test_lafan_to_smpl_joint_map_is_explicit_and_complete() -> None:
    assert len(SMPL_JOINT_MAP) == 22
    assert [index for _, index in SMPL_JOINT_MAP] == list(range(22))
    assert SMPL_JOINT_MAP[0] == ("Hips", 0)
    assert SMPL_JOINT_MAP[-2:] == (("LeftHand", 20), ("RightHand", 21))


def test_fitted_skin_contract_cannot_claim_dataset_native_smpl(tmp_path: Path) -> None:
    motion = _motion()
    motion.validate()
    path = tmp_path / "skin.npz"
    motion.save(path)
    loaded = SmplSkinMotion.load(path)
    assert np.array_equal(loaded.pose_aa_zup, motion.pose_aa_zup)

    invalid = SmplSkinMotion(
        **{
            **motion.__dict__,
            "metadata": {
                "source_representation": "LAFAN1_BVH",
                "dataset_native_smpl": True,
            },
        }
    )
    with pytest.raises(ValueError, match="dataset-native"):
        invalid.validate()


def test_local_fitted_skin_and_mesh_are_finite_when_assets_exist() -> None:
    path = default_skin_cache(".")
    if not path.is_file():
        if FINAL_MODE:
            pytest.fail("final fitted LAFAN-to-SMPL skin is missing")
        pytest.skip("licensed/generated SMPL skin is not present")
    motion = SmplSkinMotion.load(path)
    assert len(motion.pose_aa_zup) == 600
    assert motion.metadata["dataset_native_smpl"] is False
    assert motion.metadata["root_aligned_mpjpe_m"] < 0.05
    vertices, faces, joints = smpl_mesh_sequence(
        motion, ".", frame_limit=2, batch_size=1, device="cpu"
    )
    assert vertices.shape == (2, 6890, 3)
    assert faces.shape == (13776, 3)
    assert joints.shape == (2, 24, 3)
    assert np.isfinite(vertices).all()
    assert np.allclose(joints[:, 0], motion.root_positions_zup[:2], atol=1e-6)
    evidence = json.loads(
        path.with_name("lafan_bvh_fitted_smpl.evidence.json").read_text()
    )
    assert evidence["dataset_native_smpl"] is False
    assert evidence["output_sha256"]
