from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

import retargeting_comparison.phc_visualization as phc


FINAL_MODE = os.environ.get("RTCMP_FINAL_TESTS") == "1"


def test_phc_public_g1_contract_is_not_canonical_g1_29() -> None:
    assert phc.PHC_ACTUATED_DOFS == 37
    assert phc.PHC_BODY_DOFS == 23
    assert phc.PHC_HAND_DOFS == 14
    assert phc.PHC_BODY_DOFS + phc.PHC_HAND_DOFS == phc.PHC_ACTUATED_DOFS
    assert phc.PHC_VISUAL_MESHES == 43


def test_phc_motion_rejects_false_g1_29_equivalence() -> None:
    qpos = np.zeros((2, 44))
    qpos[:, 3] = 1.0
    motion = phc.PhcPreparedMotion(
        qpos=qpos,
        smpl_joints=np.zeros((2, 24, 3)),
        shape_betas=np.zeros(10),
        body_scale=1.0,
        source_frame_idx=np.arange(2),
        fps=30.0,
        metadata={"canonical_g1_29_compatible": True},
    )
    with pytest.raises(ValueError, match="must not claim"):
        motion.validate()


def test_root_local_transforms_remove_root_pose() -> None:
    root_rotation = np.asarray(
        [[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]]
    )
    roots = np.asarray([[1.0, 2.0, 0.5]])
    local = np.asarray([[[0.2, -0.1, 0.4]]])
    world = np.einsum("tvi,tji->tvj", local, root_rotation) + roots[:, None]
    restored = phc._root_local_vertices(world, roots, root_rotation)
    assert np.allclose(restored, local)


def test_local_phc_preparation_and_rerun_manifests_when_present() -> None:
    preparation = Path("manifests/phc_visualization_preparation.json")
    recording = Path("manifests/phc_rerun_visualization.json")
    if not preparation.is_file() or not recording.is_file():
        if FINAL_MODE:
            pytest.fail("final PHC visualization manifests are missing")
        pytest.skip("generated PHC visualization evidence is not present")
    prepared_value = json.loads(preparation.read_text())
    recording_value = json.loads(recording.read_text())
    assert prepared_value["dataset_native_smpl"] is False
    assert prepared_value["phc"]["actuated_dofs"] == 37
    assert prepared_value["phc"]["canonical_g1_29_compatible"] is False
    assert recording_value["frames"] == 600
    assert recording_value["phc"]["visual_meshes"] == 43
    assert recording_value["phc"]["canonical_g1_29_compatible"] is False

    sequence_id = prepared_value["sequence_id"]
    motion = phc.PhcPreparedMotion.load(
        phc._prepared_motion_path(Path.cwd(), sequence_id)
    )
    visuals = phc.PhcRobotVisualCache.load(
        phc._visual_cache_path(Path.cwd(), sequence_id)
    )
    assert motion.qpos.shape == (600, 44)
    assert visuals.translations.shape == (600, 43, 3)
    assert 0.0 < motion.body_scale < 1.0
