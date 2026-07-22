from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from retargeting_comparison.interaction import (
    _interaction_file_record,
    _interaction_output_paths,
    _verify_interaction_output_ledger,
    _verify_provenance_file_records,
    _write_interaction_output_ledger,
    intended_contact_diagnostics,
    mesh_surface_diagnostics,
    resolve_intended_contact_mapping,
)
from retargeting_comparison.io_utils import sha256_file


def test_surface_metrics_use_geometry_distance_not_object_origin() -> None:
    mujoco = pytest.importorskip("mujoco")

    xml = """
    <mujoco><worldbody>
      <body name='robot' pos='1.2 0 0'><freejoint/><geom name='robot_surface' type='sphere' size='0.1'/></body>
      <body name='largebox_link'><geom name='largebox' type='box' size='1 1 1'/></body>
    </worldbody></mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    qpos = np.tile(model.qpos0, (2, 1))
    rows, summary = mesh_surface_diagnostics(
        model,
        qpos,
        "largebox",
        ["robot"],
        [{"left": False}, {"left": False}],
        0.001,
    )
    # Origin distance is 1.2 m, but the actual box/sphere surfaces are 0.1 m apart.
    assert rows[0]["minimum_robot_object_surface_distance_m"] == pytest.approx(0.1)
    assert summary["proximity_10cm_frame_rate"] == pytest.approx(1.0)


def test_deep_penetration_is_never_counted_as_successful_contact() -> None:
    mujoco = pytest.importorskip("mujoco")
    xml = """
    <mujoco><worldbody>
      <body name='robot' pos='0.5 0 0'><freejoint/><geom name='robot_surface' type='sphere' size='0.75'/></body>
      <body name='largebox_link'><geom name='largebox' type='box' size='1 1 1'/></body>
    </worldbody></mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    rows, summary = mesh_surface_diagnostics(
        model,
        np.tile(model.qpos0, (1, 1)),
        "largebox",
        ["robot"],
        [{"left": False}],
        0.001,
    )
    assert rows[0]["minimum_robot_object_surface_distance_m"] < -0.0011
    assert rows[0]["strict_contact_2cm"] is False
    assert summary["strict_contact_2cm_frame_rate"] == 0.0
    assert summary["penetration_frame_rate"] == 1.0


def test_intended_contact_is_source_conditioned_and_pair_specific(tmp_path) -> None:
    mujoco = pytest.importorskip("mujoco")
    trimesh = pytest.importorskip("trimesh")
    mesh_path = tmp_path / "cube.obj"
    trimesh.creation.box(extents=(2.0, 2.0, 2.0)).export(mesh_path)
    xml = """
    <mujoco><worldbody>
      <body name='unrelated_torso'><geom name='torso_geom' type='sphere' size='0.5'/></body>
      <body name='left_rubber_hand_link' pos='1.03 0 0'>
        <geom name='left_hand_geom' type='sphere' size='0.02'/>
      </body>
      <body name='right_rubber_hand_link' pos='-1.03 0 0'>
        <geom name='right_hand_geom' type='sphere' size='0.02'/>
      </body>
      <body name='largebox_link'>
        <geom name='largebox' type='box' size='1 1 1'/>
      </body>
    </worldbody></mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    demo_joints = ["L_Wrist", "R_Wrist"]
    mapping = {
        "L_Wrist": "left_rubber_hand_link",
        "R_Wrist": "right_rubber_hand_link",
    }
    human = np.asarray(
        [
            [[1.03, 0.0, 0.0], [-1.03, 0.0, 0.0]],
            [[1.03, 0.0, 0.0], [-1.03, 0.0, 0.0]],
        ]
    )
    poses = np.tile(np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), (2, 1))
    paired_points = np.asarray(
        [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], [1.0, -1.0, 1.0]]
    )
    rows, summary, contract, arrays = intended_contact_diagnostics(
        model,
        np.tile(model.qpos0, (2, 1)),
        "box",
        "largebox",
        demo_joints,
        mapping,
        human,
        poses,
        paired_points,
        paired_points,
        mesh_path,
    )
    assert len(rows) == 4
    assert summary["intended_contact_pair_asserted"] is True
    assert summary["intended_contact_semantic_count"] == 2
    # A joint-centre proxy can miss the strict 2 cm surface band while still
    # defining the frozen 5 cm source contact envelope.
    assert summary["intended_source_contact_observations_2cm"] == 0
    assert summary["intended_source_contact_observations_5cm"] == 4
    assert (
        summary["intended_contact_preservation_2cm_given_source_contact_5cm"]
        == 1.0
    )
    assert all(
        row["mapped_robot_object_signed_surface_distance_m"]
        == pytest.approx(0.01)
        for row in rows
    )
    # The unrelated torso is embedded in the box.  It must not influence the
    # frozen hand/body pair metric.
    assert all(row["closest_mapped_robot_geom_name"].endswith("hand_geom") for row in rows)
    assert contract["mappings"][0]["robot_body_name"] == "left_rubber_hand_link"
    assert arrays["source_joint_positions"].shape == (2, 2, 3)


def test_intended_contact_mapping_fails_closed_on_missing_source_joint() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco><worldbody>
          <body name='left_rubber_hand_link'><geom name='left_hand' size='0.01'/></body>
          <body name='right_rubber_hand_link'><geom name='right_hand' size='0.01'/></body>
          <body name='largebox_link'><geom name='largebox' type='box' size='1 1 1'/></body>
        </worldbody></mujoco>
        """
    )
    with pytest.raises(RuntimeError, match="R_Wrist.*absent"):
        resolve_intended_contact_mapping(
            model,
            "box",
            ["L_Wrist"],
            {"L_Wrist": "left_rubber_hand_link"},
            "largebox",
        )


def test_official_holosoma_scenes_resolve_every_frozen_intended_pair(
    monkeypatch,
) -> None:
    mujoco = pytest.importorskip("mujoco")
    package = (
        Path(__file__).resolve().parents[1]
        / "external/holosoma/src/holosoma_retargeting/holosoma_retargeting"
    )
    if not package.is_dir():
        pytest.skip("Official Holosoma checkout is not available")
    monkeypatch.chdir(package)
    cases = (
        (
            "box",
            "models/g1/g1_29dof_w_largebox.xml",
            "largebox",
            ["L_Wrist", "R_Wrist"],
            {
                "L_Wrist": "left_rubber_hand_link",
                "R_Wrist": "right_rubber_hand_link",
            },
            2,
        ),
        (
            "climb",
            "demo_data/climb/mocap_climb_seq_0/g1_29dof_spherehand_w_multi_boxes.xml",
            "multi_boxes",
            [
                "LeftHandMiddle3",
                "RightHandMiddle3",
                "LeftFoot",
                "RightFoot",
                "LeftToeBase",
                "RightToeBase",
            ],
            {
                "LeftHandMiddle3": "left_sphere_hand_link",
                "RightHandMiddle3": "right_sphere_hand_link",
                "LeftFoot": "left_ankle_intermediate_1_link",
                "RightFoot": "right_ankle_intermediate_1_link",
                "LeftToeBase": "left_ankle_roll_sphere_5_link",
                "RightToeBase": "right_ankle_roll_sphere_5_link",
            },
            6,
        ),
    )
    for case, scene, token, joints, official_mapping, expected in cases:
        model = mujoco.MjModel.from_xml_path(scene)
        resolved, object_geoms = resolve_intended_contact_mapping(
            model, case, joints, official_mapping, token
        )
        assert len(resolved) == expected
        assert object_geoms
        assert all(item["robot_geom_ids"] for item in resolved)


def _fake_provenance_payload(root: Path) -> dict[str, object]:
    records = [
        _interaction_file_record(root, root / "human.npy", ["primary_human_motion"]),
        _interaction_file_record(root, root / "solver.py", ["holosoma_runtime_code"]),
    ]
    records.sort(key=lambda value: str(value["path"]))
    return {
        "files": records,
        "file_count": len(records),
        "file_bundle_sha256": hashlib.sha256(
            json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


@pytest.mark.parametrize("tampered", ["solver.py", "human.npy"])
def test_interaction_provenance_rejects_tampered_upstream_or_input(
    tmp_path: Path, tampered: str
) -> None:
    (tmp_path / "solver.py").write_text("solver-v1\n", encoding="utf-8")
    (tmp_path / "human.npy").write_bytes(b"complete-human-input")
    payload = _fake_provenance_payload(tmp_path)
    _verify_provenance_file_records(tmp_path, payload)

    (tmp_path / tampered).write_bytes(b"tampered")

    with pytest.raises(ValueError, match="provenance input changed"):
        _verify_provenance_file_records(tmp_path, payload)


def _fake_interaction_outputs(run_dir: Path) -> dict[str, object]:
    paths = _interaction_output_paths(run_dir, "box")
    for index, path in enumerate(paths.values()):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"artifact-{index}".encode())
    summary: dict[str, object] = {
        "provenance_ledger_sha256": "a" * 64,
        "qpos_sha256": sha256_file(paths["raw_qpos"]),
        "expanded_scene_sha256": sha256_file(paths["expanded_scene"]),
        "per_frame_metrics_sha256": sha256_file(paths["per_frame_metrics"]),
        "intended_contact_mapping_artifact_sha256": sha256_file(
            paths["intended_contact_mapping"]
        ),
        "intended_contact_source_contract_sha256": sha256_file(
            paths["intended_contact_source_contract"]
        ),
        "intended_contact_per_semantic_sha256": sha256_file(
            paths["intended_contact_per_semantic"]
        ),
    }
    _, ledger_hash, _ = _write_interaction_output_ledger(
        run_dir, "box", "full", "a" * 64
    )
    summary["output_artifact_ledger_sha256"] = ledger_hash
    return summary


@pytest.mark.parametrize(
    "artifact", ["raw_qpos", "expanded_scene", "intended_contact_source_contract"]
)
def test_interaction_output_ledger_rejects_tampered_artifacts(
    tmp_path: Path, artifact: str
) -> None:
    summary = _fake_interaction_outputs(tmp_path)
    _verify_interaction_output_ledger(
        tmp_path, "box", "full", "a" * 64, summary
    )

    _interaction_output_paths(tmp_path, "box")[artifact].write_bytes(b"tampered")

    with pytest.raises(ValueError, match="output artifact changed"):
        _verify_interaction_output_ledger(
            tmp_path, "box", "full", "a" * 64, summary
        )


def test_interaction_output_ledger_rejects_stale_hash(tmp_path: Path) -> None:
    summary = _fake_interaction_outputs(tmp_path)
    summary["output_artifact_ledger_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="output ledger is stale"):
        _verify_interaction_output_ledger(
            tmp_path, "box", "full", "a" * 64, summary
        )
