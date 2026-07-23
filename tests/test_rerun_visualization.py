from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from retargeting_comparison.rerun_visualization import (
    CLOSEUP_METHOD_KEYS,
    EXPECTED_TRAJECTORY_KEYS,
    EXPECTED_VIEW_INSTANCE_COUNTS,
    METHOD_STYLES,
    VIEW_METHOD_KEYS,
    UrdfSemanticKinematics,
    _build_evidence_binding,
    _frame_label,
    _metric_path,
    _run_output_path,
    inspect_rerun_recording,
    load_stage1_visualization,
    root_frame_points,
    validate_rerun_manifest_contract,
    visual_instance_poses,
    visualize_stage1,
)
from retargeting_comparison.constants import STAGE1_RUN_DIRECTORIES
from retargeting_comparison.io_utils import sha256_file
from retargeting_comparison.robot_model import CanonicalRobotModel, default_robot_scene


URDF = Path(
    "external/holosoma/src/holosoma/holosoma/data/robots/g1/g1_29dof.urdf"
)
FINAL_MODE = os.environ.get("RTCMP_FINAL_TESTS") == "1"


def _missing_artifact(message: str) -> None:
    if FINAL_MODE:
        pytest.fail(message)
    pytest.skip(message)


def test_all_operating_points_and_sparse_diagnostics_are_declared() -> None:
    assert [style.key for style in METHOD_STYLES] == [
        "sparse-neutral",
        "dense",
        "gmr",
        "omniretarget",
        "protomotions-v2.3",
        "protomotions-v3",
        "unitree-reference",
        "sparse-a",
        "sparse-b",
    ]
    assert [style.key for style in METHOD_STYLES if style.operating_point] == [
        "sparse-neutral",
        "dense",
        "gmr",
        "omniretarget",
        "protomotions-v2.3",
        "protomotions-v3",
    ]
    reference = next(style for style in METHOD_STYLES if style.key == "unitree-reference")
    assert reference.external_reference
    assert not reference.operating_point
    assert not reference.annotate_artifact_causes
    assert reference.display_name == "Unitree-attributed external reference"
    assert "unitree-reference" in VIEW_METHOD_KEYS["world"]
    assert "unitree-reference" in VIEW_METHOD_KEYS["root_frame"]
    assert tuple(style.key for style in METHOD_STYLES) == EXPECTED_TRAJECTORY_KEYS
    assert set(EXPECTED_TRAJECTORY_KEYS) == set(STAGE1_RUN_DIRECTORIES)
    assert CLOSEUP_METHOD_KEYS == EXPECTED_TRAJECTORY_KEYS
    assert EXPECTED_VIEW_INSTANCE_COUNTS == {
        "grid": 9,
        "world": 7,
        "root_frame": 7,
        "seeds": 3,
        "closeups": 9,
    }


def test_visualization_paths_have_no_legacy_fallbacks(tmp_path: Path) -> None:
    sequence = "pilot"
    for key in EXPECTED_TRAJECTORY_KEYS:
        assert _run_output_path(tmp_path, sequence, key) == (
            tmp_path
            / "runs"
            / sequence
            / STAGE1_RUN_DIRECTORIES[key]
            / "canonical_g1.npz"
        )
        assert _metric_path(tmp_path, key) == (
            tmp_path
            / "metrics"
            / "stage1_publication"
            / "runs"
            / f"{key}_per_frame.csv"
        )


def test_output_metric_source_evaluator_robot_binding_rejects_stale_output(
    tmp_path: Path,
) -> None:
    key = "gmr"
    output = _run_output_path(tmp_path, "pilot", key)
    metric = _metric_path(tmp_path, key)
    metric_summary = metric.with_name(f"{key}_summary.json")
    source = tmp_path / "source/canonical.npz"
    evaluator_path = tmp_path / "manifests/evaluator.yaml"
    robot = tmp_path / "robot/scene.xml"
    urdf = tmp_path / "robot/g1.urdf"
    publication = tmp_path / "metrics/stage1_core_summary.csv"
    for path, content in (
        (output, b"canonical-output"),
        (metric, b"source_frame_idx,rf_kpe_all_m\n0,0.1\n"),
        (source, b"canonical-source"),
        (evaluator_path, b"schema_version: 3\n"),
        (robot, b"<mujoco/>"),
        (urdf, b"<robot/>"),
        (publication, b"key,output_sha256\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    human = SimpleNamespace(source_sha256="a" * 64, timestamps=np.arange(2))
    evaluator = {
        "schema_version": 3,
        "robot_xml": robot.relative_to(tmp_path).as_posix(),
        "robot_xml_sha256": sha256_file(robot),
        "source_sha256": human.source_sha256,
    }
    summary = {
        "canonical_source_sha256": human.source_sha256,
        "evaluator_protocol_sha256": sha256_file(evaluator_path),
        "robot_model_sha256": sha256_file(robot),
        "frames": 2,
    }
    metric_summary.write_text(json.dumps(summary), encoding="utf-8")
    row = {
        **{key: str(value) for key, value in summary.items()},
        "output_path": output.relative_to(tmp_path).as_posix(),
        "output_sha256": sha256_file(output),
    }
    binding = _build_evidence_binding(
        root=tmp_path,
        key=key,
        output_path=output,
        metrics_path=metric,
        metrics_summary_path=metric_summary,
        publication_row=row,
        publication_path=publication,
        human=human,
        source_path=source,
        evaluator_path=evaluator_path,
        evaluator=evaluator,
        urdf_path=urdf,
    )
    assert binding["verified"] is True
    assert binding["output"]["sha256"] == sha256_file(output)

    stale = dict(row)
    stale["output_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="output hash is stale"):
        _build_evidence_binding(
            root=tmp_path,
            key=key,
            output_path=output,
            metrics_path=metric,
            metrics_summary_path=metric_summary,
            publication_row=stale,
            publication_path=publication,
            human=human,
            source_path=source,
            evaluator_path=evaluator_path,
            evaluator=evaluator,
            urdf_path=urdf,
        )


def test_skipped_recording_cannot_replace_acceptance_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    diagnostic_data = SimpleNamespace(
        missing_methods={"protomotions-v3": "output missing"}
    )
    monkeypatch.setattr(
        "retargeting_comparison.rerun_visualization.load_stage1_visualization",
        lambda *args, **kwargs: diagnostic_data,
    )
    with pytest.raises(ValueError, match="cannot write the Stage 1 acceptance manifest"):
        visualize_stage1(
            repo_root=tmp_path,
            skip_missing_methods=True,
        )


def test_external_reference_label_never_assigns_artifact_status() -> None:
    metrics = {
        "invalid_artifact": np.asarray([1.0]),
        "ground_penetration_artifact": np.asarray([1.0]),
        "joint_limit_artifact": np.asarray([1.0]),
        "left_foot_skating": np.asarray([1.0]),
        "right_foot_skating": np.asarray([1.0]),
    }
    method = SimpleNamespace(metrics=metrics)
    reference = next(style for style in METHOD_STYLES if style.key == "unitree-reference")
    omniretarget = next(style for style in METHOD_STYLES if style.key == "omniretarget")
    assert _frame_label(reference, method, 0) == reference.display_name
    assert "ARTIFACT" not in _frame_label(reference, method, 0).upper()
    assert _frame_label(omniretarget, method, 0).endswith(
        "INVALID+PENETRATION+JOINT-LIMIT+SKATING-L+SKATING-R"
    )


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
    not FINAL_MODE and not URDF.is_file(),
    reason="frozen Holosoma checkout is not present",
)
def test_holosoma_g1_visual_assets_are_complete() -> None:
    robot = UrdfSemanticKinematics(URDF)
    assert len(robot.visuals) == 35
    assert len({visual.link_name for visual in robot.visuals}) == 35
    assert all(visual.mesh_path.is_file() for visual in robot.visuals)
    assert sum(visual.mesh_path.stat().st_size for visual in robot.visuals) > 19_000_000
    assert {visual.rgba for visual in robot.visuals} == {
        (51, 51, 51, 255),
        (178, 178, 178, 255),
    }


@pytest.mark.skipif(
    not FINAL_MODE and (not URDF.is_file() or not default_robot_scene().is_file()),
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
    if not (Path("source/canonical_human") / f"{sequence_id}.npz").is_file():
        _missing_artifact("licensed/generated local Stage 1 artifacts are not present")
    try:
        data = load_stage1_visualization(".")
    except FileNotFoundError as error:
        _missing_artifact(f"complete local Stage 1 artifacts are not present: {error}")
    assert data.frame_count == 450
    assert set(data.methods) == {style.key for style in METHOD_STYLES}
    assert len(data.robot_visuals) == 35
    assert all(method.positions.shape[0] >= 450 for method in data.methods.values())
    assert all(method.positions.shape[1:] == (17, 3) for method in data.methods.values())
    assert all(
        method.link_transforms["pelvis"].shape[0] >= 450
        and method.link_transforms["pelvis"].shape[1:] == (4, 4)
        for method in data.methods.values()
    )
    assert all(
        np.isfinite(value).all()
        for method in data.methods.values()
        for value in method.metrics.values()
    )
    visual = data.robot_visuals[0]
    method_keys = tuple(style.key for style in METHOD_STYLES)
    transforms = {key: data.methods[key].link_transforms for key in method_keys}
    translations, rotations, scales = visual_instance_poses(
        visual,
        method_keys,
        transforms,
        frame=0,
    )
    assert translations.shape == (len(METHOD_STYLES), 3)
    assert rotations.shape == (len(METHOD_STYLES), 3, 3)
    assert scales.shape == (len(METHOD_STYLES), 3)
    assert np.allclose(np.linalg.det(rotations), 1.0, atol=1e-8)


def test_diagnostic_loading_keeps_available_reference_as_full_g1() -> None:
    sequence_id = "dance1_subject1_f000000_000600"
    required = (
        Path("source/canonical_human") / f"{sequence_id}.npz",
        Path("runs")
        / sequence_id
        / "unitree-attributed-reference"
        / "canonical_g1.npz",
        Path("metrics/stage1_publication/runs/unitree-reference_per_frame.csv"),
        URDF,
    )
    if not all(path.is_file() for path in required):
        _missing_artifact("licensed/generated Unitree reference artifacts are not present")
    data = load_stage1_visualization(".", skip_missing_methods=True)
    reference = data.methods["unitree-reference"]
    assert reference.positions.shape == (600, 17, 3)
    assert len(reference.link_transforms) >= 35
    assert reference.link_transforms["pelvis"].shape == (600, 4, 4)
    assert np.isfinite(reference.link_transforms["pelvis"]).all()


def test_large_random_file_cannot_impersonate_rerun(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = tmp_path / "fake.rrd"
    fake.write_bytes(b"not-an-rrd" * 150_000)
    rejected = subprocess.CompletedProcess(
        args=["rerun", "rrd", "verify"],
        returncode=1,
        stdout="",
        stderr="invalid RRD",
    )
    monkeypatch.setattr(
        "retargeting_comparison.rerun_visualization._run_rerun_cli",
        lambda arguments: rejected,
    )
    with pytest.raises(ValueError, match="rrd verify.*rejected"):
        inspect_rerun_recording(fake, frames=600)


def test_stale_rerun_manifest_schema_is_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "stale.json"
    manifest.write_text(json.dumps({"schema_version": 4}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema v5"):
        validate_rerun_manifest_contract(Path.cwd(), manifest, verify_recording=False)


def test_final_rerun_manifest_and_rrd_pass_deep_validation() -> None:
    manifest = Path("manifests/rerun_visualization.json")
    if not manifest.is_file() or json.loads(manifest.read_text()).get("schema_version") != 5:
        _missing_artifact("final schema-v5 Rerun manifest is not built")
    result = validate_rerun_manifest_contract(".")
    assert result["result"] == "verified"
    assert result["frame_marker_rows"] == 450
    assert result["exact_grid_trajectory_set"] == sorted(EXPECTED_TRAJECTORY_KEYS)


def test_tampered_final_rerun_hash_is_rejected(tmp_path: Path) -> None:
    manifest = Path("manifests/rerun_visualization.json")
    if not manifest.is_file() or json.loads(manifest.read_text()).get("schema_version") != 5:
        _missing_artifact("final schema-v5 Rerun manifest is not built")
    value = json.loads(manifest.read_text())
    value["output_sha256"] = "0" * 64
    tampered = tmp_path / "tampered-rerun.json"
    tampered.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_rerun_manifest_contract(".", tampered, verify_recording=False)
