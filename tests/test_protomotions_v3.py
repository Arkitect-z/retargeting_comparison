from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from retargeting_comparison.constants import G1_JOINT_NAMES
from retargeting_comparison.protomotions_v3 import (
    TARGET_RAW_FRAMES,
    audit_robot_assets,
    build_native_command,
    convert_native_output,
    joint_limit_diagnostics,
    native_environment,
    prepare_scale_variant_keypoints,
    solver_log_diagnostics,
    validate_canonical_fk,
    validate_keypoint_input,
)
from retargeting_comparison.protomotions_v3_campaign import (
    CampaignJob,
    SCALE_VARIANTS,
    _expected_implementation_hashes,
    _standard_timing,
    _valid_evidence,
    _valid_output,
    campaign_jobs,
)
from retargeting_comparison.protomotions_v3_native import (
    _canonical_human_mapping,
    _scale_mapping,
)
from retargeting_comparison.schemas import CanonicalG1, CanonicalHuman
from retargeting_comparison.scale_worker import scale_execution_contract
from retargeting_comparison.io_utils import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def test_native_command_freezes_600_frames_and_formal_timing(tmp_path) -> None:
    command = build_native_command(
        python="/usr/bin/python3",
        repo_root=ROOT,
        source=tmp_path / "canonical_human.npz",
        keypoints=tmp_path / "keypoints.npy",
        native_output=tmp_path / "native.npz",
        timing_json=tmp_path / "timing.json",
    )
    assert command[command.index("--target-raw-frames") + 1] == "600"
    assert command[command.index("--warmup-runs") + 1] == "1"
    assert command[command.index("--measured-runs") + 1] == "3"
    assert command[command.index("--source") + 1] == str(
        (tmp_path / "canonical_human.npz").resolve()
    )


def test_cpu_native_environment_is_explicit_and_suppresses_cuda_probe() -> None:
    env = native_environment(ROOT, "/usr/bin/python3", device="cpu")
    assert env["JAX_PLATFORMS"] == "cpu"
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["JAX_SKIP_CUDA_CONSTRAINTS_CHECK"] == "1"


def test_campaign_orders_formal_timing_then_four_pre_solver_variants(tmp_path) -> None:
    jobs = campaign_jobs(tmp_path, "pilot")
    assert jobs[0].name == "formal_timing"
    assert jobs[0].warmup_runs == 1
    assert jobs[0].measured_runs == 3
    assert [job.name for job in jobs[1:]] == list(SCALE_VARIANTS)
    assert all(job.warmup_runs == 0 and job.measured_runs == 1 for job in jobs[1:])


def _campaign_witness(
    tmp_path: Path, *, variant: str = "root_minus_5"
) -> tuple[CampaignJob, CanonicalHuman, Path, Path]:
    source = ROOT / "source/canonical_human/dance1_subject1_f000000_000600.npz"
    if not source.is_file():
        pytest.skip("Generated canonical Pilot source is intentionally not committed")
    human = CanonicalHuman.load(source)
    root_multiplier, local_multiplier = SCALE_VARIANTS[variant]
    output = tmp_path / "canonical_g1.npz"
    evidence = tmp_path / "evidence.json"
    job = CampaignJob(
        name=variant,
        output=output,
        work_dir=tmp_path / "work",
        evidence=evidence,
        warmup_runs=0,
        measured_runs=1,
        root_multiplier=root_multiplier,
        local_multiplier=local_multiplier,
    )
    repetition = {
        "role": "measured",
        "timing_boundary": "canonical_source_file_to_canonical_g1_in_memory",
        "native_boundary": "native_input_ready_to_native_g1_in_memory",
        "thread_limit": 1,
        "cpu_affinity": [0],
        "timing_artifact_sha256": "a" * 64,
    }
    qpos = np.zeros((len(human.timestamps), 36), dtype=np.float64)
    qpos[:, 3] = 1.0
    motion = CanonicalG1(
        qpos=qpos,
        fps=human.fps,
        source_frame_idx=np.arange(len(qpos)),
        valid=np.ones(len(qpos), dtype=bool),
        per_frame_solve_time_s=np.zeros(len(qpos)),
        metadata={
            "method": "protomotions_v3",
            "upstream_commit": "49fe5ad69de67ebbc07ea2b25d41b0f622c15c3c",
            "completion_status": "succeeded",
            "canonical_source_sha256": human.source_sha256,
            "canonical_source_file_sha256": sha256_file(source),
            "config_sha256": sha256_file(ROOT / "configs/protomotions_v3.yaml"),
            "scale_protocol_sha256": sha256_file(
                ROOT / "configs/scale_policy_sensitivity.yaml"
            ),
            "scale_variant": variant,
            "root_scale_multiplier": root_multiplier,
            "local_scale_multiplier": local_multiplier,
            "experiment_role": "native_response_fixed_canonical_contact",
            "fixed_contact_labels": True,
            "implementation_hashes": _expected_implementation_hashes(ROOT),
            "scale_execution_contract": scale_execution_contract(
                ROOT, "protomotions_v3"
            ),
            "scale_runtime_environment": {"environment_name": "egoallo"},
            "formal_timing_boundary": (
                "canonical_source_file_to_canonical_g1_in_memory"
            ),
            "timing_protocol": {"repetitions": [repetition]},
        },
    )
    motion.save(output, source_frame_count=len(human.timestamps))
    evidence.write_text(
        json.dumps(
            {
                "status": "succeeded",
                "canonical_output": str(output.resolve()),
                "canonical_output_sha256": sha256_file(output),
                "scale_variant": variant,
                "root_scale_multiplier": root_multiplier,
                "local_scale_multiplier": local_multiplier,
                "canonical_source_file_sha256": sha256_file(source),
                "canonical_source_embedded_sha256": human.source_sha256,
                "config_sha256": sha256_file(
                    ROOT / "configs/protomotions_v3.yaml"
                ),
                "scale_protocol_sha256": sha256_file(
                    ROOT / "configs/scale_policy_sensitivity.yaml"
                ),
                "implementation_hashes": _expected_implementation_hashes(ROOT),
                "scale_execution_contract": scale_execution_contract(
                    ROOT, "protomotions_v3"
                ),
                "scale_runtime_environment": {"environment_name": "egoallo"},
            }
        )
    )
    return job, human, source, evidence


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (
        ("scale_variant", "root_plus_5"),
        ("root_scale_multiplier", 1.05),
        ("canonical_source_sha256", "0" * 64),
        ("config_sha256", "1" * 64),
        ("implementation_hashes", {"wrapper": "2" * 64}),
    ),
)
def test_campaign_reuse_fails_closed_on_variant_source_config_and_code(
    tmp_path: Path, field: str, bad_value: object
) -> None:
    job, human, source, _ = _campaign_witness(tmp_path)
    assert _valid_output(
        job.output,
        human,
        root=ROOT,
        job=job,
        source_path=source,
        expected_warmup=0,
        expected_measured=1,
    )
    motion = CanonicalG1.load(job.output)
    motion.metadata[field] = bad_value
    motion.save(job.output, source_frame_count=len(human.timestamps))
    assert not _valid_output(
        job.output,
        human,
        root=ROOT,
        job=job,
        source_path=source,
        expected_warmup=0,
        expected_measured=1,
    )


def test_campaign_evidence_binds_exact_output_and_all_registered_hashes(
    tmp_path: Path,
) -> None:
    job, human, source, evidence = _campaign_witness(tmp_path)
    assert _valid_evidence(
        evidence, root=ROOT, job=job, human=human, source_path=source
    )
    value = json.loads(evidence.read_text())
    value["local_scale_multiplier"] = 1.05
    evidence.write_text(json.dumps(value))
    assert not _valid_evidence(
        evidence, root=ROOT, job=job, human=human, source_path=source
    )


def test_standard_timing_requires_independent_cold_and_same_process_one_three(
    tmp_path,
) -> None:
    def save_motion(path: Path, repetitions: list[dict[str, object]]) -> None:
        for repetition in repetitions:
            repetition.setdefault("canonical_qpos_sha256", "a" * 64)
            repetition.setdefault("canonical_g1_content_sha256", "b" * 64)
            repetition.setdefault(
                "runtime_witness_enabled", path.name == "cold.npz"
            )
        motion = CanonicalG1(
            qpos=np.column_stack(
                (
                    np.zeros((4, 3)),
                    np.tile([1.0, 0.0, 0.0, 0.0], (4, 1)),
                    np.zeros((4, 29)),
                )
            ),
            fps=2.0,
            source_frame_idx=np.arange(4),
            valid=np.ones(4, dtype=bool),
            per_frame_solve_time_s=np.zeros(4),
            metadata={
                "completion_status": "succeeded",
                "timing_protocol": {
                    "initialization_time_s": 0.25,
                    "repetitions": repetitions,
                },
            },
        )
        motion.save(path, source_frame_count=4)

    cold = tmp_path / "cold.npz"
    formal = tmp_path / "formal.npz"
    cold_evidence = tmp_path / "cold.json"
    formal_evidence = tmp_path / "formal.json"
    save_motion(
        cold,
        [
            {
                "role": "measured",
                "wall_time_s": 10.0,
                "timing_boundary": "canonical_source_file_to_canonical_g1_in_memory",
                "native_boundary": "native_input_ready_to_native_g1_in_memory",
            }
        ],
    )
    save_motion(
        formal,
        [
            {"role": "warmup", "wall_time_s": 8.0},
            {
                "role": "measured",
                "wall_time_s": 4.0,
                "steady_end_to_end_total_s": 5.0,
                "native_core_s": 3.0,
            },
            {
                "role": "measured",
                "wall_time_s": 6.0,
                "steady_end_to_end_total_s": 7.0,
                "native_core_s": 5.0,
            },
            {
                "role": "measured",
                "wall_time_s": 5.0,
                "steady_end_to_end_total_s": 6.0,
                "native_core_s": 4.0,
            },
        ],
    )
    cold_evidence.write_text(json.dumps({"process_wall_time_s": 11.0}))
    formal_evidence.write_text(json.dumps({"process_wall_time_s": 24.0}))
    timing = _standard_timing(
        cold_output=cold,
        cold_evidence=cold_evidence,
        formal_output=formal,
        formal_evidence=formal_evidence,
        sequence_id="pilot",
    )
    assert timing["end_to_end_rtf_raw"] == [2.5, 3.5, 3.0]
    assert timing["end_to_end_rtf_median"] == 3.0
    assert timing["native_core_rtf_raw"] == [1.5, 2.5, 2.0]
    assert timing["native_core_rtf_median"] == 2.0
    assert timing["cold_process_wall_s"] == 11.0
    assert timing["warm_process_wall_s"] == 24.0
    assert timing["protocol"]["end_to_end_boundary"] == (
        "canonical_source_file_to_canonical_g1_in_memory"
    )
    assert timing["protocol"]["native_core_boundary"] == (
        "native_input_ready_to_native_g1_in_memory"
    )


def test_solver_log_diagnostics_separates_repeated_calls_and_flags_ceiling(
    tmp_path,
) -> None:
    log = tmp_path / "solver.log"
    log.write_text(
        "\n".join(
            (
                "INFO - step #0: cost=1.2e+03 lambd=0.1",
                "INFO - step #1: cost=9.0e+02 lambd=0.2",
                "INFO - Terminated @ iteration #2: cost=8.9e+02 criteria=[1 0 0], term_deltas=1e-06,2e-01,3e-04",
                "INFO - step #0: cost=1.2e+03 lambd=0.1",
                "INFO - step #1: cost=8.5e+02 lambd=0.2",
                "INFO - step #2: cost=8.0e+02 lambd=0.4",
                "INFO - Terminated @ iteration #3: cost=7.9e+02 criteria=[0 0 1], term_deltas=2e-05,3e-01,4e-07",
            )
        )
    )
    diagnostics = solver_log_diagnostics(log, configured_max_iterations=3)
    assert diagnostics["solver_call_count"] == 2
    assert diagnostics["solver_calls"][0]["logged_iteration_count"] == 2
    assert diagnostics["solver_calls"][1]["logged_iteration_count"] == 3
    assert diagnostics["solver_calls"][1]["final_logged_cost"] == 800.0
    assert diagnostics["solver_calls"][1]["final_cost"] == 790.0
    assert diagnostics["solver_calls"][1]["termination_criteria"]["parameters"] is True
    assert diagnostics["all_calls_have_termination_record"] is True
    assert diagnostics["any_configured_iteration_ceiling_reached"] is True


def test_frozen_keypoint_adapter_is_finite_metric_and_600_frames() -> None:
    path = (
        ROOT
        / "source_adapters/protomotions_v3/dance1_subject1_f000000_000600/keypoints.npy"
    )
    if not path.is_file():
        pytest.skip("Generated source adapter is intentionally not committed")
    audit = validate_keypoint_input(path)
    assert audit["frames"] == TARGET_RAW_FRAMES
    assert audit["keypoints"] == 18
    assert audit["coordinate_unit"] == "metre"
    assert audit["max_rotation_orthogonality_error"] < 1e-12


def test_timed_in_memory_adapter_is_exactly_the_declared_file_contract() -> None:
    source = (
        ROOT
        / "source/canonical_human/dance1_subject1_f000000_000600.npz"
    )
    declared = (
        ROOT
        / "source_adapters/protomotions_v3/dance1_subject1_f000000_000600/keypoints.npy"
    )
    if not source.is_file() or not declared.is_file():
        pytest.skip("Generated Pilot source adapters are intentionally not committed")
    human = CanonicalHuman.load(source)
    actual = _canonical_human_mapping(human)
    expected = np.load(declared, allow_pickle=True).item()
    for key in (
        "positions",
        "orientations",
        "left_foot_contacts",
        "right_foot_contacts",
        "fps",
    ):
        assert np.array_equal(np.asarray(actual[key]), np.asarray(expected[key]))

    identity = _scale_mapping(actual, root_multiplier=1.0, local_multiplier=1.0)
    for key in (
        "positions",
        "orientations",
        "left_foot_contacts",
        "right_foot_contacts",
        "fps",
    ):
        assert np.array_equal(np.asarray(identity[key]), np.asarray(expected[key]))

    scaled = _scale_mapping(actual, root_multiplier=1.05, local_multiplier=0.95)
    root = np.asarray(actual["positions"])[:, 0]
    scaled_root = np.asarray(scaled["positions"])[:, 0]
    assert np.allclose(scaled_root - scaled_root[0], 1.05 * (root - root[0]))
    assert np.array_equal(scaled["orientations"], actual["orientations"])


def test_scale_variant_changes_root_and_local_targets_before_native_scale(tmp_path) -> None:
    source = tmp_path / "source.npy"
    output = tmp_path / "variant.npy"
    frames = 3
    positions = np.zeros((frames, 18, 3), dtype=np.float64)
    positions[:, 0, 0] = [2.0, 3.0, 4.0]
    positions[:, 1:, 0] = positions[:, :1, 0] + 1.0
    positions[:, :, 2] = np.linspace(0.0, 1.0, 18)
    orientations = np.tile(np.eye(3), (frames, 18, 1, 1))
    contacts = np.zeros((frames, 2), dtype=np.int64)
    np.save(
        source,
        {
            "positions": positions,
            "orientations": orientations,
            "left_foot_contacts": contacts,
            "right_foot_contacts": contacts,
            "fps": 30.0,
        },
        allow_pickle=True,
    )
    # The synthetic one-metre skeleton passes the metre sanity check.
    audit = prepare_scale_variant_keypoints(
        source, output, root_scale_multiplier=0.5, local_scale_multiplier=2.0
    )
    variant = np.load(output, allow_pickle=True).item()
    assert np.allclose(variant["positions"][:, 0, 0], [2.0, 2.5, 3.0])
    assert np.allclose(
        variant["positions"][:, 1, 0] - variant["positions"][:, 0, 0], 2.0
    )
    assert audit["orientations_unchanged"] is True
    assert audit["contact_labels_unchanged"] is True


def test_robot_asset_joint_order_fk_and_15_limit_differences() -> None:
    if not (ROOT / "external/ProtoMotions").is_dir() or not (
        ROOT / "external/holosoma"
    ).is_dir():
        pytest.skip("External pinned assets are intentionally not committed")
    audit = audit_robot_assets(ROOT)
    native = audit["native_retargeting_asset"]
    evaluator = audit["canonical_evaluator_asset"]
    assert tuple(native["proto_joint_order"]) == G1_JOINT_NAMES
    assert tuple(native["reference_joint_order"]) == G1_JOINT_NAMES
    assert native["origin_differences"] == []
    assert native["axis_differences"] == []
    assert native["limit_difference_count"] == 15
    assert native["max_actuated_child_fk_position_difference_m"] < 1e-12
    # The evaluator deliberately uses a separate Holosoma asset revision.
    assert set(evaluator["origin_differences"]) == {
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "right_shoulder_pitch_joint",
    }


def _native_file(path: Path, frames: int, *, include_nan: bool = False) -> None:
    root = np.arange(frames * 3, dtype=np.float32).reshape(frames, 3) / 10.0
    quaternion = np.tile(np.asarray([-2.0, 0.0, 0.0, 0.0]), (frames, 1))
    if frames > 1:
        quaternion[1] *= -1.0
    joints = np.arange(frames * 29, dtype=np.float32).reshape(frames, 29) / 100.0
    if include_nan:
        joints[0, 0] = np.nan
    np.savez_compressed(
        path,
        base_frame_pos=root,
        base_frame_wxyz=quaternion,
        joint_angles=joints,
        fps=30.0,
        source_fps=30.0,
        subsample_factor=1,
    )


def test_native_output_becomes_float64_wxyz_canonical_and_preserves_timing(tmp_path) -> None:
    native = tmp_path / "native.npz"
    timing = tmp_path / "timing.json"
    _native_file(native, 4)
    timing.write_text(
        json.dumps(
            {
                "repetitions": [
                    {"role": "warmup", "wall_time_s": 8.0},
                    {"role": "measured", "wall_time_s": 4.0},
                    {"role": "measured", "wall_time_s": 6.0},
                    {"role": "measured", "wall_time_s": 5.0},
                ]
            }
        )
    )
    reversed_names = tuple(reversed(G1_JOINT_NAMES))
    motion = convert_native_output(
        native,
        source_frame_count=4,
        native_joint_names=reversed_names,
        timing_json=timing,
    )
    assert motion.qpos.shape == (4, 36)
    assert motion.qpos.dtype == np.float64
    assert np.allclose(motion.qpos[:, 3], 1.0)
    assert np.allclose(np.linalg.norm(motion.qpos[:, 3:7], axis=1), 1.0)
    assert np.allclose(motion.per_frame_solve_time_s, 1.25)
    assert motion.metadata["completion_status"] == "succeeded"
    original = np.arange(4 * 29, dtype=np.float32).reshape(4, 29) / 100.0
    assert np.array_equal(motion.qpos[:, 7:], original[:, ::-1])


def test_native_output_rejects_nonfinite_and_marks_short_output_incomplete(tmp_path) -> None:
    bad = tmp_path / "bad.npz"
    _native_file(bad, 2, include_nan=True)
    with pytest.raises(ValueError, match="NaN/Inf"):
        convert_native_output(bad, source_frame_count=2)
    short = tmp_path / "short.npz"
    _native_file(short, 2)
    motion = convert_native_output(short, source_frame_count=600)
    assert motion.metadata["completion_status"] == "incomplete"


def test_canonical_mujoco_fk_accepts_converted_qpos(tmp_path) -> None:
    robot_xml = (
        ROOT
        / "external/holosoma/src/holosoma/holosoma/data/robots/g1/scenes/scene_g1_29dof_wbt_plane.xml"
    )
    if not robot_xml.is_file():
        pytest.skip("External canonical robot asset is intentionally not committed")
    native = tmp_path / "native.npz"
    frames = 3
    np.savez_compressed(
        native,
        base_frame_pos=np.tile([0.0, 0.0, 0.8], (frames, 1)),
        base_frame_wxyz=np.tile([1.0, 0.0, 0.0, 0.0], (frames, 1)),
        joint_angles=np.zeros((frames, 29)),
        fps=30.0,
        source_fps=30.0,
        subsample_factor=1,
    )
    motion = convert_native_output(native, source_frame_count=frames)
    audit = validate_canonical_fk(motion, robot_xml, sample_count=frames)
    assert audit["finite"] is True
    assert audit["sampled_frames"] == [0, 1, 2]
    proto_urdf = (
        ROOT
        / "external/ProtoMotions/protomotions/data/assets/urdf/for_retargeting/g1.urdf"
    )
    limits = joint_limit_diagnostics(motion, proto_urdf)
    assert limits["maximum_violation_rad"] == 0.0
    assert limits["violating_frame_count"] == 0
