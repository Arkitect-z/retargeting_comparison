from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml
import numpy as np
import retargeting_comparison.stage2 as stage2_module

from retargeting_comparison.io_utils import load_yaml
from retargeting_comparison.schemas import CanonicalG1, CanonicalHuman, RunManifest, RunStatus
from retargeting_comparison.stage2 import (
    MethodSpec,
    SequenceSpec,
    Stage2Error,
    build_stage2_plan,
    deterministic_schedule,
    discover_lafan_sequences,
    execute_stage2,
    job_is_complete,
    project_resources,
    resolve_methods,
    select_budgeted_design,
    simplify_methods_before_dataset_reduction,
    validate_plan_for_launch,
    verify_plan_hash,
    _account_prior_execution_wall,
    _ExecutionGuard,
    _execute_stage2_locked,
    _git_identity,
    _run_job,
    _validate_stage1_formal_outputs,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _sequence(
    index: int, *, reference: bool = False, duration_s: float = 10.0
) -> SequenceSpec:
    frames = int(duration_s * 30)
    return SequenceSpec(
        sequence_id=f"sequence-{index:02d}",
        relative_path=f"subject/sequence_{index:02d}.bvh",
        source_sha256=_digest(f"source-{index}"),
        source_size_bytes=1000 + index,
        frames=frames,
        fps=30.0,
        duration_s=duration_s,
        reference_relative_path=f"sequence_{index:02d}.csv" if reference else None,
        reference_sha256=_digest(f"reference-{index}") if reference else None,
        reference_size_bytes=500 if reference else 0,
        reference_frames=frames if reference else None,
    )


def _method(
    name: str = "omniretarget",
    *,
    workers: int = 6,
    rtf: float = 10.0,
    stratum: str = "native_public_pipeline",
) -> MethodSpec:
    return MethodSpec(
        method=name,
        stratum=stratum,
        public_pipeline=stratum == "native_public_pipeline",
        environment="test",
        workers=workers,
        rtf=rtf,
        startup_s=1.0,
        retained_bytes_per_frame=100,
        retained_bytes_per_sequence=1000,
        estimate_source="test",
    )


def _projection_config(*, wall_h: float, minimum: int = 2) -> dict:
    return {
        "budget": {
            "wall_time_hours": wall_h,
            "retained_storage_gb": 200.0,
            "safety_factor": 1.5,
        },
        "dataset": {
            "selection_seed": "test-source-only-seed",
            "reduced_min_sequences": minimum,
            "reduced_selection": "deterministic_hash_prefix_with_reference_anchor",
        },
        "scheduling": {
            "phase_order": ["omniretarget"],
            "production_passes_per_job": 1,
        },
        "storage_projection": {
            "fixed_manifest_and_summary_bytes": 1000,
            "retain_raw_inputs_in_incremental_budget": True,
            "canonical_source_bytes_per_sequence": 100,
            "canonical_source_bytes_per_frame": 100,
        },
    }


def test_deterministic_lpt_schedule_uses_six_omniretarget_workers() -> None:
    sequences = [_sequence(index, duration_s=5.0 + index) for index in range(12)]
    method = _method()

    first, first_wall, phases = deterministic_schedule(
        sequences, [method], ["omniretarget"]
    )
    second, second_wall, _ = deterministic_schedule(
        sequences, [method], ["omniretarget"]
    )

    assert first == second
    assert first_wall == second_wall
    assert {job.worker_index for job in first} == set(range(6))
    assert phases[0]["workers"] == 6
    serial = sum(method.startup_s + method.rtf * item.duration_s for item in sequences)
    assert first_wall < serial / 4


def test_reference_jobs_use_filename_intersection_only() -> None:
    sequences = [_sequence(0, reference=True), _sequence(1), _sequence(2)]
    reference = _method(
        "unitree-reference", workers=2, rtf=0.01, stratum="external_reference"
    )

    jobs, _, _ = deterministic_schedule(sequences, [reference], ["unitree-reference"])

    assert [job.sequence_id for job in jobs] == ["sequence-00"]
    assert jobs[0].reference_relative_path == "sequence_00.csv"


def test_reference_path_is_never_forwarded_to_non_reference_jobs() -> None:
    sequence = _sequence(0, reference=True)
    dense = _method(
        "dense", workers=1, rtf=0.1, stratum="controlled_common_per_sequence_scale"
    )
    reference = _method(
        "unitree-reference", workers=1, rtf=0.01, stratum="external_reference"
    )
    jobs, _, _ = deterministic_schedule(
        [sequence], [dense, reference], ["dense", "unitree-reference"]
    )
    assert jobs[0].method == "dense"
    assert jobs[0].reference_relative_path is None
    assert jobs[1].reference_relative_path == "sequence_00.csv"


def test_full_design_is_never_reduced_when_it_fits() -> None:
    sequences = [_sequence(index) for index in range(5)]
    config = _projection_config(wall_h=100.0)

    design, selected, projection, fallback = select_budgeted_design(
        sequences, [_method(workers=2, rtf=1.0)], config
    )

    assert design.startswith("full-lafan1-")
    assert selected == sequences
    assert projection["within_budget"] is True
    assert fallback is None


def test_design_id_changes_when_method_or_policy_contract_changes() -> None:
    sequences = [_sequence(0)]
    config = _projection_config(wall_h=100.0)
    first = _method(workers=1, rtf=1.0)
    second = replace(first, policy_sha256="1" * 64)
    first_design = select_budgeted_design(sequences, [first], config)[0]
    second_design = select_budgeted_design(sequences, [second], config)[0]
    assert first_design != second_design
    assert first_design.rsplit("-", 1)[-1] != second_design.rsplit("-", 1)[-1]


def test_budget_overflow_uses_deterministic_named_reduced_prefix() -> None:
    sequences = [_sequence(index) for index in range(7)]
    # Each sequence is ~1001 seconds raw.  With 1.5x safety, a two-sequence
    # design fits one hour while three do not.
    config = _projection_config(wall_h=1.0, minimum=2)
    method = _method(workers=1, rtf=100.0)

    first = select_budgeted_design(sequences, [method], config)
    second = select_budgeted_design(sequences, [method], config)

    assert first == second
    design, selected, projection, fallback = first
    assert design.startswith("reduced-lafan1-n02-")
    assert len(selected) == 2
    assert projection["within_budget"] is True
    assert fallback is not None
    assert fallback["result_metrics_used_for_selection"] is False
    assert fallback["full_projection"]["within_budget"] is False


def test_projection_is_one_production_pass_and_parallel_not_serial() -> None:
    sequences = [_sequence(index, duration_s=20.0) for index in range(6)]
    method = _method(workers=6, rtf=9.0)
    projection = project_resources(sequences, [method], _projection_config(wall_h=48.0))

    one_job = method.startup_s + method.rtf * 20.0
    assert projection["production_passes_per_job"] == 1
    assert projection["timing_repetitions_in_stage2"] == 1
    assert projection["raw_projected_wall_s"] == pytest.approx(one_job)
    assert projection["phase_summaries"][0]["worker_loads_s"] == pytest.approx(
        [one_job] * 6
    )


def test_projection_accounts_preparation_evaluation_and_failure_allowances() -> None:
    config = _projection_config(wall_h=48.0)
    config["budget_allowances"] = {
        "source_preparation_rtf": 0.2,
        "source_preparation_startup_s_per_sequence": 1.0,
        "source_preparation_workers": 1,
        "evaluator_rtf_per_method": 0.4,
        "evaluator_startup_s_per_job": 2.0,
        "evaluator_workers": 1,
        "failure_runtime_fraction": 0.1,
        "failure_storage_fraction": 0.05,
        "analysis_retained_bytes_per_job": 100,
        "analysis_fixed_retained_bytes": 1000,
    }
    projection = project_resources([_sequence(0)], [_method(rtf=1.0)], config)
    components = projection["wall_components_s"]
    assert components["method_phases"] > 0.0
    assert components["source_preparation"] > 0.0
    assert components["evaluation"] > 0.0
    assert components["failure_allowance"] == pytest.approx(
        0.1
        * (
            components["method_phases"]
            + components["source_preparation"]
            + components["evaluation"]
        )
    )
    assert projection["storage_failure_allowance_bytes"] > 0


def test_parallel_contention_factor_inflates_every_multiworker_projection(
    tmp_path: Path,
) -> None:
    config = {
        "scheduling": {"physical_cores": 8, "phase_order": ["omniretarget"]},
        "production_design": {"required": False},
        "methods": {
            "omniretarget": {
                "readiness": "always",
                "stratum": "native_public_pipeline",
                "public_pipeline": True,
                "environment": "test",
                "workers": 6,
                "rtf": 2.0,
                "startup_s_per_sequence": 3.0,
                "retained_bytes_per_frame": 10,
                "retained_bytes_per_sequence": 20,
                "estimate_source": "test",
            }
        },
    }
    methods, _ = resolve_methods(
        tmp_path,
        config,
        cost_probe={
            "conservative_output_bytes_per_frame": 10,
            "conservative_parallel_contention_factor": 1.5,
        },
    )
    assert methods[0].rtf == pytest.approx(3.0)
    assert methods[0].startup_s == pytest.approx(4.5)
    assert methods[0].parallel_contention_factor == pytest.approx(1.5)


def test_registered_high_cost_method_is_removed_before_dataset_reduction() -> None:
    sequences = [_sequence(index, duration_s=100.0) for index in range(5)]
    fast = _method("omniretarget", workers=1, rtf=1.0)
    slow = _method("protomotions-v3", workers=1, rtf=1000.0)
    config = _projection_config(wall_h=1.0, minimum=2)
    config["scheduling"]["phase_order"] = ["omniretarget", "protomotions-v3"]
    config["budget_simplification"] = {
        "policy": "exclude_registered_high_cost_methods_before_reducing_dataset",
        "method_exclusion_order": ["protomotions-v3"],
        "stage1_evidence_retained": True,
        "result_metrics_used_for_decision": False,
    }

    selected, exclusions, initial = simplify_methods_before_dataset_reduction(
        sequences, [fast, slow], config
    )

    assert initial["within_budget"] is False
    assert [method.method for method in selected] == ["omniretarget"]
    assert exclusions[0]["method"] == "protomotions-v3"
    assert exclusions[0]["full_design_fits_after_exclusion"] is True
    assert exclusions[0]["result_metrics_used_for_selection"] is False


def test_method_simplification_is_noop_when_full_design_already_fits() -> None:
    sequences = [_sequence(index, duration_s=10.0) for index in range(3)]
    methods = [_method("omniretarget", workers=3, rtf=1.0)]
    config = _projection_config(wall_h=100.0)
    config["budget_simplification"] = {
        "policy": "exclude_registered_high_cost_methods_before_reducing_dataset",
        "method_exclusion_order": ["omniretarget"],
        "stage1_evidence_retained": True,
        "result_metrics_used_for_decision": False,
    }

    selected, exclusions, initial = simplify_methods_before_dataset_reduction(
        sequences, methods, config
    )

    assert selected == methods
    assert exclusions == []
    assert initial["within_budget"] is True


def test_resume_requires_manifest_plan_and_output_hash(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    attempt_root = run_root / "attempts/dense__sequence-00/attempt-001"
    output = attempt_root / "canonical_g1.npz"
    sequence_manifest = tmp_path / "sequence.json"
    sequence = {
        "schema_version": 2,
        "source_frames": 3,
        "fps": 30.0,
        "source_sha256": "1" * 64,
        "canonical_sha256": "2" * 64,
        "calibration_evidence_sha256": "3" * 64,
        "common_scale": {
            "local_body_scale": 1.0,
            "root_displacement_scale": 1.0,
            "root_alignment_translation_m": [0.0, 0.0, 0.0],
        },
    }
    sequence["payload_sha256"] = stage2_module._canonical_sha256(sequence)
    sequence_manifest.write_text(json.dumps(sequence))
    job = {
        "job_id": "dense__sequence-00",
        "method": "dense",
        "sequence_id": "sequence-00",
        "stratum": "controlled_common_per_sequence_scale",
        "plan_sha256": "a" * 64,
        "job_contract_sha256": "b" * 64,
        "method_contract_sha256": "c" * 64,
        "policy_sha256": "d" * 64,
    }
    execution_path = attempt_root / "execution_contract.json"
    python = tmp_path / "env/bin/python"
    history = tmp_path / "env/conda-meta/history"
    python.parent.mkdir(parents=True)
    history.parent.mkdir(parents=True)
    python.write_text("python")
    history.write_text("history")
    command = [str(python), "worker"]
    execution = {
        "schema_version": 2,
        "design_id": "test-design",
        "config_sha256": "0" * 64,
        "plan_sha256": job["plan_sha256"],
        "job_contract_sha256": job["job_contract_sha256"],
        "method_contract_sha256": job["method_contract_sha256"],
        "policy_sha256": job["policy_sha256"],
        "repository_contract_sha256": "e" * 64,
        "hardware_contract_sha256": "f" * 64,
        "method_contract": {
            "registered_revision": "dense-test",
            "timing_evidence_sha256": "7" * 64,
            "environment_provenance_sha256": "8" * 64,
        },
        "planned_job": {key: job[key] for key in sorted(job) if key != "plan_sha256"},
        "sequence_manifest_sha256": hashlib.sha256(
            sequence_manifest.read_bytes()
        ).hexdigest(),
        "sequence_payload_sha256": sequence["payload_sha256"],
        "reference_sha256": None,
        "unitree_reference_contract_sha256": None,
        "environment": {
            "python_sha256": hashlib.sha256(python.read_bytes()).hexdigest(),
            "conda_history_sha256": hashlib.sha256(history.read_bytes()).hexdigest(),
        },
        "command": command,
        "command_sha256": stage2_module._canonical_sha256(command),
        "expected_output_path": str(output.resolve()),
        "work_directory": str((attempt_root / "work").resolve()),
    }
    execution["environment_provenance_sha256"] = stage2_module._canonical_sha256(
        execution["environment"]
    )
    execution["execution_contract_sha256"] = stage2_module._canonical_sha256(
        execution
    )
    execution_path.parent.mkdir(parents=True)
    execution_path.write_text(json.dumps(execution))
    motion = CanonicalG1(
        qpos=np.column_stack(
            [np.zeros((3, 3)), np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)), np.zeros((3, 29))]
        ),
        fps=30.0,
        source_frame_idx=np.arange(3),
        valid=np.ones(3, dtype=bool),
        per_frame_solve_time_s=np.zeros(3),
        metadata={
            "method": "dense",
            "completion_status": "succeeded",
            "stage2_method_id": "dense",
            "stage2_sequence_id": "sequence-00",
            "experiment_stratum": "controlled_common_per_sequence_scale",
            "stage2_plan_sha256": job["plan_sha256"],
            "stage2_design_id": execution["design_id"],
            "stage2_config_sha256": execution["config_sha256"],
            "stage2_repository_contract_sha256": execution[
                "repository_contract_sha256"
            ],
            "stage2_hardware_contract_sha256": execution[
                "hardware_contract_sha256"
            ],
            "stage2_job_contract_sha256": job["job_contract_sha256"],
            "stage2_execution_contract_sha256": execution[
                "execution_contract_sha256"
            ],
            "stage2_method_contract_sha256": job["method_contract_sha256"],
            "stage2_policy_sha256": job["policy_sha256"],
            "stage2_sequence_manifest_sha256": hashlib.sha256(
                sequence_manifest.read_bytes()
            ).hexdigest(),
            "stage2_sequence_payload_sha256": sequence["payload_sha256"],
            "stage2_source_sha256": sequence["source_sha256"],
            "stage2_canonical_source_sha256": sequence["canonical_sha256"],
            "stage2_reference_sha256": None,
            "stage2_unitree_reference_contract_sha256": None,
            "stage2_environment_provenance_sha256": execution[
                "environment_provenance_sha256"
            ],
            "stage2_exact_upstream_config_asset_policy_contract": {
                "method": job["method"],
                "stratum": job["stratum"],
                "registered_revision": execution["method_contract"][
                    "registered_revision"
                ],
                "method_contract_sha256": job["method_contract_sha256"],
                "policy_sha256": job["policy_sha256"],
                "timing_evidence_sha256": execution["method_contract"][
                    "timing_evidence_sha256"
                ],
                "environment_provenance_sha256": execution["method_contract"][
                    "environment_provenance_sha256"
                ],
            },
            "stage2_authoritative_scale_anchor": {
                "calibration_evidence_sha256": sequence[
                    "calibration_evidence_sha256"
                ],
                **sequence["common_scale"],
                "policy": "controlled_common_per_sequence_scale",
                "replaces_stage1_pilot_scale_anchor_metadata": True,
            },
            "stage2_pre_solver_target_contract": {
                "target_labels": ["root"],
                "target_tensor_shape": [3, 4],
                "target_tensor_sha256": "9" * 64,
                "position_tensor_sha256": "8" * 64,
                "root_yaw_tensor_sha256": "7" * 64,
                "observed_boundary": "immediately_before_controlled_mink_frame_tasks",
            },
        },
    )
    motion.save(output, source_frame_count=3)
    stdout = attempt_root / "logs/stdout.log"
    stderr = attempt_root / "logs/stderr.log"
    stdout.parent.mkdir(parents=True)
    stdout.write_text("")
    stderr.write_text("")
    attempt = {
        "attempt": 1,
        "exit_code": 0,
        "wall_time_s": 0.0,
        "command": command,
        "command_sha256": stage2_module._canonical_sha256(command),
        "thread_environment": {},
        "thread_environment_sha256": stage2_module._canonical_sha256({}),
        "python": str(python),
        "python_sha256": hashlib.sha256(python.read_bytes()).hexdigest(),
        "conda_history": str(history),
        "conda_history_sha256": hashlib.sha256(history.read_bytes()).hexdigest(),
        "execution_contract_path": str(execution_path),
        "execution_contract_sha256": execution["execution_contract_sha256"],
        "expected_output_path": str(output.resolve()),
        "stdout_log": str(stdout),
        "stderr_log": str(stderr),
        "stdout_log_sha256": hashlib.sha256(stdout.read_bytes()).hexdigest(),
        "stderr_log_sha256": hashlib.sha256(stderr.read_bytes()).hexdigest(),
    }
    cleanup_payload = {
        "role": "rebuildable_native_intermediates_deleted_after_hash_capture",
        "files": [],
        "files_sha256": stage2_module._canonical_sha256([]),
        "total_bytes_before_deletion": 0,
        "deleted": True,
    }
    attempt["rebuildable_work_cleanup"] = {
        **cleanup_payload,
        "receipt_sha256": stage2_module._canonical_sha256(cleanup_payload),
    }
    from retargeting_comparison.native_target_capture import tensor_sha256

    manifest = run_root / "job_manifests/dense__sequence-00.json"
    stage2_module._write_job_manifest(
        manifest,
        {
            "schema_version": 2,
            **job,
            "status": "succeeded",
            "repository_contract_sha256": execution[
                "repository_contract_sha256"
            ],
            "hardware_contract_sha256": execution["hardware_contract_sha256"],
            "execution_contract_sha256": execution[
                "execution_contract_sha256"
            ],
            "sequence_manifest": str(sequence_manifest),
            "sequence_manifest_sha256": hashlib.sha256(
                sequence_manifest.read_bytes()
            ).hexdigest(),
            "sequence_payload_sha256": sequence["payload_sha256"],
            "output_path": str(output),
            "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "output_bytes": output.stat().st_size,
            "output_qpos_sha256": tensor_sha256(motion.qpos),
            "completion_ratio": 1.0,
            "cumulative_accounted_attempt_wall_s": 0.0,
            "attempts": [attempt],
        },
    )

    assert job_is_complete(manifest, job) is True
    execution_bytes = execution_path.read_bytes()
    tampered_execution = json.loads(execution_path.read_text())
    tampered_execution["policy_sha256"] = "0" * 64
    execution_path.write_text(json.dumps(tampered_execution))
    assert job_is_complete(manifest, job) is False
    execution_path.write_bytes(execution_bytes)
    sequence_bytes = sequence_manifest.read_bytes()
    tampered_sequence = json.loads(sequence_manifest.read_text())
    tampered_sequence["source_sha256"] = "0" * 64
    sequence_manifest.write_text(json.dumps(tampered_sequence))
    assert job_is_complete(manifest, job) is False
    sequence_manifest.write_bytes(sequence_bytes)
    assert job_is_complete(manifest, {**job, "policy_sha256": "0" * 64}) is False
    output.write_bytes(b"corrupt")
    assert job_is_complete(manifest, job) is False


def test_discovery_hashes_inventory_and_matches_reference_basename(
    tmp_path: Path,
) -> None:
    source = tmp_path / "data/lafan1/sub"
    reference = tmp_path / "reference"
    source.mkdir(parents=True)
    reference.mkdir()
    bvh = source / "walk_subject1.bvh"
    bvh.write_text("HIERARCHY\nMOTION\nFrames: 3\nFrame Time: 0.0333333333\n")
    csv = reference / "walk_subject1.csv"
    csv.write_text(",".join(["0"] * 36) + "\n" + ",".join(["0"] * 36) + "\n")
    config = {
        "dataset": {
            "root": "data/lafan1",
            "glob": "**/*.bvh",
            "expected_files": 1,
            "expected_frames": 3,
        },
        "reference_corpus": {"root": "reference", "glob": "*.csv"},
    }

    sequences = discover_lafan_sequences(tmp_path, config)

    assert len(sequences) == 1
    assert sequences[0].sequence_id == "sub-walk-subject1"
    assert sequences[0].frames == 3
    assert sequences[0].reference_relative_path == "walk_subject1.csv"
    assert sequences[0].reference_frames == 2


def test_reference_inventory_rejects_nonfinite_and_frame_mismatch(
    tmp_path: Path,
) -> None:
    source = tmp_path / "data/lafan1"
    reference = tmp_path / "reference"
    source.mkdir(parents=True)
    reference.mkdir()
    (source / "walk_subject1.bvh").write_text(
        "HIERARCHY\nMOTION\nFrames: 2\nFrame Time: 0.0333333333\n"
    )
    csv = reference / "walk_subject1.csv"
    csv.write_text(",".join(["0"] * 35 + ["nan"]) + "\n")
    config = {
        "dataset": {
            "root": "data/lafan1",
            "glob": "**/*.bvh",
            "expected_files": 1,
            "expected_frames": 2,
        },
        "reference_corpus": {
            "root": "reference",
            "glob": "*.csv",
            "expected_files": 1,
            "expected_intersection_files": 1,
            "expected_columns": 36,
            "expected_actor_ids": ["subject1"],
            "require_finite_numeric_values": True,
            "require_exact_source_frame_count": True,
        },
    }
    with pytest.raises(Stage2Error, match="NaN/Inf"):
        discover_lafan_sequences(tmp_path, config)
    csv.write_text(",".join(["0"] * 36) + "\n")
    with pytest.raises(Stage2Error, match="frame mismatch"):
        discover_lafan_sequences(tmp_path, config)


def test_cost_probe_rejects_payload_and_bound_input_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bound = tmp_path / "input.bin"
    bound.write_bytes(b"frozen")
    environment_python = tmp_path / "env/bin/python"
    environment_history = tmp_path / "env/conda-meta/history"
    environment_python.parent.mkdir(parents=True)
    environment_history.parent.mkdir(parents=True)
    environment_python.write_bytes(b"test-python")
    environment_history.write_bytes(b"test-history")
    monkeypatch.setenv("RTCMP_TEST_PYTHON", str(environment_python))

    frames = 20
    rotations = np.zeros((frames, 1, 4), dtype=np.float64)
    rotations[..., 0] = 1.0
    canonical_source = tmp_path / "pilot-canonical.npz"
    CanonicalHuman(
        joint_names=np.asarray(["Hips"]),
        parent_indices=np.asarray([-1]),
        local_rotations=rotations,
        world_rotations=rotations.copy(),
        world_positions=np.zeros((frames, 1, 3), dtype=np.float64),
        root_translation=np.zeros((frames, 3), dtype=np.float64),
        fps=2.0,
        timestamps=np.arange(frames, dtype=np.float64) / 2.0,
        foot_contact_labels=np.zeros((frames, 2), dtype=bool),
        source_sha256="1" * 64,
    ).save(canonical_source)
    formal_paths = {
        method: tmp_path / f"{method}.npz"
        for method in stage2_module.STAGE1_FORMAL_RETARGETERS
    }
    for output in formal_paths.values():
        output.write_bytes(b"x" * 600)
    contention_artifact_root = tmp_path / "runs/stage2_cost_probe/attempt-001"
    contention_artifact_root.mkdir(parents=True)

    def receipt(path: Path) -> dict[str, object]:
        return {
            "path": path.relative_to(tmp_path).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }

    def contention_row(
        workers: int, process_makespan_s: float, steady_worker_s: float
    ) -> dict[str, object]:
        process_receipts = []
        for worker in range(workers):
            worker_root = (
                contention_artifact_root
                / f"level-{workers}/rep-0/worker-{worker}"
            )
            worker_root.mkdir(parents=True)
            stdout = worker_root / "stdout.log"
            stderr = worker_root / "stderr.log"
            stdout.write_text("")
            stderr.write_text("")
            (worker_root / "canonical_g1.npz").write_bytes(b"canonical-output")
            command = [
                "taskset",
                "--cpu-list",
                str(worker),
                str(environment_python),
                "-m",
                "retargeting_comparison.method_worker",
                "--method",
                "omniretarget",
                "--repo-root",
                str(tmp_path),
                "--source",
                str(canonical_source),
                "--output",
                str(worker_root / "canonical_g1.npz"),
                "--work-dir",
                str(worker_root / "work"),
                "--timing-json",
                str(worker_root / "timing.json"),
                "--warmup-runs",
                "0",
                "--measured-runs",
                "1",
                "--max-frames",
                "20",
            ]
            process_receipts.append({
                "exit_code": 0,
                "process_wall_s": process_makespan_s,
                "command": command,
                "command_sha256": stage2_module._canonical_sha256(command),
                "stdout_log": receipt(stdout),
                "stderr_log": receipt(stderr),
            })
        timing_receipts = []
        for worker in range(workers):
            worker_root = (
                contention_artifact_root
                / f"level-{workers}/rep-0/worker-{worker}"
            )
            artifact = worker_root / "canonical_g1.measured_000.npz"
            artifact.write_bytes(f"artifact-{workers}-{worker}".encode())
            payload = {
                "repetitions": [
                    {
                        "index": 0,
                        "role": "measured",
                        "frame_count": 20,
                        "steady_end_to_end_total_s": steady_worker_s,
                        "timing_boundary": (
                            "canonical_source_file_to_canonical_g1_in_memory"
                        ),
                        "cpu_affinity": [worker],
                        "thread_limit": 1,
                        "timing_artifact_path": str(artifact),
                        "timing_artifact_sha256": hashlib.sha256(
                            artifact.read_bytes()
                        ).hexdigest(),
                    }
                ]
            }
            timing_path = worker_root / "timing.json"
            timing_path.write_text(json.dumps(payload))
            timing_receipts.append(
                {
                    "timing_json": receipt(timing_path),
                    "payload_sha256": stage2_module._canonical_sha256(payload),
                    "payload": payload,
                }
            )
        process_slowdown = process_makespan_s
        steady_slowdown = steady_worker_s
        combined = max(process_slowdown, steady_slowdown)
        return {
            "workers": workers,
            "distinct_cpu_ids": list(range(workers)),
            "level_wall_s_raw": [process_makespan_s],
            "level_wall_s_median": process_makespan_s,
            "process_wall_s_raw": [
                [process_makespan_s for _ in range(workers)]
            ],
            "process_makespan_s_raw": [process_makespan_s],
            "process_makespan_s_median": process_makespan_s,
            "max_individual_process_wall_s_raw": [process_makespan_s],
            "steady_in_memory_s_raw": [
                [steady_worker_s for _ in range(workers)]
            ],
            "steady_worker_s_median": steady_worker_s,
            "process_receipts": [process_receipts],
            "timing_receipts": [timing_receipts],
            "process_makespan_slowdown_vs_one_process": process_slowdown,
            "steady_per_worker_slowdown_vs_one_process": steady_slowdown,
            "combined_slowdown_vs_one_process": combined,
            "per_worker_slowdown_vs_one_process": combined,
            "observed_aggregate_speedup": workers / combined,
        }

    probe = {
        "schema_version": 1,
        "repetitions": 1,
        "statistic": "median",
        "conservative_multiplier": 1.0,
        "pilot_frames": 20,
        "pilot_duration_s": 10.0,
        "pilot_canonical_source": receipt(canonical_source),
        "source_preparation_wall_s_raw": [1.0],
        "evaluator_wall_s_raw": [3.0],
        "canonical_source_bytes_raw": [200],
        "accepted_output_bytes_raw": {
            method: 600 for method in stage2_module.STAGE1_FORMAL_RETARGETERS
        },
        "accepted_formal_outputs": {
            method: receipt(path) for method, path in formal_paths.items()
        },
        "measured_source_preparation_rtf": 0.1,
        "conservative_source_preparation_rtf": 0.2,
        "measured_evaluator_rtf_per_method": 0.3,
        "conservative_evaluator_rtf_per_method": 0.4,
        "measured_canonical_source_bytes_per_frame": 10.0,
        "conservative_canonical_source_bytes_per_frame": 20.0,
        "measured_output_bytes_per_frame_max": 30.0,
        "conservative_output_bytes_per_frame": 40.0,
        "parallel_contention_probe": {
            "method": "omniretarget",
            "environment": "test",
            "levels": [1, 2, 6],
            "timing_boundary": "canonical_source_file_to_canonical_g1_in_memory",
            "command_template": "worker --method omniretarget",
            "repetitions_per_level": 1,
            "source_frames": 20,
            "cpu_assignment": "distinct_allowed_cpus",
            "results": [
                contention_row(1, 1.0, 1.0),
                contention_row(2, 1.2, 1.1),
                contention_row(6, 1.5, 1.4),
            ],
            "apply_to_all_methods_with_workers_gt_one": True,
            "observed_max_combined_slowdown": 1.5,
            "observed_max_per_worker_slowdown": 1.5,
            "conservative_slowdown_multiplier": 1.1,
            "conservative_parallel_contention_factor": 1.65,
            "python": receipt(environment_python),
            "conda_history": receipt(environment_history),
        },
        "conservative_parallel_contention_factor": 1.65,
        "measurement_hardware": {"production_cpu_ids": list(range(6))},
        "bound_inputs": [
            receipt(bound),
            receipt(canonical_source),
            *(receipt(path) for path in formal_paths.values()),
        ],
    }
    probe["bound_inputs_sha256"] = stage2_module._canonical_sha256(
        probe["bound_inputs"]
    )
    probe["contention_artifact_root"] = contention_artifact_root.relative_to(
        tmp_path
    ).as_posix()
    probe["contention_artifact_inventory"] = stage2_module._directory_receipts(
        tmp_path, contention_artifact_root
    )
    probe["contention_artifact_inventory_sha256"] = (
        stage2_module._canonical_sha256(probe["contention_artifact_inventory"])
    )
    probe["payload_sha256"] = stage2_module._canonical_sha256(probe)
    path = tmp_path / "probe.json"
    path.write_text(json.dumps(probe))
    config = {
        "preflight_cost_probe": {
            "required": True,
            "manifest": "probe.json",
            "repetitions": 1,
            "statistic": "median",
            "conservative_multiplier": 1.0,
            "contention_probe": {
                "method": "omniretarget",
                "environment": "test",
                "levels": [1, 2, 6],
                "repetitions_per_level": 1,
                "source_frames": 20,
                "cpu_assignment": "distinct_allowed_cpus",
                "conservative_slowdown_multiplier": 1.1,
                "apply_to_all_methods_with_workers_gt_one": True,
            },
        },
        "budget_allowances": {
            "source_preparation_rtf": 0.2,
            "evaluator_rtf_per_method": 0.4,
        },
        "storage_projection": {"canonical_source_bytes_per_frame": 20.0},
        "methods": {"test": {"retained_bytes_per_frame": 40.0}},
        "production_design": {"required": False},
    }
    assert stage2_module._validate_cost_probe(tmp_path, config)["manifest"][
        "sha256"
    ] == hashlib.sha256(path.read_bytes()).hexdigest()
    alternate_source = tmp_path / "same-frame-count-source.npz"
    alternate_source.write_bytes(canonical_source.read_bytes())
    source_tampered = json.loads(json.dumps(probe))
    process_receipt = source_tampered["parallel_contention_probe"]["results"][0][
        "process_receipts"
    ][0][0]
    source_flag = process_receipt["command"].index("--source")
    process_receipt["command"][source_flag + 1] = str(alternate_source)
    process_receipt["command_sha256"] = stage2_module._canonical_sha256(
        process_receipt["command"]
    )
    source_tampered.pop("payload_sha256")
    source_tampered["payload_sha256"] = stage2_module._canonical_sha256(
        source_tampered
    )
    path.write_text(json.dumps(source_tampered))
    with pytest.raises(Stage2Error, match="process command is not exact"):
        stage2_module._validate_cost_probe(tmp_path, config)
    duration_tampered = json.loads(json.dumps(probe))
    duration_tampered["pilot_duration_s"] = 11.0
    duration_tampered.pop("payload_sha256")
    duration_tampered["payload_sha256"] = stage2_module._canonical_sha256(
        duration_tampered
    )
    path.write_text(json.dumps(duration_tampered))
    with pytest.raises(Stage2Error, match="raw measurement protocol"):
        stage2_module._validate_cost_probe(tmp_path, config)
    output_size_tampered = json.loads(json.dumps(probe))
    output_size_tampered["accepted_output_bytes_raw"]["dense"] = 601
    output_size_tampered.pop("payload_sha256")
    output_size_tampered["payload_sha256"] = stage2_module._canonical_sha256(
        output_size_tampered
    )
    path.write_text(json.dumps(output_size_tampered))
    with pytest.raises(Stage2Error, match="raw measurement protocol"):
        stage2_module._validate_cost_probe(tmp_path, config)
    cpu_tampered = json.loads(json.dumps(probe))
    cpu_tampered["parallel_contention_probe"]["results"][1][
        "distinct_cpu_ids"
    ] = [0, 99]
    cpu_tampered.pop("payload_sha256")
    cpu_tampered["payload_sha256"] = stage2_module._canonical_sha256(cpu_tampered)
    path.write_text(json.dumps(cpu_tampered))
    with pytest.raises(Stage2Error, match="contention probe row"):
        stage2_module._validate_cost_probe(tmp_path, config)
    affinity_tampered = json.loads(json.dumps(probe))
    timing_receipt = affinity_tampered["parallel_contention_probe"]["results"][0][
        "timing_receipts"
    ][0][0]
    timing_file = tmp_path / timing_receipt["timing_json"]["path"]
    original_timing_bytes = timing_file.read_bytes()
    timing_receipt["payload"]["repetitions"][0]["cpu_affinity"] = [99]
    timing_file.write_text(json.dumps(timing_receipt["payload"]))
    timing_receipt["timing_json"] = receipt(timing_file)
    timing_receipt["payload_sha256"] = stage2_module._canonical_sha256(
        timing_receipt["payload"]
    )
    affinity_tampered["contention_artifact_inventory"] = (
        stage2_module._directory_receipts(tmp_path, contention_artifact_root)
    )
    affinity_tampered["contention_artifact_inventory_sha256"] = (
        stage2_module._canonical_sha256(
            affinity_tampered["contention_artifact_inventory"]
        )
    )
    affinity_tampered.pop("payload_sha256")
    affinity_tampered["payload_sha256"] = stage2_module._canonical_sha256(
        affinity_tampered
    )
    path.write_text(json.dumps(affinity_tampered))
    with pytest.raises(Stage2Error, match="raw timing receipt"):
        stage2_module._validate_cost_probe(tmp_path, config)
    timing_file.write_bytes(original_timing_bytes)
    environment_tampered = json.loads(json.dumps(probe))
    environment_tampered["parallel_contention_probe"]["python"] = receipt(bound)
    environment_tampered.pop("payload_sha256")
    environment_tampered["payload_sha256"] = stage2_module._canonical_sha256(
        environment_tampered
    )
    path.write_text(json.dumps(environment_tampered))
    with pytest.raises(Stage2Error, match="contention environment"):
        stage2_module._validate_cost_probe(tmp_path, config)
    timing_tampered = json.loads(json.dumps(probe))
    timing_tampered["parallel_contention_probe"]["results"][1][
        "timing_receipts"
    ][0][0]["payload"]["repetitions"][0]["steady_end_to_end_total_s"] = 9.0
    timing_tampered.pop("payload_sha256")
    timing_tampered["payload_sha256"] = stage2_module._canonical_sha256(
        timing_tampered
    )
    path.write_text(json.dumps(timing_tampered))
    with pytest.raises(Stage2Error, match="raw timing receipt"):
        stage2_module._validate_cost_probe(tmp_path, config)
    raw_tampered = json.loads(json.dumps(probe))
    raw_tampered["source_preparation_wall_s_raw"] = [5.0]
    raw_tampered.pop("payload_sha256")
    raw_tampered["payload_sha256"] = stage2_module._canonical_sha256(
        raw_tampered
    )
    path.write_text(json.dumps(raw_tampered))
    with pytest.raises(Stage2Error, match="not derived from raw data"):
        stage2_module._validate_cost_probe(tmp_path, config)
    path.write_text(json.dumps(probe))
    bound.write_bytes(b"tampered")
    with pytest.raises(Stage2Error, match="input changed"):
        stage2_module._validate_cost_probe(tmp_path, config)


def _write_go_manifest(root: Path, decision: str = "GO") -> None:
    path = root / "manifests/stage1_validation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"decision": decision, "checks": {"complete": True}}))


def _write_test_config(root: Path, *, authorized: bool = True) -> Path:
    repository = Path(__file__).resolve().parents[1]
    config = load_yaml(repository / "configs/stage2.yaml")
    config["authorization"]["explicitly_authorized"] = authorized
    config["dataset"].update(
        {"root": "data/lafan1", "expected_files": 1, "expected_frames": 300}
    )
    config["canonicalization"]["output_root"] = "runs/stage2"
    config["reference_corpus"]["root"] = "reference"
    config["reference_corpus"].update(
        {
            "expected_files": 0,
            "expected_intersection_files": 0,
            "expected_actor_ids": [],
        }
    )
    config["stage1_gate"]["require_formal_outputs"] = False
    config["stage1_gate"]["require_strict_source_identity"] = False
    config["stage1_gate"]["formal_timing_campaign"]["required"] = False
    config["stage1_gate"]["require_bound_validation"] = False
    config["preflight_cost_probe"]["required"] = False
    config["reference_corpus"]["contract"]["required"] = False
    config["production_design"]["required"] = False
    config["budget_simplification"][
        "exclude_registered_method_for_production"
    ] = False
    config["repository_gate"]["require_clean_worktree"] = False
    config["repository_gate"]["require_external_clean"] = False
    config["repository_gate"]["external_worktrees"] = {}
    config["repository_gate"]["required_assets"] = []
    config["publication"]["tracked_output_root"] = "stage2_results"
    config["scheduling"]["physical_cores"] = 1
    for method in config["methods"].values():
        method["workers"] = 1
    # Conditional methods have no accepted output in this isolated fixture.
    path = root / "configs/stage2.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def test_plan_requires_authorization_and_stage1_go(tmp_path: Path) -> None:
    _write_go_manifest(tmp_path)
    config = _write_test_config(tmp_path, authorized=False)
    with pytest.raises(Stage2Error, match="explicit user authorization"):
        build_stage2_plan(
            tmp_path,
            config,
            sequences=[_sequence(0)],
            output_path=tmp_path / "plan.json",
        )
    config = _write_test_config(tmp_path, authorized=True)
    _write_go_manifest(tmp_path, decision="NO-GO")
    with pytest.raises(Stage2Error, match="Stage 1 is not GO"):
        build_stage2_plan(
            tmp_path,
            config,
            sequences=[_sequence(0)],
            output_path=tmp_path / "plan.json",
        )


def test_bound_stage1_gate_rejects_unverified_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import retargeting_comparison.stage1_publication as publication

    config = {
        "stage1_gate": {
            "validation_manifest": "manifests/stage1_validation.json",
            "accepted_decisions": ["GO"],
            "require_all_checks": True,
            "require_bound_validation": True,
        }
    }
    monkeypatch.setattr(
        publication,
        "resolve_bound_stage1_validation",
        lambda _root: {
            "decision": "PENDING",
            "binding_status": "validation_unbound_or_invalid",
            "reason": "forged verdict",
            "validation": None,
        },
    )
    with pytest.raises(Stage2Error, match="schema-v5 verdict"):
        stage2_module._load_stage1_gate(tmp_path, config)
    verified = {
        "decision": "GO",
        "binding_status": "verified",
        "reason": "bound",
        "validation": {"checks": {"all": True}},
    }
    monkeypatch.setattr(
        publication, "resolve_bound_stage1_validation", lambda _root: verified
    )
    value, _ = stage2_module._load_stage1_gate(tmp_path, config)
    assert value == verified


def test_plan_is_resumable_and_launch_requires_exact_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_go_manifest(tmp_path)
    config_path = _write_test_config(tmp_path)
    source = tmp_path / "data/lafan1/sequence_00.bvh"
    source.parent.mkdir(parents=True)
    source.write_text("HIERARCHY\nMOTION\nFrames: 300\nFrame Time: 0.0333333333\n")
    discovered_fps = 1.0 / 0.0333333333
    sequence = replace(
        _sequence(0),
        relative_path="sequence_00.bvh",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_size_bytes=source.stat().st_size,
        fps=discovered_fps,
        duration_s=300 / discovered_fps,
    )
    plan_path = tmp_path / "runs/stage2/test-plan.json"

    first = build_stage2_plan(
        tmp_path,
        config_path,
        sequences=[sequence],
        output_path=plan_path,
    )
    second = build_stage2_plan(
        tmp_path,
        config_path,
        sequences=[sequence],
        output_path=plan_path,
    )

    assert first == second
    assert verify_plan_hash(first) is True
    tampered = {**first, "design_id": "tampered"}
    assert verify_plan_hash(tampered) is False
    with pytest.raises(Stage2Error, match="not acknowledged exactly"):
        validate_plan_for_launch(
            tmp_path,
            plan_path,
            config_path,
            expected_plan_sha256="0" * 64,
        )
    loaded, _, _ = validate_plan_for_launch(
        tmp_path,
        plan_path,
        config_path,
        expected_plan_sha256=first["plan_sha256"],
    )
    assert loaded["gates"]["launchable"] is True
    assert first["hardware"]["configured_physical_cores"] == 1
    original_hardware = dict(first["hardware"])
    monkeypatch.setattr(
        stage2_module,
        "_hardware_identity",
        lambda _config: {**original_hardware, "hostname": "different-host"},
    )
    with pytest.raises(Stage2Error, match="hardware changed"):
        validate_plan_for_launch(
            tmp_path,
            plan_path,
            config_path,
            expected_plan_sha256=first["plan_sha256"],
        )
    monkeypatch.setattr(
        stage2_module, "_hardware_identity", lambda _config: original_hardware
    )
    preview = execute_stage2(
        tmp_path,
        plan_path,
        config_path=config_path,
        expected_plan_sha256=first["plan_sha256"],
    )
    assert preview["status"] == "preview"
    assert preview["launched"] is False
    assert not (tmp_path / "runs/stage2" / first["design_id"] / "sources").exists()


def test_method_budget_exclusion_keeps_full_dataset_label(tmp_path: Path) -> None:
    _write_go_manifest(tmp_path)
    config_path = _write_test_config(tmp_path)
    plan = build_stage2_plan(
        tmp_path,
        config_path,
        sequences=[_sequence(0)],
        output_path=tmp_path / "plan.json",
        readiness_overrides={"protomotions-v3": True},
        timing_overrides={"protomotions-v3": (20_000.0, 0.0)},
    )

    assert plan["dataset_kind"] == "full_lafan1"
    assert len(plan["selected_sequences"]) == 1
    assert plan["fallback"]["dataset_simplification"] is None
    assert (
        plan["fallback"]["method_simplification"]["dataset_reduction_also_required"]
        is False
    )
    assert any(
        value["method"] == "protomotions-v3"
        and value["status"] == "excluded_from_stage2_for_budget"
        for value in plan["method_exclusions"]
    )


def test_empty_or_early_stopped_execution_cannot_report_complete(
    tmp_path: Path,
) -> None:
    config = {
        "budget": {
            "wall_time_hours": 1.0,
            "retained_storage_gb": 1.0,
            "stop_before_limit_fraction": 0.98,
        }
    }
    plan = {
        "design_id": "empty-test",
        "plan_sha256": "a" * 64,
        "repository": {},
        "selected_sequences": [],
        "projection": {"jobs": []},
    }
    summary = _execute_stage2_locked(tmp_path, tmp_path / "run", plan, config)
    assert summary["status"] == "incomplete"
    assert summary["planned_job_count"] == 0
    assert summary["verified_succeeded_job_count"] == 0


def test_resume_uses_cumulative_wall_budget_and_closes_crash_record(
    tmp_path: Path,
) -> None:
    attempts = tmp_path / "execution_attempts"
    attempts.mkdir(parents=True)
    (attempts / "attempt-001.json").write_text(
        json.dumps(
            {
                "status": "finished_incomplete",
                "active_wall_s": 12.5,
                "started_at_utc": "2026-07-22T00:00:00+00:00",
            }
        )
    )
    (attempts / "attempt-002.json").write_text(
        json.dumps(
            {
                "status": "running",
                "started_at_utc": "2026-07-22T00:00:00+00:00",
            }
        )
    )

    consumed = _account_prior_execution_wall(tmp_path)
    assert consumed > 12.5
    closed = json.loads((attempts / "attempt-002.json").read_text())
    assert closed["status"] == "abandoned_conservatively_accounted"
    assert closed["active_wall_s"] > 0.0


def test_sequence_manifest_cannot_relax_frozen_reference_fps_tolerance(
    tmp_path: Path,
) -> None:
    sequence = replace(
        _sequence(0, reference=True),
        relative_path="subject1_walk.bvh",
        reference_relative_path="subject1_walk.csv",
    )
    config = {
        "dataset": {
            "root": "data/lafan1",
            "source_position_scale": 1.0,
        },
        "reference_corpus": {
            "root": "reference",
            "revision": "unitree-test",
            "native_fps_relative_tolerance": 2.0e-5,
        },
        "canonicalization": {
            "contract": "test-contract",
            "common_scale": {"definition": "test-scale"},
        },
    }
    plan = {
        "plan_sha256": "a" * 64,
        "design_id": "test-design",
        "selected_inventory_sha256": "b" * 64,
        "unitree_reference_contract": {"contract_sha256": "c" * 64},
        "calibration_contract": {"contract_sha256": "d" * 64},
    }
    preparation_contract = {
        "plan_sha256": plan["plan_sha256"],
        "design_id": plan["design_id"],
        "selected_inventory_sha256": plan["selected_inventory_sha256"],
        "source": sequence.as_dict(),
        "source_position_scale": 1.0,
        "canonicalization": dict(config["canonicalization"]),
        "reference_revision": "unitree-test",
        "unitree_reference_contract_sha256": "c" * 64,
        "calibration_contract_sha256": "d" * 64,
    }
    value = {
        "schema_version": 2,
        "plan_sha256": plan["plan_sha256"],
        "design_id": plan["design_id"],
        "selected_inventory_sha256": plan["selected_inventory_sha256"],
        "sequence_id": sequence.sequence_id,
        "actor_id": stage2_module._actor_id(sequence.relative_path),
        "source_path": f"data/lafan1/{sequence.relative_path}",
        "source_sha256": sequence.source_sha256,
        "source_frames": sequence.frames,
        "fps": sequence.fps,
        "canonical_path": "unused.npz",
        "canonical_contract": "test-contract",
        "preparation_contract_sha256": stage2_module._canonical_sha256(
            preparation_contract
        ),
        "reference_relative_path": sequence.reference_relative_path,
        "reference_path": f"reference/{sequence.reference_relative_path}",
        "reference_sha256": sequence.reference_sha256,
        "reference_frames": sequence.reference_frames,
        # Self-consistent payload tampering must not be able to weaken this.
        "reference_native_fps_relative_tolerance": 1.0,
    }
    value["payload_sha256"] = stage2_module._canonical_sha256(value)

    with pytest.raises(Stage2Error, match="not bound"):
        stage2_module._validate_sequence_calibration_manifest(
            tmp_path, value, sequence, config, plan
        )


def test_formal_stage1_gate_requires_all_six_exact_outputs(tmp_path: Path) -> None:
    methods = (
        "sparse-neutral",
        "dense",
        "gmr",
        "omniretarget",
        "protomotions-v2.3",
        "protomotions-v3",
    )
    formal = {}
    for method in methods:
        path = tmp_path / "formal" / method / "canonical_g1.npz"
        motion = CanonicalG1(
            qpos=np.column_stack(
                [
                    np.zeros((3, 3)),
                    np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)),
                    np.zeros((3, 29)),
                ]
            ),
            fps=30.0,
            source_frame_idx=np.arange(3),
            valid=np.ones(3, dtype=bool),
            per_frame_solve_time_s=np.zeros(3),
            metadata={"method": method, "completion_status": "succeeded"},
        )
        motion.save(path, source_frame_count=3)
        formal[method] = {
            "path": str(path.relative_to(tmp_path)),
            "metadata_methods": [method],
        }
    config = {
        "stage1_gate": {
            "require_formal_outputs": True,
            "expected_frames": 3,
            "expected_fps": 30.0,
            "formal_outputs": formal,
        },
        "scheduling": {"phase_order": [*methods, "unitree-reference"]},
    }
    evidence = _validate_stage1_formal_outputs(tmp_path, config)
    assert set(evidence) == set(methods)
    assert all(len(value["sha256"]) == 64 for value in evidence.values())
    (tmp_path / formal["protomotions-v3"]["path"]).unlink()
    with pytest.raises(Stage2Error, match="formal output is missing"):
        _validate_stage1_formal_outputs(tmp_path, config)


def test_strict_formal_gate_uses_run_manifest_output_and_confines_attempt_path(
    tmp_path: Path,
) -> None:
    methods = (
        "sparse-neutral",
        "dense",
        "gmr",
        "omniretarget",
        "protomotions-v2.3",
        "protomotions-v3",
    )
    source_sha = "7" * 64
    frames = 3
    identity = np.tile([1.0, 0.0, 0.0, 0.0], (frames, 1, 1))
    human = CanonicalHuman(
        joint_names=np.asarray(["Hips"]),
        parent_indices=np.asarray([-1]),
        local_rotations=identity,
        world_rotations=identity.copy(),
        world_positions=np.zeros((frames, 1, 3)),
        root_translation=np.zeros((frames, 3)),
        fps=30.0,
        timestamps=np.arange(frames) / 30.0,
        foot_contact_labels=np.zeros((frames, 2), dtype=bool),
        source_sha256=source_sha,
    )
    source_path = tmp_path / "source/canonical.npz"
    human.save(source_path)
    source_manifest = tmp_path / "manifests/pilot.yaml"
    source_manifest.parent.mkdir(parents=True)
    source_manifest.write_text(
        yaml.safe_dump(
            {
                "sequence_id": "pilot",
                "canonical_path": str(source_path.relative_to(tmp_path)),
                "cropped_sha256": source_sha,
            }
        )
    )
    formal = {}
    for method in methods:
        revision = f"{method}-revision"
        registered = tmp_path / "runs/pilot" / revision
        output = registered / "attempt_001/canonical_g1.npz"
        motion = CanonicalG1(
            qpos=np.column_stack(
                [
                    np.zeros((frames, 3)),
                    np.tile([1.0, 0.0, 0.0, 0.0], (frames, 1)),
                    np.zeros((frames, 29)),
                ]
            ),
            fps=30.0,
            source_frame_idx=np.arange(frames),
            valid=np.ones(frames, dtype=bool),
            per_frame_solve_time_s=np.ones(frames),
            metadata={
                "method": method,
                "completion_status": "succeeded",
                "canonical_source_sha256": source_sha,
            },
        )
        motion.save(output, source_frame_count=frames)
        manifest_path = tmp_path / "runs/manifests" / f"pilot__{revision}.json"
        RunManifest(
            run_id=f"pilot__{revision}",
            method=method,
            status=RunStatus.SUCCEEDED,
            command=["worker"],
            environment="test",
            repo_commit="a" * 40,
            config_sha256="b" * 64,
            device="cpu",
            started_at="2026-01-01T00:00:00+00:00",
            finished_at="2026-01-01T00:00:01+00:00",
            wall_time_s=1.0,
            exit_code=0,
            output_path=str(output.relative_to(tmp_path)),
            output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
            source_sha256=source_sha,
        ).save(manifest_path)
        formal[method] = {
            "path": str((registered / "canonical_g1.npz").relative_to(tmp_path)),
            "revision": revision,
            "run_manifest": str(manifest_path.relative_to(tmp_path)),
            "run_id": f"pilot__{revision}",
            "registered_directory": str(registered.relative_to(tmp_path)),
            "metadata_methods": [method],
        }
    config = {
        "stage1_gate": {
            "require_formal_outputs": True,
            "require_strict_source_identity": True,
            "expected_frames": frames,
            "canonical_source_manifest": str(source_manifest.relative_to(tmp_path)),
            "formal_outputs": formal,
        },
        "scheduling": {"phase_order": [*methods, "unitree-reference"]},
    }
    evidence = _validate_stage1_formal_outputs(tmp_path, config)
    assert all("attempt_001" in row["path"] for row in evidence.values())

    entry = formal["dense"]
    manifest = RunManifest.load(tmp_path / entry["run_manifest"])
    escaped = tmp_path / "escaped/canonical_g1.npz"
    escaped.parent.mkdir(parents=True)
    escaped.write_bytes((tmp_path / evidence["dense"]["path"]).read_bytes())
    manifest.output_path = str(escaped.relative_to(tmp_path))
    manifest.output_sha256 = hashlib.sha256(escaped.read_bytes()).hexdigest()
    manifest.save(tmp_path / entry["run_manifest"])
    with pytest.raises(Stage2Error, match="revision/path contract"):
        _validate_stage1_formal_outputs(tmp_path, config)


def test_fake_worker_obeys_exact_contract_and_shared_time_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "fake_worker.py"
    script.write_text(
        """import json, pathlib, sys, time
import numpy as np
output = pathlib.Path(sys.argv[1])
mode, method, sequence, stratum, contract_path, sequence_path = sys.argv[2:8]
if mode == 'slow':
    time.sleep(60)
output.parent.mkdir(parents=True, exist_ok=True)
frames = 3
qpos = np.zeros((frames, 36)); qpos[:, 3] = 1.0
contract = json.loads(pathlib.Path(contract_path).read_text())
source = json.loads(pathlib.Path(sequence_path).read_text())
metadata = {'method': method, 'completion_status': 'succeeded',
            'stage2_method_id': method, 'stage2_sequence_id': sequence,
            'experiment_stratum': stratum,
            'stage2_plan_sha256': contract['plan_sha256'],
            'stage2_design_id': contract['design_id'],
            'stage2_config_sha256': contract['config_sha256'],
            'stage2_repository_contract_sha256': contract['repository_contract_sha256'],
            'stage2_hardware_contract_sha256': contract['hardware_contract_sha256'],
            'stage2_job_contract_sha256': contract['job_contract_sha256'],
            'stage2_execution_contract_sha256': contract['execution_contract_sha256'],
            'stage2_method_contract_sha256': contract['method_contract_sha256'],
            'stage2_policy_sha256': contract['policy_sha256'],
            'stage2_sequence_manifest_sha256': contract['sequence_manifest_sha256'],
            'stage2_sequence_payload_sha256': contract['sequence_payload_sha256'],
            'stage2_source_sha256': source['source_sha256'],
            'stage2_canonical_source_sha256': source['canonical_sha256'],
            'stage2_reference_sha256': contract['reference_sha256'],
            'stage2_unitree_reference_contract_sha256': contract['unitree_reference_contract_sha256'],
            'stage2_environment_provenance_sha256': contract['environment_provenance_sha256'],
            'stage2_exact_upstream_config_asset_policy_contract': {
                'method': method, 'stratum': stratum,
                'registered_revision': contract['method_contract']['registered_revision'],
                'method_contract_sha256': contract['method_contract_sha256'],
                'policy_sha256': contract['policy_sha256'],
                'timing_evidence_sha256': contract['method_contract']['timing_evidence_sha256'],
                'environment_provenance_sha256': contract['method_contract']['environment_provenance_sha256']},
            'stage2_authoritative_scale_anchor': {
                'calibration_evidence_sha256': source['calibration_evidence_sha256'],
                'policy': 'controlled_common_per_sequence_scale',
                'replaces_stage1_pilot_scale_anchor_metadata': True,
                **source['common_scale']},
            'stage2_pre_solver_target_contract': source['expected_target_contract']}
np.savez_compressed(output, qpos=qpos, fps=np.asarray(30.0),
    source_frame_idx=np.arange(frames), valid=np.ones(frames, dtype=bool),
    per_frame_solve_time_s=np.zeros(frames),
    metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
"""
    )
    controlled_config = tmp_path / "configs/controlled_mink.yaml"
    controlled_config.parent.mkdir(parents=True)
    controlled_config.write_text(
        (Path(__file__).resolve().parents[1] / "configs/controlled_mink.yaml").read_text()
    )
    joint_names = np.asarray(
        [
            "Hips",
            "Spine2",
            "Head",
            "LeftArm",
            "RightArm",
            "LeftForeArm",
            "RightForeArm",
            "LeftHand",
            "RightHand",
            "LeftUpLeg",
            "RightUpLeg",
            "LeftLeg",
            "RightLeg",
            "LeftFoot",
            "RightFoot",
            "LeftToe",
            "RightToe",
        ]
    )
    frames = 3
    rotations = np.zeros((frames, len(joint_names), 4), dtype=np.float64)
    rotations[..., 0] = 1.0
    positions = np.zeros((frames, len(joint_names), 3), dtype=np.float64)
    positions[..., 2] = np.linspace(0.0, 1.6, len(joint_names))[None, :]
    positions[:, :, 0] += np.arange(frames)[:, None] * 0.01
    name_to_index = {name: index for index, name in enumerate(joint_names)}
    positions[:, name_to_index["LeftUpLeg"], 1] = 0.1
    positions[:, name_to_index["RightUpLeg"], 1] = -0.1
    positions[:, name_to_index["LeftArm"], 1] = 0.25
    positions[:, name_to_index["RightArm"], 1] = -0.25
    human = CanonicalHuman(
        joint_names=joint_names,
        parent_indices=np.asarray([-1, *([0] * (len(joint_names) - 1))]),
        local_rotations=rotations,
        world_rotations=rotations.copy(),
        world_positions=positions,
        root_translation=positions[:, 0],
        fps=30.0,
        timestamps=np.arange(frames) / 30.0,
        foot_contact_labels=np.zeros((frames, 2), dtype=bool),
        source_sha256="1" * 64,
    )
    canonical_path = tmp_path / "canonical_human.npz"
    human.save(canonical_path)
    sequence_manifest = tmp_path / "sequence.json"
    sequence_value = {
        "schema_version": 2,
        "source_frames": 3,
        "fps": 30.0,
        "source_sha256": "1" * 64,
        "canonical_path": canonical_path.name,
        "canonical_sha256": hashlib.sha256(canonical_path.read_bytes()).hexdigest(),
        "calibration_evidence_sha256": "3" * 64,
        "common_scale": {
            "local_body_scale": 1.0,
            "root_displacement_scale": 1.0,
            "root_alignment_translation_m": [0.0, 0.0, 0.0],
        },
    }
    sequence_value["expected_target_contract"] = (
        stage2_module._controlled_pre_solver_target_contract(
            tmp_path, human, sequence_value, "dense"
        )
    )
    sequence_value["payload_sha256"] = stage2_module._canonical_sha256(
        sequence_value
    )
    sequence_manifest.write_text(json.dumps(sequence_value), encoding="utf-8")
    job = {
        "job_id": "dense__sequence-00",
        "method": "dense",
        "sequence_id": "sequence-00",
        "stratum": "controlled_common_per_sequence_scale",
        "estimated_runtime_s": 0.01,
        "worker_index": 0,
        "job_contract_sha256": "4" * 64,
        "method_contract_sha256": "5" * 64,
        "policy_sha256": "6" * 64,
    }
    plan = {
        "plan_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "design_id": "test-design",
        "repository": {},
        "hardware": {"production_cpu_ids": stage2_module._physical_cpu_ids(1)},
        "stage1_timing_evidence": {},
        "unitree_reference_contract": {},
        "method_specs": [
            {
                "method": "dense",
                "stratum": "controlled_common_per_sequence_scale",
                "public_pipeline": False,
                "environment": "test",
                "workers": 1,
                "rtf": 1.0,
                "startup_s": 0.0,
                "retained_bytes_per_frame": 10,
                "retained_bytes_per_sequence": 10,
                "estimate_source": "test",
                "method_contract_sha256": job["method_contract_sha256"],
                "timing_evidence_sha256": "",
                "environment_provenance_sha256": "",
                "policy_sha256": job["policy_sha256"],
                "registered_revision": "test",
            }
        ],
    }
    config = {
        "budget": {"safety_factor": 1.0},
        "storage_projection": {
            "delete_rebuildable_native_intermediates_after_hash_capture": True
        },
        "methods": {
            "dense": {
                "environment": "test",
                "retained_bytes_per_sequence": 10,
                "retained_bytes_per_frame": 10,
            }
        },
    }
    monkeypatch.setattr(
        stage2_module, "_resolve_python", lambda _environment: Path(sys.executable)
    )
    monkeypatch.setattr(stage2_module, "_git_head", lambda _root: "c" * 40)

    mode = {"value": "success"}

    def fake_command(
        _root, _python, value, sequence_path, output, _work, execution_contract
    ):
        return [
            sys.executable,
            str(script),
            str(output),
            mode["value"],
            value["method"],
            value["sequence_id"],
            value["stratum"],
            str(execution_contract),
            str(sequence_path),
        ]

    monkeypatch.setattr(stage2_module, "_worker_command", fake_command)
    success_root = tmp_path / "success"
    guard = _ExecutionGuard(
        success_root,
        deadline_monotonic=time.monotonic() + 10.0,
        storage_stop_bytes=10_000_000,
    )
    result = _run_job(
        tmp_path, success_root, plan, config, job, sequence_manifest, guard
    )
    assert result["status"] == "succeeded"
    assert result["completion_ratio"] == 1.0

    # Model an orchestrator crash after the running manifest was committed but
    # before the attempt exit receipt was persisted. Resume must close and
    # conservatively charge attempt 1 before attempt 2 can be accepted.
    crash_manifest = success_root / "job_manifests/dense__sequence-00.json"
    crashed = stage2_module._load_job_manifest(crash_manifest)
    crashed.pop("payload_sha256")
    crashed["status"] = "running"
    for key in (
        "output_path",
        "output_sha256",
        "output_bytes",
        "output_qpos_sha256",
        "completion_ratio",
        "cumulative_accounted_attempt_wall_s",
    ):
        crashed.pop(key, None)
    first_attempt = crashed["attempts"][0]
    first_attempt["started_at_utc"] = (
        datetime.now(timezone.utc) - timedelta(seconds=2.0)
    ).isoformat()
    for key in (
        "exit_code",
        "wall_time_s",
        "finished_at_utc",
        "budget_stop_reason",
    ):
        first_attempt.pop(key, None)
    original_command = list(first_attempt["command"])
    original_stdout_receipt = first_attempt["stdout_log_sha256"]
    stage2_module._write_job_manifest(crash_manifest, crashed)

    resume_guard = _ExecutionGuard(
        success_root,
        deadline_monotonic=time.monotonic() + 10.0,
        storage_stop_bytes=10_000_000,
    )
    resumed = _run_job(
        tmp_path,
        success_root,
        plan,
        config,
        job,
        sequence_manifest,
        resume_guard,
    )
    assert resumed["status"] == "succeeded"
    assert len(resumed["attempts"]) == 2
    abandoned, accepted = resumed["attempts"]
    assert abandoned["status"] == "abandoned_conservatively_accounted"
    assert abandoned["exit_code"] == stage2_module.ABANDONED_RUNNING_EXIT_CODE
    assert abandoned["termination_reason"] == stage2_module.ABANDONED_RUNNING_REASON
    assert abandoned["wall_time_s"] >= 2.0
    assert abandoned["command"] == original_command
    assert abandoned["stdout_log_sha256"] == original_stdout_receipt
    assert accepted["exit_code"] == 0
    assert resumed["cumulative_accounted_attempt_wall_s"] == pytest.approx(
        abandoned["wall_time_s"] + accepted["wall_time_s"], abs=1e-9
    )
    assert job_is_complete(
        crash_manifest, {**job, "plan_sha256": plan["plan_sha256"]}
    )

    mode["value"] = "slow"
    slow_root = tmp_path / "slow"
    slow_guard = _ExecutionGuard(
        slow_root,
        deadline_monotonic=time.monotonic() + 0.2,
        storage_stop_bytes=10_000_000,
    )
    started = time.monotonic()
    result = _run_job(
        tmp_path, slow_root, plan, config, job, sequence_manifest, slow_guard
    )
    assert result["status"] == "failed"
    assert result["attempts"][-1]["budget_stop_reason"] == "wall_time_guard"
    assert time.monotonic() - started < 7.0
    assert not slow_guard._processes
    failed_manifest = slow_root / "job_manifests/dense__sequence-00.json"
    tampered = json.loads(failed_manifest.read_text())
    tampered.pop("payload_sha256")
    tampered["policy_sha256"] = "0" * 64
    stage2_module._write_job_manifest(failed_manifest, tampered)
    resume_guard = _ExecutionGuard(
        slow_root,
        deadline_monotonic=time.monotonic() + 10.0,
        storage_stop_bytes=10_000_000,
    )
    with pytest.raises(Stage2Error, match="not resumable"):
        _run_job(
            tmp_path,
            slow_root,
            plan,
            config,
            job,
            sequence_manifest,
            resume_guard,
        )


def test_concurrent_storage_reservation_cancels_all_lanes(tmp_path: Path) -> None:
    guard = _ExecutionGuard(
        tmp_path,
        deadline_monotonic=time.monotonic() + 30.0,
        storage_stop_bytes=100,
    )
    assert guard.reserve("first", 60, 1.0)
    assert not guard.reserve("second", 50, 1.0)
    assert guard.cancel_event.is_set()
    assert guard.reason == "retained_storage_reservation_guard"


def test_repository_identity_excludes_only_generated_stage2_results(
    tmp_path: Path,
) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Stage2 Test"],
        check=True,
    )
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("frozen")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-q", "-m", "fixture"], check=True
    )
    baseline = _git_identity(tmp_path, generated_output_prefixes=("stage2_results",))
    generated = tmp_path / "stage2_results/design/report.md"
    generated.parent.mkdir(parents=True)
    generated.write_text("partial analysis")
    assert (
        _git_identity(tmp_path, generated_output_prefixes=("stage2_results",))
        == baseline
    )
    (tmp_path / "untracked_code.py").write_text("unsafe = True\n")
    changed = _git_identity(tmp_path, generated_output_prefixes=("stage2_results",))
    assert changed["dirty"] is True
    assert changed["untracked_file_count"] == 1
