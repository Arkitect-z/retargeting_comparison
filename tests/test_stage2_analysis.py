from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from retargeting_comparison.io_utils import sha256_file
from retargeting_comparison.schemas import CanonicalG1
from retargeting_comparison.stage2 import SCHEMA_VERSION, _plan_sha256
from retargeting_comparison.stage2_analysis import (
    ANALYSIS_METRICS,
    BOOTSTRAP_SEED,
    EXTERNAL_STRATUM,
    AnalysisTables,
    LedgerBundle,
    Stage2AnalysisError,
    build_reference_intersection,
    build_calibration_scale_tables,
    build_direct_reference_trajectory_metrics,
    build_leave_one_subject_out,
    build_stage2_analysis,
    build_stage2_charts,
    build_within_stratum_pairs,
    bootstrap_sequence_uncertainty,
    collect_stage2_job_ledger,
    load_stage2_analysis_plan,
    recompute_stage2_budget_evidence,
    render_stage2_reports,
    validate_stage2,
    VerifiedJob,
    verify_unitree_reference_evaluator_contract,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _canonical_digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _attempt_record(plan: dict, status: str, wall: float, *, schema: int) -> dict:
    value = {
        "schema_version": schema,
        "plan_sha256": plan["plan_sha256"],
        "status": status,
        "active_wall_s": wall,
    }
    value["payload_sha256"] = _canonical_digest(value)
    return value


def _plan(sequence_ids=("sequence-00",), methods=None) -> dict:
    if methods is None:
        methods = [
            ("sparse-neutral", "controlled_common_per_sequence_scale"),
        ]
    selected = [
        {
            "sequence_id": sequence_id,
            "relative_path": f"{sequence_id}.bvh",
            "source_sha256": _digest(f"source-{sequence_id}"),
            "source_size_bytes": 10,
            "frames": 3,
            "fps": 30.0,
            "duration_s": 0.1,
            "reference_relative_path": (
                f"{sequence_id}.csv"
                if any(stratum == EXTERNAL_STRATUM for _, stratum in methods)
                else None
            ),
            "reference_sha256": (
                _digest(f"reference-{sequence_id}")
                if any(stratum == EXTERNAL_STRATUM for _, stratum in methods)
                else None
            ),
            "reference_size_bytes": 5
            if any(stratum == EXTERNAL_STRATUM for _, stratum in methods)
            else 0,
            "reference_frames": 3
            if any(stratum == EXTERNAL_STRATUM for _, stratum in methods)
            else None,
        }
        for sequence_id in sequence_ids
    ]
    jobs = []
    selected_by_id = {value["sequence_id"]: value for value in selected}
    for method, stratum in methods:
        for sequence_id in sequence_ids:
            job = {
                "job_id": f"{method}__{sequence_id}",
                "method": method,
                "sequence_id": sequence_id,
                "phase_index": 0,
                "worker_index": 0,
                "estimated_runtime_s": 1.0,
                "projected_start_s": 0.0,
                "projected_end_s": 1.0,
                "stratum": stratum,
                "method_contract_sha256": _digest(f"method-{method}"),
                "policy_sha256": _digest(f"policy-{method}"),
                "reference_relative_path": (
                    f"{sequence_id}.csv" if stratum == EXTERNAL_STRATUM else None
                ),
            }
            job["job_contract_sha256"] = _canonical_digest(
                {
                    "job_id": job["job_id"],
                    "method": method,
                    "sequence": selected_by_id[sequence_id],
                    "stratum": stratum,
                    "method_contract_sha256": job["method_contract_sha256"],
                    "policy_sha256": job["policy_sha256"],
                    "reference_relative_path": job["reference_relative_path"],
                }
            )
            jobs.append(job)
    plan = {
        "schema_version": SCHEMA_VERSION,
        "stage": 2,
        "experiment_id": "test",
        "design_id": "test-design",
        "dataset_kind": "reduced_lafan1",
        "selected_sequences": selected,
        "method_specs": [
            {
                "method": method,
                "stratum": stratum,
                "public_pipeline": stratum == "native_public_pipeline",
                "method_contract_sha256": _digest(f"method-{method}"),
                "policy_sha256": _digest(f"policy-{method}"),
            }
            for method, stratum in methods
        ],
        "method_exclusions": [],
        "evidence_strata": {
            "cross_stratum_ranking_forbidden": True,
        },
        "projection": {
            "jobs": jobs,
            "wall_limit_hours": 48.0,
            "storage_limit_gb": 200.0,
        },
    }
    basis_fields = (
        "config_sha256",
        "repository",
        "hardware",
        "stage1_validation_sha256",
        "stage1_bound_verdict",
        "stage1_formal_evidence",
        "stage1_timing_evidence",
        "preflight_cost_probe",
        "unitree_reference_contract",
        "calibration_contract",
        "full_inventory_sha256",
        "selected_inventory_sha256",
        "method_specs",
        "method_design_sha256",
        "method_evidence_contract_sha256",
        "method_exclusions",
        "projection",
        "fallback",
    )
    plan["plan_basis_sha256"] = _canonical_digest(
        {key: plan.get(key) for key in basis_fields}
    )
    plan["plan_sha256"] = _plan_sha256(plan)
    return plan


def _motion(frames: int, method: str, sequence_id: str, stratum: str) -> CanonicalG1:
    qpos = np.zeros((frames, 36), dtype=float)
    qpos[:, 3] = 1.0
    metadata = {
        "method": method,
        "completion_status": "succeeded" if frames >= 3 else "incomplete",
        "experiment_stratum": stratum,
        "stage2_method_id": method,
        "stage2_sequence_id": sequence_id,
    }
    if stratum == EXTERNAL_STRATUM:
        metadata.update(
            {
                "timing_available": False,
                "verified_official_ground_truth": False,
            }
        )
    return CanonicalG1(
        qpos=qpos,
        fps=30.0,
        source_frame_idx=np.arange(frames),
        valid=np.ones(frames, dtype=bool),
        per_frame_solve_time_s=np.zeros(frames),
        metadata=metadata,
    )


def _write_succeeded_job(
    root: Path,
    run_root: Path,
    plan: dict,
    *,
    method="sparse-neutral",
    sequence_id="sequence-00",
    stratum="controlled_common_per_sequence_scale",
    frames=3,
) -> tuple[Path, Path, Path]:
    canonical = run_root / "sources" / sequence_id / "canonical_human.npz"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_bytes(b"frozen-human")
    sequence = {
        "sequence_id": sequence_id,
        "source_path": f"data/{sequence_id}.bvh",
        "source_sha256": _digest(f"source-{sequence_id}"),
        "source_frames": 3,
        "fps": 30.0,
        "canonical_path": str(canonical),
        "canonical_sha256": sha256_file(canonical),
        "common_scale": {
            "definition": (
                "frame0_root_heading_aligned_shared_semantic_landmark_weighted_"
                "scalar_least_squares"
            ),
            "formula": "argmin_s sum_k w_k ||s(h_k-h_root) - (r_k-r_root)||^2",
            "root_anchor": "test neutral root",
            "method_independent": True,
            "value": 0.75,
            "local_body_scale": 0.75,
            "root_displacement_scale": 0.75,
            "root_alignment_translation_m": [0.0, 0.0, 0.0],
            "source_landmarks": [
                "Head", "LeftArm", "RightArm", "LeftUpLeg", "RightUpLeg",
                "LeftLeg", "RightLeg", "LeftFoot", "RightFoot", "LeftToe",
                "RightToe",
            ],
            "robot_landmarks": [
                "head", "left_shoulder", "right_shoulder", "left_hip",
                "right_hip", "left_knee", "right_knee", "left_ankle",
                "right_ankle", "left_toe", "right_toe",
            ],
            "landmark_weights": [1.0] * 11,
            "least_squares_numerator_m2": 0.75,
            "least_squares_denominator_m2": 1.0,
            "weighted_residual_rmse_m": 0.1,
            "source_heading_yaw_rad": 0.0,
            "source_heading_alignment_matrix": np.eye(3).tolist(),
            "head_to_toe_diagnostic_scale": 0.74,
            "source_head_to_toe_span_m": 1.6,
            "robot_head_to_toe_span_m": 1.184,
            "robot_asset_sha256": "a" * 64,
        },
    }
    sequence_path = canonical.parent / "sequence.json"
    sequence_path.write_text(json.dumps(sequence), encoding="utf-8")
    output = run_root / "attempts" / f"{method}__{sequence_id}" / "canonical_g1.npz"
    motion = _motion(frames, method, sequence_id, stratum)
    motion.save(output, source_frame_count=3)
    manifest = {
        "schema_version": 1,
        "job_id": f"{method}__{sequence_id}",
        "method": method,
        "sequence_id": sequence_id,
        "stratum": stratum,
        "status": "succeeded",
        "plan_sha256": plan["plan_sha256"],
        "sequence_manifest": str(sequence_path),
        "sequence_manifest_sha256": sha256_file(sequence_path),
        "output_path": str(output),
        "output_sha256": sha256_file(output),
        "completion_ratio": frames / 3,
        "attempts": [{"wall_time_s": 1.25}],
    }
    manifest_path = run_root / "job_manifests" / f"{method}__{sequence_id}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return sequence_path, output, manifest_path


def test_plan_loader_requires_valid_hash_and_cross_stratum_guard(
    tmp_path: Path,
) -> None:
    plan = _plan()
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    loaded, resolved, run_root = load_stage2_analysis_plan(tmp_path, path)
    assert loaded["plan_sha256"] == plan["plan_sha256"]
    assert resolved == path
    assert run_root == tmp_path

    plan["design_id"] = "tampered"
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(Stage2AnalysisError, match="SHA-256"):
        load_stage2_analysis_plan(tmp_path, path)

    plan = _plan()
    plan["projection"]["jobs"][0]["worker_index"] = 7
    plan["plan_sha256"] = _plan_sha256(plan)
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(Stage2AnalysisError, match="decision-basis"):
        load_stage2_analysis_plan(tmp_path, path)


def test_ledger_accepts_only_exact_full_succeeded_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "retargeting_comparison.stage2_analysis.job_is_complete",
        lambda *_args, **_kwargs: True,
    )
    plan = _plan()
    run_root = tmp_path / "run"
    _write_succeeded_job(tmp_path, run_root, plan)
    bundle = collect_stage2_job_ledger(
        tmp_path, plan, run_root, run_root / "analysis/metrics"
    )
    assert len(bundle.verified) == 1
    assert bundle.ledger.integrity_status.iloc[0] == "verified"
    assert bundle.completion.complete_for_selected_design.iloc[0]
    assert (run_root / "analysis/metrics/completion_failure_ledger.parquet").is_file()


def test_ledger_fails_closed_on_corrupt_or_incomplete_claimed_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "retargeting_comparison.stage2_analysis.job_is_complete",
        lambda *_args, **_kwargs: True,
    )
    plan = _plan()
    run_root = tmp_path / "corrupt"
    _, output, _ = _write_succeeded_job(tmp_path, run_root, plan)
    output.write_bytes(b"corrupt")
    with pytest.raises(Stage2AnalysisError, match="failed integrity"):
        collect_stage2_job_ledger(
            tmp_path, plan, run_root, run_root / "analysis/metrics"
        )
    ledger = pd.read_csv(run_root / "analysis/metrics/completion_failure_ledger.csv")
    assert ledger.integrity_status.iloc[0] == "invalid"

    incomplete_root = tmp_path / "incomplete"
    _write_succeeded_job(tmp_path, incomplete_root, plan, frames=2)
    with pytest.raises(Stage2AnalysisError, match="failed integrity"):
        collect_stage2_job_ledger(
            tmp_path,
            plan,
            incomplete_root,
            incomplete_root / "analysis/metrics",
        )


def test_ledger_rejects_forged_success_without_planner_contract(tmp_path: Path) -> None:
    plan = _plan()
    run_root = tmp_path / "forged"
    _write_succeeded_job(tmp_path, run_root, plan)
    with pytest.raises(Stage2AnalysisError, match="failed integrity"):
        collect_stage2_job_ledger(
            tmp_path, plan, run_root, run_root / "analysis/metrics"
        )
    ledger = pd.read_csv(run_root / "analysis/metrics/completion_failure_ledger.csv")
    assert ledger.integrity_status.eq("invalid").all()
    assert ledger.message.str.contains("planner-provided").all()


def test_budget_recompute_charges_failed_reruns_and_raw_inputs(tmp_path: Path) -> None:
    plan = _plan()
    run_root = tmp_path / "run"
    analysis_root = tmp_path / "publication"
    execution = run_root / "execution_attempts"
    attempts = analysis_root / "analysis_attempts"
    execution.mkdir(parents=True)
    attempts.mkdir(parents=True)
    (execution / "attempt-001.json").write_text(
        json.dumps(_attempt_record(plan, "finished_error", 3.0, schema=SCHEMA_VERSION))
    )
    (execution / "attempt-002.json").write_text(
        json.dumps(
            _attempt_record(plan, "finished_complete", 4.0, schema=SCHEMA_VERSION)
        )
    )
    (attempts / "attempt-001.json").write_text(
        json.dumps(_attempt_record(plan, "failed", 2.0, schema=1))
    )
    (attempts / "attempt-002.json").write_text(
        json.dumps(_attempt_record(plan, "published", 1.5, schema=1))
    )
    (run_root / "failed.bin").write_bytes(b"x" * 23)
    (run_root / "execution_summary.json").write_text(
        json.dumps(
            {
                "cumulative_active_wall_s": 0.0,
                "retained_bytes": 0,
                "status": "forged-complete",
            }
        )
    )
    evidence = recompute_stage2_budget_evidence(plan, run_root, analysis_root)
    assert evidence.execution_attempt_wall_s == pytest.approx(7.0)
    assert evidence.prior_analysis_attempt_wall_s == pytest.approx(3.5)
    assert evidence.total_accounted_wall_s == pytest.approx(10.5)
    assert evidence.raw_input_baseline_bytes == 10
    assert evidence.total_accounted_retained_bytes == (
        evidence.retained_tree_bytes + 10
    )

    tampered = json.loads((attempts / "attempt-001.json").read_text())
    tampered["active_wall_s"] = 0.0
    (attempts / "attempt-001.json").write_text(json.dumps(tampered))
    with pytest.raises(Stage2AnalysisError, match="payload/plan contract"):
        recompute_stage2_budget_evidence(plan, run_root, analysis_root)
    (attempts / "attempt-001.json").write_text(
        json.dumps(_attempt_record(plan, "failed", 2.0, schema=1))
    )

    plan["projection"]["wall_limit_hours"] = 10.0 / 3600.0
    exceeded = recompute_stage2_budget_evidence(plan, run_root, analysis_root)
    assert not exceeded.wall_compliant

    (attempts / "attempt-003.json").write_text(
        json.dumps(_attempt_record(plan, "running", 0.1, schema=1))
    )
    with pytest.raises(Stage2AnalysisError, match="not finalized"):
        recompute_stage2_budget_evidence(plan, run_root, analysis_root)


def test_reference_contract_rehashes_assets_and_equivalence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision = "frozen-revision"
    files = {
        "manifest": tmp_path / "manifests/unitree.json",
        "config": tmp_path / "configs/unitree.yaml",
        "adapter": tmp_path / "src/adapter.py",
        "reference_urdf": tmp_path / "assets/reference.urdf",
        "canonical_urdf": tmp_path / "assets/canonical.urdf",
        "canonical_scene": tmp_path / "assets/scene.xml",
        "evaluator": tmp_path / "manifests/evaluator.yaml",
    }
    for name, path in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"frozen-{name}", encoding="utf-8")
    manifest_value = {
        "schema_version": 2,
        "role": "external_reference_not_verified_ground_truth",
        "verified_official_ground_truth": False,
        "eligible_for_runtime_comparison": False,
    }
    files["manifest"].write_text(json.dumps(manifest_value), encoding="utf-8")
    csv_root = (
        tmp_path
        / "data/external/unitree_lafan1_reference"
        / revision
        / "g1"
    )
    csv_root.mkdir(parents=True)
    csv = csv_root / "sequence-00.csv"
    csv.write_text("0,1\n", encoding="utf-8")
    inventory = hashlib.sha256()
    inventory.update(csv.name.encode())
    inventory.update(bytes.fromhex(sha256_file(csv)))

    def receipt(path: Path) -> dict:
        return {
            "path": path.relative_to(tmp_path).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }

    urdf_audit = {
        "kinematic_contract_equivalent": True,
        "canonical_g1_order_match": True,
    }
    fk_audit = {
        "random_sample_count": 3,
        "seed": 9,
        "position_tolerance_m": 1e-5,
        "rotation_tolerance_rad": 1e-5,
        "kinematic_fk_equivalent": True,
    }
    monkeypatch.setattr(
        "retargeting_comparison.stage2_analysis.load_evaluator_protocol",
        lambda _path: {"robot_xml_sha256": sha256_file(files["canonical_scene"])},
    )
    monkeypatch.setattr(
        "retargeting_comparison.unitree_reference.compare_urdf_kinematics",
        lambda *_args, **_kwargs: dict(urdf_audit),
    )
    monkeypatch.setattr(
        "retargeting_comparison.unitree_reference.compare_urdf_to_mujoco_fk",
        lambda *_args, **_kwargs: dict(fk_audit),
    )
    manifest_value.update(
        {
            "upstream_revision": revision,
            "source_binding": {
                "alignment_basis": "same basename and frame indices 0:600",
                "byte_identical_human_source_verified": False,
                "exact_timestamp_identity_claimed": False,
            },
            "pinned_resources": {
                "g1_urdf": {
                    "path": files["reference_urdf"].relative_to(tmp_path).as_posix(),
                    "sha256": sha256_file(files["reference_urdf"]),
                    "size_bytes": files["reference_urdf"].stat().st_size,
                }
            },
            "g1_csv_inventory": {
                "files": 1,
                "aggregate_sha256": inventory.hexdigest(),
            },
            "asset_binding": {
                "canonical_evaluator_urdf_path": files["canonical_urdf"]
                .relative_to(tmp_path)
                .as_posix(),
                "canonical_evaluator_urdf_sha256": sha256_file(
                    files["canonical_urdf"]
                ),
                "canonical_evaluator_scene_path": files["canonical_scene"]
                .relative_to(tmp_path)
                .as_posix(),
                "canonical_evaluator_scene_sha256": sha256_file(
                    files["canonical_scene"]
                ),
                "evaluator_manifest_path": files["evaluator"]
                .relative_to(tmp_path)
                .as_posix(),
                "evaluator_manifest_sha256": sha256_file(files["evaluator"]),
            },
            "adapter": {
                "implementation_path": files["adapter"]
                .relative_to(tmp_path)
                .as_posix(),
                "implementation_sha256": sha256_file(files["adapter"]),
            },
        }
    )
    files["manifest"].write_text(json.dumps(manifest_value), encoding="utf-8")
    contract = {
        "schema_version": 2,
        "manifest": receipt(files["manifest"]),
        "config": receipt(files["config"]),
        "revision": revision,
        "role": "external_reference_not_verified_ground_truth",
        "alignment_contract": "same_basename_and_frame_indices_only",
        "byte_identical_human_source_verified": False,
        "exact_timestamp_identity_claimed": False,
        "pinned_resources": {"g1_urdf": receipt(files["reference_urdf"])},
        "g1_csv_inventory": {
            "files": 1,
            "aggregate_sha256": inventory.hexdigest(),
        },
        "asset_binding": {
            "canonical_evaluator_urdf_path": receipt(files["canonical_urdf"]),
            "canonical_evaluator_scene_path": receipt(files["canonical_scene"]),
            "evaluator_manifest_path": receipt(files["evaluator"]),
        },
        "adapter": receipt(files["adapter"]),
        "urdf_kinematic_audit": urdf_audit,
        "urdf_to_mujoco_fk_audit": fk_audit,
    }
    contract["contract_sha256"] = _canonical_digest(contract)
    plan = _plan(
        methods=(("unitree-reference", EXTERNAL_STRATUM),)
    )
    plan["unitree_reference_contract"] = contract
    evidence = verify_unitree_reference_evaluator_contract(tmp_path, plan)
    assert evidence["verified"]
    assert evidence["pairing"] == "normalized basename and integer frame index only"

    files["canonical_scene"].write_text("tampered", encoding="utf-8")
    with pytest.raises(Stage2AnalysisError, match="bound file changed"):
        verify_unitree_reference_evaluator_contract(tmp_path, plan)


def _metric_frame() -> pd.DataFrame:
    rows = []
    definitions = [
        ("controlled_common_per_sequence_scale", "sparse-neutral", 0.00),
        ("controlled_common_per_sequence_scale", "dense", -0.01),
        ("native_public_pipeline", "gmr", 0.02),
        ("native_public_pipeline", "omniretarget", 0.01),
        (EXTERNAL_STRATUM, "unitree-reference", 0.005),
    ]
    for stratum, method, offset in definitions:
        for index in range(4):
            row = {
                "stratum": stratum,
                "method": method,
                "sequence_id": f"sequence-{index:02d}",
                "display": method,
            }
            row.update(
                {
                    metric: 0.1 + index * 0.01 + offset + metric_index * 0.001
                    for metric_index, metric in enumerate(ANALYSIS_METRICS)
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def test_sequence_bootstrap_is_deterministic_and_order_invariant() -> None:
    frame = _metric_frame().loc[lambda value: value.method == "gmr"]
    first = bootstrap_sequence_uncertainty(
        frame,
        group_columns=("stratum", "method"),
        resamples=500,
    )
    second = bootstrap_sequence_uncertainty(
        frame.sample(frac=1.0, random_state=9),
        group_columns=("stratum", "method"),
        resamples=500,
    )
    pd.testing.assert_frame_equal(first, second)
    assert (first.bootstrap_seed == BOOTSTRAP_SEED).all()
    assert (first.uncertainty_unit == "subject_cluster").all()


def test_actor_cluster_bootstrap_and_loso_keep_subjects_intact() -> None:
    frame = _metric_frame().loc[lambda value: value.method == "gmr"].copy()
    frame["subject_id"] = frame.sequence_id.map(
        {
            "sequence-00": "subject1",
            "sequence-01": "subject1",
            "sequence-02": "subject2",
            "sequence-03": "subject2",
        }
    )
    summary = bootstrap_sequence_uncertainty(
        frame, group_columns=("stratum", "method"), resamples=500
    )
    assert (summary.subject_count == 2).all()
    loso = build_leave_one_subject_out(frame)
    assert set(loso.held_out_subject_id) == {"subject1", "subject2"}
    assert (loso.retained_subject_count == 1).all()
    assert (loso.retained_sequence_count == 2).all()


def test_pairs_never_cross_strata_and_reference_is_not_accuracy() -> None:
    frame = _metric_frame()
    paired, paired_summary = build_within_stratum_pairs(frame, resamples=500)
    assert not paired.empty
    assert not paired.cross_stratum_comparison.any()
    assert set(paired.stratum) == {
        "controlled_common_per_sequence_scale",
        "native_public_pipeline",
    }
    assert not (paired.method_a.eq("sparse-neutral") & paired.method_b.eq("gmr")).any()
    assert not paired_summary.cross_stratum_comparison.any()

    reference, summary = build_reference_intersection(frame, resamples=500)
    assert reference.sequence_id.nunique() == 4
    assert reference.interpretation.str.contains("not ground truth").all()
    assert summary.interpretation.str.contains("not truth").all()
    assert "rank" not in " ".join(reference.columns).lower()


def test_direct_reference_metrics_compare_same_g1_trajectory(tmp_path: Path) -> None:
    sequence_path = tmp_path / "sequence.json"
    sequence_path.write_text(json.dumps({"actor_id": "subject1"}), encoding="utf-8")
    method_motion = _motion(
        3, "dense", "walk-subject1", "controlled_common_per_sequence_scale"
    )
    method_motion.qpos[:, 0] = 0.1
    method_motion.qpos[:, 7:] = 0.2
    method_output = tmp_path / "method.npz"
    method_motion.save(method_output, source_frame_count=3)
    reference_motion = _motion(
        3, "unitree-reference", "walk-subject1", EXTERNAL_STRATUM
    )
    reference_output = tmp_path / "reference.npz"
    reference_motion.save(reference_output, source_frame_count=3)

    class FakeRobot:
        body_ids = {"root": 0}

        @staticmethod
        def semantic_positions(qpos):
            root = np.asarray(qpos[:3], dtype=float)
            return {"root": root, "head": root + np.asarray([0.0, 0.0, 1.0])}

    dummy_manifest = tmp_path / "job.json"
    dummy_manifest.write_text("{}")
    jobs = [
        VerifiedJob(
            job_id="dense__walk-subject1",
            method="dense",
            sequence_id="walk-subject1",
            stratum="controlled_common_per_sequence_scale",
            job_manifest_path=dummy_manifest,
            sequence_manifest_path=sequence_path,
            output_path=method_output,
            production_wall_s=1.0,
        ),
        VerifiedJob(
            job_id="unitree-reference__walk-subject1",
            method="unitree-reference",
            sequence_id="walk-subject1",
            stratum=EXTERNAL_STRATUM,
            job_manifest_path=dummy_manifest,
            sequence_manifest_path=sequence_path,
            output_path=reference_output,
            production_wall_s=0.0,
        ),
    ]
    direct, summary = build_direct_reference_trajectory_metrics(
        jobs, FakeRobot(), resamples=100
    )
    assert len(direct) == 1
    assert direct.root_translation_agreement_mean_m.iloc[0] == pytest.approx(0.1)
    assert direct.root_anchor_frame0_agreement_m.iloc[0] == pytest.approx(0.1)
    assert direct.root_displacement_agreement_mean_m.iloc[0] == pytest.approx(0.0)
    assert direct.root_displacement_agreement_p95_m.iloc[0] == pytest.approx(0.0)
    assert direct.joint_angle_agreement_rmse_rad.iloc[0] == pytest.approx(0.2)
    assert direct.fk_root_frame_semantic_agreement_mean_m.iloc[0] == pytest.approx(0.0)
    assert direct.interpretation.str.contains("basename and integer frame index").all()
    assert direct.interpretation.str.contains("exact timestamp identity").all()
    assert summary.intersection_subject_count.eq(1).all()


def test_calibration_tables_are_per_sequence_and_sensitivity_is_noncausal(
    tmp_path: Path,
) -> None:
    plan = _plan()
    run_root = tmp_path / "run"
    sequence_path, output, manifest_path = _write_succeeded_job(
        tmp_path, run_root, plan
    )
    verified = [
        VerifiedJob(
            job_id="sparse-neutral__sequence-00",
            method="sparse-neutral",
            sequence_id="sequence-00",
            stratum="controlled_common_per_sequence_scale",
            job_manifest_path=manifest_path,
            sequence_manifest_path=sequence_path,
            output_path=output,
            production_wall_s=1.0,
        )
    ]
    row = {
        "sequence_id": "sequence-00",
        "subject_id": "subject1",
        "stratum": "controlled_common_per_sequence_scale",
        "method": "sparse-neutral",
        "rf_kpe_all_mean_m": 0.1,
        "root_translation_common_scale_mean_m": 0.2,
        "root_translation_scale_invariant_mean_m": 0.05,
        "artifact_rate": 0.0,
    }
    calibration, variability, sensitivity, conclusion = (
        build_calibration_scale_tables(
            verified, pd.DataFrame([row]), tmp_path / "metrics"
        )
    )
    assert calibration.calibration_unit.eq(
        "one frozen estimate per source sequence"
    ).all()
    assert calibration.sequence_id.nunique() == 1
    assert variability.estimand.eq("sequence-level calibration variability").all()
    assert sensitivity.interpretation.str.contains("not a causal").any()
    assert conclusion.forbidden_conclusion.str.contains("cross-stratum").all()


def test_analysis_attempt_publishes_manifest_last_and_records_wall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan()
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "job_manifests").mkdir()
    execution_attempts = run_root / "execution_attempts"
    execution_attempts.mkdir()
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    raw_source = raw_root / "sequence-00.bvh"
    raw_source.write_bytes(b"0123456789")
    plan["selected_sequences"][0]["source_sha256"] = sha256_file(raw_source)
    plan["selected_sequences"][0]["source_size_bytes"] = raw_source.stat().st_size
    job = plan["projection"]["jobs"][0]
    job["job_contract_sha256"] = _canonical_digest(
        {
            "job_id": job["job_id"],
            "method": job["method"],
            "sequence": plan["selected_sequences"][0],
            "stratum": job["stratum"],
            "method_contract_sha256": job["method_contract_sha256"],
            "policy_sha256": job["policy_sha256"],
            "reference_relative_path": job["reference_relative_path"],
        }
    )
    config = tmp_path / "stage2.yaml"
    config.write_text("dataset:\n  root: raw\n", encoding="utf-8")
    plan["config_path"] = config.relative_to(tmp_path).as_posix()
    plan["config_sha256"] = sha256_file(config)
    plan["tracked_plan_path"] = "publication/STAGE2_PLAN.json"
    basis_fields = (
        "config_sha256",
        "repository",
        "hardware",
        "stage1_validation_sha256",
        "stage1_bound_verdict",
        "stage1_formal_evidence",
        "stage1_timing_evidence",
        "preflight_cost_probe",
        "unitree_reference_contract",
        "calibration_contract",
        "full_inventory_sha256",
        "selected_inventory_sha256",
        "method_specs",
        "method_design_sha256",
        "method_evidence_contract_sha256",
        "method_exclusions",
        "projection",
        "fallback",
    )
    plan["plan_basis_sha256"] = _canonical_digest(
        {key: plan.get(key) for key in basis_fields}
    )
    plan["plan_sha256"] = _plan_sha256(plan)
    (execution_attempts / "attempt-001.json").write_text(
        json.dumps(
            _attempt_record(
                plan, "finished_complete", 0.1, schema=SCHEMA_VERSION
            )
        )
    )
    plan_path = run_root / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    sequence_path, output, job_manifest = _write_succeeded_job(
        tmp_path, run_root, plan
    )
    verified = VerifiedJob(
        job_id="sparse-neutral__sequence-00",
        method="sparse-neutral",
        sequence_id="sequence-00",
        stratum="controlled_common_per_sequence_scale",
        job_manifest_path=job_manifest,
        sequence_manifest_path=sequence_path,
        output_path=output,
        production_wall_s=1.0,
    )
    ledger_frame = pd.DataFrame(
        [
            {
                "job_id": verified.job_id,
                "method": verified.method,
                "sequence_id": verified.sequence_id,
                "stratum": verified.stratum,
                "status": "succeeded",
                "included_in_quality_analysis": True,
                "integrity_status": "verified",
                "message": "verified",
            }
        ]
    )
    completion = pd.DataFrame(
        [
            {
                "method": verified.method,
                "stratum": verified.stratum,
                "expected_jobs": 1,
                "verified_succeeded_jobs": 1,
                "failed_jobs": 0,
                "incomplete_jobs": 0,
                "missing_jobs": 0,
                "integrity_invalid_jobs": 0,
                "completion_fraction": 1.0,
                "complete_for_selected_design": True,
            }
        ]
    )
    scope = pd.DataFrame(
        [
            {
                "method": verified.method,
                "stratum": verified.stratum,
                "stage2_scope_status": "selected",
                "reason": "test",
                "result_metrics_used_for_scope_decision": False,
            }
        ]
    )
    bundle = LedgerBundle(ledger_frame, completion, scope, [verified])
    metric_row = (
        _metric_frame()
        .loc[lambda value: (value.method == "sparse-neutral")]
        .iloc[[0]]
        .copy()
    )
    metric_row["job_id"] = verified.job_id
    metric_row["subject_id"] = "subject1"
    metric_row["completion_ratio"] = 1.0
    monkeypatch.setattr(
        "retargeting_comparison.stage2_analysis.collect_stage2_job_ledger",
        lambda *_args, **_kwargs: bundle,
    )
    monkeypatch.setattr(
        "retargeting_comparison.stage2_analysis.evaluate_verified_stage2_jobs",
        lambda *_args, **_kwargs: metric_row,
    )

    class FakeRobot:
        body_ids = {}

    monkeypatch.setattr(
        "retargeting_comparison.stage2_analysis.CanonicalRobotModel",
        lambda _path: FakeRobot(),
    )
    monkeypatch.setattr(
        "retargeting_comparison.stage2_analysis.default_robot_scene",
        lambda _root: tmp_path / "unused.xml",
    )
    manifest = build_stage2_analysis(tmp_path, plan_path)
    publication = tmp_path / "publication"
    assert manifest["status"] == "complete"
    assert (publication / "analysis_manifest.json").is_file()
    attempt = json.loads(
        (publication / "analysis_attempts/attempt-001.json").read_text()
    )
    assert attempt["status"] == "published"
    assert attempt["published"] is True
    assert attempt["active_wall_s"] > 0.0
    assert attempt["publication_manifest_sha256"] == sha256_file(
        publication / "analysis_manifest.json"
    )


def test_validation_is_a_charged_atomic_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan()
    plan["tracked_plan_path"] = "publication/STAGE2_PLAN.json"
    basis_fields = (
        "config_sha256",
        "repository",
        "hardware",
        "stage1_validation_sha256",
        "stage1_bound_verdict",
        "stage1_formal_evidence",
        "stage1_timing_evidence",
        "preflight_cost_probe",
        "unitree_reference_contract",
        "calibration_contract",
        "full_inventory_sha256",
        "selected_inventory_sha256",
        "method_specs",
        "method_design_sha256",
        "method_evidence_contract_sha256",
        "method_exclusions",
        "projection",
        "fallback",
    )
    plan["plan_basis_sha256"] = _canonical_digest(
        {key: plan.get(key) for key in basis_fields}
    )
    plan["plan_sha256"] = _plan_sha256(plan)
    run_root = tmp_path / "run"
    (run_root / "job_manifests").mkdir(parents=True)
    execution = run_root / "execution_attempts"
    execution.mkdir()
    (execution / "attempt-001.json").write_text(
        json.dumps(
            _attempt_record(
                plan, "finished_incomplete", 0.1, schema=SCHEMA_VERSION
            )
        )
    )
    plan_path = run_root / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    monkeypatch.setattr(
        "retargeting_comparison.stage2_analysis.recompute_stage2_decision_evidence",
        lambda *_args, **_kwargs: {
            "tracked_plan_exact_equality": False,
            "selected_sequence_count": 1,
            "expected_job_count": 1,
            "verified_succeeded_job_count": 0,
            "output_rows": [],
            "output_evidence_sha256": _canonical_digest([]),
            "reference_contract": {"verified": True},
            "budget": {
                "wall_compliant": True,
                "storage_compliant": True,
                "total_accounted_retained_bytes": 0,
                "storage_limit_bytes": 200_000_000_000,
            },
            "decision_basis_sha256": _digest("partial"),
        },
    )
    result = validate_stage2(tmp_path, plan_path)
    assert result["decision"] == "NO-GO"
    validation_path = tmp_path / "publication/STAGE2_VALIDATION.json"
    assert validation_path.is_file()
    attempts = sorted((tmp_path / "publication/analysis_attempts").glob("*.json"))
    assert len(attempts) == 1
    attempt = json.loads(attempts[0].read_text())
    assert attempt["attempt_kind"] == "stage2_validation"
    assert attempt["status"] == "validation_published"
    assert attempt["published"] is True
    assert attempt["publication_manifest_sha256"] == sha256_file(validation_path)
    payload = dict(attempt)
    recorded = payload.pop("payload_sha256")
    assert recorded == _canonical_digest(payload)


def test_missing_plan_public_entry_never_creates_run_state(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_stage2_analysis(tmp_path, "missing-plan.json")
    assert not (tmp_path / "runs").exists()


def test_analysis_manifest_exposes_cli_status_alias() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src/retargeting_comparison/stage2_analysis.py"
    )
    text = source.read_text(encoding="utf-8")
    assert '"status": analysis_status' in text
    assert '"analysis_status": analysis_status' in text


def test_stage2_charts_and_reports_render_complete_stratified_evidence(
    tmp_path: Path,
) -> None:
    per_sequence = _metric_frame()
    per_method = bootstrap_sequence_uncertainty(
        per_sequence,
        group_columns=("stratum", "method"),
        resamples=500,
    )
    pairs, pair_summary = build_within_stratum_pairs(per_sequence, resamples=500)
    reference, reference_summary = build_reference_intersection(
        per_sequence, resamples=500
    )
    tables = AnalysisTables(
        per_sequence_method=per_sequence,
        per_method_metric=per_method,
        within_stratum_pairs=pairs,
        within_stratum_pair_summary=pair_summary,
        reference_intersection=reference,
        reference_summary=reference_summary,
    )
    completion = pd.DataFrame(
        [
            {
                "method": method,
                "stratum": stratum,
                "expected_jobs": 4,
                "verified_succeeded_jobs": 4,
                "failed_jobs": 0,
                "incomplete_jobs": 0,
                "missing_jobs": 0,
                "integrity_invalid_jobs": 0,
                "completion_fraction": 1.0,
                "complete_for_selected_design": True,
            }
            for stratum, method in per_sequence[["stratum", "method"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        ]
    )
    ledger_frame = pd.DataFrame(
        [
            {
                "job_id": f"{row.method}__{row.sequence_id}",
                "method": row.method,
                "sequence_id": row.sequence_id,
                "stratum": row.stratum,
                "status": "succeeded",
                "included_in_quality_analysis": True,
                "integrity_status": "verified",
                "message": "verified",
            }
            for row in per_sequence.itertuples()
        ]
    )
    method_scope = completion[["method", "stratum"]].copy()
    method_scope["stage2_scope_status"] = "selected"
    method_scope["reason"] = "test plan"
    method_scope["result_metrics_used_for_scope_decision"] = False
    ledger = LedgerBundle(
        ledger=ledger_frame,
        completion=completion,
        method_scope=method_scope,
        verified=[],
    )
    methods = list(
        per_sequence[["method", "stratum"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    plan = _plan(
        sequence_ids=tuple(f"sequence-{index:02d}" for index in range(4)),
        methods=methods,
    )
    plan["_analysis_plan_path"] = "runs/stage2/test-design/plan.json"
    charts = build_stage2_charts(tmp_path, ledger, tables)
    reports = render_stage2_reports(tmp_path, plan, ledger, tables)
    assert len(charts) == 7 * 4
    assert len(reports) == 3
    assert all(path.is_file() and path.stat().st_size for path in charts + reports)
    text = (tmp_path / "STAGE2_REPORT.md").read_text(encoding="utf-8")
    assert "no cross-stratum causal rank" in text
    assert "not verified official ground truth" in text
    assert "Method − reference" in text
    render_stage2_reports(
        tmp_path,
        plan,
        ledger,
        tables,
        budget_compliant=False,
    )
    text = (tmp_path / "STAGE2_REPORT.md").read_text(encoding="utf-8")
    assert text.startswith("# Stage 2 LAFAN Analysis Report — PARTIAL")
    assert "48-hour/200-GB accounting check" in text
