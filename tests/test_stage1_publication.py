from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from retargeting_comparison.schemas import CanonicalG1, RunManifest, RunStatus
from retargeting_comparison.scale_metric_registry import (
    SCALE_RANK_METRICS,
    SCALE_RESPONSE_METRIC_NAMES,
)
from retargeting_comparison.stage1_publication import (
    CORE_METHODS,
    EXPECTED_FRAMES,
    FORMAL_NATIVE_SCALE_ROLE,
    HISTORICAL_STAGE1_STOP,
    NATIVE_SCALE_METHODS,
    QUALITY_COLUMNS,
    REFERENCE_METHOD,
    REPORT_FILES,
    SCALE_VARIANTS,
    CONTACT_ROBUSTNESS_ROLE,
    PUBLICATION_EVIDENCE_SCHEMA_VERSION,
    VALIDATION_SCHEMA_VERSION,
    SCALE_INPUT_LEDGER,
    SCALE_INPUT_LEDGER_SCHEMA_VERSION,
    EvaluationBundle,
    ScaleBundle,
    TRANSPLANT_POLICIES,
    _validate_motion_contract,
    _write_stage1_markdown,
    _canonical_sha256,
    _expected_scale_evidence_keys,
    _runtime_scale_evidence,
    build_direct_reference_comparison,
    build_reference_comparison,
    build_stage1_figures,
    collect_scale_evidence,
    derive_scale_tables,
    parse_timing_record,
    presentation_slide_count,
    inspect_publication_evidence,
    method_output_paths,
    resolve_bound_stage1_validation,
    render_stage1_reports,
    require_stage1_outputs,
    save_publication_chart,
    validation_decision_basis_sha256,
    verify_scale_publication_input_ledger,
)
from retargeting_comparison.io_utils import sha256_file


def _write_pending_evidence_manifest(root: Path) -> dict[str, object]:
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    (root / "inputs").mkdir(parents=True, exist_ok=True)
    (root / "outputs").mkdir(parents=True, exist_ok=True)
    source = root / "inputs" / "source.bin"
    output = root / "outputs" / "metric.csv"
    source.write_bytes(b"source-v1")
    output.write_text("metric\n1\n", encoding="utf-8")

    def row(path: Path) -> dict[str, object]:
        return {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }

    inputs = [row(source)]
    outputs = [row(output)]
    manifest: dict[str, object] = {
        "schema_version": PUBLICATION_EVIDENCE_SCHEMA_VERSION,
        "decision": "PENDING",
        "state": "evidence_ready_for_independent_validation",
        "decision_authority": "manifests/stage1_validation.json",
        "input_hashes": inputs,
        "input_bundle_sha256": _canonical_sha256(inputs),
        "outputs": outputs,
        "output_bundle_sha256": _canonical_sha256(outputs),
    }
    (root / "manifests" / "stage1_publication.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _write_bound_validation(
    root: Path, checks: dict[str, bool], *, decision: str | None = None
) -> dict[str, object]:
    binding = inspect_publication_evidence(root)
    resolved_decision = decision or ("GO" if all(checks.values()) else "NO-GO")
    value: dict[str, object] = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "authority": "independent_fail_closed_stage1_validator",
        "decision": resolved_decision,
        "checks": checks,
        "check_count": len(checks),
        "passed_check_count": sum(checks.values()),
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "publication_evidence_binding": binding,
        "decision_basis_sha256": validation_decision_basis_sha256(
            resolved_decision, checks, binding
        ),
    }
    (root / "manifests" / "stage1_validation.json").write_text(
        json.dumps(value, sort_keys=True) + "\n", encoding="utf-8"
    )
    return value


def _motion(metadata: dict[str, object]) -> CanonicalG1:
    qpos = np.zeros((EXPECTED_FRAMES, 36), dtype=np.float64)
    qpos[:, 3] = 1.0
    return CanonicalG1(
        qpos=qpos,
        fps=30.0,
        source_frame_idx=np.arange(EXPECTED_FRAMES),
        valid=np.ones(EXPECTED_FRAMES, dtype=bool),
        per_frame_solve_time_s=np.full(EXPECTED_FRAMES, 0.001),
        metadata={"completion_status": "succeeded", **metadata},
    )


def test_publication_method_identity_and_roles_are_unambiguous() -> None:
    assert [spec.key for spec in CORE_METHODS] == [
        "sparse-neutral",
        "dense",
        "gmr",
        "omniretarget",
        "protomotions-v2.3",
        "protomotions-v3",
    ]
    assert "Mink LAFAN port" in CORE_METHODS[4].display_name
    assert "PHC" not in CORE_METHODS[4].display_name
    assert "modified-PyRoki" in CORE_METHODS[5].display_name
    assert "LAFAN port" in CORE_METHODS[5].display_name


def _write_output_path_manifest(
    root: Path, *, output_path: Path, run_directory: str = "dense-v6"
) -> None:
    sequence_id = "pilot"
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    (root / "manifests/pilot_sequence.yaml").write_text(
        f"sequence_id: {sequence_id}\n", encoding="utf-8"
    )
    RunManifest(
        run_id=f"{sequence_id}__{run_directory}",
        method=run_directory,
        status=RunStatus.SUCCEEDED,
        command=["formal"],
        environment="conda:capture",
        repo_commit="0" * 40,
        config_sha256="1" * 64,
        device="cpu",
        exit_code=0,
        output_path=output_path.as_posix(),
        output_sha256="2" * 64,
        source_sha256="3" * 64,
    ).save(root / "runs/manifests" / f"{sequence_id}__{run_directory}.json")


def test_method_output_paths_follow_accepted_attempt_manifest(tmp_path: Path) -> None:
    accepted = tmp_path / "runs/pilot/dense-v6/attempt_20260722/canonical_g1.npz"
    _write_output_path_manifest(tmp_path, output_path=accepted)

    assert method_output_paths(tmp_path)["dense"] == accepted.resolve()


def test_method_output_paths_reject_manifest_escape(tmp_path: Path) -> None:
    escaped = tmp_path / "runs/pilot/not-dense/canonical_g1.npz"
    _write_output_path_manifest(tmp_path, output_path=escaped)

    with pytest.raises(ValueError, match="escapes its registered revision"):
        method_output_paths(tmp_path)
    assert "common_shared_semantic_landmark_ls" in TRANSPLANT_POLICIES
    assert "common_frame0_head_toe_span" not in TRANSPLANT_POLICIES
    assert REFERENCE_METHOD.timed is False
    assert "not verified official" in REFERENCE_METHOD.evidence_role
    assert len(REPORT_FILES) == 11


def test_publication_manifest_is_evidence_only_and_cannot_self_issue_go(
    tmp_path: Path,
) -> None:
    manifest = _write_pending_evidence_manifest(tmp_path)
    binding = inspect_publication_evidence(tmp_path)
    assert manifest["decision"] == "PENDING"
    assert binding["input_bundle_sha256"] == manifest["input_bundle_sha256"]

    manifest["decision"] = "GO"
    (tmp_path / "manifests" / "stage1_publication.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="PENDING evidence ledger"):
        inspect_publication_evidence(tmp_path)


def test_bound_validation_is_authoritative_and_missing_validation_is_pending(
    tmp_path: Path,
) -> None:
    _write_pending_evidence_manifest(tmp_path)
    missing = resolve_bound_stage1_validation(tmp_path)
    assert missing["decision"] == "PENDING"
    assert missing["binding_status"] == "validation_missing"

    _write_bound_validation(tmp_path, {"experiments": True, "reports": True})
    resolved = resolve_bound_stage1_validation(tmp_path)
    assert resolved["decision"] == "GO"
    assert resolved["binding_status"] == "verified"


def test_validation_verdict_forgery_and_stale_input_fail_closed(
    tmp_path: Path,
) -> None:
    _write_pending_evidence_manifest(tmp_path)
    forged = _write_bound_validation(
        tmp_path,
        {"experiments": False, "reports": True},
        decision="GO",
    )
    # Even a recomputed digest cannot make GO consistent with a failed check.
    assert forged["decision_basis_sha256"]
    resolution = resolve_bound_stage1_validation(tmp_path)
    assert resolution["decision"] == "PENDING"
    assert resolution["binding_status"] == "validation_unbound_or_invalid"

    _write_bound_validation(tmp_path, {"experiments": True, "reports": True})
    (tmp_path / "inputs" / "source.bin").write_bytes(b"source-v2")
    stale = resolve_bound_stage1_validation(tmp_path)
    assert stale["decision"] == "PENDING"
    assert stale["binding_status"] == "invalid_publication_evidence"


def test_bound_failed_validation_resolves_to_no_go(tmp_path: Path) -> None:
    _write_pending_evidence_manifest(tmp_path)
    _write_bound_validation(tmp_path, {"experiments": True, "reports": False})
    resolved = resolve_bound_stage1_validation(tmp_path)
    assert resolved["decision"] == "NO-GO"
    assert resolved["binding_status"] == "verified"


def test_missing_v3_is_a_hard_error_not_an_na_row(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "pilot_sequence.yaml").write_text(
        "sequence_id: pilot\n"
        "num_frames: 600\n"
        "canonical_path: source/pilot.npz\n",
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="protomotions-v3"):
        require_stage1_outputs(tmp_path)


def test_reference_contract_forbids_timing_and_ground_truth_claims() -> None:
    valid = _motion(
        {
            "method": "unitree_reference",
            "timing_available": False,
            "verified_official_ground_truth": False,
        }
    )
    _validate_motion_contract(valid, REFERENCE_METHOD, EXPECTED_FRAMES)
    bad = _motion(
        {
            "method": "unitree_reference",
            "timing_available": False,
            "verified_official_ground_truth": True,
        }
    )
    with pytest.raises(ValueError, match="ground truth"):
        _validate_motion_contract(bad, REFERENCE_METHOD, EXPECTED_FRAMES)


def test_formal_timing_parser_preserves_repetitions_and_recomputes_rtf() -> None:
    entry = {
        "frame_count": 600,
        "steady_end_to_end_total_s": 2.0,
        "wall_time_s": 2.5,
        "native_total_s": 1.0,
        "initialization_time_s": 0.2,
        "timing_boundary": "canonical_source_file_to_canonical_g1_in_memory",
        "timing_artifact_path": "timing-artifact.npz",
        "timing_artifact_sha256": "0" * 64,
    }
    payload = {
        "cold": entry,
        "warmup": [entry],
        "measured_warm": [entry, {**entry, "steady_end_to_end_total_s": 2.2}, entry],
    }
    raw, summary = parse_timing_record(
        "method",
        payload,
        fps=30.0,
        frame_count=600,
        fallback_native_total_s=1.0,
    )
    assert len(raw) == 5
    assert summary["timing_evidence_grade"] == "formal_repeated"
    assert summary["measured_repetitions"] == 3
    assert summary["end_to_end_rtf_median"] == pytest.approx(0.1)
    assert summary["native_core_rtf_median"] == pytest.approx(0.05)


def test_timing_summary_only_is_explicitly_lower_grade() -> None:
    raw, summary = parse_timing_record(
        "slow-method",
        {"end_to_end_rtf_median": 7.5, "native_core_rtf_median": 7.0},
        fps=30.0,
        frame_count=600,
        fallback_native_total_s=140.0,
    )
    assert len(raw) == 1
    assert raw.role.iloc[0] == "reported-summary"
    assert summary["timing_evidence_grade"] == "reported_summary_only"


def _scale_summary() -> pd.DataFrame:
    rows = []
    offsets = {
        "native": 0.0,
        "root_minus_5": -0.05,
        "root_plus_5": 0.05,
        "local_minus_5": -0.02,
        "local_plus_5": 0.02,
    }
    for method_index, method in enumerate(NATIVE_SCALE_METHODS, start=1):
        for variant in SCALE_VARIANTS:
            delta = offsets[variant]
            rows.append(
                {
                    "experiment_role": FORMAL_NATIVE_SCALE_ROLE,
                    "method": method,
                    "variant": variant,
                    "rf_kpe_all_mean_m": 0.1 * method_index + delta * 0.01,
                    "root_translation_common_scale_mean_m": 0.2 * method_index + delta,
                    "artifact_rate": 0.05 * method_index + delta * 0.01,
                    "foot_skating_frame_rate": 0.04 * method_index + delta * 0.01,
                }
            )
            for metric in SCALE_RESPONSE_METRIC_NAMES:
                rows[-1].setdefault(
                    metric,
                    1.0
                    if metric == "completion_ratio"
                    else 0.01 * method_index + delta * 0.001,
                )
    for policy in TRANSPLANT_POLICIES:
        rows.append(
            {
                "experiment_role": "controlled_policy",
                "method": "controlled",
                "variant": policy,
                "rf_kpe_all_mean_m": 0.1,
                "root_translation_common_scale_mean_m": 0.2,
                "artifact_rate": 0.1,
                "foot_skating_frame_rate": 0.1,
            }
        )
        for metric in SCALE_RESPONSE_METRIC_NAMES:
            rows[-1].setdefault(
                metric, 1.0 if metric == "completion_ratio" else 0.1
            )
    return pd.DataFrame(rows)


def test_scale_derivation_requires_full_matrix_and_computes_slopes() -> None:
    source = _scale_summary()
    slopes, ranks = derive_scale_tables(source)
    assert len(slopes) == len(NATIVE_SCALE_METHODS) * len(
        SCALE_RESPONSE_METRIC_NAMES
    ) * 2
    assert len(ranks) == len(NATIVE_SCALE_METHODS) * len(SCALE_VARIANTS) * len(
        SCALE_RANK_METRICS
    )
    gmr_root = slopes.loc[
        (slopes.method == "gmr")
        & (slopes.factor == "root")
        & (slopes.metric == "root_translation_common_scale_mean_m")
    ].iloc[0]
    assert gmr_root.central_difference_per_unit_multiplier == pytest.approx(1.0)
    assert set(ranks.variant) == set(SCALE_VARIANTS)

    incomplete = source.loc[
        ~(
            (source.method == "protomotions_v3")
            & (source.variant == "root_plus_5")
        )
    ]
    with pytest.raises(ValueError, match="protomotions_v3"):
        derive_scale_tables(incomplete)


def test_scale_collector_separates_fixed_contact_and_robustness_roles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = tmp_path / "metrics"
    outputs = tmp_path / "outputs"
    metrics.mkdir()
    outputs.mkdir()
    rows = []

    def add_row(role: str, method: str, variant: str, index: int) -> None:
        output = outputs / f"{role}__{method}__{variant}.npz"
        output.write_bytes(f"artifact-{index}".encode())
        row = {
            "experiment_role": role,
            "method": method,
            "variant": variant,
            "frames": 600,
            "completion_ratio": 1.0,
            "output_path": str(output.relative_to(tmp_path)),
            "output_sha256": sha256_file(output),
        }
        row.update({metric: 0.01 * (index + 1) for metric in QUALITY_COLUMNS})
        row.update(
            {
                metric: 1.0
                if metric == "completion_ratio"
                else 0.01 * (index + 1)
                for metric in SCALE_RESPONSE_METRIC_NAMES
            }
        )
        rows.append(row)

    index = 0
    for policy in TRANSPLANT_POLICIES:
        add_row("controlled_policy", "controlled_dense", policy, index)
        index += 1
    for method in NATIVE_SCALE_METHODS:
        for variant in SCALE_VARIANTS:
            add_row(FORMAL_NATIVE_SCALE_ROLE, method, variant, index)
            index += 1
    for variant in SCALE_VARIANTS:
        add_row(CONTACT_ROBUSTNESS_ROLE, "omniretarget", variant, index)
        index += 1
    pd.DataFrame(rows).to_csv(
        metrics / "scale_policy_sensitivity_summary.csv", index=False
    )
    pd.DataFrame(
        {
            "method": ["omniretarget"] * len(SCALE_VARIANTS),
            "variant": SCALE_VARIANTS,
            "recomputed_label_flip_rate": [0.0, 0.001, 0.0025, 0.0, 0.0],
            "native_recomputed_vs_formal_canonical_label_disagreement_rate": [
                0.2575
            ]
            * len(SCALE_VARIANTS),
        }
    ).to_csv(metrics / "contact_constraint_flip.csv", index=False)

    # This unit isolates role separation; the strict 30-row input ledger and
    # raw-metric reconstruction have dedicated fail-closed tests below.
    monkeypatch.setattr(
        "retargeting_comparison.stage1_publication.build_scale_publication_input_ledger",
        lambda root, summary: pd.DataFrame(),
    )
    monkeypatch.setattr(
        "retargeting_comparison.stage1_publication.verify_scale_publication_input_ledger",
        lambda root, recompute_metrics=True: True,
    )

    bundle = collect_scale_evidence(tmp_path)
    assert len(bundle.native_response) == 20
    assert len(bundle.contact_robustness) == 5
    assert len(bundle.contact_robustness_slopes) == 2 * len(
        SCALE_RESPONSE_METRIC_NAMES
    )
    assert set(bundle.summary.experiment_role) == {
        "controlled_policy",
        FORMAL_NATIVE_SCALE_ROLE,
        CONTACT_ROBUSTNESS_ROLE,
    }


def _scale_key_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "experiment_role": role,
                "method": method,
                "variant": variant,
            }
            for role, method, variant in sorted(_expected_scale_evidence_keys())
        ]
    )


def test_scale_input_ledger_missing_row_fails_before_summary_trust(
    tmp_path: Path,
) -> None:
    (tmp_path / "manifests").mkdir()
    (tmp_path / "metrics").mkdir()
    rows = _scale_key_frame()
    rows["schema_version"] = SCALE_INPUT_LEDGER_SCHEMA_VERSION
    rows["fixed_contact_causal_arm"] = rows.experiment_role.ne(
        CONTACT_ROBUSTNESS_ROLE
    )
    rows.iloc[:-1].to_csv(tmp_path / SCALE_INPUT_LEDGER, index=False)
    rows.to_csv(tmp_path / "metrics/stage1_scale_policy_summary.csv", index=False)

    with pytest.raises(ValueError, match=r"exactly 25\+5"):
        verify_scale_publication_input_ledger(
            tmp_path, recompute_metrics=False
        )


def test_scale_input_ledger_tampered_raw_hash_fails_closed(tmp_path: Path) -> None:
    (tmp_path / "manifests").mkdir()
    (tmp_path / "metrics").mkdir()
    evidence = tmp_path / "evidence.bin"
    evidence.write_bytes(b"hash-bound-scale-evidence")
    rows = _scale_key_frame()
    rows["schema_version"] = SCALE_INPUT_LEDGER_SCHEMA_VERSION
    rows["fixed_contact_causal_arm"] = rows.experiment_role.ne(
        CONTACT_ROBUSTNESS_ROLE
    )
    path_columns = (
        "output",
        "raw_per_frame_csv",
        "raw_per_frame_parquet",
        "raw_evaluator_summary",
        "target_manifest",
        "registered_target",
        "runtime_evidence",
        "config",
        "method_code",
        "environment_lock",
        "source",
        "evaluator_protocol",
        "evaluator_code",
        "scale_protocol",
        "provenance_table",
        "solver_asset",
        "evaluator_asset",
    )
    for prefix in path_columns:
        rows[f"{prefix}_path"] = "evidence.bin"
        rows[f"{prefix}_sha256"] = "0" * 64
    rows.to_csv(tmp_path / SCALE_INPUT_LEDGER, index=False)
    rows[["experiment_role", "method", "variant"]].to_csv(
        tmp_path / "metrics/stage1_scale_policy_summary.csv", index=False
    )

    with pytest.raises(ValueError, match="hash mismatch"):
        verify_scale_publication_input_ledger(
            tmp_path, recompute_metrics=False
        )


def test_protomotions_v3_native_scale_point_requires_byte_identical_formal_binding(
    tmp_path: Path,
) -> None:
    sequence_id = "pilot"
    output = tmp_path / "runs/scale/native/canonical_g1.npz"
    formal = (
        tmp_path
        / "runs"
        / sequence_id
        / "protomotions-v3-v2/attempt_001/canonical_g1.npz"
    )
    output.parent.mkdir(parents=True)
    formal.parent.mkdir(parents=True)
    (tmp_path / "manifests/runs").mkdir(parents=True)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/protomotions_v3.yaml").write_text(
        "device: cpu\n", encoding="utf-8"
    )
    (tmp_path / "configs/scale_policy_sensitivity.yaml").write_text(
        "variants: five\n", encoding="utf-8"
    )
    (tmp_path / "source").mkdir()
    canonical_source = tmp_path / "source/pilot.npz"
    canonical_source.write_bytes(b"canonical-source")
    (tmp_path / "manifests/pilot_sequence.yaml").write_text(
        f"sequence_id: {sequence_id}\n"
        "canonical_path: source/pilot.npz\n"
        f"cropped_sha256: {'2' * 64}\n",
        encoding="utf-8",
    )
    motion = _motion(
        {
            "method": "protomotions_v3",
            "runtime_pre_solver_capture": {
                "observed_during_solver_run": True,
                "target_keypoints_sha256": "1" * 64,
            },
            "implementation_hashes": {"worker": "3" * 64},
        }
    )
    motion.save(output, source_frame_count=EXPECTED_FRAMES)
    shutil.copy2(output, formal)
    formal_evidence = formal.parent / "formal/evidence.json"
    formal_evidence.parent.mkdir()
    formal_evidence.write_text('{"status":"succeeded"}\n', encoding="utf-8")
    RunManifest(
        run_id=f"{sequence_id}__protomotions-v3",
        method="protomotions-v3",
        status=RunStatus.SUCCEEDED,
        command=["formal"],
        environment="conda:egoallo",
        repo_commit="0" * 40,
        config_sha256=sha256_file(tmp_path / "configs/protomotions_v3.yaml"),
        device="cpu",
        exit_code=0,
        output_path=formal.relative_to(tmp_path).as_posix(),
        output_sha256=sha256_file(formal),
        source_sha256="2" * 64,
    ).save(
        tmp_path
        / "manifests/runs"
        / f"{sequence_id}__protomotions-v3.json"
    )
    binding = {
        "schema_version": 1,
        "binding_type": "byte_identical_published_formal_output_materialization",
        "solver_invoked_for_materialization": False,
        "scale_variant": "native",
        "root_scale_multiplier": 1.0,
        "local_scale_multiplier": 1.0,
        "native_scale_output_path": output.relative_to(tmp_path).as_posix(),
        "native_scale_output_sha256": sha256_file(output),
        "formal_output_path": formal.relative_to(tmp_path).as_posix(),
        "formal_output_sha256": sha256_file(formal),
        "formal_evidence_path": formal_evidence.relative_to(tmp_path).as_posix(),
        "formal_evidence_sha256": sha256_file(formal_evidence),
        "canonical_source_sha256": "2" * 64,
        "canonical_source_file_sha256": sha256_file(canonical_source),
        "config_sha256": sha256_file(tmp_path / "configs/protomotions_v3.yaml"),
        "scale_protocol_sha256": sha256_file(
            tmp_path / "configs/scale_policy_sensitivity.yaml"
        ),
        "implementation_hashes": {"worker": "3" * 64},
    }
    output.with_name("formal_output_binding.json").write_text(
        json.dumps(binding), encoding="utf-8"
    )

    evidence = _runtime_scale_evidence(
        tmp_path,
        FORMAL_NATIVE_SCALE_ROLE,
        "protomotions_v3",
        "native",
        output,
        motion,
        pd.Series(dtype=object),
    )
    assert evidence["runtime_boundary_observed"] is True
    assert evidence["formal_native_binding_sha256"] == sha256_file(
        output.with_name("formal_output_binding.json")
    )

    formal.write_bytes(b"tampered-formal-output")
    with pytest.raises(ValueError, match="not bound to the formal run"):
        _runtime_scale_evidence(
            tmp_path,
            FORMAL_NATIVE_SCALE_ROLE,
            "protomotions_v3",
            "native",
            output,
            motion,
            pd.Series(dtype=object),
        )


def _quality_rows(keys: list[str]) -> pd.DataFrame:
    rows = []
    for index, key in enumerate(keys, start=1):
        row = {
            "key": key,
            "display_name": key,
        }
        row.update({metric: index / 100.0 for metric in QUALITY_COLUMNS})
        rows.append(row)
    return pd.DataFrame(rows)


def test_reference_comparison_is_descriptive_and_never_accuracy() -> None:
    core = _quality_rows(["a", "b"])
    reference = _quality_rows(["unitree-reference"])
    comparison = build_reference_comparison(core, reference)
    assert len(comparison) == 2 * len(QUALITY_COLUMNS)
    assert comparison.interpretation.str.contains("not ground truth").all()
    row = comparison.loc[
        (comparison.key == "a") & (comparison.metric == QUALITY_COLUMNS[0])
    ].iloc[0]
    assert row.signed_method_minus_reference == pytest.approx(0.0)


def test_direct_reference_comparison_uses_same_g1_trajectories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeRobot:
        body_ids = {"root": 0, "left_toe": 1}

        def __init__(self, _path: Path) -> None:
            pass

        def semantic_positions(self, qpos: np.ndarray) -> dict[str, np.ndarray]:
            root = np.asarray(qpos[:3], dtype=float)
            return {
                "root": root,
                "left_toe": root + np.asarray([qpos[7], 0.0, 0.0]),
                "head": root + np.asarray([0.0, 0.0, 1.0]),
            }

    monkeypatch.setattr(
        "retargeting_comparison.stage1_publication.CanonicalRobotModel", FakeRobot
    )
    q_reference = np.zeros((2, 36), dtype=np.float64)
    q_reference[:, 3] = 1.0
    q_method = q_reference.copy()
    q_method[:, 0] = 0.1
    q_method[:, 7] = 0.2

    def save(path: Path, qpos: np.ndarray, method: str) -> None:
        CanonicalG1(
            qpos=qpos,
            fps=30.0,
            source_frame_idx=np.arange(2),
            valid=np.ones(2, dtype=bool),
            per_frame_solve_time_s=np.zeros(2),
            metadata={"method": method, "completion_status": "succeeded"},
        ).save(path, source_frame_count=2)

    paths: dict[str, Path] = {}
    reference_path = tmp_path / "reference.npz"
    save(reference_path, q_reference, "unitree-reference")
    paths[REFERENCE_METHOD.key] = reference_path
    for spec in CORE_METHODS:
        path = tmp_path / f"{spec.key}.npz"
        save(path, q_method, spec.metadata_methods[0])
        paths[spec.key] = path
    direct = build_direct_reference_comparison(tmp_path, paths)
    assert len(direct) == len(CORE_METHODS) * 11
    assert direct.comparison_type.eq(
        "direct_same_g1_frame_index_aligned_trajectory_disagreement"
    ).all()
    assert direct.timeline_alignment.eq(
        "frame_index_only_not_exact_timestamp"
    ).all()
    root = direct.loc[direct.metric.eq("direct_root_translation_mean_m")]
    assert np.allclose(root.method_value, 0.1)
    displacement = direct.loc[
        direct.metric.eq("direct_root_displacement_mean_m")
    ]
    assert np.allclose(displacement.method_value, 0.0)


def test_direct_reference_velocity_is_periodic_before_differencing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeRobot:
        body_ids = {"root": 0}

        def __init__(self, _path: Path) -> None:
            pass

        @staticmethod
        def semantic_positions(qpos: np.ndarray) -> dict[str, np.ndarray]:
            root = np.asarray(qpos[:3], dtype=float)
            return {"root": root, "head": root + np.asarray([0.0, 0.0, 1.0])}

    monkeypatch.setattr(
        "retargeting_comparison.stage1_publication.CanonicalRobotModel", FakeRobot
    )
    reference = np.zeros((3, 36), dtype=np.float64)
    reference[:, 3] = 1.0
    method = reference.copy()
    method[:, 7] = np.asarray([3.13, -3.13, -3.12])

    def save(path: Path, qpos: np.ndarray, method_name: str) -> None:
        CanonicalG1(
            qpos=qpos,
            fps=30.0,
            source_frame_idx=np.arange(3),
            valid=np.ones(3, dtype=bool),
            per_frame_solve_time_s=np.zeros(3),
            metadata={"method": method_name, "completion_status": "succeeded"},
        ).save(path, source_frame_count=3)

    paths: dict[str, Path] = {}
    reference_path = tmp_path / "reference.npz"
    save(reference_path, reference, "unitree-reference")
    paths[REFERENCE_METHOD.key] = reference_path
    for spec in CORE_METHODS:
        path = tmp_path / f"{spec.key}.npz"
        save(path, method, spec.metadata_methods[0])
        paths[spec.key] = path

    direct = build_direct_reference_comparison(tmp_path, paths)
    velocity = direct.loc[
        direct.metric.eq("direct_joint_velocity_rmse_rad_s"), "method_value"
    ]
    # The joint moves smoothly through +pi/-pi.  Treating the wrapped pose
    # disagreement as a Euclidean signal would yield a spurious ~2*pi jump.
    assert (velocity < 0.2).all()


def test_markdown_writer_enforces_historical_stop_and_slide_limit(tmp_path: Path) -> None:
    path = tmp_path / "report.md"
    _write_stage1_markdown(path, "# Report\n\nEvidence.")
    text = path.read_text(encoding="utf-8")
    assert text.splitlines()[-1] == HISTORICAL_STAGE1_STOP
    assert text.count(HISTORICAL_STAGE1_STOP) == 1
    slides = "# Deck\n\n" + "\n\n".join(
        f"## Slide {index} — Test" for index in range(1, 16)
    )
    assert presentation_slide_count(slides) == 15


def test_chart_writer_emits_source_svg_pdf_and_300dpi_png(tmp_path: Path) -> None:
    source = pd.DataFrame({"x": [0.0, 1.0], "y": [1.0, 0.0]})

    def renderer(plt, data):
        figure, axis = plt.subplots(figsize=(2.0, 1.5), constrained_layout=True)
        axis.plot(data.x, data.y)
        return figure

    paths = save_publication_chart(tmp_path, "smoke", source, renderer)
    assert [path.suffix for path in paths] == [".csv", ".svg", ".pdf", ".png"]
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
    assert pd.read_csv(paths[0]).equals(source)


def test_all_figures_and_reports_render_from_synthetic_complete_evidence(
    tmp_path: Path,
) -> None:
    core = _quality_rows([spec.key for spec in CORE_METHODS])
    core["display_name"] = [spec.display_name for spec in CORE_METHODS]
    reference = _quality_rows([REFERENCE_METHOD.key])
    reference["display_name"] = [REFERENCE_METHOD.display_name]
    evaluations = EvaluationBundle(
        core=core,
        reference=reference,
        diagnostics=pd.DataFrame(),
    )
    timing = pd.DataFrame(
        [
            {
                "key": spec.key,
                "display_name": spec.display_name,
                "end_to_end_rtf_median": 0.1 * (index + 1),
                "native_core_rtf_median": 0.08 * (index + 1),
                "measured_repetitions": 3,
                "timing_evidence_grade": "formal_repeated",
            }
            for index, spec in enumerate(CORE_METHODS)
        ]
    )
    sparse_pairs = pd.DataFrame(
        [
            {
                "pair": pair,
                "joint_angle_rms_mean_rad": 0.1,
                "joint_angle_rms_p95_rad": 0.2,
                "robot_rf_point_rms_mean_m": 0.01,
                "robot_rf_point_rms_p95_m": 0.02,
            }
            for pair in (
                "sparse-neutral__sparse-a",
                "sparse-neutral__sparse-b",
                "sparse-a__sparse-b",
            )
        ]
    )
    sparse_variance = pd.DataFrame(
        {
            "targeted_point_variance_m2": [0.001, 0.002],
            "untracked_point_variance_m2": [0.002, 0.003],
            "all_point_variance_m2": [0.002, 0.004],
        }
    )
    interaction = pd.DataFrame(
        [
            {
                "case": case,
                "variant": variant,
                "strict_contact_2cm_frame_rate": 0.8,
                "near_contact_5cm_frame_rate": 0.9,
                "proximity_10cm_frame_rate": 1.0,
                "penetration_frame_rate": 0.1 if variant == "no-hard" else 0.0,
                "foot_sticking_violation_frame_rate": 0.2 if variant == "no-hard" else 0.0,
                "end_to_end_rtf": 2.0,
            }
            for case in ("box", "climb")
            for variant in ("full", "no-hard")
        ]
    )
    summary = _scale_summary()
    summary["effective_root_xy_scale"] = 0.75
    slopes, ranks = derive_scale_tables(summary)
    transplants = summary.loc[summary.experiment_role == "controlled_policy"].copy()
    native = summary.loc[summary.experiment_role == FORMAL_NATIVE_SCALE_ROLE].copy()
    robustness_rows = []
    for variant in SCALE_VARIANTS:
        row = native.loc[
            (native.method == "omniretarget") & (native.variant == variant)
        ].iloc[0].copy()
        row["experiment_role"] = CONTACT_ROBUSTNESS_ROLE
        row["artifact_rate"] += 0.01
        robustness_rows.append(row)
    robustness = pd.DataFrame(robustness_rows)
    robustness_slopes = slopes.loc[slopes.method == "omniretarget"].copy()
    contact_table = native.loc[
        native.method == "omniretarget",
        [
            "variant",
            "rf_kpe_all_mean_m",
            "root_translation_common_scale_mean_m",
            "artifact_rate",
            "foot_skating_frame_rate",
        ],
    ].merge(
        robustness[
            [
                "variant",
                "rf_kpe_all_mean_m",
                "root_translation_common_scale_mean_m",
                "artifact_rate",
                "foot_skating_frame_rate",
            ]
        ],
        on="variant",
        suffixes=("_fixed_canonical", "_native_recomputed"),
    )
    contact_diagnostics = pd.DataFrame(
        {
            "method": ["omniretarget"] * len(SCALE_VARIANTS),
            "variant": list(SCALE_VARIANTS),
            "recomputed_label_flip_rate": [0.0, 0.001, 0.0025, 0.0, 0.0],
            "native_recomputed_vs_formal_canonical_label_disagreement_rate": [
                0.2575
            ]
            * len(SCALE_VARIANTS),
        }
    )
    scale = ScaleBundle(
        summary=summary,
        transplants=transplants,
        native_response=native,
        slopes=slopes,
        ranks=ranks,
        contact_robustness=contact_table,
        contact_robustness_slopes=robustness_slopes,
        contact_diagnostics=contact_diagnostics,
    )
    comparison = build_reference_comparison(core, reference)

    figure_paths = build_stage1_figures(
        tmp_path,
        evaluations,
        timing,
        sparse_pairs,
        interaction,
        scale,
        comparison,
    )
    report_paths = render_stage1_reports(
        tmp_path,
        evaluations,
        timing,
        sparse_pairs,
        sparse_variance,
        interaction,
        scale,
        comparison,
    )
    assert len(figure_paths) == 12 * 4
    assert len(report_paths) == len(REPORT_FILES)
    assert all(path.is_file() for path in figure_paths + report_paths)
    assert all(
        path.read_text(encoding="utf-8").splitlines()[-1]
        == HISTORICAL_STAGE1_STOP
        for path in report_paths
    )
    assert presentation_slide_count(
        (tmp_path / "PRESENTATION.md").read_text(encoding="utf-8")
    ) <= 15
    decision_report = (tmp_path / "GO_NO_GO.md").read_text(encoding="utf-8")
    assert decision_report.startswith("# Stage 1 Decision: PENDING")
    assert "does not self-certify" not in decision_report.lower()
    assert "Independent fail-closed validation has not yet bound" in decision_report
