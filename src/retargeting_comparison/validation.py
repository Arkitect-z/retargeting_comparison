"""Stage 1 acceptance checks and Full-LAFAN hard stop enforcement."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .calibration import load_evaluator_protocol
from .constants import FULL_LAFAN_STOP_MESSAGE, STAGE1_RUN_DIRECTORIES
from .io_utils import atomic_write_json, load_yaml, sha256_file
from .reporting import CORE_LABELS, REPORTS, _run_paths, artifact_manifest
from .schemas import CanonicalG1, RunManifest, RunStatus


def _core_motion_check(motion: CanonicalG1, source_frame_count: int) -> bool:
    """Return a JSON-native acceptance value for one canonical trajectory."""

    return bool(
        len(motion.qpos) == source_frame_count
        and motion.metadata.get("completion_status") == "succeeded"
        and np.isfinite(motion.qpos).all()
    )


def _run_hashes_check(root: Path, sequence_id: str) -> bool:
    for label, output in _run_paths(root).items():
        run_label = STAGE1_RUN_DIRECTORIES[label]
        manifest_path = root / "manifests" / "runs" / f"{sequence_id}__{run_label}.json"
        if not manifest_path.is_file() or not output.is_file():
            return False
        manifest = RunManifest.load(manifest_path)
        if manifest.output_sha256 != sha256_file(output):
            return False
    return True


def _smoke_check(root: Path, sequence_id: str) -> bool:
    labels = (
        "sparse-neutral-v3-smoke2",
        "dense-v3-smoke2",
        "gmr-smoke2",
        "omniretarget-smoke2",
    )
    for label in labels:
        output = root / "runs" / sequence_id / label / "canonical_g1.npz"
        manifest_path = root / "runs" / "manifests" / f"{sequence_id}__{label}.json"
        if not output.is_file() or not manifest_path.is_file():
            return False
        manifest = RunManifest.load(manifest_path)
        motion = CanonicalG1.load(output)
        if (
            manifest.status != RunStatus.INCOMPLETE
            or manifest.output_sha256 != sha256_file(output)
            or len(motion.qpos) != 2
            or not np.isfinite(motion.qpos).all()
        ):
            return False
    return True


def _body_models_check(root: Path) -> bool:
    path = root / "manifests" / "body_models.yaml"
    if not path.is_file():
        return False
    value = load_yaml(path)
    for name in ("smpl", "smplx"):
        model = value.get(name, {})
        asset = (root / str(model.get("path", ""))).resolve()
        if (
            model.get("finite_forward") is not True
            or not asset.is_file()
            or model.get("sha256") != sha256_file(asset)
        ):
            return False
    return value.get("original_smpl_pickle_used") is False


def _source_adapters_check(root: Path) -> bool:
    path = root / "metrics" / "source_adapter_errors.csv"
    manifest_path = root / "manifests" / "source_adapters.yaml"
    if not path.is_file() or not manifest_path.is_file():
        return False
    rows = pd.read_csv(path)
    if set(rows.method.astype(str)) != {"gmr", "omniretarget"}:
        return False
    required = ("frame_count_match", "fps_match", "left_right_match")
    for column in required:
        values = rows[column]
        normalized = values if values.dtype == bool else values.astype(str).str.lower().eq("true")
        if not bool(normalized.all()):
            return False
    numeric_columns = (
        "common_joints",
        "root_aligned_mpjpe_m",
        "max_joint_error_m",
        "bone_length_error_mean_m",
        "bone_length_error_max_m",
        "root_translation_error_mean_m",
        "root_translation_error_max_m",
        "yaw_error_mean_rad",
        "yaw_error_max_rad",
        "foot_contact_agreement",
    )
    numeric = rows[list(numeric_columns)].to_numpy()
    manifest = load_yaml(manifest_path)
    return bool(
        rows.status.astype(str).str.lower().eq("passed").all()
        and np.isfinite(numeric).all()
        and (rows.common_joints > 0).all()
        and manifest.get("schema_version") == 2
        and manifest.get("metrics_sha256") == sha256_file(path)
        and manifest.get("native_artifacts_committed_to_git") is False
    )


def _artifact_hashes_check(root: Path) -> bool:
    path = root / "manifests" / "artifacts.csv"
    if not path.is_file():
        return False
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return False
    for row in rows:
        # stage1_validation.json is rewritten by this validator before the
        # artifact manifest is regenerated, so checking its prior self-entry
        # would create a false recursive hash mismatch.
        if row["path"] == "manifests/stage1_validation.json":
            continue
        artifact = root / row["path"]
        if (
            not artifact.is_file()
            or artifact.stat().st_size != int(row["size_bytes"])
            or sha256_file(artifact) != row["sha256"]
        ):
            return False
    return True


def _rerun_visualization_check(root: Path, sequence_id: str) -> bool:
    path = root / "manifests" / "rerun_visualization.json"
    if not path.is_file():
        return False
    value = json.loads(path.read_text())
    output = Path(value.get("output", ""))
    if not output.is_absolute():
        output = root / output
    expected_methods = {
        "sparse-neutral",
        "dense",
        "gmr",
        "omniretarget",
        "sparse-a",
        "sparse-b",
    }
    if (
        value.get("schema_version") != 3
        or value.get("sequence_id") != sequence_id
        or set(value.get("methods", [])) != expected_methods
        or value.get("frames_logged") != 600
        or value.get("rendering") != "articulated_g1_visual_meshes"
        or value.get("robot_visual_asset_count") != 35
        or value.get("robot_instances_per_frame") != 17
        or tuple(int(part) for part in str(value.get("rerun_version", "0.0")).split(".")[:2])
        < (0, 34)
        or not output.is_file()
        or int(value.get("output_size_bytes", 0)) < 40_000_000
        or value.get("output_sha256") != sha256_file(output)
    ):
        return False
    assets = value.get("robot_visual_assets", {})
    if len(assets) != 35:
        return False
    for asset_path, expected_hash in assets.items():
        asset = root / asset_path
        if not asset.is_file() or sha256_file(asset) != expected_hash:
            return False
    for label in expected_methods:
        run = (
            root
            / "runs"
            / sequence_id
            / STAGE1_RUN_DIRECTORIES[label]
            / "canonical_g1.npz"
        )
        if not run.is_file() or value.get("method_outputs", {}).get(label) != sha256_file(run):
            return False
    return True


def _evaluator_protocol_check(root: Path, sequence: dict[str, Any]) -> bool:
    """Verify that every result uses one frozen, method-independent protocol."""

    path = root / "manifests" / "evaluator.yaml"
    if not path.is_file():
        return False
    try:
        protocol = load_evaluator_protocol(path)
    except (TypeError, ValueError):
        return False
    common_scale = float(protocol["scale"]["common_static_scale"])
    config = load_yaml(root / "configs" / "controlled_mink.yaml")
    common = config["common"]
    if not (
        protocol.get("source_sha256") == sequence.get("cropped_sha256")
        and protocol.get("full_lafan_authorized") is False
        and protocol["heading"].get("uses_bvh_root_quaternion") is False
        and np.isclose(common["position_scale_root_torso_legs"], common_scale)
        and np.isclose(common["position_scale_arms"], common_scale)
    ):
        return False
    protocol_sha256 = sha256_file(path)
    for label in CORE_LABELS:
        summary_path = root / "metrics" / "runs" / f"{label}_summary.json"
        if not summary_path.is_file():
            return False
        summary = json.loads(summary_path.read_text())
        if not (
            summary.get("evaluator_schema_version") == 2
            and summary.get("evaluator_protocol_sha256") == protocol_sha256
            and np.isclose(summary.get("common_static_scale", np.nan), common_scale)
        ):
            return False
    return True


def _scientific_evidence_check(root: Path) -> bool:
    """Check the disaggregated evidence required for a scientific Pilot."""

    required_files = (
        "metrics/root_scale_diagnostics.csv",
        "metrics/controlled_task_residuals.csv",
        "metrics/sparse_seed_variance.csv",
        "metrics/sparse_seed_variance_per_frame.csv",
        "figures/root_scale_policies.svg",
        "figures/root_error_decomposition.svg",
        "figures/artifact_components.svg",
    )
    if not all((root / name).is_file() for name in required_files):
        return False
    scales = pd.read_csv(root / "metrics" / "root_scale_diagnostics.csv")
    residuals = pd.read_csv(root / "metrics" / "controlled_task_residuals.csv")
    expected = set(CORE_LABELS)
    if set(scales.label.astype(str)) != expected:
        return False
    if set(residuals.label.astype(str)) != {
        "sparse-neutral",
        "sparse-a",
        "sparse-b",
        "dense",
    }:
        return False
    required_scale_columns = (
        "common_static_scale",
        "native_root_scale",
        "effective_root_xy_scale",
        "root_translation_common_scale_mean_m",
        "root_translation_native_scale_mean_m",
        "root_translation_scale_invariant_mean_m",
    )
    numeric = scales[list(required_scale_columns)].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        return False
    common_values = scales.common_static_scale.to_numpy(dtype=float)
    if not np.allclose(common_values, common_values[0], atol=1e-12, rtol=0.0):
        return False
    controlled = scales[scales.label.isin(("sparse-neutral", "sparse-a", "sparse-b", "dense"))]
    if not np.allclose(
        controlled.native_root_scale,
        controlled.common_static_scale,
        atol=1e-12,
        rtol=0.0,
    ):
        return False
    return bool(np.isfinite(residuals.select_dtypes(include=[np.number]).to_numpy()).all())


def _timing_protocol_check(root: Path) -> bool:
    sequence_id = _sequence_id(root)
    for label in CORE_LABELS:
        path = root / "runs" / sequence_id / STAGE1_RUN_DIRECTORIES[label] / "timing.json"
        if not path.is_file():
            return False
        timing = json.loads(path.read_text())
        protocol = timing.get("protocol", {})
        if not (
            protocol.get("cold_processes") == 1
            and protocol.get("warmup_runs") == 1
            and protocol.get("measured_warm_runs") == 3
            and protocol.get("threads") == 1
            and protocol.get("visualization") is False
            and len(timing.get("end_to_end_rtf_raw", [])) == 3
            and len(timing.get("native_core_rtf_raw", [])) == 3
        ):
            return False
    return True


def _sequence_id(root: Path) -> str:
    return str(load_yaml(root / "manifests" / "pilot_sequence.yaml")["sequence_id"])


def _stage0_evidence_check(root: Path) -> bool:
    matrix_path = root / "research" / "human_to_g1_method_matrix.csv"
    claims_path = root / "research" / "claims.csv"
    candidates_path = root / "metrics" / "conditional_candidates.csv"
    required_visuals = (
        "research/lineage_graph.svg",
        "research/lineage_graph.pdf",
        "research/lineage_graph.png",
        "research/lineage_graph_source.csv",
    )
    if not (
        matrix_path.is_file()
        and claims_path.is_file()
        and candidates_path.is_file()
        and all((root / path).is_file() for path in required_visuals)
    ):
        return False
    matrix = pd.read_csv(matrix_path)
    required_columns = {
        "name",
        "version_or_commit",
        "publication_year",
        "open_source_status",
        "source_representation",
        "target_robot",
        "g1_29dof_supported",
        "outputs_g1_reference_trajectory",
        "retargeting_family",
        "optimizer",
        "kinematics_backend",
        "collision_backend",
        "temporal_scope",
        "contact_handling",
        "object_or_terrain_support",
        "requires_training",
        "requires_rl_training",
        "requires_pre_retargeted_input",
        "experiment_status",
        "exclusion_reason",
        "official_source",
    }
    if not required_columns.issubset(matrix.columns) or len(matrix) < 15:
        return False
    indexed = matrix.set_index("name")
    if not {
        "ProtoMotions v2.3",
        "ProtoMotions v3",
        "PHC retargeter",
    }.issubset(indexed.index):
        return False
    if not (
        str(indexed.loc["ProtoMotions v2.3", "stage1_role"]) == "required"
        and str(indexed.loc["ProtoMotions v3", "stage1_role"]) == "required"
        and str(indexed.loc["PHC retargeter", "g1_29dof_supported"]) == "no"
    ):
        return False
    claims = pd.read_csv(claims_path)
    if len(claims) < 8 or not {"historical", "experimental", "scope"}.issubset(
        set(claims.claim_type.astype(str))
    ):
        return False
    candidates = pd.read_csv(candidates_path)
    if set(candidates.candidate.astype(str)) != {"ProtoMotions v3", "PHC"}:
        return False
    allowed = {"passed", "na"}
    return bool(
        candidates.status.astype(str).isin(allowed).all()
        and (candidates.elapsed_s.astype(float) <= candidates.gate_limit_s.astype(float)).all()
    )


def _test_evidence_check(root: Path) -> bool:
    path = root / "manifests" / "test_evidence.json"
    if not path.is_file():
        return False
    value = json.loads(path.read_text())
    capture = value.get("capture_suite", {})
    rerun = value.get("rerun_recording", {})
    rerun_manifest_path = root / "manifests" / "rerun_visualization.json"
    rerun_manifest = (
        json.loads(rerun_manifest_path.read_text())
        if rerun_manifest_path.is_file()
        else {}
    )
    return bool(
        value.get("schema_version") == 3
        and value.get("full_lafan_authorized") is False
        and capture.get("result") == "passed"
        and int(capture.get("passed", 0)) >= 35
        and rerun.get("result") == "verified"
        and int(rerun.get("frames", 0)) == 600
        and rerun.get("rendering") == "articulated_g1_visual_meshes"
        and int(rerun.get("visual_asset_count", 0)) == 35
        and int(rerun.get("methods", 0)) == 6
        and int(rerun.get("robot_instances_per_frame", 0)) == 17
        and int(rerun.get("manifest_schema_version", 0)) == 3
        and rerun.get("output_sha256") == rerun_manifest.get("output_sha256")
        and rerun.get("evaluator_protocol_sha256")
        == rerun_manifest.get("evaluator_protocol_sha256")
    )


def validate_stage1(repo_root: str | Path = ".") -> dict[str, Any]:
    root = Path(repo_root).resolve()
    checks: dict[str, bool] = {}
    stage = load_yaml(root / "configs" / "stage1.yaml")
    sequence = load_yaml(root / "manifests" / "pilot_sequence.yaml")
    checks["full_lafan_not_authorized"] = (
        stage.get("full_lafan_authorized") is False
        and sequence.get("full_lafan_authorized") is False
    )
    for label, path in _run_paths(root).items():
        key = f"core_{label}"
        try:
            motion = CanonicalG1.load(path)
            motion.validate(source_frame_count=int(sequence["num_frames"]))
            checks[key] = _core_motion_check(motion, int(sequence["num_frames"]))
        except Exception:
            checks[key] = False
    checks["core_output_hashes"] = _run_hashes_check(root, sequence["sequence_id"])
    checks["four_method_smoke_tests"] = _smoke_check(root, sequence["sequence_id"])
    checks["body_models_finite_and_hashed"] = _body_models_check(root)
    checks["source_adapters_passed"] = _source_adapters_check(root)
    checks["evaluator_protocol_frozen"] = _evaluator_protocol_check(root, sequence)
    checks["scientific_evidence_complete"] = _scientific_evidence_check(root)
    checks["timing_protocol_complete"] = _timing_protocol_check(root)
    checks["stage0_evidence_complete"] = _stage0_evidence_check(root)
    policy_audit = root / "research" / "OFFICIAL_SCALE_AND_PREPROCESSING_AUDIT.md"
    checks["official_preprocessing_policy_audit"] = bool(
        policy_audit.is_file()
        and policy_audit.read_text().rstrip().endswith(FULL_LAFAN_STOP_MESSAGE)
    )
    sensitivity_config_path = root / "configs" / "scale_policy_sensitivity.yaml"
    sensitivity_config = (
        load_yaml(sensitivity_config_path) if sensitivity_config_path.is_file() else {}
    )
    registered_variants = sensitivity_config.get("within_method_variants", [])
    checks["scale_sensitivity_protocol_frozen"] = bool(
        sensitivity_config.get("full_lafan_authorized") is False
        and sensitivity_config.get("amass_motion_runs_authorized") is False
        and len(registered_variants) == 5
        and sensitivity_config.get("acceptance", {}).get(
            "all_required_variants_mandatory"
        )
        is True
    )
    for method, directory in (
        ("protomotions_v2_3", "protomotions-v2-3"),
        ("protomotions_v3", "protomotions-v3"),
    ):
        path = (
            root
            / "runs"
            / str(sequence["sequence_id"])
            / directory
            / "canonical_g1.npz"
        )
        try:
            motion = CanonicalG1.load(path)
            motion.validate(source_frame_count=int(sequence["num_frames"]))
            checks[f"revised_required_{method}"] = _core_motion_check(
                motion, int(sequence["num_frames"])
            )
        except Exception:
            checks[f"revised_required_{method}"] = False
    revised_artifacts = {
        "pre_solver_target_manifest": root / "manifests" / "pre_solver_targets.csv",
        "scale_policy_sensitivity_results": root
        / "metrics"
        / "scale_policy_sensitivity_summary.csv",
        "actor_shape_target_probe": root / "metrics" / "actor_shape_target_probe.csv",
        "contact_constraint_flip": root / "metrics" / "contact_constraint_flip.csv",
        "method_rank_stability": root / "metrics" / "method_rank_stability.csv",
    }
    for name, path in revised_artifacts.items():
        checks[name] = path.is_file() and path.stat().st_size > 0
    checks["artifact_hashes_valid"] = _artifact_hashes_check(root)
    checks["rerun_visualization_complete"] = _rerun_visualization_check(
        root, sequence["sequence_id"]
    )
    checks["test_evidence_complete"] = _test_evidence_check(root)
    for document in (
        "docs/RERUN_VISUALIZATION.md",
        "docs/STAGE1_COMPLETION_AUDIT.md",
    ):
        path = root / document
        checks[f"document_{Path(document).name}"] = bool(
            path.is_file() and path.read_text().rstrip().endswith(FULL_LAFAN_STOP_MESSAGE)
        )
    for case in ("box", "climb"):
        for variant in ("full", "no-hard"):
            path = root / "runs" / "interaction_manifests" / f"interaction__{case}__{variant}.json"
            manifest = RunManifest.load(path) if path.is_file() else None
            output_path = Path(manifest.output_path) if manifest and manifest.output_path else None
            if output_path is not None and not output_path.is_absolute():
                output_path = root / output_path
            checks[f"interaction_{case}_{variant}"] = bool(
                manifest
                and manifest.status == RunStatus.SUCCEEDED
                and output_path
                and output_path.is_file()
            )
    numeric_files = (
        root / "metrics" / "core_summary.csv",
        root / "metrics" / "timing_raw.csv",
        root / "metrics" / "interaction_summary.csv",
    )
    metrics_valid = True
    for path in numeric_files:
        if not path.is_file():
            metrics_valid = False
            continue
        numeric = pd.read_csv(path).select_dtypes(include=[np.number]).to_numpy()
        metrics_valid &= bool(np.isfinite(numeric).all())
    checks["metrics_finite"] = metrics_valid
    for report in REPORTS:
        path = root / report
        checks[f"report_{report}"] = (
            path.is_file() and path.read_text().rstrip().endswith(FULL_LAFAN_STOP_MESSAGE)
        )
    presentation = root / "PRESENTATION.md"
    checks["presentation_at_most_15_sections"] = (
        presentation.is_file() and presentation.read_text().count("\n## ") <= 15
    )
    artifact_path = root / "manifests" / "artifacts.csv"
    checks["artifact_manifest_present"] = artifact_path.is_file()
    projection_path = root / "metrics" / "stage2_projection.json"
    projection = json.loads(projection_path.read_text()) if projection_path.is_file() else {}
    mandatory_complete = all(checks.values())
    if mandatory_complete and projection.get("within_wall_budget") and projection.get(
        "within_storage_budget"
    ):
        decision = "GO"
    elif mandatory_complete:
        decision = "GO WITH CHANGES"
    else:
        decision = "NO-GO"
    result = {
        "decision": decision,
        "checks": checks,
        "stage2_projection": projection,
        "hard_stop_message": FULL_LAFAN_STOP_MESSAGE,
    }
    atomic_write_json(root / "manifests" / "stage1_validation.json", result)
    artifact_manifest(root)
    return result
