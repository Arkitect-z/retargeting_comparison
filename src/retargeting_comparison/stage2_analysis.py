"""Fail-closed aggregation and review for frozen Stage-2 trajectories.

The module never launches or resumes Stage 2.  It reads an immutable plan,
frozen per-sequence manifests, and job manifests.  Only jobs whose manifests
say ``succeeded`` and whose provenance, bytes, and exact frame coverage verify
are evaluated.  Missing/failed jobs remain in a completion ledger; a claimed
success with an integrity defect aborts the analysis after writing that ledger.

Evidence strata are preserved throughout.  Controlled, native-public,
benchmark-port, and external-reference rows are never collapsed into a causal ranking.  The
external corpus is an untimed descriptive reference, not verified ground
truth or an upper bound.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .calibration import load_evaluator_protocol
from .evaluator import evaluate_motion, save_evaluation
from .io_utils import atomic_write_json, atomic_write_text, sha256_file
from .robot_model import CanonicalRobotModel, default_robot_scene
from .rotations import quaternion_wxyz_to_matrix, yaw_from_matrix
from .schemas import CanonicalG1, CanonicalHuman
from .stage2 import (
    SCHEMA_VERSION as STAGE2_SCHEMA_VERSION,
    SequenceSpec,
    inventory_sha256,
    job_is_complete,
    verify_plan_hash,
)


ANALYSIS_SCHEMA_VERSION = 2
ANALYSIS_ATTEMPT_SCHEMA_VERSION = 1
ANALYSIS_PUBLICATION_RESERVE_BYTES = 1_048_576
BOOTSTRAP_SEED = 20260722
BOOTSTRAP_RESAMPLES = 2000
ALLOWED_STRATA = (
    "controlled_common_per_sequence_scale",
    "native_public_pipeline",
    "benchmark_public_retargeter_port",
    "external_reference",
)
EXTERNAL_STRATUM = "external_reference"

ANALYSIS_METRICS = (
    "rf_kpe_all_mean_m",
    "rf_kpe_targeted_mean_m",
    "rf_kpe_untracked_mean_m",
    "bone_direction_mean_rad",
    "bend_plane_mean_rad",
    "root_translation_common_scale_mean_m",
    "root_translation_scale_invariant_mean_m",
    "root_yaw_mean_rad",
    "joint_velocity_rms_mean_rad_s",
    "joint_acceleration_rms_mean_rad_s2",
    "joint_jerk_rms_p95_rad_s3",
    "pose_jump_p95_m",
    "foot_skating_frame_rate",
    "ground_penetration_frame_rate",
    "joint_limit_violation_frame_rate",
    "invalid_frame_rate",
    "artifact_rate",
)

UNCERTAINTY_METRICS = (
    "rf_kpe_all_mean_m",
    "rf_kpe_targeted_mean_m",
    "rf_kpe_untracked_mean_m",
    "root_translation_common_scale_mean_m",
    "root_translation_scale_invariant_mean_m",
    "root_yaw_mean_rad",
    "joint_jerk_rms_p95_rad_s3",
    "pose_jump_p95_m",
    "foot_skating_frame_rate",
    "ground_penetration_frame_rate",
    "artifact_rate",
)


class Stage2AnalysisError(RuntimeError):
    """Raised when frozen Stage-2 evidence fails an analysis integrity gate."""


@dataclass(frozen=True)
class VerifiedJob:
    job_id: str
    method: str
    sequence_id: str
    stratum: str
    job_manifest_path: Path
    sequence_manifest_path: Path
    output_path: Path
    production_wall_s: float


@dataclass
class LedgerBundle:
    ledger: pd.DataFrame
    completion: pd.DataFrame
    method_scope: pd.DataFrame
    verified: list[VerifiedJob]


@dataclass
class AnalysisTables:
    per_sequence_method: pd.DataFrame
    per_method_metric: pd.DataFrame
    within_stratum_pairs: pd.DataFrame
    within_stratum_pair_summary: pd.DataFrame
    reference_intersection: pd.DataFrame
    reference_summary: pd.DataFrame
    reference_direct_trajectory: pd.DataFrame = field(default_factory=pd.DataFrame)
    reference_direct_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    leave_one_subject_out: pd.DataFrame = field(default_factory=pd.DataFrame)
    calibration_per_sequence: pd.DataFrame = field(default_factory=pd.DataFrame)
    calibration_variability: pd.DataFrame = field(default_factory=pd.DataFrame)
    scale_sensitivity: pd.DataFrame = field(default_factory=pd.DataFrame)
    conclusion_ledger: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass(frozen=True)
class BudgetEvidence:
    execution_attempt_wall_s: float
    prior_analysis_attempt_wall_s: float
    current_analysis_attempt_wall_s: float
    total_accounted_wall_s: float
    raw_input_baseline_bytes: int
    retained_tree_bytes: int
    total_accounted_retained_bytes: int
    wall_limit_s: float
    storage_limit_bytes: int
    wall_compliant: bool
    storage_compliant: bool
    execution_attempt_count: int
    finalized_analysis_attempt_count: int
    evidence_sha256: str


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _portable(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path.resolve())


def _float_or_nan(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def _subject_id(value: str) -> str:
    match = re.search(r"(?:^|[_-])(subject\d+)(?:[_-]|\.|$)", value, re.I)
    return match.group(1).lower() if match else value.lower()


def _write_frame(frame: pd.DataFrame, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(stem.with_suffix(".csv"), index=False)
    frame.to_parquet(stem.with_suffix(".parquet"), index=False)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _unique_tree_bytes(*trees: Path) -> int:
    """Return exact retained bytes without double-counting nested roots."""

    files: dict[Path, int] = {}
    for tree in trees:
        if not tree.exists():
            continue
        if tree.is_file():
            files[tree.resolve()] = tree.stat().st_size
            continue
        for path in tree.rglob("*"):
            if path.is_file():
                files[path.resolve()] = path.stat().st_size
    return int(sum(files.values()))


def _raw_input_baseline_bytes(plan: Mapping[str, Any]) -> int:
    """Charge every selected raw BVH and reference CSV exactly once.

    Raw inputs live outside the run tree, but the registered storage policy
    retains them.  Omitting this baseline would let analysis retries appear to
    fit by ignoring the largest immutable inputs.
    """

    total = 0
    seen_source: set[tuple[str, str]] = set()
    seen_reference: set[tuple[str, str]] = set()
    for value in plan.get("selected_sequences", []):
        source_key = (
            str(value.get("relative_path", "")),
            str(value.get("source_sha256", "")),
        )
        if not source_key[0] or len(source_key[1]) != 64:
            raise Stage2AnalysisError("Selected source lacks immutable raw-input identity")
        if source_key not in seen_source:
            size = int(value.get("source_size_bytes", -1))
            if size < 0:
                raise Stage2AnalysisError("Selected source lacks a valid byte size")
            total += size
            seen_source.add(source_key)
        reference_path = value.get("reference_relative_path")
        if reference_path is not None:
            reference_key = (
                str(reference_path),
                str(value.get("reference_sha256", "")),
            )
            if len(reference_key[1]) != 64:
                raise Stage2AnalysisError(
                    "Reference intersection lacks immutable raw-input identity"
                )
            if reference_key not in seen_reference:
                size = int(value.get("reference_size_bytes", -1))
                if size < 0:
                    raise Stage2AnalysisError("Reference input lacks a valid byte size")
                total += size
                seen_reference.add(reference_key)
    return total


def _execution_attempt_evidence(
    run_root: Path, *, expected_plan_sha256: str
) -> tuple[float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    directory = run_root / "execution_attempts"
    for path in sorted(directory.glob("attempt-*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            wall = float(value["active_wall_s"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise Stage2AnalysisError(
                f"Execution-attempt accounting is unreadable: {path}"
            ) from error
        payload = dict(value)
        recorded_payload = payload.pop("payload_sha256", None)
        if (
            value.get("schema_version") != STAGE2_SCHEMA_VERSION
            or recorded_payload != _canonical_sha256(payload)
            or value.get("plan_sha256") != expected_plan_sha256
        ):
            raise Stage2AnalysisError(
                f"Execution attempt payload/plan contract failed: {path}"
            )
        if not np.isfinite(wall) or wall < 0.0 or value.get("status") == "running":
            raise Stage2AnalysisError(
                f"Execution attempt is not finalized with finite wall time: {path}"
            )
        rows.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "status": str(value.get("status")),
                "active_wall_s": wall,
            }
        )
    return float(sum(row["active_wall_s"] for row in rows)), rows


def _analysis_attempt_evidence(
    analysis_root: Path,
    *,
    expected_plan_sha256: str,
    current_attempt_path: Path | None = None,
) -> tuple[float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    directory = analysis_root / "analysis_attempts"
    for path in sorted(directory.glob("attempt-*.json")):
        if current_attempt_path is not None and path.resolve() == current_attempt_path.resolve():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            wall = float(value["active_wall_s"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise Stage2AnalysisError(
                f"Analysis-attempt accounting is unreadable: {path}"
            ) from error
        payload = dict(value)
        recorded_payload = payload.pop("payload_sha256", None)
        if (
            value.get("schema_version") != ANALYSIS_ATTEMPT_SCHEMA_VERSION
            or recorded_payload != _canonical_sha256(payload)
            or value.get("plan_sha256") != expected_plan_sha256
        ):
            raise Stage2AnalysisError(
                f"Analysis attempt payload/plan contract failed: {path}"
            )
        if not np.isfinite(wall) or wall < 0.0 or value.get("status") == "running":
            raise Stage2AnalysisError(
                f"Analysis attempt is not finalized with finite wall time: {path}"
            )
        rows.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "status": str(value.get("status")),
                "active_wall_s": wall,
            }
        )
    return float(sum(row["active_wall_s"] for row in rows)), rows


def recompute_stage2_budget_evidence(
    plan: Mapping[str, Any],
    run_root: str | Path,
    analysis_root: str | Path,
    *,
    current_analysis_wall_s: float = 0.0,
    current_attempt_path: Path | None = None,
) -> BudgetEvidence:
    """Independently charge cumulative execution, analysis, and retained bytes."""

    run = Path(run_root).resolve()
    analysis = Path(analysis_root).resolve()
    expected_plan_sha256 = str(plan["plan_sha256"])
    execution_wall, execution_rows = _execution_attempt_evidence(
        run, expected_plan_sha256=expected_plan_sha256
    )
    prior_analysis_wall, analysis_rows = _analysis_attempt_evidence(
        analysis,
        expected_plan_sha256=expected_plan_sha256,
        current_attempt_path=current_attempt_path,
    )
    current = float(current_analysis_wall_s)
    if not np.isfinite(current) or current < 0.0:
        raise Stage2AnalysisError("Current analysis wall time is invalid")
    raw_bytes = _raw_input_baseline_bytes(plan)
    tree_bytes = _unique_tree_bytes(run, analysis)
    total_bytes = tree_bytes + raw_bytes
    projection = plan.get("projection", {})
    wall_limit_s = float(projection.get("wall_limit_hours", np.nan)) * 3600.0
    storage_limit_bytes = int(
        float(projection.get("storage_limit_gb", np.nan)) * 1_000_000_000.0
    )
    if not np.isfinite(wall_limit_s) or wall_limit_s <= 0.0:
        raise Stage2AnalysisError("Plan lacks a finite positive wall-time limit")
    if storage_limit_bytes <= 0:
        raise Stage2AnalysisError("Plan lacks a finite positive storage limit")
    total_wall = execution_wall + prior_analysis_wall + current
    basis = {
        "execution_attempts": execution_rows,
        "analysis_attempts": analysis_rows,
        "current_analysis_wall_s": current,
        "raw_input_baseline_bytes": raw_bytes,
        "retained_tree_bytes": tree_bytes,
        "total_accounted_retained_bytes": total_bytes,
        "wall_limit_s": wall_limit_s,
        "storage_limit_bytes": storage_limit_bytes,
    }
    return BudgetEvidence(
        execution_attempt_wall_s=execution_wall,
        prior_analysis_attempt_wall_s=prior_analysis_wall,
        current_analysis_attempt_wall_s=current,
        total_accounted_wall_s=total_wall,
        raw_input_baseline_bytes=raw_bytes,
        retained_tree_bytes=tree_bytes,
        total_accounted_retained_bytes=total_bytes,
        wall_limit_s=wall_limit_s,
        storage_limit_bytes=storage_limit_bytes,
        wall_compliant=total_wall <= wall_limit_s,
        storage_compliant=total_bytes <= storage_limit_bytes,
        execution_attempt_count=len(execution_rows),
        finalized_analysis_attempt_count=len(analysis_rows),
        evidence_sha256=_canonical_sha256(basis),
    )


def _write_analysis_attempt(path: Path, value: Mapping[str, Any]) -> None:
    payload = dict(value)
    payload.pop("payload_sha256", None)
    payload["payload_sha256"] = _canonical_sha256(payload)
    atomic_write_json(path, payload)


def _begin_analysis_attempt(
    analysis_root: Path,
    plan: Mapping[str, Any],
    *,
    attempt_kind: str = "analysis_build",
) -> tuple[Path, Path, dict[str, Any]]:
    """Conservatively close a crashed analysis and open a new atomic attempt."""

    directory = analysis_root / "analysis_attempts"
    directory.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    existing = sorted(directory.glob("attempt-*.json"))
    for path in existing:
        value = json.loads(path.read_text(encoding="utf-8"))
        payload = dict(value)
        recorded_payload = payload.pop("payload_sha256", None)
        if (
            value.get("schema_version") != ANALYSIS_ATTEMPT_SCHEMA_VERSION
            or recorded_payload != _canonical_sha256(payload)
            or value.get("plan_sha256") != plan.get("plan_sha256")
        ):
            raise Stage2AnalysisError(
                f"Existing analysis-attempt payload/plan contract failed: {path}"
            )
        if value.get("status") != "running":
            continue
        started = _parse_utc(str(value["started_at_utc"]))
        wall = max(0.0, (now - started).total_seconds())
        value.update(
            {
                "status": "abandoned_conservatively_accounted",
                "finished_at_utc": now.isoformat(),
                "active_wall_s": wall,
                "published": False,
                "accounting_note": (
                    "prior analysis did not finalize; elapsed UTC interval was "
                    "charged at the next attempt"
                ),
            }
        )
        _write_analysis_attempt(path, value)
    count = len(existing)
    identifier = f"attempt-{count + 1:03d}"
    path = directory / f"{identifier}.json"
    staging = directory / f"{identifier}-staging"
    staging.mkdir(parents=True, exist_ok=False)
    value = {
        "schema_version": ANALYSIS_ATTEMPT_SCHEMA_VERSION,
        "attempt_id": identifier,
        "attempt_kind": attempt_kind,
        "status": "running",
        "published": False,
        "started_at_utc": now.isoformat(),
        "plan_sha256": str(plan["plan_sha256"]),
        "design_id": str(plan["design_id"]),
        "staging_path": str(staging),
    }
    _write_analysis_attempt(path, value)
    return path, staging, value


def _finalize_analysis_attempt(
    path: Path,
    value: Mapping[str, Any],
    *,
    started_monotonic: float,
    status: str,
    published: bool,
    error: str | None,
    publication_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    finished = dict(value)
    finished.update(
        {
            "status": status,
            "published": published,
            "finished_at_utc": _utc_now(),
            "active_wall_s": max(0.0, time.perf_counter() - started_monotonic),
            "error": error,
            "publication_manifest_sha256": publication_manifest_sha256,
        }
    )
    _write_analysis_attempt(path, finished)
    return finished


def _enforce_live_analysis_budget(
    plan: Mapping[str, Any],
    run_root: Path,
    analysis_root: Path,
    *,
    attempt_path: Path,
    attempt_started_monotonic: float,
    publication_reserve_bytes: int = ANALYSIS_PUBLICATION_RESERVE_BYTES,
) -> BudgetEvidence:
    evidence = recompute_stage2_budget_evidence(
        plan,
        run_root,
        analysis_root,
        current_analysis_wall_s=time.perf_counter() - attempt_started_monotonic,
        current_attempt_path=attempt_path,
    )
    if not evidence.wall_compliant:
        raise Stage2AnalysisError(
            "Cumulative Stage-2 wall-time budget was exhausted during analysis"
        )
    if (
        evidence.total_accounted_retained_bytes + int(publication_reserve_bytes)
        > evidence.storage_limit_bytes
    ):
        raise Stage2AnalysisError(
            "Cumulative Stage-2 retained-storage budget was exhausted during analysis"
        )
    if plan.get("projection", {}).get("jobs") and evidence.execution_attempt_count < 1:
        raise Stage2AnalysisError(
            "Stage-2 outputs lack a payload-hashed execution-attempt ledger"
        )
    return evidence


def _publish_staging_tree(staging: Path, destination: Path) -> list[Path]:
    """Atomically replace files, leaving the publication manifest for last."""

    source_files = sorted(
        (path for path in staging.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(staging).as_posix(),
    )
    published: list[Path] = []
    for source in source_files:
        relative = source.relative_to(staging)
        if relative.as_posix() == "analysis_manifest.json":
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, target)
        published.append(target)
    return published


def load_stage2_analysis_plan(
    repo_root: str | Path, plan_path: str | Path
) -> tuple[dict[str, Any], Path, Path]:
    """Load and structurally validate an immutable Stage-2 plan."""

    root = Path(repo_root).resolve()
    path = _resolve(root, plan_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Stage-2 plan is missing: {path}")
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Stage2AnalysisError(f"Cannot read Stage-2 plan: {path}") from error
    if (
        plan.get("stage") != 2
        or int(plan.get("schema_version", 0)) != STAGE2_SCHEMA_VERSION
    ):
        raise Stage2AnalysisError("Unsupported Stage-2 plan schema")
    if not verify_plan_hash(plan):
        raise Stage2AnalysisError("Stage-2 plan SHA-256 is invalid")
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
    if "plan_basis_sha256" not in plan:
        raise Stage2AnalysisError("Stage-2 plan lacks a decision-basis SHA-256")
    basis = {key: plan.get(key) for key in basis_fields}
    if _canonical_sha256(basis) != str(plan["plan_basis_sha256"]):
        raise Stage2AnalysisError("Stage-2 plan decision-basis SHA-256 is invalid")
    if (
        plan.get("evidence_strata", {}).get("cross_stratum_ranking_forbidden")
        is not True
    ):
        raise Stage2AnalysisError("Plan does not forbid cross-stratum ranking")
    jobs = list(plan.get("projection", {}).get("jobs", []))
    if not jobs:
        raise Stage2AnalysisError("Stage-2 plan contains no jobs")
    identifiers = [str(job.get("job_id", "")) for job in jobs]
    if any(not value for value in identifiers) or len(set(identifiers)) != len(
        identifiers
    ):
        raise Stage2AnalysisError("Stage-2 job IDs are empty or duplicated")
    selected = list(plan.get("selected_sequences", []))
    sequence_ids = [str(value.get("sequence_id", "")) for value in selected]
    if any(not value for value in sequence_ids) or len(set(sequence_ids)) != len(
        sequence_ids
    ):
        raise Stage2AnalysisError("Selected sequence IDs are empty or duplicated")
    selected_set = set(sequence_ids)
    selected_by_id = {
        str(value["sequence_id"]): dict(value) for value in selected
    }
    if plan.get("selected_inventory_sha256") is not None:
        try:
            selected_digest = inventory_sha256(
                [SequenceSpec.from_dict(value) for value in selected]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise Stage2AnalysisError("Selected inventory contract is invalid") from error
        if selected_digest != str(plan["selected_inventory_sha256"]):
            raise Stage2AnalysisError("Selected inventory SHA-256 is invalid")
    method_specs = {
        str(value["method"]): dict(value) for value in plan.get("method_specs", [])
    }
    method_strata = {
        method: str(value["stratum"]) for method, value in method_specs.items()
    }
    for method, stratum in method_strata.items():
        if stratum not in ALLOWED_STRATA:
            raise Stage2AnalysisError(f"Unknown stratum for {method}: {stratum}")
    for job in jobs:
        method = str(job.get("method", ""))
        sequence = str(job.get("sequence_id", ""))
        stratum = str(job.get("stratum", ""))
        if method not in method_strata or method_strata[method] != stratum:
            raise Stage2AnalysisError(f"Job stratum disagrees with method spec: {job}")
        method_spec = method_specs[method]
        if (
            str(job.get("method_contract_sha256"))
            != str(method_spec.get("method_contract_sha256"))
            or str(job.get("policy_sha256"))
            != str(method_spec.get("policy_sha256"))
        ):
            raise Stage2AnalysisError(
                f"Job method/policy contract disagrees with method spec: {job}"
            )
        if sequence not in selected_set:
            raise Stage2AnalysisError(f"Job references an unselected sequence: {job}")
        expected_job_contract = {
            "job_id": str(job["job_id"]),
            "method": method,
            "sequence": selected_by_id[sequence],
            "stratum": stratum,
            "method_contract_sha256": str(job.get("method_contract_sha256")),
            "policy_sha256": str(job.get("policy_sha256")),
            "reference_relative_path": job.get("reference_relative_path"),
        }
        if _canonical_sha256(expected_job_contract) != str(
            job.get("job_contract_sha256")
        ):
            raise Stage2AnalysisError(
                f"Scheduled job contract SHA-256 is invalid: {job.get('job_id')}"
            )
        if stratum == EXTERNAL_STRATUM and job.get("reference_relative_path") is None:
            raise Stage2AnalysisError(
                "External-reference job lacks intersection provenance"
            )
        if (
            stratum != EXTERNAL_STRATUM
            and job.get("reference_relative_path") is not None
        ):
            raise Stage2AnalysisError(
                "Only external-reference jobs may receive reference trajectory paths"
            )
        for digest_name in (
            "method_contract_sha256",
            "policy_sha256",
            "job_contract_sha256",
        ):
            digest = job.get(digest_name)
            if digest is not None and (not isinstance(digest, str) or len(digest) != 64):
                raise Stage2AnalysisError(
                    f"Job has invalid {digest_name}: {job.get('job_id')}"
                )
    projection = dict(plan.get("projection", {}))
    for limit_name in ("wall_limit_hours", "storage_limit_gb"):
        if limit_name in projection:
            limit = float(projection[limit_name])
            if not np.isfinite(limit) or limit <= 0.0:
                raise Stage2AnalysisError(f"Plan has invalid {limit_name}")
    if any(str(job.get("stratum")) == EXTERNAL_STRATUM for job in jobs):
        if not isinstance(plan.get("unitree_reference_contract"), Mapping):
            raise Stage2AnalysisError(
                "External-reference jobs lack a frozen Unitree asset/evaluator contract"
            )

    default_run_root = root / "runs" / "stage2" / str(plan["design_id"])
    candidates = [path.parent, default_run_root]
    run_root = next(
        (
            candidate
            for candidate in candidates
            if (candidate / "job_manifests").is_dir()
            or (candidate / "sources").is_dir()
        ),
        path.parent,
    )
    return plan, path, run_root.resolve()


def verify_unitree_reference_evaluator_contract(
    repo_root: str | Path, plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Independently re-hash and re-run the Unitree↔canonical G1 audits."""

    root = Path(repo_root).resolve()
    external_jobs = [
        job
        for job in plan.get("projection", {}).get("jobs", [])
        if job.get("stratum") == EXTERNAL_STRATUM
    ]
    if not external_jobs:
        return {"required": False, "verified": True}
    contract = dict(plan.get("unitree_reference_contract", {}))
    recorded_digest = contract.pop("contract_sha256", None)
    if not isinstance(recorded_digest, str) or recorded_digest != _canonical_sha256(
        contract
    ):
        raise Stage2AnalysisError("Unitree reference contract digest is invalid")

    receipts: list[Mapping[str, Any]] = []
    for key in ("manifest", "config", "adapter"):
        receipts.append(dict(contract.get(key, {})))
    receipts.extend(
        dict(value) for value in contract.get("pinned_resources", {}).values()
    )
    receipts.extend(dict(value) for value in contract.get("asset_binding", {}).values())
    for receipt in receipts:
        path = _resolve(root, str(receipt.get("path", "")))
        if (
            not path.is_file()
            or sha256_file(path) != str(receipt.get("sha256"))
            or path.stat().st_size != int(receipt.get("bytes", -1))
        ):
            raise Stage2AnalysisError(f"Unitree bound file changed: {path}")

    manifest_path = _resolve(root, str(contract["manifest"]["path"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 2
        or manifest.get("role") != "external_reference_not_verified_ground_truth"
        or manifest.get("verified_official_ground_truth") is not False
        or manifest.get("eligible_for_runtime_comparison") is not False
        or contract.get("alignment_contract")
        != "same_basename_and_frame_indices_only"
        or contract.get("byte_identical_human_source_verified") is not False
        or contract.get("exact_timestamp_identity_claimed") is not False
    ):
        raise Stage2AnalysisError("Unitree evidence-role/alignment contract changed")
    if str(manifest.get("upstream_revision")) != str(contract.get("revision")):
        raise Stage2AnalysisError("Unitree manifest revision differs from the plan")
    source_binding = dict(manifest.get("source_binding", {}))
    if (
        source_binding.get("alignment_basis")
        != "same basename and frame indices 0:600"
        or source_binding.get("byte_identical_human_source_verified") is not False
        or source_binding.get("exact_timestamp_identity_claimed") is not False
    ):
        raise Stage2AnalysisError("Unitree source binding overclaims input/timeline identity")
    manifest_pinned = dict(manifest.get("pinned_resources", {}))
    for name, receipt in contract.get("pinned_resources", {}).items():
        value = dict(manifest_pinned.get(name, {}))
        if (
            str(value.get("path")) != str(receipt.get("path"))
            or str(value.get("sha256")) != str(receipt.get("sha256"))
            or int(value.get("size_bytes", -1)) != int(receipt.get("bytes", -2))
        ):
            raise Stage2AnalysisError(
                f"Unitree manifest/plan pinned-resource mismatch: {name}"
            )
    revision = str(contract["revision"])
    csv_files = sorted(
        (root / "data/external/unitree_lafan1_reference" / revision / "g1").glob(
            "*.csv"
        )
    )
    inventory_digest = hashlib.sha256()
    for path in csv_files:
        inventory_digest.update(path.name.encode("utf-8"))
        inventory_digest.update(bytes.fromhex(sha256_file(path)))
    inventory = dict(contract["g1_csv_inventory"])
    if (
        len(csv_files) != int(inventory.get("files", -1))
        or inventory_digest.hexdigest() != str(inventory.get("aggregate_sha256"))
    ):
        raise Stage2AnalysisError("Unitree CSV inventory changed after planning")
    if dict(manifest.get("g1_csv_inventory", {})) != inventory:
        raise Stage2AnalysisError("Unitree manifest/plan CSV inventory mismatch")

    assets = contract["asset_binding"]
    canonical_urdf = _resolve(
        root, str(assets["canonical_evaluator_urdf_path"]["path"])
    )
    canonical_scene = _resolve(
        root, str(assets["canonical_evaluator_scene_path"]["path"])
    )
    evaluator_path = _resolve(root, str(assets["evaluator_manifest_path"]["path"]))
    reference_urdf = _resolve(
        root, str(contract["pinned_resources"]["g1_urdf"]["path"])
    )
    manifest_assets = dict(manifest.get("asset_binding", {}))
    asset_manifest_keys = {
        "canonical_evaluator_urdf_path": "canonical_evaluator_urdf_sha256",
        "canonical_evaluator_scene_path": "canonical_evaluator_scene_sha256",
        "evaluator_manifest_path": "evaluator_manifest_sha256",
    }
    for key, hash_key in asset_manifest_keys.items():
        receipt = dict(assets[key])
        if (
            str(manifest_assets.get(key)) != str(receipt.get("path"))
            or str(manifest_assets.get(hash_key)) != str(receipt.get("sha256"))
        ):
            raise Stage2AnalysisError(
                f"Unitree manifest/plan evaluator-asset mismatch: {key}"
            )
    manifest_adapter = dict(manifest.get("adapter", {}))
    if (
        str(manifest_adapter.get("implementation_path"))
        != str(contract["adapter"].get("path"))
        or str(manifest_adapter.get("implementation_sha256"))
        != str(contract["adapter"].get("sha256"))
    ):
        raise Stage2AnalysisError("Unitree manifest/plan adapter mismatch")
    evaluator = load_evaluator_protocol(evaluator_path)
    if evaluator.get("robot_xml_sha256") != sha256_file(canonical_scene):
        raise Stage2AnalysisError(
            "Canonical evaluator manifest is not bound to the audited G1 scene"
        )
    from .unitree_reference import (
        compare_urdf_kinematics,
        compare_urdf_to_mujoco_fk,
    )

    urdf_audit = compare_urdf_kinematics(reference_urdf, canonical_urdf)
    expected_urdf = dict(contract["urdf_kinematic_audit"])
    if urdf_audit != expected_urdf or urdf_audit.get(
        "kinematic_contract_equivalent"
    ) is not True:
        raise Stage2AnalysisError("Reference/canonical URDF equivalence changed")
    expected_fk = dict(contract["urdf_to_mujoco_fk_audit"])
    fk_audit = compare_urdf_to_mujoco_fk(
        reference_urdf,
        canonical_scene,
        random_sample_count=int(expected_fk["random_sample_count"]),
        seed=int(expected_fk["seed"]),
        position_tolerance_m=float(expected_fk["position_tolerance_m"]),
        rotation_tolerance_rad=float(expected_fk["rotation_tolerance_rad"]),
    )
    if fk_audit != expected_fk or fk_audit.get("kinematic_fk_equivalent") is not True:
        raise Stage2AnalysisError("Reference URDF/canonical evaluator FK audit changed")
    return {
        "required": True,
        "verified": True,
        "contract_sha256": recorded_digest,
        "manifest_sha256": sha256_file(manifest_path),
        "reference_urdf_sha256": sha256_file(reference_urdf),
        "canonical_urdf_sha256": sha256_file(canonical_urdf),
        "canonical_scene_sha256": sha256_file(canonical_scene),
        "evaluator_manifest_sha256": sha256_file(evaluator_path),
        "pairing": "normalized basename and integer frame index only",
    }


def _attempt_wall_time(manifest: Mapping[str, Any]) -> float:
    attempts = manifest.get("attempts", [])
    if not isinstance(attempts, list) or not attempts:
        return float("nan")
    value = attempts[-1].get("wall_time_s")
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) and result >= 0.0 else float("nan")


def _integrity_row(
    job: Mapping[str, Any],
    manifest_path: Path,
    *,
    status: str,
    included: bool,
    integrity: str,
    message: str,
    output_path: str = "",
    completion_ratio: float = float("nan"),
    production_wall_s: float = float("nan"),
) -> dict[str, Any]:
    return {
        "job_id": str(job["job_id"]),
        "method": str(job["method"]),
        "sequence_id": str(job["sequence_id"]),
        "stratum": str(job["stratum"]),
        "status": status,
        "included_in_quality_analysis": included,
        "integrity_status": integrity,
        "message": message,
        "job_manifest_path": str(manifest_path),
        "output_path": output_path,
        "completion_ratio": completion_ratio,
        "production_wall_s": production_wall_s,
    }


def collect_stage2_job_ledger(
    repo_root: str | Path,
    plan: Mapping[str, Any],
    run_root: str | Path,
    output_root: str | Path,
    *,
    persist: bool = True,
) -> LedgerBundle:
    """Verify succeeded jobs and ledger every expected plan job.

    Failed/missing jobs are recorded and excluded.  Any integrity defect in a
    manifest that claims success aborts after the ledger is persisted.
    """

    root = Path(repo_root).resolve()
    run = Path(run_root).resolve()
    output = Path(output_root).resolve()
    selected = {
        str(value["sequence_id"]): dict(value) for value in plan["selected_sequences"]
    }
    rows: list[dict[str, Any]] = []
    verified: list[VerifiedJob] = []
    integrity_errors: list[str] = []
    sequence_identity: dict[str, tuple[Path, str]] = {}

    for job in plan["projection"]["jobs"]:
        manifest_path = run / "job_manifests" / f"{job['job_id']}.json"
        if not manifest_path.is_file():
            rows.append(
                _integrity_row(
                    job,
                    manifest_path,
                    status="missing",
                    included=False,
                    integrity="not_applicable",
                    message="expected job manifest is absent",
                )
            )
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            rows.append(
                _integrity_row(
                    job,
                    manifest_path,
                    status="unreadable",
                    included=False,
                    integrity="invalid",
                    message=str(error),
                )
            )
            integrity_errors.append(str(job["job_id"]))
            continue
        status = str(manifest.get("status", "unknown"))
        wall = _attempt_wall_time(manifest)
        identity_ok = all(
            (
                str(manifest.get("job_id")) == str(job["job_id"]),
                str(manifest.get("method")) == str(job["method"]),
                str(manifest.get("sequence_id")) == str(job["sequence_id"]),
                str(manifest.get("stratum")) == str(job["stratum"]),
                str(manifest.get("plan_sha256")) == str(plan["plan_sha256"]),
            )
        )
        if not identity_ok:
            rows.append(
                _integrity_row(
                    job,
                    manifest_path,
                    status=status,
                    included=False,
                    integrity="invalid",
                    message="job identity or plan hash mismatch",
                    production_wall_s=wall,
                )
            )
            integrity_errors.append(str(job["job_id"]))
            continue
        if status != "succeeded":
            rows.append(
                _integrity_row(
                    job,
                    manifest_path,
                    status=status,
                    included=False,
                    integrity="not_evaluated",
                    message=str(manifest.get("message", "job did not succeed")),
                    output_path=str(manifest.get("output_path", "")),
                    completion_ratio=_float_or_nan(manifest.get("completion_ratio")),
                    production_wall_s=wall,
                )
            )
            continue

        planned_contract = {**dict(job), "plan_sha256": plan["plan_sha256"]}
        if not job_is_complete(manifest_path, planned_contract):
            rows.append(
                _integrity_row(
                    job,
                    manifest_path,
                    status=status,
                    included=False,
                    integrity="invalid",
                    message=(
                        "planner-provided job/method/execution contract or payload "
                        "hash failed strict verification"
                    ),
                    output_path=str(manifest.get("output_path", "")),
                    completion_ratio=_float_or_nan(manifest.get("completion_ratio")),
                    production_wall_s=wall,
                )
            )
            integrity_errors.append(str(job["job_id"]))
            continue

        error_message = ""
        try:
            sequence_path = _resolve(root, str(manifest["sequence_manifest"]))
            if not sequence_path.is_file():
                raise Stage2AnalysisError("sequence manifest is missing")
            sequence_hash = sha256_file(sequence_path)
            if sequence_hash != str(manifest.get("sequence_manifest_sha256")):
                raise Stage2AnalysisError("sequence manifest hash mismatch")
            sequence = json.loads(sequence_path.read_text(encoding="utf-8"))
            expected_sequence = selected[str(job["sequence_id"])]
            if (
                str(sequence.get("sequence_id")) != str(job["sequence_id"])
                or str(sequence.get("source_sha256"))
                != str(expected_sequence["source_sha256"])
                or int(sequence.get("source_frames", -1))
                != int(expected_sequence["frames"])
                or not np.isclose(
                    float(sequence.get("fps", np.nan)),
                    float(expected_sequence["fps"]),
                    rtol=0.0,
                    atol=1e-9,
                )
            ):
                raise Stage2AnalysisError(
                    "sequence manifest disagrees with frozen plan"
                )
            prior = sequence_identity.get(str(job["sequence_id"]))
            identity = (sequence_path.resolve(), sequence_hash)
            if prior is not None and prior != identity:
                raise Stage2AnalysisError(
                    "methods reference different sequence manifests"
                )
            sequence_identity[str(job["sequence_id"])] = identity
            canonical_path = _resolve(root, str(sequence["canonical_path"]))
            if not canonical_path.is_file() or sha256_file(canonical_path) != str(
                sequence.get("canonical_sha256")
            ):
                raise Stage2AnalysisError("canonical human package hash mismatch")
            common = sequence.get("common_scale", {})
            if (
                common.get("method_independent") is not True
                or not np.isfinite(float(common.get("value", np.nan)))
                or float(common.get("value", 0.0)) <= 0.0
            ):
                raise Stage2AnalysisError(
                    "per-sequence common evaluator scale is invalid"
                )
            output_path = _resolve(root, str(manifest["output_path"]))
            if not output_path.is_file():
                raise Stage2AnalysisError("canonical G1 output is missing")
            output_hash = sha256_file(output_path)
            if output_hash != str(manifest.get("output_sha256")):
                raise Stage2AnalysisError("canonical G1 output hash mismatch")
            motion = CanonicalG1.load(output_path)
            frames = int(sequence["source_frames"])
            motion.validate(source_frame_count=frames)
            if len(motion.qpos) != frames:
                raise Stage2AnalysisError("succeeded output is not frame-complete")
            if not np.isclose(
                float(motion.fps), float(sequence["fps"]), rtol=0.0, atol=1e-9
            ):
                raise Stage2AnalysisError(
                    "succeeded output fps differs from source timeline"
                )
            if not np.array_equal(motion.source_frame_idx, np.arange(frames)):
                raise Stage2AnalysisError(
                    "succeeded output does not cover every source frame"
                )
            if not np.asarray(motion.valid, dtype=bool).all():
                raise Stage2AnalysisError("succeeded output contains invalid frames")
            if (
                not np.isfinite(np.asarray(motion.qpos, dtype=float)).all()
                or not np.isfinite(
                    np.asarray(motion.per_frame_solve_time_s, dtype=float)
                ).all()
            ):
                raise Stage2AnalysisError("succeeded output contains NaN/Inf")
            if not np.allclose(
                np.linalg.norm(np.asarray(motion.qpos)[:, 3:7], axis=1),
                1.0,
                rtol=0.0,
                atol=1e-6,
            ):
                raise Stage2AnalysisError(
                    "succeeded output has non-unit root quaternion"
                )
            if not np.isclose(float(manifest.get("completion_ratio", np.nan)), 1.0):
                raise Stage2AnalysisError(
                    "succeeded manifest completion ratio is not 1.0"
                )
            if str(motion.metadata.get("experiment_stratum")) != str(job["stratum"]):
                raise Stage2AnalysisError("output metadata stratum mismatch")
            if str(motion.metadata.get("stage2_method_id")) != str(job["method"]):
                raise Stage2AnalysisError("output metadata method ID mismatch")
            if str(motion.metadata.get("stage2_sequence_id")) != str(
                job["sequence_id"]
            ):
                raise Stage2AnalysisError("output metadata sequence ID mismatch")
            if str(job["stratum"]) == EXTERNAL_STRATUM:
                if motion.metadata.get("timing_available") is not False:
                    raise Stage2AnalysisError(
                        "external reference must be explicitly untimed"
                    )
                if motion.metadata.get("verified_official_ground_truth") is not False:
                    raise Stage2AnalysisError(
                        "external reference cannot claim ground truth"
                    )
            verified.append(
                VerifiedJob(
                    job_id=str(job["job_id"]),
                    method=str(job["method"]),
                    sequence_id=str(job["sequence_id"]),
                    stratum=str(job["stratum"]),
                    job_manifest_path=manifest_path,
                    sequence_manifest_path=sequence_path,
                    output_path=output_path,
                    production_wall_s=wall,
                )
            )
            rows.append(
                _integrity_row(
                    job,
                    manifest_path,
                    status="succeeded",
                    included=True,
                    integrity="verified",
                    message="exact hashes and full frame coverage verified",
                    output_path=str(output_path),
                    completion_ratio=1.0,
                    production_wall_s=wall,
                )
            )
        except (
            KeyError,
            OSError,
            ValueError,
            json.JSONDecodeError,
            Stage2AnalysisError,
        ) as error:
            error_message = str(error)
            rows.append(
                _integrity_row(
                    job,
                    manifest_path,
                    status="succeeded",
                    included=False,
                    integrity="invalid",
                    message=error_message,
                    output_path=str(manifest.get("output_path", "")),
                    completion_ratio=_float_or_nan(manifest.get("completion_ratio")),
                    production_wall_s=wall,
                )
            )
            integrity_errors.append(str(job["job_id"]))

    ledger = pd.DataFrame(rows)
    expected_methods = [str(value["method"]) for value in plan["method_specs"]]
    completion_rows = []
    for method in expected_methods:
        group = ledger.loc[ledger.method == method]
        expected = len(group)
        succeeded = int((group.integrity_status == "verified").sum())
        completion_rows.append(
            {
                "method": method,
                "stratum": str(group.stratum.iloc[0]) if expected else "unknown",
                "expected_jobs": expected,
                "verified_succeeded_jobs": succeeded,
                "failed_jobs": int(group.status.eq("failed").sum()),
                "incomplete_jobs": int(group.status.eq("incomplete").sum()),
                "missing_jobs": int(group.status.eq("missing").sum()),
                "integrity_invalid_jobs": int(
                    group.integrity_status.eq("invalid").sum()
                ),
                "completion_fraction": succeeded / expected if expected else 0.0,
                "complete_for_selected_design": succeeded == expected and expected > 0,
            }
        )
    completion = pd.DataFrame(completion_rows)

    selected_methods = {
        str(value["method"]): str(value["stratum"]) for value in plan["method_specs"]
    }
    scope_rows = [
        {
            "method": method,
            "stratum": stratum,
            "stage2_scope_status": "selected",
            "reason": "included by frozen budgeted plan",
            "result_metrics_used_for_scope_decision": False,
        }
        for method, stratum in selected_methods.items()
    ]
    for value in plan.get("method_exclusions", []):
        scope_rows.append(
            {
                "method": str(value.get("method")),
                "stratum": "not_in_stage2_plan",
                "stage2_scope_status": str(value.get("status", "excluded")),
                "reason": str(value.get("reason", "plan exclusion")),
                "result_metrics_used_for_scope_decision": bool(
                    value.get("result_metrics_used_for_selection", False)
                ),
            }
        )
    method_scope = pd.DataFrame(scope_rows)
    if persist:
        _write_frame(ledger, output / "completion_failure_ledger")
        _write_frame(completion, output / "completion_by_method")
        _write_frame(method_scope, output / "method_scope_ledger")
    if integrity_errors:
        raise Stage2AnalysisError(
            "Claimed-success Stage-2 jobs failed integrity verification: "
            + ", ".join(integrity_errors)
        )
    return LedgerBundle(
        ledger=ledger,
        completion=completion,
        method_scope=method_scope,
        verified=verified,
    )


def build_sequence_evaluator_protocol(
    repo_root: str | Path,
    sequence_manifest: Mapping[str, Any],
    robot: CanonicalRobotModel,
) -> dict[str, Any]:
    """Create one method-independent evaluator-v3 contract for a sequence."""

    root = Path(repo_root).resolve()
    base_path = root / "manifests" / "evaluator.yaml"
    if not base_path.is_file():
        raise FileNotFoundError(f"Base evaluator protocol is missing: {base_path}")
    base = load_evaluator_protocol(base_path)
    common = dict(sequence_manifest["common_scale"])
    if str(common.get("robot_asset_sha256")) != robot.sha256:
        raise Stage2AnalysisError(
            f"Per-sequence robot hash differs from canonical evaluator: "
            f"{sequence_manifest.get('sequence_id')}"
        )
    local_scale = float(common["local_body_scale"])
    root_scale = float(common["root_displacement_scale"])
    alignment = np.asarray(common["root_alignment_translation_m"], dtype=float)
    if alignment.shape != (3,) or not np.isfinite(alignment).all():
        raise Stage2AnalysisError(
            "Per-sequence root alignment is not a finite 3-vector"
        )
    native_scales = dict(base.get("native_root_scales", {}))
    for method_name in (
        "sparse/controlled-common-per-sequence-scale",
        "dense/controlled-common-per-sequence-scale",
    ):
        native_scales[method_name] = {
            "value": root_scale,
            "policy": (
                "Stage-2 method-independent per-sequence LS root-displacement scale"
            ),
        }
    native_scales["unitree_reference"] = {
        "value": root_scale,
        "policy": (
            "external native scale undocumented; common evaluator scale used "
            "for descriptive root metric"
        ),
    }
    return {
        "schema_version": 3,
        "name": f"stage2-sequence-{sequence_manifest['sequence_id']}-evaluator-v3",
        "source_sha256": str(sequence_manifest["source_sha256"]),
        "source_path": str(sequence_manifest["canonical_path"]),
        "robot_xml": str(base["robot_xml"]),
        "robot_xml_sha256": robot.sha256,
        "robot_joint_order": list(robot.joint_names),
        "robot_joint_order_sha256": robot.joint_order_sha256,
        "heading": dict(base["heading"]),
        "scale": {
            "definition": str(common["definition"]),
            "method_independent": True,
            "source_frame": 0,
            "estimator_formula": str(common["formula"]),
            "source_root_frame": (
                "frame0 Hips origin; geometry heading removed; canonical +X forward"
            ),
            "robot_root_frame": "neutral canonical Holosoma G1 pelvis; +X forward",
            "landmark_weights_policy": "registered_equal_weight_per_landmark",
            "source_landmarks": list(common["source_landmarks"]),
            "robot_landmarks": list(common["robot_landmarks"]),
            "landmark_weights": list(common["landmark_weights"]),
            "excluded_landmarks": dict(base["scale"]["excluded_landmarks"]),
            "least_squares_numerator_m2": float(
                common["least_squares_numerator_m2"]
            ),
            "least_squares_denominator_m2": float(
                common["least_squares_denominator_m2"]
            ),
            "weighted_residual_rmse_m": float(
                common["weighted_residual_rmse_m"]
            ),
            "source_heading_yaw_rad": float(common["source_heading_yaw_rad"]),
            "source_heading_alignment_matrix": list(
                common["source_heading_alignment_matrix"]
            ),
            "common_local_body_scale": local_scale,
            "local_body_scale_definition": "shared_semantic_landmark_ls_value",
            "common_root_displacement_scale": root_scale,
            "root_displacement_scale_definition": (
                "separately frozen scalar; numerically initialized from shared LS"
            ),
            "common_static_scale": local_scale,
            "common_root_alignment_translation_m": alignment.tolist(),
            "root_alignment_definition": (
                "neutral_g1_root_minus_root_displacement_scale_times_source_frame0_root"
            ),
            "root_anchor_policy": str(common["root_anchor"]),
            "root_local_and_anchor_frozen_separately": True,
            "diagnostics": {
                "head_to_toe_definition": (
                    "neutral_g1_head_to_mean_toe_over_source_frame0_head_to_mean_toe"
                ),
                "head_to_toe_role": "diagnostic_only_not_common_policy",
                "source_head_to_toe_span_m": float(
                    common["source_head_to_toe_span_m"]
                ),
                "robot_head_to_toe_span_m": float(
                    common["robot_head_to_toe_span_m"]
                ),
                "head_to_toe_scale": float(
                    common["head_to_toe_diagnostic_scale"]
                ),
            },
        },
        "native_root_scales": native_scales,
        "thresholds": dict(base["thresholds"]),
        "reporting": {
            **dict(base.get("reporting", {})),
            "sequence_level_common_contract": True,
            "cross_stratum_causal_ranking_forbidden": True,
            "external_reference_is_ground_truth": False,
        },
        "stage2_sequence_id": str(sequence_manifest["sequence_id"]),
        "sequence_manifest_sha256": sha256_file(
            _resolve(root, str(sequence_manifest["_manifest_path"]))
        ),
    }


def evaluate_verified_stage2_jobs(
    repo_root: str | Path,
    verified: Sequence[VerifiedJob],
    output_root: str | Path,
    *,
    budget_guard: Callable[[], None] | None = None,
) -> pd.DataFrame:
    """Evaluate every verified canonical output with its sequence contract."""

    if not verified:
        raise Stage2AnalysisError("No verified Stage-2 jobs are available to evaluate")
    root = Path(repo_root).resolve()
    output = Path(output_root).resolve()
    robot = CanonicalRobotModel(default_robot_scene(root))
    protocol_dir = output / "evaluator_protocols"
    run_metrics = output / "runs"
    protocol_cache: dict[str, tuple[dict[str, Any], str]] = {}
    rows: list[dict[str, Any]] = []

    for job in sorted(
        verified, key=lambda value: (value.sequence_id, value.stratum, value.method)
    ):
        if budget_guard is not None:
            budget_guard()
        sequence = json.loads(job.sequence_manifest_path.read_text(encoding="utf-8"))
        sequence["_manifest_path"] = str(job.sequence_manifest_path)
        canonical_path = _resolve(root, str(sequence["canonical_path"]))
        human = CanonicalHuman.load(canonical_path)
        frames = int(sequence["source_frames"])
        if len(human.timestamps) != frames:
            raise Stage2AnalysisError(
                f"Canonical human frame mismatch during evaluation: {job.sequence_id}"
            )
        if not np.isclose(human.fps, float(sequence["fps"]), rtol=0.0, atol=1e-9):
            raise Stage2AnalysisError(
                f"Canonical human fps mismatch during evaluation: {job.sequence_id}"
            )
        if job.sequence_id not in protocol_cache:
            protocol = build_sequence_evaluator_protocol(root, sequence, robot)
            protocol_path = protocol_dir / f"{job.sequence_id}.json"
            atomic_write_json(protocol_path, protocol)
            protocol_hash = sha256_file(protocol_path)
            protocol["manifest_sha256"] = protocol_hash
            protocol_cache[job.sequence_id] = (protocol, protocol_hash)
        protocol, protocol_hash = protocol_cache[job.sequence_id]
        motion = CanonicalG1.load(job.output_path)
        table, summary = evaluate_motion(human, motion, robot, protocol)
        save_evaluation(table, summary, run_metrics, job.job_id)
        duration_s = frames / float(sequence["fps"])
        production_rtf = (
            job.production_wall_s / duration_s
            if np.isfinite(job.production_wall_s)
            else float("nan")
        )
        rows.append(
            {
                **summary,
                "job_id": job.job_id,
                "method": job.method,
                "output_metadata_method": str(motion.metadata.get("method", "")),
                "sequence_id": job.sequence_id,
                "subject_id": str(
                    sequence.get("actor_id") or _subject_id(job.sequence_id)
                ),
                "stratum": job.stratum,
                "source_frames": frames,
                "source_duration_s": duration_s,
                "sequence_manifest_path": str(job.sequence_manifest_path),
                "sequence_manifest_sha256": sha256_file(job.sequence_manifest_path),
                "evaluator_protocol_sha256": protocol_hash,
                "output_path": str(job.output_path),
                "output_sha256": sha256_file(job.output_path),
                "production_wall_s": job.production_wall_s,
                "production_rtf_descriptive": production_rtf,
                "timing_comparable": job.stratum != EXTERNAL_STRATUM,
                "external_reference_verified_ground_truth": False,
            }
        )
        if budget_guard is not None:
            budget_guard()
    frame = (
        pd.DataFrame(rows)
        .sort_values(["stratum", "method", "sequence_id"])
        .reset_index(drop=True)
    )
    required_numeric = [*ANALYSIS_METRICS, "completion_ratio"]
    if not np.isfinite(frame[required_numeric].to_numpy(dtype=float)).all():
        raise Stage2AnalysisError("Stage-2 evaluator summary contains NaN/Inf")
    if not np.allclose(frame.completion_ratio, 1.0):
        raise Stage2AnalysisError("Evaluated Stage-2 output is incomplete")
    _write_frame(frame, output / "per_sequence_method_summary")
    return frame


def _group_seed(seed: int, values: Iterable[Any]) -> int:
    payload = "|".join([str(seed), *(str(value) for value in values)])
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def _ensure_subject_column(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "subject_id" not in result:
        result["subject_id"] = result.sequence_id.astype(str).map(_subject_id)
    if result.subject_id.isna().any() or result.subject_id.astype(str).eq("").any():
        raise Stage2AnalysisError("Subject-cluster identity is missing")
    return result


def _cluster_bootstrap_mean(
    frame: pd.DataFrame,
    value_column: str,
    *,
    seed: int,
    resamples: int,
) -> tuple[np.ndarray, int]:
    """Resample actors, retaining every sequence within each sampled actor."""

    frame = _ensure_subject_column(frame)
    clusters = [
        group[value_column].to_numpy(dtype=float)
        for _, group in frame.groupby("subject_id", sort=True)
    ]
    clusters = [values[np.isfinite(values)] for values in clusters]
    clusters = [values for values in clusters if len(values)]
    if not clusters:
        return np.asarray([], dtype=float), 0
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(clusters), size=(resamples, len(clusters)))
    boot = np.empty(resamples, dtype=float)
    for index, draw in enumerate(sampled):
        boot[index] = float(np.mean(np.concatenate([clusters[item] for item in draw])))
    return boot, len(clusters)


def bootstrap_sequence_uncertainty(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    metrics: Sequence[str] = UNCERTAINTY_METRICS,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> pd.DataFrame:
    """Deterministic actor-cluster bootstrap of sequence-level method means."""

    if resamples < 100:
        raise ValueError("At least 100 bootstrap resamples are required")
    missing = (set(group_columns) | {"sequence_id", *metrics}) - set(frame)
    if missing:
        raise ValueError(f"Bootstrap table lacks columns: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    group_key: str | list[str]
    group_key = group_columns[0] if len(group_columns) == 1 else list(group_columns)
    for key, group in frame.groupby(group_key, sort=True, dropna=False):
        keys = (key,) if len(group_columns) == 1 else tuple(key)
        ordered = _ensure_subject_column(group.sort_values("sequence_id"))
        if ordered.sequence_id.duplicated().any():
            raise ValueError(f"Bootstrap group has duplicate sequence IDs: {keys}")
        for metric in metrics:
            values = ordered[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if not len(values):
                continue
            local_seed = _group_seed(seed, (*keys, metric))
            boot, subject_count = _cluster_bootstrap_mean(
                ordered.loc[np.isfinite(ordered[metric].to_numpy(dtype=float))],
                metric,
                seed=local_seed,
                resamples=resamples,
            )
            row = {
                column: value for column, value in zip(group_columns, keys, strict=True)
            }
            row.update(
                {
                    "metric": metric,
                    "sequence_count": len(values),
                    "subject_count": subject_count,
                    "sequence_mean": float(np.mean(values)),
                    "sequence_median": float(np.median(values)),
                    "sequence_std": float(np.std(values, ddof=1))
                    if len(values) > 1
                    else 0.0,
                    "bootstrap_mean": float(np.mean(boot)),
                    "bootstrap_ci95_low": float(np.percentile(boot, 2.5)),
                    "bootstrap_ci95_high": float(np.percentile(boot, 97.5)),
                    "bootstrap_standard_error": float(np.std(boot, ddof=1)),
                    "bootstrap_seed": seed,
                    "group_seed": local_seed,
                    "bootstrap_resamples": resamples,
                    "uncertainty_unit": "subject_cluster",
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def build_leave_one_subject_out(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str] = ("stratum", "method"),
    metrics: Sequence[str] = UNCERTAINTY_METRICS,
) -> pd.DataFrame:
    """Expose influence of each LAFAN actor on every primary method mean."""

    frame = _ensure_subject_column(frame)
    rows: list[dict[str, Any]] = []
    group_key: str | list[str] = (
        group_columns[0] if len(group_columns) == 1 else list(group_columns)
    )
    for key, group in frame.groupby(group_key, sort=True, dropna=False):
        keys = (key,) if len(group_columns) == 1 else tuple(key)
        subjects = sorted(group.subject_id.astype(str).unique())
        for held_out in subjects:
            retained = group.loc[group.subject_id.astype(str) != held_out]
            for metric in metrics:
                values = retained[metric].to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                if not len(values):
                    continue
                row = {
                    column: value
                    for column, value in zip(group_columns, keys, strict=True)
                }
                row.update(
                    {
                        "held_out_subject_id": held_out,
                        "metric": metric,
                        "retained_subject_count": int(
                            retained.subject_id.astype(str).nunique()
                        ),
                        "retained_sequence_count": int(len(values)),
                        "leave_one_subject_out_mean": float(np.mean(values)),
                        "full_sequence_mean": float(
                            np.nanmean(group[metric].to_numpy(dtype=float))
                        ),
                    }
                )
                row["mean_shift_from_full"] = (
                    row["leave_one_subject_out_mean"] - row["full_sequence_mean"]
                )
                rows.append(row)
    return pd.DataFrame(rows)


def build_within_stratum_pairs(
    per_sequence: pd.DataFrame,
    *,
    metrics: Sequence[str] = UNCERTAINTY_METRICS,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build paired differences only among methods in the same stratum."""

    per_sequence = _ensure_subject_column(per_sequence)
    rows: list[dict[str, Any]] = []
    for stratum, stratum_frame in per_sequence.groupby("stratum", sort=True):
        methods = sorted(stratum_frame.method.unique())
        for method_a, method_b in combinations(methods, 2):
            left = stratum_frame.loc[stratum_frame.method == method_a]
            right = stratum_frame.loc[stratum_frame.method == method_b]
            merged = left.merge(
                right,
                on="sequence_id",
                suffixes=("_a", "_b"),
                validate="one_to_one",
            )
            if not (merged.subject_id_a == merged.subject_id_b).all():
                raise Stage2AnalysisError("Paired methods disagree on subject identity")
            for value in merged.itertuples():
                for metric in metrics:
                    rows.append(
                        {
                            "stratum": stratum,
                            "method_a": method_a,
                            "method_b": method_b,
                            "sequence_id": value.sequence_id,
                            "subject_id": value.subject_id_a,
                            "metric": metric,
                            "value_a": float(getattr(value, f"{metric}_a")),
                            "value_b": float(getattr(value, f"{metric}_b")),
                            "paired_difference_b_minus_a": float(
                                getattr(value, f"{metric}_b")
                                - getattr(value, f"{metric}_a")
                            ),
                            "cross_stratum_comparison": False,
                        }
                    )
    columns = (
        "stratum",
        "method_a",
        "method_b",
        "sequence_id",
        "subject_id",
        "metric",
        "value_a",
        "value_b",
        "paired_difference_b_minus_a",
        "cross_stratum_comparison",
    )
    paired = pd.DataFrame(rows, columns=columns)
    if paired.empty:
        summary = pd.DataFrame(
            columns=(
                "stratum",
                "method_a",
                "method_b",
                "metric",
                "sequence_count",
                "subject_count",
                "mean_paired_difference_b_minus_a",
                "bootstrap_ci95_low",
                "bootstrap_ci95_high",
                "bootstrap_seed",
                "bootstrap_resamples",
            )
        )
        return paired, summary
    summary_rows: list[dict[str, Any]] = []
    for key, group in paired.groupby(
        ["stratum", "method_a", "method_b", "metric"], sort=True
    ):
        ordered = group.sort_values("sequence_id")
        values = ordered.paired_difference_b_minus_a.to_numpy(dtype=float)
        local_seed = _group_seed(seed, key)
        boot, subject_count = _cluster_bootstrap_mean(
            ordered,
            "paired_difference_b_minus_a",
            seed=local_seed,
            resamples=resamples,
        )
        summary_rows.append(
            {
                "stratum": key[0],
                "method_a": key[1],
                "method_b": key[2],
                "metric": key[3],
                "sequence_count": len(values),
                "subject_count": subject_count,
                "mean_paired_difference_b_minus_a": float(np.mean(values)),
                "bootstrap_ci95_low": float(np.percentile(boot, 2.5)),
                "bootstrap_ci95_high": float(np.percentile(boot, 97.5)),
                "bootstrap_seed": seed,
                "group_seed": local_seed,
                "bootstrap_resamples": resamples,
                "cross_stratum_comparison": False,
            }
        )
    return paired, pd.DataFrame(summary_rows)


def build_reference_intersection(
    per_sequence: pd.DataFrame,
    *,
    metrics: Sequence[str] = UNCERTAINTY_METRICS,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare to the external corpus descriptively on filename intersection."""

    per_sequence = _ensure_subject_column(per_sequence)
    reference = per_sequence.loc[per_sequence.stratum == EXTERNAL_STRATUM].copy()
    if reference.empty:
        columns = (
            "method",
            "method_stratum",
            "reference_method",
            "sequence_id",
            "subject_id",
            "metric",
            "method_value",
            "reference_value",
            "method_minus_reference",
            "interpretation",
        )
        summary_columns = (
            "method",
            "method_stratum",
            "reference_method",
            "metric",
            "intersection_sequence_count",
            "intersection_subject_count",
            "mean_method_minus_reference",
            "bootstrap_ci95_low",
            "bootstrap_ci95_high",
            "bootstrap_seed",
            "bootstrap_resamples",
            "interpretation",
        )
        return pd.DataFrame(columns=columns), pd.DataFrame(columns=summary_columns)
    if reference.groupby("sequence_id").size().max() != 1:
        raise Stage2AnalysisError(
            "External-reference stratum has duplicate sequence rows"
        )
    methods = per_sequence.loc[per_sequence.stratum != EXTERNAL_STRATUM]
    rows: list[dict[str, Any]] = []
    for (stratum, method), group in methods.groupby(["stratum", "method"], sort=True):
        merged = group.merge(
            reference,
            on="sequence_id",
            suffixes=("_method", "_reference"),
            validate="one_to_one",
        )
        if not (merged.subject_id_method == merged.subject_id_reference).all():
            raise Stage2AnalysisError("Method/reference subject identity mismatch")
        for value in merged.itertuples():
            for metric in metrics:
                method_value = float(getattr(value, f"{metric}_method"))
                reference_value = float(getattr(value, f"{metric}_reference"))
                rows.append(
                    {
                        "method": method,
                        "method_stratum": stratum,
                        "reference_method": str(value.method_reference),
                        "sequence_id": str(value.sequence_id),
                        "subject_id": str(value.subject_id_method),
                        "metric": metric,
                        "method_value": method_value,
                        "reference_value": reference_value,
                        "method_minus_reference": method_value - reference_value,
                        "interpretation": (
                            "descriptive difference on filename intersection; "
                            "external reference is not ground truth"
                        ),
                    }
                )
    intersection = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    if not intersection.empty:
        for key, group in intersection.groupby(
            ["method", "method_stratum", "reference_method", "metric"], sort=True
        ):
            ordered = group.sort_values("sequence_id")
            values = ordered.method_minus_reference.to_numpy(dtype=float)
            local_seed = _group_seed(seed, key)
            boot, subject_count = _cluster_bootstrap_mean(
                ordered,
                "method_minus_reference",
                seed=local_seed,
                resamples=resamples,
            )
            summary_rows.append(
                {
                    "method": key[0],
                    "method_stratum": key[1],
                    "reference_method": key[2],
                    "metric": key[3],
                    "intersection_sequence_count": len(values),
                    "intersection_subject_count": subject_count,
                    "mean_method_minus_reference": float(np.mean(values)),
                    "bootstrap_ci95_low": float(np.percentile(boot, 2.5)),
                    "bootstrap_ci95_high": float(np.percentile(boot, 97.5)),
                    "bootstrap_seed": seed,
                    "group_seed": local_seed,
                    "bootstrap_resamples": resamples,
                    "interpretation": (
                        "descriptive difference; zero is reference agreement, not truth"
                    ),
                }
            )
    return intersection, pd.DataFrame(summary_rows)


DIRECT_REFERENCE_METRICS = (
    "root_translation_agreement_mean_m",
    "root_translation_agreement_p95_m",
    "root_anchor_frame0_agreement_m",
    "root_displacement_agreement_mean_m",
    "root_displacement_agreement_p95_m",
    "root_yaw_agreement_mean_rad",
    "root_orientation_agreement_mean_rad",
    "joint_angle_agreement_rmse_rad",
    "joint_velocity_agreement_rmse_rad_s",
    "fk_world_semantic_agreement_mean_m",
    "fk_root_frame_semantic_agreement_mean_m",
)


def _wrap_angle(value: np.ndarray) -> np.ndarray:
    return (np.asarray(value, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def build_direct_reference_trajectory_metrics(
    verified: Sequence[VerifiedJob],
    robot: CanonicalRobotModel,
    *,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
    budget_guard: Callable[[], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure G1 trajectory disagreement on the basename/frame-index overlap.

    The corpus audit does not establish byte-identical human inputs or exact
    timestamp identity.  Consequently, this routine pairs only a normalized
    sequence basename and integer frame index.  Absolute root translation is
    retained, but frame-0 anchor and frame-0-subtracted displacement are also
    reported so a placement offset cannot masquerade as path-shape disagreement.
    """

    references = {
        job.sequence_id: job for job in verified if job.stratum == EXTERNAL_STRATUM
    }
    if len(references) != sum(job.stratum == EXTERNAL_STRATUM for job in verified):
        raise Stage2AnalysisError(
            "Duplicate verified reference trajectory for one sequence"
        )
    rows: list[dict[str, Any]] = []
    semantic_names = tuple(sorted(robot.body_ids)) + ("head",)
    for job in sorted(verified, key=lambda value: (value.method, value.sequence_id)):
        if budget_guard is not None:
            budget_guard()
        reference_job = references.get(job.sequence_id)
        if job.stratum == EXTERNAL_STRATUM or reference_job is None:
            continue
        method_motion = CanonicalG1.load(job.output_path)
        reference_motion = CanonicalG1.load(reference_job.output_path)
        if (
            method_motion.qpos.shape != reference_motion.qpos.shape
            or not np.isclose(
                method_motion.fps, reference_motion.fps, atol=1e-9, rtol=0.0
            )
            or not np.array_equal(
                method_motion.source_frame_idx, reference_motion.source_frame_idx
            )
        ):
            raise Stage2AnalysisError(
                f"Direct reference frame-index pairing mismatch: {job.job_id}"
            )
        q_method = np.asarray(method_motion.qpos, dtype=float)
        q_reference = np.asarray(reference_motion.qpos, dtype=float)
        root_delta = np.linalg.norm(q_method[:, :3] - q_reference[:, :3], axis=1)
        root_anchor_delta = float(
            np.linalg.norm(q_method[0, :3] - q_reference[0, :3])
        )
        method_displacement = q_method[:, :3] - q_method[0, :3]
        reference_displacement = q_reference[:, :3] - q_reference[0, :3]
        root_displacement_delta = np.linalg.norm(
            method_displacement - reference_displacement, axis=1
        )
        method_rotation = quaternion_wxyz_to_matrix(q_method[:, 3:7])
        reference_rotation = quaternion_wxyz_to_matrix(q_reference[:, 3:7])
        method_yaw = yaw_from_matrix(method_rotation)
        reference_yaw = yaw_from_matrix(reference_rotation)
        yaw_delta = np.abs(_wrap_angle(method_yaw - reference_yaw))
        relative_rotation = np.einsum(
            "tji,tjk->tik", reference_rotation, method_rotation
        )
        cosine = np.clip(
            (np.trace(relative_rotation, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0
        )
        orientation_delta = np.arccos(cosine)
        joint_delta = _wrap_angle(q_method[:, 7:] - q_reference[:, 7:])
        if len(q_method) > 1:
            method_joint_step = _wrap_angle(np.diff(q_method[:, 7:], axis=0))
            reference_joint_step = _wrap_angle(
                np.diff(q_reference[:, 7:], axis=0)
            )
            velocity_delta = (
                method_joint_step - reference_joint_step
            ) * float(method_motion.fps)
            velocity_rmse = float(np.sqrt(np.mean(np.square(velocity_delta))))
        else:
            velocity_rmse = 0.0
        world_distances: list[float] = []
        root_frame_distances: list[float] = []
        for frame_index, (method_qpos, reference_qpos) in enumerate(
            zip(q_method, q_reference, strict=True)
        ):
            method_positions = robot.semantic_positions(method_qpos)
            reference_positions = robot.semantic_positions(reference_qpos)
            method_root = method_positions["root"]
            reference_root = reference_positions["root"]
            for name in semantic_names:
                world_distances.append(
                    float(
                        np.linalg.norm(
                            method_positions[name] - reference_positions[name]
                        )
                    )
                )
                method_local = method_positions[name] - method_root
                reference_local = reference_positions[name] - reference_root
                method_cos = np.cos(-method_yaw[frame_index])
                method_sin = np.sin(-method_yaw[frame_index])
                reference_cos = np.cos(-reference_yaw[frame_index])
                reference_sin = np.sin(-reference_yaw[frame_index])
                method_root_frame = np.asarray(
                    [
                        method_cos * method_local[0]
                        - method_sin * method_local[1],
                        method_sin * method_local[0]
                        + method_cos * method_local[1],
                        method_local[2],
                    ]
                )
                reference_root_frame = np.asarray(
                    [
                        reference_cos * reference_local[0]
                        - reference_sin * reference_local[1],
                        reference_sin * reference_local[0]
                        + reference_cos * reference_local[1],
                        reference_local[2],
                    ]
                )
                root_frame_distances.append(
                    float(np.linalg.norm(method_root_frame - reference_root_frame))
                )
        sequence = json.loads(job.sequence_manifest_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "method": job.method,
                "method_stratum": job.stratum,
                "reference_method": reference_job.method,
                "sequence_id": job.sequence_id,
                "subject_id": str(
                    sequence.get("actor_id") or _subject_id(job.sequence_id)
                ),
                "frames": len(q_method),
                "fps": float(method_motion.fps),
                "method_output_sha256": sha256_file(job.output_path),
                "reference_output_sha256": sha256_file(reference_job.output_path),
                "root_translation_agreement_mean_m": float(np.mean(root_delta)),
                "root_translation_agreement_p95_m": float(
                    np.percentile(root_delta, 95.0)
                ),
                "root_anchor_frame0_agreement_m": root_anchor_delta,
                "root_displacement_agreement_mean_m": float(
                    np.mean(root_displacement_delta)
                ),
                "root_displacement_agreement_p95_m": float(
                    np.percentile(root_displacement_delta, 95.0)
                ),
                "root_yaw_agreement_mean_rad": float(np.mean(yaw_delta)),
                "root_orientation_agreement_mean_rad": float(
                    np.mean(orientation_delta)
                ),
                "joint_angle_agreement_rmse_rad": float(
                    np.sqrt(np.mean(np.square(joint_delta)))
                ),
                "joint_velocity_agreement_rmse_rad_s": velocity_rmse,
                "fk_world_semantic_agreement_mean_m": float(np.mean(world_distances)),
                "fk_root_frame_semantic_agreement_mean_m": float(
                    np.mean(root_frame_distances)
                ),
                "interpretation": (
                    "direct G1 disagreement on normalized basename and integer frame "
                    "index only; byte-identical human input and exact timestamp identity "
                    "are not established; lower means closer to the external trajectory, "
                    "not higher verified accuracy"
                ),
            }
        )
        if budget_guard is not None:
            budget_guard()
    direct = pd.DataFrame(rows)
    if direct.empty:
        direct_columns = (
            "method",
            "method_stratum",
            "reference_method",
            "sequence_id",
            "subject_id",
            *DIRECT_REFERENCE_METRICS,
        )
        summary_columns = (
            "method",
            "method_stratum",
            "reference_method",
            "metric",
            "intersection_sequence_count",
            "intersection_subject_count",
            "sequence_mean_disagreement",
            "bootstrap_ci95_low",
            "bootstrap_ci95_high",
        )
        return pd.DataFrame(columns=direct_columns), pd.DataFrame(
            columns=summary_columns
        )
    numeric = direct.loc[:, DIRECT_REFERENCE_METRICS].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise Stage2AnalysisError("Direct reference trajectory metrics contain NaN/Inf")
    summary_rows: list[dict[str, Any]] = []
    for key, group in direct.groupby(
        ["method", "method_stratum", "reference_method"], sort=True
    ):
        for metric in DIRECT_REFERENCE_METRICS:
            local_seed = _group_seed(seed, (*key, metric, "direct-reference"))
            boot, subject_count = _cluster_bootstrap_mean(
                group,
                metric,
                seed=local_seed,
                resamples=resamples,
            )
            values = group[metric].to_numpy(dtype=float)
            summary_rows.append(
                {
                    "method": key[0],
                    "method_stratum": key[1],
                    "reference_method": key[2],
                    "metric": metric,
                    "intersection_sequence_count": len(values),
                    "intersection_subject_count": subject_count,
                    "sequence_mean_disagreement": float(np.mean(values)),
                    "bootstrap_ci95_low": float(np.percentile(boot, 2.5)),
                    "bootstrap_ci95_high": float(np.percentile(boot, 97.5)),
                    "bootstrap_seed": seed,
                    "group_seed": local_seed,
                    "bootstrap_resamples": resamples,
                    "uncertainty_unit": "subject_cluster",
                    "interpretation": (
                        "descriptive G1 agreement on basename/frame-index overlap only; "
                        "not a same-source, same-timestamp, or ground-truth error"
                    ),
                }
            )
    return direct, pd.DataFrame(summary_rows)


def aggregate_stage2_tables(
    per_sequence: pd.DataFrame,
    output_root: str | Path,
    *,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> AnalysisTables:
    """Build paired, uncertainty, and reference-intersection evidence tables."""

    output = Path(output_root).resolve()
    per_method = bootstrap_sequence_uncertainty(
        per_sequence,
        group_columns=("stratum", "method"),
        seed=seed,
        resamples=resamples,
    )
    pairs, pair_summary = build_within_stratum_pairs(
        per_sequence, seed=seed, resamples=resamples
    )
    reference, reference_summary = build_reference_intersection(
        per_sequence, seed=seed, resamples=resamples
    )
    leave_one_subject_out = build_leave_one_subject_out(per_sequence)
    _write_frame(per_method, output / "per_method_metric_summary")
    _write_frame(pairs, output / "within_stratum_paired_per_sequence")
    _write_frame(pair_summary, output / "within_stratum_paired_summary")
    _write_frame(reference, output / "reference_intersection_per_sequence")
    _write_frame(reference_summary, output / "reference_intersection_summary")
    _write_frame(leave_one_subject_out, output / "leave_one_subject_out")
    return AnalysisTables(
        per_sequence_method=per_sequence,
        per_method_metric=per_method,
        within_stratum_pairs=pairs,
        within_stratum_pair_summary=pair_summary,
        reference_intersection=reference,
        reference_summary=reference_summary,
        leave_one_subject_out=leave_one_subject_out,
    )


def build_calibration_scale_tables(
    verified: Sequence[VerifiedJob],
    per_sequence: pd.DataFrame,
    output_root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Serialize per-sequence calibration and non-causal scale sensitivity.

    Calibration is recomputed once for each sequence, never once per actor.
    Sensitivity here is an observational diagnostic across the frozen
    per-sequence scales and a comparison of common-scale vs scale-invariant
    root metrics. It is not a counterfactual retargeter rerun.
    """

    output = Path(output_root).resolve()
    manifests: dict[str, Path] = {}
    for job in verified:
        prior = manifests.get(job.sequence_id)
        if prior is not None and prior.resolve() != job.sequence_manifest_path.resolve():
            raise Stage2AnalysisError(
                f"Multiple calibration manifests for sequence {job.sequence_id}"
            )
        manifests[job.sequence_id] = job.sequence_manifest_path
    calibration_rows: list[dict[str, Any]] = []
    for sequence_id, path in sorted(manifests.items()):
        value = json.loads(path.read_text(encoding="utf-8"))
        common = dict(value["common_scale"])
        alignment = np.asarray(common["root_alignment_translation_m"], dtype=float)
        if alignment.shape != (3,) or not np.isfinite(alignment).all():
            raise Stage2AnalysisError("Calibration root alignment is invalid")
        calibration_rows.append(
            {
                "sequence_id": sequence_id,
                "subject_id": str(value.get("actor_id") or _subject_id(sequence_id)),
                "sequence_manifest_sha256": sha256_file(path),
                "common_scale": float(common["value"]),
                "local_body_scale": float(common["local_body_scale"]),
                "root_displacement_scale": float(common["root_displacement_scale"]),
                "weighted_residual_rmse_m": float(
                    common["weighted_residual_rmse_m"]
                ),
                "head_to_toe_diagnostic_scale": float(
                    common["head_to_toe_diagnostic_scale"]
                ),
                "source_head_to_toe_span_m": float(
                    common["source_head_to_toe_span_m"]
                ),
                "root_anchor_translation_norm_m": float(np.linalg.norm(alignment)),
                "calibration_unit": "one frozen estimate per source sequence",
                "method_independent": bool(common.get("method_independent")),
            }
        )
    calibration = pd.DataFrame(calibration_rows)
    numeric_columns = (
        "common_scale",
        "local_body_scale",
        "root_displacement_scale",
        "weighted_residual_rmse_m",
        "head_to_toe_diagnostic_scale",
        "source_head_to_toe_span_m",
        "root_anchor_translation_norm_m",
    )
    if calibration.empty or not np.isfinite(
        calibration.loc[:, numeric_columns].to_numpy(dtype=float)
    ).all():
        raise Stage2AnalysisError("Per-sequence calibration table is empty or non-finite")
    if not calibration.method_independent.all():
        raise Stage2AnalysisError("A per-sequence common calibration is method-dependent")

    variability_rows: list[dict[str, Any]] = []
    groups: list[tuple[str, pd.DataFrame]] = [("all_sequences", calibration)]
    groups.extend(
        (f"subject:{subject}", group)
        for subject, group in calibration.groupby("subject_id", sort=True)
    )
    for scope, group in groups:
        for metric in (
            "common_scale",
            "weighted_residual_rmse_m",
            "head_to_toe_diagnostic_scale",
        ):
            values = group[metric].to_numpy(dtype=float)
            mean = float(np.mean(values))
            standard_deviation = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
            variability_rows.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "sequence_count": len(values),
                    "mean": mean,
                    "standard_deviation": standard_deviation,
                    "median": float(np.median(values)),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                    "coefficient_of_variation": (
                        standard_deviation / abs(mean) if abs(mean) > 1e-12 else 0.0
                    ),
                    "estimand": "sequence-level calibration variability",
                }
            )
    variability = pd.DataFrame(variability_rows)

    merged = per_sequence.merge(
        calibration[["sequence_id", "common_scale"]],
        on="sequence_id",
        how="left",
        validate="many_to_one",
    )
    if merged.common_scale.isna().any():
        raise Stage2AnalysisError("Metrics lack per-sequence calibration evidence")
    sensitivity_rows: list[dict[str, Any]] = []
    for (stratum, method), group in merged.groupby(["stratum", "method"], sort=True):
        scales = group.common_scale.to_numpy(dtype=float)
        for metric in (
            "rf_kpe_all_mean_m",
            "root_translation_common_scale_mean_m",
            "root_translation_scale_invariant_mean_m",
            "artifact_rate",
        ):
            values = group[metric].to_numpy(dtype=float)
            if len(values) > 1 and float(np.ptp(scales)) > 1e-12:
                slope = float(np.polyfit(scales, values, deg=1)[0])
                scale_rank = pd.Series(scales).rank(method="average").to_numpy()
                value_rank = pd.Series(values).rank(method="average").to_numpy()
                if float(np.ptp(values)) > 1e-12:
                    pearson = float(np.corrcoef(scales, values)[0, 1])
                    spearman = float(np.corrcoef(scale_rank, value_rank)[0, 1])
                else:
                    pearson = spearman = 0.0
            else:
                slope = pearson = spearman = 0.0
            sensitivity_rows.append(
                {
                    "stratum": stratum,
                    "method": method,
                    "metric": metric,
                    "sequence_count": len(values),
                    "common_scale_min": float(np.min(scales)),
                    "common_scale_max": float(np.max(scales)),
                    "metric_mean": float(np.mean(values)),
                    "ols_slope_per_scale_unit": slope,
                    "pearson_correlation": pearson,
                    "spearman_correlation": spearman,
                    "interpretation": (
                        "observational across-sequence diagnostic; actor, motion, pose, "
                        "and calibration co-vary; not a causal scale intervention"
                    ),
                }
            )
        common_root = group.root_translation_common_scale_mean_m.to_numpy(dtype=float)
        invariant_root = group.root_translation_scale_invariant_mean_m.to_numpy(
            dtype=float
        )
        sensitivity_rows.append(
            {
                "stratum": stratum,
                "method": method,
                "metric": "common_minus_scale_invariant_root_error",
                "sequence_count": len(group),
                "common_scale_min": float(np.min(scales)),
                "common_scale_max": float(np.max(scales)),
                "metric_mean": float(np.mean(common_root - invariant_root)),
                "ols_slope_per_scale_unit": 0.0,
                "pearson_correlation": 0.0,
                "spearman_correlation": 0.0,
                "interpretation": (
                    "registered metric-definition contrast, not a retargeter rerun or "
                    "causal scale intervention"
                ),
            }
        )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity_numeric = (
        "common_scale_min",
        "common_scale_max",
        "metric_mean",
        "ols_slope_per_scale_unit",
        "pearson_correlation",
        "spearman_correlation",
    )
    if sensitivity.empty or not np.isfinite(
        sensitivity.loc[:, sensitivity_numeric].to_numpy(dtype=float)
    ).all():
        raise Stage2AnalysisError("Scale-sensitivity diagnostics contain NaN/Inf")

    conclusion_rows: list[dict[str, Any]] = []
    for stratum, group in per_sequence.groupby("stratum", sort=True):
        conclusion_rows.append(
            {
                "question": "selected-design kinematic quality",
                "evidence_stratum": stratum,
                "methods": ", ".join(sorted(group.method.unique())),
                "sequence_count": int(group.sequence_id.nunique()),
                "evidence_status": "verified descriptive evidence",
                "allowed_conclusion": (
                    "sequence-level distribution and within-stratum paired differences"
                ),
                "forbidden_conclusion": (
                    "cross-stratum solver ranking, dynamics feasibility, or universal winner"
                ),
            }
        )
    conclusion = pd.DataFrame(conclusion_rows)
    _write_frame(calibration, output / "calibration_per_sequence")
    _write_frame(variability, output / "calibration_scale_variability")
    _write_frame(sensitivity, output / "scale_sensitivity_diagnostics")
    _write_frame(conclusion, output / "conclusion_ledger")
    return calibration, variability, sensitivity, conclusion


STRATUM_COLORS = {
    "controlled_common_per_sequence_scale": "#4C78A8",
    "native_public_pipeline": "#F58518",
    "benchmark_public_retargeter_port": "#B279A2",
    "external_reference": "#222222",
}


def save_stage2_chart(
    output_root: str | Path,
    name: str,
    source: pd.DataFrame,
    renderer: Callable[[Any, pd.DataFrame], Any],
) -> tuple[Path, Path, Path, Path]:
    """Write exact chart data plus SVG, PDF, and 300-dpi PNG."""

    root = Path(output_root).resolve()
    figures = root / "figures"
    sources = figures / "source_data"
    figures.mkdir(parents=True, exist_ok=True)
    sources.mkdir(parents=True, exist_ok=True)
    csv_path = sources / f"{name}.csv"
    source.to_csv(csv_path, index=False)

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    style = {
        "font.family": "DejaVu Sans",
        "font.size": 9.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "x",
        "grid.alpha": 0.20,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.hashsalt": "retargeting-comparison-stage2",
    }
    with mpl.rc_context(style):
        figure = renderer(plt, source.copy())
        if figure is None:
            raise ValueError(f"Stage-2 chart renderer {name!r} returned no figure")
        paths: list[Path] = []
        for suffix, kwargs in (
            ("svg", {"metadata": {"Date": None, "Creator": "rtcmp"}}),
            ("pdf", {"metadata": {"CreationDate": None, "Creator": "rtcmp"}}),
            ("png", {"dpi": 300, "metadata": {"Software": "rtcmp"}}),
        ):
            path = figures / f"{name}.{suffix}"
            figure.savefig(path, bbox_inches="tight", **kwargs)
            paths.append(path)
        plt.close(figure)
    return csv_path, paths[0], paths[1], paths[2]


def _completion_chart(plt: Any, data: pd.DataFrame) -> Any:
    figure, axis = plt.subplots(figsize=(8.2, 4.5), constrained_layout=True)
    methods = list(data.method.astype(str))
    y = np.arange(len(data))
    succeeded = data.verified_succeeded_jobs.to_numpy(dtype=float)
    failed = data.failed_jobs.to_numpy(dtype=float) + data.incomplete_jobs.to_numpy(
        dtype=float
    )
    missing = data.missing_jobs.to_numpy(dtype=float)
    invalid = data.integrity_invalid_jobs.to_numpy(dtype=float)
    axis.barh(y, succeeded, color="#54A24B", label="verified succeeded")
    axis.barh(y, failed, left=succeeded, color="#E45756", label="failed/incomplete")
    axis.barh(y, missing, left=succeeded + failed, color="#B8B8B8", label="missing")
    axis.barh(
        y,
        invalid,
        left=succeeded + failed + missing,
        color="#B279A2",
        label="integrity invalid",
    )
    axis.set_yticks(y, methods)
    axis.set_xlabel("Expected Stage-2 jobs")
    axis.set_title("Completion and integrity ledger")
    axis.legend(frameon=False, ncol=2, fontsize=8)
    return figure


def _quality_uncertainty_chart(plt: Any, data: pd.DataFrame) -> Any:
    metrics = (
        "rf_kpe_all_mean_m",
        "root_translation_common_scale_mean_m",
        "root_yaw_mean_rad",
    )
    selected = data.loc[data.metric.isin(metrics)].copy()
    strata = [value for value in ALLOWED_STRATA if value in set(selected.stratum)]
    figure, axes = plt.subplots(
        len(metrics),
        max(1, len(strata)),
        figsize=(5.0 * max(1, len(strata)), 3.0 * len(metrics)),
        constrained_layout=True,
        squeeze=False,
    )
    units = {
        "rf_kpe_all_mean_m": "RF-KPE-all mean (m)",
        "root_translation_common_scale_mean_m": "Common-scale root error (m)",
        "root_yaw_mean_rad": "Root yaw error (rad)",
    }
    if strata:
        for column, stratum in enumerate(strata):
            panel = selected.loc[selected.stratum == stratum]
            methods = sorted(panel.method.unique())
            for row_index, metric in enumerate(metrics):
                axis = axes[row_index, column]
                values = panel.loc[panel.metric == metric]
                ybase = {method: index for index, method in enumerate(methods)}
                for row in values.itertuples():
                    y = ybase[row.method]
                    axis.errorbar(
                        row.sequence_mean,
                        y,
                        xerr=np.asarray(
                            [
                                [row.sequence_mean - row.bootstrap_ci95_low],
                                [row.bootstrap_ci95_high - row.sequence_mean],
                            ]
                        ),
                        fmt="o",
                        color=STRATUM_COLORS[stratum],
                        markersize=4.5,
                        capsize=2,
                    )
                axis.set_yticks(np.arange(len(methods)), methods, fontsize=7.5)
                axis.set_xlabel(f"{units[metric]} with 95% actor-cluster bootstrap CI")
                if row_index == 0:
                    axis.set_title(stratum.replace("_", " "))
    else:
        for axis in axes[:, 0]:
            axis.text(0.5, 0.5, "No verified quality rows", ha="center", va="center")
            axis.set_axis_off()
    figure.suptitle(
        "Quality uncertainty shown within evidence strata; no cross-stratum rank"
    )
    return figure


def _targeted_untracked_chart(plt: Any, data: pd.DataFrame) -> Any:
    strata = [value for value in ALLOWED_STRATA if value in set(data.stratum)]
    figure, axes = plt.subplots(
        1,
        max(1, len(strata)),
        figsize=(5.0 * max(1, len(strata)), 4.3),
        constrained_layout=True,
        squeeze=False,
    )
    for axis, stratum in zip(axes[0], strata, strict=False):
        panel = data.loc[data.stratum == stratum]
        for method, group in panel.groupby("method", sort=True):
            axis.scatter(
                group.rf_kpe_targeted_mean_m,
                group.rf_kpe_untracked_mean_m,
                s=18,
                alpha=0.35,
                color=STRATUM_COLORS[stratum],
            )
            axis.scatter(
                group.rf_kpe_targeted_mean_m.mean(),
                group.rf_kpe_untracked_mean_m.mean(),
                s=65,
                marker="D",
                color=STRATUM_COLORS[stratum],
                edgecolor="white",
                linewidth=0.8,
            )
            axis.annotate(
                method,
                (
                    group.rf_kpe_targeted_mean_m.mean(),
                    group.rf_kpe_untracked_mean_m.mean(),
                ),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7.5,
            )
        axis.set_xlabel("Targeted RF-KPE (m)")
        axis.set_ylabel("Untracked RF-KPE (m)")
        axis.set_title(stratum.replace("_", " "))
    if not strata:
        axes[0, 0].text(0.5, 0.5, "No verified quality rows", ha="center", va="center")
    figure.suptitle(
        "Per-sequence trade-off; diamonds are stratum-preserving method means"
    )
    return figure


def _artifact_chart(plt: Any, data: pd.DataFrame) -> Any:
    methods = list(
        data[["stratum", "method"]]
        .drop_duplicates()
        .sort_values(["stratum", "method"])
        .itertuples(index=False, name=None)
    )
    labels = [f"{method}\n[{stratum.split('_')[0]}]" for stratum, method in methods]
    x = np.arange(len(methods))
    figure, axis = plt.subplots(
        figsize=(max(8.0, len(methods) * 1.15), 4.5), constrained_layout=True
    )
    width = 0.24
    for offset, metric, label in (
        (-width, "foot_skating_frame_rate", "foot skating"),
        (0.0, "ground_penetration_frame_rate", "ground penetration"),
        (width, "artifact_rate", "union artifact"),
    ):
        means = [
            data.loc[(data.stratum == stratum) & (data.method == method), metric].mean()
            for stratum, method in methods
        ]
        axis.bar(x + offset, means, width, label=label)
    axis.set_xticks(x, labels, rotation=22, ha="right", fontsize=7.5)
    axis.set_ylabel("Mean sequence-level frame rate")
    axis.set_title("Named artifact rates; strata remain labelled")
    axis.legend(frameon=False, fontsize=8)
    return figure


def _paired_chart(plt: Any, data: pd.DataFrame) -> Any:
    selected = data.loc[
        data.metric.isin(("rf_kpe_all_mean_m", "root_translation_common_scale_mean_m"))
    ].copy()
    figure, axis = plt.subplots(
        figsize=(9.5, max(3.2, 0.38 * len(selected) + 1.8)), constrained_layout=True
    )
    if selected.empty:
        axis.text(0.5, 0.5, "No within-stratum method pair", ha="center", va="center")
        axis.set_axis_off()
        return figure
    selected["label"] = (
        selected.method_b.astype(str)
        + " − "
        + selected.method_a.astype(str)
        + " | "
        + selected.metric.astype(str)
    )
    y = np.arange(len(selected))
    for index, row in enumerate(selected.itertuples()):
        axis.errorbar(
            row.mean_paired_difference_b_minus_a,
            index,
            xerr=np.asarray(
                [
                    [row.mean_paired_difference_b_minus_a - row.bootstrap_ci95_low],
                    [row.bootstrap_ci95_high - row.mean_paired_difference_b_minus_a],
                ]
            ),
            fmt="o",
            color=STRATUM_COLORS[row.stratum],
            capsize=2,
        )
    axis.axvline(0.0, color="#222222", linewidth=0.9)
    axis.set_yticks(y, selected.label, fontsize=7.2)
    axis.set_xlabel("Paired sequence mean difference (method B − method A)")
    axis.set_title("Within-stratum paired effects only")
    return figure


def _reference_chart(plt: Any, data: pd.DataFrame) -> Any:
    selected = data.loc[
        data.metric.isin(("rf_kpe_all_mean_m", "root_translation_common_scale_mean_m"))
    ].copy()
    figure, axis = plt.subplots(
        figsize=(9.5, max(3.2, 0.38 * len(selected) + 1.8)), constrained_layout=True
    )
    if selected.empty:
        axis.text(
            0.5, 0.5, "No verified reference intersection", ha="center", va="center"
        )
        axis.set_axis_off()
        return figure
    selected["label"] = (
        selected.method.astype(str) + " | " + selected.metric.astype(str)
    )
    y = np.arange(len(selected))
    for index, row in enumerate(selected.itertuples()):
        axis.errorbar(
            row.mean_method_minus_reference,
            index,
            xerr=np.asarray(
                [
                    [row.mean_method_minus_reference - row.bootstrap_ci95_low],
                    [row.bootstrap_ci95_high - row.mean_method_minus_reference],
                ]
            ),
            fmt="o",
            color=STRATUM_COLORS[row.method_stratum],
            capsize=2,
        )
    axis.axvline(0.0, color="#222222", linewidth=0.9)
    axis.set_yticks(y, selected.label, fontsize=7.2)
    axis.set_xlabel("Method − external reference (descriptive)")
    axis.set_title("Filename intersection: zero means agreement, not truth")
    return figure


def _direct_reference_chart(plt: Any, data: pd.DataFrame) -> Any:
    if "metric" not in data:
        data = pd.DataFrame(columns=("metric",))
    selected = data.loc[
        data.metric.isin(
            (
                "fk_root_frame_semantic_agreement_mean_m",
                "root_anchor_frame0_agreement_m",
                "root_displacement_agreement_mean_m",
            )
        )
    ].copy()
    figure, axis = plt.subplots(
        figsize=(9.5, max(3.2, 0.38 * len(selected) + 1.8)), constrained_layout=True
    )
    if selected.empty:
        axis.text(
            0.5, 0.5, "No direct reference trajectory overlap", ha="center", va="center"
        )
        axis.set_axis_off()
        return figure
    selected["label"] = (
        selected.method.astype(str) + " | " + selected.metric.astype(str)
    )
    y = np.arange(len(selected))
    for index, row in enumerate(selected.itertuples()):
        axis.errorbar(
            row.sequence_mean_disagreement,
            index,
            xerr=np.asarray(
                [
                    [row.sequence_mean_disagreement - row.bootstrap_ci95_low],
                    [row.bootstrap_ci95_high - row.sequence_mean_disagreement],
                ]
            ),
            fmt="o",
            color=STRATUM_COLORS[row.method_stratum],
            capsize=2,
        )
    axis.set_yticks(y, selected.label, fontsize=7.2)
    axis.set_xlabel("Direct method↔external-reference trajectory disagreement")
    axis.set_title("Same G1, source frame, and timeline; lower is agreement, not truth")
    return figure


def build_stage2_charts(
    output_root: str | Path,
    ledger: LedgerBundle,
    tables: AnalysisTables,
) -> list[Path]:
    chart_specs = (
        ("completion_and_integrity", ledger.completion, _completion_chart),
        (
            "quality_sequence_uncertainty",
            tables.per_method_metric,
            _quality_uncertainty_chart,
        ),
        (
            "targeted_vs_untracked_by_stratum",
            tables.per_sequence_method,
            _targeted_untracked_chart,
        ),
        ("artifact_rates_by_stratum", tables.per_sequence_method, _artifact_chart),
        (
            "within_stratum_paired_differences",
            tables.within_stratum_pair_summary,
            _paired_chart,
        ),
        (
            "reference_intersection_differences",
            tables.reference_summary,
            _reference_chart,
        ),
        (
            "reference_direct_trajectory_agreement",
            tables.reference_direct_summary,
            _direct_reference_chart,
        ),
    )
    outputs: list[Path] = []
    for name, source, renderer in chart_specs:
        outputs.extend(save_stage2_chart(output_root, name, source, renderer))
    return outputs


def _markdown_table(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    labels: Mapping[str, str] | None = None,
    digits: int = 4,
) -> str:
    if frame.empty:
        return "No rows in this evidence stratum."
    selected = frame.loc[:, list(columns)].copy()
    for column in selected.select_dtypes(include=[np.number]).columns:
        selected[column] = selected[column].map(
            lambda value: f"{float(value):.{digits}f}" if pd.notna(value) else "N/A"
        )
    if labels:
        selected = selected.rename(columns=dict(labels))
    return selected.to_markdown(index=False)


def _stratum_narrative(per_method: pd.DataFrame) -> str:
    lines = []
    for stratum in ALLOWED_STRATA:
        panel = per_method.loc[
            (per_method.stratum == stratum) & (per_method.metric == "rf_kpe_all_mean_m")
        ]
        if panel.empty:
            continue
        methods = ", ".join(panel.method.astype(str))
        lower = float(panel.sequence_mean.min())
        upper = float(panel.sequence_mean.max())
        lines.append(
            f"- `{stratum}`: {methods}; sequence-level mean RF-KPE-all spans "
            f"{lower:.4f}–{upper:.4f} m. This is a within-stratum descriptive range."
        )
    return "\n".join(lines) if lines else "- No verified stratum has RF-KPE evidence."


def render_stage2_reports(
    output_root: str | Path,
    plan: Mapping[str, Any],
    ledger: LedgerBundle,
    tables: AnalysisTables,
    *,
    budget_compliant: bool | None = None,
) -> list[Path]:
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    complete = bool(
        ledger.completion.complete_for_selected_design.all()
        and budget_compliant is not False
    )
    status = "COMPLETE" if complete else "PARTIAL"
    expected_jobs = int(ledger.completion.expected_jobs.sum())
    succeeded_jobs = int(ledger.completion.verified_succeeded_jobs.sum())
    selected_sequences = len(plan["selected_sequences"])
    dataset_kind = str(plan.get("dataset_kind", "unknown"))
    bootstrap_rows = tables.per_method_metric.loc[
        tables.per_method_metric.metric.isin(
            (
                "rf_kpe_all_mean_m",
                "rf_kpe_targeted_mean_m",
                "rf_kpe_untracked_mean_m",
                "root_translation_common_scale_mean_m",
                "root_yaw_mean_rad",
                "artifact_rate",
            )
        )
    ]
    quality_table = _markdown_table(
        bootstrap_rows,
        (
            "stratum",
            "method",
            "metric",
            "sequence_count",
            "sequence_mean",
            "bootstrap_ci95_low",
            "bootstrap_ci95_high",
        ),
        labels={
            "stratum": "Evidence stratum",
            "method": "Method",
            "metric": "Metric",
            "sequence_count": "N sequences",
            "sequence_mean": "Sequence mean",
            "bootstrap_ci95_low": "CI low",
            "bootstrap_ci95_high": "CI high",
        },
    )
    completion_table = _markdown_table(
        ledger.completion,
        (
            "stratum",
            "method",
            "expected_jobs",
            "verified_succeeded_jobs",
            "failed_jobs",
            "incomplete_jobs",
            "missing_jobs",
            "integrity_invalid_jobs",
            "completion_fraction",
        ),
        labels={
            "stratum": "Stratum",
            "method": "Method",
            "expected_jobs": "Expected",
            "verified_succeeded_jobs": "Verified",
            "failed_jobs": "Failed",
            "incomplete_jobs": "Incomplete",
            "missing_jobs": "Missing",
            "integrity_invalid_jobs": "Invalid integrity",
            "completion_fraction": "Completion",
        },
    )
    scope_table = _markdown_table(
        ledger.method_scope,
        (
            "method",
            "stratum",
            "stage2_scope_status",
            "reason",
            "result_metrics_used_for_scope_decision",
        ),
        labels={
            "method": "Method",
            "stratum": "Stratum",
            "stage2_scope_status": "Scope status",
            "reason": "Reason",
            "result_metrics_used_for_scope_decision": "Result metrics used",
        },
    )
    pair_focus = tables.within_stratum_pair_summary.loc[
        tables.within_stratum_pair_summary.metric.isin(
            ("rf_kpe_all_mean_m", "root_translation_common_scale_mean_m")
        )
    ]
    pair_table = _markdown_table(
        pair_focus,
        (
            "stratum",
            "method_a",
            "method_b",
            "metric",
            "sequence_count",
            "mean_paired_difference_b_minus_a",
            "bootstrap_ci95_low",
            "bootstrap_ci95_high",
        ),
        labels={
            "stratum": "Stratum",
            "method_a": "Method A",
            "method_b": "Method B",
            "metric": "Metric",
            "sequence_count": "Paired N",
            "mean_paired_difference_b_minus_a": "B − A",
            "bootstrap_ci95_low": "CI low",
            "bootstrap_ci95_high": "CI high",
        },
    )
    reference_focus = tables.reference_summary.loc[
        tables.reference_summary.metric.isin(
            ("rf_kpe_all_mean_m", "root_translation_common_scale_mean_m")
        )
    ]
    reference_table = _markdown_table(
        reference_focus,
        (
            "method_stratum",
            "method",
            "metric",
            "intersection_sequence_count",
            "mean_method_minus_reference",
            "bootstrap_ci95_low",
            "bootstrap_ci95_high",
        ),
        labels={
            "method_stratum": "Method stratum",
            "method": "Method",
            "metric": "Metric",
            "intersection_sequence_count": "Intersection N",
            "mean_method_minus_reference": "Method − reference",
            "bootstrap_ci95_low": "CI low",
            "bootstrap_ci95_high": "CI high",
        },
    )
    direct_summary = tables.reference_direct_summary
    if "metric" not in direct_summary:
        direct_summary = pd.DataFrame(
            columns=(
                "method_stratum",
                "method",
                "metric",
                "intersection_sequence_count",
                "intersection_subject_count",
                "sequence_mean_disagreement",
                "bootstrap_ci95_low",
                "bootstrap_ci95_high",
            )
        )
    direct_focus = direct_summary.loc[
        direct_summary.metric.isin(
            (
                "fk_root_frame_semantic_agreement_mean_m",
                "root_translation_agreement_mean_m",
                "root_anchor_frame0_agreement_m",
                "root_displacement_agreement_mean_m",
                "joint_angle_agreement_rmse_rad",
            )
        )
    ]
    direct_reference_table = _markdown_table(
        direct_focus,
        (
            "method_stratum",
            "method",
            "metric",
            "intersection_sequence_count",
            "intersection_subject_count",
            "sequence_mean_disagreement",
            "bootstrap_ci95_low",
            "bootstrap_ci95_high",
        ),
        labels={
            "method_stratum": "Method stratum",
            "method": "Method",
            "metric": "Direct trajectory metric",
            "intersection_sequence_count": "N sequences",
            "intersection_subject_count": "N actors",
            "sequence_mean_disagreement": "Mean disagreement",
            "bootstrap_ci95_low": "CI low",
            "bootstrap_ci95_high": "CI high",
        },
    )
    failure_rows = ledger.ledger.loc[~ledger.ledger.included_in_quality_analysis]
    failure_table = _markdown_table(
        failure_rows,
        ("job_id", "status", "integrity_status", "message"),
        labels={
            "job_id": "Job",
            "status": "Status",
            "integrity_status": "Integrity",
            "message": "Reason",
        },
    )
    reference_sequences = (
        int(tables.reference_intersection.sequence_id.nunique())
        if not tables.reference_intersection.empty
        else 0
    )
    stratum_narrative = _stratum_narrative(tables.per_method_metric)
    calibration_table = _markdown_table(
        tables.calibration_per_sequence,
        (
            "sequence_id",
            "subject_id",
            "common_scale",
            "weighted_residual_rmse_m",
            "head_to_toe_diagnostic_scale",
            "root_anchor_translation_norm_m",
        ),
        labels={
            "sequence_id": "Sequence",
            "subject_id": "Actor label",
            "common_scale": "Registered LS scale",
            "weighted_residual_rmse_m": "Calibration residual (m)",
            "head_to_toe_diagnostic_scale": "Head/toe diagnostic",
            "root_anchor_translation_norm_m": "Anchor norm (m)",
        },
    )
    variability_focus = tables.calibration_variability.loc[
        tables.calibration_variability.scope.eq("all_sequences")
    ] if "scope" in tables.calibration_variability else tables.calibration_variability
    variability_table = _markdown_table(
        variability_focus,
        (
            "metric",
            "sequence_count",
            "mean",
            "standard_deviation",
            "minimum",
            "median",
            "maximum",
        ),
    )
    sensitivity_focus = tables.scale_sensitivity.loc[
        tables.scale_sensitivity.metric.isin(
            (
                "root_translation_common_scale_mean_m",
                "root_translation_scale_invariant_mean_m",
                "common_minus_scale_invariant_root_error",
            )
        )
    ] if "metric" in tables.scale_sensitivity else tables.scale_sensitivity
    sensitivity_table = _markdown_table(
        sensitivity_focus,
        (
            "stratum",
            "method",
            "metric",
            "sequence_count",
            "metric_mean",
            "ols_slope_per_scale_unit",
            "spearman_correlation",
        ),
    )
    conclusion_table = _markdown_table(
        tables.conclusion_ledger,
        (
            "question",
            "evidence_stratum",
            "methods",
            "sequence_count",
            "allowed_conclusion",
            "forbidden_conclusion",
        ),
    )

    report = f"""# Stage 2 LAFAN Analysis Report — {status}

## Scope and completion

This report aggregates immutable plan `{plan["design_id"]}` (`{dataset_kind}`): {selected_sequences} selected source sequences and {expected_jobs} expected method×sequence jobs. Exactly {succeeded_jobs} jobs passed status, provenance, byte-hash, full-frame, valid-frame, and stratum verification. The analysis is **{status}** for the selected design; production builds also require the final 48-hour/200-GB accounting check to pass. Missing or failed work is never interpolated or relabelled as success.

Budget accounting is cumulative across every finalized execution attempt and every failed, interrupted, repeated, or published analysis attempt. Retained storage is measured from the run/publication trees and then adds the selected raw BVH/reference-CSV baseline exactly once. Editable `execution_summary.json` and analysis-manifest totals are not decision authorities; final validation recomputes these quantities from attempt ledgers and bytes.

{completion_table}

![Completion ledger](figures/completion_and_integrity.svg)

## Scientific strata

{scope_table}

The controlled stratum uses one method-independent common scale recomputed and frozen separately for each source sequence. Native-public pipelines retain their own preprocessing. The benchmark-port stratum contains public retargeter logic with a non-upstream LAFAN input adapter and is not relabelled as an official LAFAN path. The external reference is a precomputed filename-intersection corpus. These are different interventions: results are displayed in separate strata and the report makes no cross-stratum causal rank. Actor labels are used only as dependence clusters for uncertainty, never as the calibration unit.

### Calibration and scale variability

{calibration_table}

{variability_table}

Head/toe scale is diagnostic only. The registered shared-landmark least-squares scale, its residual, and its root anchor remain separate fields so morphology fit and world placement cannot be conflated.

### Scale-sensitivity diagnostics

{sensitivity_table}

These slopes/correlations are observational across sequences: actor, motion, frame-0 pose, and calibration co-vary. The common-vs-scale-invariant root contrast is a metric-definition check. Neither table is a counterfactual retargeter rerun, so neither supports a causal claim about changing an official scale policy.

## Per-method sequence uncertainty

{quality_table}

The primary aggregation unit is a sequence, not a frame. Means and 95% intervals use {BOOTSTRAP_RESAMPLES} deterministic actor-cluster resamples with seed {BOOTSTRAP_SEED}; every sampled actor contributes all of their sequences, so repeated motions from one performer are not treated as independent actors. Long sequences do not silently dominate the primary estimate. `leave_one_subject_out.csv` exposes each actor's influence.

{stratum_narrative}

![Quality uncertainty](figures/quality_sequence_uncertainty.svg)

## Targeted and untracked pose

RF-KPE is root-frame keypoint position error under each sequence's frozen common evaluator scale. Targeted wrists/ankles and untracked body landmarks remain separate. Root translation and yaw are separate metrics rather than hidden inside RF-KPE.

![Targeted and untracked](figures/targeted_vs_untracked_by_stratum.svg)

## Temporal and artifact evidence

Artifact rate is the union of named per-frame causes, not a method label. Foot skating, penetration, joint-limit, invalid-frame, jerk, and pose-jump fields remain available in `per_sequence_method_summary.csv` and the paired tables.

![Artifact evidence](figures/artifact_rates_by_stratum.svg)

## Within-stratum paired differences

{pair_table}

Every row pairs the same source sequence and reports method B minus method A. Pairs are constructed only within one evidence stratum; no controlled-vs-native causal effect is inferred.

![Within-stratum pairs](figures/within_stratum_paired_differences.svg)

## External-reference intersection

The verified reference intersection contributes {reference_sequences} source sequences to descriptive difference estimates.

{reference_table}

`Method − reference` measures agreement with one external trajectory. The reference has no comparable runtime provenance and is not verified official ground truth, an accuracy target, or a quality upper bound. A difference nearer zero cannot by itself establish better retargeting.

![Reference intersection](figures/reference_intersection_differences.svg)

### Direct audited-G1 trajectory agreement

{direct_reference_table}

These rows compare method and reference qpos on a kinematically audited G1 contract, paired by normalized sequence basename and integer frame index only. They report absolute root translation, frame-0 root-anchor offset, frame-0-subtracted root displacement, root yaw, joint angles and velocities, and MuJoCo FK landmarks. Byte-identical human inputs and exact timestamp identity are not established. Unlike the metric-delta table above, these are direct trajectory-disagreement measurements, but they still measure resemblance to an externally supplied trajectory rather than error to verified truth.

The audit binds and rechecks the reference URDF, canonical evaluator URDF, canonical MuJoCo scene, evaluator manifest, 29-joint order/tree/origins/axes/limits, and deterministic neutral/random-pose FK. “Audited G1” means kinematic-contract equivalence only; geometry, collision, inertia, actuation, and dynamics equivalence are outside this claim.

![Direct trajectory agreement](figures/reference_direct_trajectory_agreement.svg)

## Failure and exclusion ledger

{failure_table}

Plan-level method exclusions are budget decisions registered before Stage-2 result metrics existed. They remain in `method_scope_ledger.csv` and do not erase Stage-1 evidence. If this report is partial, its numerical rows describe only verified successes and must not be presented as the completed selected-design comparison.

## Conclusion ledger

{conclusion_table}

The ledger is the claim boundary: it records what each evidence stratum can and cannot establish. Numerical ordering across strata is never promoted to a solver-effect conclusion.

## Reproducibility boundary

The builder reads only the frozen plan, sequence manifests, succeeded job manifests, and their hashed canonical packages. It launches no job. Per-run evaluator CSV/Parquet, per-sequence and per-method summaries, paired tables, bootstrap seeds, exact chart source CSVs, cumulative analysis-attempt records, and the final analysis manifest are retained below this analysis directory. Each build stages privately; the manifest is replaced last as the publication commit point.
"""

    review = f"""# Stage 2 Analysis Review — {status}

## Design review

- **Plan freeze:** plan SHA-256, unique job IDs, selected sequence inventory, method specs, and the explicit cross-stratum-ranking prohibition are validated before reading results.
- **Per-sequence evaluator:** every method on a source sequence uses the same common morphology scale, root anchor, heading definition, thresholds, and canonical G1 asset hash; calibration is recomputed per sequence, not per actor.
- **Paired design:** same-sequence differences are formed only within one evidence stratum.
- **Uncertainty:** the performer is the resampling cluster, all sequences from a sampled actor move together, and leave-one-subject-out influence is serialized.
- **Reference boundary:** the external corpus is untimed and descriptive, never a truth label; direct pairing proves normalized-basename/frame-index overlap, not byte-identical source motion or exact timestamp identity.

## Execution-evidence review

The selected design expected {expected_jobs} jobs and has {succeeded_jobs} verified successes. Claimed successes require an exact output SHA-256, exact sequence-manifest SHA-256, exact canonical-human SHA-256, full source-frame coverage, all `valid=True`, completion ratio 1.0, matching plan identity, and matching output stratum. A violation aborts the quality build after preserving the integrity ledger.

Final validation does not accept counts or budget flags copied from `analysis_manifest.json`. It reloads the exact tracked/run plan pair, re-hashes the plan decision basis, raw source/reference files, payload-hashed sequence and job manifests, execution contracts, canonical outputs, and reference/evaluator assets; it then recomputes cumulative wall/storage and decisive counts.

## Statistical review

Sequence-level means avoid frame-count weighting, while actor-cluster bootstrap intervals avoid treating motions from the same performer as independent people. Leave-one-subject-out rows expose actor influence. Raw per-sequence rows remain available for alternative preregistered estimands. Intervals describe actor/sequence variability in this selected design; they are not confidence bounds for all human motion domains. No multiple-comparison-adjusted hypothesis testing or universal winner claim is made.

## Completeness review

{completion_table}

Status is **{status}**. A partial build is useful for diagnosing successful jobs and failures, but it is not a completed Stage-2 benchmark. A complete build supports selected-design, stratum-specific descriptive conclusions only.

## Threats to validity

- Native, controlled, and benchmark-port strata differ in preprocessing and/or input integration, so cross-stratum numerical ordering is not a solver effect.
- Kinematic metrics do not establish torque feasibility, balance, dynamics, or controller trackability.
- LAFAN coverage does not establish performance on AMASS or interaction datasets.
- The reference intersection is smaller than the source inventory and its method provenance is incomplete.
- Stage-2 production wall time is one pass and is retained for operations, not treated as the repeated Stage-1 timing experiment.
- Plan-level budget exclusion can change method coverage; excluded methods must stay visible in scope reporting.

## Audit conclusion

The aggregation design is internally coherent and fail-closed. The evidence-package conclusion is **{status}** for plan `{plan["design_id"]}`; all substantive claims must retain the plan's dataset scope, method scope, and evidence stratum.
"""

    reproduce = f"""# Reproduce Stage 2 Analysis

From the repository environment with evaluation dependencies installed:

```bash
python -c "from retargeting_comparison.stage2_analysis import build_stage2_analysis; build_stage2_analysis('.', '{plan.get("_analysis_plan_path", "runs/stage2/<design>/plan.json")}')"
```

The function does not launch Stage 2. It verifies plan and artifact hashes, rebuilds per-sequence evaluator contracts, evaluates only verified succeeded jobs, writes completion and exclusion ledgers, performs deterministic actor-cluster bootstrap and leave-one-subject-out analysis with seed {BOOTSTRAP_SEED}, computes direct G1 disagreement on the normalized-basename/frame-index overlap, and regenerates every report figure from its adjacent source CSV. Every invocation creates a numbered analysis attempt; failed/rerun wall time and retained staging bytes remain charged.

If any manifest that claims success has an invalid hash or incomplete output, the function stops after writing the ledger. Missing or genuinely failed jobs remain explicit and produce a partial analysis rather than synthetic trajectories.
"""
    bodies = {
        "STAGE2_REPORT.md": report,
        "STAGE2_REVIEW.md": review,
        "REPRODUCE_STAGE2_ANALYSIS.md": reproduce,
    }
    paths: list[Path] = []
    for name, body in bodies.items():
        path = output / name
        atomic_write_text(path, body.strip() + "\n")
        paths.append(path)
    return paths


def build_stage2_analysis(
    repo_root: str | Path,
    plan_path: str | Path,
) -> dict[str, Any]:
    """Build Stage-2 evidence as a cumulatively budgeted atomic attempt."""

    invocation_started = time.perf_counter()
    root = Path(repo_root).resolve()
    plan_value, resolved_plan, run_root = load_stage2_analysis_plan(root, plan_path)
    plan = dict(plan_value)
    plan["_analysis_plan_path"] = _portable(root, resolved_plan)
    tracked_plan = _resolve(
        root,
        str(
            plan.get("tracked_plan_path")
            or Path("stage2_results") / str(plan["design_id"]) / "STAGE2_PLAN.json"
        ),
    )
    analysis_root = tracked_plan.parent
    analysis_root.mkdir(parents=True, exist_ok=True)
    plan_snapshot = {
        key: value for key, value in plan.items() if key != "_analysis_plan_path"
    }
    if tracked_plan.is_file():
        try:
            existing_snapshot = json.loads(tracked_plan.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise Stage2AnalysisError("Tracked Stage-2 plan is unreadable") from error
        if existing_snapshot != plan_snapshot:
            raise Stage2AnalysisError(
                f"Tracked Stage-2 plan is not exactly equal to the run plan: {tracked_plan}"
            )

    attempt_started = invocation_started
    attempt_path, staging, attempt = _begin_analysis_attempt(analysis_root, plan)
    manifest: dict[str, Any] | None = None
    try:
        def live_guard() -> None:
            _enforce_live_analysis_budget(
                plan,
                run_root,
                analysis_root,
                attempt_path=attempt_path,
                attempt_started_monotonic=attempt_started,
            )

        _enforce_live_analysis_budget(
            plan,
            run_root,
            analysis_root,
            attempt_path=attempt_path,
            attempt_started_monotonic=attempt_started,
        )
        reference_contract_evidence = verify_unitree_reference_evaluator_contract(
            root, plan
        )
        atomic_write_json(staging / "STAGE2_PLAN.json", plan_snapshot)
        metrics_root = staging / "metrics"
        ledger = collect_stage2_job_ledger(root, plan, run_root, metrics_root)
        _enforce_live_analysis_budget(
            plan,
            run_root,
            analysis_root,
            attempt_path=attempt_path,
            attempt_started_monotonic=attempt_started,
        )
        per_sequence = evaluate_verified_stage2_jobs(
            root, ledger.verified, metrics_root, budget_guard=live_guard
        )
        tables = aggregate_stage2_tables(per_sequence, metrics_root)
        calibration, variability, sensitivity, conclusion = (
            build_calibration_scale_tables(ledger.verified, per_sequence, metrics_root)
        )
        tables.calibration_per_sequence = calibration
        tables.calibration_variability = variability
        tables.scale_sensitivity = sensitivity
        tables.conclusion_ledger = conclusion
        robot = CanonicalRobotModel(default_robot_scene(root))
        direct, direct_summary = build_direct_reference_trajectory_metrics(
            ledger.verified, robot, budget_guard=live_guard
        )
        _write_frame(
            direct, metrics_root / "reference_direct_trajectory_per_sequence"
        )
        _write_frame(
            direct_summary, metrics_root / "reference_direct_trajectory_summary"
        )
        tables.reference_direct_trajectory = direct
        tables.reference_direct_summary = direct_summary
        _enforce_live_analysis_budget(
            plan,
            run_root,
            analysis_root,
            attempt_path=attempt_path,
            attempt_started_monotonic=attempt_started,
        )
        chart_paths = build_stage2_charts(staging, ledger, tables)
        live_budget = _enforce_live_analysis_budget(
            plan,
            run_root,
            analysis_root,
            attempt_path=attempt_path,
            attempt_started_monotonic=attempt_started,
        )
        report_paths = render_stage2_reports(
            staging,
            plan,
            ledger,
            tables,
            budget_compliant=(
                live_budget.wall_compliant and live_budget.storage_compliant
            ),
        )
        final_budget = _enforce_live_analysis_budget(
            plan,
            run_root,
            analysis_root,
            attempt_path=attempt_path,
            attempt_started_monotonic=attempt_started,
        )
        dataset_complete = bool(
            ledger.completion.complete_for_selected_design.all()
        )
        budget_compliant = bool(
            final_budget.wall_compliant and final_budget.storage_compliant
        )
        analysis_status = (
            "complete" if dataset_complete and budget_compliant else "partial"
        )
        evidence_paths = [path for path in metrics_root.rglob("*") if path.is_file()]
        evidence_paths.extend(chart_paths)
        evidence_paths.extend(report_paths)
        evidence_paths.append(staging / "STAGE2_PLAN.json")
        evidence_paths = list(dict.fromkeys(sorted(evidence_paths)))
        output_rows = [
            {
                "path": str(path.relative_to(staging)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in evidence_paths
        ]
        source_bindings: dict[str, dict[str, Any]] = {}
        job_bindings: list[dict[str, Any]] = []
        for verified in sorted(ledger.verified, key=lambda value: value.job_id):
            sequence_value = json.loads(
                verified.sequence_manifest_path.read_text(encoding="utf-8")
            )
            canonical_path = _resolve(
                root, str(sequence_value["canonical_path"])
            )
            source_bindings[verified.sequence_id] = {
                "sequence_id": verified.sequence_id,
                "sequence_manifest_sha256": sha256_file(
                    verified.sequence_manifest_path
                ),
                "sequence_payload_sha256": str(
                    sequence_value.get("payload_sha256", "")
                ),
                "source_sha256": str(sequence_value["source_sha256"]),
                "canonical_source_sha256": sha256_file(canonical_path),
                "calibration_evidence_sha256": str(
                    sequence_value.get("calibration_evidence_sha256", "")
                ),
            }
            job_value = json.loads(
                verified.job_manifest_path.read_text(encoding="utf-8")
            )
            job_bindings.append(
                {
                    "job_id": verified.job_id,
                    "job_manifest_sha256": sha256_file(
                        verified.job_manifest_path
                    ),
                    "sequence_manifest_sha256": sha256_file(
                        verified.sequence_manifest_path
                    ),
                    "job_contract_sha256": str(
                        job_value.get("job_contract_sha256", "")
                    ),
                    "method_contract_sha256": str(
                        job_value.get("method_contract_sha256", "")
                    ),
                    "execution_contract_sha256": str(
                        job_value.get("execution_contract_sha256", "")
                    ),
                    "output_sha256": sha256_file(verified.output_path),
                    "output_bytes": verified.output_path.stat().st_size,
                    "output_qpos_sha256": str(
                        job_value.get("output_qpos_sha256", "")
                    ),
                }
            )
        source_binding_rows = [source_bindings[key] for key in sorted(source_bindings)]
        manifest = {
            "schema_version": ANALYSIS_SCHEMA_VERSION,
            "stage": 2,
            "status": analysis_status,
            "analysis_status": analysis_status,
            "analysis_attempt_id": str(attempt["attempt_id"]),
            "design_id": str(plan["design_id"]),
            "dataset_kind": str(plan.get("dataset_kind")),
            "plan_path": str(resolved_plan),
            "plan_sha256": str(plan["plan_sha256"]),
            "plan_basis_sha256": str(plan.get("plan_basis_sha256", "")),
            "tracked_plan_exact_equality_verified": True,
            "selected_sequence_count": len(plan["selected_sequences"]),
            "expected_job_count": int(ledger.completion.expected_jobs.sum()),
            "verified_succeeded_job_count": int(
                ledger.completion.verified_succeeded_jobs.sum()
            ),
            "source_evidence_sha256": _canonical_sha256(source_binding_rows),
            "verified_job_output_evidence_sha256": _canonical_sha256(job_bindings),
            "analysis_wall_s": final_budget.current_analysis_attempt_wall_s,
            "cumulative_execution_wall_s": final_budget.execution_attempt_wall_s,
            "cumulative_prior_analysis_wall_s": (
                final_budget.prior_analysis_attempt_wall_s
            ),
            "total_accounted_stage2_wall_s": final_budget.total_accounted_wall_s,
            "wall_budget_compliant": final_budget.wall_compliant,
            "raw_input_baseline_bytes": final_budget.raw_input_baseline_bytes,
            "retained_tree_bytes": final_budget.retained_tree_bytes,
            "total_accounted_retained_bytes": (
                final_budget.total_accounted_retained_bytes
            ),
            "storage_budget_compliant": final_budget.storage_compliant,
            "budget_compliant": budget_compliant,
            "budget_evidence_sha256": final_budget.evidence_sha256,
            "execution_attempt_count": final_budget.execution_attempt_count,
            "prior_finalized_analysis_attempt_count": (
                final_budget.finalized_analysis_attempt_count
            ),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "uncertainty_unit": "subject_cluster_with_sequence_level_estimand",
            "leave_one_subject_out_performed": True,
            "calibration_unit": "one frozen estimate per source sequence",
            "scale_sensitivity_is_causal_intervention": False,
            "evidence_strata": list(ALLOWED_STRATA),
            "cross_stratum_causal_ranking_performed": False,
            "external_reference_verified_ground_truth": False,
            "external_reference_pairing": (
                "normalized basename and integer frame index only; byte-identical "
                "source and exact timestamp identity not established"
            ),
            "external_reference_in_timing_ranking": False,
            "unitree_reference_evaluator_contract": reference_contract_evidence,
            "direct_reference_trajectory_rows": len(
                tables.reference_direct_trajectory
            ),
            "stage2_jobs_launched_by_analysis": False,
            "outputs": output_rows,
            "output_evidence_sha256": _canonical_sha256(output_rows),
        }
        atomic_write_json(staging / "analysis_manifest.json", manifest)
        ready = _finalize_analysis_attempt(
            attempt_path,
            attempt,
            started_monotonic=attempt_started,
            status="ready_to_publish",
            published=False,
            error=None,
            publication_manifest_sha256=sha256_file(
                staging / "analysis_manifest.json"
            ),
        )
        post_attempt_budget = recompute_stage2_budget_evidence(
            plan, run_root, analysis_root
        )
        if not (
            post_attempt_budget.wall_compliant
            and post_attempt_budget.storage_compliant
        ):
            raise Stage2AnalysisError(
                "Finalized analysis attempt exceeds a hard Stage-2 budget"
            )
        _publish_staging_tree(staging, analysis_root)
        published_manifest = analysis_root / "analysis_manifest.json"
        os.replace(staging / "analysis_manifest.json", published_manifest)
        ready.update(
            {
                "status": "published",
                "published": True,
                "published_at_utc": _utc_now(),
                "publication_manifest_sha256": sha256_file(published_manifest),
            }
        )
        _write_analysis_attempt(attempt_path, ready)
        return manifest
    except BaseException as error:
        _finalize_analysis_attempt(
            attempt_path,
            attempt,
            started_monotonic=attempt_started,
            status="failed",
            published=False,
            error=f"{type(error).__name__}: {error}",
        )
        raise


def recompute_stage2_decision_evidence(
    repo_root: str | Path,
    plan: Mapping[str, Any],
    run_root: str | Path,
    analysis_root: str | Path,
    *,
    current_attempt_path: Path | None = None,
    current_analysis_wall_s: float = 0.0,
) -> dict[str, Any]:
    """Recompute every decisive plan/source/job/output/budget fact from bytes."""

    root = Path(repo_root).resolve()
    run = Path(run_root).resolve()
    analysis = Path(analysis_root).resolve()
    jobs = list(plan.get("projection", {}).get("jobs", []))
    bundle = collect_stage2_job_ledger(
        root, plan, run, analysis / ".validation-scratch", persist=False
    )
    config_path = _resolve(root, str(plan.get("config_path", "")))
    if not config_path.is_file() or sha256_file(config_path) != str(
        plan.get("config_sha256")
    ):
        raise Stage2AnalysisError("Stage-2 config no longer matches the plan")
    from .io_utils import load_yaml

    config = load_yaml(config_path)
    source_rows: list[dict[str, Any]] = []
    for selected in plan.get("selected_sequences", []):
        source = root / str(config["dataset"]["root"]) / str(
            selected["relative_path"]
        )
        source_row = {
            "sequence_id": str(selected["sequence_id"]),
            "path": _portable(root, source),
            "sha256": sha256_file(source) if source.is_file() else "missing",
            "bytes": source.stat().st_size if source.is_file() else -1,
        }
        if (
            source_row["sha256"] != str(selected["source_sha256"])
            or source_row["bytes"] != int(selected["source_size_bytes"])
        ):
            raise Stage2AnalysisError(
                f"Selected raw source changed: {selected['sequence_id']}"
            )
        reference_relative = selected.get("reference_relative_path")
        if reference_relative is not None:
            reference = root / str(config["reference_corpus"]["root"]) / str(
                reference_relative
            )
            reference_row = {
                "path": _portable(root, reference),
                "sha256": sha256_file(reference)
                if reference.is_file()
                else "missing",
                "bytes": reference.stat().st_size if reference.is_file() else -1,
            }
            if (
                reference_row["sha256"] != str(selected["reference_sha256"])
                or reference_row["bytes"] != int(selected["reference_size_bytes"])
            ):
                raise Stage2AnalysisError(
                    f"Selected raw reference changed: {selected['sequence_id']}"
                )
            source_row["reference"] = reference_row
        source_rows.append(source_row)

    output_rows: list[dict[str, Any]] = []
    for verified in sorted(bundle.verified, key=lambda value: value.job_id):
        manifest = json.loads(
            verified.job_manifest_path.read_text(encoding="utf-8")
        )
        output_rows.append(
            {
                "job_id": verified.job_id,
                "job_manifest_sha256": sha256_file(verified.job_manifest_path),
                "sequence_manifest_sha256": sha256_file(
                    verified.sequence_manifest_path
                ),
                "output_sha256": sha256_file(verified.output_path),
                "output_bytes": verified.output_path.stat().st_size,
                "output_qpos_sha256": str(manifest.get("output_qpos_sha256")),
                "job_contract_sha256": str(
                    manifest.get("job_contract_sha256")
                ),
                "method_contract_sha256": str(
                    manifest.get("method_contract_sha256")
                ),
                "execution_contract_sha256": str(
                    manifest.get("execution_contract_sha256")
                ),
            }
        )
    budget = recompute_stage2_budget_evidence(
        plan,
        run,
        analysis,
        current_analysis_wall_s=current_analysis_wall_s,
        current_attempt_path=current_attempt_path,
    )
    reference = verify_unitree_reference_evaluator_contract(root, plan)
    tracked_plan = analysis / "STAGE2_PLAN.json"
    tracked_exact = False
    if tracked_plan.is_file():
        tracked_exact = json.loads(tracked_plan.read_text(encoding="utf-8")) == dict(
            plan
        )
    evidence = {
        "plan_sha256": str(plan["plan_sha256"]),
        "plan_basis_sha256": str(plan.get("plan_basis_sha256", "")),
        "tracked_plan_exact_equality": tracked_exact,
        "selected_sequence_count": len(plan.get("selected_sequences", [])),
        "expected_job_count": len(jobs),
        "verified_succeeded_job_count": len(bundle.verified),
        "source_rows": source_rows,
        "source_evidence_sha256": _canonical_sha256(source_rows),
        "output_rows": output_rows,
        "output_evidence_sha256": _canonical_sha256(output_rows),
        "budget": {
            "execution_attempt_wall_s": budget.execution_attempt_wall_s,
            "analysis_attempt_wall_s": budget.prior_analysis_attempt_wall_s,
            "total_accounted_wall_s": budget.total_accounted_wall_s,
            "raw_input_baseline_bytes": budget.raw_input_baseline_bytes,
            "retained_tree_bytes": budget.retained_tree_bytes,
            "total_accounted_retained_bytes": budget.total_accounted_retained_bytes,
            "wall_limit_s": budget.wall_limit_s,
            "storage_limit_bytes": budget.storage_limit_bytes,
            "wall_compliant": budget.wall_compliant,
            "storage_compliant": budget.storage_compliant,
            "evidence_sha256": budget.evidence_sha256,
            "execution_attempt_count": budget.execution_attempt_count,
            "finalized_analysis_attempt_count": (
                budget.finalized_analysis_attempt_count
            ),
        },
        "reference_contract": reference,
    }
    evidence["decision_basis_sha256"] = _canonical_sha256(evidence)
    return evidence


def validate_stage2(
    repo_root: str | Path,
    plan_path: str | Path,
) -> dict[str, Any]:
    """Validate Stage 2 by recomputing decisive evidence, not trusting summaries."""

    validation_started = time.perf_counter()
    root = Path(repo_root).resolve()
    plan, resolved_plan, run_root = load_stage2_analysis_plan(root, plan_path)
    analysis_root = _resolve(
        root,
        str(
            Path(
                plan.get("tracked_plan_path")
                or Path("stage2_results") / str(plan["design_id"]) / "STAGE2_PLAN.json"
            ).parent
        ),
    )
    analysis_root.mkdir(parents=True, exist_ok=True)
    validation_attempt_path, validation_staging, validation_attempt = (
        _begin_analysis_attempt(
            analysis_root, plan, attempt_kind="stage2_validation"
        )
    )
    manifest_path = analysis_root / "analysis_manifest.json"
    checks: dict[str, bool] = {
        "analysis_manifest_exists": manifest_path.is_file(),
        "plan_hash_valid": verify_plan_hash(plan),
    }
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
    decision_evidence: dict[str, Any] = {}
    decision_error: str | None = None
    try:
        decision_evidence = recompute_stage2_decision_evidence(
            root,
            plan,
            run_root,
            analysis_root,
            current_attempt_path=validation_attempt_path,
            current_analysis_wall_s=time.perf_counter() - validation_started,
        )
    except (OSError, ValueError, KeyError, Stage2AnalysisError) as error:
        decision_error = f"{type(error).__name__}: {error}"
    expected_jobs = len(plan.get("projection", {}).get("jobs", []))
    selected_sequences = len(plan.get("selected_sequences", []))
    budget = dict(decision_evidence.get("budget", {}))
    checks.update(
        {
            "analysis_complete": manifest.get("status") == "complete",
            "independent_decision_evidence_recomputed": decision_error is None,
            "hard_budget_compliant": (
                budget.get("wall_compliant") is True
                and budget.get("storage_compliant") is True
                and int(budget.get("total_accounted_retained_bytes", 2**63 - 1))
                + ANALYSIS_PUBLICATION_RESERVE_BYTES
                <= int(budget.get("storage_limit_bytes", -1))
            ),
            "analysis_plan_bound": manifest.get("plan_sha256")
            == plan.get("plan_sha256"),
            "tracked_plan_exact_equality": decision_evidence.get(
                "tracked_plan_exact_equality"
            )
            is True,
            "selected_source_count_recomputed": int(
                decision_evidence.get("selected_sequence_count", -1)
            )
            == selected_sequences,
            "all_expected_jobs_verified": (
                int(decision_evidence.get("expected_job_count", -1))
                == expected_jobs
                == int(
                    decision_evidence.get("verified_succeeded_job_count", -2)
                )
            ),
            "job_output_evidence_binding": (
                manifest.get("verified_job_output_evidence_sha256")
                == decision_evidence.get("output_evidence_sha256")
            ),
            "cross_stratum_rank_forbidden": (
                manifest.get("cross_stratum_causal_ranking_performed") is False
            ),
            "reference_not_ground_truth": (
                manifest.get("external_reference_verified_ground_truth") is False
            ),
            "actor_cluster_uncertainty": (
                manifest.get("uncertainty_unit")
                == "subject_cluster_with_sequence_level_estimand"
                and manifest.get("leave_one_subject_out_performed") is True
            ),
            "per_sequence_calibration_declared": (
                manifest.get("calibration_unit")
                == "one frozen estimate per source sequence"
                and manifest.get("scale_sensitivity_is_causal_intervention") is False
            ),
            "reference_evaluator_contract_verified": (
                decision_evidence.get("reference_contract", {}).get("verified")
                is True
            ),
        }
    )
    output_integrity = True
    for value in manifest.get("outputs", []):
        path = analysis_root / str(value.get("path", ""))
        if (
            not path.is_file()
            or sha256_file(path) != str(value.get("sha256"))
            or path.stat().st_size != int(value.get("bytes", -1))
        ):
            output_integrity = False
            break
    checks["analysis_output_hashes"] = (
        bool(manifest.get("outputs")) and output_integrity
    )
    required = (
        "STAGE2_PLAN.json",
        "STAGE2_REPORT.md",
        "STAGE2_REVIEW.md",
        "REPRODUCE_STAGE2_ANALYSIS.md",
        "metrics/completion_by_method.csv",
        "metrics/leave_one_subject_out.csv",
        "metrics/reference_direct_trajectory_per_sequence.csv",
        "metrics/reference_direct_trajectory_summary.csv",
        "metrics/calibration_per_sequence.csv",
        "metrics/calibration_scale_variability.csv",
        "metrics/scale_sensitivity_diagnostics.csv",
        "metrics/conclusion_ledger.csv",
    )
    checks["tracked_deliverables_complete"] = all(
        (analysis_root / relative).is_file() for relative in required
    )
    external_jobs = [
        job
        for job in plan.get("projection", {}).get("jobs", [])
        if job.get("stratum") == EXTERNAL_STRATUM
    ]
    non_external_methods = {
        str(job["method"])
        for job in plan.get("projection", {}).get("jobs", [])
        if job.get("stratum") != EXTERNAL_STRATUM
    }
    expected_direct = len(external_jobs) * len(non_external_methods)
    direct_path = analysis_root / "metrics/reference_direct_trajectory_per_sequence.csv"
    direct_frame = pd.read_csv(direct_path) if direct_path.is_file() else pd.DataFrame()
    direct_rows = len(direct_frame) if direct_path.is_file() else -1
    checks["direct_reference_trajectory_complete"] = (
        expected_direct == 0 or direct_rows == expected_direct
    )
    direct_required = set(DIRECT_REFERENCE_METRICS) | {
        "method_output_sha256",
        "reference_output_sha256",
        "interpretation",
    }
    independently_verified_output_hashes = {
        str(row["job_id"]): str(row["output_sha256"])
        for row in decision_evidence.get("output_rows", [])
    }
    direct_hashes_match = False
    if not direct_frame.empty and direct_required.issubset(direct_frame.columns):
        direct_hashes_match = all(
            str(row.method_output_sha256)
            == independently_verified_output_hashes.get(
                f"{row.method}__{row.sequence_id}"
            )
            and str(row.reference_output_sha256)
            == independently_verified_output_hashes.get(
                f"{row.reference_method}__{row.sequence_id}"
            )
            for row in direct_frame.itertuples()
        )
    checks["direct_reference_anchor_displacement_split"] = (
        expected_direct == 0
        or (
            direct_required.issubset(direct_frame.columns)
            and direct_frame.interpretation.astype(str)
            .str.contains("basename and integer frame index", regex=False)
            .all()
            and direct_frame.interpretation.astype(str)
            .str.contains("exact timestamp identity", regex=False)
            .all()
            and direct_hashes_match
        )
    )
    selected_actor_by_sequence = {
        str(value["sequence_id"]): _subject_id(str(value["relative_path"]))
        for value in plan.get("selected_sequences", [])
    }
    expected_selected_actors = set(selected_actor_by_sequence.values())
    loso_path = analysis_root / "metrics/leave_one_subject_out.csv"
    observed_loso_actors = (
        set(pd.read_csv(loso_path).held_out_subject_id.astype(str))
        if loso_path.is_file()
        else set()
    )
    checks["leave_one_subject_out_actor_coverage"] = (
        observed_loso_actors == expected_selected_actors
    )
    expected_reference_actors = {
        selected_actor_by_sequence[str(job["sequence_id"])] for job in external_jobs
    }
    observed_direct_actors = (
        set(direct_frame.subject_id.astype(str))
        if direct_path.is_file() and direct_rows > 0
        else set()
    )
    checks["direct_reference_actor_coverage"] = (
        not expected_reference_actors
        or observed_direct_actors == expected_reference_actors
    )
    calibration_path = analysis_root / "metrics/calibration_per_sequence.csv"
    calibration = (
        pd.read_csv(calibration_path) if calibration_path.is_file() else pd.DataFrame()
    )
    expected_sequence_hashes = {
        str(row["sequence_manifest_sha256"])
        for row in decision_evidence.get("output_rows", [])
    }
    checks["calibration_sequence_coverage"] = (
        not calibration.empty
        and calibration.sequence_id.astype(str).nunique() == selected_sequences
        and calibration.calibration_unit.eq(
            "one frozen estimate per source sequence"
        ).all()
        and calibration.method_independent.astype(bool).all()
        and set(calibration.sequence_manifest_sha256.astype(str))
        == expected_sequence_hashes
    )
    per_sequence_path = analysis_root / "metrics/per_sequence_method_summary.csv"
    per_sequence_frame = (
        pd.read_csv(per_sequence_path)
        if per_sequence_path.is_file()
        else pd.DataFrame()
    )
    verified_rows = int(decision_evidence.get("verified_succeeded_job_count", -1))
    expected_output_by_job = {
        str(row["job_id"]): row for row in decision_evidence.get("output_rows", [])
    }
    metric_provenance_matches = False
    if not per_sequence_frame.empty and {
        "job_id",
        "output_sha256",
        "sequence_manifest_sha256",
    }.issubset(per_sequence_frame.columns):
        metric_provenance_matches = all(
            str(row.output_sha256)
            == str(expected_output_by_job.get(str(row.job_id), {}).get("output_sha256"))
            and str(row.sequence_manifest_sha256)
            == str(
                expected_output_by_job.get(str(row.job_id), {}).get(
                    "sequence_manifest_sha256"
                )
            )
            for row in per_sequence_frame.itertuples()
        )
    checks["quality_rows_bound_to_verified_jobs"] = (
        not per_sequence_frame.empty
        and len(per_sequence_frame) == verified_rows
        and per_sequence_frame.job_id.astype(str).nunique() == verified_rows
        and set(per_sequence_frame.job_id.astype(str))
        == {
            str(row["job_id"])
            for row in decision_evidence.get("output_rows", [])
        }
        and metric_provenance_matches
    )
    attempt_id = str(manifest.get("analysis_attempt_id", ""))
    attempt_path = analysis_root / "analysis_attempts" / f"{attempt_id}.json"
    attempt: dict[str, Any] = {}
    if attempt_path.is_file():
        try:
            attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            attempt = {}
    checks["atomic_analysis_publication"] = (
        attempt.get("status") == "published"
        and attempt.get("published") is True
        and attempt.get("publication_manifest_sha256")
        == (sha256_file(manifest_path) if manifest_path.is_file() else "missing")
        and np.isfinite(_float_or_nan(attempt.get("active_wall_s")))
    )
    ready_attempt = _finalize_analysis_attempt(
        validation_attempt_path,
        validation_attempt,
        started_monotonic=validation_started,
        status="validation_ready_to_publish",
        published=False,
        error=decision_error,
    )
    try:
        exact_budget = recompute_stage2_budget_evidence(
            plan, run_root, analysis_root
        )
        exact_budget_row = {
            "execution_attempt_wall_s": exact_budget.execution_attempt_wall_s,
            "analysis_attempt_wall_s": exact_budget.prior_analysis_attempt_wall_s,
            "total_accounted_wall_s": exact_budget.total_accounted_wall_s,
            "raw_input_baseline_bytes": exact_budget.raw_input_baseline_bytes,
            "retained_tree_bytes": exact_budget.retained_tree_bytes,
            "total_accounted_retained_bytes": (
                exact_budget.total_accounted_retained_bytes
            ),
            "wall_limit_s": exact_budget.wall_limit_s,
            "storage_limit_bytes": exact_budget.storage_limit_bytes,
            "wall_compliant": exact_budget.wall_compliant,
            "storage_compliant": exact_budget.storage_compliant,
            "evidence_sha256": exact_budget.evidence_sha256,
            "execution_attempt_count": exact_budget.execution_attempt_count,
            "finalized_analysis_attempt_count": (
                exact_budget.finalized_analysis_attempt_count
            ),
        }
        if decision_evidence:
            decision_evidence["budget"] = exact_budget_row
            decision_evidence.pop("decision_basis_sha256", None)
            decision_evidence["decision_basis_sha256"] = _canonical_sha256(
                decision_evidence
            )
        checks["hard_budget_compliant"] = bool(
            exact_budget.wall_compliant
            and exact_budget.storage_compliant
            and exact_budget.total_accounted_retained_bytes
            + ANALYSIS_PUBLICATION_RESERVE_BYTES
            <= exact_budget.storage_limit_bytes
        )
        checks["attempt_ledgers_present"] = bool(
            exact_budget.execution_attempt_count >= 1
            and exact_budget.finalized_analysis_attempt_count >= 1
        )
    except Stage2AnalysisError as error:
        decision_error = decision_error or f"{type(error).__name__}: {error}"
        checks["hard_budget_compliant"] = False
        checks["attempt_ledgers_present"] = False
        checks["independent_decision_evidence_recomputed"] = False
    decision = "GO" if checks and all(checks.values()) else "NO-GO"
    result = {
        "schema_version": 2,
        "stage": 2,
        "decision": decision,
        "design_id": plan.get("design_id"),
        "plan_path": str(resolved_plan),
        "plan_sha256": plan.get("plan_sha256"),
        "checks": checks,
        "expected_direct_reference_rows": expected_direct,
        "observed_direct_reference_rows": direct_rows,
        "decision_basis": decision_evidence,
        "decision_basis_sha256": decision_evidence.get("decision_basis_sha256"),
        "decision_evidence_error": decision_error,
        "analysis_manifest_is_not_decision_authority": True,
        "validation_does_not_launch_jobs": True,
    }
    staged_validation = validation_staging / "STAGE2_VALIDATION.json"
    atomic_write_json(staged_validation, result)
    published_validation = analysis_root / "STAGE2_VALIDATION.json"
    os.replace(staged_validation, published_validation)
    ready_attempt.update(
        {
            "status": "validation_published",
            "published": True,
            "published_at_utc": _utc_now(),
            "publication_manifest_sha256": sha256_file(published_validation),
        }
    )
    _write_analysis_attempt(validation_attempt_path, ready_attempt)
    return result
