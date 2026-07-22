"""Reproducible publication builder for the expanded Stage 1 Pilot.

This module is intentionally independent from :mod:`reporting`.  It consumes
only frozen canonical outputs and evidence tables; it never launches a
retargeter.  A build is all-or-nothing: six required 600-frame operating
points, the three Sparse seeds, the external Unitree-attributed reference,
the complete scale-policy matrix, and four interaction runs must all exist.

The Unitree-attributed corpus is treated as an untimed, external reference.
Its repository provenance does not establish that it is official, ground
truth, or an upper bound.
"""

from __future__ import annotations

import json
import hashlib
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from .calibration import load_evaluator_protocol
from .constants import STAGE1_RUN_DIRECTORIES
from .evaluator import HUMAN_SEMANTIC_JOINTS, evaluate_motion, save_evaluation
from .io_utils import atomic_write_json, atomic_write_text, load_yaml, sha256_file
from .interaction import (
    INTERACTION_METRIC_REVISION,
    verify_interaction_ablation_pair,
    verify_interaction_attempt,
)
from .robot_model import CanonicalRobotModel, default_robot_scene
from .rotations import quaternion_wxyz_to_matrix, yaw_from_matrix
from .scale_metric_registry import (
    SCALE_RANK_METRICS,
    SCALE_RESPONSE_METRIC_NAMES,
    SCALE_RESPONSE_VARIANTS,
    derive_registered_rank_stability,
    derive_registered_scale_slopes,
)
from .schemas import CanonicalG1, CanonicalHuman, RunManifest


EXPECTED_FRAMES = 600
HISTORICAL_STAGE1_STOP = (
    "FULL-LAFAN EXPERIMENTS NOT STARTED — WAITING FOR USER APPROVAL."
)
PUBLICATION_EVIDENCE_SCHEMA_VERSION = 2
VALIDATION_SCHEMA_VERSION = 5


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hashed_file_rows(root: Path, paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted({path.resolve() for path in paths}):
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(f"Publication evidence must be inside the repository: {path}") from error
        if not path.is_file():
            raise FileNotFoundError(f"Publication evidence is missing: {path}")
        rows.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return rows


def _verify_hashed_file_rows(root: Path, rows: Any, label: str) -> bool:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Publication {label} ledger is empty")
    seen: set[str] = set()
    for item in rows:
        if not isinstance(item, Mapping):
            raise ValueError(f"Publication {label} ledger contains a non-mapping row")
        relative = str(item.get("path", ""))
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts or relative in seen:
            raise ValueError(f"Unsafe or duplicate publication {label} path: {relative!r}")
        seen.add(relative)
        path = root / candidate
        if (
            not path.is_file()
            or path.stat().st_size != int(item.get("bytes", -1))
            or sha256_file(path) != str(item.get("sha256", ""))
        ):
            raise ValueError(f"Publication {label} hash mismatch: {relative}")
    return True


def inspect_publication_evidence(repo_root: str | Path = ".") -> dict[str, Any]:
    """Verify the immutable PENDING evidence ledger used by the validator.

    The publication builder is deliberately not an acceptance authority.  Its
    manifest can only describe hash-bound inputs and outputs in ``PENDING``
    state.  A separate validator may bind a decision to this exact ledger.
    """

    root = Path(repo_root).resolve()
    path = root / "manifests" / "stage1_publication.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != PUBLICATION_EVIDENCE_SCHEMA_VERSION
        or manifest.get("decision") != "PENDING"
        or manifest.get("state") != "evidence_ready_for_independent_validation"
        or manifest.get("decision_authority") != "manifests/stage1_validation.json"
    ):
        raise ValueError("Stage 1 publication manifest is not a PENDING evidence ledger")
    inputs = manifest.get("input_hashes")
    outputs = manifest.get("outputs")
    _verify_hashed_file_rows(root, inputs, "input")
    _verify_hashed_file_rows(root, outputs, "output")
    input_digest = _canonical_sha256(inputs)
    output_digest = _canonical_sha256(outputs)
    if input_digest != manifest.get("input_bundle_sha256"):
        raise ValueError("Publication input-bundle digest mismatch")
    if output_digest != manifest.get("output_bundle_sha256"):
        raise ValueError("Publication output-bundle digest mismatch")
    return {
        "publication_manifest_path": "manifests/stage1_publication.json",
        "publication_manifest_sha256": sha256_file(path),
        "input_bundle_sha256": input_digest,
        "output_bundle_sha256": output_digest,
    }


def validation_decision_basis_sha256(
    decision: str,
    checks: Mapping[str, Any],
    publication_evidence_binding: Mapping[str, Any],
) -> str:
    """Return the deterministic digest that prevents verdict-only editing."""

    return _canonical_sha256(
        {
            "decision": decision,
            "checks": dict(checks),
            "publication_evidence_binding": dict(publication_evidence_binding),
        }
    )


def resolve_bound_stage1_validation(repo_root: str | Path = ".") -> dict[str, Any]:
    """Resolve the authoritative verdict, failing closed to ``PENDING``.

    A validation file is accepted only when it binds the current verified
    publication ledger, its decision agrees with every boolean check, and its
    decision-basis digest recomputes exactly.  This catches stale ledgers and
    simple verdict forgery without letting the publication certify itself.
    """

    root = Path(repo_root).resolve()
    try:
        binding = inspect_publication_evidence(root)
    except Exception as error:
        return {
            "decision": "PENDING",
            "binding_status": "invalid_publication_evidence",
            "reason": f"{type(error).__name__}: {error}",
            "validation": None,
        }
    path = root / "manifests" / "stage1_validation.json"
    if not path.is_file():
        return {
            "decision": "PENDING",
            "binding_status": "validation_missing",
            "reason": "Independent Stage 1 validation has not been published",
            "validation": None,
            "publication_evidence_binding": binding,
        }
    try:
        validation = json.loads(path.read_text(encoding="utf-8"))
        checks = validation["checks"]
        if not isinstance(checks, dict) or not checks or not all(
            isinstance(value, bool) for value in checks.values()
        ):
            raise ValueError("Validation checks are not a non-empty boolean mapping")
        decision = str(validation.get("decision", ""))
        expected_decision = "GO" if all(checks.values()) else "NO-GO"
        if (
            validation.get("schema_version") != VALIDATION_SCHEMA_VERSION
            or validation.get("authority") != "independent_fail_closed_stage1_validator"
            or validation.get("publication_evidence_binding") != binding
            or decision != expected_decision
            or int(validation.get("check_count", -1)) != len(checks)
            or int(validation.get("passed_check_count", -1)) != sum(checks.values())
            or validation.get("failed_checks")
            != [name for name, passed in checks.items() if not passed]
            or validation.get("decision_basis_sha256")
            != validation_decision_basis_sha256(decision, checks, binding)
        ):
            raise ValueError("Validation verdict or evidence binding is inconsistent")
    except Exception as error:
        return {
            "decision": "PENDING",
            "binding_status": "validation_unbound_or_invalid",
            "reason": f"{type(error).__name__}: {error}",
            "validation": None,
            "publication_evidence_binding": binding,
        }
    return {
        "decision": decision,
        "binding_status": "verified",
        "reason": "Verdict is bound to the current publication evidence ledger",
        "validation": validation,
        "validation_sha256": sha256_file(path),
        "publication_evidence_binding": binding,
    }


@dataclass(frozen=True)
class MethodSpec:
    key: str
    display_name: str
    run_directory: str
    metadata_methods: tuple[str, ...]
    evidence_role: str
    timed: bool = True


CORE_METHODS = (
    MethodSpec(
        "sparse-neutral",
        "Sparse Mink (neutral)",
        STAGE1_RUN_DIRECTORIES["sparse-neutral"],
        ("sparse-neutral",),
        "controlled operating point",
    ),
    MethodSpec(
        "dense",
        "Dense Mink",
        STAGE1_RUN_DIRECTORIES["dense"],
        ("dense",),
        "controlled operating point",
    ),
    MethodSpec(
        "gmr",
        "GMR",
        STAGE1_RUN_DIRECTORIES["gmr"],
        ("gmr",),
        "published public pipeline",
    ),
    MethodSpec(
        "omniretarget",
        "OmniRetarget / Holosoma",
        STAGE1_RUN_DIRECTORIES["omniretarget"],
        ("omniretarget",),
        "published public pipeline",
    ),
    MethodSpec(
        "protomotions-v2.3",
        "ProtoMotions v2.3 Mink LAFAN port",
        STAGE1_RUN_DIRECTORIES["protomotions-v2.3"],
        ("protomotions_v2_3_mink", "protomotions-v2.3"),
        "canonical LAFAN port of the v2.3 Mink retargeter",
    ),
    MethodSpec(
        "protomotions-v3",
        "ProtoMotions v3 modified-PyRoki LAFAN port",
        STAGE1_RUN_DIRECTORIES["protomotions-v3"],
        ("protomotions_v3", "protomotions-v3"),
        "canonical LAFAN port of the published generic modified-PyRoki retargeter",
    ),
)

SPARSE_DIAGNOSTICS = (
    MethodSpec(
        "sparse-a",
        "Sparse Mink (seed A)",
        STAGE1_RUN_DIRECTORIES["sparse-a"],
        ("sparse-a",),
        "initialization diagnostic only",
        timed=False,
    ),
    MethodSpec(
        "sparse-b",
        "Sparse Mink (seed B)",
        STAGE1_RUN_DIRECTORIES["sparse-b"],
        ("sparse-b",),
        "initialization diagnostic only",
        timed=False,
    ),
)

REFERENCE_METHOD = MethodSpec(
    "unitree-reference",
    "Unitree-attributed reference corpus",
    STAGE1_RUN_DIRECTORIES["unitree-reference"],
    ("unitree_reference",),
    "external precomputed reference; not verified official or ground truth",
    timed=False,
)

REPORT_FILES = (
    "PILOT_REPORT.md",
    "EXECUTIVE_SUMMARY.md",
    "GO_NO_GO.md",
    "METHOD_SCOPE.md",
    "SPARSE_IK_ANALYSIS.md",
    "INTERACTION_CASE_STUDY.md",
    "REPRODUCE_PILOT.md",
    "PRESENTATION.md",
    "SCALE_POLICY_SENSITIVITY.md",
    "UNITREE_REFERENCE_COMPARISON.md",
    "STAGE1_REVIEW.md",
)

NATIVE_SCALE_METHODS = (
    "gmr",
    "omniretarget",
    "protomotions_v2_3_mink",
    "protomotions_v3",
)
SCALE_VARIANTS = SCALE_RESPONSE_VARIANTS
TRANSPLANT_POLICIES = (
    "common_shared_semantic_landmark_ls",
    "holosoma_lafan_uniform",
    "gmr_region_wise",
    "protomotions_v2_3_world_axis",
    "protomotions_v3_lower_upper_axis",
)
FORMAL_NATIVE_SCALE_ROLE = "native_response_fixed_canonical_contact"
CONTACT_ROBUSTNESS_ROLE = "native_response_native_recomputed_contact"
SCALE_INPUT_LEDGER = "manifests/stage1_scale_input_ledger.csv"
SCALE_INPUT_LEDGER_SCHEMA_VERSION = 1

_RAW_SCALE_METHOD = {
    "gmr": "gmr",
    "omniretarget": "omniretarget",
    "protomotions_v2_3_mink": "protomotions_v2_3",
    "protomotions_v3": "protomotions_v3",
    "controlled_dense": "controlled_dense",
}

_NATIVE_POLICY_BY_METHOD = {
    "gmr": "gmr_region_wise",
    "omniretarget": "holosoma_lafan_uniform",
    "protomotions_v2_3_mink": "protomotions_v2_3_world_axis",
    "protomotions_v3": "protomotions_v3_lower_upper_axis",
}

QUALITY_COLUMNS = (
    "rf_kpe_all_mean_m",
    "rf_kpe_targeted_mean_m",
    "rf_kpe_untracked_mean_m",
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


@dataclass
class EvaluationBundle:
    core: pd.DataFrame
    reference: pd.DataFrame
    diagnostics: pd.DataFrame


@dataclass
class ScaleBundle:
    summary: pd.DataFrame
    transplants: pd.DataFrame
    native_response: pd.DataFrame
    slopes: pd.DataFrame
    ranks: pd.DataFrame
    contact_robustness: pd.DataFrame
    contact_robustness_slopes: pd.DataFrame
    contact_diagnostics: pd.DataFrame


def _sequence(root: Path) -> dict[str, Any]:
    return load_yaml(root / "manifests" / "pilot_sequence.yaml")


def _sequence_root(root: Path) -> Path:
    return root / "runs" / str(_sequence(root)["sequence_id"])


def method_output_paths(repo_root: str | Path) -> dict[str, Path]:
    """Resolve every publication input from its accepted run manifest.

    A failed or stale immutable run is retried below an ``attempt_*``
    directory.  The accepted :class:`RunManifest` is therefore the source of
    truth; reconstructing ``<revision>/canonical_g1.npz`` would silently read
    the superseded artifact.  The external reference has no timed run
    manifest and deliberately falls back to its registered direct path.
    """

    root = Path(repo_root).resolve()
    sequence_id = str(_sequence(root)["sequence_id"])
    base = _sequence_root(root)
    specs = (*CORE_METHODS, *SPARSE_DIAGNOSTICS, REFERENCE_METHOD)
    resolved: dict[str, Path] = {}
    for spec in specs:
        direct = base / spec.run_directory / "canonical_g1.npz"
        run_id = f"{sequence_id}__{spec.run_directory}"
        candidates = (
            root / "runs" / "manifests" / f"{run_id}.json",
            root / "manifests" / "runs" / f"{run_id}.json",
        )
        manifest_path = next((path for path in candidates if path.is_file()), None)
        if manifest_path is None:
            resolved[spec.key] = direct
            continue
        manifest = RunManifest.load(manifest_path)
        if manifest.run_id != run_id:
            raise ValueError(
                f"Stage-1 manifest run id differs from the registered revision: "
                f"{manifest_path}"
            )
        output = Path(str(manifest.output_path))
        if not output.is_absolute():
            output = root / output
        output = output.resolve()
        revision_root = (base / spec.run_directory).resolve()
        if output.name != "canonical_g1.npz" or not output.is_relative_to(
            revision_root
        ):
            raise ValueError(
                f"Stage-1 manifest output escapes its registered revision: {output}"
            )
        resolved[spec.key] = output
    return resolved


def require_stage1_outputs(repo_root: str | Path) -> dict[str, Path]:
    """Require the exact expanded Stage 1 motion set.

    In particular, a missing ProtoMotions v3 result is an error rather than an
    N/A row or a substituted backend demo.
    """

    root = Path(repo_root).resolve()
    sequence = _sequence(root)
    if int(sequence.get("num_frames", -1)) != EXPECTED_FRAMES:
        raise ValueError(
            f"Publication requires the frozen {EXPECTED_FRAMES}-frame Pilot"
        )
    source = root / str(sequence["canonical_path"])
    paths = method_output_paths(root)
    missing = [str(path) for path in (source, *paths.values()) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Expanded Stage 1 is incomplete; required canonical outputs are missing: "
            + ", ".join(missing)
        )
    return paths


def _validate_motion_contract(
    motion: CanonicalG1, spec: MethodSpec, source_frames: int
) -> None:
    motion.validate(source_frame_count=source_frames)
    if source_frames != EXPECTED_FRAMES or len(motion.qpos) != EXPECTED_FRAMES:
        raise ValueError(
            f"{spec.key} must contain exactly {EXPECTED_FRAMES} frames, got "
            f"{len(motion.qpos)}/{source_frames}"
        )
    if not np.array_equal(
        np.asarray(motion.source_frame_idx), np.arange(EXPECTED_FRAMES)
    ):
        raise ValueError(f"{spec.key} does not cover source frames 0..599 exactly")
    if not np.asarray(motion.valid, dtype=bool).all():
        raise ValueError(f"{spec.key} contains invalid frames")
    method = str(motion.metadata.get("method", ""))
    if method not in spec.metadata_methods:
        raise ValueError(
            f"{spec.key} metadata method {method!r} is not one of "
            f"{spec.metadata_methods!r}"
        )
    if spec is REFERENCE_METHOD:
        if motion.metadata.get("timing_available") is not False:
            raise ValueError("The external reference must explicitly declare no timing")
        if motion.metadata.get("verified_official_ground_truth") is not False:
            raise ValueError(
                "The external reference must not be represented as verified ground truth"
            )


def _write_frame(frame: pd.DataFrame, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(stem.with_suffix(".csv"), index=False)
    frame.to_parquet(stem.with_suffix(".parquet"), index=False)


def evaluate_stage1_outputs(repo_root: str | Path) -> EvaluationBundle:
    """Evaluate every required motion under one evaluator-v3 protocol."""

    root = Path(repo_root).resolve()
    paths = require_stage1_outputs(root)
    sequence = _sequence(root)
    human = CanonicalHuman.load(root / str(sequence["canonical_path"]))
    if len(human.timestamps) != EXPECTED_FRAMES:
        raise ValueError("Canonical human package is not the frozen 600-frame Pilot")
    robot = CanonicalRobotModel(default_robot_scene(root))
    protocol_path = root / "manifests" / "evaluator.yaml"
    protocol = load_evaluator_protocol(protocol_path)
    protocol["manifest_sha256"] = sha256_file(protocol_path)
    output_dir = root / "metrics" / "stage1_publication" / "runs"

    rows: list[dict[str, Any]] = []
    for scope, specs in (
        ("core", CORE_METHODS),
        ("diagnostic", SPARSE_DIAGNOSTICS),
        ("reference", (REFERENCE_METHOD,)),
    ):
        for spec in specs:
            path = paths[spec.key]
            motion = CanonicalG1.load(path)
            _validate_motion_contract(motion, spec, len(human.timestamps))
            table, summary = evaluate_motion(human, motion, robot, protocol)
            save_evaluation(table, summary, output_dir, spec.key)
            rows.append(
                {
                    **summary,
                    "key": spec.key,
                    "display_name": spec.display_name,
                    "scope": scope,
                    "evidence_role": spec.evidence_role,
                    "timed": spec.timed,
                    "output_path": str(path.relative_to(root)),
                    "output_sha256": sha256_file(path),
                }
            )

    frame = pd.DataFrame(rows)
    order = [
        *(spec.key for spec in CORE_METHODS),
        *(spec.key for spec in SPARSE_DIAGNOSTICS),
        REFERENCE_METHOD.key,
    ]
    frame["_order"] = frame.key.map({key: index for index, key in enumerate(order)})
    frame = frame.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    core = frame.loc[frame.scope == "core"].reset_index(drop=True)
    diagnostics = frame.loc[frame.scope == "diagnostic"].reset_index(drop=True)
    reference = frame.loc[frame.scope == "reference"].reset_index(drop=True)
    _write_frame(core, root / "metrics" / "stage1_core_summary")
    _write_frame(reference, root / "metrics" / "stage1_reference_summary")
    _write_frame(diagnostics, root / "metrics" / "stage1_sparse_seed_evaluations")
    return EvaluationBundle(core=core, reference=reference, diagnostics=diagnostics)


def _numeric(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"Timing field {name} must be finite")
    return result


def _timing_entry(
    method: str,
    entry: Mapping[str, Any],
    role: str,
    repetition: int,
    fps: float,
    default_frames: int,
    default_native_total_s: float,
) -> dict[str, Any]:
    frames = int(entry.get("frame_count", default_frames))
    if frames <= 0:
        raise ValueError(f"{method} timing entry has no frames")
    duration_s = frames / fps
    steady_s = entry.get("steady_end_to_end_total_s", entry.get("wall_time_s"))
    if steady_s is None:
        raise ValueError(f"{method} timing entry has no end-to-end duration")
    native_s = entry.get("native_total_s")
    if native_s is None and entry.get("native_frame_times_s") is not None:
        native_s = float(np.sum(np.asarray(entry["native_frame_times_s"], dtype=float)))
    if native_s is None:
        native_s = default_native_total_s
    boundary = str(entry.get("timing_boundary", "")).strip()
    artifact_path = str(entry.get("timing_artifact_path", "")).strip()
    artifact_sha256 = str(entry.get("timing_artifact_sha256", "")).strip()
    if not boundary:
        raise ValueError(f"{method} timing entry has no explicit timing boundary")
    if not artifact_path or len(artifact_sha256) != 64:
        raise ValueError(
            f"{method} timing entry has no hash-bound canonical artifact"
        )
    return {
        "key": method,
        "role": role,
        "repetition": repetition,
        "frame_count": frames,
        "steady_end_to_end_s": _numeric(steady_s, "steady_end_to_end_total_s"),
        "process_wall_s": _numeric(
            entry.get("wall_time_s", steady_s), "wall_time_s"
        ),
        "native_total_s": _numeric(native_s, "native_total_s"),
        "end_to_end_rtf": _numeric(steady_s, "steady_end_to_end_total_s")
        / duration_s,
        "native_core_rtf": _numeric(native_s, "native_total_s") / duration_s,
        "initialization_time_s": _numeric(
            entry.get("initialization_time_s", 0.0), "initialization_time_s"
        ),
        "timing_boundary": boundary,
        "timing_artifact_path": artifact_path,
        "timing_artifact_sha256": artifact_sha256,
    }


def parse_timing_record(
    method: str,
    payload: Mapping[str, Any],
    *,
    fps: float,
    frame_count: int,
    fallback_native_total_s: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse formal timing JSON or an auditable summary-only record."""

    entries: list[dict[str, Any]] = []
    cold = payload.get("cold")
    if isinstance(cold, Mapping):
        entries.append(
            _timing_entry(
                method,
                cold,
                "cold",
                0,
                fps,
                frame_count,
                fallback_native_total_s,
            )
        )
    warmups = payload.get("warmup", [])
    if isinstance(warmups, Mapping):
        warmups = [warmups]
    for index, entry in enumerate(warmups):
        entries.append(
            _timing_entry(
                method,
                entry,
                "warmup",
                index,
                fps,
                frame_count,
                fallback_native_total_s,
            )
        )
    measured = payload.get("measured_warm", payload.get("measured", []))
    if isinstance(measured, Mapping):
        measured = [measured]
    for index, entry in enumerate(measured):
        entries.append(
            _timing_entry(
                method,
                entry,
                "measured",
                index,
                fps,
                frame_count,
                fallback_native_total_s,
            )
        )

    if entries:
        raw = pd.DataFrame(entries)
        formal = raw.loc[raw.role == "measured"]
        if formal.empty:
            raise ValueError(f"{method} timing JSON has no measured repetitions")
        computed_e2e = float(formal.end_to_end_rtf.median())
        computed_native = float(formal.native_core_rtf.median())
        reported_e2e = _numeric(
            payload.get("end_to_end_rtf_median", computed_e2e),
            "end_to_end_rtf_median",
        )
        reported_native = _numeric(
            payload.get("native_core_rtf_median", computed_native),
            "native_core_rtf_median",
        )
        summary = {
            "key": method,
            "timing_evidence_grade": (
                "formal_repeated" if len(formal) >= 3 else "formal_under_repeated"
            ),
            "measured_repetitions": int(len(formal)),
            "warmup_repetitions": int((raw.role == "warmup").sum()),
            "end_to_end_rtf_median": reported_e2e,
            "native_core_rtf_median": reported_native,
            "recomputed_end_to_end_rtf_median": computed_e2e,
            "recomputed_native_core_rtf_median": computed_native,
            "reported_minus_recomputed_end_to_end_rtf": reported_e2e - computed_e2e,
            "reported_minus_recomputed_native_core_rtf": (
                reported_native - computed_native
            ),
            "cold_process_wall_s": float(
                payload.get(
                    "cold_process_wall_s",
                    raw.loc[raw.role == "cold", "process_wall_s"].iloc[0]
                    if (raw.role == "cold").any()
                    else np.nan,
                )
            ),
            "initialization_and_adapter_s_cold": float(
                payload.get(
                    "initialization_and_adapter_s_cold",
                    raw.loc[raw.role == "cold", "initialization_time_s"].iloc[0]
                    if (raw.role == "cold").any()
                    else np.nan,
                )
            ),
        }
        return raw, summary

    if "end_to_end_rtf_median" in payload:
        e2e = _numeric(payload["end_to_end_rtf_median"], "end_to_end_rtf_median")
        native = _numeric(
            payload.get(
                "native_core_rtf_median",
                fallback_native_total_s / (frame_count / fps),
            ),
            "native_core_rtf_median",
        )
        raw = pd.DataFrame(
            [
                {
                    "key": method,
                    "role": "reported-summary",
                    "repetition": 0,
                    "frame_count": frame_count,
                    "steady_end_to_end_s": e2e * frame_count / fps,
                    "process_wall_s": np.nan,
                    "native_total_s": native * frame_count / fps,
                    "end_to_end_rtf": e2e,
                    "native_core_rtf": native,
                    "initialization_time_s": np.nan,
                }
            ]
        )
        return raw, {
            "key": method,
            "timing_evidence_grade": "reported_summary_only",
            "measured_repetitions": 0,
            "warmup_repetitions": 0,
            "end_to_end_rtf_median": e2e,
            "native_core_rtf_median": native,
            "recomputed_end_to_end_rtf_median": e2e,
            "recomputed_native_core_rtf_median": native,
            "reported_minus_recomputed_end_to_end_rtf": 0.0,
            "cold_process_wall_s": np.nan,
            "initialization_and_adapter_s_cold": np.nan,
        }
    raise ValueError(f"{method} timing payload is not auditable")


def collect_stage1_timing(repo_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect timing for core methods only; never synthesize reference timing."""

    root = Path(repo_root).resolve()
    paths = require_stage1_outputs(root)
    raw_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for spec in CORE_METHODS:
        motion = CanonicalG1.load(paths[spec.key])
        run_dir = paths[spec.key].parent
        candidates = (run_dir / "timing_refined.json", run_dir / "timing.json")
        timing_path = next((path for path in candidates if path.is_file()), None)
        fallback_native = float(np.sum(motion.per_frame_solve_time_s))
        if timing_path is not None:
            payload = json.loads(timing_path.read_text(encoding="utf-8"))
            raw, summary = parse_timing_record(
                spec.key,
                payload,
                fps=motion.fps,
                frame_count=len(motion.qpos),
                fallback_native_total_s=fallback_native,
            )
            source = str(timing_path.relative_to(root))
            source_hash = sha256_file(timing_path)
            source_kind = "timing_json"
        else:
            total = motion.metadata.get("steady_end_to_end_total_s")
            if total is None:
                raise FileNotFoundError(
                    f"{spec.key} has neither timing JSON nor auditable metadata timing"
                )
            duration = len(motion.qpos) / motion.fps
            payload = {
                "end_to_end_rtf_median": _numeric(
                    total, "metadata.steady_end_to_end_total_s"
                )
                / duration,
                "native_core_rtf_median": fallback_native / duration,
            }
            raw, summary = parse_timing_record(
                spec.key,
                payload,
                fps=motion.fps,
                frame_count=len(motion.qpos),
                fallback_native_total_s=fallback_native,
            )
            summary["timing_evidence_grade"] = "metadata_single_aggregate"
            source = str(paths[spec.key].relative_to(root)) + "#metadata_json"
            source_hash = sha256_file(paths[spec.key])
            source_kind = "canonical_output_metadata"
        raw["timing_source"] = source
        raw["timing_source_sha256"] = source_hash
        artifact_verified: list[bool] = []
        for row in raw.itertuples(index=False):
            artifact = Path(str(row.timing_artifact_path))
            if not artifact.is_absolute():
                artifact = root / artifact
            verified = bool(
                artifact.is_file()
                and sha256_file(artifact) == str(row.timing_artifact_sha256)
            )
            artifact_verified.append(verified)
        raw["timing_artifact_hash_verified"] = artifact_verified
        if not all(artifact_verified):
            raise ValueError(
                f"{spec.key} has a missing or hash-mismatched timing artifact"
            )
        summary.update(
            {
                "display_name": spec.display_name,
                "timing_source": source,
                "timing_source_sha256": source_hash,
                "timing_source_kind": source_kind,
            }
        )
        raw_parts.append(raw)
        summary_rows.append(summary)

    raw_frame = pd.concat(raw_parts, ignore_index=True)
    summary_frame = pd.DataFrame(summary_rows)
    order = {spec.key: index for index, spec in enumerate(CORE_METHODS)}
    summary_frame["_order"] = summary_frame.key.map(order)
    summary_frame = summary_frame.sort_values("_order").drop(columns="_order")
    _write_frame(raw_frame, root / "metrics" / "stage1_timing_raw")
    _write_frame(summary_frame, root / "metrics" / "stage1_timing_summary")
    untimed = pd.DataFrame(
        [
            {
                "key": REFERENCE_METHOD.key,
                "display_name": REFERENCE_METHOD.display_name,
                "timed": False,
                "reason": "precomputed external corpus; runtime provenance unavailable",
                "excluded_from_rtf_plots": True,
            }
        ]
    )
    untimed.to_csv(root / "metrics" / "stage1_untimed_references.csv", index=False)
    return raw_frame, summary_frame


def _robot_root_frame(robot: CanonicalRobotModel, motion: CanonicalG1) -> np.ndarray:
    semantics = [name for name in HUMAN_SEMANTIC_JOINTS if name != "root"]
    frames = [robot.semantic_positions(qpos) for qpos in motion.qpos]
    root = np.stack([frame["root"] for frame in frames])
    points = (
        np.stack([[frame[name] for name in semantics] for frame in frames])
        - root[:, None]
    )
    yaw = yaw_from_matrix(quaternion_wxyz_to_matrix(motion.qpos[:, 3:7]))
    cosine, sine = np.cos(-yaw), np.sin(-yaw)
    x = cosine[:, None] * points[..., 0] - sine[:, None] * points[..., 1]
    y = sine[:, None] * points[..., 0] + cosine[:, None] * points[..., 1]
    return np.stack([x, y, points[..., 2]], axis=-1)


def compute_sparse_seed_diagnostics(
    repo_root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute pairwise and three-seed null-space sensitivity."""

    root = Path(repo_root).resolve()
    paths = require_stage1_outputs(root)
    robot = CanonicalRobotModel(default_robot_scene(root))
    specs = (CORE_METHODS[0], *SPARSE_DIAGNOSTICS)
    motions = {spec.key: CanonicalG1.load(paths[spec.key]) for spec in specs}
    root_frames = {
        key: _robot_root_frame(robot, motion) for key, motion in motions.items()
    }
    rows: list[dict[str, Any]] = []
    for left, right in combinations(motions, 2):
        angle_delta = np.arctan2(
            np.sin(motions[left].qpos[:, 7:] - motions[right].qpos[:, 7:]),
            np.cos(motions[left].qpos[:, 7:] - motions[right].qpos[:, 7:]),
        )
        angle_rms = np.sqrt(np.mean(angle_delta**2, axis=1))
        point_rms = np.sqrt(
            np.mean(
                np.sum((root_frames[left] - root_frames[right]) ** 2, axis=2),
                axis=1,
            )
        )
        rows.extend(
            {
                "pair": f"{left}__{right}",
                "source_frame_idx": frame,
                "time_s": frame / motions[left].fps,
                "joint_angle_rms_rad": float(angle_rms[frame]),
                "robot_rf_point_rms_m": float(point_rms[frame]),
            }
            for frame in range(EXPECTED_FRAMES)
        )
    per_frame = pd.DataFrame(rows)
    pair_summary = (
        per_frame.groupby("pair", as_index=False)
        .agg(
            joint_angle_rms_mean_rad=("joint_angle_rms_rad", "mean"),
            joint_angle_rms_p95_rad=("joint_angle_rms_rad", lambda x: x.quantile(0.95)),
            joint_angle_rms_max_rad=("joint_angle_rms_rad", "max"),
            robot_rf_point_rms_mean_m=("robot_rf_point_rms_m", "mean"),
            robot_rf_point_rms_p95_m=("robot_rf_point_rms_m", lambda x: x.quantile(0.95)),
            robot_rf_point_rms_max_m=("robot_rf_point_rms_m", "max"),
        )
        .reset_index(drop=True)
    )

    semantics = [name for name in HUMAN_SEMANTIC_JOINTS if name != "root"]
    targeted = [
        semantics.index(name)
        for name in ("left_wrist", "right_wrist", "left_ankle", "right_ankle")
    ]
    untracked = [index for index in range(len(semantics)) if index not in targeted]
    stacked = np.stack([root_frames[spec.key] for spec in specs])
    variance_norm = np.sum(np.var(stacked, axis=0), axis=2)
    joints = np.stack([motions[spec.key].qpos[:, 7:] for spec in specs])
    circular_mean = np.arctan2(
        np.mean(np.sin(joints), axis=0), np.mean(np.cos(joints), axis=0)
    )
    circular_delta = np.arctan2(
        np.sin(joints - circular_mean[None]),
        np.cos(joints - circular_mean[None]),
    )
    variance = pd.DataFrame(
        {
            "source_frame_idx": np.arange(EXPECTED_FRAMES),
            "targeted_point_variance_m2": np.mean(variance_norm[:, targeted], axis=1),
            "untracked_point_variance_m2": np.mean(
                variance_norm[:, untracked], axis=1
            ),
            "all_point_variance_m2": np.mean(variance_norm, axis=1),
            "qpos_circular_variance_rad2": np.mean(
                circular_delta**2, axis=(0, 2)
            ),
        }
    )
    per_frame.to_csv(
        root / "metrics" / "stage1_sparse_seed_pairwise_per_frame.csv", index=False
    )
    pair_summary.to_csv(
        root / "metrics" / "stage1_sparse_seed_pairwise_summary.csv", index=False
    )
    variance.to_csv(
        root / "metrics" / "stage1_sparse_seed_variance_per_frame.csv", index=False
    )
    return per_frame, pair_summary, variance


def _boolean_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def collect_interaction_evidence(repo_root: str | Path) -> pd.DataFrame:
    """Rebuild the four-case summary from frozen surface-distance artifacts."""

    root = Path(repo_root).resolve()
    rows: list[dict[str, Any]] = []
    for case in ("box", "climb"):
        for variant in ("full", "no-hard"):
            manifest_path = (
                root
                / "runs"
                / "interaction_manifests"
                / f"interaction__{case}__{variant}.json"
            )
            manifest = RunManifest.load(manifest_path)
            summary_path, value = verify_interaction_attempt(
                root, case, variant, manifest
            )
            directory = summary_path.parent
            per_frame_path = directory / "per_frame_metrics.csv"
            intended_path = directory / "intended_contact_per_semantic.csv"
            mapping_path = directory / "intended_contact_mapping.json"
            contract_path = directory / "intended_contact_source_contract.npz"
            if not all(
                path.is_file()
                for path in (
                    summary_path,
                    per_frame_path,
                    intended_path,
                    mapping_path,
                    contract_path,
                )
            ):
                raise FileNotFoundError(
                    f"Missing interaction evidence for {case}/{variant}: {directory}"
                )
            value.pop("native_frame_times_s", None)
            per_frame = pd.read_csv(per_frame_path)
            required = {
                "minimum_robot_object_surface_distance_m",
                "penetration_depth_m",
                "left_stance",
                "right_stance",
                "left_foot_xy_displacement_m",
                "right_foot_xy_displacement_m",
                "intended_source_contact_count_2cm",
                "intended_source_contact_count_5cm",
                "intended_source_contact_count_10cm",
                "intended_min_source_surface_distance_m",
                "intended_min_mapped_robot_signed_surface_distance_m",
                "intended_max_mapped_robot_penetration_depth_m",
            }
            missing = required - set(per_frame)
            if missing:
                raise ValueError(
                    f"Interaction per-frame table {per_frame_path} lacks {sorted(missing)}"
                )
            distance = per_frame.minimum_robot_object_surface_distance_m.to_numpy(
                dtype=float
            )
            depth = np.maximum(0.0, -distance)
            penetration_tolerance = 0.0011
            valid_separation = distance >= -penetration_tolerance
            left_stance = _boolean_series(per_frame.left_stance)
            right_stance = _boolean_series(per_frame.right_stance)
            tolerance = np.sqrt(2.0) * 0.001 + 1e-4
            left_violation = left_stance & (
                per_frame.left_foot_xy_displacement_m > tolerance
            )
            right_violation = right_stance & (
                per_frame.right_foot_xy_displacement_m > tolerance
            )
            value.update(
                {
                    "case": case,
                    "variant": variant,
                    "strict_contact_2cm_frame_rate": float(
                        np.mean(valid_separation & (distance <= 0.02))
                    ),
                    "near_contact_2_to_5cm_frame_rate": float(
                        np.mean((distance > 0.02) & (distance <= 0.05))
                    ),
                    "proximity_5_to_10cm_frame_rate": float(
                        np.mean((distance > 0.05) & (distance <= 0.10))
                    ),
                    "near_contact_5cm_frame_rate": float(
                        np.mean(valid_separation & (distance <= 0.05))
                    ),
                    "proximity_10cm_frame_rate": float(
                        np.mean(valid_separation & (distance <= 0.10))
                    ),
                    "penetration_any_frame_rate": float(np.mean(depth > 0.0)),
                    "penetration_frame_rate": float(
                        np.mean(depth > penetration_tolerance)
                    ),
                    "penetration_primary_threshold_m": penetration_tolerance,
                    "contact_penetration_tolerance_m": penetration_tolerance,
                    "distance_scope": (
                        "minimum over all enabled robot-object collision pairs"
                    ),
                    "task_intended_contact_pair_asserted": False,
                    "foot_sticking_violation_frame_rate": float(
                        np.mean(left_violation | right_violation)
                    ),
                    "reported_metric_revision": (
                        INTERACTION_METRIC_REVISION
                    ),
                    "summary_sha256": sha256_file(summary_path),
                    "per_frame_sha256": sha256_file(per_frame_path),
                    "intended_contact_per_semantic_sha256": sha256_file(
                        intended_path
                    ),
                    "intended_contact_mapping_artifact_sha256": sha256_file(
                        mapping_path
                    ),
                    "intended_contact_source_contract_sha256": sha256_file(
                        contract_path
                    ),
                }
            )
            if value.get("status") != "succeeded":
                raise ValueError(f"Interaction {case}/{variant} did not succeed")
            if "mj_geomDistance" not in str(value.get("distance_backend", "")):
                raise ValueError("Interaction distance is not a mesh/surface query")
            if (
                value.get("metric_revision")
                != INTERACTION_METRIC_REVISION
                or value.get("intended_contact_pair_asserted") is not True
                or value.get("intended_contact_fail_closed") is not True
                or "point-to-triangle" not in str(
                    value.get("source_surface_distance_backend", "")
                )
                or "mj_geomDistance" not in str(
                    value.get("mapped_robot_surface_distance_backend", "")
                )
            ):
                raise ValueError(
                    f"Missing source-conditioned intended-contact evidence for {case}/{variant}"
                )
            expected_hard = variant == "full"
            if bool(value.get("activate_obj_non_penetration")) != expected_hard:
                raise ValueError(f"Unexpected non-penetration flag for {case}/{variant}")
            if bool(value.get("activate_foot_sticking")) != expected_hard:
                raise ValueError(f"Unexpected foot-sticking flag for {case}/{variant}")
            if value.get("activate_joint_limits") is not True:
                raise ValueError(f"Joint limits are not active for {case}/{variant}")
            graph = value.get("constraint_graph_audit", {})
            if graph.get("runtime_graph_not_source_ast_only") is not True:
                raise ValueError(f"Missing runtime constraint graph for {case}/{variant}")
            minimum = graph.get("component_count_min", {})
            maximum = graph.get("component_count_max", {})
            if int(minimum.get("joint_limits", -1)) != 2:
                raise ValueError(f"Joint-limit graph changed for {case}/{variant}")
            if variant == "full":
                if int(maximum.get("object_non_penetration", 0)) <= 0:
                    raise ValueError(f"Full lacks object constraints for {case}")
                if int(maximum.get("foot_sticking_and_lock", 0)) <= 0:
                    raise ValueError(f"Full lacks foot-sticking constraints for {case}")
            elif (
                int(maximum.get("object_non_penetration", -1)) != 0
                or int(maximum.get("foot_sticking_and_lock", -1)) != 0
            ):
                raise ValueError(f"No-Hard still constructed disabled constraints for {case}")
            rows.append(value)
        verify_interaction_ablation_pair(root, case)
    frame = pd.DataFrame(rows)
    order = {("box", "full"): 0, ("box", "no-hard"): 1, ("climb", "full"): 2, ("climb", "no-hard"): 3}
    frame["_order"] = [order[(case, variant)] for case, variant in zip(frame.case, frame.variant)]
    frame = frame.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    _write_frame(frame, root / "metrics" / "stage1_interaction_summary")
    return frame


def _normalise_scale_method(value: str) -> str:
    aliases = {
        "protomotions_v2_3": "protomotions_v2_3_mink",
        "protomotions-v2.3": "protomotions_v2_3_mink",
        "protomotions-v3": "protomotions_v3",
    }
    return aliases.get(value, value)


def _derive_scale_slopes(
    response: pd.DataFrame, methods: Iterable[str]
) -> pd.DataFrame:
    selected = response.copy()
    selected["method"] = selected.method.astype(str).map(_normalise_scale_method)
    method_order = tuple(methods)
    selected = selected.loc[selected.method.isin(method_order)]
    if set(selected.method.astype(str)) != set(method_order):
        raise ValueError("Scale slope input does not contain the exact method set")
    return derive_registered_scale_slopes(
        selected,
        group_columns=("method",),
    )


def derive_scale_tables(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derive formal fixed-contact sensitivities and rank stability."""

    response = summary.loc[
        summary.experiment_role == FORMAL_NATIVE_SCALE_ROLE
    ].copy()
    response["method"] = response.method.astype(str).map(_normalise_scale_method)
    slopes = _derive_scale_slopes(response, NATIVE_SCALE_METHODS)
    ranks = derive_registered_rank_stability(
        response,
        methods=NATIVE_SCALE_METHODS,
        experiment_role=FORMAL_NATIVE_SCALE_ROLE,
    )
    return slopes, ranks


def _expected_scale_evidence_keys() -> set[tuple[str, str, str]]:
    fixed = {
        ("controlled_policy", "controlled_dense", policy)
        for policy in TRANSPLANT_POLICIES
    }
    fixed.update(
        (FORMAL_NATIVE_SCALE_ROLE, method, variant)
        for method in NATIVE_SCALE_METHODS
        for variant in SCALE_VARIANTS
    )
    robustness = {
        (CONTACT_ROBUSTNESS_ROLE, "omniretarget", variant)
        for variant in SCALE_VARIANTS
    }
    return fixed | robustness


def _scale_row_keys(frame: pd.DataFrame) -> set[tuple[str, str, str]]:
    return set(
        zip(
            frame.experiment_role.astype(str),
            frame.method.astype(str).map(_normalise_scale_method),
            frame.variant.astype(str),
        )
    )


def _relative_hashed_path(root: Path, path: Path) -> tuple[str, str]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"Scale evidence must stay inside the repository: {path}") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"Scale evidence is missing: {resolved}")
    return relative, sha256_file(resolved)


def _controlled_scale_execution_contract(root: Path) -> dict[str, Any]:
    """Return the same immutable identities used by the controlled transplants."""

    files = {
        "harness": root / "src/retargeting_comparison/scale_sensitivity.py",
        "method_adapter": root / "src/retargeting_comparison/controlled_mink.py",
        "evaluator_implementation": root / "src/retargeting_comparison/evaluator.py",
        "evaluator_protocol": root / "manifests/evaluator.yaml",
        "environment_lock": root / "environments/environment-locks.yaml",
    }
    return {
        "schema_version": 1,
        "environment_name": "robot",
        **{
            f"{name}_path": path.relative_to(root).as_posix()
            for name, path in files.items()
        },
        **{f"{name}_sha256": sha256_file(path) for name, path in files.items()},
    }


def _scale_execution_binding(
    root: Path, role: str, method: str
) -> dict[str, Any]:
    if role == "controlled_policy":
        return _controlled_scale_execution_contract(root)
    from .scale_worker import scale_execution_contract

    raw_method = _RAW_SCALE_METHOD[method]
    return scale_execution_contract(root, raw_method)


def _scale_config_path(root: Path, role: str, method: str) -> Path:
    if role == "controlled_policy":
        return root / "configs/controlled_mink.yaml"
    paths = {
        "gmr": (
            root
            / "external/GMR/general_motion_retargeting/ik_configs"
            / "bvh_lafan1_to_g1.json"
        ),
        "omniretarget": (
            root
            / "external/holosoma/src/holosoma_retargeting"
            / "holosoma_retargeting/config_types/data_type.py"
        ),
        "protomotions_v2_3_mink": root / "configs/protomotions_v2.yaml",
        "protomotions_v3": root / "configs/protomotions_v3.yaml",
    }
    return paths[method]


def _scale_asset_binding(
    root: Path, role: str, method: str
) -> dict[str, Any]:
    evaluator = default_robot_scene(root)
    if role == "controlled_policy":
        solver = evaluator
    else:
        solver = {
            "gmr": root / "external/GMR/assets/unitree_g1/g1_mocap_29dof.xml",
            "omniretarget": (
                root
                / "external/holosoma/src/holosoma_retargeting"
                / "holosoma_retargeting/models/g1/g1_29dof.xml"
            ),
            "protomotions_v2_3_mink": (
                root
                / "external/ProtoMotions-v2.3/protomotions/data/assets/mjcf/g1.xml"
            ),
            "protomotions_v3": (
                root
                / "external/ProtoMotions/protomotions/data/assets/urdf"
                / "for_retargeting/g1.urdf"
            ),
        }[method]
    solver_path, solver_hash = _relative_hashed_path(root, solver)
    evaluator_path, evaluator_hash = _relative_hashed_path(root, evaluator)
    return {
        "solver_asset_path": solver_path,
        "solver_asset_sha256": solver_hash,
        "evaluator_asset_path": evaluator_path,
        "evaluator_asset_sha256": evaluator_hash,
        "solver_and_evaluator_assets_identical": solver.resolve()
        == evaluator.resolve(),
    }


def _scale_raw_metric_paths(
    root: Path, role: str, method: str, variant: str
) -> dict[str, str]:
    raw_method = _RAW_SCALE_METHOD[method]
    label = f"{role}__{raw_method}__{variant}".replace("/", "-")
    base = root / "metrics/scale_sensitivity" / label
    result: dict[str, str] = {}
    for name, suffix in (
        ("raw_per_frame_csv", "_per_frame.csv"),
        ("raw_per_frame_parquet", "_per_frame.parquet"),
        ("raw_evaluator_summary", "_summary.json"),
    ):
        path, digest = _relative_hashed_path(root, Path(str(base) + suffix))
        result[f"{name}_path"] = path
        result[f"{name}_sha256"] = digest
    return result


def _target_contract_row(
    targets: pd.DataFrame, role: str, method: str, variant: str
) -> pd.Series:
    if role == "controlled_policy":
        scope = "controlled_dense"
        policy = variant
        target_variant = "native"
    else:
        scope = _RAW_SCALE_METHOD[method]
        policy = _NATIVE_POLICY_BY_METHOD[method]
        target_variant = variant
    match = targets.loc[
        targets.method_scope.astype(str).eq(scope)
        & targets.policy_id.astype(str).eq(policy)
        & targets.variant.astype(str).eq(target_variant)
    ]
    if len(match) != 1:
        raise ValueError(
            "Scale target contract must contain exactly one row for "
            f"{role}/{method}/{variant}"
        )
    return match.iloc[0]


def _registered_target_tensor_sha256(path: Path) -> str:
    from .scale_sensitivity import _array_sha256

    with np.load(path, allow_pickle=False) as archive:
        positions = np.asarray(archive["target_positions"], dtype=np.float64)
        labels = np.asarray(archive["landmark_names"]).astype(str).tolist()
    return _array_sha256(positions, labels)


def _runtime_scale_evidence(
    root: Path,
    role: str,
    method: str,
    variant: str,
    output: Path,
    motion: CanonicalG1,
    target: pd.Series,
) -> dict[str, Any]:
    """Bind exact runtime evidence without conflating it with common diagnostics."""

    if role == "controlled_policy":
        path = root / str(target.artifact_path)
        relative, digest = _relative_hashed_path(root, path)
        return {
            "runtime_evidence_kind": "deterministic_controlled_target_constructor",
            "runtime_evidence_path": relative,
            "runtime_evidence_sha256": digest,
            "runtime_target_sha256": str(target.target_sha256),
            "runtime_boundary_observed": False,
            "formal_native_binding_path": "",
            "formal_native_binding_sha256": "",
        }

    capture = motion.metadata.get("runtime_pre_solver_capture")
    if method != "protomotions_v3":
        hash_key = {
            "gmr": "position_tensor_sha256",
            "omniretarget": "target_tensor_sha256",
            "protomotions_v2_3_mink": "position_tensor_sha256",
        }[method]
        if (
            not isinstance(capture, Mapping)
            or capture.get("observed_during_solver_run") is not True
            or not re.fullmatch(r"[0-9a-f]{64}", str(capture.get(hash_key, "")))
        ):
            raise ValueError(
                f"Scale runtime target witness is incomplete for {method}/{variant}"
            )
        relative, digest = _relative_hashed_path(root, output)
        return {
            "runtime_evidence_kind": "exact_runtime_pre_solver_capture_in_output_metadata",
            "runtime_evidence_path": relative,
            "runtime_evidence_sha256": digest,
            "runtime_target_sha256": str(capture[hash_key]),
            "runtime_boundary_observed": True,
            "formal_native_binding_path": "",
            "formal_native_binding_sha256": "",
        }

    if variant == "native":
        binding = output.with_name("formal_output_binding.json")
        binding_path, binding_hash = _relative_hashed_path(root, binding)
        value = json.loads(binding.read_text(encoding="utf-8"))
        formal_output = root / str(value.get("formal_output_path", ""))
        sequence = _sequence(root)
        sequence_id = str(sequence["sequence_id"])
        canonical_source = root / str(sequence["canonical_path"])
        manifest_path = (
            root / "manifests/runs" / f"{sequence_id}__protomotions-v3.json"
        )
        manifest = RunManifest.load(manifest_path)
        accepted_output = Path(str(manifest.output_path))
        if not accepted_output.is_absolute():
            accepted_output = root / accepted_output
        accepted_output = accepted_output.resolve()
        registered_root = (
            root
            / "runs"
            / sequence_id
            / STAGE1_RUN_DIRECTORIES["protomotions-v3"]
        ).resolve()
        try:
            accepted_output.relative_to(registered_root)
        except ValueError as error:
            raise ValueError(
                "ProtoMotions v3 accepted formal output escaped its registered run directory"
            ) from error
        formal_evidence = root / str(value.get("formal_evidence_path", ""))
        try:
            formal_evidence.resolve().relative_to(registered_root)
        except ValueError as error:
            raise ValueError(
                "ProtoMotions v3 formal evidence escaped its registered run directory"
            ) from error
        if (
            value.get("schema_version") != 1
            or value.get("binding_type")
            != "byte_identical_published_formal_output_materialization"
            or value.get("additional_solver_invocation") is True
            or value.get("solver_invoked_for_materialization") is not False
            or value.get("scale_variant") != "native"
            or float(value.get("root_scale_multiplier", np.nan)) != 1.0
            or float(value.get("local_scale_multiplier", np.nan)) != 1.0
            or manifest.status.value != "succeeded"
            or manifest.exit_code != 0
            or manifest.run_id != f"{sequence_id}__protomotions-v3"
            or manifest.method != "protomotions-v3"
            or manifest.source_sha256 != str(sequence["cropped_sha256"])
            or accepted_output != formal_output.resolve()
            or manifest.output_sha256 != sha256_file(accepted_output)
            or value.get("formal_output_sha256") != manifest.output_sha256
            or manifest.config_sha256
            != sha256_file(root / "configs/protomotions_v3.yaml")
            or value.get("config_sha256") != manifest.config_sha256
            or value.get("scale_protocol_sha256")
            != sha256_file(root / "configs/scale_policy_sensitivity.yaml")
            or value.get("canonical_source_sha256")
            != str(sequence["cropped_sha256"])
            or value.get("canonical_source_file_sha256")
            != sha256_file(canonical_source)
            or value.get("native_scale_output_sha256") != sha256_file(output)
            or (root / str(value.get("native_scale_output_path", ""))).resolve()
            != output.resolve()
            or not formal_output.is_file()
            or sha256_file(formal_output) != sha256_file(output)
            or not formal_evidence.is_file()
            or value.get("formal_evidence_sha256") != sha256_file(formal_evidence)
            or not isinstance(value.get("implementation_hashes"), Mapping)
            or not value["implementation_hashes"]
            or value.get("implementation_hashes")
            != motion.metadata.get("implementation_hashes")
            or not isinstance(capture, Mapping)
            or capture.get("observed_during_solver_run") is not True
        ):
            raise ValueError("ProtoMotions v3 native point is not bound to the formal run")
        relative, digest = _relative_hashed_path(root, output)
        return {
            "runtime_evidence_kind": "formal_output_plus_independent_cold_runtime_capture",
            "runtime_evidence_path": relative,
            "runtime_evidence_sha256": digest,
            "runtime_target_sha256": str(capture.get("target_keypoints_sha256", "")),
            "runtime_boundary_observed": True,
            "formal_native_binding_path": binding_path,
            "formal_native_binding_sha256": binding_hash,
            "accepted_formal_manifest_path": manifest_path.relative_to(root).as_posix(),
            "accepted_formal_manifest_sha256": sha256_file(manifest_path),
            "accepted_formal_output_path": accepted_output.relative_to(root).as_posix(),
            "accepted_formal_output_sha256": sha256_file(accepted_output),
            "accepted_formal_evidence_path": formal_evidence.relative_to(root).as_posix(),
            "accepted_formal_evidence_sha256": sha256_file(formal_evidence),
        }

    evidence = output.with_name("evidence.json")
    evidence_path, evidence_hash = _relative_hashed_path(root, evidence)
    value = json.loads(evidence.read_text(encoding="utf-8"))
    input_value = value.get("input")
    if not isinstance(input_value, Mapping):
        raise ValueError(f"ProtoMotions v3 scale evidence lacks input for {variant}")
    target_path = Path(str(input_value.get("path", "")))
    if not target_path.is_absolute():
        target_path = root / target_path
    target_relative, target_hash = _relative_hashed_path(root, target_path)
    if (
        value.get("status") != "succeeded"
        or Path(str(value.get("canonical_output", ""))).resolve() != output.resolve()
        or value.get("canonical_output_sha256") != sha256_file(output)
        or input_value.get("sha256") != target_hash
    ):
        raise ValueError(f"ProtoMotions v3 scale evidence is stale for {variant}")
    return {
        "runtime_evidence_kind": "solver_invocation_bound_official_loader_input",
        "runtime_evidence_path": evidence_path,
        "runtime_evidence_sha256": evidence_hash,
        "runtime_target_sha256": target_hash,
        "runtime_boundary_observed": False,
        "runtime_input_path": target_relative,
        "runtime_input_sha256": target_hash,
        "formal_native_binding_path": "",
        "formal_native_binding_sha256": "",
    }


def build_scale_publication_input_ledger(
    repo_root: str | Path, summary: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Create the 30-row, hash-bound scale input ledger used for publication."""

    root = Path(repo_root).resolve()
    if summary is None:
        summary = pd.read_csv(root / "metrics/scale_policy_sensitivity_summary.csv")
    summary = summary.copy()
    summary["method"] = summary.method.astype(str).map(_normalise_scale_method)
    expected = _expected_scale_evidence_keys()
    if len(summary) != len(expected) or _scale_row_keys(summary) != expected:
        raise ValueError("Scale summary does not contain the exact 25+5 evidence matrix")
    targets_path = root / "manifests/pre_solver_targets.csv"
    targets = pd.read_csv(targets_path)
    expected_target_keys = {
        ("controlled_dense", policy, "native") for policy in TRANSPLANT_POLICIES
    } | {
        (_RAW_SCALE_METHOD[method], _NATIVE_POLICY_BY_METHOD[method], variant)
        for method in NATIVE_SCALE_METHODS
        for variant in SCALE_VARIANTS
    }
    actual_target_keys = set(
        zip(
            targets.method_scope.astype(str),
            targets.policy_id.astype(str),
            targets.variant.astype(str),
        )
    )
    if len(targets) != 25 or actual_target_keys != expected_target_keys:
        raise ValueError("Pre-solver target contract does not contain exactly 25 rows")

    sequence = _sequence(root)
    source = root / str(sequence["canonical_path"])
    human = CanonicalHuman.load(source)
    source_path, source_hash = _relative_hashed_path(root, source)
    evaluator_path = root / "manifests/evaluator.yaml"
    evaluator_relative, evaluator_hash = _relative_hashed_path(root, evaluator_path)
    evaluator_code = root / "src/retargeting_comparison/evaluator.py"
    evaluator_code_relative, evaluator_code_hash = _relative_hashed_path(
        root, evaluator_code
    )
    scale_protocol = root / "configs/scale_policy_sensitivity.yaml"
    scale_protocol_relative, scale_protocol_hash = _relative_hashed_path(
        root, scale_protocol
    )
    provenance = root / "manifests/scale_run_provenance.csv"
    provenance_relative, provenance_hash = _relative_hashed_path(root, provenance)
    target_manifest_relative, target_manifest_hash = _relative_hashed_path(
        root, targets_path
    )

    rows: list[dict[str, Any]] = []
    for summary_row in summary.sort_values(
        ["experiment_role", "method", "variant"]
    ).itertuples(index=False):
        role = str(summary_row.experiment_role)
        method = str(summary_row.method)
        variant = str(summary_row.variant)
        output = root / str(summary_row.output_path)
        output_path, output_hash = _relative_hashed_path(root, output)
        if output_hash != str(summary_row.output_sha256):
            raise ValueError(f"Scale summary output hash is stale: {output_path}")
        motion = CanonicalG1.load(output)
        motion.validate(source_frame_count=EXPECTED_FRAMES)
        target = _target_contract_row(targets, role, method, variant)
        target_artifact = root / str(target.artifact_path)
        target_path, target_hash = _relative_hashed_path(root, target_artifact)
        target_tensor_hash = _registered_target_tensor_sha256(target_artifact)
        if (
            target_hash != str(target.artifact_file_sha256)
            or target_tensor_hash != str(target.target_sha256)
        ):
            raise ValueError(f"Scale target artifact hash is stale: {target_path}")
        execution = _scale_execution_binding(root, role, method)
        config = _scale_config_path(root, role, method)
        config_path, config_hash = _relative_hashed_path(root, config)
        assets = _scale_asset_binding(root, role, method)
        environment_lock = root / str(execution["environment_lock_path"])
        environment_path, environment_hash = _relative_hashed_path(
            root, environment_lock
        )
        method_code = root / str(execution["method_adapter_path"])
        method_code_path, method_code_hash = _relative_hashed_path(root, method_code)
        runtime = _runtime_scale_evidence(
            root, role, method, variant, output, motion, target
        )
        rows.append(
            {
                "schema_version": SCALE_INPUT_LEDGER_SCHEMA_VERSION,
                "experiment_role": role,
                "method": method,
                "variant": variant,
                "fixed_contact_causal_arm": role != CONTACT_ROBUSTNESS_ROLE,
                "output_path": output_path,
                "output_sha256": output_hash,
                **_scale_raw_metric_paths(root, role, method, variant),
                "target_manifest_path": target_manifest_relative,
                "target_manifest_sha256": target_manifest_hash,
                "registered_target_path": target_path,
                "registered_target_file_sha256": target_hash,
                "registered_target_tensor_sha256": target_tensor_hash,
                **runtime,
                "scale_execution_contract_json": json.dumps(
                    execution, sort_keys=True, separators=(",", ":")
                ),
                "scale_execution_contract_sha256": _canonical_sha256(execution),
                "config_path": config_path,
                "config_sha256": config_hash,
                "method_code_path": method_code_path,
                "method_code_sha256": method_code_hash,
                "environment_name": str(execution["environment_name"]),
                "environment_lock_path": environment_path,
                "environment_lock_sha256": environment_hash,
                "source_path": source_path,
                "source_file_sha256": source_hash,
                "source_content_sha256": human.source_sha256,
                "evaluator_protocol_path": evaluator_relative,
                "evaluator_protocol_sha256": evaluator_hash,
                "evaluator_code_path": evaluator_code_relative,
                "evaluator_code_sha256": evaluator_code_hash,
                "scale_protocol_path": scale_protocol_relative,
                "scale_protocol_sha256": scale_protocol_hash,
                "provenance_table_path": provenance_relative,
                "provenance_table_sha256": provenance_hash,
                **assets,
            }
        )
    ledger = pd.DataFrame(rows)
    ledger.to_csv(root / SCALE_INPUT_LEDGER, index=False)
    return ledger


_RAW_SCALE_REDUCTIONS: dict[str, tuple[str, str]] = {
    "rf_kpe_all_mean_m": ("rf_kpe_all_m", "mean"),
    "rf_kpe_all_p95_m": ("rf_kpe_all_m", "p95"),
    "rf_kpe_targeted_mean_m": ("rf_kpe_targeted_m", "mean"),
    "rf_kpe_targeted_p95_m": ("rf_kpe_targeted_m", "p95"),
    "rf_kpe_untracked_mean_m": ("rf_kpe_untracked_m", "mean"),
    "rf_kpe_untracked_p95_m": ("rf_kpe_untracked_m", "p95"),
    "bone_direction_mean_rad": ("bone_direction_error_rad", "mean"),
    "bone_direction_p95_rad": ("bone_direction_error_rad", "p95"),
    "bend_plane_mean_rad": ("bend_plane_error_rad", "mean"),
    "bend_plane_p95_rad": ("bend_plane_error_rad", "p95"),
    "root_translation_common_scale_mean_m": (
        "root_translation_common_scale_error_m",
        "mean",
    ),
    "root_translation_common_scale_p95_m": (
        "root_translation_common_scale_error_m",
        "p95",
    ),
    "root_translation_native_scale_mean_m": (
        "root_translation_native_scale_error_m",
        "mean",
    ),
    "root_translation_native_scale_p95_m": (
        "root_translation_native_scale_error_m",
        "p95",
    ),
    "root_translation_scale_invariant_mean_m": (
        "root_translation_scale_invariant_error_m",
        "mean",
    ),
    "root_translation_scale_invariant_p95_m": (
        "root_translation_scale_invariant_error_m",
        "p95",
    ),
    "root_yaw_mean_rad": ("root_yaw_error_rad", "mean"),
    "root_yaw_p95_rad": ("root_yaw_error_rad", "p95"),
    "joint_velocity_rms_mean_rad_s": ("joint_velocity_rms_rad_s", "mean"),
    "joint_velocity_rms_p95_rad_s": ("joint_velocity_rms_rad_s", "p95"),
    "joint_acceleration_rms_mean_rad_s2": (
        "joint_acceleration_rms_rad_s2",
        "mean",
    ),
    "joint_acceleration_rms_p95_rad_s2": (
        "joint_acceleration_rms_rad_s2",
        "p95",
    ),
    "joint_jerk_rms_mean_rad_s3": ("joint_jerk_rms_rad_s3", "mean"),
    "joint_jerk_rms_p95_rad_s3": ("joint_jerk_rms_rad_s3", "p95"),
    "pose_jump_mean_m": ("pose_jump_rms_m", "mean"),
    "pose_jump_p95_m": ("pose_jump_rms_m", "p95"),
    "foot_skating_frame_rate": ("foot_skating", "mean"),
    "ground_penetration_frame_rate": ("ground_penetration_artifact", "mean"),
    "ground_penetration_p95_m": ("ground_penetration_depth_m", "p95"),
    "joint_limit_violation_frame_rate": ("joint_limit_artifact", "mean"),
    "joint_limit_violation_p95_rad": ("joint_limit_violation_rad", "p95"),
    "invalid_frame_rate": ("invalid_artifact", "mean"),
    "artifact_rate": ("artifact", "mean"),
}


def _reduce_scale_per_frame(table: pd.DataFrame) -> dict[str, float]:
    result: dict[str, float] = {}
    for metric, (column, reduction) in _RAW_SCALE_REDUCTIONS.items():
        if column not in table:
            raise ValueError(f"Scale per-frame table lacks {column}")
        values = table[column].to_numpy(dtype=float)
        if len(values) != EXPECTED_FRAMES or not np.isfinite(values).all():
            raise ValueError(f"Scale per-frame metric {column} is incomplete")
        result[metric] = (
            float(np.mean(values))
            if reduction == "mean"
            else float(np.percentile(values, 95))
        )
    result["completion_ratio"] = float(len(table) / EXPECTED_FRAMES)
    return result


def _scale_target_diagnostics_from_artifact(
    robot: CanonicalRobotModel,
    motion: CanonicalG1,
    target_path: Path,
    reach_cache: dict[tuple[str, ...], dict[str, float]],
) -> dict[str, float]:
    from .scale_sensitivity import _reach_envelope_m

    with np.load(target_path, allow_pickle=False) as archive:
        target = np.asarray(archive["target_positions"], dtype=np.float64)
        labels = tuple(np.asarray(archive["landmark_names"]).astype(str).tolist())
    if target.shape != (EXPECTED_FRAMES, len(labels), 3) or "root" not in labels:
        raise ValueError(f"Invalid registered scale target artifact: {target_path}")
    frame_indices = np.asarray(motion.source_frame_idx, dtype=np.int64)
    target = target[frame_indices]
    robot_frames = [robot.semantic_positions(qpos) for qpos in motion.qpos]
    robot_target = np.stack(
        [[frame[label] for label in labels] for frame in robot_frames], axis=0
    )
    root_index = labels.index("root")
    target_root = target[:, root_index]
    robot_root = robot_target[:, root_index]
    target_local = target - target_root[:, None, :]
    robot_local = robot_target - robot_root[:, None, :]
    residual = np.linalg.norm(robot_local - target_local, axis=2)
    residual[:, root_index] = np.linalg.norm(
        (robot_root - robot_root[0]) - (target_root - target_root[0]), axis=1
    )
    if labels not in reach_cache:
        reach_cache[labels] = _reach_envelope_m(robot, list(labels), samples=512)
    envelope = reach_cache[labels]
    exceedance = np.zeros((len(target), len(labels)), dtype=bool)
    for index, label in enumerate(labels):
        if label != "root":
            exceedance[:, index] = (
                np.linalg.norm(target_local[:, index], axis=1)
                > float(envelope[label]) + 0.02
            )
    return {
        "standardized_semantic_target_residual_mean_m": float(residual.mean()),
        "standardized_semantic_target_residual_p95_m": float(
            np.percentile(residual, 95)
        ),
        "sampled_reach_envelope_exceedance_frame_rate": float(
            np.any(exceedance, axis=1).mean()
        ),
    }


def _scale_native_divergence(
    robot: CanonicalRobotModel,
    motion: CanonicalG1,
    native: CanonicalG1 | None,
    fk_cache: dict[str, np.ndarray],
    current_key: str,
    native_key: str,
) -> dict[str, float]:
    if native is None:
        return {"qpos_rms_from_native": np.nan, "fk_rms_from_native_m": np.nan}
    difference = np.asarray(motion.qpos - native.qpos, dtype=np.float64)
    difference[:, 7:] = np.arctan2(
        np.sin(difference[:, 7:]), np.cos(difference[:, 7:])
    )

    def fk(key: str, value: CanonicalG1) -> np.ndarray:
        if key not in fk_cache:
            fk_cache[key] = np.stack(
                [
                    np.concatenate(list(robot.semantic_positions(qpos).values()))
                    for qpos in value.qpos
                ]
            )
        return fk_cache[key]

    return {
        "qpos_rms_from_native": float(np.sqrt(np.mean(difference**2))),
        "fk_rms_from_native_m": float(
            np.sqrt(np.mean((fk(current_key, motion) - fk(native_key, native)) ** 2))
        ),
    }


def _scale_timing_metrics(
    motion: CanonicalG1, source_fps: float
) -> dict[str, float]:
    end_to_end = float(
        motion.metadata.get(
            "single_measurement_wall_time_s",
            motion.metadata.get(
                "steady_end_to_end_total_s",
                np.sum(motion.per_frame_solve_time_s),
            ),
        )
    )
    native = float(
        motion.metadata.get(
            "native_core_total_s", np.sum(motion.per_frame_solve_time_s)
        )
    )
    duration = len(motion.qpos) / float(source_fps)
    return {
        "scale_end_to_end_single_wall_s": end_to_end,
        "scale_end_to_end_single_rtf": end_to_end / duration,
        "scale_native_core_total_s": native,
        "scale_native_core_rtf": native / duration,
    }


def _assert_scale_value(actual: Any, expected: float, label: str) -> None:
    actual_value = float(actual)
    if np.isnan(expected):
        if not np.isnan(actual_value):
            raise ValueError(f"Scale metric {label} should be N/A")
    elif not np.isclose(actual_value, expected, rtol=1e-10, atol=1e-12):
        raise ValueError(
            f"Scale metric {label} differs from raw evidence: "
            f"{actual_value!r} != {expected!r}"
        )


def _assert_derived_frame_equal(
    actual: pd.DataFrame, expected: pd.DataFrame, sort_by: list[str], label: str
) -> None:
    actual = actual.sort_values(sort_by).reset_index(drop=True)
    expected = expected.sort_values(sort_by).reset_index(drop=True)
    if list(actual.columns) != list(expected.columns):
        raise ValueError(f"{label} columns differ from the registered derivation")
    try:
        pd.testing.assert_frame_equal(
            actual,
            expected,
            check_dtype=False,
            check_exact=False,
            rtol=1e-10,
            atol=1e-12,
        )
    except AssertionError as error:
        raise ValueError(f"{label} differs from the registered derivation") from error


def verify_scale_publication_input_ledger(
    repo_root: str | Path, *, recompute_metrics: bool = True
) -> bool:
    """Fail closed on missing rows, stale hashes, or summary-only scale claims."""

    root = Path(repo_root).resolve()
    ledger = pd.read_csv(root / SCALE_INPUT_LEDGER, keep_default_na=False)
    summary = pd.read_csv(root / "metrics/stage1_scale_policy_summary.csv")
    summary["method"] = summary.method.astype(str).map(_normalise_scale_method)
    expected_keys = _expected_scale_evidence_keys()
    if (
        len(SCALE_RESPONSE_METRIC_NAMES) != 43
        or len(ledger) != 30
        or len(summary) != 30
        or _scale_row_keys(ledger) != expected_keys
        or _scale_row_keys(summary) != expected_keys
        or ledger[["experiment_role", "method", "variant"]].duplicated().any()
        or not (ledger.schema_version.astype(int) == SCALE_INPUT_LEDGER_SCHEMA_VERSION).all()
    ):
        raise ValueError("Scale publication ledger does not contain exactly 25+5 rows")
    fixed = ledger.loc[ledger.fixed_contact_causal_arm.astype(str).str.lower().eq("true")]
    robustness = ledger.loc[
        ledger.experiment_role.astype(str).eq(CONTACT_ROBUSTNESS_ROLE)
    ]
    if len(fixed) != 25 or len(robustness) != 5:
        raise ValueError("Scale fixed-contact and robustness arms are not separated")

    # Authenticate the two files needed to interpret the rest of the ledger
    # before parsing either one.
    for row in ledger.itertuples(index=False):
        for path_column, hash_column in (
            ("output_path", "output_sha256"),
            ("target_manifest_path", "target_manifest_sha256"),
        ):
            path = root / str(getattr(row, path_column))
            if not path.is_file() or sha256_file(path) != str(
                getattr(row, hash_column)
            ):
                raise ValueError(
                    f"Scale ledger hash mismatch: {getattr(row, path_column)}"
                )

    targets = pd.read_csv(root / str(ledger.iloc[0].target_manifest_path))
    expected_target_keys = {
        ("controlled_dense", policy, "native") for policy in TRANSPLANT_POLICIES
    } | {
        (_RAW_SCALE_METHOD[method], _NATIVE_POLICY_BY_METHOD[method], variant)
        for method in NATIVE_SCALE_METHODS
        for variant in SCALE_VARIANTS
    }
    if len(targets) != 25 or set(
        zip(
            targets.method_scope.astype(str),
            targets.policy_id.astype(str),
            targets.variant.astype(str),
        )
    ) != expected_target_keys:
        raise ValueError("Scale target manifest does not contain exactly 25 rows")
    source_human = CanonicalHuman.load(root / str(ledger.iloc[0].source_path))
    if source_human.source_sha256 != str(ledger.iloc[0].source_content_sha256):
        raise ValueError("Scale ledger canonical source content hash is stale")

    path_hash_pairs = (
        ("output_path", "output_sha256"),
        ("raw_per_frame_csv_path", "raw_per_frame_csv_sha256"),
        ("raw_per_frame_parquet_path", "raw_per_frame_parquet_sha256"),
        ("raw_evaluator_summary_path", "raw_evaluator_summary_sha256"),
        ("target_manifest_path", "target_manifest_sha256"),
        ("registered_target_path", "registered_target_file_sha256"),
        ("runtime_evidence_path", "runtime_evidence_sha256"),
        ("config_path", "config_sha256"),
        ("method_code_path", "method_code_sha256"),
        ("environment_lock_path", "environment_lock_sha256"),
        ("source_path", "source_file_sha256"),
        ("evaluator_protocol_path", "evaluator_protocol_sha256"),
        ("evaluator_code_path", "evaluator_code_sha256"),
        ("scale_protocol_path", "scale_protocol_sha256"),
        ("provenance_table_path", "provenance_table_sha256"),
        ("solver_asset_path", "solver_asset_sha256"),
        ("evaluator_asset_path", "evaluator_asset_sha256"),
    )
    for row in ledger.itertuples(index=False):
        for path_column, hash_column in path_hash_pairs:
            path = root / str(getattr(row, path_column))
            if not path.is_file() or sha256_file(path) != str(getattr(row, hash_column)):
                raise ValueError(
                    f"Scale ledger hash mismatch: {getattr(row, path_column)}"
                )
        for optional_path, optional_hash in (
            ("runtime_input_path", "runtime_input_sha256"),
            ("formal_native_binding_path", "formal_native_binding_sha256"),
            ("accepted_formal_manifest_path", "accepted_formal_manifest_sha256"),
            ("accepted_formal_output_path", "accepted_formal_output_sha256"),
            ("accepted_formal_evidence_path", "accepted_formal_evidence_sha256"),
        ):
            relative = str(getattr(row, optional_path, ""))
            if relative:
                path = root / relative
                if not path.is_file() or sha256_file(path) != str(
                    getattr(row, optional_hash)
                ):
                    raise ValueError(f"Scale optional evidence hash mismatch: {relative}")
        execution = json.loads(str(row.scale_execution_contract_json))
        if (
            _canonical_sha256(execution) != str(row.scale_execution_contract_sha256)
            or execution
            != _scale_execution_binding(
                root, str(row.experiment_role), str(row.method)
            )
        ):
            raise ValueError("Scale execution contract is stale")
        motion = CanonicalG1.load(root / str(row.output_path))
        motion.validate(source_frame_count=EXPECTED_FRAMES)
        if (
            len(motion.qpos) != EXPECTED_FRAMES
            or not np.array_equal(motion.source_frame_idx, np.arange(EXPECTED_FRAMES))
            or not np.asarray(motion.valid, dtype=bool).all()
            or motion.metadata.get("canonical_source_sha256")
            != str(row.source_content_sha256)
        ):
            raise ValueError("Scale output violates the canonical motion contract")
        role = str(row.experiment_role)
        method = str(row.method)
        target_contract = _target_contract_row(
            targets, role, method, str(row.variant)
        )
        target_tensor_hash = _registered_target_tensor_sha256(
            root / str(row.registered_target_path)
        )
        if (
            target_tensor_hash != str(row.registered_target_tensor_sha256)
            or target_tensor_hash != str(target_contract.target_sha256)
            or not re.fullmatch(r"[0-9a-f]{64}", str(row.runtime_target_sha256))
        ):
            raise ValueError("Scale registered/runtime target hash is stale")
        runtime_expected = _runtime_scale_evidence(
            root,
            role,
            method,
            str(row.variant),
            root / str(row.output_path),
            motion,
            target_contract,
        )
        for field, expected_value in runtime_expected.items():
            actual_value = getattr(row, field, "")
            if isinstance(expected_value, bool):
                if str(actual_value).lower() != str(expected_value).lower():
                    raise ValueError(f"Scale runtime evidence field is stale: {field}")
            elif str(actual_value) != str(expected_value):
                raise ValueError(f"Scale runtime evidence field is stale: {field}")
        if role == "controlled_policy":
            metadata_ok = bool(
                motion.metadata.get("experiment_role") == "controlled_policy_ablation"
                and motion.metadata.get("policy_id") == str(row.variant)
                and motion.metadata.get("config_sha256") == str(row.config_sha256)
                and motion.metadata.get("evaluator_sha256")
                == str(row.evaluator_protocol_sha256)
                and motion.metadata.get("canonical_robot_xml_sha256")
                == str(row.evaluator_asset_sha256)
                and motion.metadata.get("fixed_contact_labels") is True
            )
        else:
            metadata_ok = bool(
                motion.metadata.get("scale_execution_contract") == execution
                and motion.metadata.get("config_sha256") == str(row.config_sha256)
                and motion.metadata.get("scale_protocol_sha256")
                == str(row.scale_protocol_sha256)
                and isinstance(motion.metadata.get("scale_runtime_environment"), Mapping)
                and motion.metadata["scale_runtime_environment"].get("environment_name")
                == str(row.environment_name)
            )
            if method == "omniretarget":
                metadata_ok = bool(
                    metadata_ok
                    and motion.metadata.get("solver_robot_xml_sha256")
                    == str(row.solver_asset_sha256)
                    and motion.metadata.get("canonical_evaluator_robot_scene_sha256")
                    == str(row.evaluator_asset_sha256)
                    and motion.metadata.get("solver_and_evaluator_assets_identical")
                    is False
                    and str(row.solver_asset_sha256)
                    != str(row.evaluator_asset_sha256)
                )
        if not metadata_ok:
            raise ValueError(f"Scale output metadata is stale for {role}/{method}/{row.variant}")

    provenance = pd.read_csv(root / str(ledger.iloc[0].provenance_table_path))
    provenance["method"] = provenance.method.astype(str).map(_normalise_scale_method)
    joined = ledger.merge(
        provenance,
        on=["experiment_role", "method", "variant", "output_path", "output_sha256"],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if len(provenance) != 30 or not joined._merge.astype(str).eq("both").all():
        raise ValueError("Scale provenance table does not match the 30-row ledger")
    for column in (
        "source_match",
        "config_match",
        "upstream_match",
        "method_match",
        "robot_match",
        "role_match",
        "scale_protocol_match",
        "execution_contract_match",
        "variant_match",
        "contact_policy_match",
        "exact_full_frames",
        "accepted",
    ):
        values = provenance[column]
        if not values.astype(str).str.lower().isin({"true", "1"}).all():
            raise ValueError(f"Scale provenance rejected column {column}")

    if not recompute_metrics:
        return True

    human = source_human
    robot = CanonicalRobotModel(default_robot_scene(root))
    reach_cache: dict[tuple[str, ...], dict[str, float]] = {}
    fk_cache: dict[str, np.ndarray] = {}
    motion_by_key = {
        (str(row.experiment_role), str(row.method), str(row.variant)): CanonicalG1.load(
            root / str(row.output_path)
        )
        for row in ledger.itertuples(index=False)
    }
    summary_lookup = summary.set_index(["experiment_role", "method", "variant"])
    for row in ledger.itertuples(index=False):
        key = (str(row.experiment_role), str(row.method), str(row.variant))
        motion = motion_by_key[key]
        csv_table = pd.read_csv(root / str(row.raw_per_frame_csv_path))
        parquet_table = pd.read_parquet(root / str(row.raw_per_frame_parquet_path))
        try:
            pd.testing.assert_frame_equal(
                csv_table,
                parquet_table,
                check_dtype=False,
                check_exact=False,
                rtol=1e-10,
                atol=1e-12,
            )
        except AssertionError as error:
            raise ValueError(f"Scale CSV/Parquet mismatch for {key}") from error
        if not np.array_equal(
            csv_table.source_frame_idx.to_numpy(dtype=np.int64),
            np.arange(EXPECTED_FRAMES, dtype=np.int64),
        ):
            raise ValueError(f"Scale raw frame indices are incomplete for {key}")
        recomputed = _reduce_scale_per_frame(csv_table)
        recomputed.update(
            _scale_target_diagnostics_from_artifact(
                robot,
                motion,
                root / str(row.registered_target_path),
                reach_cache,
            )
        )
        native_key = (key[0], key[1], "native")
        native_motion = None if key[0] == "controlled_policy" else motion_by_key[native_key]
        recomputed.update(
            _scale_native_divergence(
                robot,
                motion,
                native_motion,
                fk_cache,
                "|".join(key),
                "|".join(native_key),
            )
        )
        recomputed.update(_scale_timing_metrics(motion, human.fps))
        if set(recomputed) != set(SCALE_RESPONSE_METRIC_NAMES):
            raise ValueError("Scale raw reconstruction does not cover the exact 43 metrics")
        summary_row = summary_lookup.loc[key]
        evaluator_summary = json.loads(
            (root / str(row.raw_evaluator_summary_path)).read_text(encoding="utf-8")
        )
        if (
            evaluator_summary.get("evaluator_protocol_sha256")
            != str(row.evaluator_protocol_sha256)
            or evaluator_summary.get("robot_model_sha256")
            != str(row.evaluator_asset_sha256)
        ):
            raise ValueError(f"Scale evaluator summary is stale for {key}")
        for metric, expected_value in recomputed.items():
            _assert_scale_value(summary_row[metric], expected_value, f"{key}/{metric}")
            if metric in _RAW_SCALE_REDUCTIONS or metric == "completion_ratio":
                _assert_scale_value(
                    evaluator_summary[metric], expected_value, f"{key}/raw/{metric}"
                )

    slopes_expected, ranks_expected = derive_scale_tables(summary.reset_index(drop=True))
    slopes = pd.read_csv(root / "metrics/stage1_scale_sensitivity_slopes.csv")
    ranks = pd.read_csv(root / "metrics/stage1_method_rank_stability.csv")
    if len(slopes) != 4 * 2 * 43 or len(ranks) != 4 * 5 * len(SCALE_RANK_METRICS):
        raise ValueError("Scale slope/rank tables have unexpected row counts")
    _assert_derived_frame_equal(
        slopes,
        slopes_expected,
        ["method", "factor", "metric"],
        "Scale slopes",
    )
    _assert_derived_frame_equal(
        ranks,
        ranks_expected,
        ["metric", "variant", "method"],
        "Scale ranks",
    )
    robustness_summary = summary.loc[
        summary.experiment_role.astype(str).eq(CONTACT_ROBUSTNESS_ROLE)
    ]
    robustness_expected = _derive_scale_slopes(
        robustness_summary, ("omniretarget",)
    )
    robustness_slopes = pd.read_csv(
        root / "metrics/stage1_holosoma_native_contact_slopes.csv"
    )
    if len(robustness_slopes) != 2 * 43:
        raise ValueError("Holosoma robustness slope table has unexpected row count")
    _assert_derived_frame_equal(
        robustness_slopes,
        robustness_expected,
        ["method", "factor", "metric"],
        "Holosoma robustness slopes",
    )
    return True


def collect_scale_evidence(repo_root: str | Path) -> ScaleBundle:
    root = Path(repo_root).resolve()
    path = root / "metrics" / "scale_policy_sensitivity_summary.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Scale-policy summary is missing: {path}")
    summary = pd.read_csv(path)
    required_columns = {
        "experiment_role",
        "method",
        "variant",
        "frames",
        "completion_ratio",
        "output_path",
        "output_sha256",
        *QUALITY_COLUMNS,
        *SCALE_RESPONSE_METRIC_NAMES,
    }
    missing_columns = required_columns - set(summary)
    if missing_columns:
        raise ValueError(f"Scale-policy summary lacks {sorted(missing_columns)}")
    summary["method"] = summary.method.astype(str).map(_normalise_scale_method)
    expected_keys = _expected_scale_evidence_keys()
    if len(summary) != len(expected_keys) or _scale_row_keys(summary) != expected_keys:
        raise ValueError("Scale-policy summary must contain exactly 25+5 registered rows")
    transplants = summary.loc[summary.experiment_role == "controlled_policy"].copy()
    native = summary.loc[
        summary.experiment_role == FORMAL_NATIVE_SCALE_ROLE
    ].copy()
    robustness = summary.loc[
        summary.experiment_role == CONTACT_ROBUSTNESS_ROLE
    ].copy()
    missing_policies = set(TRANSPLANT_POLICIES) - set(transplants.variant.astype(str))
    if missing_policies:
        raise ValueError(f"Controlled policy transplants lack {sorted(missing_policies)}")
    if len(transplants.loc[transplants.variant.isin(TRANSPLANT_POLICIES)]) != len(
        TRANSPLANT_POLICIES
    ):
        raise ValueError("Controlled policy transplants contain duplicates")
    for method in NATIVE_SCALE_METHODS:
        variants = set(native.loc[native.method == method, "variant"].astype(str))
        missing = set(SCALE_VARIANTS) - variants
        if missing:
            raise ValueError(f"Native scale response for {method} lacks {sorted(missing)}")
    robustness_methods = set(robustness.method.astype(str).map(_normalise_scale_method))
    if robustness_methods != {"omniretarget"}:
        raise ValueError(
            "The native-recomputed-contact robustness arm must contain Holosoma only"
        )
    robustness_variants = set(robustness.variant.astype(str))
    missing_robustness = set(SCALE_VARIANTS) - robustness_variants
    if missing_robustness:
        raise ValueError(
            "Holosoma native-recomputed-contact arm lacks "
            + str(sorted(missing_robustness))
        )
    if len(robustness) != len(SCALE_VARIANTS):
        raise ValueError("Holosoma native-recomputed-contact variants are duplicated")
    selected = pd.concat([transplants, native, robustness], ignore_index=True)
    if not (selected.frames.astype(int) == EXPECTED_FRAMES).all():
        raise ValueError("Every scale-policy output must contain 600 frames")
    if not np.allclose(selected.completion_ratio.astype(float), 1.0):
        raise ValueError("Every scale-policy output must be complete")
    numeric = selected.loc[:, [column for column in QUALITY_COLUMNS if column in selected]]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Scale-policy evidence contains NaN/Inf")
    for row in selected.itertuples():
        output = root / str(row.output_path)
        if not output.is_file():
            raise FileNotFoundError(f"Scale output is missing: {output}")
        if sha256_file(output) != str(row.output_sha256):
            raise ValueError(f"Scale output hash mismatch: {output}")
    slopes, ranks = derive_scale_tables(summary)
    robustness_slopes = _derive_scale_slopes(robustness.copy(), ("omniretarget",))
    formal_holosoma = native.loc[native.method == "omniretarget"].copy()
    comparison_columns = [
        "variant",
        "rf_kpe_all_mean_m",
        "root_translation_common_scale_mean_m",
        "artifact_rate",
        "foot_skating_frame_rate",
    ]
    contact_robustness = formal_holosoma[comparison_columns].merge(
        robustness[comparison_columns],
        on="variant",
        suffixes=("_fixed_canonical", "_native_recomputed"),
        validate="one_to_one",
    )
    contact_robustness["_order"] = contact_robustness.variant.map(
        {variant: index for index, variant in enumerate(SCALE_VARIANTS)}
    )
    contact_robustness = contact_robustness.sort_values("_order").drop(
        columns="_order"
    )
    for metric in comparison_columns[1:]:
        contact_robustness[f"{metric}_native_recomputed_minus_fixed"] = (
            contact_robustness[f"{metric}_native_recomputed"]
            - contact_robustness[f"{metric}_fixed_canonical"]
        )
    diagnostics_path = root / "metrics" / "contact_constraint_flip.csv"
    if not diagnostics_path.is_file():
        raise FileNotFoundError(f"Contact-label diagnostic is missing: {diagnostics_path}")
    diagnostics = pd.read_csv(diagnostics_path)
    holosoma_diagnostics = diagnostics.loc[
        diagnostics.method.astype(str).eq("omniretarget")
    ].copy()
    if set(holosoma_diagnostics.variant.astype(str)) != set(SCALE_VARIANTS):
        raise ValueError("Holosoma contact-label diagnostics are incomplete")
    required_contact_columns = {
        "recomputed_label_flip_rate",
        "native_recomputed_vs_formal_canonical_label_disagreement_rate",
    }
    if not required_contact_columns.issubset(holosoma_diagnostics):
        raise ValueError("Holosoma contact-label diagnostics lack disagreement fields")
    _write_frame(summary, root / "metrics" / "stage1_scale_policy_summary")
    slopes.to_csv(root / "metrics" / "stage1_scale_sensitivity_slopes.csv", index=False)
    ranks.to_csv(root / "metrics" / "stage1_method_rank_stability.csv", index=False)
    contact_robustness.to_csv(
        root / "metrics" / "stage1_holosoma_contact_robustness.csv", index=False
    )
    robustness_slopes.to_csv(
        root / "metrics" / "stage1_holosoma_native_contact_slopes.csv", index=False
    )
    build_scale_publication_input_ledger(root, summary)
    verify_scale_publication_input_ledger(root, recompute_metrics=True)
    return ScaleBundle(
        summary=summary,
        transplants=transplants,
        native_response=native,
        slopes=slopes,
        ranks=ranks,
        contact_robustness=contact_robustness,
        contact_robustness_slopes=robustness_slopes,
        contact_diagnostics=holosoma_diagnostics,
    )


def build_reference_comparison(
    core: pd.DataFrame, reference: pd.DataFrame
) -> pd.DataFrame:
    """Build descriptive deltas without treating the reference as truth."""

    if len(reference) != 1:
        raise ValueError("Exactly one external reference row is required")
    ref = reference.iloc[0]
    rows: list[dict[str, Any]] = []
    for method in core.itertuples():
        for metric in QUALITY_COLUMNS:
            method_value = _numeric(getattr(method, metric), metric)
            reference_value = _numeric(ref[metric], metric)
            rows.append(
                {
                    "key": method.key,
                    "display_name": method.display_name,
                    "reference_key": str(ref.key),
                    "reference_display_name": str(ref.display_name),
                    "metric": metric,
                    "method_value": method_value,
                    "reference_value": reference_value,
                    "signed_method_minus_reference": method_value - reference_value,
                    "absolute_difference": abs(method_value - reference_value),
                    "comparison_type": "evaluator_metric_delta",
                    "interpretation": "descriptive difference; reference is not ground truth",
                }
            )
    return pd.DataFrame(rows)


def build_direct_reference_comparison(
    repo_root: str | Path,
    paths: Mapping[str, Path] | None = None,
) -> pd.DataFrame:
    """Measure direct same-G1 trajectory disagreement on the frozen Pilot."""

    root = Path(repo_root).resolve()
    outputs = dict(paths or method_output_paths(root))
    reference_path = outputs[REFERENCE_METHOD.key]
    reference = CanonicalG1.load(reference_path)
    robot = CanonicalRobotModel(default_robot_scene(root))
    semantic_names = tuple(sorted(robot.body_ids)) + ("head",)
    rows: list[dict[str, Any]] = []
    for spec in CORE_METHODS:
        motion_path = outputs[spec.key]
        motion = CanonicalG1.load(motion_path)
        if (
            motion.qpos.shape != reference.qpos.shape
            or not np.array_equal(motion.source_frame_idx, reference.source_frame_idx)
            or not np.asarray(motion.valid, dtype=bool).all()
            or not np.asarray(reference.valid, dtype=bool).all()
        ):
            raise ValueError(
                f"Direct external-reference timeline mismatch for {spec.key}"
            )
        method_qpos = np.asarray(motion.qpos, dtype=np.float64)
        reference_qpos = np.asarray(reference.qpos, dtype=np.float64)
        root_distance = np.linalg.norm(
            method_qpos[:, :3] - reference_qpos[:, :3], axis=1
        )
        root_anchor_distance = float(
            np.linalg.norm(method_qpos[0, :3] - reference_qpos[0, :3])
        )
        method_root_displacement = method_qpos[:, :3] - method_qpos[0, :3]
        reference_root_displacement = (
            reference_qpos[:, :3] - reference_qpos[0, :3]
        )
        root_displacement_distance = np.linalg.norm(
            method_root_displacement - reference_root_displacement, axis=1
        )
        method_rotation = quaternion_wxyz_to_matrix(method_qpos[:, 3:7])
        reference_rotation = quaternion_wxyz_to_matrix(reference_qpos[:, 3:7])
        method_yaw = yaw_from_matrix(method_rotation)
        reference_yaw = yaw_from_matrix(reference_rotation)
        yaw_delta = np.abs(
            (method_yaw - reference_yaw + np.pi)
            % (2.0 * np.pi)
            - np.pi
        )
        relative_rotation = np.einsum(
            "tji,tjk->tik", reference_rotation, method_rotation
        )
        cosine = np.clip(
            (np.trace(relative_rotation, axis1=1, axis2=2) - 1.0) / 2.0,
            -1.0,
            1.0,
        )
        orientation_delta = np.arccos(cosine)
        joint_delta = (
            method_qpos[:, 7:] - reference_qpos[:, 7:] + np.pi
        ) % (2.0 * np.pi) - np.pi
        # Differentiate each periodic trajectory before taking the velocity
        # difference.  Differentiating the already wrapped pose disagreement
        # would create artificial 2*pi spikes whenever that disagreement
        # crosses the -pi/pi branch cut.
        method_joint_step = (
            np.diff(method_qpos[:, 7:], axis=0) + np.pi
        ) % (2.0 * np.pi) - np.pi
        reference_joint_step = (
            np.diff(reference_qpos[:, 7:], axis=0) + np.pi
        ) % (2.0 * np.pi) - np.pi
        joint_velocity_delta = (
            method_joint_step - reference_joint_step
        ) * float(motion.fps)
        world_distances: list[float] = []
        root_frame_distances: list[float] = []
        for frame_index, (method_frame, reference_frame) in enumerate(
            zip(method_qpos, reference_qpos, strict=True)
        ):
            method_positions = robot.semantic_positions(method_frame)
            reference_positions = robot.semantic_positions(reference_frame)
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
        metrics = {
            # These two absolute-position metrics intentionally include the
            # two trajectories' different frame-zero anchors.
            "direct_root_translation_mean_m": float(np.mean(root_distance)),
            "direct_root_translation_p95_m": float(
                np.percentile(root_distance, 95.0)
            ),
            "direct_root_anchor_frame0_m": root_anchor_distance,
            "direct_root_displacement_mean_m": float(
                np.mean(root_displacement_distance)
            ),
            "direct_root_displacement_p95_m": float(
                np.percentile(root_displacement_distance, 95.0)
            ),
            "direct_root_yaw_mean_rad": float(np.mean(yaw_delta)),
            "direct_root_orientation_mean_rad": float(
                np.mean(orientation_delta)
            ),
            "direct_joint_angle_rmse_rad": float(
                np.sqrt(np.mean(np.square(joint_delta)))
            ),
            "direct_joint_velocity_rmse_rad_s": float(
                np.sqrt(np.mean(np.square(joint_velocity_delta)))
            ),
            "direct_fk_world_semantic_mean_m": float(np.mean(world_distances)),
            "direct_fk_root_frame_semantic_mean_m": float(
                np.mean(root_frame_distances)
            ),
        }
        if not np.isfinite(np.asarray(list(metrics.values()), dtype=float)).all():
            raise ValueError(f"Direct reference metrics are non-finite for {spec.key}")
        for metric, value in metrics.items():
            rows.append(
                {
                    "key": spec.key,
                    "display_name": spec.display_name,
                    "reference_key": REFERENCE_METHOD.key,
                    "reference_display_name": REFERENCE_METHOD.display_name,
                    "metric": metric,
                    "method_value": value,
                    "reference_value": 0.0,
                    "signed_method_minus_reference": value,
                    "absolute_difference": value,
                    "comparison_type": (
                        "direct_same_g1_frame_index_aligned_trajectory_disagreement"
                    ),
                    "method_output_sha256": sha256_file(motion_path),
                    "reference_output_sha256": sha256_file(reference_path),
                    "method_fps": float(motion.fps),
                    "reference_fps": float(reference.fps),
                    "frames": len(motion.qpos),
                    "timeline_alignment": "frame_index_only_not_exact_timestamp",
                    "absolute_root_metrics_include_frame0_anchor": metric.startswith(
                        "direct_root_translation_"
                    ),
                    "interpretation": (
                        "frame-index-aligned same-G1 trajectory disagreement; lower "
                        "is closer to the external trajectory under this metric, not "
                        "higher verified accuracy; absolute root translation includes "
                        "the frame-zero anchor while displacement metrics remove it"
                    ),
                }
            )
    return pd.DataFrame(rows)


PALETTE = {
    "sparse-neutral": "#4C78A8",
    "dense": "#72B7B2",
    "gmr": "#F58518",
    "omniretarget": "#E45756",
    "protomotions-v2.3": "#54A24B",
    "protomotions-v3": "#B279A2",
    "unitree-reference": "#222222",
}


def save_publication_chart(
    repo_root: str | Path,
    name: str,
    source: pd.DataFrame,
    renderer: Callable[[Any, pd.DataFrame], Any],
) -> tuple[Path, Path, Path, Path]:
    """Save one chart and its exact CSV source in all publication formats."""

    root = Path(repo_root).resolve()
    figure_dir = root / "figures" / "stage1_publication"
    source_dir = figure_dir / "source_data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    csv_path = source_dir / f"{name}.csv"
    source.to_csv(csv_path, index=False)

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    style = {
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.alpha": 0.20,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.hashsalt": "retargeting-comparison-stage1",
    }
    with mpl.rc_context(style):
        figure = renderer(plt, source.copy())
        if figure is None:
            raise ValueError(f"Chart renderer {name!r} returned no figure")
        outputs = []
        for suffix, kwargs in (
            ("svg", {"metadata": {"Date": None, "Creator": "rtcmp"}}),
            ("pdf", {"metadata": {"CreationDate": None, "Creator": "rtcmp"}}),
            ("png", {"dpi": 300, "metadata": {"Software": "rtcmp"}}),
        ):
            path = figure_dir / f"{name}.{suffix}"
            figure.savefig(path, bbox_inches="tight", **kwargs)
            outputs.append(path)
        plt.close(figure)
    return csv_path, outputs[0], outputs[1], outputs[2]


def _annotated_scatter(plt: Any, data: pd.DataFrame) -> Any:
    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for row in data.itertuples():
        reference = row.key == REFERENCE_METHOD.key
        axis.scatter(
            row.rf_kpe_targeted_mean_m,
            row.rf_kpe_untracked_mean_m,
            s=145 if reference else 65,
            marker="*" if reference else "o",
            color=PALETTE[row.key],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        axis.annotate(
            row.display_name,
            (row.rf_kpe_targeted_mean_m, row.rf_kpe_untracked_mean_m),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlabel("Targeted hand/foot RF-KPE (m; lower is better)")
    axis.set_ylabel("Untracked-body RF-KPE (m; lower is better)")
    axis.set_title("One Pilot: targeted fidelity does not determine untracked pose")
    return figure


def _timing_quality_chart(plt: Any, data: pd.DataFrame) -> Any:
    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for row in data.itertuples():
        axis.scatter(
            row.end_to_end_rtf_median,
            row.rf_kpe_all_mean_m,
            s=70,
            color=PALETTE[row.key],
            edgecolor="white",
            linewidth=0.8,
        )
        axis.annotate(
            row.display_name,
            (row.end_to_end_rtf_median, row.rf_kpe_all_mean_m),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xscale("log")
    axis.axvline(1.0, color="#777777", linestyle="--", linewidth=1, label="real time")
    axis.set_xlabel("Steady end-to-end RTF (log scale; lower is faster)")
    axis.set_ylabel("RF-KPE-all mean (m; lower is better)")
    axis.set_title("Quality/runtime operating points; reference excluded from timing")
    axis.legend(frameon=False, loc="best")
    return figure


def _root_yaw_chart(plt: Any, data: pd.DataFrame) -> Any:
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.3), constrained_layout=True)
    x = np.arange(len(data))
    width = 0.37
    colors = [PALETTE[key] for key in data.key]
    axes[0].bar(
        x - width / 2,
        data.root_translation_common_scale_mean_m,
        width,
        color=colors,
        alpha=0.95,
        label="common-scale error",
    )
    axes[0].bar(
        x + width / 2,
        data.root_translation_scale_invariant_mean_m,
        width,
        color=colors,
        alpha=0.38,
        hatch="//",
        label="scale-invariant path error",
    )
    axes[0].set_ylabel("Mean root translation error (m)")
    axes[0].set_title("Scale magnitude vs path shape")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].bar(x, data.root_yaw_mean_rad, color=colors)
    axes[1].set_ylabel("Mean root yaw error (rad)")
    axes[1].set_title("Heading fidelity")
    labels = [str(value).replace(" / ", "/\n") for value in data.display_name]
    for axis in axes:
        axis.set_xticks(x, labels, rotation=28, ha="right", fontsize=7.5)
    return figure


def _temporal_artifact_chart(plt: Any, data: pd.DataFrame) -> Any:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    x = np.arange(len(data))
    colors = [PALETTE[key] for key in data.key]
    axes[0].bar(x, data.joint_jerk_rms_p95_rad_s3, color=colors, label="joint jerk p95")
    temporal_right = axes[0].twinx()
    temporal_right.spines["right"].set_visible(True)
    temporal_right.plot(
        x,
        data.pose_jump_p95_m,
        color="#222222",
        marker="D",
        linewidth=1.2,
        label="pose jump p95",
    )
    axes[0].set_yscale("symlog", linthresh=1.0)
    axes[0].set_ylabel("Joint jerk p95 (rad/s³; symlog)")
    temporal_right.set_ylabel("Pose jump p95 (m)")
    axes[0].set_title("Temporal diagnostics")
    left_handles, left_labels = axes[0].get_legend_handles_labels()
    right_handles, right_labels = temporal_right.get_legend_handles_labels()
    axes[0].legend(
        left_handles + right_handles,
        left_labels + right_labels,
        frameon=False,
        fontsize=7.5,
    )
    bottom = np.zeros(len(data))
    for column, label, color in (
        ("foot_skating_frame_rate", "foot skating", "#4C78A8"),
        ("ground_penetration_frame_rate", "ground penetration", "#F58518"),
        ("joint_limit_violation_frame_rate", "joint limit", "#E45756"),
        ("invalid_frame_rate", "invalid", "#777777"),
    ):
        values = data[column].to_numpy(dtype=float)
        axes[1].bar(x, values, bottom=bottom, color=color, label=label)
        bottom += values
    axes[1].set_ylabel("Component frame rates (stacked; causes may overlap)")
    axes[1].set_title("Named artifact components")
    axes[1].legend(frameon=False, fontsize=7.5)
    labels = [str(value).replace(" / ", "/\n") for value in data.display_name]
    for axis in axes:
        axis.set_xticks(x, labels, rotation=28, ha="right", fontsize=7.5)
    return figure


def _scale_transplant_chart(plt: Any, data: pd.DataFrame) -> Any:
    figure, axes = plt.subplots(1, 2, figsize=(10.3, 4.2), constrained_layout=True)
    labels = [str(value).replace("_", "\n") for value in data.variant]
    x = np.arange(len(data))
    axes[0].bar(x, data.root_translation_common_scale_mean_m, color="#4C78A8")
    axes[0].set_ylabel("Common-scale root error (m)")
    axes[0].set_title("Same Dense solver, transplanted scale formula")
    axes[1].bar(x, data.rf_kpe_all_mean_m, color="#72B7B2")
    axes[1].set_ylabel("RF-KPE-all (m)")
    axes[1].set_title("Morphology-referenced pose fidelity")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=25, ha="right", fontsize=7.2)
    return figure


def _scale_slope_chart(plt: Any, data: pd.DataFrame) -> Any:
    metrics = [
        "rf_kpe_all_mean_m",
        "root_translation_common_scale_mean_m",
        "artifact_rate",
        "foot_skating_frame_rate",
    ]
    columns = [(factor, metric) for metric in metrics for factor in ("root", "local")]
    matrix = np.asarray(
        [
            [
                data.loc[
                    (data.method == method)
                    & (data.factor == factor)
                    & (data.metric == metric),
                    "elasticity_at_native",
                ].iloc[0]
                for factor, metric in columns
            ]
            for method in NATIVE_SCALE_METHODS
        ],
        dtype=float,
    )
    finite = np.abs(matrix[np.isfinite(matrix)])
    limit = max(float(np.percentile(finite, 90)) if len(finite) else 1.0, 1e-6)
    clipped = np.clip(matrix, -limit, limit)
    figure, axis = plt.subplots(figsize=(10.5, 4.1), constrained_layout=True)
    image = axis.imshow(clipped, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:.2g}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(clipped[row, column]) > limit * 0.55 else "black",
            )
    axis.set_yticks(np.arange(len(NATIVE_SCALE_METHODS)), NATIVE_SCALE_METHODS)
    axis.set_xticks(
        np.arange(len(columns)),
        [f"{metric.replace('_mean_m', '').replace('_frame_rate', '')}\n{factor}" for factor, metric in columns],
        rotation=30,
        ha="right",
        fontsize=7,
    )
    axis.set_title("Native ±5% response elasticity (numbers unclipped; color robust-clipped)")
    figure.colorbar(image, ax=axis, label="elasticity at native")
    return figure


def _rank_stability_chart(plt: Any, data: pd.DataFrame) -> Any:
    grouped = (
        data.groupby(["metric", "variant"], as_index=False)
        .agg(spearman_vs_native=("spearman_vs_native", "first"))
    )
    metrics = list(SCALE_RANK_METRICS)
    matrix = (
        grouped.pivot(index="metric", columns="variant", values="spearman_vs_native")
        .reindex(index=metrics, columns=SCALE_VARIANTS)
        .to_numpy(dtype=float)
    )
    figure, axis = plt.subplots(
        figsize=(9.6, max(5.0, 0.28 * len(metrics))), constrained_layout=True
    )
    image = axis.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if abs(value) > 0.55 else "black",
            )
    axis.set_xticks(
        np.arange(len(SCALE_VARIANTS)),
        [value.replace("_", "\n") for value in SCALE_VARIANTS],
    )
    axis.set_yticks(
        np.arange(len(metrics)),
        [metric.replace("_", " ") for metric in metrics],
        fontsize=6.5,
    )
    axis.set_title("Per-metric method-rank stability under ±5% target scale")
    figure.colorbar(image, ax=axis, label="Spearman rank correlation vs native")
    return figure


def _contact_robustness_chart(plt: Any, data: pd.DataFrame) -> Any:
    figure, axes = plt.subplots(1, 2, figsize=(9.8, 4.1), constrained_layout=True)
    x = np.arange(len(data))
    width = 0.37
    for axis, metric, ylabel in (
        (axes[0], "root_translation_common_scale_mean_m", "Common-scale root error (m)"),
        (axes[1], "artifact_rate", "Artifact frame rate"),
    ):
        axis.bar(
            x - width / 2,
            data[f"{metric}_fixed_canonical"],
            width,
            color="#4C78A8",
            label="fixed canonical contacts",
        )
        axis.bar(
            x + width / 2,
            data[f"{metric}_native_recomputed"],
            width,
            color="#F58518",
            label="native recomputed contacts",
        )
        axis.set_xticks(x, [value.replace("_", "\n") for value in data.variant], fontsize=7)
        axis.set_ylabel(ylabel)
    axes[0].set_title("Holosoma scale response")
    axes[1].set_title("Contact-policy robustness")
    axes[1].legend(frameon=False, fontsize=7.5)
    figure.suptitle("Robustness arm is separate from the fixed-contact causal analysis")
    return figure


def _reference_delta_chart(plt: Any, data: pd.DataFrame) -> Any:
    selected = data.loc[
        data.metric.isin(
            (
                "rf_kpe_targeted_mean_m",
                "rf_kpe_untracked_mean_m",
                "root_translation_common_scale_mean_m",
                "root_yaw_mean_rad",
            )
        )
    ].copy()
    pivot = selected.pivot(index="key", columns="metric", values="signed_method_minus_reference")
    pivot = pivot.reindex([spec.key for spec in CORE_METHODS])
    figure, axes = plt.subplots(2, 2, figsize=(10.2, 6.8), constrained_layout=True)
    for axis, metric in zip(axes.flat, pivot.columns):
        values = pivot[metric]
        colors = [PALETTE[key] for key in pivot.index]
        axis.bar(np.arange(len(values)), values, color=colors)
        axis.axhline(0.0, color="#222222", linewidth=0.9)
        axis.set_title(metric.replace("_", " "), fontsize=9)
        axis.set_xticks(
            np.arange(len(values)),
            [key.replace("protomotions-", "proto-") for key in pivot.index],
            rotation=28,
            ha="right",
            fontsize=7,
        )
        axis.set_ylabel("method minus external reference")
    figure.suptitle("Descriptive deltas only: the external corpus is not ground truth")
    return figure


def _direct_reference_chart(plt: Any, data: pd.DataFrame) -> Any:
    selected = data.loc[
        data.metric.isin(
            (
                "direct_root_displacement_mean_m",
                "direct_fk_world_semantic_mean_m",
                "direct_fk_root_frame_semantic_mean_m",
                "direct_root_yaw_mean_rad",
                "direct_joint_angle_rmse_rad",
            )
        )
    ].copy()
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), constrained_layout=True)
    if selected.empty:
        for axis in axes:
            axis.text(0.5, 0.5, "Direct trajectory evidence unavailable", ha="center")
            axis.set_axis_off()
        return figure
    groups = (
        (
            axes[0],
            (
                "direct_root_displacement_mean_m",
                "direct_fk_world_semantic_mean_m",
                "direct_fk_root_frame_semantic_mean_m",
            ),
            "Distance disagreement (m)",
        ),
        (
            axes[1],
            ("direct_root_yaw_mean_rad", "direct_joint_angle_rmse_rad"),
            "Angular disagreement (rad)",
        ),
    )
    keys = [spec.key for spec in CORE_METHODS]
    x = np.arange(len(keys))
    for axis, metrics, ylabel in groups:
        width = 0.8 / len(metrics)
        for index, metric in enumerate(metrics):
            values = (
                selected.loc[selected.metric.eq(metric)]
                .set_index("key")
                .reindex(keys)
                .method_value
            )
            axis.bar(
                x + (index - (len(metrics) - 1) / 2.0) * width,
                values,
                width=width,
                label=metric.removeprefix("direct_").replace("_mean", "").replace("_", " "),
            )
        axis.set_xticks(
            x,
            [key.replace("protomotions-", "proto-") for key in keys],
            rotation=28,
            ha="right",
            fontsize=7,
        )
        axis.set_ylabel(ylabel)
        axis.legend(frameon=False, fontsize=7)
    figure.suptitle(
        "Direct same-G1 disagreement with external trajectory (zero means identity, not truth)"
    )
    return figure


def _sparse_seed_chart(plt: Any, data: pd.DataFrame) -> Any:
    figure, axis = plt.subplots(figsize=(7.2, 4.3), constrained_layout=True)
    x = np.arange(len(data))
    axis.bar(x, data.robot_rf_point_rms_mean_m, color="#4C78A8")
    axis.errorbar(
        x,
        data.robot_rf_point_rms_mean_m,
        yerr=np.maximum(
            data.robot_rf_point_rms_p95_m - data.robot_rf_point_rms_mean_m, 0
        ),
        fmt="none",
        ecolor="#222222",
        capsize=3,
    )
    axis.set_xticks(x, [value.replace("__", "\nvs\n") for value in data.pair])
    axis.set_ylabel("Robot root-frame point RMS (m)")
    axis.set_title("Sparse initialization sensitivity: mean with p95 whisker")
    return figure


def _interaction_chart(plt: Any, data: pd.DataFrame) -> Any:
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 4.0), constrained_layout=True)
    for axis, case in zip(axes, ("box", "climb")):
        values = data.loc[data.case == case].set_index("variant").reindex(("full", "no-hard"))
        x = np.arange(2)
        width = 0.25
        for offset, column, label, color in (
            (-width, "strict_contact_2cm_frame_rate", "strict contact ≤2 cm", "#54A24B"),
            (0.0, "penetration_frame_rate", "penetration >1.1 mm", "#E45756"),
            (width, "foot_sticking_violation_frame_rate", "foot-stick violation", "#F58518"),
        ):
            axis.bar(x + offset, values[column], width, label=label, color=color)
        axis.set_xticks(x, ("Full", "No-Hard"))
        axis.set_ylim(0, 1.02)
        axis.set_ylabel("Frame rate")
        axis.set_title(f"{case.capitalize()} case")
    axes[1].legend(frameon=False, fontsize=7.5, loc="upper center")
    figure.suptitle("Two-case interaction ablation; no dataset-level generalization")
    return figure


def build_stage1_figures(
    repo_root: str | Path,
    evaluations: EvaluationBundle,
    timing: pd.DataFrame,
    sparse_pairs: pd.DataFrame,
    interaction: pd.DataFrame,
    scale: ScaleBundle,
    reference_comparison: pd.DataFrame,
) -> list[Path]:
    root = Path(repo_root).resolve()
    core_reference = pd.concat(
        [evaluations.core, evaluations.reference], ignore_index=True
    )
    timing_quality = evaluations.core.merge(
        timing[["key", "end_to_end_rtf_median", "timing_evidence_grade"]],
        on="key",
        validate="one_to_one",
    )
    temporal_columns = [
        "key",
        "display_name",
        "joint_jerk_rms_p95_rad_s3",
        "pose_jump_p95_m",
        "foot_skating_frame_rate",
        "ground_penetration_frame_rate",
        "joint_limit_violation_frame_rate",
        "invalid_frame_rate",
        "artifact_rate",
    ]
    charts = (
        (
            "quality_targeted_vs_untracked",
            core_reference[["key", "display_name", "rf_kpe_targeted_mean_m", "rf_kpe_untracked_mean_m"]],
            _annotated_scatter,
        ),
        (
            "timing_vs_quality",
            timing_quality[["key", "display_name", "end_to_end_rtf_median", "rf_kpe_all_mean_m", "timing_evidence_grade"]],
            _timing_quality_chart,
        ),
        (
            "root_translation_and_yaw",
            core_reference[["key", "display_name", "root_translation_common_scale_mean_m", "root_translation_scale_invariant_mean_m", "root_yaw_mean_rad"]],
            _root_yaw_chart,
        ),
        (
            "temporal_and_artifact_components",
            core_reference[temporal_columns],
            _temporal_artifact_chart,
        ),
        (
            "controlled_scale_policy_transplant",
            scale.transplants[["variant", "rf_kpe_all_mean_m", "root_translation_common_scale_mean_m", "artifact_rate"]].sort_values("variant"),
            _scale_transplant_chart,
        ),
        ("native_scale_response_elasticity", scale.slopes, _scale_slope_chart),
        ("scale_rank_stability", scale.ranks, _rank_stability_chart),
        (
            "holosoma_contact_policy_robustness",
            scale.contact_robustness,
            _contact_robustness_chart,
        ),
        ("unitree_reference_descriptive_delta", reference_comparison, _reference_delta_chart),
        (
            "unitree_reference_direct_trajectory",
            reference_comparison.loc[
                reference_comparison.comparison_type.astype(str).eq(
                    "direct_same_g1_frame_index_aligned_trajectory_disagreement"
                )
            ],
            _direct_reference_chart,
        ),
        ("sparse_seed_divergence", sparse_pairs, _sparse_seed_chart),
        (
            "interaction_full_vs_no_hard",
            interaction[["case", "variant", "strict_contact_2cm_frame_rate", "near_contact_5cm_frame_rate", "proximity_10cm_frame_rate", "penetration_frame_rate", "foot_sticking_violation_frame_rate"]],
            _interaction_chart,
        ),
    )
    outputs: list[Path] = []
    for name, source, renderer in charts:
        outputs.extend(save_publication_chart(root, name, source, renderer))
    return outputs


def _markdown_table(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    labels: Mapping[str, str] | None = None,
    digits: int = 4,
) -> str:
    selected = frame.loc[:, list(columns)].copy()
    for column in selected.select_dtypes(include=[np.number]).columns:
        selected[column] = selected[column].map(
            lambda value: f"{float(value):.{digits}f}" if pd.notna(value) else "N/A"
        )
    if labels:
        selected = selected.rename(columns=dict(labels))
    return selected.to_markdown(index=False)


def _write_stage1_markdown(path: Path, body: str) -> None:
    text = body.strip()
    if text.endswith(HISTORICAL_STAGE1_STOP):
        text = text[: -len(HISTORICAL_STAGE1_STOP)].rstrip()
    atomic_write_text(path, f"{text}\n\n{HISTORICAL_STAGE1_STOP}\n")


def presentation_slide_count(text: str) -> int:
    return len(re.findall(r"^## Slide\s+\d+\b", text, flags=re.MULTILINE))


def _display_scale_method(value: str) -> str:
    return {
        "gmr": "GMR",
        "omniretarget": "OmniRetarget / Holosoma",
        "protomotions_v2_3_mink": "ProtoMotions v2.3 Mink LAFAN port",
        "protomotions_v3": "ProtoMotions v3 modified-PyRoki LAFAN port",
    }.get(value, value)


def render_stage1_reports(
    repo_root: str | Path,
    evaluations: EvaluationBundle,
    timing: pd.DataFrame,
    sparse_pairs: pd.DataFrame,
    sparse_variance: pd.DataFrame,
    interaction: pd.DataFrame,
    scale: ScaleBundle,
    reference_comparison: pd.DataFrame,
    *,
    decision_resolution: Mapping[str, Any] | None = None,
) -> list[Path]:
    """Render the complete English Markdown evidence package.

    With no resolution this is the bootstrap ``PENDING`` report set.  A final
    GO/NO-GO rendering requires a resolution returned by
    :func:`resolve_bound_stage1_validation`; arbitrary caller-supplied verdicts
    are rejected.
    """

    root = Path(repo_root).resolve()
    core = evaluations.core.copy()
    reference = evaluations.reference.copy()
    all_quality = pd.concat([core, reference], ignore_index=True)
    core_timing = core.merge(timing, on=["key", "display_name"], validate="one_to_one")

    quality_table = _markdown_table(
        core,
        (
            "display_name",
            "rf_kpe_all_mean_m",
            "rf_kpe_targeted_mean_m",
            "rf_kpe_untracked_mean_m",
            "root_translation_common_scale_mean_m",
            "root_yaw_mean_rad",
            "artifact_rate",
        ),
        labels={
            "display_name": "Method",
            "rf_kpe_all_mean_m": "RF-KPE all (m)",
            "rf_kpe_targeted_mean_m": "Targeted (m)",
            "rf_kpe_untracked_mean_m": "Untracked (m)",
            "root_translation_common_scale_mean_m": "Root common (m)",
            "root_yaw_mean_rad": "Yaw (rad)",
            "artifact_rate": "Artifact rate",
        },
    )
    reference_table = _markdown_table(
        reference,
        (
            "display_name",
            "rf_kpe_all_mean_m",
            "rf_kpe_targeted_mean_m",
            "rf_kpe_untracked_mean_m",
            "root_translation_common_scale_mean_m",
            "root_translation_scale_invariant_mean_m",
            "root_yaw_mean_rad",
            "artifact_rate",
        ),
        labels={
            "display_name": "External reference",
            "rf_kpe_all_mean_m": "RF-KPE all (m)",
            "rf_kpe_targeted_mean_m": "Targeted (m)",
            "rf_kpe_untracked_mean_m": "Untracked (m)",
            "root_translation_common_scale_mean_m": "Root common (m)",
            "root_translation_scale_invariant_mean_m": "Root shape (m)",
            "root_yaw_mean_rad": "Yaw (rad)",
            "artifact_rate": "Artifact rate",
        },
    )
    timing_table = _markdown_table(
        core_timing,
        (
            "display_name",
            "end_to_end_rtf_median",
            "native_core_rtf_median",
            "measured_repetitions",
            "timing_evidence_grade",
        ),
        labels={
            "display_name": "Method",
            "end_to_end_rtf_median": "End-to-end RTF",
            "native_core_rtf_median": "Native-core RTF",
            "measured_repetitions": "Measured repeats",
            "timing_evidence_grade": "Evidence grade",
        },
    )
    temporal_table = _markdown_table(
        all_quality,
        (
            "display_name",
            "joint_velocity_rms_mean_rad_s",
            "joint_acceleration_rms_mean_rad_s2",
            "joint_jerk_rms_p95_rad_s3",
            "pose_jump_p95_m",
            "foot_skating_frame_rate",
            "ground_penetration_frame_rate",
            "joint_limit_violation_frame_rate",
        ),
        labels={
            "display_name": "Method/reference",
            "joint_velocity_rms_mean_rad_s": "Velocity RMS",
            "joint_acceleration_rms_mean_rad_s2": "Acceleration RMS",
            "joint_jerk_rms_p95_rad_s3": "Jerk p95",
            "pose_jump_p95_m": "Pose jump p95 (m)",
            "foot_skating_frame_rate": "Skating",
            "ground_penetration_frame_rate": "Penetration",
            "joint_limit_violation_frame_rate": "Joint limit",
        },
    )
    sparse_table = _markdown_table(
        sparse_pairs,
        (
            "pair",
            "joint_angle_rms_mean_rad",
            "joint_angle_rms_p95_rad",
            "robot_rf_point_rms_mean_m",
            "robot_rf_point_rms_p95_m",
        ),
        labels={
            "pair": "Seed pair",
            "joint_angle_rms_mean_rad": "Joint RMS mean (rad)",
            "joint_angle_rms_p95_rad": "Joint RMS p95 (rad)",
            "robot_rf_point_rms_mean_m": "RF point RMS mean (m)",
            "robot_rf_point_rms_p95_m": "RF point RMS p95 (m)",
        },
    )
    interaction_table = _markdown_table(
        interaction,
        (
            "case",
            "variant",
            "strict_contact_2cm_frame_rate",
            "near_contact_5cm_frame_rate",
            "penetration_frame_rate",
            "foot_sticking_violation_frame_rate",
            "end_to_end_rtf",
        ),
        labels={
            "case": "Case",
            "variant": "Variant",
            "strict_contact_2cm_frame_rate": "Contact ≤2 cm",
            "near_contact_5cm_frame_rate": "Contact ≤5 cm",
            "penetration_frame_rate": "Penetration",
            "foot_sticking_violation_frame_rate": "Foot-stick violation",
            "end_to_end_rtf": "RTF",
        },
    )
    transplant_table = _markdown_table(
        scale.transplants.sort_values("variant"),
        (
            "variant",
            "effective_root_xy_scale",
            "rf_kpe_all_mean_m",
            "root_translation_common_scale_mean_m",
            "artifact_rate",
        ),
        labels={
            "variant": "Scale formula in same Dense solver",
            "effective_root_xy_scale": "Effective XY scale",
            "rf_kpe_all_mean_m": "RF-KPE all (m)",
            "root_translation_common_scale_mean_m": "Root common (m)",
            "artifact_rate": "Artifact rate",
        },
    )
    slope_focus = scale.slopes.loc[
        scale.slopes.metric.isin(
            ("rf_kpe_all_mean_m", "root_translation_common_scale_mean_m")
        )
    ].copy()
    slope_focus["method"] = slope_focus.method.map(_display_scale_method)
    slope_table = _markdown_table(
        slope_focus,
        (
            "method",
            "factor",
            "metric",
            "native_value",
            "central_difference_per_unit_multiplier",
            "elasticity_at_native",
        ),
        labels={
            "method": "Method",
            "factor": "Perturbed target",
            "metric": "Metric",
            "native_value": "Native",
            "central_difference_per_unit_multiplier": "Central slope",
            "elasticity_at_native": "Elasticity",
        },
        digits=3,
    )
    rank_summary = (
        scale.ranks.groupby(["metric", "variant"], as_index=False)
        .agg(
            spearman_vs_native=("spearman_vs_native", "first"),
            changed_methods=("rank_changed", "sum"),
        )
    )
    rank_table = _markdown_table(
        rank_summary,
        ("metric", "variant", "spearman_vs_native", "changed_methods"),
        labels={
            "metric": "Metric",
            "variant": "Variant",
            "spearman_vs_native": "Spearman vs native",
            "changed_methods": "Changed method ranks",
        },
        digits=3,
    )
    contact_robustness_table = _markdown_table(
        scale.contact_robustness,
        (
            "variant",
            "rf_kpe_all_mean_m_fixed_canonical",
            "rf_kpe_all_mean_m_native_recomputed",
            "root_translation_common_scale_mean_m_fixed_canonical",
            "root_translation_common_scale_mean_m_native_recomputed",
            "artifact_rate_fixed_canonical",
            "artifact_rate_native_recomputed",
        ),
        labels={
            "variant": "Scale variant",
            "rf_kpe_all_mean_m_fixed_canonical": "RF-KPE fixed",
            "rf_kpe_all_mean_m_native_recomputed": "RF-KPE recomputed",
            "root_translation_common_scale_mean_m_fixed_canonical": "Root fixed",
            "root_translation_common_scale_mean_m_native_recomputed": "Root recomputed",
            "artifact_rate_fixed_canonical": "Artifact fixed",
            "artifact_rate_native_recomputed": "Artifact recomputed",
        },
    )
    comparison_focus = reference_comparison.loc[
        reference_comparison.metric.isin(
            (
                "rf_kpe_all_mean_m",
                "root_translation_common_scale_mean_m",
                "root_translation_scale_invariant_mean_m",
                "root_yaw_mean_rad",
                "artifact_rate",
            )
        )
    ]
    comparison_table = _markdown_table(
        comparison_focus,
        (
            "display_name",
            "metric",
            "method_value",
            "reference_value",
            "signed_method_minus_reference",
        ),
        labels={
            "display_name": "Method",
            "metric": "Metric",
            "method_value": "Method",
            "reference_value": "External reference",
            "signed_method_minus_reference": "Method − reference",
        },
    )
    direct_focus = reference_comparison.loc[
        reference_comparison.metric.isin(
            (
                "direct_root_translation_mean_m",
                "direct_root_anchor_frame0_m",
                "direct_root_displacement_mean_m",
                "direct_root_yaw_mean_rad",
                "direct_joint_angle_rmse_rad",
                "direct_joint_velocity_rmse_rad_s",
                "direct_fk_world_semantic_mean_m",
                "direct_fk_root_frame_semantic_mean_m",
            )
        )
    ]
    direct_reference_table = _markdown_table(
        direct_focus,
        ("display_name", "metric", "method_value"),
        labels={
            "display_name": "Method",
            "metric": "Direct disagreement metric",
            "method_value": "Value",
        },
    )

    best_all = core.loc[core.rf_kpe_all_mean_m.idxmin()]
    best_targeted = core.loc[core.rf_kpe_targeted_mean_m.idxmin()]
    best_untracked = core.loc[core.rf_kpe_untracked_mean_m.idxmin()]
    best_root = core.loc[core.root_translation_common_scale_mean_m.idxmin()]
    best_yaw = core.loc[core.root_yaw_mean_rad.idxmin()]
    fastest = core_timing.loc[core_timing.end_to_end_rtf_median.idxmin()]
    formal_count = int((timing.timing_evidence_grade == "formal_repeated").sum())
    root_transplant_span = float(
        scale.transplants.root_translation_common_scale_mean_m.max()
        - scale.transplants.root_translation_common_scale_mean_m.min()
    )
    max_seed_frame = int(sparse_variance.all_point_variance_m2.idxmax())
    max_seed_variance = float(sparse_variance.all_point_variance_m2.max())
    quality_scale_slopes = scale.slopes.loc[scale.slopes.rank_eligible.astype(bool)]
    highest_scale_elasticity = quality_scale_slopes.loc[
        quality_scale_slopes.elasticity_at_native.abs().idxmax()
    ]
    native_contact_row = scale.contact_diagnostics.loc[
        scale.contact_diagnostics.variant.astype(str).eq("native")
    ].iloc[0]
    native_contact_disagreement = float(
        native_contact_row.native_recomputed_vs_formal_canonical_label_disagreement_rate
    )
    scale_induced_contact_flip_max = float(
        scale.contact_diagnostics.recomputed_label_flip_rate.max()
    )

    if decision_resolution is None:
        decision = "PENDING"
        validation_note = (
            "Independent fail-closed validation has not yet bound a verdict to "
            "this evidence ledger."
        )
        decision_action = (
            "Do not treat this publication build as Stage 1 acceptance or as "
            "authority to proceed."
        )
        failed_check_block = "Validation checks are not available until the independent pass runs."
    else:
        if decision_resolution.get("binding_status") != "verified":
            raise ValueError("Final reports require a verified validation binding")
        decision = str(decision_resolution.get("decision", ""))
        if decision not in {"GO", "NO-GO"}:
            raise ValueError(f"Unsupported independently validated decision: {decision!r}")
        validation_hash = str(decision_resolution.get("validation_sha256", ""))
        validation_note = (
            "This verdict is issued by `manifests/stage1_validation.json` and is "
            f"bound to the current publication evidence ledger (validation SHA-256 "
            f"`{validation_hash}`)."
        )
        decision_action = (
            "The independently validated evidence gate permits budget-gated Stage 2 "
            "design/execution; this is not a universal method endorsement."
            if decision == "GO"
            else "One or more fail-closed acceptance checks failed; Stage 1 is not accepted."
        )
        failed_checks = list(
            decision_resolution.get("validation", {}).get("failed_checks", [])
        )
        failed_check_block = (
            "All independently evaluated acceptance checks passed."
            if not failed_checks
            else "Failed checks:\n\n"
            + "\n".join(f"- `{name}`" for name in failed_checks)
        )
    decision_sentence = f"Decision: **{decision}**. {validation_note} {decision_action}"
    decision_audit = (
        f"The authoritative Stage 1 decision is **{decision}**. {validation_note} "
        f"{decision_action} The review does not endorse a universal winner and "
        "does not retroactively alter the frozen Pilot, thresholds, or scale formulas."
    )

    pilot_report = f"""# Human-to-G1 Retargeting: Expanded Stage 1 Pilot Report

## Executive finding

The publication builder found six required operating points covering the frozen 600-frame, 19.9998-second `dance1_subject1` Pilot and evaluated them with one canonical Holosoma G1-29 model and evaluator-v3 protocol. This evidence inventory is not itself the Stage 1 acceptance decision. Evaluator v3 uses the registered shared-semantic-landmark least-squares local/body scale, separately declares root-displacement gain and root anchor, and retains head/toe span as a diagnostic only. Sparse seeds A/B remain diagnostics rather than extra methods. The Unitree-attributed corpus is an external, untimed comparison reference; its hosting provenance does **not** verify that it is official, ground truth, or a quality upper bound.

On this single Pilot, `{best_all.display_name}` has the lowest RF-KPE-all ({best_all.rf_kpe_all_mean_m:.4f} m), `{best_targeted.display_name}` the lowest targeted RF-KPE ({best_targeted.rf_kpe_targeted_mean_m:.4f} m), `{best_untracked.display_name}` the lowest untracked RF-KPE ({best_untracked.rf_kpe_untracked_mean_m:.4f} m), `{best_root.display_name}` the lowest common-scale root-path error ({best_root.root_translation_common_scale_mean_m:.4f} m), and `{best_yaw.display_name}` the lowest yaw error ({best_yaw.root_yaw_mean_rad:.4f} rad). These are metric-specific observations, not a universal method ranking.

## Frozen design and method identity

- Sparse Mink (neutral) and Dense Mink share solver, G1, limits, scale, initialization policy, regularization, and warm start; only their declared task set differs.
- GMR and OmniRetarget/Holosoma use their frozen public LAFAN paths with I/O, provenance, and timing adapters.
- ProtoMotions v2.3 is the **Mink LAFAN port**, not PHC. Its public v2.3 retargeter infrastructure is preserved while the canonical LAFAN package supplies semantic targets.
- ProtoMotions v3 is the **canonical LAFAN port of the published generic modified-PyRoki retargeter**, not an official/native LAFAN entry point and not a bare PyRoki demonstration.
- All quality conclusions are limited to one preselected Pilot.

## Unified quality results

{quality_table}

RF-KPE is root-frame keypoint position error: both human and robot landmarks are expressed relative to their own root and heading, then the human morphology is mapped with the one frozen common scale. Consequently RF-KPE intentionally removes global translation and most heading differences. Targeted RF-KPE covers wrists and ankles; untracked RF-KPE covers the remaining semantic body landmarks. Root translation and yaw must therefore be read in their dedicated columns.

![Targeted versus untracked](figures/stage1_publication/quality_targeted_vs_untracked.svg)

## Why the motions can look similar

All methods animate the same G1 morphology, and a root-frame overlay deliberately removes two visually dominant differences: global path and heading. Several solvers also satisfy similar hand/foot objectives on this sequence. Their separation appears in different channels and moments: target-vs-untracked trade-offs, root scale and path shape, yaw, high-frequency temporal behavior, stance artifacts, and Sparse null-space choice. A synchronized articulated-G1 view is therefore necessary, but it cannot replace the decomposed metrics.

## Root scale, path shape, and yaw

Common-scale root error compares every method to the same source-to-G1 scale. Native-scale error (preserved in the CSV) asks whether a method followed its own declared target scale. Scale-invariant root error refits one scalar only for diagnosis; it does not excuse a scale mismatch. This decomposition explains why two methods can have similar RF-KPE yet very different root translation error.

![Root and yaw](figures/stage1_publication/root_translation_and_yaw.svg)

## Temporal and artifact evidence

{temporal_table}

`artifact_rate` is a per-frame union of named causes, never a method label. Component rates may overlap, so their stack is diagnostic rather than an additive score. Self-collision remains outside the primary metric because a common validated pair set was not frozen.

![Temporal and artifact components](figures/stage1_publication/temporal_and_artifact_components.svg)

## Timing

{timing_table}

The external reference has no reproducible runtime and is excluded from every RTF plot. `{fastest.display_name}` is fastest on this Pilot at {fastest.end_to_end_rtf_median:.4f} end-to-end RTF. {formal_count}/6 methods have the full repeated timing grade; any summary-only or metadata-derived row remains explicitly marked and should not support fine-grained speed claims.

![Timing versus quality](figures/stage1_publication/timing_vs_quality.svg)

## Scale-policy experiments

Applying the five frozen scale formulas to the same Dense Mink solver changes common-scale root error across a {root_transplant_span:.4f} m span. Thus scale preprocessing alone can manufacture a substantial part of the apparent method gap even when source sequence, solver, and target robot are fixed.

{transplant_table}

The native ±5% experiment perturbs root-path scale and root-relative local scale before each method's own solver. Across the pre-registered quality subset, the largest absolute elasticity is `{highest_scale_elasticity.method}` / `{highest_scale_elasticity.factor}` / `{highest_scale_elasticity.metric}` at {highest_scale_elasticity.elasticity_at_native:.3g}. The complete response registry also retains non-ranking trajectory-divergence and single-run timing diagnostics. These are local sensitivities around published policies, not recommendations to retune a method.

The formal causal arm freezes canonical contact labels. Holosoma's native recomputed labels disagree with those labels on {native_contact_disagreement:.2%} of entries at the native setting, while changing scale causes at most {scale_induced_contact_flip_max:.2%} additional native-label flips. A separate recomputed-contact robustness arm is reported rather than mixed into formal slopes or method ranks.

![Scale transplant](figures/stage1_publication/controlled_scale_policy_transplant.svg)

## External reference comparison

{reference_table}

Coordinate normalization is a fixed world-frame rotation only; it does not rescale, translate, ground-align, or resample the reference. Descriptive deltas are reported because the external corpus documents neither comparable timing nor enough method provenance to justify a ground-truth or upper-bound claim.

![External reference deltas](figures/stage1_publication/unitree_reference_descriptive_delta.svg)

The direct same-G1 comparison below is stronger than subtracting evaluator summaries: it compares root, joints, velocities, and canonical MuJoCo FK frame by frame. Zero means identical trajectories; it still does not mean that the external trajectory is truth.

{direct_reference_table}

![Direct external-trajectory disagreement](figures/stage1_publication/unitree_reference_direct_trajectory.svg)

## Sparse initialization sensitivity

{sparse_table}

The maximum three-seed all-point variance is {max_seed_variance:.6g} m² at frame {max_seed_frame}. Seed neutral is the operating point; A/B quantify underconstraint and are never averaged into a method score.

## Interaction case study

{interaction_table}

The Full/No-Hard comparisons use actual collision-surface distance, retain joint limits, and change only object non-penetration and foot-sticking flags. This is evidence for exactly one box and one climb sequence, not for LAFAN or interactions generally.

## Conclusion and limits

The evidence package contains six complete method outputs, one clearly delimited external reference, controlled and native scale tests, sparse-seed diagnostics, timing provenance, and two interaction ablations. {decision_sentence} It does not establish dataset-level ranking, dynamic feasibility, controller robustness, statistical uncertainty across actors/actions, or that proximity to the external reference means higher quality. Those require Stage 2 or a separate dynamics study. Stage 2 uses method-first budget simplification: a method, including v3, may be excluded if its measured projection cannot satisfy the frozen runtime/storage gate; that does not erase its completed Stage 1 evidence. The final marker below is retained as the historical Stage 1 boundary snapshot.
"""

    executive_summary = f"""# Executive Summary

The evidence inventory contains one frozen 20-second Pilot for Sparse-neutral, Dense, GMR, OmniRetarget/Holosoma, ProtoMotions v2.3 Mink LAFAN port, and the ProtoMotions v3 modified-PyRoki LAFAN port, each with 600 valid G1-29 frames. This inventory does not self-certify Stage 1 acceptance. Sparse A/B are diagnostics. The Unitree-attributed corpus is an untimed external reference, not verified official data, ground truth, or an upper bound.

No single winner is supported. `{best_all.display_name}` minimizes RF-KPE-all, `{best_root.display_name}` minimizes common-scale root translation error, `{best_yaw.display_name}` minimizes yaw error, and `{fastest.display_name}` is fastest on this Pilot. The apparent visual similarity is expected because all outputs share G1 morphology and root-frame inspection removes global trajectory; the decomposed metrics expose scale, path, temporal, artifact, and null-space differences.

The strongest design result is causal: transplanting official scale formulas into the same Dense solver changes root error by {root_transplant_span:.4f} m. Published preprocessing is therefore part of the method and must be reported, while the common-scale evaluator remains necessary for cross-method comparison. {decision_sentence}
"""

    go_no_go = f"""# Stage 1 Decision: {decision}

## Publication evidence inventory (not the acceptance authority)

| Requirement | Result |
|---|---|
| Six required canonical G1-29 operating points | PASS: 6 × 600 valid frames |
| Sparse neutral plus deterministic A/B diagnostics | PASS |
| Unified targeted, untracked, root, yaw, temporal, and artifact metrics | PASS |
| Unitree-attributed external reference, excluded from timing | PASS |
| Five same-solver scale-policy transplants | PASS |
| Four methods × fixed-contact native/root±5%/local±5% response | PASS |
| Holosoma native-recomputed-contact robustness arm | PASS, kept separate |
| Sparse rank/seed and two-case interaction evidence | PASS |
| Source CSV + SVG + PDF + 300-dpi PNG for every main chart | PASS |

{decision_sentence}

{failed_check_block}

This decision does not mean any method is best in general, that the external corpus is ground truth, or that kinematic quality predicts controller success. {formal_count}/6 timing rows have formal repeated evidence; lower-grade rows remain usable only as explicitly labelled engineering estimates. Stage 2 has its own method-first budget gate, so v3 may be omitted there if its projection is infeasible without changing or discarding its Stage 1 evidence.
"""

    method_scope = """# Method Scope

## Primary operating points

| Label | Role | What the result is not |
|---|---|---|
| Sparse Mink (neutral) | Controlled sparse root/wrist/ankle IK | Seeds A/B are not additional methods |
| Dense Mink | Same controlled solver with added body targets | Not a learned prior |
| GMR | Frozen public LAFAN→G1 pipeline | Not the bare Mink backend |
| OmniRetarget / Holosoma | Frozen public LAFAN→G1 pipeline | Not labelled `Artifact`; artifacts are per-frame causes |
| ProtoMotions v2.3 Mink LAFAN port | Canonical LAFAN target adapter around v2.3 Mink retargeting | Not PHC and not an official LAFAN entry point |
| ProtoMotions v3 modified-PyRoki LAFAN port | Canonical LAFAN adapter around the published generic whole-trajectory solver | Not an official/native LAFAN entry point and not bare PyRoki |

## External reference

The `lvhaidong/LAFAN1_Retargeting_Dataset` G1 trajectory is a **Unitree-attributed reference corpus**. It is precomputed, untimed, and compared after a fixed coordinate-frame rotation with no scale, translation, ground, or time correction. Repository copyright attribution and a personal hosting account do not by themselves verify official publication, method identity, ground truth, or an upper bound.

## Claim boundary

All numerical comparisons are for the preselected `dance1_subject1`, frames 0–599. Interaction results cover exactly two separate case studies. Backends, controllers, dynamics, and policy-tracking benchmarks are not silently mixed into the retargeter scatter.
"""

    sparse_report = f"""# Sparse IK Analysis

Sparse-neutral tracks root translation/yaw plus both wrists and ankles. Dense adds head, torso, shoulders, elbows, hips, knees, ankles, and toes while keeping the same G1 model, common scale, solver family, limits, posture/temporal regularization, iteration budget, and sequential warm start. This makes the Sparse/Dense contrast a task-density experiment rather than a change of backend or robot.

{sparse_table}

Across all three deterministic seeds, targeted point variance averages {sparse_variance.targeted_point_variance_m2.mean():.6g} m² and untracked point variance averages {sparse_variance.untracked_point_variance_m2.mean():.6g} m². The maximum all-point variance occurs at frame {max_seed_frame}. Similar hand/foot satisfaction can therefore coexist with distinct untracked full-body solutions; neutral alone is the primary operating point and A/B only diagnose that null space.

![Sparse seed divergence](figures/stage1_publication/sparse_seed_divergence.svg)
"""

    interaction_report = f"""# Interaction Case Study

{interaction_table}

Full enables object non-penetration, foot sticking, and joint limits. No-Hard disables only object non-penetration and foot sticking; the input, initialization, solver, iteration budget, and joint limits remain fixed. Strict contact (≤2 cm), near contact (≤5 cm), proximity (≤10 cm), and penetration use `mujoco.mj_geomDistance` over collision surfaces rather than object-origin distance. Penetration uses the frozen 1.1 mm tolerance-aware threshold.

The box and climb results demonstrate constraint effects in these two cases only. They do not estimate dataset-level interaction quality or physical executability.

![Interaction ablation](figures/stage1_publication/interaction_full_vs_no_hard.svg)
"""

    reproduce = """# Reproduce the Expanded Stage 1 Publication

1. Restore the frozen upstream worktrees, model assets, LAFAN source, canonical Pilot, six method outputs, three Sparse seeds, interaction artifacts, and scale-sensitivity outputs recorded by their manifests and hashes.
2. Install the project with the `evaluation` extra in an environment containing MuJoCo, pandas, PyArrow, SciPy, Matplotlib, and tabulate.
3. From the repository root run:

```bash
python -c "from retargeting_comparison.stage1_publication import build_stage1_publication; build_stage1_publication('.')"
```

4. Confirm `metrics/stage1_core_summary.csv` has six rows, `metrics/stage1_reference_summary.csv` has one untimed row, and every output covers source frames 0–599 exactly.
5. Confirm the formal fixed-contact scale matrix contains four methods × five variants, the Holosoma native-contact robustness arm contains five variants, and the controlled transplant matrix contains all five frozen formulas.
6. Inspect `figures/stage1_publication/source_data/`; every figure has one source CSV plus SVG, PDF, and 300-dpi PNG siblings.
7. Read `manifests/stage1_publication.json` for the PENDING evidence ledger and `manifests/stage1_validation.json` for the independently bound decision. The builder evaluates and reports existing outputs; it never launches an experiment.
"""

    scale_report = f"""# Official Scale-Policy Sensitivity

## Why equal source and robot do not imply equal target coordinates

Every method receives the same frozen human motion and targets the same G1-29 robot, but its published preprocessing defines a different mapping from human world/root-relative coordinates to robot targets. Scale is therefore an algorithmic input policy, not a property of the final robot. A common evaluator scale answers cross-method morphology fidelity; native-scale tracking answers whether a solver followed its own policy.

## Same-solver policy transplant

{transplant_table}

Only the target-coordinate formula changes; Dense Mink, G1, source, initialization, and evaluator remain fixed. The {root_transplant_span:.4f} m root-error span demonstrates that preprocessing alone can explain a large apparent gap. These transplant rows are controlled ablations, not upstream-method results.

## Native ±5% response

{slope_table}

Root-path and root-relative local targets are perturbed before each solver. Qpos post-scaling is forbidden. Central slopes divide the +5%−−5% response by 0.10; elasticity normalizes that slope by the native metric. Large artifact elasticities near a threshold must be interpreted as discontinuous label sensitivity, not smooth physical response.

The displayed table is a two-metric reading aid. `metrics/stage1_scale_sensitivity_slopes.csv` contains all {len(SCALE_RESPONSE_METRIC_NAMES)} pre-registered response metrics, with metric family, preferred direction, raw ±5% values and deltas, centered slope, elasticity-defined status, and rank eligibility. No composite score is constructed.

![Native sensitivity](figures/stage1_publication/native_scale_response_elasticity.svg)

## Holosoma contact-policy robustness

The formal scale-causality arm holds canonical contact labels fixed. Native Holosoma contact extraction disagrees with those labels on {native_contact_disagreement:.2%} of entries at native scale, whereas the largest scale-induced recomputation flip is {scale_induced_contact_flip_max:.2%}. Contact-definition choice is therefore a separate preprocessing factor and is not folded into a scale slope.

{contact_robustness_table}

![Contact robustness](figures/stage1_publication/holosoma_contact_policy_robustness.svg)

## Rank stability

{rank_table}

![Rank stability](figures/stage1_publication/scale_rank_stability.svg)

Only the pre-registered quality subset is ranked, using the explicit minimize/maximize direction stored in every row; timing and native-trajectory divergence are excluded. Rank correlations describe only four methods, one sequence, and local ±5% interventions. They do not form a composite score and do not justify selecting a globally optimal scale or retuning upstream methods after seeing results.
"""

    reference_report = f"""# Unitree-Attributed Reference Comparison

## Provenance status

The Hugging Face corpus is attributed to Unitree in its repository metadata, but the available hosting and discussion evidence does not verify an official release chain or disclose enough method detail to call it ground truth. The scientific label is therefore **Unitree-attributed external reference corpus**. It is untimed and excluded from RTF comparisons.

## Coordinate contract

The published root is viewed through one predeclared active world rotation `Rz(-π/2)` and quaternion conversion `xyzw→wxyz`. No fitted scale, translation, ground alignment, or resampling is applied. This avoids manufacturing root/yaw error from a known coordinate convention while preserving the trajectory and pose.

{reference_table}

## Descriptive differences

{comparison_table}

![Descriptive deltas](figures/stage1_publication/unitree_reference_descriptive_delta.svg)

## Direct same-G1 trajectory disagreement

{direct_reference_table}

These values compare the method qpos and the external qpos on identical source-frame indices and the same canonical MuJoCo G1: root translation and orientation, wrapped joint angles and velocities, world-frame FK, and root-frame FK. This is actual trajectory resemblance rather than a difference between two evaluator summaries. Lower still means only closer to the external trajectory.

![Direct trajectory disagreement](figures/stage1_publication/unitree_reference_direct_trajectory.svg)

A smaller absolute delta means only that a method is numerically closer to this external trajectory under a named metric. It does not establish correctness or superior retargeting quality. The reference itself is evaluated for temporal and artifact behavior rather than presumed clean.
"""

    review = f"""# Comprehensive Stage 1 Review

## Design review

The design now contains six actual retargeter operating points rather than backend or lineage placeholders. Sparse/Dense isolate task density. GMR and Holosoma retain their public LAFAN paths. ProtoMotions v2.3 and v3 are explicitly identified as canonical LAFAN ports rather than official/native LAFAN entry points. The external reference is separated from methods and timing. The common evaluator reports targeted, untracked, root, yaw, temporal, and named artifact channels without collapsing them into a composite score.

## Execution review

All nine required canonical trajectories (six core, two additional Sparse seeds, one external reference) contain exactly 600 finite valid G1-29 frames with source indices 0–599. The scale evidence contains five controlled transplants, four native methods × five fixed-contact variants, and a separate five-variant Holosoma native-contact robustness arm. Both interaction cases have Full and No-Hard results with surface distances and verified flag differences. Every main chart has inspectable source data and three render formats.

## Analysis review

The report avoids three earlier category errors: RF-KPE is not root tracking; an artifact is a per-frame union of named causes rather than a method label; and closeness to the external corpus is not accuracy against ground truth. It also separates native published preprocessing from controlled scale transplants. The method-specific minima in the main report are descriptive and deliberately not compressed into a winner.

## Threats to validity

- **External validity:** one 20-second dance sequence cannot represent LAFAN, AMASS, actors, or interaction tasks.
- **Construct validity:** kinematic keypoint and artifact metrics do not prove dynamics, torque feasibility, balance, or policy trackability.
- **Reference validity:** external provenance and method details are incomplete; no upper-bound claim is allowed.
- **Timing validity:** only {formal_count}/6 methods currently have the highest repeated timing grade; other grades must remain visible.
- **Scale locality:** ±5% central differences probe a local neighborhood and thresholded artifacts can be non-smooth.
- **Contact policy:** Holosoma native and formal canonical labels disagree by {native_contact_disagreement:.2%}; fixed-contact causality and native-contact robustness remain separate.
- **Model transfer:** upstream and canonical G1 assets can differ slightly; evaluator transfer error is a measurement floor, not retargeter error.
- **Interaction scope:** two cases support a case study only.

## Decision audit

{decision_audit}
"""

    presentation = f"""# Human-to-G1 Retargeting — Stage 1 Evidence Deck

## Slide 1 — Question and frozen Pilot

Can sparse IK match denser/public retargeters, and how much of the apparent gap comes from scale preprocessing? One preselected 600-frame, 20-second LAFAN1 Pilot; no result-driven sequence choice.

## Slide 2 — Required method set

Sparse-neutral, Dense, GMR, OmniRetarget/Holosoma, ProtoMotions v2.3 Mink LAFAN port, and ProtoMotions v3 modified-PyRoki LAFAN port. Sparse A/B are diagnostics, not extra methods.

## Slide 3 — External reference boundary

The Unitree-attributed corpus is precomputed and untimed. It is not verified official, ground truth, or an upper bound; it receives only a fixed coordinate-view rotation.

## Slide 4 — What RF-KPE measures

Root-frame keypoint error removes global translation and heading. Targeted wrists/ankles and untracked body are separated; root translation and yaw are separate metrics.

## Slide 5 — Quality landscape

![Quality](figures/stage1_publication/quality_targeted_vs_untracked.svg)

Metric-specific minima differ; there is no one-Pilot universal winner.

## Slide 6 — Why outputs look similar

The same G1 morphology plus root-frame overlays hide path and heading. Differences emerge in scale, untracked pose, temporal spikes, stance artifacts, and Sparse null-space choices.

## Slide 7 — Root and yaw

![Root/yaw](figures/stage1_publication/root_translation_and_yaw.svg)

Common-scale, native-scale, and scale-invariant root errors answer different questions.

## Slide 8 — Timing versus quality

![Timing](figures/stage1_publication/timing_vs_quality.svg)

The external reference is excluded; timing evidence grades remain visible.

## Slide 9 — Scale policy is causal

![Transplant](figures/stage1_publication/controlled_scale_policy_transplant.svg)

The same Dense solver changes root error by {root_transplant_span:.4f} m across published formulas.

## Slide 10 — Native scale sensitivity

![Sensitivity](figures/stage1_publication/native_scale_response_elasticity.svg)

Root/local ±5% targets are changed before each solver; no qpos post-scaling.

Formal slopes freeze canonical contacts. Holosoma native-recomputed contacts form a separate robustness arm because their native label definition disagrees by {native_contact_disagreement:.2%}.

## Slide 11 — External reference comparison

![Reference deltas](figures/stage1_publication/unitree_reference_descriptive_delta.svg)

![Direct same-G1 disagreement](figures/stage1_publication/unitree_reference_direct_trajectory.svg)

Evaluator-summary deltas and direct qpos/FK disagreement are descriptive, never accuracy-to-truth.

## Slide 12 — Sparse null space

![Sparse seeds](figures/stage1_publication/sparse_seed_divergence.svg)

Neutral is primary; A/B expose hidden full-body sensitivity.

## Slide 13 — Interaction ablation

![Interaction](figures/stage1_publication/interaction_full_vs_no_hard.svg)

One box and one climb case; Full/No-Hard differences are not dataset evidence.

## Slide 14 — Decision and limits

The evidence contains six complete methods, one delimited external reference, scale causal tests, decomposed metrics, interaction, and reproducible figures. Authoritative decision: **{decision}**. {decision_action} No general ranking until multi-sequence evidence exists.
"""
    if presentation_slide_count(presentation) > 15:
        raise AssertionError("PRESENTATION.md exceeds 15 slide-level sections")

    bodies = {
        "PILOT_REPORT.md": pilot_report,
        "EXECUTIVE_SUMMARY.md": executive_summary,
        "GO_NO_GO.md": go_no_go,
        "METHOD_SCOPE.md": method_scope,
        "SPARSE_IK_ANALYSIS.md": sparse_report,
        "INTERACTION_CASE_STUDY.md": interaction_report,
        "REPRODUCE_PILOT.md": reproduce,
        "PRESENTATION.md": presentation,
        "SCALE_POLICY_SENSITIVITY.md": scale_report,
        "UNITREE_REFERENCE_COMPARISON.md": reference_report,
        "STAGE1_REVIEW.md": review,
    }
    outputs = []
    for name in REPORT_FILES:
        path = root / name
        _write_stage1_markdown(path, bodies[name])
        outputs.append(path)
    return outputs


def _direct_publication_input_paths(root: Path) -> list[Path]:
    """Return the direct, non-derived inputs consumed by the publication build."""

    sequence = _sequence(root)
    paths = method_output_paths(root)
    inputs = [
        root / "manifests" / "pilot_sequence.yaml",
        root / "manifests" / "evaluator.yaml",
        root / str(sequence["canonical_path"]),
        *paths.values(),
        root / "metrics" / "scale_policy_sensitivity_summary.csv",
        root / "metrics" / "contact_constraint_flip.csv",
        root / SCALE_INPUT_LEDGER,
    ]
    scale_table = pd.read_csv(root / "metrics" / "scale_policy_sensitivity_summary.csv")
    if "output_path" not in scale_table:
        raise ValueError("Scale-policy input table lacks output_path")
    inputs.extend(root / str(path) for path in scale_table.output_path.astype(str))
    scale_ledger = pd.read_csv(root / SCALE_INPUT_LEDGER, keep_default_na=False)
    for column in (
        "raw_per_frame_csv_path",
        "raw_per_frame_parquet_path",
        "raw_evaluator_summary_path",
        "target_manifest_path",
        "registered_target_path",
        "runtime_evidence_path",
        "runtime_input_path",
        "formal_native_binding_path",
        "accepted_formal_manifest_path",
        "accepted_formal_output_path",
        "accepted_formal_evidence_path",
        "config_path",
        "method_code_path",
        "environment_lock_path",
        "source_path",
        "evaluator_protocol_path",
        "evaluator_code_path",
        "scale_protocol_path",
        "provenance_table_path",
        "solver_asset_path",
        "evaluator_asset_path",
    ):
        if column in scale_ledger:
            inputs.extend(
                root / value
                for value in scale_ledger[column].astype(str)
                if value
            )
    sequence_id = str(sequence["sequence_id"])
    for spec in CORE_METHODS:
        run_dir = paths[spec.key].parent
        timing = next(
            (
                candidate
                for candidate in (run_dir / "timing_refined.json", run_dir / "timing.json")
                if candidate.is_file()
            ),
            None,
        )
        if timing is not None:
            inputs.append(timing)
        for candidate in (
            root / "runs" / "manifests" / f"{sequence_id}__{spec.run_directory}.json",
            root / "manifests" / "runs" / f"{sequence_id}__{spec.run_directory}.json",
        ):
            if candidate.is_file():
                inputs.append(candidate)
                break
    for case in ("box", "climb"):
        for variant in ("full", "no-hard"):
            manifest_path = (
                root
                / "runs"
                / "interaction_manifests"
                / f"interaction__{case}__{variant}.json"
            )
            inputs.append(manifest_path)
            manifest = RunManifest.load(manifest_path)
            summary = Path(str(manifest.output_path))
            if not summary.is_absolute():
                summary = root / summary
            inputs.extend(
                (
                    summary,
                    summary.parent / "per_frame_metrics.csv",
                    summary.parent / "intended_contact_per_semantic.csv",
                    summary.parent / "intended_contact_mapping.json",
                    summary.parent / "intended_contact_source_contract.npz",
                )
            )
    return inputs


def _load_final_report_inputs(
    root: Path,
) -> tuple[
    EvaluationBundle,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    ScaleBundle,
    pd.DataFrame,
]:
    """Load already-validated derived tables without rebuilding evidence."""

    evaluations = EvaluationBundle(
        core=pd.read_csv(root / "metrics" / "stage1_core_summary.csv"),
        reference=pd.read_csv(root / "metrics" / "stage1_reference_summary.csv"),
        diagnostics=pd.read_csv(root / "metrics" / "stage1_sparse_seed_evaluations.csv"),
    )
    timing = pd.read_csv(root / "metrics" / "stage1_timing_summary.csv")
    sparse_pairs = pd.read_csv(
        root / "metrics" / "stage1_sparse_seed_pairwise_summary.csv"
    )
    sparse_variance = pd.read_csv(
        root / "metrics" / "stage1_sparse_seed_variance_per_frame.csv"
    )
    interaction = pd.read_csv(root / "metrics" / "stage1_interaction_summary.csv")
    summary = pd.read_csv(root / "metrics" / "stage1_scale_policy_summary.csv")
    scale = ScaleBundle(
        summary=summary,
        transplants=summary.loc[summary.experiment_role.eq("controlled_policy")].copy(),
        native_response=summary.loc[
            summary.experiment_role.eq(FORMAL_NATIVE_SCALE_ROLE)
        ].copy(),
        slopes=pd.read_csv(root / "metrics" / "stage1_scale_sensitivity_slopes.csv"),
        ranks=pd.read_csv(root / "metrics" / "stage1_method_rank_stability.csv"),
        contact_robustness=pd.read_csv(
            root / "metrics" / "stage1_holosoma_contact_robustness.csv"
        ),
        contact_robustness_slopes=pd.read_csv(
            root / "metrics" / "stage1_holosoma_native_contact_slopes.csv"
        ),
        contact_diagnostics=pd.read_csv(
            root / "metrics" / "contact_constraint_flip.csv"
        ).loc[lambda frame: frame.method.astype(str).eq("omniretarget")],
    )
    comparison = pd.read_csv(root / "metrics" / "stage1_reference_comparison.csv")
    return (
        evaluations,
        timing,
        sparse_pairs,
        sparse_variance,
        interaction,
        scale,
        comparison,
    )


def finalize_stage1_publication(repo_root: str | Path = ".") -> dict[str, Any]:
    """Render final Markdown only from an independently bound validation result."""

    root = Path(repo_root).resolve()
    resolution = resolve_bound_stage1_validation(root)
    inputs = _load_final_report_inputs(root)
    render_stage1_reports(
        root,
        *inputs,
        decision_resolution=(
            resolution if resolution.get("binding_status") == "verified" else None
        ),
    )
    return resolution


def build_stage1_publication(repo_root: str | Path = ".") -> dict[str, Any]:
    """Build Stage 1 evidence and a non-authoritative ``PENDING`` manifest.

    This entry point performs evaluation and publication only.  It does not run
    or resume any retargeting or scale experiment, and it cannot issue GO.
    """

    root = Path(repo_root).resolve()
    # Invalidate any previously bound verdict before derived evidence is
    # regenerated.  If the build is interrupted, consumers fail closed rather
    # than continuing to display a stale GO from an older ledger.
    atomic_write_json(
        root / "manifests" / "stage1_publication.json",
        {
            "schema_version": PUBLICATION_EVIDENCE_SCHEMA_VERSION,
            "decision": "PENDING",
            "state": "evidence_build_in_progress",
            "decision_authority": "manifests/stage1_validation.json",
        },
    )
    evaluations = evaluate_stage1_outputs(root)
    timing_raw, timing = collect_stage1_timing(root)
    _, sparse_pairs, sparse_variance = compute_sparse_seed_diagnostics(root)
    interaction = collect_interaction_evidence(root)
    scale = collect_scale_evidence(root)
    comparison = pd.concat(
        (
            build_reference_comparison(evaluations.core, evaluations.reference),
            build_direct_reference_comparison(root),
        ),
        ignore_index=True,
        sort=False,
    )
    _write_frame(comparison, root / "metrics" / "stage1_reference_comparison")
    figure_paths = build_stage1_figures(
        root,
        evaluations,
        timing,
        sparse_pairs,
        interaction,
        scale,
        comparison,
    )
    render_stage1_reports(
        root,
        evaluations,
        timing,
        sparse_pairs,
        sparse_variance,
        interaction,
        scale,
        comparison,
    )
    metric_paths = [
        path
        for path in (root / "metrics").glob("stage1_*")
        if path.is_file()
    ]
    metric_paths.extend(
        path
        for path in (root / "metrics" / "stage1_publication").rglob("*")
        if path.is_file()
    )
    evidence_paths = list(
        dict.fromkeys([*sorted(metric_paths), *figure_paths])
    )
    input_hashes = _hashed_file_rows(root, _direct_publication_input_paths(root))
    output_hashes = _hashed_file_rows(root, evidence_paths)
    manifest = {
        "schema_version": PUBLICATION_EVIDENCE_SCHEMA_VERSION,
        "decision": "PENDING",
        "state": "evidence_ready_for_independent_validation",
        "decision_authority": "manifests/stage1_validation.json",
        "claim_scope": "one frozen 600-frame Pilot only",
        "core_methods": [spec.key for spec in CORE_METHODS],
        "sparse_diagnostics": [spec.key for spec in SPARSE_DIAGNOSTICS],
        "external_reference": {
            "key": REFERENCE_METHOD.key,
            "timed": False,
            "verified_official_ground_truth": False,
        },
        "core_rows": len(evaluations.core),
        "reference_rows": len(evaluations.reference),
        "timing_rows": len(timing_raw),
        "scale_rows": len(scale.summary),
        "scale_fixed_contact_rows": 25,
        "scale_native_contact_robustness_rows": 5,
        "scale_registered_metric_count": len(SCALE_RESPONSE_METRIC_NAMES),
        "scale_slope_rows": len(scale.slopes),
        "scale_rank_rows": len(scale.ranks),
        "scale_input_ledger": SCALE_INPUT_LEDGER,
        "scale_input_ledger_sha256": sha256_file(root / SCALE_INPUT_LEDGER),
        "interaction_rows": len(interaction),
        "presentation_slides": presentation_slide_count(
            (root / "PRESENTATION.md").read_text(encoding="utf-8")
        ),
        "input_hashes": input_hashes,
        "input_bundle_sha256": _canonical_sha256(input_hashes),
        "outputs": output_hashes,
        "output_bundle_sha256": _canonical_sha256(output_hashes),
        "report_files": list(REPORT_FILES),
        "historical_stage1_stop_marker": HISTORICAL_STAGE1_STOP,
        "retargeting_experiments_launched_by_builder": False,
    }
    atomic_write_json(root / "manifests" / "stage1_publication.json", manifest)
    return manifest
