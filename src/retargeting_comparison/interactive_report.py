"""Build the deterministic, standalone Stage 1 interactive research report."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .constants import FULL_LAFAN_STOP_MESSAGE
from .io_utils import atomic_write_json, atomic_write_text, load_yaml, sha256_file
from .stage1_publication import resolve_bound_stage1_validation


OPERATING_POINTS = (
    "sparse-neutral",
    "dense",
    "gmr",
    "omniretarget",
    "protomotions-v2.3",
    "protomotions-v3",
)
EXTERNAL_REFERENCES = ("unitree-reference",)
TRAJECTORY_SERIES = (*OPERATING_POINTS, *EXTERNAL_REFERENCES)
FRAME_FIELDS = (
    "source_frame_idx",
    "rf_kpe_all_m",
    "rf_kpe_targeted_m",
    "rf_kpe_untracked_m",
    "artifact",
    "foot_skating",
    "ground_penetration_depth_m",
    "pose_jump_rms_m",
    "solve_time_s",
    "artifact_causes",
)

_STATIC_CLOSING_COPY = (
    "Six complete operating points, an explicitly delimited external reference, "
    "exact pre-solver target capture, causal scale interventions, repeated timing, "
    "and two interaction cases support a Pilot-level GO. Dataset-level ranking "
    "remains deliberately deferred to the budget-gated Stage 2 run."
)


def _apply_decision_copy(
    template: str, javascript: str, decision: str
) -> tuple[str, str]:
    """Remove the legacy static-GO copy from the rendered browser artifact."""

    closing = {
        "GO": (
            "The independently validated, hash-bound Stage 1 evidence supports a "
            "Pilot-level GO. Dataset-level ranking remains deferred to Stage 2."
        ),
        "NO-GO": (
            "Independent fail-closed validation found unmet Stage 1 acceptance "
            "conditions. This evidence package does not authorize a GO."
        ),
        "PENDING": (
            "The evidence package awaits an independently validated, hash-bound "
            "Stage 1 decision. The publication builder cannot issue GO."
        ),
    }.get(decision)
    if closing is None:
        raise ValueError(f"Unsupported interactive decision: {decision!r}")
    if template.count(_STATIC_CLOSING_COPY) != 1:
        raise ValueError("Interactive template static decision copy changed unexpectedly")
    template = template.replace(_STATIC_CLOSING_COPY, closing)

    old_subtitle = (
        ': decision === "GO WITH CHANGES" ? "WITH CHANGES" : "GATE CLOSED";'
    )
    new_subtitle = (
        ': decision === "GO WITH CHANGES" ? "WITH CHANGES"\n'
        '      : decision === "PENDING" ? "AWAITING VALIDATION" : "GATE CLOSED";'
    )
    old_copy = (
        ': "The frozen validation manifest contains at least one unmet Stage 1 '
        'acceptance condition.";'
    )
    new_copy = (
        ': decision === "PENDING"\n'
        '      ? "Independent Stage 1 validation has not yet bound a verdict to '
        'the current evidence ledger."\n'
        '      : "The frozen validation manifest contains at least one unmet Stage 1 '
        'acceptance condition.";'
    )
    if javascript.count(old_subtitle) != 1 or javascript.count(old_copy) != 1:
        raise ValueError("Interactive verdict script changed unexpectedly")
    javascript = javascript.replace(old_subtitle, new_subtitle).replace(
        old_copy, new_copy
    )
    return template, javascript


def _coerce(value: str) -> Any:
    stripped = value.strip()
    if stripped == "":
        return None
    lowered = stripped.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if not any(character in stripped for character in ".eE"):
            return int(stripped)
        return float(stripped)
    except ValueError:
        return value


def _csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return [
            {key: _coerce(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def _frame_series(root: Path) -> dict[str, dict[str, list[Any]]]:
    series: dict[str, dict[str, list[Any]]] = {}
    for method in TRAJECTORY_SERIES:
        path = (
            root
            / "metrics"
            / "stage1_publication"
            / "runs"
            / f"{method}_per_frame.csv"
        )
        rows = _csv_rows(path)
        series[method] = {
            field: [row[field] for row in rows]
            for field in FRAME_FIELDS
        }
    return series


def _seed_series(root: Path) -> dict[str, dict[str, list[float]]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"frame": [], "joint_angle_rms_rad": [], "robot_rf_point_rms_m": []}
    )
    for row in _csv_rows(
        root / "metrics" / "stage1_sparse_seed_pairwise_per_frame.csv"
    ):
        pair = str(row["pair"])
        grouped[pair]["frame"].append(int(row["source_frame_idx"]))
        grouped[pair]["joint_angle_rms_rad"].append(float(row["joint_angle_rms_rad"]))
        grouped[pair]["robot_rf_point_rms_m"].append(float(row["robot_rf_point_rms_m"]))
    return dict(grouped)


def _input_hashes(root: Path) -> list[dict[str, Any]]:
    paths = [
        root / "metrics" / "stage1_core_summary.csv",
        root / "metrics" / "stage1_reference_summary.csv",
        root / "metrics" / "stage1_reference_comparison.csv",
        root / "metrics" / "stage1_timing_raw.csv",
        root / "metrics" / "stage1_timing_summary.csv",
        root / "metrics" / "stage1_scale_policy_summary.csv",
        root / "metrics" / "stage1_scale_sensitivity_slopes.csv",
        root / "metrics" / "stage1_method_rank_stability.csv",
        root / "metrics" / "stage1_holosoma_contact_robustness.csv",
        root / "metrics" / "stage1_holosoma_native_contact_slopes.csv",
        root / "metrics" / "actor_shape_policy_formula_probe.csv",
        root / "metrics" / "contact_constraint_flip.csv",
        root / "metrics" / "stage1_interaction_summary.csv",
        root / "metrics" / "stage2_projection.csv",
        root / "metrics" / "stage2_projection.json",
        root / "metrics" / "stage1_sparse_seed_pairwise_per_frame.csv",
        root / "metrics" / "stage1_sparse_seed_pairwise_summary.csv",
        root / "metrics" / "source_adapter_errors.csv",
        root / "metrics" / "conditional_candidates.csv",
        root / "research" / "unitree_reference_provenance.csv",
        root / "research" / "OFFICIAL_SCALE_AND_PREPROCESSING_AUDIT.md",
        root / "configs" / "scale_policy_sensitivity.yaml",
        root / "configs" / "stage2.yaml",
        root / "manifests" / "evaluator.yaml",
        root / "manifests" / "source_adapters.yaml",
        root / "manifests" / "conditional_candidates.json",
        root / "manifests" / "pilot_sequence.yaml",
        root / "manifests" / "dataset.yaml",
        root / "manifests" / "hardware.yaml",
        root / "manifests" / "body_models.yaml",
        root / "manifests" / "stage1_publication.json",
        root / "interactive_report" / "template.html",
        root / "interactive_report" / "report.css",
        root / "interactive_report" / "report.js",
    ]
    paths.extend(
        root / "metrics" / "stage1_publication" / "runs" / f"{method}_per_frame.csv"
        for method in TRAJECTORY_SERIES
    )
    rerun_manifest = root / "manifests" / "rerun_visualization.json"
    if rerun_manifest.is_file():
        paths.append(rerun_manifest)
    validation_manifest = root / "manifests" / "stage1_validation.json"
    if validation_manifest.is_file():
        paths.append(validation_manifest)
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]


def collect_interactive_data(
    repo_root: str | Path = ".",
    *,
    force_pending: bool = False,
) -> dict[str, Any]:
    """Collect only committed, publication-safe evidence for the browser report."""

    root = Path(repo_root).resolve()
    core = _csv_rows(root / "metrics" / "stage1_core_summary.csv")
    core = [row for row in core if row.get("key") in OPERATING_POINTS]
    for row in core:
        row["label"] = row["key"]
    core.sort(key=lambda row: OPERATING_POINTS.index(str(row["label"])))
    reference = _csv_rows(root / "metrics" / "stage1_reference_summary.csv")
    for row in reference:
        row["label"] = row["key"]
    if [row.get("label") for row in reference] != list(EXTERNAL_REFERENCES):
        raise ValueError("Interactive report requires the frozen external reference")
    timing_summary = _csv_rows(root / "metrics" / "stage1_timing_summary.csv")
    timing_summary = [row for row in timing_summary if row.get("key") in OPERATING_POINTS]
    for row in timing_summary:
        row["label"] = row["key"]
    timing_by_label = {str(row["label"]): row for row in timing_summary}
    for row in core:
        row.update(timing_by_label[str(row["label"])])
    interaction = _csv_rows(root / "metrics" / "stage1_interaction_summary.csv")
    interaction.sort(key=lambda row: (str(row["case"]), str(row["variant"])))
    timing_raw = _csv_rows(root / "metrics" / "stage1_timing_raw.csv")
    timing_raw = [row for row in timing_raw if row.get("key") in OPERATING_POINTS]
    for row in timing_raw:
        row["method"] = row["key"]
    projection_rows = _csv_rows(root / "metrics" / "stage2_projection.csv")
    repositories = _csv_rows(root / "manifests" / "repositories.csv")
    adapters = _csv_rows(root / "metrics" / "source_adapter_errors.csv")
    seed_summary = _csv_rows(root / "metrics" / "stage1_sparse_seed_pairwise_summary.csv")
    scale_summary = _csv_rows(root / "metrics" / "stage1_scale_policy_summary.csv")
    scale_slopes = _csv_rows(root / "metrics" / "stage1_scale_sensitivity_slopes.csv")
    rank_stability = _csv_rows(root / "metrics" / "stage1_method_rank_stability.csv")
    contact_robustness = _csv_rows(
        root / "metrics" / "stage1_holosoma_contact_robustness.csv"
    )
    actor_shape = _csv_rows(
        root / "metrics" / "actor_shape_policy_formula_probe.csv"
    )
    reference_comparison = _csv_rows(
        root / "metrics" / "stage1_reference_comparison.csv"
    )
    direct_metrics = {
        "direct_root_translation_mean_m",
        "direct_root_translation_p95_m",
        "direct_root_anchor_frame0_m",
        "direct_root_displacement_mean_m",
        "direct_root_displacement_p95_m",
        "direct_root_yaw_mean_rad",
        "direct_root_orientation_mean_rad",
        "direct_joint_angle_rmse_rad",
        "direct_joint_velocity_rmse_rad_s",
        "direct_fk_world_semantic_mean_m",
        "direct_fk_root_frame_semantic_mean_m",
    }
    direct_rows = [
        row
        for row in reference_comparison
        if row.get("comparison_type")
        == "direct_same_g1_frame_index_aligned_trajectory_disagreement"
    ]
    if (
        len(direct_rows) != len(OPERATING_POINTS) * len(direct_metrics)
        or {str(row.get("key")) for row in direct_rows} != set(OPERATING_POINTS)
        or {str(row.get("metric")) for row in direct_rows} != direct_metrics
        or {
            str(row.get("timeline_alignment")) for row in direct_rows
        }
        != {"frame_index_only_not_exact_timestamp"}
        or not all(
            isinstance(row.get("method_value"), (int, float))
            and math.isfinite(float(row["method_value"]))
            and float(row["method_value"]) >= 0.0
            for row in direct_rows
        )
    ):
        raise ValueError(
            "Interactive report requires the complete frame-index-aligned "
            "same-G1 reference table"
        )
    reference_provenance = _csv_rows(
        root / "research" / "unitree_reference_provenance.csv"
    )
    conditional_candidates = _csv_rows(root / "metrics" / "conditional_candidates.csv")
    resolution = (
        {
            "decision": "PENDING",
            "binding_status": "bootstrap_pending",
            "reason": "Interactive structure is being rendered before independent validation",
            "validation": None,
        }
        if force_pending
        else resolve_bound_stage1_validation(root)
    )
    validation_payload = resolution.get("validation") or {
        "schema_version": None,
        "decision": "PENDING",
        "authority": None,
        "checks": {},
        "failed_checks": [],
        "provisional": True,
    }

    input_hashes = _input_hashes(root)
    data = {
        "schema_version": 5,
        "title": "Human-to-G1 Retargeting — Stage 1 Pilot Evidence",
        "decision": resolution["decision"],
        "decision_authority": {
            "path": "manifests/stage1_validation.json",
            "binding_status": resolution["binding_status"],
            "reason": resolution["reason"],
            "validation_sha256": resolution.get("validation_sha256"),
            "publication_evidence_binding": resolution.get(
                "publication_evidence_binding"
            ),
        },
        "hard_stop_message": FULL_LAFAN_STOP_MESSAGE,
        "operating_points": list(OPERATING_POINTS),
        "external_references": list(EXTERNAL_REFERENCES),
        "trajectory_series": list(TRAJECTORY_SERIES),
        "core": core,
        "reference": reference,
        "reference_comparison": reference_comparison,
        "reference_provenance": reference_provenance,
        "frame_series": _frame_series(root),
        "seed_series": _seed_series(root),
        "seed_summary": seed_summary,
        "scale_summary": scale_summary,
        "scale_slopes": scale_slopes,
        "rank_stability": rank_stability,
        "contact_robustness": contact_robustness,
        "actor_shape": actor_shape,
        "conditional_candidates": conditional_candidates,
        "evaluator": load_yaml(root / "manifests" / "evaluator.yaml"),
        "interaction": interaction,
        "timing_raw": timing_raw,
        "timing_summary": timing_summary,
        "stage2_projection_rows": projection_rows,
        "stage2_projection": _json(root / "metrics" / "stage2_projection.json"),
        "validation": validation_payload,
        "rerun_visualization": (
            _json(root / "manifests" / "rerun_visualization.json")
            if (root / "manifests" / "rerun_visualization.json").is_file()
            else None
        ),
        "pilot": load_yaml(root / "manifests" / "pilot_sequence.yaml"),
        "dataset": load_yaml(root / "manifests" / "dataset.yaml"),
        "hardware": load_yaml(root / "manifests" / "hardware.yaml"),
        "body_models": load_yaml(root / "manifests" / "body_models.yaml"),
        "methods": load_yaml(root / "manifests" / "methods.yaml"),
        "repositories": repositories,
        "adapters": adapters,
        "input_hashes": input_hashes,
        "evidence_bundle_sha256": hashlib.sha256(
            json.dumps(
                input_hashes,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest(),
        "links": {
            "pilot_report": "PILOT_REPORT.md",
            "decision": "GO_NO_GO.md",
            "interaction": "INTERACTION_CASE_STUDY.md",
            "sparse": "SPARSE_IK_ANALYSIS.md",
            "reproduce": "REPRODUCE_PILOT.md",
            "rerun_guide": "docs/RERUN_VISUALIZATION.md",
            "completion_audit": "docs/STAGE1_COMPLETION_AUDIT.md",
            "preprocessing_audit": "research/OFFICIAL_SCALE_AND_PREPROCESSING_AUDIT.md",
            "scale_report": "SCALE_POLICY_SENSITIVITY.md",
            "reference_report": "UNITREE_REFERENCE_COMPARISON.md",
            "stage1_review": "STAGE1_REVIEW.md",
            "scale_sensitivity_config": "configs/scale_policy_sensitivity.yaml",
            "rerun_manifest": "manifests/rerun_visualization.json",
            "artifacts": "manifests/artifacts.csv",
            "core_csv": "metrics/stage1_core_summary.csv",
            "reference_csv": "metrics/stage1_reference_summary.csv",
            "scale_csv": "metrics/stage1_scale_policy_summary.csv",
            "interaction_csv": "metrics/stage1_interaction_summary.csv",
            "timing_csv": "metrics/stage1_timing_raw.csv",
            "projection_csv": "metrics/stage2_projection.csv",
        },
    }
    return data


def build_interactive_report(
    repo_root: str | Path = ".",
    output_path: str | Path = "INTERACTIVE_REPORT.html",
    *,
    force_pending: bool = False,
) -> Path:
    """Render a single-file, dependency-free interactive scientific report."""

    root = Path(repo_root).resolve()
    source_dir = root / "interactive_report"
    template = (source_dir / "template.html").read_text(encoding="utf-8")
    css = (source_dir / "report.css").read_text(encoding="utf-8")
    javascript = (source_dir / "report.js").read_text(encoding="utf-8")
    data = collect_interactive_data(root, force_pending=force_pending)
    template, javascript = _apply_decision_copy(
        template, javascript, str(data["decision"])
    )
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    embedded_data_sha256 = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    replacements = {
        "__RTCMP_INLINE_CSS__": css,
        "__RTCMP_INLINE_DATA__": encoded,
        "__RTCMP_INLINE_JS__": javascript,
        "__RTCMP_DATA_SHA256__": embedded_data_sha256,
    }
    rendered = template
    for marker, value in replacements.items():
        if rendered.count(marker) != 1:
            raise ValueError(f"Template marker must occur exactly once: {marker}")
        rendered = rendered.replace(marker, value)
    output = Path(output_path)
    if not output.is_absolute():
        output = root / output
    atomic_write_text(output, rendered.rstrip() + "\n")
    audit = audit_interactive_delivery(
        root, output, require_rerun=not force_pending
    )
    try:
        portable_output = output.relative_to(root).as_posix()
    except ValueError:
        portable_output = str(output)
    delivery_manifest = {
        **audit,
        "output": portable_output,
    }
    atomic_write_json(
        output.with_suffix(output.suffix + ".manifest.json"), delivery_manifest
    )
    return output


def audit_interactive_delivery(
    repo_root: str | Path,
    output_path: str | Path = "INTERACTIVE_REPORT.html",
    *,
    require_rerun: bool = True,
) -> dict[str, Any]:
    """Verify the rendered standalone report and every embedded evidence hash."""

    root = Path(repo_root).resolve()
    output = Path(output_path)
    if not output.is_absolute():
        output = root / output
    if not output.is_file():
        raise FileNotFoundError(f"Interactive report is missing: {output}")
    text = output.read_text(encoding="utf-8")
    if "__RTCMP_INLINE_" in text or "__RTCMP_DATA_SHA256__" in text:
        raise ValueError("Interactive template markers remain in the rendered artifact")
    if re.search(r"<(?:script|link)[^>]+(?:src|href)=[\"']https?://", text):
        raise ValueError("Interactive report has an external runtime dependency")

    data_match = re.search(
        r'<script id="rtcmp-data" type="application/json">(.*?)</script>',
        text,
        flags=re.DOTALL,
    )
    digest_match = re.search(
        r'<meta name="rtcmp-data-sha256" content="([0-9a-f]{64})">', text
    )
    if data_match is None or digest_match is None:
        raise ValueError("Interactive report lacks its embedded data digest")
    embedded = data_match.group(1)
    embedded_sha = hashlib.sha256(embedded.encode("utf-8")).hexdigest()
    if embedded_sha != digest_match.group(1):
        raise ValueError("Interactive embedded data was modified after rendering")
    data = json.loads(embedded)
    if not isinstance(data, dict) or int(data.get("schema_version", 0)) != 5:
        raise ValueError("Interactive evidence schema v5 is required")
    if data.get("operating_points") != list(OPERATING_POINTS):
        raise ValueError("Interactive operating-point order is stale")
    if data.get("external_references") != list(EXTERNAL_REFERENCES):
        raise ValueError("Interactive reference set is stale")
    if data.get("trajectory_series") != list(TRAJECTORY_SERIES):
        raise ValueError("Interactive trajectory series is stale")

    hashes = data.get("input_hashes")
    if not isinstance(hashes, list) or not hashes:
        raise ValueError("Interactive report has no input hash ledger")
    paths = [str(row.get("path", "")) for row in hashes]
    if len(paths) != len(set(paths)):
        raise ValueError("Interactive input hash ledger contains duplicates")
    for row in hashes:
        candidate = (root / str(row.get("path", ""))).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError("Interactive evidence path escapes the repository")
        if (
            not candidate.is_file()
            or candidate.stat().st_size != int(row.get("size_bytes", -1))
            or sha256_file(candidate) != str(row.get("sha256", ""))
        ):
            raise ValueError(
                f"Interactive evidence hash/size mismatch: {row.get('path')}"
            )
    bundle_sha = hashlib.sha256(
        json.dumps(
            hashes,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    if bundle_sha != data.get("evidence_bundle_sha256"):
        raise ValueError("Interactive evidence bundle digest is stale")

    rerun = data.get("rerun_visualization")
    if require_rerun:
        from .rerun_visualization import EXPECTED_TRAJECTORY_KEYS

        rerun_path = root / "manifests/rerun_visualization.json"
        if rerun != _json(rerun_path):
            raise ValueError("Interactive report embeds a stale Rerun manifest")
        if (
            int(rerun.get("schema_version", 0)) != 7
            or rerun.get("methods") != list(EXPECTED_TRAJECTORY_KEYS)
            or rerun.get("rrd_verification", {}).get("result") != "verified"
        ):
            raise ValueError("Interactive report does not bind the verified nine-trajectory RRD")

    required_sections = (
        "design",
        "frontier",
        "scale",
        "reference",
        "frame-story",
        "sparse",
        "interaction",
        "timing",
        "budget",
        "evidence",
    )
    missing_sections = [
        section for section in required_sections if f'id="{section}"' not in text
    ]
    if missing_sections:
        raise ValueError(f"Interactive narrative sections are missing: {missing_sections}")

    return {
        "schema_version": 2,
        "result": "verified",
        "output_size_bytes": output.stat().st_size,
        "output_sha256": sha256_file(output),
        "embedded_data_sha256": embedded_sha,
        "evidence_bundle_sha256": bundle_sha,
        "operating_points": list(OPERATING_POINTS),
        "external_references": list(EXTERNAL_REFERENCES),
        "rerun_required": require_rerun,
        "rerun_manifest_sha256": (
            sha256_file(root / "manifests/rerun_visualization.json")
            if require_rerun
            else None
        ),
        "required_sections": list(required_sections),
        "external_runtime_dependencies": 0,
    }


def validate_interactive_delivery_manifest(
    repo_root: str | Path,
    output_path: str | Path = "INTERACTIVE_REPORT.html",
) -> dict[str, Any]:
    """Re-audit a final HTML delivery and compare its immutable sidecar."""

    root = Path(repo_root).resolve()
    output = Path(output_path)
    if not output.is_absolute():
        output = root / output
    sidecar = output.with_suffix(output.suffix + ".manifest.json")
    frozen = json.loads(sidecar.read_text(encoding="utf-8"))
    current = audit_interactive_delivery(root, output, require_rerun=True)
    for key, value in current.items():
        if frozen.get(key) != value:
            raise ValueError(f"Interactive delivery sidecar is stale for {key}")
    return current
