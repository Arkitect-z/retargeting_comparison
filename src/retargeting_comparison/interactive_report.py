"""Build the deterministic, standalone Stage 1 interactive research report."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .constants import FULL_LAFAN_STOP_MESSAGE
from .io_utils import atomic_write_text, load_yaml, sha256_file


OPERATING_POINTS = ("sparse-neutral", "dense", "gmr", "omniretarget")
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
    for method in OPERATING_POINTS:
        rows = _csv_rows(root / "metrics" / "runs" / f"{method}_per_frame.csv")
        series[method] = {
            field: [row[field] for row in rows]
            for field in FRAME_FIELDS
        }
    return series


def _seed_series(root: Path) -> dict[str, dict[str, list[float]]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"frame": [], "joint_angle_rms_rad": [], "robot_rf_point_rms_m": []}
    )
    for row in _csv_rows(root / "metrics" / "sparse_seed_divergence_per_frame.csv"):
        pair = str(row["pair"])
        grouped[pair]["frame"].append(int(row["frame"]))
        grouped[pair]["joint_angle_rms_rad"].append(float(row["joint_angle_rms_rad"]))
        grouped[pair]["robot_rf_point_rms_m"].append(float(row["robot_rf_point_rms_m"]))
    return dict(grouped)


def _input_hashes(root: Path) -> list[dict[str, Any]]:
    paths = [
        root / "metrics" / "core_summary.csv",
        root / "metrics" / "interaction_summary.csv",
        root / "metrics" / "timing_raw.csv",
        root / "metrics" / "timing_summary.csv",
        root / "metrics" / "stage2_projection.csv",
        root / "metrics" / "stage2_projection.json",
        root / "metrics" / "sparse_seed_divergence_per_frame.csv",
        root / "metrics" / "source_adapter_errors.csv",
        root / "metrics" / "root_scale_diagnostics.csv",
        root / "metrics" / "controlled_task_residuals.csv",
        root / "metrics" / "conditional_candidates.csv",
        root / "manifests" / "evaluator.yaml",
        root / "manifests" / "source_adapters.yaml",
        root / "manifests" / "conditional_candidates.json",
        root / "manifests" / "pilot_sequence.yaml",
        root / "manifests" / "dataset.yaml",
        root / "manifests" / "hardware.yaml",
        root / "manifests" / "body_models.yaml",
        root / "manifests" / "stage1_validation.json",
    ]
    paths.extend(
        root / "metrics" / "runs" / f"{method}_per_frame.csv"
        for method in OPERATING_POINTS
    )
    rerun_manifest = root / "manifests" / "rerun_visualization.json"
    if rerun_manifest.is_file():
        paths.append(rerun_manifest)
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]


def collect_interactive_data(repo_root: str | Path = ".") -> dict[str, Any]:
    """Collect only committed, publication-safe evidence for the browser report."""

    root = Path(repo_root).resolve()
    core = [
        row
        for row in _csv_rows(root / "metrics" / "core_summary.csv")
        if row.get("label") in OPERATING_POINTS
    ]
    core.sort(key=lambda row: OPERATING_POINTS.index(str(row["label"])))
    timing_summary = _csv_rows(root / "metrics" / "timing_summary.csv")
    timing_summary = [row for row in timing_summary if row.get("label") in OPERATING_POINTS]
    timing_by_label = {str(row["label"]): row for row in timing_summary}
    for row in core:
        row.update(timing_by_label[str(row["label"])])
    interaction = _csv_rows(root / "metrics" / "interaction_summary.csv")
    interaction.sort(key=lambda row: (str(row["case"]), str(row["variant"])))
    timing_raw = _csv_rows(root / "metrics" / "timing_raw.csv")
    timing_raw = [row for row in timing_raw if row.get("method") in OPERATING_POINTS]
    projection_rows = _csv_rows(root / "metrics" / "stage2_projection.csv")
    repositories = _csv_rows(root / "manifests" / "repositories.csv")
    adapters = _csv_rows(root / "metrics" / "source_adapter_errors.csv")
    seed_summary = _csv_rows(root / "metrics" / "sparse_seed_divergence.csv")
    scale_diagnostics = _csv_rows(root / "metrics" / "root_scale_diagnostics.csv")
    controlled_residuals = _csv_rows(root / "metrics" / "controlled_task_residuals.csv")
    conditional_candidates = _csv_rows(root / "metrics" / "conditional_candidates.csv")

    return {
        "schema_version": 2,
        "title": "Human-to-G1 Retargeting — Stage 1 Pilot",
        "decision": _json(root / "manifests" / "stage1_validation.json")["decision"],
        "hard_stop_message": FULL_LAFAN_STOP_MESSAGE,
        "operating_points": list(OPERATING_POINTS),
        "core": core,
        "frame_series": _frame_series(root),
        "seed_series": _seed_series(root),
        "seed_summary": seed_summary,
        "scale_diagnostics": scale_diagnostics,
        "controlled_residuals": controlled_residuals,
        "conditional_candidates": conditional_candidates,
        "evaluator": load_yaml(root / "manifests" / "evaluator.yaml"),
        "interaction": interaction,
        "timing_raw": timing_raw,
        "timing_summary": timing_summary,
        "stage2_projection_rows": projection_rows,
        "stage2_projection": _json(root / "metrics" / "stage2_projection.json"),
        "validation": _json(root / "manifests" / "stage1_validation.json"),
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
        "input_hashes": _input_hashes(root),
        "links": {
            "pilot_report": "PILOT_REPORT.md",
            "decision": "GO_NO_GO.md",
            "interaction": "INTERACTION_CASE_STUDY.md",
            "sparse": "SPARSE_IK_ANALYSIS.md",
            "reproduce": "REPRODUCE_PILOT.md",
            "rerun_guide": "docs/RERUN_VISUALIZATION.md",
            "completion_audit": "docs/STAGE1_COMPLETION_AUDIT.md",
            "rerun_manifest": "manifests/rerun_visualization.json",
            "artifacts": "manifests/artifacts.csv",
            "core_csv": "metrics/core_summary.csv",
            "interaction_csv": "metrics/interaction_summary.csv",
            "timing_csv": "metrics/timing_raw.csv",
            "projection_csv": "metrics/stage2_projection.csv",
        },
    }


def build_interactive_report(
    repo_root: str | Path = ".",
    output_path: str | Path = "INTERACTIVE_REPORT.html",
) -> Path:
    """Render a single-file, dependency-free interactive scientific report."""

    root = Path(repo_root).resolve()
    source_dir = root / "interactive_report"
    template = (source_dir / "template.html").read_text(encoding="utf-8")
    css = (source_dir / "report.css").read_text(encoding="utf-8")
    javascript = (source_dir / "report.js").read_text(encoding="utf-8")
    data = collect_interactive_data(root)
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    replacements = {
        "__RTCMP_INLINE_CSS__": css,
        "__RTCMP_INLINE_DATA__": encoded,
        "__RTCMP_INLINE_JS__": javascript,
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
    return output
