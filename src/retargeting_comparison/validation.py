"""Stage 1 acceptance checks and Full-LAFAN hard stop enforcement."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import FULL_LAFAN_STOP_MESSAGE
from .io_utils import atomic_write_json, load_yaml, sha256_file
from .reporting import CORE_LABELS, REPORTS, _run_paths
from .schemas import CanonicalG1, RunManifest, RunStatus


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
            checks[key] = (
                len(motion.qpos) == int(sequence["num_frames"])
                and motion.metadata.get("completion_status") == "succeeded"
                and np.isfinite(motion.qpos).all()
            )
        except Exception:
            checks[key] = False
    for case in ("box", "climb"):
        for variant in ("full", "no-hard"):
            path = root / "runs" / "interaction_manifests" / f"interaction__{case}__{variant}.json"
            checks[f"interaction_{case}_{variant}"] = (
                path.is_file() and RunManifest.load(path).status == RunStatus.SUCCEEDED
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
    core_complete = all(checks[f"core_{label}"] for label in CORE_LABELS)
    interaction_complete = all(
        checks[f"interaction_{case}_{variant}"]
        for case in ("box", "climb")
        for variant in ("full", "no-hard")
    )
    reports_complete = all(checks[f"report_{report}"] for report in REPORTS)
    if all(checks.values()) and projection.get("within_wall_budget"):
        decision = "GO"
    elif core_complete and interaction_complete and reports_complete:
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
    return result
