"""Stage 1 acceptance checks and Full-LAFAN hard stop enforcement."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import FULL_LAFAN_STOP_MESSAGE
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
        manifest_path = root / "manifests" / "runs" / f"{sequence_id}__{label}.json"
        if not manifest_path.is_file() or not output.is_file():
            return False
        manifest = RunManifest.load(manifest_path)
        if manifest.output_sha256 != sha256_file(output):
            return False
    return True


def _smoke_check(root: Path, sequence_id: str) -> bool:
    labels = ("sparse-neutral-smoke2", "dense-smoke2", "gmr-smoke2", "omniretarget-smoke2")
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
    if not path.is_file():
        return False
    rows = pd.read_csv(path)
    if len(rows) < 2:
        return False
    required = ("frame_count_match", "fps_match", "left_right_match")
    for column in required:
        values = rows[column]
        normalized = values if values.dtype == bool else values.astype(str).str.lower().eq("true")
        if not bool(normalized.all()):
            return False
    numeric = rows[["common_joints", "root_aligned_mpjpe_m", "max_joint_error_m"]].to_numpy()
    return bool(
        rows.status.astype(str).str.lower().eq("passed").all()
        and np.isfinite(numeric).all()
        and (rows.common_joints > 0).all()
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
        value.get("schema_version") != 2
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
        run = root / "runs" / sequence_id / label / "canonical_g1.npz"
        if not run.is_file() or value.get("method_outputs", {}).get(label) != sha256_file(run):
            return False
    return True


def _test_evidence_check(root: Path) -> bool:
    path = root / "manifests" / "test_evidence.json"
    if not path.is_file():
        return False
    value = json.loads(path.read_text())
    capture = value.get("capture_suite", {})
    rerun = value.get("rerun_recording", {})
    return bool(
        value.get("schema_version") == 2
        and value.get("full_lafan_authorized") is False
        and capture.get("result") == "passed"
        and int(capture.get("passed", 0)) >= 31
        and rerun.get("result") == "verified"
        and int(rerun.get("frames", 0)) == 600
        and rerun.get("rendering") == "articulated_g1_visual_meshes"
        and int(rerun.get("visual_asset_count", 0)) == 35
        and int(rerun.get("visual_asset_entities", 0)) == 140
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
    core_complete = all(checks[f"core_{label}"] for label in CORE_LABELS)
    interaction_complete = all(
        checks[f"interaction_{case}_{variant}"]
        for case in ("box", "climb")
        for variant in ("full", "no-hard")
    )
    reports_complete = all(checks[f"report_{report}"] for report in REPORTS)
    if (
        all(checks.values())
        and projection.get("within_wall_budget")
        and projection.get("within_storage_budget")
    ):
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
    artifact_manifest(root)
    return result
