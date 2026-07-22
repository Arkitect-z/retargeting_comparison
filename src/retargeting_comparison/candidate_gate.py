"""Legacy bounded integration-readiness evidence.

The revised Stage 1 promotes ProtoMotions v2.3/v3 to required methods and keeps
PHC as noncanonical lineage evidence.  This module preserves the earlier gate
record for provenance; an N/A row can no longer satisfy revised completion.
"""

from __future__ import annotations

import csv
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .constants import FULL_LAFAN_STOP_MESSAGE
from .io_utils import atomic_write_json, load_yaml, sha256_file
from .schemas import CanonicalG1


NA_OUTCOME = "N/A — public human→G1 pipeline not integration-ready under the Pilot budget"
NONCANONICAL_OUTCOME = (
    "N/A — official public G1 fitting asset is not the canonical 29-DoF embodiment"
)
GATE_LIMIT_S = 7200.0


def _git_commit(checkout: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _module_available(python: Path, module: str) -> bool:
    if not python.is_file():
        return False
    command = [
        str(python),
        "-c",
        f"import importlib.util; raise SystemExit(0 if importlib.util.find_spec({module!r}) else 1)",
    ]
    return subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def _configured_pythons() -> list[Path]:
    home = Path.home()
    roots = (home / "anaconda3" / "envs", home / "miniconda3" / "envs")
    return sorted(
        python
        for root in roots
        if root.is_dir()
        for python in root.glob("*/bin/python")
        if python.is_file()
    )


def _candidate_output(root: Path, key: str, sequence_id: str) -> Path:
    return root / "runs" / sequence_id / key / "canonical_g1.npz"


def _canonical_candidate_passed(path: Path, source_frames: int) -> bool:
    if not path.is_file():
        return False
    try:
        motion = CanonicalG1.load(path)
        motion.validate(source_frame_count=source_frames)
    except Exception:
        return False
    return bool(
        len(motion.qpos) == source_frames
        and motion.metadata.get("completion_status") == "succeeded"
    )


def _row(
    *,
    candidate: str,
    order: int,
    checkout: Path,
    expected_commit: str,
    input_ready: bool,
    environment_ready: bool,
    canonical_output: Path,
    source_frames: int,
    g1_29dof_declared: bool,
    role: str,
    evidence: list[str],
    started: float,
) -> dict[str, Any]:
    actual_commit = _git_commit(checkout)
    output_ready = _canonical_candidate_passed(canonical_output, source_frames)
    passed = bool(
        g1_29dof_declared
        and actual_commit == expected_commit
        and input_ready
        and environment_ready
        and output_ready
    )
    elapsed = time.perf_counter() - started
    missing = []
    if not g1_29dof_declared:
        missing.append("canonical G1-29 embodiment")
    if actual_commit != expected_commit:
        missing.append("frozen checkout")
    if not input_ready:
        missing.append("same-source native adapter")
    if not environment_ready:
        missing.append("frozen runnable environment")
    if not output_ready:
        missing.append("complete canonical 600-frame output")
    return {
        "candidate": candidate,
        "order": order,
        "status": "passed" if passed else "na",
        "outcome": (
            "Passed all Pilot integration gates"
            if passed
            else NONCANONICAL_OUTCOME
            if not g1_29dof_declared
            else NA_OUTCOME
        ),
        "reason": "all mandatory gates passed" if passed else "missing: " + "; ".join(missing),
        "upstream_commit": actual_commit,
        "expected_commit": expected_commit,
        "input_ready": input_ready,
        "g1_29dof_declared": g1_29dof_declared,
        "revised_stage1_role": role,
        "environment_ready": environment_ready,
        "canonical_output_ready": output_ready,
        "full_sequence_run": output_ready,
        "elapsed_s": elapsed,
        "gate_limit_s": GATE_LIMIT_S,
        "evidence": " | ".join(evidence),
    }


def gate_candidates(repo_root: str | Path = ".") -> list[dict[str, Any]]:
    root = Path(repo_root).resolve()
    sequence = load_yaml(root / "manifests" / "pilot_sequence.yaml")
    if sequence.get("full_lafan_authorized") is not False:
        raise RuntimeError("Conditional gates require the Full-LAFAN hard stop")
    sequence_id = str(sequence["sequence_id"])
    source_frames = int(sequence["num_frames"])
    pythons = _configured_pythons()

    proto_started = time.perf_counter()
    proto = root / "external" / "ProtoMotions"
    proto_script = proto / "pyroki" / "batch_retarget_to_g1_from_keypoints.py"
    proto_adapter = (
        root
        / "source_adapters"
        / "protomotions_v3"
        / sequence_id
        / "keypoints.npy"
    )
    proto_environment = any(
        python.parent.parent.name in {"pyroki", "rtcmp_pyroki"}
        and all(_module_available(python, module) for module in ("jax", "pyroki", "jaxls"))
        for python in pythons
    )
    proto_text = proto_script.read_text() if proto_script.is_file() else ""
    proto_row = _row(
        candidate="ProtoMotions v3",
        order=1,
        checkout=proto,
        expected_commit="49fe5ad69de67ebbc07ea2b25d41b0f622c15c3c",
        input_ready=proto_adapter.is_file(),
        environment_ready=proto_environment,
        canonical_output=_candidate_output(root, "protomotions-v3", sequence_id),
        source_frames=source_frames,
        g1_29dof_declared=True,
        role="required_pending_full_integration",
        evidence=[
            "official pyroki/batch_retarget_to_g1_from_keypoints.py",
            f"target_raw_frames_cli={('--target-raw-frames' in proto_text)}",
            f"native_adapter={proto_adapter.relative_to(root)}",
            "requires separate JAX/PyRoki environment per official workflow",
        ],
        started=proto_started,
    )

    phc_started = time.perf_counter()
    phc = root / "external" / "PHC"
    phc_adapter = root / "source_adapters" / "phc" / sequence_id / "motion.npz"
    phc_environment = any(
        python.parent.parent.name in {"phc", "rtcmp_phc"}
        and all(_module_available(python, module) for module in ("smpl_sim", "hydra", "torch"))
        for python in pythons
    )
    phc_row = _row(
        candidate="PHC",
        order=2,
        checkout=phc,
        expected_commit="846988d433ce1f341e85ac6fbd2cd51911bb3341",
        input_ready=phc_adapter.is_file(),
        environment_ready=phc_environment,
        canonical_output=_candidate_output(root, "phc", sequence_id),
        source_frames=source_frames,
        g1_29dof_declared=False,
        role="lineage_and_amass_policy_evidence_only",
        evidence=[
            "official docs/retargeting.md",
            "official scripts/data_process/fit_smpl_shape.py",
            "official scripts/data_process/fit_smpl_motion.py",
            "official fitting asset has 37 motors and is not canonical G1-29",
            f"native_adapter={phc_adapter.relative_to(root)}",
            "requires AMASS/SMPL parameters plus pre-fitted robot shape; raw BVH is not a native input",
        ],
        started=phc_started,
    )
    rows = [proto_row, phc_row]
    output = root / "metrics" / "conditional_candidates.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sequence_id": sequence_id,
        "source_frames": source_frames,
        "per_candidate_gate_limit_s": GATE_LIMIT_S,
        "rows_sha256": sha256_file(output),
        "results": rows,
        "full_lafan_authorized": False,
        "hard_stop_message": FULL_LAFAN_STOP_MESSAGE,
    }
    atomic_write_json(root / "manifests" / "conditional_candidates.json", manifest)
    return rows
