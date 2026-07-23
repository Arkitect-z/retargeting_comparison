"""Resumable formal timing and scale-response campaign for ProtoMotions v3."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import atomic_write_json, sha256_file
from .protomotions_v3 import TARGET_RAW_FRAMES
from .schemas import CanonicalG1, CanonicalHuman, RunManifest, RunStatus


SCALE_VARIANTS = {
    "root_minus_5": (0.95, 1.0),
    "root_plus_5": (1.05, 1.0),
    "local_minus_5": (1.0, 0.95),
    "local_plus_5": (1.0, 1.05),
}
FORMAL_RUN_DIRECTORY = "protomotions-v3-v3"


def _variant_name(root_multiplier: float, local_multiplier: float) -> str:
    mapping = {(1.0, 1.0): "native", **{
        value: name for name, value in SCALE_VARIANTS.items()
    }}
    try:
        return mapping[(root_multiplier, local_multiplier)]
    except KeyError as error:
        raise ValueError(
            "ProtoMotions campaign received an unregistered scale multiplier pair"
        ) from error


def _expected_implementation_hashes(root: Path) -> dict[str, str]:
    return {
        "wrapper": sha256_file(
            root / "src/retargeting_comparison/protomotions_v3_worker.py"
        ),
        "native_worker": sha256_file(
            root / "src/retargeting_comparison/protomotions_v3_native.py"
        ),
        "adapter": sha256_file(
            root / "src/retargeting_comparison/protomotions_v3.py"
        ),
        "source_adapter": sha256_file(
            root / "src/retargeting_comparison/source_adapter_audit.py"
        ),
        "campaign": sha256_file(
            root / "src/retargeting_comparison/protomotions_v3_campaign.py"
        ),
        "official_solver": sha256_file(
            root
            / "external/ProtoMotions/pyroki/batch_retarget_to_g1_from_keypoints.py"
        ),
    }


@dataclass(frozen=True)
class CampaignJob:
    name: str
    output: Path
    work_dir: Path
    evidence: Path
    warmup_runs: int
    measured_runs: int
    target_raw_frames: int = TARGET_RAW_FRAMES
    device: str = "cuda"
    root_multiplier: float = 1.0
    local_multiplier: float = 1.0


def campaign_jobs(root: Path, sequence_id: str) -> tuple[CampaignJob, ...]:
    method_root = root / "runs" / sequence_id / FORMAL_RUN_DIRECTORY
    formal = CampaignJob(
        name="formal_timing",
        output=method_root / "formal/canonical_g1.npz",
        work_dir=method_root / "formal/work",
        evidence=method_root / "formal/evidence.json",
        warmup_runs=1,
        measured_runs=3,
    )
    variants = []
    for name, (root_multiplier, local_multiplier) in SCALE_VARIANTS.items():
        run_dir = (
            root
            / "runs"
            / sequence_id
            / "scale-sensitivity/native-response/protomotions_v3"
            / name
        )
        variants.append(
            CampaignJob(
                name=name,
                output=run_dir / "canonical_g1.npz",
                work_dir=run_dir / "work",
                evidence=run_dir / "evidence.json",
                warmup_runs=0,
                measured_runs=1,
                root_multiplier=root_multiplier,
                local_multiplier=local_multiplier,
            )
        )
    return (formal, *variants)


def _valid_output(
    path: Path,
    human: CanonicalHuman,
    *,
    root: Path,
    job: CampaignJob,
    source_path: Path,
    expected_warmup: int | None = None,
    expected_measured: int | None = None,
) -> bool:
    if not path.is_file():
        return False
    try:
        motion = CanonicalG1.load(path)
        motion.validate(source_frame_count=len(human.timestamps))
    except Exception:
        return False
    expected_variant = _variant_name(job.root_multiplier, job.local_multiplier)
    from .scale_worker import scale_execution_contract

    if not (
        len(motion.qpos) == job.target_raw_frames
        and motion.metadata.get("completion_status") == "incomplete"
        and motion.metadata.get("full_source_completion_status") == "incomplete"
        and motion.metadata.get("full_source_completion_ratio")
        == job.target_raw_frames / len(human.timestamps)
        and motion.metadata.get("native_contract_completion_status") == "succeeded"
        and motion.metadata.get("native_contract_completion_ratio") == 1.0
        and motion.metadata.get("native_contract_frame_count")
        == job.target_raw_frames
        and np.array_equal(
            motion.source_frame_idx,
            np.arange(job.target_raw_frames, dtype=np.int64),
        )
        and bool(np.all(motion.valid))
        and motion.metadata.get("formal_timing_boundary")
        == "canonical_source_file_to_canonical_g1_in_memory"
        and isinstance(motion.metadata.get("implementation_hashes"), dict)
        and motion.metadata.get("method") == "protomotions_v3"
        and motion.metadata.get("upstream_commit")
        == "49fe5ad69de67ebbc07ea2b25d41b0f622c15c3c"
        and motion.metadata.get("canonical_source_sha256") == human.source_sha256
        and motion.metadata.get("canonical_source_file_sha256")
        == sha256_file(source_path)
        and motion.metadata.get("config_sha256")
        == sha256_file(root / "configs/protomotions_v3.yaml")
        and motion.metadata.get("scale_protocol_sha256")
        == sha256_file(root / "configs/scale_policy_sensitivity.yaml")
        and motion.metadata.get("scale_variant") == expected_variant
        and motion.metadata.get("root_scale_multiplier") == job.root_multiplier
        and motion.metadata.get("local_scale_multiplier") == job.local_multiplier
        and motion.metadata.get("experiment_role")
        == "native_response_fixed_canonical_contact"
        and motion.metadata.get("fixed_contact_labels") is True
        and motion.metadata.get("implementation_hashes")
        == _expected_implementation_hashes(root)
        and motion.metadata.get("scale_execution_contract")
        == scale_execution_contract(root, "protomotions_v3")
        and isinstance(motion.metadata.get("scale_runtime_environment"), dict)
        and motion.metadata["scale_runtime_environment"].get("environment_name")
        == "egoallo"
    ):
        return False
    protocol = motion.metadata.get("timing_protocol")
    if not isinstance(protocol, dict):
        return False
    repetitions = protocol.get("repetitions")
    if not isinstance(repetitions, list):
        return False
    if expected_warmup is not None:
        warmup = [item for item in repetitions if item.get("role") == "warmup"]
        if len(warmup) != expected_warmup:
            return False
    if expected_measured is not None:
        measured = [item for item in repetitions if item.get("role") == "measured"]
        if len(measured) != expected_measured:
            return False
    return all(
        item.get("timing_boundary")
        == "canonical_source_file_to_canonical_g1_in_memory"
        and item.get("native_boundary")
        == "native_input_ready_to_native_g1_in_memory"
        and item.get("thread_limit") == 1
        and isinstance(item.get("cpu_affinity"), list)
        and len(item["cpu_affinity"]) == 1
        and isinstance(item.get("timing_artifact_sha256"), str)
        and len(item["timing_artifact_sha256"]) == 64
        for item in repetitions
    )


def _valid_evidence(
    path: Path,
    *,
    root: Path,
    job: CampaignJob,
    human: CanonicalHuman,
    source_path: Path,
) -> bool:
    if not path.is_file() or not job.output.is_file():
        return False
    try:
        value = _evidence(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    from .scale_worker import scale_execution_contract

    canonical = Path(str(value.get("canonical_output", ""))).resolve()
    return bool(
        value.get("status") == "succeeded"
        and canonical == job.output.resolve()
        and value.get("canonical_output_sha256") == sha256_file(job.output)
        and value.get("scale_variant")
        == _variant_name(job.root_multiplier, job.local_multiplier)
        and value.get("root_scale_multiplier") == job.root_multiplier
        and value.get("local_scale_multiplier") == job.local_multiplier
        and value.get("canonical_source_file_sha256") == sha256_file(source_path)
        and value.get("canonical_source_embedded_sha256") == human.source_sha256
        and value.get("config_sha256")
        == sha256_file(root / "configs/protomotions_v3.yaml")
        and value.get("scale_protocol_sha256")
        == sha256_file(root / "configs/scale_policy_sensitivity.yaml")
        and value.get("implementation_hashes")
        == _expected_implementation_hashes(root)
        and value.get("scale_execution_contract")
        == scale_execution_contract(root, "protomotions_v3")
        and isinstance(value.get("scale_runtime_environment"), dict)
        and value["scale_runtime_environment"].get("environment_name")
        == "egoallo"
    )


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _worker_command(
    *,
    job: CampaignJob,
    root: Path,
    native_python: Path,
    source: Path,
    keypoints: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "retargeting_comparison.protomotions_v3_worker",
        "--repo-root",
        str(root),
        "--python",
        str(native_python),
        "--source",
        str(source),
        "--keypoints",
        str(keypoints),
        "--output",
        str(job.output),
        "--work-dir",
        str(job.work_dir),
        "--evidence-json",
        str(job.evidence),
        "--target-raw-frames",
        str(job.target_raw_frames),
        "--warmup-runs",
        str(job.warmup_runs),
        "--measured-runs",
        str(job.measured_runs),
        "--device",
        job.device,
        "--root-scale-multiplier",
        str(job.root_multiplier),
        "--local-scale-multiplier",
        str(job.local_multiplier),
    ]
    if job.name == "independent_cold":
        command.append("--capture-runtime-witness")
    return command


def run_protomotions_v3_smoke(
    repo_root: str | Path = ".",
    *,
    native_python: str | Path | None = None,
) -> dict[str, Any]:
    """Run the dedicated two-frame v3 worker smoke and publish a receipt."""

    from .io_utils import load_yaml
    from .native_target_capture import tensor_sha256
    from .runner import _conda_python
    from .stage1_timing_campaign import CampaignLock, _proto_provenance_receipt

    root = Path(repo_root).resolve()
    sequence = load_yaml(root / "manifests/pilot_sequence.yaml")
    sequence_id = str(sequence["sequence_id"])
    source = root / str(sequence["canonical_path"])
    keypoints = (
        root / "source_adapters/protomotions_v3" / sequence_id / "keypoints.npy"
    )
    run_dir = root / "runs" / sequence_id / "protomotions-v3-v3-smoke2"
    job = CampaignJob(
        name="smoke2",
        output=run_dir / "canonical_g1.npz",
        work_dir=run_dir / "work",
        evidence=run_dir / "evidence.json",
        warmup_runs=0,
        measured_runs=1,
        target_raw_frames=2,
    )
    python = (
        Path(native_python).resolve()
        if native_python is not None
        else _conda_python("egoallo")
    )
    command = _worker_command(
        job=job,
        root=root,
        native_python=python,
        source=source,
        keypoints=keypoints,
    )
    before = _proto_provenance_receipt(root)
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    worker_env = os.environ.copy()
    with CampaignLock(root / "runs/.stage1_formal_timing.lock") as lock:
        if lock.token is None:
            raise RuntimeError("ProtoMotions smoke could not acquire a campaign token")
        worker_env["RTCMP_FORMAL_CAMPAIGN_TOKEN"] = lock.token
        result = subprocess.run(command, cwd=root, env=worker_env, check=False)
    process_wall = time.perf_counter() - started
    after = _proto_provenance_receipt(root)
    if before != after:
        raise RuntimeError("ProtoMotions implementation changed during smoke run")
    if result.returncode != 0 or not job.output.is_file() or not job.evidence.is_file():
        raise RuntimeError("ProtoMotions two-frame smoke worker failed")
    motion = CanonicalG1.load(job.output)
    motion.validate(source_frame_count=int(sequence["num_frames"]))
    if len(motion.qpos) != 2 or motion.metadata.get("completion_status") != "incomplete":
        raise RuntimeError("ProtoMotions smoke output is not the requested two frames")
    receipt = {
        "schema_version": 1,
        "status": "succeeded",
        "method": "protomotions-v3",
        "frames": 2,
        "solver_invoked": True,
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "process_wall_time_s": process_wall,
        "command": command,
        "command_sha256": hashlib.sha256(
            json.dumps(command, separators=(",", ":")).encode()
        ).hexdigest(),
        "output_path": str(job.output.relative_to(root)),
        "output_sha256": sha256_file(job.output),
        "qpos_sha256": tensor_sha256(np.asarray(motion.qpos, dtype=np.float64)),
        "evidence_path": str(job.evidence.relative_to(root)),
        "evidence_sha256": sha256_file(job.evidence),
        "source_path": str(source.relative_to(root)),
        "source_sha256": sha256_file(source),
        "source_content_sha256": motion.metadata.get("canonical_source_sha256"),
        "implementation_receipt": before,
        "implementation_sha256": before["aggregate_sha256"],
    }
    receipt_path = run_dir / "smoke_receipt.json"
    atomic_write_json(receipt_path, receipt)
    receipt["receipt_path"] = str(receipt_path.relative_to(root))
    receipt["receipt_sha256"] = sha256_file(receipt_path)
    return receipt


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    motion = CanonicalG1.load(path)
    timing = motion.metadata.get("timing_protocol", {})
    repetitions = timing.get("repetitions", [])
    return {
        "path": str(path.relative_to(root)),
        "sha256": sha256_file(path),
        "frames": len(motion.qpos),
        "fps": motion.fps,
        "completion_status": motion.metadata.get("completion_status"),
        "device": motion.metadata.get("device"),
        "scale_variant": motion.metadata.get("scale_variant"),
        "repetitions": repetitions,
        "measured_wall_time_s": [
            float(item["wall_time_s"])
            for item in repetitions
            if item.get("role") == "measured"
        ],
        "warmup_wall_time_s": [
            float(item["wall_time_s"])
            for item in repetitions
            if item.get("role") == "warmup"
        ],
    }


def _evidence(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Invalid evidence mapping: {path}")
    return value


def _standard_timing(
    *,
    cold_output: Path,
    cold_evidence: Path,
    formal_output: Path,
    formal_evidence: Path,
    sequence_id: str,
) -> dict[str, Any]:
    cold_motion = CanonicalG1.load(cold_output)
    formal_motion = CanonicalG1.load(formal_output)
    cold_protocol = cold_motion.metadata["timing_protocol"]
    formal_protocol = formal_motion.metadata["timing_protocol"]
    cold_repetitions = list(cold_protocol["repetitions"])
    formal_repetitions = list(formal_protocol["repetitions"])
    cold = [item for item in cold_repetitions if item["role"] == "measured"]
    warmup = [item for item in formal_repetitions if item["role"] == "warmup"]
    measured = [item for item in formal_repetitions if item["role"] == "measured"]
    if len(cold) != 1 or len(warmup) != 1 or len(measured) != 3:
        raise ValueError("ProtoMotions timing does not satisfy independent cold + 1/3 protocol")
    repetitions = cold + warmup + measured
    qpos_hashes = [item.get("canonical_qpos_sha256") for item in repetitions]
    content_hashes = [
        item.get("canonical_g1_content_sha256") for item in repetitions
    ]
    if (
        any(not isinstance(value, str) or len(value) != 64 for value in qpos_hashes)
        or len(set(qpos_hashes)) != 1
        or any(
            not isinstance(value, str) or len(value) != 64
            for value in content_hashes
        )
        or len(set(content_hashes)) != 1
    ):
        raise ValueError("ProtoMotions cold/warm canonical qpos is not deterministic")
    if cold[0].get("runtime_witness_enabled") is not True or any(
        item.get("runtime_witness_enabled") is not False
        for item in warmup + measured
    ):
        raise ValueError("ProtoMotions runtime capture must be cold-only")
    cold_entry = dict(cold[0])
    cold_entry["timing_artifact_path"] = str(cold_output)
    cold_entry["timing_artifact_sha256"] = sha256_file(cold_output)
    if cold_entry.get("timing_boundary") != (
        "canonical_source_file_to_canonical_g1_in_memory"
    ):
        raise ValueError("ProtoMotions cold run uses an obsolete timing boundary")
    if cold_entry.get("native_boundary") != (
        "native_input_ready_to_native_g1_in_memory"
    ):
        raise ValueError("ProtoMotions cold run lacks the native in-memory boundary")
    duration = len(formal_motion.qpos) / formal_motion.fps
    raw_rtf = [
        float(item.get("steady_end_to_end_total_s", item["wall_time_s"])) / duration
        for item in measured
    ]
    native_rtf = [
        float(item.get("native_total_s", item.get("native_core_s", item["wall_time_s"])))
        / duration
        for item in measured
    ]
    cold_info = _evidence(cold_evidence)
    formal_info = _evidence(formal_evidence)
    return {
        "run_id": f"{sequence_id}__protomotions-v3",
        "method": "protomotions-v3",
        "protocol": {
            "cold_processes": 1,
            "warmup_runs": 1,
            "measured_warm_runs": 3,
            "threads": 1,
            "cpu_affinity": measured[0].get("cpu_affinity"),
            "visualization": False,
            "device": formal_motion.metadata.get("device"),
            "trajectory_scope": (
                "official fixed 450-frame whole trajectory; source frames 0:450"
            ),
            "frozen_source_frames": 600,
            "official_native_contract_frames": TARGET_RAW_FRAMES,
            "full_source_coverage_ratio": TARGET_RAW_FRAMES / 600,
            "end_to_end_boundary": "canonical_source_file_to_canonical_g1_in_memory",
            "native_core_boundary": "native_input_ready_to_native_g1_in_memory",
            "intermediate_and_final_artifact_writes_excluded": True,
            "output_hash_required_per_repetition": True,
            "canonical_qpos_hash_required_per_repetition": True,
            "cold_warm_qpos_determinism_required": True,
            "runtime_witness_policy": (
                "independent cold solver observation; disabled for warmup/measured"
            ),
        },
        "cold": cold_entry,
        "warmup": warmup,
        "measured": measured,
        "cold_process_wall_s": float(cold_info["process_wall_time_s"]),
        "warm_process_wall_s": float(formal_info["process_wall_time_s"]),
        "end_to_end_rtf_raw": raw_rtf,
        "end_to_end_rtf_median": float(np.median(raw_rtf)),
        "native_core_rtf_raw": native_rtf,
        "native_core_rtf_median": float(np.median(native_rtf)),
        "initialization_time_s": {
            "cold": float(cold_protocol.get("initialization_time_s", 0.0)),
            "formal_process": float(formal_protocol.get("initialization_time_s", 0.0)),
        },
        "timing_granularity": "whole trajectory",
        "per_frame_timing": (
            "canonical per-frame values are the measured whole-trajectory median "
            "amortized uniformly; no independent frame solve was observed"
        ),
        "raw_timing_artifacts": {
            "cold": cold_motion.metadata["timing_protocol"],
            "formal": formal_motion.metadata["timing_protocol"],
        },
        "canonical_qpos_sha256": qpos_hashes[0],
        "canonical_g1_content_sha256": content_hashes[0],
        "all_cold_warm_qpos_identical": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--native-python", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--keypoints", required=True)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--cold-output", required=True)
    parser.add_argument("--cold-evidence", required=True)
    parser.add_argument(
        "--run-fresh-cold",
        action="store_true",
        help="create the independent cold witness in a new subprocess; refuse reuse",
    )
    parser.add_argument(
        "--timing-only",
        action="store_true",
        help="stop after formal timing; registered scale jobs may run later",
    )
    parser.add_argument(
        "--summary-json", default="manifests/protomotions_v3_campaign.json"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.repo_root).resolve()
    native_python = Path(args.native_python).resolve()
    source = Path(args.source).resolve()
    keypoints = Path(args.keypoints).resolve()
    cold_output = Path(args.cold_output).resolve()
    cold_evidence = Path(args.cold_evidence).resolve()
    summary_path = (root / args.summary_json).resolve()
    human = CanonicalHuman.load(source)
    source_frames = len(human.timestamps)
    if source_frames != 600:
        raise SystemExit("ProtoMotions v3 campaign requires the frozen 600-frame Pilot")
    campaign_started_at = datetime.now(timezone.utc).isoformat()
    fresh_cold_entry: dict[str, Any] | None = None
    cold_job = CampaignJob(
        name="independent_cold",
        output=cold_output,
        work_dir=cold_output.parent / "work",
        evidence=cold_evidence,
        warmup_runs=0,
        measured_runs=1,
    )
    if args.run_fresh_cold:
        if cold_output.exists() or cold_evidence.exists():
            raise SystemExit(
                "--run-fresh-cold refuses to reuse an existing cold output/evidence"
            )
        cold_command = _worker_command(
            job=cold_job,
            root=root,
            native_python=native_python,
            source=source,
            keypoints=keypoints,
        )
        cold_started_at = datetime.now(timezone.utc).isoformat()
        cold_started = time.perf_counter()
        cold_result = subprocess.run(
            cold_command, cwd=root, env=os.environ.copy(), check=False
        )
        cold_finished_at = datetime.now(timezone.utc).isoformat()
        if cold_result.returncode != 0:
            raise SystemExit(cold_result.returncode)
        fresh_cold_entry = {
            "status": "succeeded",
            "command": cold_command,
            "started_at_utc": cold_started_at,
            "finished_at_utc": cold_finished_at,
            "process_wall_time_s": time.perf_counter() - cold_started,
            "returncode": cold_result.returncode,
        }
    if not _valid_output(
        cold_output,
        human,
        root=root,
        job=cold_job,
        source_path=source,
        expected_warmup=0,
        expected_measured=1,
    ) or not _valid_evidence(
        cold_evidence,
        root=root,
        job=cold_job,
        human=human,
        source_path=source,
    ):
        raise SystemExit("Independent CUDA cold output is missing or invalid")

    cold_snapshot = (
        root / "runs" / args.sequence_id / FORMAL_RUN_DIRECTORY / "cold/evidence.json"
    )
    cold_snapshot.parent.mkdir(parents=True, exist_ok=True)
    if cold_evidence != cold_snapshot:
        shutil.copy2(cold_evidence, cold_snapshot)
    summary: dict[str, Any] = {
        "method": "ProtoMotions v3 / modified PyRoki",
        "status": "running",
        "sequence_id": args.sequence_id,
        "source_frames": source_frames,
        "device": "cuda",
        "started_at_utc": campaign_started_at,
        "independent_cold": _artifact(cold_output, root),
        "independent_cold_evidence": str(cold_snapshot.relative_to(root)),
        "fresh_cold_attempt": fresh_cold_entry,
        "timing_only": bool(args.timing_only),
        "formal_timing": None,
        "scale_variants": {
            "native": {
                "reused_from": "formal_timing",
                "root_scale_multiplier": 1.0,
                "local_scale_multiplier": 1.0,
            }
        },
        "jobs": [],
        "superseded_600_frame_gpu_hardware_evidence": {
            "status": "failed",
            "reason": "XLA kernel requested 131072 shared-memory bytes; RTX 3090 Ti exposed 101376",
            "stderr_log": (
                f"runs/{args.sequence_id}/protomotions-v3/logs/native.cuda-600f.stderr.log"
            ),
            "substitute_output_used": False,
            "interpretation": (
                "Benchmark protocol deviation: 600 overrides the upstream fixed-450 "
                "contract and is excluded from the official operating point."
            ),
        },
    }
    atomic_write_json(summary_path, summary)
    jobs = campaign_jobs(root, args.sequence_id)
    if args.timing_only:
        jobs = jobs[:1]
    for job in jobs:
        command = _worker_command(
            job=job,
            root=root,
            native_python=native_python,
            source=source,
            keypoints=keypoints,
        )
        entry = {
            "name": job.name,
            "status": "running",
            "command": command,
            "output": str(job.output.relative_to(root)),
            "evidence": str(job.evidence.relative_to(root)),
        }
        summary["jobs"].append(entry)
        atomic_write_json(summary_path, summary)
        if args.run_fresh_cold and (job.output.exists() or job.evidence.exists()):
            raise SystemExit(
                f"Fresh timing campaign refuses pre-existing {job.name} artifacts"
            )
        if _valid_output(
            job.output,
            human,
            root=root,
            job=job,
            source_path=source,
            expected_warmup=job.warmup_runs,
            expected_measured=job.measured_runs,
        ) and _valid_evidence(
            job.evidence,
            root=root,
            job=job,
            human=human,
            source_path=source,
        ):
            entry["status"] = "reused"
        else:
            result = subprocess.run(command, cwd=root, env=os.environ.copy(), check=False)
            entry["returncode"] = result.returncode
            if result.returncode != 0 or not _valid_output(
                job.output,
                human,
                root=root,
                job=job,
                source_path=source,
                expected_warmup=job.warmup_runs,
                expected_measured=job.measured_runs,
            ) or not _valid_evidence(
                job.evidence,
                root=root,
                job=job,
                human=human,
                source_path=source,
            ):
                entry["status"] = "failed"
                summary["status"] = "failed"
                atomic_write_json(summary_path, summary)
                return result.returncode or 2
            entry["status"] = "succeeded"
        artifact = _artifact(job.output, root)
        entry["artifact"] = artifact
        entry["evidence_sha256"] = sha256_file(job.evidence)
        if job.name == "formal_timing":
            summary["formal_timing"] = artifact
            summary["scale_variants"]["native"]["artifact"] = artifact
        else:
            summary["scale_variants"][job.name] = artifact
        atomic_write_json(summary_path, summary)

    formal_job = jobs[0]
    formal = formal_job.output
    final_output = (
        root / "runs" / args.sequence_id / FORMAL_RUN_DIRECTORY / "canonical_g1.npz"
    )
    # Bind the cold-only runtime target observation to the deterministic warm
    # trajectory after every timed boundary.  This changes provenance metadata,
    # never qpos, and keeps capture hashing out of measured warm repetitions.
    formal_motion = CanonicalG1.load(formal)
    cold_motion = CanonicalG1.load(cold_output)
    from .native_target_capture import tensor_sha256

    formal_qpos_sha256 = tensor_sha256(
        np.asarray(formal_motion.qpos, dtype=np.float64)
    )
    cold_qpos_sha256 = tensor_sha256(
        np.asarray(cold_motion.qpos, dtype=np.float64)
    )
    if formal_qpos_sha256 != cold_qpos_sha256:
        raise RuntimeError("ProtoMotions independent cold changed canonical qpos")
    capture = cold_motion.metadata.get("runtime_pre_solver_capture")
    if not isinstance(capture, dict):
        raise RuntimeError("ProtoMotions independent cold capture is missing")
    capture = dict(capture)
    capture.update(
        {
            "witness_execution_role": "independent_cold_process",
            "witness_output_path": str(cold_output.relative_to(root)),
            "witness_output_sha256": sha256_file(cold_output),
            "witness_qpos_sha256": cold_qpos_sha256,
            "timed_warm_qpos_sha256": formal_qpos_sha256,
            "witness_and_timed_qpos_identical": True,
            "capture_overhead_excluded_from_measured_warm_runs": True,
        }
    )
    formal_motion.metadata["runtime_pre_solver_capture"] = capture
    formal_motion.metadata["runtime_witness_binding"] = {
        "cold_output_sha256": sha256_file(cold_output),
        "cold_qpos_sha256": cold_qpos_sha256,
        "warm_qpos_sha256": formal_qpos_sha256,
        "identical": True,
    }
    formal_motion.save(final_output, source_frame_count=source_frames)
    summary["canonical_output"] = _artifact(final_output, root)

    native_scale_output = (
        root
        / "runs"
        / args.sequence_id
        / "scale-sensitivity/native-response/protomotions_v3/native"
        / "canonical_g1.npz"
    )
    if native_scale_output.is_file() and sha256_file(native_scale_output) != sha256_file(
        final_output
    ):
        archive = native_scale_output.with_name(
            "canonical_g1.pre-formal-binding-"
            f"{sha256_file(native_scale_output)[:12]}.npz"
        )
        if not archive.exists():
            _atomic_copy(native_scale_output, archive)
    _atomic_copy(final_output, native_scale_output)
    if not _valid_output(
        native_scale_output,
        human,
        root=root,
        job=formal_job,
        source_path=source,
        expected_warmup=formal_job.warmup_runs,
        expected_measured=formal_job.measured_runs,
    ):
        raise RuntimeError("Materialized native scale point differs from formal output")
    native_binding = native_scale_output.with_name("formal_output_binding.json")
    atomic_write_json(
        native_binding,
        {
            "schema_version": 1,
            "binding_type": "byte_identical_published_formal_output_materialization",
            "solver_invoked_for_materialization": False,
            "scale_variant": "native",
            "root_scale_multiplier": 1.0,
            "local_scale_multiplier": 1.0,
            "formal_output_path": str(final_output.relative_to(root)),
            "formal_output_sha256": sha256_file(final_output),
            "formal_evidence_path": str(formal_job.evidence.relative_to(root)),
            "formal_evidence_sha256": sha256_file(formal_job.evidence),
            "native_scale_output_path": str(native_scale_output.relative_to(root)),
            "native_scale_output_sha256": sha256_file(native_scale_output),
            "canonical_source_sha256": human.source_sha256,
            "canonical_source_file_sha256": sha256_file(source),
            "config_sha256": sha256_file(root / "configs/protomotions_v3.yaml"),
            "scale_protocol_sha256": sha256_file(
                root / "configs/scale_policy_sensitivity.yaml"
            ),
            "implementation_hashes": _expected_implementation_hashes(root),
        },
    )
    summary["scale_variants"]["native"] = {
        **_artifact(native_scale_output, root),
        "reused_from": str(final_output.relative_to(root)),
        "binding_path": str(native_binding.relative_to(root)),
        "binding_sha256": sha256_file(native_binding),
        "additional_solver_invocation": False,
    }
    summary["status"] = "succeeded"
    summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    summary["all_five_native_scale_variants_complete"] = bool(
        not args.timing_only
        and set(summary["scale_variants"]) == {"native", *SCALE_VARIANTS}
    )
    timing_path = (
        root / "runs" / args.sequence_id / FORMAL_RUN_DIRECTORY / "timing.json"
    )
    timing = _standard_timing(
        cold_output=cold_output,
        cold_evidence=cold_snapshot,
        formal_output=formal,
        formal_evidence=jobs[0].evidence,
        sequence_id=args.sequence_id,
    )
    atomic_write_json(timing_path, timing)
    summary["standard_timing"] = str(timing_path.relative_to(root))

    repo_commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    formal_evidence = _evidence(jobs[0].evidence)
    manifest_path = (
        root / "manifests" / "runs" / f"{args.sequence_id}__protomotions-v3.json"
    )
    manifest = RunManifest(
        run_id=f"{args.sequence_id}__protomotions-v3",
        method="protomotions-v3",
        status=RunStatus.SUCCEEDED,
        command=summary["jobs"][0]["command"],
        environment="conda:egoallo (capture orchestrator)",
        repo_commit=repo_commit,
        config_sha256=sha256_file(root / "configs/protomotions_v3.yaml"),
        device=(
            "NVIDIA RTX 3090 Ti CUDA with CPU callback backend; "
            "host threads=1; whole-trajectory JAXLS"
        ),
        started_at=summary["started_at_utc"],
        finished_at=summary["finished_at_utc"],
        wall_time_s=float(timing["cold_process_wall_s"] + timing["warm_process_wall_s"]),
        exit_code=0,
        stdout_log=str(
            (
                jobs[0].work_dir
                / f"logs/native.{jobs[0].device}-{jobs[0].target_raw_frames}f.stdout.log"
            ).relative_to(root)
        ),
        stderr_log=str(
            (
                jobs[0].work_dir
                / f"logs/native.{jobs[0].device}-{jobs[0].target_raw_frames}f.stderr.log"
            ).relative_to(root)
        ),
        output_path=str(final_output.relative_to(root)),
        output_sha256=sha256_file(final_output),
        source_sha256=human.source_sha256,
        timing_path=str(timing_path.relative_to(root)),
        message=(
            "Official fixed-450 whole-trajectory solver completed on CUDA. The prior "
            "600-frame CUDA/CPU benchmark intervention is excluded and retained only "
            "as protocol-deviation evidence."
        ),
    )
    manifest.save(manifest_path)
    summary["standard_run_manifest"] = str(manifest_path.relative_to(root))
    summary["formal_evidence_sha256"] = sha256_file(jobs[0].evidence)
    summary["formal_process_wall_time_s"] = formal_evidence["process_wall_time_s"]
    atomic_write_json(summary_path, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
