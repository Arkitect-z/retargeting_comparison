"""Resumable, provenance-preserving method subprocess orchestration."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import atomic_write_json, load_yaml, sha256_file
from .schemas import CanonicalG1, RunManifest, RunStatus


METHOD_ENVIRONMENTS = {
    "sparse": "capture",
    "dense": "capture",
    "gmr": "robot",
    "omniretarget": "hsretargeting",
    "holosoma": "hsretargeting",
}


def _variant_label(method: str, seed: str, revision: str | None = None) -> str:
    label = f"sparse-{seed.lower()}" if method == "sparse" else method
    return f"{label}-{revision}" if revision else label


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _conda_python(environment: str) -> Path:
    override = os.environ.get(f"RTCMP_{environment.upper()}_PYTHON")
    if override:
        path = Path(override)
        if path.is_file():
            return path
        raise FileNotFoundError(f"Configured Python executable does not exist: {path}")
    candidates = (
        Path.home() / "anaconda3" / "envs" / environment / "bin" / "python",
        Path.home() / "miniconda3" / "envs" / environment / "bin" / "python",
    )
    for path in candidates:
        if path.is_file():
            return path
    if environment == os.environ.get("CONDA_DEFAULT_ENV"):
        return Path(sys.executable)
    raise FileNotFoundError(
        f"Cannot locate conda environment {environment!r}; set RTCMP_{environment.upper()}_PYTHON"
    )


def _aggregate_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.as_posix().encode())
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _method_config_hash(root: Path, method: str) -> str:
    if method in {"sparse", "dense"}:
        return _aggregate_sha256(
            [
                root / "configs" / "controlled_mink.yaml",
                root / "manifests" / "evaluator.yaml",
            ]
        )
    if method == "gmr":
        return _aggregate_sha256(
            [
                root
                / "external"
                / "GMR"
                / "general_motion_retargeting"
                / "ik_configs"
                / "bvh_lafan1_to_g1.json",
                root / "external" / "GMR" / "assets" / "unitree_g1" / "g1_mocap_29dof.xml",
            ]
        )
    return _aggregate_sha256(
        [
            root
            / "external"
            / "holosoma"
            / "src"
            / "holosoma_retargeting"
            / "holosoma_retargeting"
            / "config_types"
            / "retargeter.py",
            root / "patches" / "holosoma" / "interaction-hard-constraint-flags.patch",
        ]
    )


def _run_worker(
    command: list[str],
    root: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> tuple[int, float]:
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "MPLCONFIGDIR": "/tmp/rtcmp-mpl",
        }
    )
    source_path = str(root / "src")
    env["PYTHONPATH"] = source_path + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        result = subprocess.run(
            command,
            cwd=root,
            env=env,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    return result.returncode, time.perf_counter() - start


def _worker_command(
    python: Path,
    method: str,
    root: Path,
    canonical_source: Path,
    native_source: Path,
    output: Path,
    work_dir: Path,
    timing_json: Path,
    seed: str,
    warmup_runs: int,
    measured_runs: int,
    max_frames: int | None,
) -> list[str]:
    command = [
        str(python),
        "-m",
        "retargeting_comparison.method_worker",
        "--method",
        method,
        "--repo-root",
        str(root),
        "--source",
        str(canonical_source),
        "--output",
        str(output),
        "--work-dir",
        str(work_dir),
        "--timing-json",
        str(timing_json),
        "--seed",
        seed,
        "--warmup-runs",
        str(warmup_runs),
        "--measured-runs",
        str(measured_runs),
    ]
    if method == "gmr":
        command.extend(["--native-source", str(native_source)])
    if max_frames is not None:
        command.extend(["--max-frames", str(max_frames)])
    return command


def _timing_summary(
    cold_worker: dict[str, Any],
    warm_worker: dict[str, Any],
    cold_process_wall_s: float,
    warm_process_wall_s: float,
    fps: float,
) -> dict[str, Any]:
    cold = cold_worker["repetitions"][0]
    warm = [item for item in warm_worker["repetitions"] if item["role"] == "measured"]
    warmup = [item for item in warm_worker["repetitions"] if item["role"] == "warmup"]
    e2e_rtf = [
        item.get("steady_end_to_end_total_s", item["wall_time_s"])
        / (item["frame_count"] / fps)
        for item in warm
    ]
    native_rtf = [item["native_total_s"] / (item["frame_count"] / fps) for item in warm]
    return {
        "protocol": {
            "cold_processes": 1,
            "warmup_runs": len(warmup),
            "measured_warm_runs": len(warm),
            "threads": 1,
            "visualization": False,
        },
        "cold": cold,
        "cold_process_wall_s": cold_process_wall_s,
        "warmup": warmup,
        "measured_warm": warm,
        "warm_process_wall_s": warm_process_wall_s,
        "end_to_end_rtf_raw": e2e_rtf,
        "native_core_rtf_raw": native_rtf,
        "end_to_end_rtf_median": float(np.median(e2e_rtf)),
        "native_core_rtf_median": float(np.median(native_rtf)),
        "cold_import_startup_s": max(0.0, cold_process_wall_s - cold["wall_time_s"]),
        "initialization_and_adapter_s_cold": max(
            0.0, cold["wall_time_s"] - cold["native_total_s"]
        ),
        "initialization_time_s_raw": [item.get("initialization_time_s") for item in warm],
    }


def run_method(
    method: str,
    sequence_manifest: str | Path,
    repo_root: str | Path = ".",
    seed: str = "neutral",
    max_frames: int | None = None,
    revision: str | None = None,
) -> RunManifest:
    root = Path(repo_root).resolve()
    method = "omniretarget" if method == "holosoma" else method
    if method not in METHOD_ENVIRONMENTS:
        raise ValueError(f"Unknown method {method!r}")
    if method != "sparse" and seed != "neutral":
        raise ValueError("Only Sparse exposes the A/B initial-pose seeds")
    if revision and method not in {"sparse", "dense"}:
        raise ValueError("Run revisions are currently reserved for controlled baselines")
    sequence_path = Path(sequence_manifest)
    if not sequence_path.is_absolute():
        sequence_path = root / sequence_path
    sequence = load_yaml(sequence_path)
    if sequence.get("full_lafan_authorized") is not False:
        raise RuntimeError("Refusing a sequence manifest without the Stage 1 Full-LAFAN hard stop")
    canonical_source = root / sequence["canonical_path"]
    native_source = root / sequence["cropped_source_file"]
    label = _variant_label(method, seed, revision)
    if max_frames is not None:
        label += f"-smoke{max_frames}"
    run_id = f"{sequence['sequence_id']}__{label}"
    run_dir = root / "runs" / sequence["sequence_id"] / label
    manifest_path = root / "runs" / "manifests" / f"{run_id}.json"
    if manifest_path.exists():
        existing = RunManifest.load(manifest_path)
        if (
            existing.status in {RunStatus.SUCCEEDED, RunStatus.INCOMPLETE}
            and existing.output_path
            and Path(existing.output_path).is_file()
            and existing.output_sha256 == sha256_file(existing.output_path)
        ):
            return existing
        attempt = datetime.now(timezone.utc).strftime("attempt_%Y%m%dT%H%M%SZ")
        archived = manifest_path.with_name(f"{manifest_path.stem}__{attempt}.json")
        archived.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest_path, archived)
        run_dir = run_dir / attempt
    output = run_dir / "canonical_g1.npz"
    timing_path = run_dir / "timing.json"
    cold_output = run_dir / "cold" / "canonical_g1.npz"
    cold_timing = run_dir / "cold" / "worker_timing.json"
    warm_worker_timing = run_dir / "warm_worker_timing.json"
    logs = run_dir / "logs"
    environment = METHOD_ENVIRONMENTS[method]
    python = _conda_python(environment)
    base_command = _worker_command(
        python,
        method,
        root,
        canonical_source,
        native_source,
        output,
        run_dir / "work",
        warm_worker_timing,
        seed,
        1,
        3,
        max_frames,
    )
    manifest = RunManifest(
        run_id=run_id,
        method=label,
        status=RunStatus.RUNNING,
        command=base_command,
        environment=f"conda:{environment}",
        repo_commit=_git(root, "rev-parse", "HEAD"),
        config_sha256=_method_config_hash(root, method),
        device=f"CPU; {platform.machine()}; threads=1",
        started_at=utc_now(),
        stdout_log=str(logs / "warm.stdout.log"),
        stderr_log=str(logs / "warm.stderr.log"),
        output_path=str(output),
        source_sha256=sequence["cropped_sha256"],
        timing_path=str(timing_path),
    )
    manifest.save(manifest_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    overall_start = time.perf_counter()
    cold_command = _worker_command(
        python,
        method,
        root,
        canonical_source,
        native_source,
        cold_output,
        run_dir / "cold" / "work",
        cold_timing,
        seed,
        0,
        1,
        max_frames,
    )
    cold_code, cold_wall = _run_worker(
        cold_command, root, logs / "cold.stdout.log", logs / "cold.stderr.log"
    )
    if cold_code != 0:
        manifest.status = RunStatus.FAILED
        manifest.exit_code = cold_code
        manifest.finished_at = utc_now()
        manifest.wall_time_s = time.perf_counter() - overall_start
        manifest.message = "Cold process failed; inspect cold stderr log"
        manifest.save(manifest_path)
        return manifest
    warm_code, warm_wall = _run_worker(
        base_command, root, logs / "warm.stdout.log", logs / "warm.stderr.log"
    )
    manifest.exit_code = warm_code
    manifest.wall_time_s = time.perf_counter() - overall_start
    manifest.finished_at = utc_now()
    if warm_code != 0:
        manifest.status = RunStatus.FAILED
        manifest.message = "Warm timing process failed; inspect stderr log"
        manifest.save(manifest_path)
        return manifest
    motion = CanonicalG1.load(output)
    timing = _timing_summary(
        json.loads(cold_timing.read_text()),
        json.loads(warm_worker_timing.read_text()),
        cold_wall,
        warm_wall,
        motion.fps,
    )
    timing["method"] = label
    timing["run_id"] = run_id
    atomic_write_json(timing_path, timing)
    manifest.output_sha256 = sha256_file(output)
    manifest.status = RunStatus(motion.metadata["completion_status"])
    manifest.message = "Existing successful artifacts are immutable; reruns create a new attempt"
    manifest.save(manifest_path)
    return manifest


def refresh_method_timing(
    method: str,
    sequence_manifest: str | Path,
    repo_root: str | Path = ".",
    seed: str = "neutral",
    revision: str | None = None,
) -> Path:
    """Create a new immutable timing revision with initialization separated."""
    root = Path(repo_root).resolve()
    method = "omniretarget" if method == "holosoma" else method
    sequence_path = Path(sequence_manifest)
    if not sequence_path.is_absolute():
        sequence_path = root / sequence_path
    sequence = load_yaml(sequence_path)
    canonical_source = root / sequence["canonical_path"]
    native_source = root / sequence["cropped_source_file"]
    label = _variant_label(method, seed, revision)
    base_dir = root / "runs" / sequence["sequence_id"] / label
    revision = base_dir / "timing_refined"
    output_path = base_dir / "timing_refined.json"
    if output_path.is_file():
        return output_path
    python = _conda_python(METHOD_ENVIRONMENTS[method])
    cold_output = revision / "cold" / "canonical_g1.npz"
    warm_output = revision / "warm" / "canonical_g1.npz"
    cold_worker = revision / "cold" / "worker_timing.json"
    warm_worker = revision / "warm" / "worker_timing.json"
    cold_command = _worker_command(
        python, method, root, canonical_source, native_source, cold_output,
        revision / "cold" / "work", cold_worker, seed, 0, 1, None
    )
    warm_command = _worker_command(
        python, method, root, canonical_source, native_source, warm_output,
        revision / "warm" / "work", warm_worker, seed, 1, 3, None
    )
    cold_code, cold_wall = _run_worker(
        cold_command, root, revision / "logs" / "cold.stdout.log", revision / "logs" / "cold.stderr.log"
    )
    if cold_code:
        raise RuntimeError(f"Refined cold timing failed for {label}")
    warm_code, warm_wall = _run_worker(
        warm_command, root, revision / "logs" / "warm.stdout.log", revision / "logs" / "warm.stderr.log"
    )
    if warm_code:
        raise RuntimeError(f"Refined warm timing failed for {label}")
    motion = CanonicalG1.load(warm_output)
    quality_motion = CanonicalG1.load(base_dir / "canonical_g1.npz")
    if not np.allclose(motion.qpos, quality_motion.qpos, atol=1e-10, rtol=0.0):
        raise RuntimeError(f"Refined timing changed deterministic qpos for {label}")
    timing = _timing_summary(
        json.loads(cold_worker.read_text()),
        json.loads(warm_worker.read_text()),
        cold_wall,
        warm_wall,
        motion.fps,
    )
    timing.update(
        {
            "method": label,
            "timing_revision": "initialization-separated-v2",
            "repo_commit": _git(root, "rev-parse", "HEAD"),
            "quality_output_sha256": sha256_file(base_dir / "canonical_g1.npz"),
            "prior_timing_sha256": sha256_file(base_dir / "timing.json"),
            "quality_qpos_reproduced": True,
        }
    )
    atomic_write_json(output_path, timing)
    return output_path
