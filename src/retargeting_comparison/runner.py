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
    "protomotions_v2_3": "capture",
}


def _variant_label(method: str, seed: str, revision: str | None = None) -> str:
    if method == "sparse":
        label = f"sparse-{seed.lower()}"
    elif method == "protomotions_v2_3":
        label = "protomotions-v2.3"
    else:
        label = method
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


def _files_below(path: Path, suffixes: set[str] | None = None) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Provenance root is missing: {path}")
    return sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file()
        and "__pycache__" not in candidate.parts
        and (suffixes is None or candidate.suffix.lower() in suffixes)
    )


def _checkout_receipt(path: Path) -> dict[str, Any]:
    if not (path / ".git").exists():
        # Worktrees may expose a .git file rather than a directory.
        if not (path / ".git").is_file():
            raise FileNotFoundError(f"Frozen upstream checkout is missing: {path}")
    commit = _git(path, "rev-parse", "HEAD")
    status = _git(path, "status", "--porcelain=v1", "--untracked-files=all")
    return {
        "path": str(path),
        "commit": commit,
        "status_porcelain_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "clean": not bool(status),
    }


def _method_provenance_receipt(root: Path, method: str) -> dict[str, Any]:
    """Hash the executed harness, upstream tree, assets, and env history."""

    method = "omniretarget" if method == "holosoma" else method
    environment = METHOD_ENVIRONMENTS[method]
    python = _conda_python(environment)
    environment_root = python.parent.parent
    history = environment_root / "conda-meta" / "history"
    # Freeze the executable dependency closure, not unrelated reporting and
    # visualization modules.  This remains fail-closed for every file that can
    # affect a formal run while allowing reports to be generated afterwards
    # without invalidating an already published timing receipt.
    harness_names = {
        "constants.py",
        "io_utils.py",
        "method_worker.py",
        "native_target_capture.py",
        "rotations.py",
        "runner.py",
        "schemas.py",
        "stage1_timing_campaign.py",
    }
    if method in {"sparse", "dense"}:
        harness_names.update(
            {"calibration.py", "controlled_mink.py", "robot_model.py"}
        )
    elif method in {"gmr", "omniretarget"}:
        harness_names.add("method_adapters.py")
    elif method == "protomotions_v2_3":
        harness_names.add("protomotions_v2.py")
    harness_root = root / "src/retargeting_comparison"
    paths = [harness_root / name for name in sorted(harness_names)]
    paths.extend(
        [
            root / "configs/stage1_timing_campaign.yaml",
            root / "manifests/evaluator.yaml",
        ]
    )
    checkout: Path | None = None
    if method in {"sparse", "dense"}:
        paths.append(root / "configs/controlled_mink.yaml")
        paths.extend(
            _files_below(
                root
                / "external/holosoma/src/holosoma/holosoma/data/robots/g1",
                {".xml", ".urdf", ".obj", ".stl", ".dae"},
            )
        )
        checkout = root / "external/holosoma"
    elif method == "gmr":
        checkout = root / "external/GMR"
        paths.extend(
            _files_below(checkout / "general_motion_retargeting", {".py", ".json"})
        )
        paths.extend(
            _files_below(
                checkout / "assets/unitree_g1",
                {".xml", ".urdf", ".obj", ".stl", ".dae"},
            )
        )
    elif method == "protomotions_v2_3":
        checkout = root / "external/ProtoMotions-v2.3"
        paths.append(root / "configs/protomotions_v2.yaml")
        paths.extend(
            _files_below(
                checkout / "data/scripts/retargeting", {".py", ".yaml", ".json"}
            )
        )
        paths.extend(
            _files_below(
                checkout / "protomotions/data/assets/mjcf",
                {".xml", ".urdf", ".obj", ".stl", ".dae"},
            )
        )
    else:
        checkout = root / "external/holosoma"
        paths.append(root / "patches/holosoma/interaction-hard-constraint-flags.patch")
        paths.extend(
            _files_below(
                checkout / "src/holosoma_retargeting/holosoma_retargeting",
                {".py", ".yaml", ".json", ".xml", ".urdf", ".obj", ".stl", ".dae"},
            )
        )
    unique = sorted({path.resolve() for path in paths})
    records = [
        {
            "path": str(path.relative_to(root)),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in unique
    ]
    env_history = {
        "environment": environment,
        "python_path": str(python),
        "python_sha256": sha256_file(python),
        "conda_history_path": str(history),
        "conda_history_sha256": sha256_file(history),
    }
    upstream = _checkout_receipt(checkout) if checkout is not None else None
    stable = {
        "schema_version": 2,
        "method": method,
        "files": records,
        "environment": env_history,
        "upstream_checkout": upstream,
    }
    stable["aggregate_sha256"] = hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return stable


def _method_config_hash(root: Path, method: str) -> str:
    return str(_method_provenance_receipt(root, method)["aggregate_sha256"])


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
    *,
    capture_runtime_witness: bool = False,
    native_source_declared_sha256: str | None = None,
    native_source_declared_fps: float | None = None,
    native_source_declared_frames: int | None = None,
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
        if (
            native_source_declared_sha256 is None
            or native_source_declared_fps is None
            or native_source_declared_frames is None
        ):
            raise ValueError("GMR worker requires the complete declared BVH contract")
        command.extend(
            [
                "--native-source-declared-sha256",
                native_source_declared_sha256,
                "--native-source-declared-fps",
                str(native_source_declared_fps),
                "--native-source-declared-frames",
                str(native_source_declared_frames),
            ]
        )
    if capture_runtime_witness:
        command.append("--capture-runtime-witness")
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
    all_repetitions = [cold, *warmup, *warm]
    qpos_hashes = [item.get("canonical_qpos_sha256") for item in all_repetitions]
    content_hashes = [
        item.get("canonical_g1_content_sha256") for item in all_repetitions
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
        raise RuntimeError("Cold/warm canonical G1 content is not deterministic")
    return {
        "protocol": {
            "cold_processes": 1,
            "warmup_runs": len(warmup),
            "measured_warm_runs": len(warm),
            "threads": 1,
            "visualization": False,
            "cpu_affinity": warm[0].get("cpu_affinity") if warm else None,
            "end_to_end_boundary": "canonical_source_file_to_canonical_g1_in_memory",
            "native_core_boundary": "native_input_ready_to_native_g1_in_memory",
            "intermediate_and_final_artifact_writes_excluded": True,
            "canonical_artifact_write_excluded": True,
            "output_hash_required_per_repetition": True,
            "canonical_qpos_hash_required_per_repetition": True,
            "cold_warm_qpos_determinism_required": True,
            "runtime_witness_policy": (
                "independent cold solver observation; disabled for warmup/measured"
            ),
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
        "canonical_qpos_sha256": qpos_hashes[0],
        "canonical_g1_content_sha256": content_hashes[0],
        "all_cold_warm_qpos_identical": True,
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
    sequence_path = Path(sequence_manifest)
    if not sequence_path.is_absolute():
        sequence_path = root / sequence_path
    sequence = load_yaml(sequence_path)
    if sequence.get("full_lafan_authorized") is not False:
        raise RuntimeError("Refusing a sequence manifest without the Stage 1 Full-LAFAN hard stop")
    canonical_source = root / sequence["canonical_path"]
    native_source = root / sequence["cropped_source_file"]
    if not canonical_source.is_file() or not native_source.is_file():
        raise FileNotFoundError("Frozen canonical or cropped native source is missing")
    if sha256_file(native_source) != sequence["cropped_sha256"]:
        raise RuntimeError("Cropped native source differs from its declared SHA-256")
    source_frames = int(sequence["num_frames"])
    source_fps = float(sequence["fps"])
    label = _variant_label(method, seed, revision)
    if max_frames is not None:
        label += f"-smoke{max_frames}"
    run_id = f"{sequence['sequence_id']}__{label}"
    run_dir = root / "runs" / sequence["sequence_id"] / label
    manifest_path = root / "runs" / "manifests" / f"{run_id}.json"
    current_provenance = _method_provenance_receipt(root, method)
    if manifest_path.exists():
        existing = RunManifest.load(manifest_path)
        existing_output = Path(str(existing.output_path))
        existing_timing = Path(str(existing.timing_path))
        if not existing_output.is_absolute():
            existing_output = root / existing_output
        if not existing_timing.is_absolute():
            existing_timing = root / existing_timing
        provenance_path = existing_output.parent / "provenance.json"
        reusable = False
        if (
            existing.status in {RunStatus.SUCCEEDED, RunStatus.INCOMPLETE}
            and existing_output.is_file()
            and existing_timing.is_file()
            and provenance_path.is_file()
            and existing.output_sha256 == sha256_file(existing_output)
            and existing.config_sha256 == current_provenance["aggregate_sha256"]
            and existing.source_sha256 == sequence["cropped_sha256"]
        ):
            timing_value = json.loads(existing_timing.read_text(encoding="utf-8"))
            provenance_value = json.loads(
                provenance_path.read_text(encoding="utf-8")
            )
            history = provenance_value.get("history", [])
            expected_frames = max_frames if max_frames is not None else source_frames
            motion = CanonicalG1.load(existing_output)
            reusable = bool(
                timing_value.get("implementation_receipt_sha256")
                == current_provenance["aggregate_sha256"]
                and timing_value.get("provenance_sha256")
                == sha256_file(provenance_path)
                and isinstance(history, list)
                and len(history) >= 4
                and all(
                    isinstance(item, dict)
                    and item.get("receipt") == current_provenance
                    for item in history
                )
                and len(motion.qpos) == expected_frames
                and np.isfinite(motion.qpos).all()
                and np.array_equal(
                    motion.source_frame_idx,
                    np.arange(expected_frames, dtype=np.int64),
                )
                and bool(np.all(motion.valid))
            )
        if (
            reusable
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
    provenance_before = current_provenance
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
        capture_runtime_witness=False,
        native_source_declared_sha256=sequence["cropped_sha256"],
        native_source_declared_fps=source_fps,
        native_source_declared_frames=source_frames,
    )
    manifest = RunManifest(
        run_id=run_id,
        method=label,
        status=RunStatus.RUNNING,
        command=base_command,
        environment=f"conda:{environment}",
        repo_commit=_git(root, "rev-parse", "HEAD"),
        config_sha256=str(provenance_before["aggregate_sha256"]),
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
    provenance_path = run_dir / "provenance.json"
    provenance_history: list[dict[str, Any]] = [
        {"stage": "before_cold", "at_utc": utc_now(), "receipt": provenance_before}
    ]
    atomic_write_json(
        provenance_path,
        {"schema_version": 2, "method": method, "history": provenance_history},
    )
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
        capture_runtime_witness=True,
        native_source_declared_sha256=sequence["cropped_sha256"],
        native_source_declared_fps=source_fps,
        native_source_declared_frames=source_frames,
    )
    cold_code, cold_wall = _run_worker(
        cold_command, root, logs / "cold.stdout.log", logs / "cold.stderr.log"
    )
    provenance_after_cold = _method_provenance_receipt(root, method)
    provenance_history.append(
        {"stage": "after_cold", "at_utc": utc_now(), "receipt": provenance_after_cold}
    )
    atomic_write_json(
        provenance_path,
        {"schema_version": 2, "method": method, "history": provenance_history},
    )
    if provenance_after_cold["aggregate_sha256"] != manifest.config_sha256:
        cold_code = cold_code or 97
    if cold_code != 0:
        manifest.status = RunStatus.FAILED
        manifest.exit_code = cold_code
        manifest.finished_at = utc_now()
        manifest.wall_time_s = time.perf_counter() - overall_start
        manifest.message = "Cold process failed; inspect cold stderr log"
        manifest.save(manifest_path)
        return manifest
    provenance_before_warm = _method_provenance_receipt(root, method)
    if provenance_before_warm["aggregate_sha256"] != manifest.config_sha256:
        manifest.status = RunStatus.FAILED
        manifest.exit_code = 98
        manifest.finished_at = utc_now()
        manifest.message = "Method provenance changed before warm timing"
        manifest.save(manifest_path)
        return manifest
    provenance_history.append(
        {"stage": "before_warm", "at_utc": utc_now(), "receipt": provenance_before_warm}
    )
    atomic_write_json(
        provenance_path,
        {"schema_version": 2, "method": method, "history": provenance_history},
    )
    warm_code, warm_wall = _run_worker(
        base_command, root, logs / "warm.stdout.log", logs / "warm.stderr.log"
    )
    provenance_after_warm = _method_provenance_receipt(root, method)
    provenance_history.append(
        {"stage": "after_warm", "at_utc": utc_now(), "receipt": provenance_after_warm}
    )
    atomic_write_json(
        provenance_path,
        {"schema_version": 2, "method": method, "history": provenance_history},
    )
    if provenance_after_warm["aggregate_sha256"] != manifest.config_sha256:
        warm_code = warm_code or 99
    manifest.exit_code = warm_code
    manifest.wall_time_s = time.perf_counter() - overall_start
    manifest.finished_at = utc_now()
    if warm_code != 0:
        manifest.status = RunStatus.FAILED
        manifest.message = "Warm timing process failed; inspect stderr log"
        manifest.save(manifest_path)
        return manifest
    motion = CanonicalG1.load(output)
    cold_motion = CanonicalG1.load(cold_output)
    from .native_target_capture import tensor_sha256

    warm_qpos_sha256 = tensor_sha256(np.asarray(motion.qpos, dtype=np.float64))
    cold_qpos_sha256 = tensor_sha256(
        np.asarray(cold_motion.qpos, dtype=np.float64)
    )
    if warm_qpos_sha256 != cold_qpos_sha256:
        raise RuntimeError(
            f"Independent cold witness changed deterministic qpos for {label}"
        )
    runtime_capture = cold_motion.metadata.get("runtime_pre_solver_capture")
    if method in {"gmr", "omniretarget", "protomotions_v2_3"}:
        if not isinstance(runtime_capture, dict):
            raise RuntimeError(f"Cold runtime target witness is missing for {label}")
        runtime_capture = dict(runtime_capture)
        runtime_capture.update(
            {
                "witness_execution_role": "independent_cold_process",
                "witness_output_path": str(cold_output),
                "witness_output_sha256": sha256_file(cold_output),
                "witness_qpos_sha256": cold_qpos_sha256,
                "timed_warm_qpos_sha256": warm_qpos_sha256,
                "witness_and_timed_qpos_identical": True,
                "capture_overhead_excluded_from_measured_warm_runs": True,
            }
        )
        motion.metadata["runtime_pre_solver_capture"] = runtime_capture
        motion.metadata["runtime_witness_binding"] = {
            "cold_output_sha256": sha256_file(cold_output),
            "cold_qpos_sha256": cold_qpos_sha256,
            "warm_qpos_sha256": warm_qpos_sha256,
            "identical": True,
        }
        # This publication/provenance mutation is outside every recorded
        # repetition boundary.
        motion.save(output, source_frame_count=source_frames)
    timing = _timing_summary(
        json.loads(cold_timing.read_text()),
        json.loads(warm_worker_timing.read_text()),
        cold_wall,
        warm_wall,
        motion.fps,
    )
    timing["method"] = label
    timing["run_id"] = run_id
    provenance_before_publication = _method_provenance_receipt(root, method)
    if provenance_before_publication["aggregate_sha256"] != manifest.config_sha256:
        raise RuntimeError("Method provenance changed before timing publication")
    provenance_history.append(
        {
            "stage": "before_publication",
            "at_utc": utc_now(),
            "receipt": provenance_before_publication,
        }
    )
    atomic_write_json(
        provenance_path,
        {"schema_version": 2, "method": method, "history": provenance_history},
    )
    provenance_after_publication = _method_provenance_receipt(root, method)
    if provenance_after_publication["aggregate_sha256"] != manifest.config_sha256:
        raise RuntimeError("Method provenance changed during timing publication")
    provenance_history.append(
        {
            "stage": "after_publication",
            "at_utc": utc_now(),
            "receipt": provenance_after_publication,
        }
    )
    atomic_write_json(
        provenance_path,
        {"schema_version": 2, "method": method, "history": provenance_history},
    )
    timing["provenance_path"] = str(provenance_path)
    timing["provenance_sha256"] = sha256_file(provenance_path)
    timing["implementation_receipt_sha256"] = manifest.config_sha256
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
        revision / "cold" / "work", cold_worker, seed, 0, 1, None,
        capture_runtime_witness=True,
        native_source_declared_sha256=sequence["cropped_sha256"],
        native_source_declared_fps=float(sequence["fps"]),
        native_source_declared_frames=int(sequence["num_frames"]),
    )
    warm_command = _worker_command(
        python, method, root, canonical_source, native_source, warm_output,
        revision / "warm" / "work", warm_worker, seed, 1, 3, None,
        capture_runtime_witness=False,
        native_source_declared_sha256=sequence["cropped_sha256"],
        native_source_declared_fps=float(sequence["fps"]),
        native_source_declared_frames=int(sequence["num_frames"]),
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
