"""Fail-closed, sequential orchestration for the formal Stage 1 timing run.

The campaign is deliberately stricter than :mod:`runner`: an output is reusable
only when it is already bound to this campaign state, its complete provenance
still hashes to the recorded values, and its timing boundary is the current
in-memory boundary.  Files from older or interrupted attempts are moved to a
timestamped archive before a fresh process is launched.
"""

from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable

import numpy as np

from .io_utils import atomic_write_json, atomic_write_text, load_yaml, sha256_file
from .runner import (
    _checkout_receipt,
    _conda_python,
    _files_below,
    _method_config_hash,
    _method_provenance_receipt,
)
from .schemas import CanonicalG1, CanonicalHuman, RunManifest, RunStatus


END_TO_END_BOUNDARY = "canonical_source_file_to_canonical_g1_in_memory"
NATIVE_BOUNDARY = "native_input_ready_to_native_g1_in_memory"
EXPECTED_POPULATION = (
    "sparse-neutral",
    "dense",
    "gmr",
    "omniretarget",
    "protomotions-v2.3",
    "protomotions-v3",
)
EXPECTED_FROZEN_ORDER = (
    "sparse-neutral",
    "dense",
    "protomotions-v3",
    "omniretarget",
    "protomotions-v2.3",
    "gmr",
)
EXPECTED_OUTPUTS = {
    "sparse-neutral": "sparse-neutral-v6",
    "dense": "dense-v6",
    "protomotions-v3": "protomotions-v3-v2",
    "omniretarget": "omniretarget-v3",
    "protomotions-v2.3": "protomotions-v2.3-v3",
    "gmr": "gmr-v3",
}
GENERIC_LAUNCH = {
    "sparse-neutral": ("sparse", "neutral", "v6"),
    "dense": ("dense", "neutral", "v6"),
    "omniretarget": ("omniretarget", "neutral", "v3"),
    "protomotions-v2.3": ("protomotions_v2_3", "neutral", "v3"),
    "gmr": ("gmr", "neutral", "v3"),
}
FORMAL_PROCESS_MARKERS = (
    "retargeting_comparison.method_worker",
    "retargeting_comparison.protomotions_v3_worker",
    "retargeting_comparison.protomotions_v3_native",
    "retargeting_comparison.protomotions_v3_campaign",
    "retargeting_comparison.cli run-method",
    "rtcmp run-method",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _payload_hash(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("payload_sha256", None)
    return _json_hash(payload)


def _write_state(path: Path, value: dict[str, Any]) -> None:
    value["updated_at_utc"] = utc_now()
    value["payload_sha256"] = _payload_hash(value)
    atomic_write_json(path, value)
    # The sidecar makes shell-level artifact collection possible without
    # defining a recursively self-hashed JSON representation.
    sidecar = path.with_suffix(path.suffix + ".sha256")
    atomic_write_text(sidecar, f"{sha256_file(path)}  {path.name}\n")


def _load_state(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        raise ValueError(f"Campaign state SHA-256 sidecar is missing: {sidecar}")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[0] != sha256_file(path) or fields[1] != path.name:
        raise ValueError("Campaign state file hash differs from its SHA-256 sidecar")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Campaign state is not a JSON mapping: {path}")
    recorded = value.get("payload_sha256")
    computed = _payload_hash(value)
    if recorded != computed:
        raise ValueError(
            f"Campaign state payload hash mismatch: recorded={recorded}, computed={computed}"
        )
    return value


def _parse_timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO-8601 timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _aggregate_hash(paths: Iterable[Path]) -> str:
    records = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Frozen implementation input is missing: {path}")
        records.append({"path": str(path), "sha256": sha256_file(path)})
    return _json_hash(records)


def _repo_provenance(root: Path) -> dict[str, Any]:
    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    status = git("status", "--porcelain=v1", "--untracked-files=all")
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_status_porcelain_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "git_worktree_clean": not bool(status),
    }


@dataclass(frozen=True)
class FrozenCampaign:
    path: Path
    config: dict[str, Any]
    config_sha256: str
    campaign_id: str
    sequence_manifest: str
    order: tuple[str, ...]
    outputs: dict[str, str]


def load_frozen_campaign(
    repo_root: str | Path,
    config_path: str | Path = "configs/stage1_timing_campaign.yaml",
) -> FrozenCampaign:
    root = Path(repo_root).resolve()
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path
    config = load_yaml(path)
    if config.get("schema_version") != 1:
        raise ValueError("Stage 1 timing campaign schema_version must be 1")
    randomization = config.get("randomization")
    if not isinstance(randomization, dict):
        raise ValueError("Timing randomization must be a mapping")
    if randomization.get("generator") != "numpy.random.Generator.permutation":
        raise ValueError("Timing order must use numpy.random.Generator.permutation")
    if randomization.get("bit_generator") != "PCG64":
        raise ValueError("Timing order must use NumPy PCG64")
    population = tuple(randomization.get("population", ()))
    if population != EXPECTED_POPULATION or len(set(population)) != len(population):
        raise ValueError(
            "Timing population differs from the six frozen operating points"
        )
    seed = randomization.get("seed")
    if not isinstance(seed, int):
        raise ValueError("Timing permutation seed must be an integer")
    computed = tuple(
        np.random.Generator(np.random.PCG64(seed)).permutation(population).tolist()
    )
    frozen = tuple(randomization.get("frozen_order", ()))
    if computed != frozen or frozen != EXPECTED_FROZEN_ORDER:
        raise ValueError(
            f"Frozen timing order is not the reproducible PCG64 permutation: {computed}"
        )
    protocol = config.get("protocol")
    required_protocol = {
        "new_cold_processes": 1,
        "warmup_runs": 1,
        "measured_warm_runs": 3,
        "cpu_threads": 1,
        "cpu_affinity": "first_allowed_cpu",
        "visualization": False,
        "end_to_end_boundary": END_TO_END_BOUNDARY,
        "native_boundary": NATIVE_BOUNDARY,
        "initialization_jit_compile_separate": True,
        "intermediate_and_final_artifact_writes_excluded": True,
        "canonical_qpos_hash_required_per_repetition": True,
        "cold_warm_qpos_determinism_required": True,
        "runtime_witness_policy": (
            "independent cold solver observation; disabled for warmup/measured"
        ),
        "overlapping_formal_jobs_forbidden": True,
    }
    if not isinstance(protocol, dict):
        raise ValueError("Timing protocol must be a mapping")
    for key, expected in required_protocol.items():
        if protocol.get(key) != expected:
            raise ValueError(f"Timing protocol {key!r} must be {expected!r}")
    outputs = config.get("registered_outputs")
    if outputs != EXPECTED_OUTPUTS:
        raise ValueError("Registered timing outputs differ from the frozen revisions")
    campaign_id = config.get("campaign_id")
    sequence_manifest = config.get("sequence_manifest")
    if not isinstance(campaign_id, str) or not campaign_id:
        raise ValueError("campaign_id must be non-empty")
    if not isinstance(sequence_manifest, str) or not sequence_manifest:
        raise ValueError("sequence_manifest must be non-empty")
    return FrozenCampaign(
        path=path,
        config=config,
        config_sha256=sha256_file(path),
        campaign_id=campaign_id,
        sequence_manifest=sequence_manifest,
        order=frozen,
        outputs=dict(outputs),
    )


def _proto_provenance_receipt(root: Path) -> dict[str, Any]:
    python = _conda_python("egoallo")
    history = python.parent.parent / "conda-meta/history"
    checkout = root / "external/ProtoMotions"
    # This is the full import/execution closure of the formal ProtoMotions-v3
    # campaign.  Deliberately omit report/visualization modules so publishing a
    # report cannot retroactively invalidate a scientifically frozen run.
    harness_names = {
        "bvh.py",
        "calibration.py",
        "constants.py",
        "io_utils.py",
        "method_adapters.py",
        "native_target_capture.py",
        "protomotions_v3.py",
        "protomotions_v3_campaign.py",
        "protomotions_v3_native.py",
        "protomotions_v3_worker.py",
        "rotations.py",
        "runner.py",
        "scale_worker.py",
        "schemas.py",
        "source.py",
        "source_adapter_audit.py",
        "stage1_timing_campaign.py",
    }
    harness_root = root / "src/retargeting_comparison"
    paths = [harness_root / name for name in sorted(harness_names)]
    paths.extend(
        [
            root / "configs/protomotions_v3.yaml",
            root / "configs/stage1_timing_campaign.yaml",
            root
            / "external/ProtoMotions/pyroki/batch_retarget_to_g1_from_keypoints.py",
            root
            / "external/ProtoMotions/protomotions/data/assets/urdf/for_retargeting/g1.urdf",
        ]
    )
    paths.extend(
        _files_below(
            root / "external/ProtoMotions/protomotions/data/assets/mesh/G1",
            {".obj", ".stl", ".dae", ".mtl", ".png", ".jpg"},
        )
    )
    records = [
        {
            "path": _relative(path, root),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted({path.resolve() for path in paths})
    ]
    value = {
        "schema_version": 2,
        "method": "protomotions-v3",
        "files": records,
        "environment": {
            "environment": "egoallo",
            "python_path": str(python),
            "python_sha256": sha256_file(python),
            "conda_history_path": str(history),
            "conda_history_sha256": sha256_file(history),
        },
        "upstream_checkout": _checkout_receipt(checkout),
    }
    value["aggregate_sha256"] = _json_hash(value)
    return value


def _proto_implementation_hash(root: Path) -> str:
    return str(_proto_provenance_receipt(root)["aggregate_sha256"])


def build_campaign_plan(
    repo_root: str | Path = ".",
    config_path: str | Path = "configs/stage1_timing_campaign.yaml",
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    frozen = load_frozen_campaign(root, config_path)
    sequence_path = root / frozen.sequence_manifest
    sequence = load_yaml(sequence_path)
    sequence_id = str(sequence["sequence_id"])
    if sequence.get("full_lafan_authorized") is not False:
        raise RuntimeError("The formal Pilot campaign requires the Stage 1 hard stop")
    source = root / str(sequence["canonical_path"])
    human = CanonicalHuman.load(source)
    if len(human.timestamps) != int(sequence["num_frames"]):
        raise ValueError("Canonical source length differs from the sequence manifest")

    jobs: list[dict[str, Any]] = []
    for index, name in enumerate(frozen.order):
        registered = frozen.outputs[name]
        output = root / "runs" / sequence_id / registered / "canonical_g1.npz"
        if name == "protomotions-v3":
            native_python = _conda_python("egoallo")
            keypoints = (
                root / "source_adapters/protomotions_v3" / sequence_id / "keypoints.npy"
            )
            if not keypoints.is_file():
                raise FileNotFoundError(
                    f"Audited ProtoMotions v3 adapter is missing: {keypoints}"
                )
            cold_root = root / "runs" / sequence_id / registered / "fresh_cold"
            command = [
                sys.executable,
                "-m",
                "retargeting_comparison.protomotions_v3_campaign",
                "--repo-root",
                str(root),
                "--native-python",
                str(native_python),
                "--source",
                str(source),
                "--keypoints",
                str(keypoints),
                "--sequence-id",
                sequence_id,
                "--cold-output",
                str(cold_root / "canonical_g1.npz"),
                "--cold-evidence",
                str(cold_root / "evidence.json"),
                "--summary-json",
                "manifests/protomotions_v3_campaign.formal_timing_v2.json",
                "--run-fresh-cold",
                "--timing-only",
            ]
            manifest = root / "manifests/runs" / f"{sequence_id}__protomotions-v3.json"
            implementation = _proto_implementation_hash(root)
            implementation_receipt = _proto_provenance_receipt(root)
            extra_paths = [
                root / "manifests/protomotions_v3_campaign.formal_timing_v2.json"
            ]
            native_input: dict[str, Any] | None = {
                "path": _relative(keypoints, root),
                "sha256": sha256_file(keypoints),
                "native_python": str(native_python),
            }
        else:
            method, seed, revision = GENERIC_LAUNCH[name]
            command = [
                sys.executable,
                "-m",
                "retargeting_comparison.cli",
                "run-method",
                "--method",
                method,
                "--sequence",
                str(sequence_path),
                "--seed",
                seed,
                "--revision",
                revision,
                "--repo-root",
                str(root),
            ]
            manifest = root / "runs/manifests" / f"{sequence_id}__{registered}.json"
            implementation = _method_config_hash(root, method)
            implementation_receipt = _method_provenance_receipt(root, method)
            extra_paths = []
            native_input = None
        jobs.append(
            {
                "index": index,
                "name": name,
                "registered_output": registered,
                "run_directory": _relative(output.parent, root),
                "output_path": _relative(output, root),
                "manifest_path": _relative(manifest, root),
                "extra_provenance_paths": [
                    _relative(path, root) for path in extra_paths
                ],
                "command": command,
                "command_sha256": _json_hash(command),
                "implementation_sha256": implementation,
                "implementation_receipt": implementation_receipt,
                "native_input": native_input,
            }
        )
    plan = {
        "schema_version": 1,
        "campaign_id": frozen.campaign_id,
        "config_path": _relative(frozen.path, root),
        "config_sha256": frozen.config_sha256,
        "sequence_manifest": _relative(sequence_path, root),
        "sequence_manifest_sha256": sha256_file(sequence_path),
        "sequence_id": sequence_id,
        "canonical_source_path": _relative(source, root),
        "canonical_source_file_sha256": sha256_file(source),
        "canonical_source_embedded_sha256": human.source_sha256,
        "source_frames": len(human.timestamps),
        "permutation": {
            "generator": "numpy.random.Generator.permutation",
            "bit_generator": "PCG64",
            "seed": frozen.config["randomization"]["seed"],
            "population": list(EXPECTED_POPULATION),
            "computed_and_frozen_order": list(frozen.order),
        },
        "protocol": dict(frozen.config["protocol"]),
        "jobs": jobs,
        "orchestrator_sha256": sha256_file(Path(__file__)),
        "repo_provenance": _repo_provenance(root),
    }
    # Worktree cleanliness is recorded for interpretation but is deliberately
    # excluded from the resume key: creating the state JSON itself can change
    # ``git status`` while no frozen implementation input changed.
    stable_plan = dict(plan)
    stable_plan.pop("repo_provenance", None)
    plan["plan_sha256"] = _json_hash(stable_plan)
    return plan


class CampaignLock(AbstractContextManager["CampaignLock"]):
    """Advisory process lock shared by all instances of this orchestrator."""

    def __init__(self, path: Path):
        self.path = path
        self._stream: Any = None
        self.token: str | None = None

    def __enter__(self) -> "CampaignLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self._stream.close()
            self._stream = None
            raise RuntimeError(
                "Another Stage 1 formal timing campaign is active"
            ) from error
        self._stream.seek(0)
        self._stream.truncate()
        self.token = os.urandom(16).hex()
        self._stream.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "token": self.token,
                    "acquired_at_utc": utc_now(),
                }
            )
            + "\n"
        )
        self._stream.flush()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._stream is not None:
            fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
            self._stream.close()
            self._stream = None
            self.token = None


def assert_formal_campaign_ownership(root: str | Path) -> None:
    """Prevent an unregistered formal worker from overlapping the campaign."""

    lock_path = Path(root).resolve() / "runs/.stage1_formal_timing.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("RTCMP_FORMAL_CAMPAIGN_TOKEN")
    stream = lock_path.open("a+", encoding="utf-8")
    try:
        if token:
            stream.seek(0)
            try:
                ownership = json.loads(stream.read())
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    "Formal campaign ownership record is invalid"
                ) from error
            if ownership.get("token") != token:
                raise RuntimeError("Formal campaign token does not own the active lock")
            return
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "An unregistered formal worker cannot overlap the Stage 1 campaign"
            ) from error
        finally:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
    finally:
        stream.close()


def _pin_to_one_cpu() -> list[int]:
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("Formal timing requires Linux CPU-affinity support")
    allowed = sorted(os.sched_getaffinity(0))
    if not allowed:
        raise RuntimeError("Formal timing process has an empty CPU affinity set")
    selected = [int(allowed[0])]
    os.sched_setaffinity(0, set(selected))
    if sorted(os.sched_getaffinity(0)) != selected:
        raise RuntimeError("Could not freeze the campaign to exactly one CPU")
    return selected


def _ancestor_pids() -> set[int]:
    values = {os.getpid()}
    current = os.getpid()
    while current > 1:
        status = Path(f"/proc/{current}/status")
        if not status.is_file():
            break
        parent = 0
        for line in status.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("PPid:"):
                parent = int(line.split()[1])
                break
        if parent <= 0 or parent in values:
            break
        values.add(parent)
        current = parent
    return values


def external_formal_processes() -> list[dict[str, Any]]:
    """Find formal workers not belonging to the current campaign process tree."""

    excluded = _ancestor_pids()
    matches: list[dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in excluded:
            continue
        try:
            command = (
                (entry / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode("utf-8", errors="replace")
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(marker in command for marker in FORMAL_PROCESS_MARKERS):
            matches.append({"pid": pid, "command": command.strip()})
    return sorted(matches, key=lambda item: item["pid"])


def _thread_environment(root: Path, campaign_token: str) -> dict[str, str]:
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
            "RTCMP_FORMAL_CAMPAIGN_TOKEN": campaign_token,
        }
    )
    source = str(root / "src")
    env["PYTHONPATH"] = source + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    return env


def _timing_repetitions(
    timing: dict[str, Any],
) -> tuple[list[Any], list[Any], list[Any]]:
    cold_value = timing.get("cold")
    cold = [cold_value] if isinstance(cold_value, dict) else []
    warmup = timing.get("warmup", [])
    measured = timing.get("measured_warm", timing.get("measured", []))
    if not isinstance(warmup, list) or not isinstance(measured, list):
        raise ValueError("Timing warmup/measured fields must be lists")
    return cold, warmup, measured


def validate_standard_timing(
    timing: dict[str, Any],
    *,
    source_frames: int,
    expected_cpu_affinity: list[int] | None = None,
) -> dict[str, Any]:
    """Validate the exact formal timing contract and return a compact receipt."""

    protocol = timing.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("Formal timing lacks a protocol mapping")
    expected_protocol = {
        "cold_processes": 1,
        "warmup_runs": 1,
        "measured_warm_runs": 3,
        "threads": 1,
        "visualization": False,
        "end_to_end_boundary": END_TO_END_BOUNDARY,
        "native_core_boundary": NATIVE_BOUNDARY,
        "intermediate_and_final_artifact_writes_excluded": True,
    }
    for key, expected in expected_protocol.items():
        if protocol.get(key) != expected:
            raise ValueError(f"Formal timing protocol {key!r} must be {expected!r}")
    cold, warmup, measured = _timing_repetitions(timing)
    if (len(cold), len(warmup), len(measured)) != (1, 1, 3):
        raise ValueError(
            "Formal timing must contain exactly one cold, one warmup, and three measured runs"
        )
    if (
        cold[0].get("role") != "measured"
        or warmup[0].get("role") != "warmup"
        or any(item.get("role") != "measured" for item in measured)
    ):
        raise ValueError(
            "Formal timing repetition roles do not match the 1/1/3 protocol"
        )
    repetitions = cold + warmup + measured
    affinity: list[int] | None = None
    starts: list[datetime] = []
    finishes: list[datetime] = []
    qpos_hashes: list[str] = []
    content_hashes: list[str] = []
    for index, item in enumerate(repetitions):
        if not isinstance(item, dict):
            raise ValueError("Timing repetition must be a mapping")
        if item.get("timing_boundary") != END_TO_END_BOUNDARY:
            raise ValueError(
                "A repetition uses an obsolete or ambiguous end-to-end boundary"
            )
        if item.get("native_boundary") != NATIVE_BOUNDARY:
            raise ValueError(
                "A repetition uses an obsolete or ambiguous native boundary"
            )
        if item.get("frame_count") != source_frames:
            raise ValueError("A formal timing repetition is not the complete source")
        if item.get("thread_limit") != 1:
            raise ValueError("A formal timing repetition is not limited to one thread")
        current_affinity = item.get("cpu_affinity")
        if (
            not isinstance(current_affinity, list)
            or len(current_affinity) != 1
            or not isinstance(current_affinity[0], int)
        ):
            raise ValueError(
                "A formal timing repetition is not pinned to exactly one CPU"
            )
        if affinity is None:
            affinity = list(current_affinity)
        elif current_affinity != affinity:
            raise ValueError("Formal timing repetitions used different CPU affinities")
        wall = item.get("steady_end_to_end_total_s")
        native = item.get("native_total_s")
        if not all(
            isinstance(value, (int, float)) and math.isfinite(value) and value > 0.0
            for value in (wall, native)
        ):
            raise ValueError("Formal timing durations must be finite and positive")
        if float(wall) + 1e-12 < float(native):
            raise ValueError(
                "End-to-end time cannot end before the native in-memory boundary"
            )
        excluded_writes = [
            value
            for key, value in item.items()
            if key.endswith("artifact_write_time_s_excluded")
        ]
        if not excluded_writes or not all(
            isinstance(value, (int, float)) and math.isfinite(value) and value >= 0.0
            for value in excluded_writes
        ):
            raise ValueError("Each repetition must record excluded artifact-write time")
        artifact_hash = item.get("timing_artifact_sha256")
        if not isinstance(artifact_hash, str) or len(artifact_hash) != 64:
            raise ValueError("Each repetition must hash its timing witness")
        qpos_hash = item.get("canonical_qpos_sha256")
        content_hash = item.get("canonical_g1_content_sha256")
        if not all(
            isinstance(value, str) and len(value) == 64
            for value in (qpos_hash, content_hash)
        ):
            raise ValueError("Each repetition must hash canonical G1 content")
        qpos_hashes.append(str(qpos_hash))
        content_hashes.append(str(content_hash))
        start = _parse_timestamp(
            item.get("started_at_utc"), f"repetition[{index}].started"
        )
        finish = _parse_timestamp(
            item.get("output_in_memory_at_utc"), f"repetition[{index}].finished"
        )
        if finish < start:
            raise ValueError("A repetition finished before it started")
        starts.append(start)
        finishes.append(finish)
    if expected_cpu_affinity is not None and affinity != expected_cpu_affinity:
        raise ValueError("Timing evidence does not use the campaign CPU affinity")
    # The cold process completes before the separate warm process begins, and
    # warmup/measured calls are serialized within their process.
    if any(starts[index] < finishes[index - 1] for index in range(1, len(starts))):
        raise ValueError("Timing repetitions overlap")
    if len(set(qpos_hashes)) != 1 or len(set(content_hashes)) != 1:
        raise ValueError("Cold/warm canonical G1 content is not deterministic")
    if cold[0].get("runtime_witness_enabled") is not True or any(
        item.get("runtime_witness_enabled") is not False
        for item in warmup + measured
    ):
        raise ValueError(
            "Runtime target capture must be cold-only and excluded from warm timing"
        )
    return {
        "cpu_affinity": affinity,
        "cold": 1,
        "warmup": 1,
        "measured": 3,
        "first_started_at_utc": starts[0].isoformat(),
        "last_output_in_memory_at_utc": finishes[-1].isoformat(),
        "end_to_end_rtf_raw": timing.get("end_to_end_rtf_raw"),
        "native_core_rtf_raw": timing.get("native_core_rtf_raw"),
        "canonical_qpos_sha256": qpos_hashes[0],
        "canonical_g1_content_sha256": content_hashes[0],
        "all_cold_warm_qpos_identical": True,
    }


def _path_from_job(root: Path, job: dict[str, Any], key: str) -> Path:
    path = Path(str(job[key]))
    return path if path.is_absolute() else root / path


def _current_job_implementation_receipt(
    root: Path, job: dict[str, Any]
) -> dict[str, Any]:
    if job["name"] == "protomotions-v3":
        return _proto_provenance_receipt(root)
    method, _, _ = GENERIC_LAUNCH[str(job["name"])]
    return _method_provenance_receipt(root, method)


def validate_job_artifacts(
    root: Path,
    plan: dict[str, Any],
    job: dict[str, Any],
    *,
    expected_cpu_affinity: list[int] | None,
) -> dict[str, Any]:
    output = _path_from_job(root, job, "output_path")
    manifest_path = _path_from_job(root, job, "manifest_path")
    if not output.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"Formal output or manifest is missing for {job['name']}"
        )
    motion = CanonicalG1.load(output)
    source_frames = int(plan["source_frames"])
    motion.validate(source_frame_count=source_frames)
    if (
        len(motion.qpos) != source_frames
        or motion.metadata.get("completion_status") != "succeeded"
        or not bool(np.all(motion.valid))
    ):
        raise ValueError(
            f"Formal output is not a complete valid trajectory: {job['name']}"
        )
    manifest = RunManifest.load(manifest_path)
    if manifest.status != RunStatus.SUCCEEDED or manifest.exit_code != 0:
        raise ValueError(f"Formal run manifest did not succeed: {job['name']}")
    if manifest.output_sha256 != sha256_file(output):
        raise ValueError(f"Formal output hash differs from its manifest: {job['name']}")
    if not manifest.timing_path:
        raise ValueError(f"Formal run manifest has no timing path: {job['name']}")
    timing_path = Path(manifest.timing_path)
    if not timing_path.is_absolute():
        timing_path = root / timing_path
    if not timing_path.is_file():
        raise FileNotFoundError(f"Formal timing JSON is missing: {timing_path}")
    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    timing_receipt = validate_standard_timing(
        timing,
        source_frames=source_frames,
        expected_cpu_affinity=expected_cpu_affinity,
    )
    cold, warmup, measured = _timing_repetitions(timing)
    for item in cold + warmup + measured:
        artifact_value = item.get("timing_artifact_path")
        if not isinstance(artifact_value, str) or not artifact_value:
            raise ValueError("A formal timing witness path is missing")
        artifact = Path(artifact_value)
        if not artifact.is_absolute():
            artifact = root / artifact
        if not artifact.is_file():
            raise FileNotFoundError(f"Formal timing witness is missing: {artifact}")
        if sha256_file(artifact) != item["timing_artifact_sha256"]:
            raise ValueError(f"Formal timing witness hash mismatch: {artifact}")
    if job["name"] == "protomotions-v3":
        current_receipt = _proto_provenance_receipt(root)
        if current_receipt != job.get("implementation_receipt"):
            raise ValueError("ProtoMotions v3 complete provenance receipt is stale")
        if manifest.config_sha256 != sha256_file(root / "configs/protomotions_v3.yaml"):
            raise ValueError("ProtoMotions v3 manifest config hash is stale")
        evidence_path = output.parent / "formal/evidence.json"
        if not evidence_path.is_file():
            raise FileNotFoundError(
                "ProtoMotions v3 formal implementation evidence is missing"
            )
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        declared_input = evidence.get("declared_input")
        if (
            not isinstance(declared_input, dict)
            or declared_input.get("sha256") != job["native_input"]["sha256"]
        ):
            raise ValueError("ProtoMotions v3 declared native adapter hash is stale")
        hashes = evidence.get("implementation_hashes")
        if not isinstance(hashes, dict) or not hashes:
            raise ValueError("ProtoMotions v3 implementation hashes are missing")
        for label, recorded in hashes.items():
            candidate = {
                "wrapper": root
                / "src/retargeting_comparison/protomotions_v3_worker.py",
                "native_worker": root
                / "src/retargeting_comparison/protomotions_v3_native.py",
                "adapter": root / "src/retargeting_comparison/protomotions_v3.py",
                "source_adapter": root
                / "src/retargeting_comparison/source_adapter_audit.py",
                "official_solver": root
                / "external/ProtoMotions/pyroki/batch_retarget_to_g1_from_keypoints.py",
            }.get(label)
            if (
                candidate is None
                or not candidate.is_file()
                or sha256_file(candidate) != recorded
            ):
                raise ValueError(
                    f"ProtoMotions v3 implementation hash is stale: {label}"
                )
    else:
        method, _, _ = GENERIC_LAUNCH[job["name"]]
        current_receipt = _method_provenance_receipt(root, method)
        if current_receipt != job.get("implementation_receipt"):
            raise ValueError(
                f"Formal complete provenance receipt is stale: {job['name']}"
            )
        if manifest.config_sha256 != job["implementation_sha256"]:
            raise ValueError(
                f"Formal method implementation/config hash is stale: {job['name']}"
            )
    return {
        "output_path": _relative(output, root),
        "output_sha256": sha256_file(output),
        "manifest_path": _relative(manifest_path, root),
        "manifest_sha256": sha256_file(manifest_path),
        "timing_path": _relative(timing_path, root),
        "timing_sha256": sha256_file(timing_path),
        "implementation_sha256": job["implementation_sha256"],
        "implementation_receipt_sha256": _json_hash(
            job["implementation_receipt"]
        ),
        "timing_contract": timing_receipt,
    }


def _archive_paths(
    root: Path,
    plan: dict[str, Any],
    job: dict[str, Any],
    *,
    reason: str,
) -> dict[str, Any] | None:
    candidates = [
        _path_from_job(root, job, "run_directory"),
        _path_from_job(root, job, "manifest_path"),
    ]
    candidates.extend(
        Path(path) if Path(path).is_absolute() else root / path
        for path in job.get("extra_provenance_paths", [])
    )
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    destination = (
        root
        / "runs/archive/stage1_timing_campaign"
        / str(plan["campaign_id"])
        / str(job["name"])
        / stamp
    )
    destination.mkdir(parents=True, exist_ok=False)
    moved: list[dict[str, Any]] = []
    for path in existing:
        target = destination / path.name
        if target.exists():
            target = destination / f"{len(moved):02d}_{path.name}"
        shutil.move(str(path), str(target))
        moved.append({"from": _relative(path, root), "to": _relative(target, root)})
    receipt = {
        "reason": reason,
        "archived_at_utc": utc_now(),
        "paths": moved,
    }
    atomic_write_json(destination / "archive_receipt.json", receipt)
    receipt["receipt_path"] = _relative(destination / "archive_receipt.json", root)
    receipt["receipt_sha256"] = sha256_file(destination / "archive_receipt.json")
    return receipt


def _new_state(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "campaign_id": plan["campaign_id"],
        "status": "pending",
        "plan_sha256": plan["plan_sha256"],
        "config_path": plan["config_path"],
        "config_sha256": plan["config_sha256"],
        "orchestrator_sha256": plan["orchestrator_sha256"],
        "sequence_id": plan["sequence_id"],
        "permutation": plan["permutation"],
        "protocol": plan["protocol"],
        "repo_provenance_at_creation": plan["repo_provenance"],
        "started_at_utc": None,
        "finished_at_utc": None,
        "cpu_affinity": None,
        "jobs": [
            {
                **job,
                "status": "pending",
                "attempts": [],
                "archives": [],
                "receipt": None,
            }
            for job in plan["jobs"]
        ],
    }


def _archive_incompatible_state(root: Path, state_path: Path, reason: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    destination = root / "runs/archive/stage1_timing_campaign/states" / stamp
    destination.mkdir(parents=True, exist_ok=False)
    shutil.move(str(state_path), str(destination / state_path.name))
    sidecar = state_path.with_suffix(state_path.suffix + ".sha256")
    if sidecar.exists():
        shutil.move(str(sidecar), str(destination / sidecar.name))
    atomic_write_json(
        destination / "reason.json", {"reason": reason, "archived_at_utc": utc_now()}
    )


def _state_for_plan(
    root: Path, state_path: Path, plan: dict[str, Any]
) -> dict[str, Any]:
    if not state_path.is_file():
        return _new_state(plan)
    state = _load_state(state_path)
    compatibility = {
        "campaign_id": plan["campaign_id"],
        "plan_sha256": plan["plan_sha256"],
        "config_sha256": plan["config_sha256"],
        "orchestrator_sha256": plan["orchestrator_sha256"],
    }
    differences = {
        key: (state.get(key), expected)
        for key, expected in compatibility.items()
        if state.get(key) != expected
    }
    if differences:
        _archive_incompatible_state(
            root, state_path, f"campaign provenance changed: {differences}"
        )
        return _new_state(plan)
    if len(state.get("jobs", [])) != len(plan["jobs"]):
        raise ValueError("Compatible campaign state has an invalid job count")
    return state


def _validate_completed_prefix(
    root: Path,
    plan: dict[str, Any],
    state: dict[str, Any],
) -> int:
    affinity = state.get("cpu_affinity")
    first_pending = len(state["jobs"])
    saw_pending = False
    previous_finished: datetime | None = None
    for index, job in enumerate(state["jobs"]):
        if job["status"] != "succeeded":
            saw_pending = True
            first_pending = min(first_pending, index)
            continue
        if saw_pending:
            raise ValueError(
                "Campaign state has a succeeded job after an unfinished job"
            )
        receipt = validate_job_artifacts(
            root,
            plan,
            job,
            expected_cpu_affinity=affinity,
        )
        if job.get("receipt") != receipt:
            raise ValueError(
                f"Recorded receipt changed for completed job {job['name']}"
            )
        attempts = job.get("attempts", [])
        if not attempts:
            raise ValueError(f"Completed job has no attempt evidence: {job['name']}")
        start = _parse_timestamp(attempts[-1].get("started_at_utc"), "job started")
        finish = _parse_timestamp(attempts[-1].get("finished_at_utc"), "job finished")
        if finish < start or (
            previous_finished is not None and start < previous_finished
        ):
            raise ValueError("Completed formal jobs overlap or are out of order")
        previous_finished = finish
    return first_pending


def run_stage1_timing_campaign(
    repo_root: str | Path = ".",
    config_path: str | Path = "configs/stage1_timing_campaign.yaml",
    state_path: str | Path = "manifests/stage1_timing_campaign.json",
    *,
    launch: bool = False,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    plan = build_campaign_plan(root, config_path)
    state_file = Path(state_path)
    if not state_file.is_absolute():
        state_file = root / state_file
    if not launch:
        return {
            "status": "preview",
            "plan": plan,
            "state_path": _relative(state_file, root),
        }

    lock_path = root / "runs/.stage1_formal_timing.lock"
    with CampaignLock(lock_path) as campaign_lock:
        if campaign_lock.token is None:
            raise RuntimeError("Formal campaign lock did not issue an ownership token")
        state = _state_for_plan(root, state_file, plan)
        affinity = _pin_to_one_cpu()
        if state.get("cpu_affinity") not in (None, affinity):
            raise RuntimeError("Resumed campaign is not on its original CPU affinity")
        state["cpu_affinity"] = affinity
        if state["started_at_utc"] is None:
            state["started_at_utc"] = utc_now()
        state["status"] = "running"
        _write_state(state_file, state)
        try:
            first_pending = _validate_completed_prefix(root, plan, state)
        except Exception as error:
            state["status"] = "failed_completed_receipt_validation"
            state["validation_error"] = f"{type(error).__name__}: {error}"
            _write_state(state_file, state)
            return state
        if first_pending == len(state["jobs"]):
            state["status"] = "succeeded"
            state["finished_at_utc"] = state.get("finished_at_utc") or utc_now()
            _write_state(state_file, state)
            return state

        for index in range(first_pending, len(state["jobs"])):
            job = state["jobs"][index]
            if job["status"] == "succeeded":
                raise RuntimeError("Internal error: non-prefix succeeded job")
            active = external_formal_processes()
            if active:
                state["status"] = "blocked_external_formal_process"
                state["external_formal_processes"] = active
                _write_state(state_file, state)
                return state
            state.pop("external_formal_processes", None)
            current_receipt = _current_job_implementation_receipt(root, job)
            if current_receipt != job.get("implementation_receipt"):
                job["status"] = "failed"
                job["validation_error"] = (
                    "Complete implementation provenance changed before launch"
                )
                state["status"] = "failed_prelaunch_provenance"
                _write_state(state_file, state)
                return state
            archive = _archive_paths(
                root,
                plan,
                job,
                reason=(
                    "interrupted_or_failed_campaign_attempt"
                    if job["status"] in {"running", "failed", "interrupted"}
                    else "pre-existing_artifact_not_bound_to_this_campaign"
                ),
            )
            if archive is not None:
                job["archives"].append(archive)
            attempt = {
                "number": len(job["attempts"]) + 1,
                "started_at_utc": utc_now(),
                "finished_at_utc": None,
                "returncode": None,
                "command_sha256": job["command_sha256"],
            }
            job.pop("validation_error", None)
            state.pop("validation_error", None)
            job["attempts"].append(attempt)
            job["status"] = "running"
            state["status"] = "running"
            state["current_job_index"] = index
            _write_state(state_file, state)
            try:
                started = time.perf_counter()
                result = subprocess.run(
                    job["command"],
                    cwd=root,
                    env=_thread_environment(root, campaign_lock.token),
                    check=False,
                )
                attempt["process_wall_time_s"] = time.perf_counter() - started
                attempt["returncode"] = result.returncode
                attempt["finished_at_utc"] = utc_now()
            except BaseException:
                attempt["finished_at_utc"] = utc_now()
                job["status"] = "interrupted"
                state["status"] = "interrupted"
                _write_state(state_file, state)
                raise
            if result.returncode != 0:
                job["status"] = "failed"
                state["status"] = "failed"
                _write_state(state_file, state)
                return state
            active = external_formal_processes()
            if active:
                job["status"] = "failed"
                state["status"] = "failed_orphan_formal_process"
                state["external_formal_processes"] = active
                _write_state(state_file, state)
                return state
            try:
                receipt = validate_job_artifacts(
                    root,
                    plan,
                    job,
                    expected_cpu_affinity=affinity,
                )
            except Exception as error:
                job["status"] = "failed"
                job["validation_error"] = f"{type(error).__name__}: {error}"
                state["status"] = "failed_validation"
                _write_state(state_file, state)
                return state
            job["receipt"] = receipt
            job["status"] = "succeeded"
            _write_state(state_file, state)

        try:
            _validate_completed_prefix(root, plan, state)
        except Exception as error:
            state["status"] = "failed_final_receipt_validation"
            state["validation_error"] = f"{type(error).__name__}: {error}"
            _write_state(state_file, state)
            return state
        state["status"] = "succeeded"
        state["finished_at_utc"] = utc_now()
        state.pop("current_job_index", None)
        state.pop("external_formal_processes", None)
        _write_state(state_file, state)
        return state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--config", default="configs/stage1_timing_campaign.yaml")
    parser.add_argument("--state", default="manifests/stage1_timing_campaign.json")
    parser.add_argument("--launch", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_stage1_timing_campaign(
        args.repo_root,
        args.config,
        args.state,
        launch=args.launch,
    )
    print(result["status"])
    return 0 if result["status"] in {"preview", "succeeded"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
