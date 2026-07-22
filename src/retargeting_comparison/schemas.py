"""Canonical on-disk contracts for human sources, G1 motions, and runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from .constants import CANONICAL_QPOS_WIDTH, MIN_COMPLETION_RATIO
from .io_utils import atomic_write_json
from .rotations import normalize_quaternion_wxyz


def _require_unit_quaternion_wxyz(value: np.ndarray, label: str) -> None:
    """Reject malformed canonical quaternion evidence instead of repairing it."""

    quaternion = np.asarray(value, dtype=np.float64)
    # Keep the shared helper's finite/zero diagnostics, but deliberately ignore
    # its normalized return: normalization belongs at an adapter boundary.
    normalize_quaternion_wxyz(quaternion)
    norms = np.linalg.norm(quaternion, axis=-1)
    if not np.allclose(norms, 1.0, atol=1e-6, rtol=0.0):
        maximum = float(np.max(np.abs(norms - 1.0)))
        raise ValueError(
            f"{label} must contain unit wxyz quaternions; max norm error={maximum:.3e}"
        )


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    INCOMPLETE = "incomplete"
    FAILED = "failed"
    NA = "na"


@dataclass
class CanonicalHuman:
    joint_names: np.ndarray
    parent_indices: np.ndarray
    local_rotations: np.ndarray
    world_rotations: np.ndarray
    world_positions: np.ndarray
    root_translation: np.ndarray
    fps: float
    timestamps: np.ndarray
    foot_contact_labels: np.ndarray
    source_sha256: str

    def validate(self) -> None:
        names = np.asarray(self.joint_names).astype(str)
        parents = np.asarray(self.parent_indices, dtype=np.int64)
        local = np.asarray(self.local_rotations, dtype=np.float64)
        world = np.asarray(self.world_rotations, dtype=np.float64)
        positions = np.asarray(self.world_positions, dtype=np.float64)
        root = np.asarray(self.root_translation, dtype=np.float64)
        times = np.asarray(self.timestamps, dtype=np.float64)
        contacts = np.asarray(self.foot_contact_labels, dtype=bool)
        if names.ndim != 1 or len(set(names.tolist())) != len(names):
            raise ValueError("joint_names must be a unique 1-D array")
        if parents.shape != (len(names),) or parents[0] != -1:
            raise ValueError("parent_indices must match joints and start with -1")
        if np.any(parents[1:] < 0) or np.any(parents[1:] >= np.arange(1, len(names))):
            raise ValueError("Each non-root parent must precede its child")
        if local.ndim != 3 or local.shape[1:] != (len(names), 4):
            raise ValueError("local_rotations must have shape [T,J,4]")
        if world.shape != local.shape:
            raise ValueError("world_rotations must match local_rotations")
        frames = local.shape[0]
        if positions.shape != (frames, len(names), 3):
            raise ValueError("world_positions must have shape [T,J,3]")
        if root.shape != (frames, 3) or times.shape != (frames,):
            raise ValueError("root_translation/timestamps frame counts do not match")
        if contacts.shape != (frames, 2):
            raise ValueError("foot_contact_labels must have shape [T,2]")
        if not np.isfinite(positions).all() or not np.isfinite(root).all():
            raise ValueError("Human motion contains NaN/Inf")
        _require_unit_quaternion_wxyz(local, "local_rotations")
        _require_unit_quaternion_wxyz(world, "world_rotations")
        if self.fps <= 0 or frames < 1:
            raise ValueError("fps and frame count must be positive")
        if frames > 1 and not np.allclose(np.diff(times), 1.0 / self.fps, atol=1e-8):
            raise ValueError("timestamps are not uniformly sampled at fps")
        if len(self.source_sha256) != 64:
            raise ValueError("source_sha256 must be a SHA-256 hex digest")

    def save(self, path: str | Path) -> None:
        self.validate()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            joint_names=np.asarray(self.joint_names).astype("U"),
            parent_indices=np.asarray(self.parent_indices, dtype=np.int64),
            local_rotations=np.asarray(self.local_rotations, dtype=np.float64),
            world_rotations=np.asarray(self.world_rotations, dtype=np.float64),
            world_positions=np.asarray(self.world_positions, dtype=np.float64),
            root_translation=np.asarray(self.root_translation, dtype=np.float64),
            fps=np.asarray(self.fps, dtype=np.float64),
            timestamps=np.asarray(self.timestamps, dtype=np.float64),
            foot_contact_labels=np.asarray(self.foot_contact_labels, dtype=bool),
            source_sha256=np.asarray(self.source_sha256),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CanonicalHuman":
        with np.load(path, allow_pickle=False) as data:
            value = cls(
                joint_names=data["joint_names"],
                parent_indices=data["parent_indices"],
                local_rotations=data["local_rotations"],
                world_rotations=data["world_rotations"],
                world_positions=data["world_positions"],
                root_translation=data["root_translation"],
                fps=float(data["fps"]),
                timestamps=data["timestamps"],
                foot_contact_labels=data["foot_contact_labels"],
                source_sha256=str(data["source_sha256"]),
            )
        value.validate()
        return value


@dataclass
class CanonicalG1:
    qpos: np.ndarray
    fps: float
    source_frame_idx: np.ndarray
    valid: np.ndarray
    per_frame_solve_time_s: np.ndarray
    metadata: dict[str, Any]

    def validate(self, source_frame_count: int | None = None) -> None:
        qpos = np.asarray(self.qpos, dtype=np.float64)
        frames = qpos.shape[0] if qpos.ndim == 2 else 0
        if qpos.shape != (frames, CANONICAL_QPOS_WIDTH) or frames < 1:
            raise ValueError(f"qpos must have shape [T,{CANONICAL_QPOS_WIDTH}]")
        if np.asarray(self.source_frame_idx).shape != (frames,):
            raise ValueError("source_frame_idx must have shape [T]")
        if np.asarray(self.valid).shape != (frames,):
            raise ValueError("valid must have shape [T]")
        if np.asarray(self.per_frame_solve_time_s).shape != (frames,):
            raise ValueError("per_frame_solve_time_s must have shape [T]")
        if not np.isfinite(qpos).all() or not np.isfinite(self.per_frame_solve_time_s).all():
            raise ValueError("G1 output contains NaN/Inf")
        _require_unit_quaternion_wxyz(qpos[:, 3:7], "qpos root rotation")
        if self.fps <= 0 or np.any(np.diff(self.source_frame_idx) <= 0):
            raise ValueError("fps must be positive and source_frame_idx strictly increasing")
        if source_frame_count is not None:
            ratio = frames / source_frame_count
            expected = "succeeded" if ratio >= MIN_COMPLETION_RATIO else "incomplete"
            status = str(self.metadata.get("completion_status", ""))
            if status != expected:
                raise ValueError(f"completion_status must be {expected!r}, got {status!r}")

    def save(self, path: str | Path, source_frame_count: int | None = None) -> None:
        self.validate(source_frame_count)
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            qpos=np.asarray(self.qpos, dtype=np.float64),
            fps=np.asarray(self.fps, dtype=np.float64),
            source_frame_idx=np.asarray(self.source_frame_idx, dtype=np.int64),
            valid=np.asarray(self.valid, dtype=bool),
            per_frame_solve_time_s=np.asarray(self.per_frame_solve_time_s, dtype=np.float64),
            metadata_json=np.asarray(json.dumps(self.metadata, sort_keys=True)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CanonicalG1":
        with np.load(path, allow_pickle=False) as data:
            value = cls(
                qpos=data["qpos"],
                fps=float(data["fps"]),
                source_frame_idx=data["source_frame_idx"],
                valid=data["valid"],
                per_frame_solve_time_s=data["per_frame_solve_time_s"],
                metadata=json.loads(str(data["metadata_json"])),
            )
        value.validate()
        return value


@dataclass
class RunManifest:
    run_id: str
    method: str
    status: RunStatus
    command: list[str]
    environment: str
    repo_commit: str
    config_sha256: str
    device: str
    started_at: str | None = None
    finished_at: str | None = None
    wall_time_s: float | None = None
    exit_code: int | None = None
    stdout_log: str | None = None
    stderr_log: str | None = None
    output_path: str | None = None
    output_sha256: str | None = None
    source_sha256: str | None = None
    timing_path: str | None = None
    message: str | None = None

    def save(self, path: str | Path) -> None:
        data = asdict(self)
        data["status"] = self.status.value
        atomic_write_json(path, data)

    @classmethod
    def load(cls, path: str | Path) -> "RunManifest":
        data = json.loads(Path(path).read_text())
        data["status"] = RunStatus(data["status"])
        return cls(**data)
