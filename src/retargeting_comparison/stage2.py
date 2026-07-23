"""Budget-gated, deterministic, resumable Stage-2 LAFAN orchestration.

This module intentionally has no CLI registration.  Importing it or building a
plan cannot launch an experiment.  A launch requires all of the following:

* the checked-in Stage-2 authorization record;
* a Stage-1 validation manifest whose decision is ``GO``;
* a plan whose 1.5x-safe wall and retained-storage projections fit the limits;
* ``launch=True`` and an exact acknowledgement of the immutable plan SHA-256.

Public pipelines retain their native preprocessing and scale policy.  Sparse
and Dense form a separate controlled stratum whose common scale and root anchor
are calibrated independently for every source sequence before solving.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import tempfile
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .io_utils import atomic_write_json, load_yaml, sha256_file


SCHEMA_VERSION = 2
ABANDONED_RUNNING_EXIT_CODE = 255
ABANDONED_RUNNING_REASON = "orchestrator_crash_detected_on_resume"
PRODUCTION_RETARGETERS = (
    "sparse-neutral",
    "dense",
    "gmr",
    "omniretarget",
    "protomotions-v2.3",
)
STAGE1_FORMAL_RETARGETERS = (*PRODUCTION_RETARGETERS, "protomotions-v3")
PRODUCTION_METHODS = (*PRODUCTION_RETARGETERS, "unitree-reference")
THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "MPLCONFIGDIR": "/tmp/rtcmp-mpl",
}


class Stage2Error(RuntimeError):
    """Raised when a scientific, authorization, or budget gate fails."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _plan_sha256(plan: Mapping[str, Any]) -> str:
    value = dict(plan)
    value.pop("plan_sha256", None)
    return _canonical_sha256(value)


def verify_plan_hash(plan: Mapping[str, Any]) -> bool:
    digest = plan.get("plan_sha256")
    return isinstance(digest, str) and digest == _plan_sha256(plan)


def _slug(relative_path: str) -> str:
    path = Path(relative_path)
    raw = "__".join(path.with_suffix("").parts).lower()
    result = "".join(character if character.isalnum() else "-" for character in raw)
    while "--" in result:
        result = result.replace("--", "-")
    return result.strip("-")


def _actor_id(relative_path: str) -> str:
    """Return the frozen LAFAN actor cluster encoded by the filename."""

    match = re.search(
        r"(?:^|[_-])(subject\d+)(?:[_-]|\.|$)", Path(relative_path).name, re.I
    )
    if match is None:
        raise Stage2Error(f"LAFAN filename lacks a subject cluster: {relative_path}")
    return match.group(1).lower()


@dataclass(frozen=True)
class SequenceSpec:
    sequence_id: str
    relative_path: str
    source_sha256: str
    source_size_bytes: int
    frames: int
    fps: float
    duration_s: float
    reference_relative_path: str | None = None
    reference_sha256: str | None = None
    reference_size_bytes: int = 0
    reference_frames: int | None = None

    def validate(self) -> None:
        if not self.sequence_id or Path(self.relative_path).is_absolute():
            raise ValueError(
                "Sequence IDs must be non-empty and paths must be relative"
            )
        if len(self.source_sha256) != 64 or self.frames < 1 or self.fps <= 0.0:
            raise ValueError("Sequence source contract is invalid")
        if self.source_size_bytes < 1 or self.duration_s <= 0.0:
            raise ValueError("Sequence size and duration must be positive")
        reference_fields = (
            self.reference_relative_path,
            self.reference_sha256,
            self.reference_frames,
        )
        if any(value is not None for value in reference_fields):
            if not all(value is not None for value in reference_fields):
                raise ValueError("Reference provenance must be complete or absent")
            if len(str(self.reference_sha256)) != 64 or int(self.reference_frames) < 1:
                raise ValueError("Reference hash/frame contract is invalid")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SequenceSpec":
        result = cls(**dict(value))
        result.validate()
        return result


@dataclass(frozen=True)
class MethodSpec:
    method: str
    stratum: str
    public_pipeline: bool
    environment: str
    workers: int
    rtf: float
    startup_s: float
    retained_bytes_per_frame: int
    retained_bytes_per_sequence: int
    estimate_source: str
    method_contract_sha256: str = ""
    timing_evidence_sha256: str = ""
    environment_provenance_sha256: str = ""
    policy_sha256: str = ""
    registered_revision: str = ""
    parallel_contention_factor: float = 1.0

    def validate(self, physical_cores: int) -> None:
        allowed = {
            "controlled_common_per_sequence_scale",
            "native_public_pipeline",
            "benchmark_public_retargeter_port",
            "external_reference",
        }
        if self.stratum not in allowed:
            raise ValueError(f"Unknown Stage-2 evidence stratum: {self.stratum}")
        if not 1 <= self.workers <= physical_cores:
            raise ValueError(f"Invalid worker count for {self.method}")
        if self.rtf < 0.0 or self.startup_s < 0.0:
            raise ValueError(f"Invalid timing estimate for {self.method}")
        if self.parallel_contention_factor < 1.0:
            raise ValueError(f"Invalid parallel contention factor for {self.method}")
        if self.retained_bytes_per_frame < 0 or self.retained_bytes_per_sequence < 0:
            raise ValueError(f"Invalid storage estimate for {self.method}")
        for label, digest in (
            ("method contract", self.method_contract_sha256),
            ("timing evidence", self.timing_evidence_sha256),
            ("environment provenance", self.environment_provenance_sha256),
            ("policy", self.policy_sha256),
        ):
            if digest and (len(digest) != 64 or not re.fullmatch(r"[0-9a-f]{64}", digest)):
                raise ValueError(f"Invalid {label} SHA-256 for {self.method}")


@dataclass(frozen=True)
class ScheduledJob:
    job_id: str
    method: str
    sequence_id: str
    phase_index: int
    worker_index: int
    estimated_runtime_s: float
    projected_start_s: float
    projected_end_s: float
    stratum: str
    reference_relative_path: str | None
    method_contract_sha256: str
    policy_sha256: str
    job_contract_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _bvh_header(path: Path) -> tuple[int, float]:
    frames: int | None = None
    frame_time: float | None = None
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            stripped = line.strip()
            if stripped.lower().startswith("frames:"):
                frames = int(stripped.split(":", 1)[1].strip())
            elif stripped.lower().startswith("frame time:"):
                frame_time = float(stripped.split(":", 1)[1].strip())
                break
    if frames is None or frame_time is None or frames < 1 or frame_time <= 0.0:
        raise Stage2Error(f"Invalid BVH motion header: {path}")
    return frames, 1.0 / frame_time


def _csv_shape(path: Path, *, require_finite: bool = True) -> tuple[int, int]:
    rows = 0
    columns: int | None = None
    with path.open(encoding="utf-8", errors="strict") as stream:
        for line in stream:
            stripped = line.strip()
            if not stripped:
                continue
            fields = stripped.split(",")
            width = len(fields)
            if columns is None:
                columns = width
            elif width != columns:
                raise Stage2Error(f"Ragged reference CSV: {path}")
            if require_finite:
                try:
                    values = [float(field) for field in fields]
                except ValueError as error:
                    raise Stage2Error(
                        f"Non-numeric reference CSV value: {path}"
                    ) from error
                if not all(
                    value == value and abs(value) != float("inf") for value in values
                ):
                    raise Stage2Error(f"NaN/Inf in reference CSV: {path}")
            rows += 1
    if rows < 1 or columns is None:
        raise Stage2Error(f"Empty reference CSV: {path}")
    return rows, columns


def discover_lafan_sequences(
    repo_root: str | Path,
    config: Mapping[str, Any],
    *,
    enforce_expected_inventory: bool = True,
) -> list[SequenceSpec]:
    """Hash and freeze the source inventory without looking at method results."""

    root = Path(repo_root).resolve()
    dataset = config["dataset"]
    source_root = root / dataset["root"]
    files = sorted(
        source_root.glob(dataset["glob"]), key=lambda path: path.as_posix().lower()
    )
    if not files:
        raise Stage2Error(f"No LAFAN BVH files below {source_root}")

    reference_config = config["reference_corpus"]
    reference_root = root / reference_config["root"]
    reference_by_stem: dict[str, Path] = {}
    if (
        not reference_root.is_dir()
        and enforce_expected_inventory
        and int(reference_config.get("expected_files", 0)) > 0
    ):
        raise Stage2Error(f"Frozen reference corpus is missing: {reference_root}")
    if reference_root.is_dir():
        reference_files = sorted(reference_root.glob(reference_config["glob"]))
        expected_reference_files = reference_config.get("expected_files")
        if (
            enforce_expected_inventory
            and expected_reference_files is not None
            and len(reference_files) != int(expected_reference_files)
        ):
            raise Stage2Error(
                "Reference inventory differs from the frozen "
                f"{int(expected_reference_files)}-file contract"
            )
        for path in reference_files:
            stem = path.stem.lower()
            if stem in reference_by_stem:
                raise Stage2Error(f"Duplicate reference basename {stem!r}")
            reference_by_stem[stem] = path

    sequences: list[SequenceSpec] = []
    identifiers: set[str] = set()
    for path in files:
        relative = path.relative_to(source_root).as_posix()
        sequence_id = _slug(relative)
        if sequence_id in identifiers:
            raise Stage2Error(f"Sequence-ID collision for {relative}")
        identifiers.add(sequence_id)
        frames, fps = _bvh_header(path)
        reference = reference_by_stem.get(path.stem.lower())
        reference_values: dict[str, Any] = {}
        if reference is not None:
            reference_frames, width = _csv_shape(
                reference,
                require_finite=bool(
                    reference_config.get("require_finite_numeric_values", True)
                ),
            )
            expected_columns = int(reference_config.get("expected_columns", 36))
            if width != expected_columns:
                raise Stage2Error(
                    f"Reference CSV must have {expected_columns} columns: {reference}"
                )
            if (
                bool(reference_config.get("require_exact_source_frame_count", False))
                and reference_frames != frames
            ):
                raise Stage2Error(
                    f"Reference/source frame mismatch for {relative}: "
                    f"{reference_frames} != {frames}"
                )
            reference_values = {
                "reference_relative_path": reference.relative_to(
                    reference_root
                ).as_posix(),
                "reference_sha256": sha256_file(reference),
                "reference_size_bytes": reference.stat().st_size,
                "reference_frames": reference_frames,
            }
        sequence = SequenceSpec(
            sequence_id=sequence_id,
            relative_path=relative,
            source_sha256=sha256_file(path),
            source_size_bytes=path.stat().st_size,
            frames=frames,
            fps=fps,
            duration_s=frames / fps,
            **reference_values,
        )
        sequence.validate()
        sequences.append(sequence)

    if enforce_expected_inventory:
        expected_files = int(dataset["expected_files"])
        expected_frames = int(dataset["expected_frames"])
        if (
            len(sequences) != expected_files
            or sum(item.frames for item in sequences) != expected_frames
        ):
            raise Stage2Error(
                "LAFAN inventory differs from the frozen 77-file/496672-frame contract"
            )
        matched = [
            item for item in sequences if item.reference_relative_path is not None
        ]
        expected_intersection = reference_config.get("expected_intersection_files")
        if expected_intersection is not None and len(matched) != int(
            expected_intersection
        ):
            raise Stage2Error(
                "Reference/source intersection differs from the frozen "
                f"{int(expected_intersection)}-file contract"
            )
        unmatched_references = set(reference_by_stem) - {
            Path(item.relative_path).stem.lower() for item in sequences
        }
        if unmatched_references:
            raise Stage2Error(
                "Reference corpus contains filenames outside the LAFAN inventory: "
                + ", ".join(sorted(unmatched_references))
            )
        expected_actors = {
            str(value).lower()
            for value in reference_config.get("expected_actor_ids", [])
        }
        observed_actors = {_actor_id(item.relative_path) for item in matched}
        if expected_actors and observed_actors != expected_actors:
            raise Stage2Error(
                "Reference actor inventory mismatch: "
                f"expected {sorted(expected_actors)}, observed {sorted(observed_actors)}"
            )
    return sequences


def inventory_sha256(sequences: Sequence[SequenceSpec]) -> str:
    return _canonical_sha256([item.as_dict() for item in sequences])


def _file_receipt(root: Path, path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise Stage2Error(f"Required immutable evidence is missing: {path}")
    return {
        "path": _portable(root, path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _directory_receipts(root: Path, directory: Path) -> list[dict[str, Any]]:
    """Hash every persistent file below a non-symlinked evidence directory."""

    if not directory.is_dir() or directory.is_symlink():
        raise Stage2Error(f"Required evidence directory is invalid: {directory}")
    receipts: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise Stage2Error(f"Symlink is forbidden in evidence directory: {path}")
        if path.is_file():
            receipts.append(_file_receipt(root, path))
    if not receipts:
        raise Stage2Error(f"Evidence directory is empty: {directory}")
    return receipts


def _validate_unitree_reference_contract(
    root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Recompute the complete schema-v2 reference/asset/FK contract."""

    reference = config.get("reference_corpus", {})
    contract = reference.get("contract", {})
    if contract.get("required") is not True:
        return {}
    manifest_path = root / str(contract["manifest"])
    reference_config_path = root / str(contract["config"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference_config = load_yaml(reference_config_path)
    if (
        manifest.get("schema_version") != 2
        or manifest.get("role") != "external_reference_not_verified_ground_truth"
        or manifest.get("verified_official_ground_truth") is not False
        or manifest.get("eligible_for_runtime_comparison") is not False
        or manifest.get("upstream_revision") != reference.get("revision")
        or reference_config.get("upstream", {}).get("revision")
        != reference.get("revision")
        # This is an immutable Stage-1 corpus audit.  Its historical hard-stop
        # flag remains false; Stage-2 authorization lives in stage2.yaml and is
        # validated independently instead of rewriting provenance.
        or reference_config.get("schema_version") != 1
        or reference_config.get("full_lafan_authorized") is not False
    ):
        raise Stage2Error("Unitree reference schema/role/revision contract failed")

    pinned = manifest.get("pinned_resources", {})
    expected_resources = set(contract.get("required_pinned_resources", ()))
    if set(pinned) != expected_resources:
        raise Stage2Error("Unitree pinned-resource inventory changed")
    resource_receipts: dict[str, Any] = {}
    for name, entry in sorted(pinned.items()):
        path = root / str(entry.get("path", ""))
        receipt = _file_receipt(root, path)
        if (
            receipt["sha256"] != entry.get("sha256")
            or receipt["bytes"] != int(entry.get("size_bytes", -1))
        ):
            raise Stage2Error(f"Unitree pinned resource changed: {name}")
        resource_receipts[name] = receipt

    revision_root = root / "data/external/unitree_lafan1_reference" / str(
        reference["revision"]
    )
    configured_reference_root = (root / str(reference["root"])).resolve()
    if configured_reference_root != (revision_root / "g1").resolve():
        raise Stage2Error(
            "Configured Unitree CSV root differs from the audited pinned revision"
        )
    csv_files = sorted((revision_root / "g1").glob("*.csv"))
    inventory_digest = hashlib.sha256()
    for path in csv_files:
        inventory_digest.update(path.name.encode("utf-8"))
        inventory_digest.update(bytes.fromhex(sha256_file(path)))
    inventory = manifest.get("g1_csv_inventory", {})
    if (
        len(csv_files) != int(reference["expected_files"])
        or int(inventory.get("files", -1)) != len(csv_files)
        or inventory.get("aggregate_sha256") != inventory_digest.hexdigest()
    ):
        raise Stage2Error("Unitree G1 CSV aggregate changed")

    assets = manifest.get("asset_binding", {})
    asset_keys = {
        "canonical_evaluator_urdf_path": "canonical_evaluator_urdf_sha256",
        "canonical_evaluator_scene_path": "canonical_evaluator_scene_sha256",
        "evaluator_manifest_path": "evaluator_manifest_sha256",
    }
    asset_receipts: dict[str, Any] = {}
    for path_key, hash_key in asset_keys.items():
        path = root / str(assets.get(path_key, ""))
        receipt = _file_receipt(root, path)
        if receipt["sha256"] != assets.get(hash_key):
            raise Stage2Error(f"Unitree evaluator binding changed: {path_key}")
        asset_receipts[path_key] = receipt
    if assets.get("reference_urdf") != pinned.get("g1_urdf"):
        raise Stage2Error("Unitree reference URDF is not the pinned G1 resource")

    from .calibration import load_evaluator_protocol
    from .unitree_reference import (
        compare_urdf_kinematics,
        compare_urdf_to_mujoco_fk,
    )

    evaluator = load_evaluator_protocol(
        root / str(assets["evaluator_manifest_path"])
    )
    if evaluator.get("robot_xml_sha256") != assets.get(
        "canonical_evaluator_scene_sha256"
    ):
        raise Stage2Error("Evaluator protocol is not bound to the Unitree comparison scene")
    reference_urdf = root / str(pinned["g1_urdf"]["path"])
    canonical_urdf = root / str(assets["canonical_evaluator_urdf_path"])
    canonical_scene = root / str(assets["canonical_evaluator_scene_path"])
    urdf_audit = compare_urdf_kinematics(reference_urdf, canonical_urdf)
    fk_parameters = dict(contract["fk_audit"])
    fk_audit = compare_urdf_to_mujoco_fk(
        reference_urdf,
        canonical_scene,
        random_sample_count=int(fk_parameters["random_sample_count"]),
        seed=int(fk_parameters["seed"]),
        position_tolerance_m=float(fk_parameters["position_tolerance_m"]),
        rotation_tolerance_rad=float(fk_parameters["rotation_tolerance_rad"]),
    )
    if (
        urdf_audit.get("kinematic_contract_equivalent") is not True
        or urdf_audit.get("canonical_g1_order_match") is not True
        or fk_audit.get("kinematic_fk_equivalent") is not True
    ):
        raise Stage2Error("Unitree/canonical G1 kinematic equivalence audit failed")
    source_binding = manifest.get("source_binding", {})
    if (
        source_binding.get("alignment_basis")
        != "same basename and frame indices 0:600"
        or source_binding.get("byte_identical_human_source_verified") is not False
        or source_binding.get("exact_timestamp_identity_claimed") is not False
    ):
        raise Stage2Error("Unitree comparison must remain frame-index-only")
    adapter = manifest.get("adapter", {})
    adapter_path = root / str(adapter.get("implementation_path", ""))
    if sha256_file(adapter_path) != adapter.get("implementation_sha256"):
        raise Stage2Error("Unitree adapter implementation changed")
    receipt = {
        "schema_version": 2,
        "manifest": _file_receipt(root, manifest_path),
        "config": _file_receipt(root, reference_config_path),
        "revision": reference["revision"],
        "role": manifest["role"],
        "alignment_contract": "same_basename_and_frame_indices_only",
        "byte_identical_human_source_verified": False,
        "exact_timestamp_identity_claimed": False,
        "pinned_resources": resource_receipts,
        "g1_csv_inventory": {
            "files": len(csv_files),
            "aggregate_sha256": inventory_digest.hexdigest(),
        },
        "asset_binding": asset_receipts,
        "adapter": _file_receipt(root, adapter_path),
        "urdf_kinematic_audit": urdf_audit,
        "urdf_to_mujoco_fk_audit": fk_audit,
    }
    receipt["contract_sha256"] = _canonical_sha256(receipt)
    return receipt


def _legacy_stage1_timing_from_output(path: Path) -> tuple[float, float, int]:
    from .schemas import CanonicalG1

    motion = CanonicalG1.load(path)
    frames = len(motion.qpos)
    duration = frames / motion.fps
    steady = float(motion.metadata.get("steady_end_to_end_total_s", 0.0))
    if steady <= 0.0 or duration <= 0.0:
        raise Stage2Error(
            f"Accepted Stage-1 output lacks a positive timing estimate: {path}"
        )
    protocol = motion.metadata.get("timing_protocol", {})
    if (
        not isinstance(protocol, Mapping)
        or int(protocol.get("warmup_runs", 0)) < 1
        or int(protocol.get("measured_runs", 0)) < 3
    ):
        raise Stage2Error(
            f"Accepted Stage-1 output lacks the 1-warmup/3-measured timing grade: {path}"
        )
    startup_value = motion.metadata.get("initialization_time_s")
    if startup_value is None:
        startup_value = protocol.get("initialization_time_s", 0.0)
    startup = float(startup_value)
    return steady / duration, max(0.0, startup), max(1, path.stat().st_size // frames)


def _validate_stage1_timing_evidence(
    root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind Stage 2 to the exact successful formal Stage-1 timing campaign."""

    gate = config["stage1_gate"]
    timing_gate = gate.get("formal_timing_campaign", {})
    if timing_gate.get("required") is not True:
        return {}
    from .schemas import CanonicalG1, CanonicalHuman, RunManifest
    from .stage1_timing_campaign import (
        _load_state,
        build_campaign_plan,
        validate_job_artifacts,
    )
    from .native_target_capture import tensor_sha256

    state_path = root / str(timing_gate["state_manifest"])
    sidecar_path = state_path.with_suffix(state_path.suffix + ".sha256")
    state = _load_state(state_path)
    campaign_plan = build_campaign_plan(root, timing_gate["config"])
    if (
        state.get("status") != "succeeded"
        or state.get("plan_sha256") != campaign_plan.get("plan_sha256")
        or state.get("campaign_id") != campaign_plan.get("campaign_id")
    ):
        raise Stage2Error("Stage-1 formal timing campaign is not an exact success")
    state_jobs = state.get("jobs", [])
    if [job.get("name") for job in state_jobs] != list(
        campaign_plan["permutation"]["computed_and_frozen_order"]
    ):
        raise Stage2Error("Stage-1 formal timing job order changed")
    if set(job.get("name") for job in state_jobs) != set(STAGE1_FORMAL_RETARGETERS):
        raise Stage2Error("Stage-1 timing campaign must contain exactly six retargeters")

    source_manifest_path = root / str(campaign_plan["sequence_manifest"])
    source_manifest = load_yaml(source_manifest_path)
    source_path = root / str(campaign_plan["canonical_source_path"])
    canonical_source = CanonicalHuman.load(source_path)
    source_duration_s = len(canonical_source.timestamps) / canonical_source.fps
    expected_source_sha256 = str(campaign_plan["canonical_source_embedded_sha256"])
    if (
        sha256_file(source_manifest_path)
        != campaign_plan["sequence_manifest_sha256"]
        or sha256_file(source_path) != campaign_plan["canonical_source_file_sha256"]
        or source_manifest.get("cropped_sha256") != expected_source_sha256
    ):
        raise Stage2Error("Stage-1 timing campaign source identity changed")

    evidence: dict[str, Any] = {}
    for state_job in state_jobs:
        name = str(state_job["name"])
        formal_entry = gate["formal_outputs"][name]
        expected_revision = str(formal_entry["revision"])
        if (
            config.get("production_design", {}).get("required") is True
            and config.get("methods", {}).get(name, {}).get("registered_revision")
            != expected_revision
        ):
            raise Stage2Error(
                f"Stage-1/current Stage-2 revision differs for {name}"
            )
        configured_manifest = root / str(formal_entry["run_manifest"])
        if not configured_manifest.is_file():
            raise Stage2Error(f"Accepted Stage-1 RunManifest is missing for {name}")
        run_manifest = RunManifest.load(configured_manifest)
        output_path = _resolve_manifest_path(root, str(run_manifest.output_path))
        registered_directory = root / str(formal_entry["registered_directory"])
        if (
            run_manifest.run_id != formal_entry["run_id"]
            or not output_path.is_file()
            or not _path_below(output_path, registered_directory)
        ):
            raise Stage2Error(
                f"Stage-1 accepted RunManifest/revision binding failed for {name}"
            )
        # The accepted RunManifest is the sole output-path authority.  A
        # successful campaign may preserve/materialize its accepted trajectory
        # below an attempt_* directory inside the registered revision.  Reuse
        # every other frozen campaign field while validating those exact bytes.
        validation_job = dict(state_job)
        validation_job["output_path"] = _portable(root, output_path)
        validation_job["manifest_path"] = _portable(root, configured_manifest)
        receipt = validate_job_artifacts(
            root,
            campaign_plan,
            validation_job,
            expected_cpu_affinity=state.get("cpu_affinity"),
        )
        recorded_receipt = dict(state_job.get("receipt", {}))
        # Only the path may legitimately differ; content/config/timing and all
        # remaining receipts must still equal the successful campaign record.
        if recorded_receipt.get("output_sha256") != receipt.get("output_sha256"):
            raise Stage2Error(f"Stage-1 timing output bytes changed for {name}")
        recorded_receipt["output_path"] = receipt["output_path"]
        if recorded_receipt != receipt:
            raise Stage2Error(f"Stage-1 timing receipt changed for {name}")
        manifest_path = root / str(receipt["manifest_path"])
        if (
            manifest_path.resolve() != configured_manifest.resolve()
            or _resolve_manifest_path(root, str(run_manifest.output_path)).resolve()
            != output_path.resolve()
            or run_manifest.output_sha256 != receipt["output_sha256"]
        ):
            raise Stage2Error(
                f"Stage-1 accepted RunManifest/revision binding failed for {name}"
            )
        motion = CanonicalG1.load(output_path)
        registered_frames = int(
            formal_entry.get(
                "expected_output_frames", campaign_plan["source_frames"]
            )
        )
        completion_field = str(
            formal_entry.get("completion_metadata_field", "completion_status")
        )
        registered_completion = motion.metadata.get(completion_field)
        if (
            registered_completion != "succeeded"
            or len(motion.qpos) != registered_frames
            or str(
                motion.metadata.get("canonical_source_sha256")
                or motion.metadata.get("source_sha256")
                or ""
            )
            != expected_source_sha256
            or tensor_sha256(np.asarray(motion.qpos, dtype=np.float64))
            != receipt["timing_contract"]["canonical_qpos_sha256"]
        ):
            raise Stage2Error(f"Stage-1 output/source/timing binding failed for {name}")

        expected_manifest_config_sha256 = (
            sha256_file(root / "configs/protomotions_v3.yaml")
            if name == "protomotions-v3"
            else str(receipt["implementation_sha256"])
        )
        if (
            run_manifest.source_sha256 != expected_source_sha256
            or run_manifest.output_sha256 != receipt["output_sha256"]
            or run_manifest.config_sha256 != expected_manifest_config_sha256
        ):
            raise Stage2Error(f"Stage-1 run manifest binding failed for {name}")
        timing_path = root / str(receipt["timing_path"])
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
        measured_repetitions = timing.get(
            "measured_warm", timing.get("measured", [])
        )
        method_duration_s = registered_frames / float(motion.fps)
        expected_raw_rtf = [
            float(item["steady_end_to_end_total_s"]) / method_duration_s
            for item in measured_repetitions
        ]
        expected_native_rtf = [
            float(item["native_total_s"]) / method_duration_s
            for item in measured_repetitions
        ]
        raw_rtf = [float(value) for value in timing["end_to_end_rtf_raw"]]
        native_raw_rtf = [float(value) for value in timing["native_core_rtf_raw"]]
        if (
            len(raw_rtf) != 3
            or len(native_raw_rtf) != 3
            or not np.isfinite(raw_rtf).all()
            or not np.isfinite(native_raw_rtf).all()
            or not np.allclose(raw_rtf, expected_raw_rtf, rtol=0.0, atol=1e-12)
            or not np.allclose(
                native_raw_rtf, expected_native_rtf, rtol=0.0, atol=1e-12
            )
        ):
            raise Stage2Error(f"Stage-1 timing lacks three measured values for {name}")
        median_rtf = float(np.median(raw_rtf))
        if not np.isclose(
            median_rtf,
            float(timing["end_to_end_rtf_median"]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise Stage2Error(f"Stage-1 timing median was not recomputed for {name}")
        cold = timing.get("cold", {})
        cold_process_wall_s = float(timing.get("cold_process_wall_s", 0.0))
        cold_in_memory_s = float(cold.get("steady_end_to_end_total_s", 0.0))
        if (
            cold_process_wall_s <= 0.0
            or cold_in_memory_s <= 0.0
            or not np.isfinite(cold_process_wall_s)
            or not np.isfinite(cold_in_memory_s)
            or cold_process_wall_s + 1e-12 < cold_in_memory_s
            or not np.isclose(
                float(timing.get("native_core_rtf_median", np.nan)),
                float(np.median(native_raw_rtf)),
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise Stage2Error(f"Stage-1 fresh-process timing is missing for {name}")
        fresh_startup_s = cold_process_wall_s - cold_in_memory_s
        implementation = state_job.get("implementation_receipt")
        if not isinstance(implementation, Mapping):
            raise Stage2Error(f"Stage-1 implementation receipt is missing for {name}")
        environment = implementation.get("environment")
        if not isinstance(environment, Mapping):
            raise Stage2Error(f"Stage-1 environment receipt is missing for {name}")
        configured_environment = str(config["methods"][name]["environment"])
        if environment.get("environment") != configured_environment:
            raise Stage2Error(f"Stage-2 environment differs from Stage-1 timing for {name}")
        current_python = _resolve_python(configured_environment)
        current_history = current_python.parent.parent / "conda-meta/history"
        if (
            str(current_python) != environment.get("python_path")
            or sha256_file(current_python) != environment.get("python_sha256")
            or not current_history.is_file()
            or sha256_file(current_history) != environment.get("conda_history_sha256")
        ):
            raise Stage2Error(f"Stage-1 timing environment changed for {name}")
        provenance_paths = (
            [output_path.parent / "formal/evidence.json"]
            if name == "protomotions-v3"
            else [output_path.parent / "provenance.json"]
        )
        provenance_paths.extend(
            root / str(path) for path in state_job.get("extra_provenance_paths", [])
        )
        provenance = [_file_receipt(root, path) for path in provenance_paths]
        method_evidence = {
            "registered_revision": expected_revision,
            "output": _file_receipt(root, output_path),
            "run_manifest": _file_receipt(root, manifest_path),
            "run_manifest_config_sha256": expected_manifest_config_sha256,
            "timing": _file_receipt(root, timing_path),
            "provenance": provenance,
            "implementation_receipt": implementation,
            "implementation_receipt_sha256": _canonical_sha256(implementation),
            "environment": dict(environment),
            "environment_provenance_sha256": _canonical_sha256(environment),
            "source_sha256": expected_source_sha256,
            "protocol": {
                "cold_processes": 1,
                "warmup_runs": 1,
                "measured_warm_runs": 3,
                "measured_end_to_end_rtf_raw": raw_rtf,
                "measured_end_to_end_rtf_median": median_rtf,
                "fresh_process_total_s": cold_process_wall_s,
                "fresh_process_in_memory_s": cold_in_memory_s,
                "fresh_process_startup_s": fresh_startup_s,
                "end_to_end_boundary": timing["protocol"][
                    "end_to_end_boundary"
                ],
                "native_core_boundary": timing["protocol"][
                    "native_core_boundary"
                ],
                "intermediate_and_final_artifact_writes_excluded": timing[
                    "protocol"
                ]["intermediate_and_final_artifact_writes_excluded"],
            },
            "retained_output_bytes_per_frame": max(
                1, output_path.stat().st_size // len(motion.qpos)
            ),
        }
        method_evidence["evidence_sha256"] = _canonical_sha256(method_evidence)
        evidence[name] = method_evidence
    value = {
        "campaign_state": _file_receipt(root, state_path),
        "campaign_state_sidecar": _file_receipt(root, sidecar_path),
        "campaign_plan_sha256": campaign_plan["plan_sha256"],
        "campaign_config": _file_receipt(root, root / str(timing_gate["config"])),
        "source_manifest": _file_receipt(root, source_manifest_path),
        "canonical_source": _file_receipt(root, source_path),
        "source_sha256": expected_source_sha256,
        "methods": evidence,
    }
    value["contract_sha256"] = _canonical_sha256(value)
    return value


def _validate_cost_probe(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    probe_config = config.get("preflight_cost_probe", {})
    if probe_config.get("required") is not True:
        return {}
    path = root / str(probe_config["manifest"])
    value = json.loads(path.read_text(encoding="utf-8"))
    payload = dict(value)
    recorded = payload.pop("payload_sha256", None)
    if value.get("schema_version") != 1 or recorded != _canonical_sha256(payload):
        raise Stage2Error("Stage-2 cost probe payload hash is invalid")
    for entry in value.get("bound_inputs", []):
        bound = root / str(entry.get("path", ""))
        if (
            not bound.is_file()
            or sha256_file(bound) != entry.get("sha256")
            or bound.stat().st_size != int(entry.get("bytes", -1))
        ):
            raise Stage2Error("Stage-2 cost probe input changed")
    bound_inputs = value.get("bound_inputs", [])
    if value.get("bound_inputs_sha256") != _canonical_sha256(bound_inputs):
        raise Stage2Error("Stage-2 cost probe bound-input inventory hash failed")
    canonical_receipt = value.get("pilot_canonical_source", {})
    canonical_path = _resolve_manifest_path(
        root, str(canonical_receipt.get("path", ""))
    )
    accepted_receipts = value.get("accepted_formal_outputs", {})
    if (
        _file_receipt(root, canonical_path) != canonical_receipt
        or canonical_receipt not in bound_inputs
        or set(accepted_receipts) != set(STAGE1_FORMAL_RETARGETERS)
    ):
        raise Stage2Error("Stage-2 cost probe source/output receipts are not exact")
    from .schemas import CanonicalHuman

    pilot_human = CanonicalHuman.load(canonical_path)
    expected_pilot_frames = len(pilot_human.timestamps)
    expected_pilot_duration_s = expected_pilot_frames / float(pilot_human.fps)
    expected_output_size_raw: dict[str, int] = {}
    for method in STAGE1_FORMAL_RETARGETERS:
        receipt = accepted_receipts[method]
        output_path = _resolve_manifest_path(root, str(receipt.get("path", "")))
        if _file_receipt(root, output_path) != receipt or receipt not in bound_inputs:
            raise Stage2Error("Stage-2 accepted formal-output receipt changed")
        expected_output_size_raw[method] = output_path.stat().st_size
    repetitions = int(probe_config.get("repetitions", 0))
    multiplier = float(probe_config.get("conservative_multiplier", 0.0))
    preparation_raw = value.get("source_preparation_wall_s_raw", [])
    evaluator_raw = value.get("evaluator_wall_s_raw", [])
    canonical_size_raw = value.get("canonical_source_bytes_raw", [])
    output_size_raw = value.get("accepted_output_bytes_raw", {})
    pilot_frames = int(value.get("pilot_frames", 0))
    duration_s = float(value.get("pilot_duration_s", 0.0))
    if (
        repetitions < 1
        or probe_config.get("statistic") != "median"
        or value.get("statistic") != "median"
        or int(value.get("repetitions", 0)) != repetitions
        or multiplier < 1.0
        or not np.isclose(
            float(value.get("conservative_multiplier", 0.0)),
            multiplier,
            rtol=0.0,
            atol=0.0,
        )
        or pilot_frames < 1
        or pilot_frames != expected_pilot_frames
        or not np.isfinite(duration_s)
        or duration_s <= 0.0
        or not np.isclose(
            duration_s, expected_pilot_duration_s, rtol=0.0, atol=1e-12
        )
        or len(preparation_raw) != repetitions
        or len(evaluator_raw) != repetitions
        or len(canonical_size_raw) != repetitions
        or not all(
            np.isfinite(float(item)) and float(item) > 0.0
            for item in (*preparation_raw, *evaluator_raw)
        )
        or not all(int(item) > 0 for item in canonical_size_raw)
        or set(output_size_raw) != set(STAGE1_FORMAL_RETARGETERS)
        or not all(int(item) > 0 for item in output_size_raw.values())
        or {
            str(method): int(size) for method, size in output_size_raw.items()
        }
        != expected_output_size_raw
    ):
        raise Stage2Error("Stage-2 cost probe raw measurement protocol is invalid")
    measured_values = {
        "measured_source_preparation_rtf": float(
            np.median(preparation_raw) / duration_s
        ),
        "measured_evaluator_rtf_per_method": float(
            np.median(evaluator_raw) / duration_s
        ),
        "measured_canonical_source_bytes_per_frame": float(
            max(int(item) for item in canonical_size_raw) / pilot_frames
        ),
        "measured_output_bytes_per_frame_max": float(
            max(int(item) for item in output_size_raw.values()) / pilot_frames
        ),
    }
    if any(
        not np.isclose(
            float(value.get(key, 0.0)), expected, rtol=0.0, atol=1e-12
        )
        for key, expected in measured_values.items()
    ):
        raise Stage2Error("Stage-2 measured cost values were not derived from raw data")
    allowances = config.get("budget_allowances", {})
    storage = config.get("storage_projection", {})
    method_entries = config.get("methods", {})
    conservative_values = {
        "conservative_source_preparation_rtf": max(
            measured_values["measured_source_preparation_rtf"] * multiplier,
            float(allowances.get("source_preparation_rtf", 0.0)),
        ),
        "conservative_evaluator_rtf_per_method": max(
            measured_values["measured_evaluator_rtf_per_method"] * multiplier,
            float(allowances.get("evaluator_rtf_per_method", 0.0)),
        ),
        "conservative_canonical_source_bytes_per_frame": max(
            measured_values["measured_canonical_source_bytes_per_frame"] * multiplier,
            float(storage.get("canonical_source_bytes_per_frame", 0.0)),
        ),
        "conservative_output_bytes_per_frame": max(
            measured_values["measured_output_bytes_per_frame_max"] * multiplier,
            max(
                (float(entry.get("retained_bytes_per_frame", 0.0)) for entry in method_entries.values()),
                default=0.0,
            ),
        ),
    }
    if any(
        not np.isclose(
            float(value.get(key, 0.0)), expected, rtol=0.0, atol=1e-12
        )
        for key, expected in conservative_values.items()
    ):
        raise Stage2Error(
            "Stage-2 conservative cost values do not equal measurement×margin/floor"
        )
    if config.get("production_design", {}).get("required") is True:
        measurement = value.get("measurement_environment", {})
        environment_name = str(probe_config.get("measurement_environment", ""))
        python = _resolve_python(environment_name)
        history = python.parent.parent / "conda-meta/history"
        if (
            measurement.get("environment") != environment_name
            or measurement.get("python") != _file_receipt(root, python)
            or measurement.get("conda_history") != _file_receipt(root, history)
            or measurement.get("thread_environment") != THREAD_ENVIRONMENT
            or measurement.get("thread_environment_sha256")
            != _canonical_sha256(THREAD_ENVIRONMENT)
            or value.get("measurement_hardware") != _hardware_identity(config)
        ):
            raise Stage2Error("Stage-2 prep/evaluator measurement provenance changed")
        config_receipt = value.get("stage2_config", {})
        config_path = _resolve_manifest_path(
            root, str(config_receipt.get("path", ""))
        )
        if (
            _file_receipt(root, config_path) != config_receipt
            or load_yaml(config_path) != config
        ):
            raise Stage2Error("Stage-2 cost probe config binding changed")
        formal = _validate_stage1_formal_outputs(root, config)
        source_manifest_path = root / str(
            config["stage1_gate"]["canonical_source_manifest"]
        )
        source_manifest = load_yaml(source_manifest_path)
        production_canonical_path = root / str(source_manifest["canonical_path"])
        if canonical_path.resolve() != production_canonical_path.resolve():
            raise Stage2Error("Stage-2 cost probe measured a different canonical source")
        if {
            method: accepted_receipts[method] for method in STAGE1_FORMAL_RETARGETERS
        } != {
            method: _file_receipt(
                root, _resolve_manifest_path(root, str(formal[method]["path"]))
            )
            for method in STAGE1_FORMAL_RETARGETERS
        }:
            raise Stage2Error("Stage-2 cost probe did not measure six accepted outputs")
        expected_paths = [
            config_path,
            source_manifest_path,
            root / str(source_manifest["cropped_source_file"]),
            root / str(source_manifest["canonical_path"]),
            root / "manifests/evaluator.yaml",
            root / "src/retargeting_comparison/source.py",
            root / "src/retargeting_comparison/evaluator.py",
            root / "src/retargeting_comparison/method_worker.py",
            root / "src/retargeting_comparison/method_adapters.py",
            root / "src/retargeting_comparison/stage2.py",
            *(
                _resolve_manifest_path(root, str(formal[name]["path"]))
                for name in STAGE1_FORMAL_RETARGETERS
            ),
            *(
                _resolve_manifest_path(
                    root, str(formal[name]["run_manifest"]["path"])
                )
                for name in STAGE1_FORMAL_RETARGETERS
            ),
        ]
        expected_inputs = [_file_receipt(root, item) for item in expected_paths]
        if bound_inputs != expected_inputs:
            raise Stage2Error("Stage-2 cost probe bound-input set is not exact")
    required = (
        "measured_source_preparation_rtf",
        "conservative_source_preparation_rtf",
        "measured_evaluator_rtf_per_method",
        "conservative_evaluator_rtf_per_method",
        "measured_canonical_source_bytes_per_frame",
        "conservative_canonical_source_bytes_per_frame",
        "measured_output_bytes_per_frame_max",
        "conservative_output_bytes_per_frame",
    )
    if not all(
        isinstance(value.get(key), (int, float))
        and np.isfinite(float(value[key]))
        and float(value[key]) > 0.0
        for key in required
    ):
        raise Stage2Error("Stage-2 cost probe lacks finite measured/conservative values")
    for measured, conservative in (
        ("measured_source_preparation_rtf", "conservative_source_preparation_rtf"),
        ("measured_evaluator_rtf_per_method", "conservative_evaluator_rtf_per_method"),
        (
            "measured_canonical_source_bytes_per_frame",
            "conservative_canonical_source_bytes_per_frame",
        ),
        ("measured_output_bytes_per_frame_max", "conservative_output_bytes_per_frame"),
    ):
        if float(value[conservative]) < float(value[measured]):
            raise Stage2Error("A conservative Stage-2 cost probe is below its measurement")
    contention = value.get("parallel_contention_probe", {})
    rows = contention.get("results", [])
    registered_contention = probe_config.get("contention_probe", {})
    if (
        contention.get("method") != "omniretarget"
        or contention.get("levels") != [1, 2, 6]
        or int(contention.get("source_frames", 0)) < 20
        or (
            registered_contention
            and (
                contention.get("method") != registered_contention.get("method")
                or contention.get("environment")
                != registered_contention.get("environment")
                or contention.get("levels")
                != registered_contention.get("levels")
                or int(contention.get("repetitions_per_level", 0))
                != int(registered_contention.get("repetitions_per_level", -1))
                or int(contention.get("source_frames", 0))
                != int(registered_contention.get("source_frames", -1))
                or contention.get("cpu_assignment")
                != registered_contention.get("cpu_assignment")
                or not np.isclose(
                    float(contention.get("conservative_slowdown_multiplier", 0.0)),
                    float(
                        registered_contention.get(
                            "conservative_slowdown_multiplier", -1.0
                        )
                    ),
                    rtol=0.0,
                    atol=0.0,
                )
                or contention.get("apply_to_all_methods_with_workers_gt_one")
                is not registered_contention.get(
                    "apply_to_all_methods_with_workers_gt_one"
                )
            )
        )
        or contention.get("timing_boundary")
        != "canonical_source_file_to_canonical_g1_in_memory"
        or "--method omniretarget" not in str(contention.get("command_template", ""))
        or [row.get("workers") for row in rows] != [1, 2, 6]
        or contention.get("apply_to_all_methods_with_workers_gt_one") is not True
        or float(value.get("conservative_parallel_contention_factor", 0.0))
        != float(contention.get("conservative_parallel_contention_factor", -1.0))
        or float(value.get("conservative_parallel_contention_factor", 0.0)) < 1.0
        or float(contention.get("conservative_parallel_contention_factor", 0.0))
        < float(contention.get("observed_max_combined_slowdown", 0.0))
    ):
        raise Stage2Error("Stage-2 1/2/6 contention probe contract failed")
    source_frames = int(contention["source_frames"])
    repetitions = int(contention.get("repetitions_per_level", 0))
    artifact_root = _resolve_manifest_path(
        root, str(value.get("contention_artifact_root", ""))
    ).resolve()
    expected_artifact_parent = (root / "runs/stage2_cost_probe").resolve()
    if (
        artifact_root.parent != expected_artifact_parent
        or re.fullmatch(r"attempt-\d{3}", artifact_root.name) is None
    ):
        raise Stage2Error("Stage-2 contention artifact root is not confined")
    artifact_inventory = _directory_receipts(root, artifact_root)
    if (
        value.get("contention_artifact_inventory") != artifact_inventory
        or value.get("contention_artifact_inventory_sha256")
        != _canonical_sha256(artifact_inventory)
    ):
        raise Stage2Error("Stage-2 contention artifact inventory changed")
    contention_environment = str(registered_contention.get("environment", ""))
    contention_python = _resolve_python(contention_environment)
    expected_environment_receipts = {
        "python": _file_receipt(root, contention_python),
        "conda_history": _file_receipt(
            root, contention_python.parent.parent / "conda-meta/history"
        ),
    }
    for key, expected_receipt in expected_environment_receipts.items():
        if contention.get(key) != expected_receipt:
            raise Stage2Error(f"Stage-2 contention environment changed: {key}")
    measured_cpu_ids = [
        int(cpu)
        for cpu in value.get("measurement_hardware", {}).get(
            "production_cpu_ids", []
        )
    ]
    if len(measured_cpu_ids) < 6 or len(set(measured_cpu_ids)) != len(
        measured_cpu_ids
    ):
        raise Stage2Error("Stage-2 measurement hardware CPU lanes are invalid")
    for row in rows:
        workers = int(row["workers"])
        level_wall_raw = row.get("level_wall_s_raw", [])
        process_wall_raw = row.get("process_wall_s_raw", [])
        process_makespan_raw = row.get("process_makespan_s_raw", [])
        steady_raw = row.get("steady_in_memory_s_raw", [])
        process_receipts = row.get("process_receipts", [])
        timing_receipts = row.get("timing_receipts", [])
        if (
            row.get("distinct_cpu_ids") != measured_cpu_ids[:workers]
            or len(row.get("distinct_cpu_ids", [])) != workers
            or len(set(row["distinct_cpu_ids"])) != workers
            or repetitions < 1
            or len(level_wall_raw) != repetitions
            or len(process_wall_raw) != repetitions
            or len(process_makespan_raw) != repetitions
            or len(steady_raw) != repetitions
            or len(process_receipts) != repetitions
            or len(timing_receipts) != repetitions
            or any(
                len(receipts) != workers
                or any(
                    receipt.get("exit_code") != 0
                    or not np.isfinite(float(receipt.get("process_wall_s", 0.0)))
                    or float(receipt.get("process_wall_s", 0.0)) <= 0.0
                    for receipt in receipts
                )
                for receipts in process_receipts
            )
            or any(len(receipts) != workers for receipts in timing_receipts)
            or float(row.get("level_wall_s_median", 0.0)) <= 0.0
            or float(row.get("process_makespan_s_median", 0.0)) <= 0.0
            or float(row.get("steady_worker_s_median", 0.0)) <= 0.0
            or float(row.get("process_makespan_slowdown_vs_one_process", 0.0))
            < 1.0
            or float(row.get("steady_per_worker_slowdown_vs_one_process", 0.0))
            < 1.0
            or float(row.get("combined_slowdown_vs_one_process", 0.0)) < 1.0
            or float(row.get("per_worker_slowdown_vs_one_process", 0.0)) < 1.0
        ):
            raise Stage2Error("Stage-2 contention probe row is invalid")
        receipt_steady: list[list[float]] = []
        for repetition_index, repetition_receipts in enumerate(process_receipts):
            for worker_index, receipt in enumerate(repetition_receipts):
                worker_root = (
                    artifact_root
                    / f"level-{workers}"
                    / f"rep-{repetition_index}"
                    / f"worker-{worker_index}"
                )
                command = receipt.get("command", [])
                expected_command = [
                    "taskset",
                    "--cpu-list",
                    str(row["distinct_cpu_ids"][worker_index]),
                    str(contention_python),
                    "-m",
                    "retargeting_comparison.method_worker",
                    "--method",
                    "omniretarget",
                    "--repo-root",
                    str(root),
                    "--source",
                    str(canonical_path),
                    "--output",
                    str(worker_root / "canonical_g1.npz"),
                    "--work-dir",
                    str(worker_root / "work"),
                    "--timing-json",
                    str(worker_root / "timing.json"),
                    "--warmup-runs",
                    "0",
                    "--measured-runs",
                    "1",
                    "--max-frames",
                    str(source_frames),
                ]
                if (
                    command != expected_command
                    or receipt.get("command_sha256")
                    != _canonical_sha256(expected_command)
                ):
                    raise Stage2Error(
                        "Stage-2 contention process command is not exact"
                    )
                for log_name in ("stdout", "stderr"):
                    log_key = f"{log_name}_log"
                    log_receipt = receipt.get(log_key, {})
                    expected_log_path = worker_root / f"{log_name}.log"
                    if _file_receipt(root, expected_log_path) != log_receipt:
                        raise Stage2Error(
                            "Stage-2 contention process-log receipt is invalid"
                        )
        for repetition_index, repetition_receipts in enumerate(timing_receipts):
            repetition_steady: list[float] = []
            for worker_index, receipt in enumerate(repetition_receipts):
                worker_root = (
                    artifact_root
                    / f"level-{workers}"
                    / f"rep-{repetition_index}"
                    / f"worker-{worker_index}"
                )
                timing_payload = receipt.get("payload", {})
                repetitions_payload = timing_payload.get("repetitions", [])
                timing_file_receipt = receipt.get("timing_json", {})
                timing_file = worker_root / "timing.json"
                if (
                    _file_receipt(root, timing_file) != timing_file_receipt
                    or json.loads(timing_file.read_text(encoding="utf-8"))
                    != timing_payload
                    or receipt.get("payload_sha256")
                    != _canonical_sha256(timing_payload)
                    or len(repetitions_payload) != 1
                ):
                    raise Stage2Error("Stage-2 raw timing receipt is invalid")
                repetition_value = repetitions_payload[0]
                steady = float(
                    repetition_value.get("steady_end_to_end_total_s", 0.0)
                )
                if (
                    repetition_value.get("role") != "measured"
                    or repetition_value.get("timing_boundary")
                    != "canonical_source_file_to_canonical_g1_in_memory"
                    or int(repetition_value.get("frame_count", -1)) != source_frames
                    or repetition_value.get("cpu_affinity")
                    != [int(row["distinct_cpu_ids"][worker_index])]
                    or int(repetition_value.get("thread_limit", -1)) != 1
                    or not np.isfinite(steady)
                    or steady <= 0.0
                ):
                    raise Stage2Error("Stage-2 raw timing receipt is invalid")
                artifact_path = _resolve_manifest_path(
                    root, str(repetition_value.get("timing_artifact_path", ""))
                )
                if (
                    not artifact_path.is_file()
                    or not _path_below(artifact_path, worker_root)
                    or sha256_file(artifact_path)
                    != repetition_value.get("timing_artifact_sha256")
                ):
                    raise Stage2Error("Stage-2 timing witness artifact changed")
                repetition_steady.append(steady)
            receipt_steady.append(repetition_steady)
        receipt_process_wall = [
            [float(receipt["process_wall_s"]) for receipt in receipts]
            for receipts in process_receipts
        ]
        expected_individual_max = [max(values) for values in receipt_process_wall]
        expected_makespan = [float(value) for value in level_wall_raw]
        if (
            not np.allclose(process_wall_raw, receipt_process_wall, rtol=0.0, atol=1e-12)
            or not np.allclose(steady_raw, receipt_steady, rtol=0.0, atol=1e-12)
            or not np.allclose(
                process_makespan_raw, expected_makespan, rtol=0.0, atol=1e-12
            )
            or not np.allclose(
                row.get("max_individual_process_wall_s_raw", []),
                expected_individual_max,
                rtol=0.0,
                atol=1e-12,
            )
            or any(
                float(level_wall) + 1e-12 < float(makespan)
                for level_wall, makespan in zip(
                    level_wall_raw, expected_individual_max
                )
            )
            or not np.isclose(
                float(row["level_wall_s_median"]),
                float(np.median(level_wall_raw)),
                rtol=0.0,
                atol=1e-12,
            )
            or not np.isclose(
                float(row["process_makespan_s_median"]),
                float(np.median(expected_makespan)),
                rtol=0.0,
                atol=1e-12,
            )
            or not np.isclose(
                float(row["steady_worker_s_median"]),
                float(np.median(receipt_steady)),
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise Stage2Error("Stage-2 contention raw timing derivation is invalid")
    serial_makespan = float(rows[0]["process_makespan_s_median"])
    serial_steady = float(rows[0]["steady_worker_s_median"])
    observed = 1.0
    for row in rows:
        process_slowdown = max(
            1.0, float(row["process_makespan_s_median"]) / serial_makespan
        )
        steady_slowdown = max(
            1.0, float(row["steady_worker_s_median"]) / serial_steady
        )
        combined = max(process_slowdown, steady_slowdown)
        observed = max(observed, combined)
        if (
            not np.isclose(
                float(row["process_makespan_slowdown_vs_one_process"]),
                process_slowdown,
                rtol=0.0,
                atol=1e-12,
            )
            or not np.isclose(
                float(row["steady_per_worker_slowdown_vs_one_process"]),
                steady_slowdown,
                rtol=0.0,
                atol=1e-12,
            )
            or not np.isclose(
                float(row["combined_slowdown_vs_one_process"]),
                combined,
                rtol=0.0,
                atol=1e-12,
            )
            or not np.isclose(
                float(row["per_worker_slowdown_vs_one_process"]),
                combined,
                rtol=0.0,
                atol=1e-12,
            )
            or not np.isclose(
                float(row["observed_aggregate_speedup"]),
                float(row["workers"]) / combined,
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise Stage2Error("Stage-2 contention slowdown derivation is invalid")
    margin = float(contention.get("conservative_slowdown_multiplier", 0.0))
    expected_factor = observed * margin
    if (
        margin < 1.0
        or not np.isclose(
            float(contention.get("observed_max_combined_slowdown", 0.0)),
            observed,
            rtol=0.0,
            atol=1e-12,
        )
        or not np.isclose(
            float(contention.get("observed_max_per_worker_slowdown", 0.0)),
            observed,
            rtol=0.0,
            atol=1e-12,
        )
        or not np.isclose(
            float(contention.get("conservative_parallel_contention_factor", 0.0)),
            expected_factor,
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise Stage2Error("Stage-2 contention conservative factor is invalid")
    return {**value, "manifest": _file_receipt(root, path)}


def resolve_methods(
    repo_root: str | Path,
    config: Mapping[str, Any],
    *,
    readiness_overrides: Mapping[str, bool] | None = None,
    timing_overrides: Mapping[str, tuple[float, float]] | None = None,
    stage1_timing_evidence: Mapping[str, Any] | None = None,
    stage1_formal_evidence: Mapping[str, Any] | None = None,
    cost_probe: Mapping[str, Any] | None = None,
) -> tuple[list[MethodSpec], list[dict[str, str]]]:
    """Resolve conditional methods and never invent a missing timing estimate."""

    root = Path(repo_root).resolve()
    physical_cores = int(config["scheduling"]["physical_cores"])
    overrides = dict(readiness_overrides or {})
    timing = dict(timing_overrides or {})
    formal_timing = dict((stage1_timing_evidence or {}).get("methods", {}))
    formal_outputs = dict(stage1_formal_evidence or {})
    probe = dict(cost_probe or {})
    methods: list[MethodSpec] = []
    exclusions: list[dict[str, str]] = []
    for name in config["scheduling"]["phase_order"]:
        entry = config["methods"][name]
        readiness = str(entry["readiness"])
        accepted_output: Path | None = None
        if name in overrides:
            ready = bool(overrides[name])
            readiness_evidence = "explicit test/runtime override"
        elif readiness == "always":
            ready = True
            readiness_evidence = "frozen core operating point"
        elif readiness == "reference_intersection":
            ready = True
            readiness_evidence = "resolved per filename intersection"
        elif name in formal_outputs:
            if (
                config.get("production_design", {}).get("required") is True
                and entry.get("readiness_manifest")
                != formal_outputs[name].get("run_manifest", {}).get("path")
            ):
                raise Stage2Error(
                    f"Stage-2 readiness manifest differs from accepted Stage-1: {name}"
                )
            accepted_output = root / str(formal_outputs[name]["path"])
            ready = accepted_output.is_file()
            readiness_evidence = str(
                formal_outputs[name].get("run_manifest", {}).get("path", accepted_output)
            )
        elif readiness == "stage1_output":
            accepted_output = root / str(entry["readiness_path"])
            ready = accepted_output.is_file()
            readiness_evidence = str(accepted_output)
        else:
            raise Stage2Error(f"Unknown readiness rule {readiness!r} for {name}")
        if not ready:
            exclusions.append(
                {
                    "method": name,
                    "status": "na",
                    "reason": "no accepted Stage-1 canonical output; not integration-ready",
                    "evidence": readiness_evidence,
                }
            )
            continue

        rtf = entry.get("rtf")
        startup = entry.get("startup_s_per_sequence")
        bytes_per_frame = int(entry["retained_bytes_per_frame"])
        timing_evidence_sha256 = ""
        environment_provenance_sha256 = ""
        if name in timing:
            rtf, startup = timing[name]
        elif name in formal_timing:
            evidence = formal_timing[name]
            protocol = evidence["protocol"]
            rtf = protocol["measured_end_to_end_rtf_median"]
            startup = protocol["fresh_process_startup_s"]
            observed_bytes = int(evidence["retained_output_bytes_per_frame"])
            bytes_per_frame = max(bytes_per_frame, observed_bytes)
            timing_evidence_sha256 = str(evidence["evidence_sha256"])
            environment_provenance_sha256 = str(
                evidence["environment_provenance_sha256"]
            )
        elif (rtf is None or startup is None) and accepted_output is not None:
            # Test-only compatibility. Production requires the bound formal
            # timing campaign and never accepts output metadata as projection input.
            rtf, startup, observed_bytes = _legacy_stage1_timing_from_output(
                accepted_output
            )
            bytes_per_frame = max(bytes_per_frame, observed_bytes)
        if probe:
            bytes_per_frame = max(
                bytes_per_frame,
                int(np.ceil(float(probe["conservative_output_bytes_per_frame"]))),
            )
        if rtf is None or startup is None:
            raise Stage2Error(
                f"No measured projection inputs for integration-ready {name}"
            )
        contention_factor = 1.0
        if int(entry["workers"]) > 1 and probe:
            contention_factor = float(
                probe["conservative_parallel_contention_factor"]
            )
            if contention_factor < 1.0:
                raise Stage2Error("Parallel contention factor must be conservative")
            rtf = float(rtf) * contention_factor
            startup = float(startup) * contention_factor
        policy_files = list(entry.get("policy_files", []))
        if config.get("production_design", {}).get("required") is True:
            policy_file_receipts = [
                _file_receipt(root, root / str(path)) for path in policy_files
            ]
        else:
            policy_file_receipts = policy_files
        policy_basis = {
            "method": name,
            "stratum": entry["stratum"],
            "public_pipeline": bool(entry["public_pipeline"]),
            "environment": entry["environment"],
            "registered_revision": entry.get("registered_revision"),
            "policy_files": policy_file_receipts,
            "scale_policy": entry.get("scale_policy"),
            "reference_policy": entry.get("reference_policy"),
        }
        policy_sha256 = _canonical_sha256(policy_basis)
        method_contract_basis = {
            "policy": policy_basis,
            "policy_sha256": policy_sha256,
            "timing_evidence_sha256": timing_evidence_sha256,
            "environment_provenance_sha256": environment_provenance_sha256,
            "rtf": float(rtf),
            "startup_s": float(startup),
            "retained_bytes_per_frame": bytes_per_frame,
            "retained_bytes_per_sequence": int(entry["retained_bytes_per_sequence"]),
            "parallel_contention_factor": contention_factor,
        }
        method = MethodSpec(
            method=name,
            stratum=str(entry["stratum"]),
            public_pipeline=bool(entry["public_pipeline"]),
            environment=str(entry["environment"]),
            workers=int(entry["workers"]),
            rtf=float(rtf),
            startup_s=float(startup),
            retained_bytes_per_frame=bytes_per_frame,
            retained_bytes_per_sequence=int(entry["retained_bytes_per_sequence"]),
            estimate_source=str(entry["estimate_source"]),
            method_contract_sha256=_canonical_sha256(method_contract_basis),
            timing_evidence_sha256=timing_evidence_sha256,
            environment_provenance_sha256=environment_provenance_sha256,
            policy_sha256=policy_sha256,
            registered_revision=str(entry.get("registered_revision", "")),
            parallel_contention_factor=contention_factor,
        )
        method.validate(physical_cores)
        methods.append(method)
    return methods, exclusions


def _eligible_sequences(
    method: MethodSpec, sequences: Sequence[SequenceSpec]
) -> list[SequenceSpec]:
    if method.stratum == "external_reference":
        return [item for item in sequences if item.reference_relative_path is not None]
    return list(sequences)


def deterministic_schedule(
    sequences: Sequence[SequenceSpec],
    methods: Sequence[MethodSpec],
    phase_order: Sequence[str],
) -> tuple[list[ScheduledJob], float, list[dict[str, Any]]]:
    """Build fixed LPT worker lanes; method phases never oversubscribe cores."""

    by_name = {method.method: method for method in methods}
    jobs: list[ScheduledJob] = []
    phase_offset = 0.0
    phase_summaries: list[dict[str, Any]] = []
    phase_index = 0
    for name in phase_order:
        method = by_name.get(name)
        if method is None:
            continue
        candidates = _eligible_sequences(method, sequences)
        estimated = [
            (
                method.startup_s + method.rtf * sequence.duration_s,
                sequence.sequence_id,
                sequence,
            )
            for sequence in candidates
        ]
        estimated.sort(key=lambda item: (-item[0], item[1]))
        lane_ends = [0.0] * method.workers
        for runtime, _, sequence in estimated:
            lane = min(
                range(method.workers), key=lambda index: (lane_ends[index], index)
            )
            local_start = lane_ends[lane]
            local_end = local_start + runtime
            lane_ends[lane] = local_end
            job_id = f"{method.method}__{sequence.sequence_id}"
            job_contract = {
                "job_id": job_id,
                "method": method.method,
                "sequence": sequence.as_dict(),
                "stratum": method.stratum,
                "method_contract_sha256": method.method_contract_sha256,
                "policy_sha256": method.policy_sha256,
                "reference_relative_path": (
                    sequence.reference_relative_path
                    if method.stratum == "external_reference"
                    else None
                ),
            }
            jobs.append(
                ScheduledJob(
                    job_id=job_id,
                    method=method.method,
                    sequence_id=sequence.sequence_id,
                    phase_index=phase_index,
                    worker_index=lane,
                    estimated_runtime_s=runtime,
                    projected_start_s=phase_offset + local_start,
                    projected_end_s=phase_offset + local_end,
                    stratum=method.stratum,
                    reference_relative_path=(
                        sequence.reference_relative_path
                        if method.stratum == "external_reference"
                        else None
                    ),
                    method_contract_sha256=method.method_contract_sha256,
                    policy_sha256=method.policy_sha256,
                    job_contract_sha256=_canonical_sha256(job_contract),
                )
            )
        makespan = max(lane_ends, default=0.0)
        phase_summaries.append(
            {
                "phase_index": phase_index,
                "method": method.method,
                "workers": method.workers,
                "jobs": len(candidates),
                "projected_makespan_s": makespan,
                "worker_loads_s": lane_ends,
            }
        )
        phase_offset += makespan
        phase_index += 1
    return jobs, phase_offset, phase_summaries


def project_resources(
    sequences: Sequence[SequenceSpec],
    methods: Sequence[MethodSpec],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    jobs, method_wall_s, phases = deterministic_schedule(
        sequences, methods, config["scheduling"]["phase_order"]
    )
    storage = config["storage_projection"]
    raw_storage = int(storage["fixed_manifest_and_summary_bytes"])
    if bool(storage["retain_raw_inputs_in_incremental_budget"]):
        raw_storage += sum(item.source_size_bytes for item in sequences)
        raw_storage += sum(item.reference_size_bytes for item in sequences)
    raw_storage += sum(
        int(storage["canonical_source_bytes_per_sequence"])
        + item.frames * int(storage["canonical_source_bytes_per_frame"])
        for item in sequences
    )
    by_id = {item.sequence_id: item for item in sequences}
    by_method = {item.method: item for item in methods}
    for job in jobs:
        sequence = by_id[job.sequence_id]
        method = by_method[job.method]
        frames = (
            min(sequence.frames, int(sequence.reference_frames))
            if method.stratum == "external_reference"
            else sequence.frames
        )
        raw_storage += (
            method.retained_bytes_per_sequence
            + frames * method.retained_bytes_per_frame
        )

    allowances = config.get("budget_allowances", {})
    total_source_duration_s = sum(item.duration_s for item in sequences)
    preparation_workers = max(1, int(allowances.get("source_preparation_workers", 1)))
    preparation_wall_s = (
        total_source_duration_s * float(allowances.get("source_preparation_rtf", 0.0))
        + len(sequences)
        * float(allowances.get("source_preparation_startup_s_per_sequence", 0.0))
    ) / preparation_workers
    evaluator_workers = max(1, int(allowances.get("evaluator_workers", 1)))
    evaluator_work_s = sum(
        by_id[job.sequence_id].duration_s
        * float(allowances.get("evaluator_rtf_per_method", 0.0))
        + float(allowances.get("evaluator_startup_s_per_job", 0.0))
        for job in jobs
    )
    evaluator_wall_s = evaluator_work_s / evaluator_workers
    failure_runtime_fraction = float(allowances.get("failure_runtime_fraction", 0.0))
    if not 0.0 <= failure_runtime_fraction < 1.0:
        raise Stage2Error("failure_runtime_fraction must be in [0,1)")
    base_wall_s = method_wall_s + preparation_wall_s + evaluator_wall_s
    failure_runtime_allowance_s = base_wall_s * failure_runtime_fraction
    raw_wall_s = base_wall_s + failure_runtime_allowance_s

    raw_storage += int(allowances.get("analysis_fixed_retained_bytes", 0))
    raw_storage += len(jobs) * int(allowances.get("analysis_retained_bytes_per_job", 0))
    failure_storage_fraction = float(allowances.get("failure_storage_fraction", 0.0))
    if not 0.0 <= failure_storage_fraction < 1.0:
        raise Stage2Error("failure_storage_fraction must be in [0,1)")
    failure_storage_allowance_bytes = int(raw_storage * failure_storage_fraction)
    raw_storage += failure_storage_allowance_bytes

    budget = config["budget"]
    safety = float(budget["safety_factor"])
    safe_wall_h = raw_wall_s * safety / 3600.0
    safe_storage_gb = raw_storage * safety / 1_000_000_000.0
    wall_limit = float(budget["wall_time_hours"])
    storage_limit = float(budget["retained_storage_gb"])
    return {
        "production_passes_per_job": int(
            config["scheduling"]["production_passes_per_job"]
        ),
        "timing_repetitions_in_stage2": 1,
        "raw_projected_wall_s": raw_wall_s,
        "raw_projected_wall_hours": raw_wall_s / 3600.0,
        "raw_projected_retained_bytes": raw_storage,
        "raw_projected_retained_gb": raw_storage / 1_000_000_000.0,
        "safety_factor": safety,
        "safe_projected_wall_hours": safe_wall_h,
        "safe_projected_retained_gb": safe_storage_gb,
        "wall_limit_hours": wall_limit,
        "storage_limit_gb": storage_limit,
        "within_wall_budget": safe_wall_h <= wall_limit,
        "within_storage_budget": safe_storage_gb <= storage_limit,
        "within_budget": safe_wall_h <= wall_limit and safe_storage_gb <= storage_limit,
        "wall_components_s": {
            "method_phases": method_wall_s,
            "source_preparation": preparation_wall_s,
            "evaluation": evaluator_wall_s,
            "failure_allowance": failure_runtime_allowance_s,
        },
        "storage_failure_allowance_bytes": failure_storage_allowance_bytes,
        "parallel_contention_factors": {
            method.method: method.parallel_contention_factor for method in methods
        },
        "phase_summaries": phases,
        "jobs": [job.as_dict() for job in jobs],
    }


def _selection_order(
    sequences: Sequence[SequenceSpec], seed: str
) -> list[SequenceSpec]:
    ranked = sorted(
        sequences,
        key=lambda item: (
            hashlib.sha256(
                f"{seed}\0{item.relative_path}\0{item.source_sha256}".encode()
            ).hexdigest(),
            item.sequence_id,
        ),
    )
    references = [item for item in ranked if item.reference_relative_path is not None]
    if references and ranked[0] not in references:
        anchor = references[0]
        ranked.remove(anchor)
        ranked.insert(0, anchor)
    return ranked


def _method_design_sha256(
    methods: Sequence[MethodSpec], config: Mapping[str, Any]
) -> str:
    return _canonical_sha256(
        {
            "methods": [asdict(method) for method in methods],
            "phase_order": config["scheduling"]["phase_order"],
            "production_design": config.get("production_design", {}),
        }
    )


def select_budgeted_design(
    sequences: Sequence[SequenceSpec],
    methods: Sequence[MethodSpec],
    config: Mapping[str, Any],
) -> tuple[str, list[SequenceSpec], dict[str, Any], dict[str, Any] | None]:
    """Use Full LAFAN whenever it fits; otherwise choose a frozen hash prefix."""

    full_projection = project_resources(sequences, methods, config)
    full_inventory = inventory_sha256(sequences)
    method_digest = _method_design_sha256(methods, config)
    if full_projection["within_budget"]:
        design_id = f"full-lafan1-{full_inventory[:12]}-{method_digest[:12]}"
        return design_id, list(sequences), full_projection, None

    dataset = config["dataset"]
    ranked = _selection_order(sequences, str(dataset["selection_seed"]))
    minimum = int(dataset["reduced_min_sequences"])
    accepted: tuple[list[SequenceSpec], dict[str, Any]] | None = None
    for count in range(1, len(ranked)):
        candidate = ranked[:count]
        projection = project_resources(candidate, methods, config)
        if projection["within_budget"]:
            accepted = (candidate, projection)
    if accepted is None or len(accepted[0]) < minimum:
        raise Stage2Error(
            "Neither Full LAFAN nor the minimum deterministic reduced design fits the hard budget"
        )
    selected, projection = accepted
    subset_hash = inventory_sha256(selected)
    design_id = (
        f"reduced-lafan1-n{len(selected):02d}-{subset_hash[:12]}-{method_digest[:12]}"
    )
    fallback = {
        "trigger": "full_projection_exceeded_hard_budget_after_1.5x_safety",
        "full_inventory_sha256": full_inventory,
        "full_sequence_count": len(sequences),
        "full_projection": {
            key: value for key, value in full_projection.items() if key != "jobs"
        },
        "selection_rule": dataset["reduced_selection"],
        "selection_seed": dataset["selection_seed"],
        "selected_sequence_ids": [item.sequence_id for item in selected],
        "result_metrics_used_for_selection": False,
    }
    return design_id, selected, projection, fallback


def simplify_methods_before_dataset_reduction(
    sequences: Sequence[SequenceSpec],
    methods: Sequence[MethodSpec],
    config: Mapping[str, Any],
) -> tuple[list[MethodSpec], list[dict[str, Any]], dict[str, Any]]:
    """Apply the pre-registered method exclusion order before source reduction.

    The decision uses timing/storage projections only.  Quality metrics and
    per-sequence retargeting results are unavailable at planning time and are
    expressly forbidden as selection inputs.  If the all-method Full-LAFAN
    design already fits, no method is removed.
    """

    active = list(methods)
    initial = project_resources(sequences, active, config)
    simplification = config.get("budget_simplification", {})
    always_exclude = bool(simplification.get("exclude_registered_method_for_production"))
    if initial["within_budget"] and not always_exclude:
        return active, [], initial

    if simplification.get("policy") != (
        "exclude_registered_high_cost_methods_before_reducing_dataset"
    ):
        return active, [], initial
    if simplification.get("result_metrics_used_for_decision") is not False:
        raise Stage2Error(
            "Budget method simplification must explicitly forbid result-metric selection"
        )

    exclusions: list[dict[str, Any]] = []
    current = initial
    for name in simplification.get("method_exclusion_order", []):
        match = next((method for method in active if method.method == name), None)
        if match is None:
            continue
        before = current
        active = [method for method in active if method.method != name]
        if not active:
            raise Stage2Error(
                "Budget simplification cannot remove every Stage-2 method"
            )
        current = project_resources(sequences, active, config)
        exclusions.append(
            {
                "method": name,
                "status": "excluded_from_stage2_for_budget",
                "reason": (
                    "pre-registered high-cost method exclusion applied before "
                    "any dataset reduction"
                ),
                "selection_inputs": "Stage-1 timing and retained-storage projection only",
                "result_metrics_used_for_selection": False,
                "stage1_evidence_retained": bool(
                    simplification.get("stage1_evidence_retained", False)
                ),
                "safe_full_wall_hours_before": before["safe_projected_wall_hours"],
                "safe_full_wall_hours_after": current["safe_projected_wall_hours"],
                "safe_full_storage_gb_before": before["safe_projected_retained_gb"],
                "safe_full_storage_gb_after": current["safe_projected_retained_gb"],
                "full_design_fits_after_exclusion": current["within_budget"],
            }
        )
        if current["within_budget"] and not always_exclude:
            break
    return active, exclusions, initial


def _validate_authorization(config: Mapping[str, Any]) -> None:
    authorization = config.get("authorization", {})
    if authorization.get("explicitly_authorized") is not True:
        raise Stage2Error("Stage 2 lacks explicit user authorization")
    if authorization.get("full_lafan_authorized") is not True:
        raise Stage2Error("The Stage-2 Full-LAFAN authorization flag is not true")
    if authorization.get("scope") != "run_after_stage1_go_under_frozen_budget":
        raise Stage2Error(
            "Stage-2 authorization scope is missing or broader than approved"
        )


def _load_stage1_gate(
    root: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any], Path]:
    gate = config["stage1_gate"]
    path = root / gate["validation_manifest"]
    if gate.get("require_bound_validation") is True:
        from .stage1_publication import resolve_bound_stage1_validation

        value = resolve_bound_stage1_validation(root)
        if (
            value.get("binding_status") != "verified"
            or value.get("decision") not in set(gate["accepted_decisions"])
            or value.get("validation") is None
        ):
            raise Stage2Error(
                "Stage 1 lacks a verified schema-v5 verdict bound to the current "
                f"publication evidence: {value.get('reason')}"
            )
        checks = value["validation"].get("checks", {})
        if gate.get("require_all_checks") is True and (
            not checks or not all(result is True for result in checks.values())
        ):
            raise Stage2Error("Bound Stage-1 GO verdict contains a failed check")
        return value, path

    # Explicitly test-only compatibility path. Production config requires the
    # schema-v5 publication binding above.
    if not path.is_file():
        raise Stage2Error(f"Stage-1 validation manifest is missing: {path}")
    value = json.loads(path.read_text())
    if value.get("decision") not in set(gate["accepted_decisions"]):
        raise Stage2Error(f"Stage 1 is not GO (found {value.get('decision')!r})")
    checks = value.get("checks", {})
    if gate.get("require_all_checks") is True and (
        not checks or not all(result is True for result in checks.values())
    ):
        raise Stage2Error("Stage-1 GO manifest contains a failed or non-boolean check")
    return value, path


def _validate_stage1_formal_outputs(
    root: Path, config: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Fail closed unless all six frozen Stage-1 operating points are exact.

    A GO label alone is insufficient launch evidence.  This gate independently
    loads each configured canonical output and binds its byte hash, method
    identity, timeline, validity mask, and finite 29-DoF trajectory into the
    Stage-2 plan.  Later budget simplification may remove only a registered
    high-cost method; readiness resolution may not do so.
    """

    gate = config["stage1_gate"]
    if gate.get("require_formal_outputs") is not True:
        return {}
    entries = gate.get("formal_outputs", {})
    expected_methods = tuple(config["scheduling"]["phase_order"][:6])
    if expected_methods != STAGE1_FORMAL_RETARGETERS:
        raise Stage2Error(
            "Stage-2 logical method IDs/order do not match the accepted Stage-1 "
            f"operating points: expected {STAGE1_FORMAL_RETARGETERS!r}"
        )
    if set(entries) != set(expected_methods):
        raise Stage2Error(
            "Stage-1 formal-output gate must name exactly the six retargeter methods"
        )
    expected_frames = int(gate["expected_frames"])
    from .schemas import CanonicalG1, CanonicalHuman
    from .schemas import RunManifest, RunStatus

    strict_source_identity = gate.get("require_strict_source_identity") is True
    expected_source_hash = ""
    source_manifest_receipt: dict[str, Any] | None = None
    if gate.get("canonical_source_manifest"):
        source_manifest_path = root / str(gate["canonical_source_manifest"])
        source_manifest = load_yaml(source_manifest_path)
        source_path = root / str(source_manifest["canonical_path"])
        canonical_source = CanonicalHuman.load(source_path)
        if len(canonical_source.timestamps) != expected_frames:
            raise Stage2Error(
                "Canonical Stage-1 source does not have the frozen frame count"
            )
        expected_fps = float(canonical_source.fps)
        expected_source_hash = str(canonical_source.source_sha256)
        source_manifest_receipt = _file_receipt(root, source_manifest_path)
        if (
            strict_source_identity
            and source_manifest.get("cropped_sha256") != expected_source_hash
        ):
            raise Stage2Error("Stage-1 canonical/Pilot source SHA-256 differs")
    else:
        expected_fps = float(gate["expected_fps"])

    evidence: dict[str, dict[str, Any]] = {}
    source_hashes: set[str] = set()
    for method in expected_methods:
        entry = entries[method]
        accepted_run_manifest: RunManifest | None = None
        accepted_run_manifest_path: Path | None = None
        if strict_source_identity:
            accepted_run_manifest_path = root / str(entry["run_manifest"])
            if not accepted_run_manifest_path.is_file():
                raise Stage2Error(
                    f"Accepted Stage-1 run manifest is missing for {method}"
                )
            accepted_run_manifest = RunManifest.load(accepted_run_manifest_path)
            path = _resolve_manifest_path(root, str(accepted_run_manifest.output_path))
        else:
            path = root / str(entry["path"])
        revision = entry.get("revision")
        if strict_source_identity:
            registered_directory = root / str(entry["registered_directory"])
            configured_revision = (
                config.get("methods", {}).get(method, {}).get("registered_revision")
            )
            if (
                not revision
                or (
                    config.get("production_design", {}).get("required") is True
                    and configured_revision != revision
                )
                or accepted_run_manifest is None
                or accepted_run_manifest.status != RunStatus.SUCCEEDED
                or accepted_run_manifest.exit_code != 0
                or accepted_run_manifest.run_id != entry["run_id"]
                or not path.is_file()
                or not _path_below(path, registered_directory)
                or accepted_run_manifest.output_sha256 != sha256_file(path)
                or accepted_run_manifest.source_sha256 != expected_source_hash
            ):
                raise Stage2Error(f"Stage-1 revision/path contract failed for {method}")
        if not path.is_file():
            raise Stage2Error(f"Required Stage-1 formal output is missing: {path}")
        try:
            motion = CanonicalG1.load(path)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            raise Stage2Error(
                f"Invalid Stage-1 formal output for {method}: {path}"
            ) from error
        qpos = motion.qpos
        metadata_method = str(motion.metadata.get("method", ""))
        accepted_methods = {str(value) for value in entry["metadata_methods"]}
        if metadata_method not in accepted_methods:
            raise Stage2Error(
                f"Stage-1 method identity mismatch for {method}: {metadata_method!r}"
            )
        registered_frames = int(
            entry.get("expected_output_frames", expected_frames)
        )
        completion_field = str(
            entry.get("completion_metadata_field", "completion_status")
        )
        registered_completion = motion.metadata.get(completion_field)
        if (
            qpos.shape != (registered_frames, 36)
            or not all(bool(value) for value in motion.valid)
            or list(map(int, motion.source_frame_idx))
            != list(range(registered_frames))
            or abs(float(motion.fps) - expected_fps) > 1e-9
            or str(registered_completion) != "succeeded"
        ):
            raise Stage2Error(
                f"Stage-1 formal output timeline/completion contract failed for {method}"
            )
        # CanonicalG1.load already rejects NaN/Inf and malformed quaternions;
        # record the exact source identity when adapters expose it.
        source_hash = str(
            motion.metadata.get("source_sha256")
            or motion.metadata.get("canonical_source_sha256")
            or ""
        )
        if strict_source_identity and source_hash != expected_source_hash:
            raise Stage2Error(f"Stage-1 exact source identity is missing for {method}")
        if source_hash:
            if len(source_hash) != 64:
                raise Stage2Error(f"Malformed Stage-1 source hash for {method}")
            source_hashes.add(source_hash)
        evidence[method] = {
            "path": _portable(root, path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "frames": registered_frames,
            "fps": expected_fps,
            "metadata_method": metadata_method,
            "source_sha256": source_hash or None,
            "registered_revision": revision,
            "source_manifest": source_manifest_receipt,
            "run_manifest": (
                None
                if accepted_run_manifest_path is None
                else _file_receipt(root, accepted_run_manifest_path)
            ),
        }
    if len(source_hashes) > 1 or (
        strict_source_identity and source_hashes != {expected_source_hash}
    ):
        raise Stage2Error("The six Stage-1 formal outputs do not share one source hash")
    return evidence


def build_stage2_plan(
    repo_root: str | Path = ".",
    config_path: str | Path = "configs/stage2.yaml",
    *,
    sequences: Sequence[SequenceSpec] | None = None,
    output_path: str | Path | None = None,
    readiness_overrides: Mapping[str, bool] | None = None,
    timing_overrides: Mapping[str, tuple[float, float]] | None = None,
    enforce_expected_inventory: bool = True,
) -> dict[str, Any]:
    """Create an immutable preflight plan; this function never launches jobs."""

    root = Path(repo_root).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config = load_yaml(config_file)
    if int(config.get("schema_version", 0)) != SCHEMA_VERSION:
        raise Stage2Error("Unsupported Stage-2 configuration schema")
    _validate_authorization(config)
    stage1_bound_verdict, stage1_path = _load_stage1_gate(root, config)
    stage1_timing_evidence = _validate_stage1_timing_evidence(root, config)
    stage1_formal_evidence = _validate_stage1_formal_outputs(root, config)
    if stage1_timing_evidence:
        timing_outputs = {
            name: value["output"]["sha256"]
            for name, value in stage1_timing_evidence["methods"].items()
        }
        formal_outputs = {
            name: value["sha256"] for name, value in stage1_formal_evidence.items()
        }
        if timing_outputs != formal_outputs:
            raise Stage2Error("Formal-output and timing-campaign SHA-256 bindings differ")
    cost_probe = _validate_cost_probe(root, config)
    if cost_probe:
        allowances = config.setdefault("budget_allowances", {})
        allowances["source_preparation_rtf"] = float(
            cost_probe["conservative_source_preparation_rtf"]
        )
        allowances["evaluator_rtf_per_method"] = float(
            cost_probe["conservative_evaluator_rtf_per_method"]
        )
        config["storage_projection"]["canonical_source_bytes_per_frame"] = int(
            np.ceil(cost_probe["conservative_canonical_source_bytes_per_frame"])
        )
    unitree_reference_contract = _validate_unitree_reference_contract(root, config)
    inventory = (
        list(sequences)
        if sequences is not None
        else discover_lafan_sequences(
            root, config, enforce_expected_inventory=enforce_expected_inventory
        )
    )
    for sequence in inventory:
        sequence.validate()
    methods, exclusions = resolve_methods(
        root,
        config,
        readiness_overrides=readiness_overrides,
        timing_overrides=timing_overrides,
        stage1_timing_evidence=stage1_timing_evidence,
        stage1_formal_evidence=stage1_formal_evidence,
        cost_probe=cost_probe,
    )
    if not methods:
        raise Stage2Error("No integration-ready Stage-2 methods")
    if config["stage1_gate"].get("require_formal_outputs") is True:
        required = set(config["stage1_gate"]["formal_outputs"])
        resolved = {method.method for method in methods}
        if not required.issubset(resolved):
            raise Stage2Error(
                "All six Stage-1 formal methods must resolve before budget simplification; "
                f"missing {sorted(required - resolved)}"
            )
    all_ready_methods = list(methods)
    if config["stage1_gate"].get("require_formal_outputs") is True and tuple(
        method.method for method in all_ready_methods
    ) != (*STAGE1_FORMAL_RETARGETERS, "unitree-reference"):
        raise Stage2Error(
            "Pre-simplification Stage-2 methods must be exactly six Pilot "
            "retargeters plus the external reference"
        )
    methods, budget_exclusions, all_method_full_projection = (
        simplify_methods_before_dataset_reduction(inventory, methods, config)
    )
    exclusions.extend(budget_exclusions)
    production_design = config.get("production_design", {})
    if production_design.get("required") is True:
        expected = tuple(production_design.get("selected_methods", ()))
        if expected != PRODUCTION_METHODS:
            raise Stage2Error("Frozen production design does not name the exact five+reference set")
        selected_method_names = tuple(method.method for method in methods)
        if selected_method_names != expected:
            raise Stage2Error(
                f"Stage-2 method simplification produced {selected_method_names!r}, "
                f"not the frozen {expected!r}"
            )
        excluded = {
            row.get("method")
            for row in budget_exclusions
            if row.get("status") == "excluded_from_stage2_for_budget"
        }
        if excluded != set(production_design.get("required_budget_exclusions", ())):
            raise Stage2Error("Stage-2 cost exclusion set differs from the frozen design")
    design_id, selected, projection, dataset_fallback = select_budgeted_design(
        inventory, methods, config
    )
    fallback: dict[str, Any] | None = dataset_fallback
    if budget_exclusions:
        method_fallback = {
            "trigger": "all_method_full_projection_exceeded_hard_budget",
            "policy": config["budget_simplification"]["policy"],
            "method_exclusion_order": list(
                config["budget_simplification"]["method_exclusion_order"]
            ),
            "all_ready_methods": [method.method for method in all_ready_methods],
            "selected_methods": [method.method for method in methods],
            "method_exclusions": budget_exclusions,
            "all_method_full_projection": {
                key: value
                for key, value in all_method_full_projection.items()
                if key != "jobs"
            },
            "dataset_reduction_also_required": dataset_fallback is not None,
            "result_metrics_used_for_selection": False,
        }
        fallback = {
            "method_simplification": method_fallback,
            "dataset_simplification": dataset_fallback,
        }
    if not projection["within_budget"]:
        raise Stage2Error("Internal error: selected Stage-2 design exceeds budget")
    config_hash = sha256_file(config_file)
    repository = _repository_identity(root, config)
    _validate_repository_gate(repository, config)
    hardware = _hardware_identity(config)
    method_design_sha256 = _method_design_sha256(methods, config)
    method_evidence_contract_sha256 = _canonical_sha256(
        {
            "method_specs": [asdict(method) for method in methods],
            "production_design": production_design,
            "stage1_timing_contract_sha256": stage1_timing_evidence.get(
                "contract_sha256"
            ),
            "unitree_reference_contract_sha256": unitree_reference_contract.get(
                "contract_sha256"
            ),
        }
    )
    if not design_id.endswith(method_design_sha256[:12]):
        raise Stage2Error("Stage-2 design ID is not bound to the method/config digest")
    design_method_digest = method_design_sha256[:12]
    calibration_contract: dict[str, Any] = {
        "definition": config["canonicalization"]["common_scale"]["definition"],
        "reference_contract_sha256": unitree_reference_contract.get("contract_sha256"),
    }
    if production_design.get("required") is True:
        calibration_contract.update(
            {
                "evaluator_manifest": _file_receipt(
                    root, root / "manifests/evaluator.yaml"
                ),
                "calibration_implementation": _file_receipt(
                    root, root / "src/retargeting_comparison/calibration.py"
                ),
                "robot_model_implementation": _file_receipt(
                    root, root / "src/retargeting_comparison/robot_model.py"
                ),
                "stage2_preparation_implementation": _file_receipt(
                    root, Path(__file__)
                ),
            }
        )
    calibration_contract["contract_sha256"] = _canonical_sha256(calibration_contract)
    basis = {
        "config_sha256": config_hash,
        "repository": repository,
        "hardware": hardware,
        "stage1_validation_sha256": sha256_file(stage1_path),
        "stage1_bound_verdict": stage1_bound_verdict,
        "stage1_formal_evidence": stage1_formal_evidence,
        "stage1_timing_evidence": stage1_timing_evidence,
        "preflight_cost_probe": cost_probe,
        "unitree_reference_contract": unitree_reference_contract,
        "calibration_contract": calibration_contract,
        "full_inventory_sha256": inventory_sha256(inventory),
        "selected_inventory_sha256": inventory_sha256(selected),
        "method_specs": [asdict(method) for method in methods],
        "method_design_sha256": method_design_sha256,
        "method_evidence_contract_sha256": method_evidence_contract_sha256,
        "method_exclusions": exclusions,
        "projection": projection,
        "fallback": fallback,
    }
    basis_sha256 = _canonical_sha256(basis)
    destination = (
        Path(output_path)
        if output_path is not None
        else root / config["canonicalization"]["output_root"] / design_id / "plan.json"
    )
    if not destination.is_absolute():
        destination = root / destination
    if destination.exists():
        existing = json.loads(destination.read_text())
        if not verify_plan_hash(existing):
            raise Stage2Error(
                f"Existing Stage-2 plan has an invalid hash: {destination}"
            )
        if existing.get("plan_basis_sha256") != basis_sha256:
            raise Stage2Error(
                f"Refusing to overwrite a different Stage-2 plan: {destination}"
            )
        return existing

    plan: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stage": 2,
        "experiment_id": config["experiment_id"],
        "design_id": design_id,
        "created_at_utc": utc_now(),
        "repo_root": str(root),
        "config_path": str(config_file.relative_to(root)),
        "config_sha256": config_hash,
        "tracked_plan_path": str(
            Path(
                config.get("publication", {}).get(
                    "tracked_output_root", "stage2_results"
                )
            )
            / design_id
            / "STAGE2_PLAN.json"
        ),
        "repository": repository,
        "hardware": hardware,
        "plan_basis_sha256": basis_sha256,
        "stage1_validation_path": str(stage1_path.relative_to(root)),
        "stage1_validation_sha256": sha256_file(stage1_path),
        "stage1_bound_verdict": stage1_bound_verdict,
        "stage1_formal_evidence": stage1_formal_evidence,
        "stage1_timing_evidence": stage1_timing_evidence,
        "preflight_cost_probe": cost_probe,
        "unitree_reference_contract": unitree_reference_contract,
        "calibration_contract": calibration_contract,
        "method_design_sha256": method_design_sha256,
        "method_evidence_contract_sha256": method_evidence_contract_sha256,
        "design_id_method_digest": design_method_digest,
        "authorization": dict(config["authorization"]),
        "gates": {
            "user_authorization_verified": True,
            "stage1_go_verified": True,
            "preflight_within_wall_budget": projection["within_wall_budget"],
            "preflight_within_storage_budget": projection["within_storage_budget"],
            "launchable": True,
        },
        # A method-budget exclusion does not turn a complete 77-sequence
        # inventory into a reduced dataset.  Dataset scope is determined only
        # by the source-selection fallback.
        "dataset_kind": (
            "full_lafan1" if dataset_fallback is None else "reduced_lafan1"
        ),
        "full_inventory_sha256": inventory_sha256(inventory),
        "full_inventory_sequence_count": len(inventory),
        "selected_inventory_sha256": inventory_sha256(selected),
        "selected_sequences": [item.as_dict() for item in selected],
        "method_specs": [asdict(method) for method in methods],
        "selected_production_methods": [method.method for method in methods],
        "method_exclusions": exclusions,
        "evidence_strata": {
            "native_public_pipeline": [
                method.method
                for method in methods
                if method.stratum == "native_public_pipeline"
            ],
            "controlled_common_per_sequence_scale": [
                method.method
                for method in methods
                if method.stratum == "controlled_common_per_sequence_scale"
            ],
            "benchmark_public_retargeter_port": [
                method.method
                for method in methods
                if method.stratum == "benchmark_public_retargeter_port"
            ],
            "external_reference": [
                method.method
                for method in methods
                if method.stratum == "external_reference"
            ],
            "cross_stratum_ranking_forbidden": True,
        },
        "projection": projection,
        "fallback": fallback,
        "execution_contract": {
            "stage2_timing_repetitions": 1,
            "stage1_timing_protocol_reused_for_projection": True,
            "automatic_retry": False,
            "resume_only_incomplete_or_invalid_jobs": True,
            "successful_output_immutable": True,
            "launch_requires_exact_plan_sha256": True,
        },
    }
    plan["plan_sha256"] = _plan_sha256(plan)
    atomic_write_json(destination, plan)
    return plan


def _git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_identity(
    path: Path, *, generated_output_prefixes: Sequence[str] = ()
) -> dict[str, Any]:
    try:
        commit = _git_head(path)
        # Zero-context patches are byte-stable and avoid representing blank
        # context lines as space-only records when the frozen patch itself is
        # tracked by the main repository.
        diff_command = [
            "git",
            "-C",
            str(path),
            "diff",
            "--binary",
            "--unified=0",
            "HEAD",
            "--",
            ".",
        ]
        diff_command.extend(
            f":(exclude){prefix.rstrip('/')}/**" for prefix in generated_output_prefixes
        )
        diff = subprocess.run(
            diff_command,
            check=True,
            capture_output=True,
        ).stdout
        status = subprocess.run(
            [
                "git",
                "-C",
                str(path),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
        ).stdout
        untracked = subprocess.run(
            ["git", "-C", str(path), "ls-files", "--others", "--exclude-standard"],
            check=True,
            capture_output=True,
        ).stdout
        prefixes = tuple(
            prefix.rstrip("/") + "/" for prefix in generated_output_prefixes
        )
        if prefixes:
            status = b"\n".join(
                line
                for line in status.splitlines()
                if not line[3:].decode("utf-8", errors="replace").startswith(prefixes)
            )
            untracked = b"\n".join(
                line
                for line in untracked.splitlines()
                if not line.decode("utf-8", errors="replace").startswith(prefixes)
            )
        return {
            "commit": commit,
            "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "status_sha256": hashlib.sha256(status).hexdigest(),
            "untracked_path_list_sha256": hashlib.sha256(untracked).hexdigest(),
            "dirty": bool(status.strip()),
            "untracked_file_count": len(untracked.splitlines()),
        }
    except (OSError, subprocess.CalledProcessError):
        return {
            "commit": "unavailable",
            "tracked_diff_sha256": "unavailable",
            "status_sha256": "unavailable",
            "untracked_path_list_sha256": "unavailable",
            "dirty": True,
            "untracked_file_count": -1,
        }


def _repository_identity(
    root: Path, config: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Pin source, ignored upstream worktrees, environments, and key assets."""

    gate = dict((config or {}).get("repository_gate", {}))
    publication_prefix = str(
        (config or {})
        .get("publication", {})
        .get("tracked_output_root", "stage2_results")
    )
    identity: dict[str, Any] = {
        "main": _git_identity(root, generated_output_prefixes=(publication_prefix,))
    }
    external: dict[str, Any] = {}
    for name, entry in sorted(gate.get("external_worktrees", {}).items()):
        path = root / str(entry["path"])
        value = _git_identity(path)
        value["path"] = _portable(root, path)
        value["expected_commit"] = entry.get("expected_commit")
        patch = entry.get("tracked_patch")
        if patch:
            patch_path = root / str(patch)
            value["tracked_patch_path"] = str(patch)
            value["tracked_patch_sha256"] = (
                sha256_file(patch_path) if patch_path.is_file() else "missing"
            )
            value["external_diff_matches_tracked_patch"] = (
                value["tracked_patch_sha256"] == value["tracked_diff_sha256"]
            )
        external[str(name)] = value
    identity["external_worktrees"] = external

    assets: dict[str, Any] = {}
    for relative in gate.get("required_assets", []):
        path = root / str(relative)
        assets[str(relative)] = {
            "exists": path.is_file(),
            "sha256": sha256_file(path) if path.is_file() else "missing",
            "bytes": path.stat().st_size if path.is_file() else 0,
        }
    identity["required_assets"] = assets

    environments: dict[str, Any] = {}
    for environment in sorted(
        {
            str(entry["environment"])
            for entry in (config or {}).get("methods", {}).values()
        }
    ):
        try:
            python = _resolve_python(environment)
            history = python.parent.parent / "conda-meta" / "history"
            environments[environment] = {
                "python": str(python),
                "python_sha256": sha256_file(python),
                "conda_history": str(history),
                "conda_history_sha256": sha256_file(history)
                if history.is_file()
                else "missing",
            }
        except (OSError, Stage2Error):
            environments[environment] = {
                "python": "missing",
                "python_sha256": "missing",
            }
    identity["environments"] = environments
    return identity


def _validate_repository_gate(
    identity: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    gate = config.get("repository_gate", {})
    main = identity.get("main", {})
    if gate.get("require_clean_worktree") is True and (
        main.get("commit") == "unavailable" or main.get("dirty") is not False
    ):
        raise Stage2Error(
            "Stage-2 planning requires all main-repository code/config to be committed"
        )
    for name, value in identity.get("external_worktrees", {}).items():
        expected = value.get("expected_commit")
        if expected and value.get("commit") != expected:
            raise Stage2Error(f"External worktree commit mismatch for {name}")
        allows_frozen_patch = bool(value.get("tracked_patch_path"))
        if value.get("tracked_patch_sha256") == "missing":
            raise Stage2Error(f"Frozen external patch is missing for {name}")
        if (
            allows_frozen_patch
            and value.get("external_diff_matches_tracked_patch") is not True
        ):
            raise Stage2Error(
                f"External worktree diff does not equal its frozen patch: {name}"
            )
        if (
            gate.get("require_external_clean") is True
            and value.get("dirty") is not False
            and not allows_frozen_patch
        ):
            raise Stage2Error(
                f"External worktree is dirty without a frozen patch: {name}"
            )
    missing_assets = [
        path
        for path, value in identity.get("required_assets", {}).items()
        if not value["exists"]
    ]
    if missing_assets:
        raise Stage2Error(f"Required Stage-2 assets are missing: {missing_assets}")
    missing_environments = [
        name
        for name, value in identity.get("environments", {}).items()
        if value.get("python_sha256") == "missing"
    ]
    if missing_environments:
        raise Stage2Error(
            f"Required Stage-2 environments are missing: {missing_environments}"
        )


def _hardware_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    physical_pairs: set[tuple[str, str]] = set()
    physical_id: str | None = None
    core_id: str | None = None
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                if physical_id is not None and core_id is not None:
                    physical_pairs.add((physical_id, core_id))
                physical_id = core_id = None
            elif line.startswith("physical id"):
                physical_id = line.split(":", 1)[1].strip()
            elif line.startswith("core id"):
                core_id = line.split(":", 1)[1].strip()
        if physical_id is not None and core_id is not None:
            physical_pairs.add((physical_id, core_id))
    except OSError:
        pass
    physical_cores = len(physical_pairs) or (os.cpu_count() or 1)
    required_cores = int(config["scheduling"]["physical_cores"])
    if physical_cores < required_cores:
        raise Stage2Error(
            f"Stage-2 host has {physical_cores} physical cores; plan requires {required_cores}"
        )
    gpu_rows: list[str] = []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        gpu_rows = sorted(
            line.strip() for line in result.stdout.splitlines() if line.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        pass
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "physical_cores": physical_cores,
        "configured_physical_cores": required_cores,
        "production_cpu_ids": _physical_cpu_ids(required_cores),
        "gpus": gpu_rows,
    }


def _physical_cpu_ids(count: int) -> list[int]:
    """Choose one allowed logical CPU from each distinct physical core."""

    if count < 1 or not hasattr(os, "sched_getaffinity"):
        raise Stage2Error("Stage-2 CPU pinning requires Linux CPU affinity")
    selected: list[int] = []
    observed_cores: set[tuple[str, str]] = set()
    for cpu in sorted(os.sched_getaffinity(0)):
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            package = (topology / "physical_package_id").read_text().strip()
            core = (topology / "core_id").read_text().strip()
        except OSError:
            package, core = "logical", str(cpu)
        identity = (package, core)
        if identity in observed_cores:
            continue
        observed_cores.add(identity)
        selected.append(int(cpu))
        if len(selected) == count:
            return selected
    raise Stage2Error(
        f"Stage-2 needs {count} distinct allowed physical cores; found {len(selected)}"
    )


def _resolve_python(environment: str) -> Path:
    variable = f"RTCMP_{environment.upper().replace('-', '_')}_PYTHON"
    override = os.environ.get(variable)
    if override and Path(override).is_file():
        return Path(override).resolve()
    for base in (Path.home() / "anaconda3/envs", Path.home() / "miniconda3/envs"):
        candidate = base / environment / "bin/python"
        if candidate.is_file():
            return candidate.resolve()
    if os.environ.get("CONDA_DEFAULT_ENV") == environment:
        return Path(sys.executable).resolve()
    raise Stage2Error(
        f"Cannot locate conda environment {environment!r}; set {variable}"
    )


def _portable(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path.resolve())


def _controlled_pre_solver_target_contract(
    repo_root: str | Path,
    human: Any,
    sequence_manifest: Mapping[str, Any],
    variant: str,
) -> dict[str, Any]:
    """Rebuild the exact controlled target tensor before any IK solve.

    This deliberately depends only on the canonical source, the frozen
    per-source-sequence LS/anchor evidence, and the registered target-set
    policy.  Both the worker and the post-run validator call this function, so
    an arbitrary hash string cannot masquerade as pre-solver target evidence.
    """

    from .calibration import human_heading_yaw
    from .native_target_capture import tensor_sha256

    root = Path(repo_root).resolve()
    config_path = root / "configs/controlled_mink.yaml"
    controlled = load_yaml(config_path)
    if variant not in {"sparse", "dense"}:
        raise ValueError(f"Unknown controlled target variant: {variant}")
    specs = list(controlled["target_sets"][variant])
    calibration = sequence_manifest["common_scale"]
    local_scale = float(calibration["local_body_scale"])
    root_scale = float(calibration["root_displacement_scale"])
    alignment = np.asarray(
        calibration["root_alignment_translation_m"], dtype=np.float64
    )
    if alignment.shape != (3,) or not np.isfinite(alignment).all():
        raise ValueError("Controlled root alignment must be a finite xyz vector")
    indices = {
        name: index for index, name in enumerate(human.joint_names.astype(str))
    }
    missing = sorted(
        {str(spec["human_joint"]) for spec in specs}.difference(indices)
    )
    if missing:
        raise ValueError(f"Canonical source lacks controlled targets: {missing}")
    source_root0 = np.asarray(human.world_positions[0, 0], dtype=np.float64)
    robot_anchor = source_root0 * root_scale + alignment
    positions: list[list[np.ndarray]] = []
    for frame in range(len(human.timestamps)):
        source_root = np.asarray(human.world_positions[frame, 0], dtype=np.float64)
        scaled_root = robot_anchor + (source_root - source_root0) * root_scale
        positions.append(
            [
                scaled_root
                + (
                    np.asarray(
                        human.world_positions[frame, indices[str(spec["human_joint"])]],
                        dtype=np.float64,
                    )
                    - source_root
                )
                * local_scale
                for spec in specs
            ]
        )
    target_array = np.asarray(positions, dtype=np.float64)
    yaw = np.asarray(
        human_heading_yaw(human, np.arange(len(human.timestamps))),
        dtype=np.float64,
    )
    combined = np.concatenate(
        (target_array.reshape(len(yaw), -1), yaw[:, None]), axis=1
    )
    return {
        "target_labels": [str(spec["semantic"]) for spec in specs],
        "target_tensor_shape": list(combined.shape),
        "target_tensor_sha256": tensor_sha256(combined),
        "position_tensor_sha256": tensor_sha256(target_array),
        "root_yaw_tensor_sha256": tensor_sha256(yaw),
        "controlled_policy_sha256": sha256_file(config_path),
        "calibration_evidence_sha256": sequence_manifest[
            "calibration_evidence_sha256"
        ],
        "observed_boundary": "immediately_before_controlled_mink_frame_tasks",
    }


def prepare_sequence(
    repo_root: str | Path,
    run_root: str | Path,
    sequence: SequenceSpec,
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> Path:
    """Canonicalize one full BVH and freeze per-sequence common calibration."""

    root = Path(repo_root).resolve()
    output_root = Path(run_root).resolve() / "sources" / sequence.sequence_id
    manifest_path = output_root / "sequence.json"
    source_path = root / config["dataset"]["root"] / sequence.relative_path
    if not verify_plan_hash(plan):
        raise Stage2Error("Sequence preparation requires the exact immutable plan")
    calibration_contract = plan.get("calibration_contract", {})
    if not isinstance(calibration_contract, Mapping) or len(
        str(calibration_contract.get("contract_sha256", ""))
    ) != 64:
        raise Stage2Error("Stage-2 calibration contract is absent from the plan")
    preparation_contract = {
        "plan_sha256": plan["plan_sha256"],
        "design_id": plan["design_id"],
        "selected_inventory_sha256": plan["selected_inventory_sha256"],
        "source": sequence.as_dict(),
        "source_position_scale": float(config["dataset"]["source_position_scale"]),
        "canonicalization": dict(config["canonicalization"]),
        "reference_revision": str(config["reference_corpus"]["revision"]),
        "unitree_reference_contract_sha256": plan.get(
            "unitree_reference_contract", {}
        ).get("contract_sha256"),
        "calibration_contract_sha256": calibration_contract["contract_sha256"],
    }
    preparation_contract_sha256 = _canonical_sha256(preparation_contract)
    if sha256_file(source_path) != sequence.source_sha256:
        raise Stage2Error(f"Source changed after planning: {sequence.relative_path}")
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text())
        canonical_path = _resolve_manifest_path(
            root, str(existing.get("canonical_path", ""))
        )
        payload = dict(existing)
        payload_hash = payload.pop("payload_sha256", None)
        if (
            existing.get("schema_version") == SCHEMA_VERSION
            and payload_hash == _canonical_sha256(payload)
            and existing.get("plan_sha256") == plan["plan_sha256"]
            and existing.get("source_sha256") == sequence.source_sha256
            and existing.get("preparation_contract_sha256")
            == preparation_contract_sha256
            and canonical_path.is_file()
            and existing.get("canonical_sha256") == sha256_file(canonical_path)
        ):
            _validate_sequence_calibration_manifest(
                root, existing, sequence, config, plan
            )
            return manifest_path
        raise Stage2Error(
            f"Refusing to overwrite stale sequence provenance: {manifest_path}"
        )

    attempt_count = len(list((output_root / "attempts").glob("attempt-*")))
    attempt_root = output_root / "attempts" / f"attempt-{attempt_count + 1:03d}"
    canonical_path = attempt_root / "canonical_human.npz"
    helper_manifest = attempt_root / "canonicalization.yaml"

    from .calibration import shared_landmark_least_squares_calibration
    from .robot_model import CanonicalRobotModel, default_robot_scene
    from .source import canonicalize_source

    human = canonicalize_source(
        source_path,
        canonical_path,
        helper_manifest,
        position_scale=float(config["dataset"]["source_position_scale"]),
        origin_path=source_path,
        origin_frame_start=0,
        origin_frame_end=sequence.frames,
        repo_root=root,
    )
    if len(human.timestamps) != sequence.frames or abs(human.fps - sequence.fps) > 1e-9:
        raise Stage2Error(
            "Canonical source length/fps differs from the frozen inventory"
        )
    if sequence.reference_relative_path is not None:
        reference_path = (
            root / config["reference_corpus"]["root"] / sequence.reference_relative_path
        )
        if (
            not reference_path.is_file()
            or sha256_file(reference_path) != sequence.reference_sha256
        ):
            raise Stage2Error("Reference corpus file changed after planning")
    robot = CanonicalRobotModel(default_robot_scene(root))
    calibration = shared_landmark_least_squares_calibration(human, robot)
    local_scale = float(calibration["value"])
    root_scale = local_scale
    neutral = robot.semantic_positions(robot.model.qpos0.copy())
    root_alignment = neutral["root"] - root_scale * human.world_positions[0, 0]
    calibration_evidence = {
        "definition": config["canonicalization"]["common_scale"]["definition"],
        "value": local_scale,
        "local_body_scale": local_scale,
        "root_displacement_scale": root_scale,
        "source_landmarks": calibration["source_landmarks"],
        "robot_landmarks": calibration["robot_landmarks"],
        "landmark_weights": calibration["weights"],
        "least_squares_numerator_m2": calibration["numerator_m2"],
        "least_squares_denominator_m2": calibration["denominator_m2"],
        "weighted_residual_rmse_m": calibration["weighted_residual_rmse_m"],
        "source_heading_yaw_rad": calibration["source_heading_yaw_rad"],
        "source_heading_alignment_matrix": calibration[
            "source_heading_alignment_matrix"
        ],
        "head_to_toe_diagnostic_scale": calibration[
            "head_to_toe_diagnostic_scale"
        ],
        "source_head_to_toe_span_m": calibration["source_head_to_toe_span_m"],
        "robot_head_to_toe_span_m": calibration["robot_head_to_toe_span_m"],
        "root_alignment_translation_m": root_alignment.tolist(),
        "robot_asset_sha256": robot.sha256,
        "robot_joint_order_sha256": robot.joint_order_sha256,
        "calibration_contract_sha256": calibration_contract["contract_sha256"],
    }
    calibration_evidence_sha256 = _canonical_sha256(calibration_evidence)
    value = {
        "schema_version": SCHEMA_VERSION,
        "plan_sha256": plan["plan_sha256"],
        "design_id": plan["design_id"],
        "selected_inventory_sha256": plan["selected_inventory_sha256"],
        "sequence_id": sequence.sequence_id,
        "actor_id": _actor_id(sequence.relative_path),
        "source_path": _portable(root, source_path),
        "source_sha256": sequence.source_sha256,
        "source_frames": sequence.frames,
        "fps": sequence.fps,
        "canonical_path": _portable(root, canonical_path),
        "canonical_sha256": sha256_file(canonical_path),
        "canonical_content_sha256": human.source_sha256,
        "canonicalization_helper_manifest": _file_receipt(root, helper_manifest),
        "canonical_contract": config["canonicalization"]["contract"],
        "preparation_contract_sha256": preparation_contract_sha256,
        "common_scale": {
            **dict(config["canonicalization"]["common_scale"]),
            **calibration_evidence,
        },
        "calibration_evidence_sha256": calibration_evidence_sha256,
        "reference_relative_path": sequence.reference_relative_path,
        "reference_path": (
            None
            if sequence.reference_relative_path is None
            else _portable(
                root,
                root
                / config["reference_corpus"]["root"]
                / sequence.reference_relative_path,
            )
        ),
        "reference_sha256": sequence.reference_sha256,
        "reference_frames": sequence.reference_frames,
        "reference_native_fps_relative_tolerance": float(
            config["reference_corpus"].get("native_fps_relative_tolerance", 2.0e-5)
        ),
        "prepared_at_utc": utc_now(),
    }
    value["payload_sha256"] = _canonical_sha256(value)
    atomic_write_json(manifest_path, value)
    return manifest_path


def _validate_sequence_calibration_manifest(
    root: Path,
    value: Mapping[str, Any],
    sequence: SequenceSpec,
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> None:
    """Deterministically recompute LS scale/anchor and every bound input."""

    from .calibration import shared_landmark_least_squares_calibration
    from .robot_model import CanonicalRobotModel, default_robot_scene
    from .schemas import CanonicalHuman

    payload = dict(value)
    recorded_payload_sha256 = payload.pop("payload_sha256", None)
    source_path = root / config["dataset"]["root"] / sequence.relative_path
    expected_reference_path = (
        None
        if sequence.reference_relative_path is None
        else _portable(
            root,
            root
            / config["reference_corpus"]["root"]
            / sequence.reference_relative_path,
        )
    )
    expected_reference_fps_tolerance = float(
        config["reference_corpus"].get("native_fps_relative_tolerance", 2.0e-5)
    )
    try:
        recorded_reference_fps_tolerance = float(
            value["reference_native_fps_relative_tolerance"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise Stage2Error(
            "Sequence calibration lacks the frozen reference fps tolerance"
        ) from error
    preparation_contract = {
        "plan_sha256": plan["plan_sha256"],
        "design_id": plan["design_id"],
        "selected_inventory_sha256": plan["selected_inventory_sha256"],
        "source": sequence.as_dict(),
        "source_position_scale": float(config["dataset"]["source_position_scale"]),
        "canonicalization": dict(config["canonicalization"]),
        "reference_revision": str(config["reference_corpus"]["revision"]),
        "unitree_reference_contract_sha256": plan.get(
            "unitree_reference_contract", {}
        ).get("contract_sha256"),
        "calibration_contract_sha256": plan["calibration_contract"][
            "contract_sha256"
        ],
    }
    if (
        recorded_payload_sha256 != _canonical_sha256(payload)
        or value.get("plan_sha256") != plan.get("plan_sha256")
        or value.get("design_id") != plan.get("design_id")
        or value.get("selected_inventory_sha256")
        != plan.get("selected_inventory_sha256")
        or value.get("sequence_id") != sequence.sequence_id
        or value.get("actor_id") != _actor_id(sequence.relative_path)
        or value.get("source_path") != _portable(root, source_path)
        or value.get("source_sha256") != sequence.source_sha256
        or int(value.get("source_frames", -1)) != sequence.frames
        or not np.isclose(
            float(value.get("fps", 0.0)), sequence.fps, rtol=0.0, atol=1e-12
        )
        or value.get("canonical_contract") != config["canonicalization"]["contract"]
        or value.get("preparation_contract_sha256")
        != _canonical_sha256(preparation_contract)
        or value.get("reference_relative_path") != sequence.reference_relative_path
        or value.get("reference_path") != expected_reference_path
        or value.get("reference_sha256") != sequence.reference_sha256
        or value.get("reference_frames") != sequence.reference_frames
        or not np.isclose(
            recorded_reference_fps_tolerance,
            expected_reference_fps_tolerance,
            rtol=0.0,
            atol=0.0,
        )
    ):
        raise Stage2Error("Sequence calibration is not bound to this plan/source")
    canonical_path = _resolve_manifest_path(root, str(value["canonical_path"]))
    human = CanonicalHuman.load(canonical_path)
    if (
        sha256_file(canonical_path) != value.get("canonical_sha256")
        or human.source_sha256 != sequence.source_sha256
        or value.get("canonical_content_sha256") != human.source_sha256
        or len(human.timestamps) != sequence.frames
        or not np.isclose(human.fps, sequence.fps, rtol=0.0, atol=1e-9)
    ):
        raise Stage2Error("Sequence canonical source changed after preparation")
    helper = value.get("canonicalization_helper_manifest", {})
    helper_path = root / str(helper.get("path", ""))
    if _file_receipt(root, helper_path) != helper:
        raise Stage2Error("Canonicalization helper manifest changed")
    robot = CanonicalRobotModel(default_robot_scene(root))
    calibration = shared_landmark_least_squares_calibration(human, robot)
    scale = float(calibration["value"])
    root_alignment = (
        robot.semantic_positions(robot.model.qpos0.copy())["root"]
        - scale * human.world_positions[0, 0]
    )
    evidence = {
        "definition": config["canonicalization"]["common_scale"]["definition"],
        "value": scale,
        "local_body_scale": scale,
        "root_displacement_scale": scale,
        "source_landmarks": calibration["source_landmarks"],
        "robot_landmarks": calibration["robot_landmarks"],
        "landmark_weights": calibration["weights"],
        "least_squares_numerator_m2": calibration["numerator_m2"],
        "least_squares_denominator_m2": calibration["denominator_m2"],
        "weighted_residual_rmse_m": calibration["weighted_residual_rmse_m"],
        "source_heading_yaw_rad": calibration["source_heading_yaw_rad"],
        "source_heading_alignment_matrix": calibration[
            "source_heading_alignment_matrix"
        ],
        "head_to_toe_diagnostic_scale": calibration[
            "head_to_toe_diagnostic_scale"
        ],
        "source_head_to_toe_span_m": calibration["source_head_to_toe_span_m"],
        "robot_head_to_toe_span_m": calibration["robot_head_to_toe_span_m"],
        "root_alignment_translation_m": root_alignment.tolist(),
        "robot_asset_sha256": robot.sha256,
        "robot_joint_order_sha256": robot.joint_order_sha256,
        "calibration_contract_sha256": plan["calibration_contract"][
            "contract_sha256"
        ],
    }
    if (
        _canonical_sha256(evidence) != value.get("calibration_evidence_sha256")
        or value.get("common_scale")
        != {**dict(config["canonicalization"]["common_scale"]), **evidence}
    ):
        raise Stage2Error("Per-sequence LS/anchor calibration failed recomputation")
    reference_path_value = value.get("reference_path")
    if sequence.reference_relative_path is None:
        if reference_path_value is not None or value.get("reference_sha256") is not None:
            raise Stage2Error("Unexpected reference binding for a non-intersection source")
    else:
        reference_path = _resolve_manifest_path(root, str(reference_path_value))
        if (
            not reference_path.is_file()
            or sha256_file(reference_path) != sequence.reference_sha256
            or value.get("reference_sha256") != sequence.reference_sha256
            or value.get("reference_frames") != sequence.reference_frames
        ):
            raise Stage2Error("Per-sequence Unitree reference binding changed")


def build_stage2_cost_probe(
    repo_root: str | Path = ".",
    config_path: str | Path = "configs/stage2.yaml",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Measure cheap Pilot preparation/evaluation/storage projection inputs.

    This probe never runs a retargeter. It repeats deterministic source
    canonicalization and unified evaluation, then applies the pre-registered
    conservative multiplier/floors. The manifest binds every measured input.
    """

    root = Path(repo_root).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config = load_yaml(config_file)
    probe_config = config["preflight_cost_probe"]
    repetitions = int(probe_config["repetitions"])
    if repetitions < 1:
        raise Stage2Error("Stage-2 cost probe requires at least one repetition")
    multiplier = float(probe_config["conservative_multiplier"])
    if multiplier < 1.0:
        raise Stage2Error("Stage-2 conservative cost multiplier must be >= 1")
    measurement_environment = str(probe_config["measurement_environment"])
    measurement_python = _resolve_python(measurement_environment)
    if Path(sys.executable).resolve() != measurement_python:
        raise Stage2Error(
            "Stage-2 cost probe must run inside its frozen measurement environment"
        )
    os.environ.update(THREAD_ENVIRONMENT)
    measurement_history = measurement_python.parent.parent / "conda-meta/history"
    if not measurement_history.is_file():
        raise Stage2Error("Stage-2 cost-probe conda history is missing")
    source_manifest_path = root / str(config["stage1_gate"]["canonical_source_manifest"])
    source_manifest = load_yaml(source_manifest_path)
    native_source = root / str(source_manifest["cropped_source_file"])
    canonical_source = root / str(source_manifest["canonical_path"])
    formal_evidence = _validate_stage1_formal_outputs(root, config)
    output_paths = [
        root / str(formal_evidence[name]["path"])
        for name in STAGE1_FORMAL_RETARGETERS
    ]
    if not all(path.is_file() for path in output_paths):
        raise Stage2Error("Stage-2 cost probe requires all six accepted Pilot outputs")

    from .calibration import load_evaluator_protocol
    from .evaluator import evaluate_motion
    from .robot_model import CanonicalRobotModel, default_robot_scene
    from .schemas import CanonicalG1, CanonicalHuman
    from .source import canonicalize_source

    human = CanonicalHuman.load(canonical_source)
    duration_s = len(human.timestamps) / human.fps
    robot = CanonicalRobotModel(default_robot_scene(root))
    evaluator_path = root / "manifests/evaluator.yaml"
    evaluator_protocol = load_evaluator_protocol(evaluator_path)
    evaluation_motion = CanonicalG1.load(output_paths[0])
    preparation_wall: list[float] = []
    evaluation_wall: list[float] = []
    canonical_sizes: list[int] = []
    contention_base = root / "runs/stage2_cost_probe"
    contention_base.mkdir(parents=True, exist_ok=True)
    attempt_index = 1
    while True:
        contention_output = contention_base / f"attempt-{attempt_index:03d}"
        try:
            contention_output.mkdir(parents=False, exist_ok=False)
            break
        except FileExistsError:
            attempt_index += 1
    with tempfile.TemporaryDirectory(prefix="rtcmp-stage2-probe-") as directory:
        temporary = Path(directory)
        for index in range(repetitions):
            output = temporary / f"canonical-{index}.npz"
            helper = temporary / f"canonical-{index}.yaml"
            started = time.perf_counter()
            canonicalize_source(
                native_source,
                output,
                helper,
                position_scale=float(config["dataset"]["source_position_scale"]),
                origin_path=native_source,
                origin_frame_start=0,
                origin_frame_end=len(human.timestamps),
                repo_root=root,
            )
            preparation_wall.append(time.perf_counter() - started)
            canonical_sizes.append(output.stat().st_size + helper.stat().st_size)
            started = time.perf_counter()
            evaluate_motion(human, evaluation_motion, robot, evaluator_protocol)
            evaluation_wall.append(time.perf_counter() - started)
        contention = _run_parallel_contention_probe(
            root,
            canonical_source,
            contention_output,
            probe_config["contention_probe"],
        )

    prep_rtf = float(np.median(preparation_wall) / duration_s)
    eval_rtf = float(np.median(evaluation_wall) / duration_s)
    canonical_bytes_per_frame = float(max(canonical_sizes) / len(human.timestamps))
    output_bytes_per_frame = float(
        max(path.stat().st_size for path in output_paths) / len(human.timestamps)
    )
    configured_allowances = config["budget_allowances"]
    configured_storage = config["storage_projection"]
    contention_artifact_inventory = _directory_receipts(root, contention_output)
    bound_inputs = [
        _file_receipt(root, path)
        for path in (
            config_file,
            source_manifest_path,
            native_source,
            canonical_source,
            evaluator_path,
            root / "src/retargeting_comparison/source.py",
            root / "src/retargeting_comparison/evaluator.py",
            root / "src/retargeting_comparison/method_worker.py",
            root / "src/retargeting_comparison/method_adapters.py",
            root / "src/retargeting_comparison/stage2.py",
            *output_paths,
            *(
                root / str(formal_evidence[name]["run_manifest"]["path"])
                for name in STAGE1_FORMAL_RETARGETERS
            ),
        )
    ]
    value: dict[str, Any] = {
        "schema_version": 1,
        "measurement_role": "pilot_measured_then_conservatively_bounded_projection_probe",
        "repetitions": repetitions,
        "statistic": "median",
        "conservative_multiplier": multiplier,
        "pilot_frames": len(human.timestamps),
        "pilot_duration_s": duration_s,
        "pilot_canonical_source": _file_receipt(root, canonical_source),
        "source_preparation_wall_s_raw": preparation_wall,
        "evaluator_wall_s_raw": evaluation_wall,
        "canonical_source_bytes_raw": canonical_sizes,
        "accepted_output_bytes_raw": {
            name: path.stat().st_size
            for name, path in zip(STAGE1_FORMAL_RETARGETERS, output_paths)
        },
        "accepted_formal_outputs": {
            name: _file_receipt(root, path)
            for name, path in zip(STAGE1_FORMAL_RETARGETERS, output_paths)
        },
        "measured_source_preparation_rtf": prep_rtf,
        "conservative_source_preparation_rtf": max(
            prep_rtf * multiplier,
            float(configured_allowances["source_preparation_rtf"]),
        ),
        "measured_evaluator_rtf_per_method": eval_rtf,
        "conservative_evaluator_rtf_per_method": max(
            eval_rtf * multiplier,
            float(configured_allowances["evaluator_rtf_per_method"]),
        ),
        "measured_canonical_source_bytes_per_frame": canonical_bytes_per_frame,
        "conservative_canonical_source_bytes_per_frame": max(
            canonical_bytes_per_frame * multiplier,
            float(configured_storage["canonical_source_bytes_per_frame"]),
        ),
        "measured_output_bytes_per_frame_max": output_bytes_per_frame,
        "conservative_output_bytes_per_frame": max(
            output_bytes_per_frame * multiplier,
            max(
                float(entry["retained_bytes_per_frame"])
                for entry in config["methods"].values()
            ),
        ),
        "parallel_contention_probe": contention,
        "conservative_parallel_contention_factor": contention[
            "conservative_parallel_contention_factor"
        ],
        "measurement_environment": {
            "environment": measurement_environment,
            "python": _file_receipt(root, measurement_python),
            "conda_history": _file_receipt(root, measurement_history),
            "thread_environment": dict(THREAD_ENVIRONMENT),
            "thread_environment_sha256": _canonical_sha256(THREAD_ENVIRONMENT),
        },
        "measurement_hardware": _hardware_identity(config),
        "contention_artifact_root": _portable(root, contention_output),
        "contention_artifact_inventory": contention_artifact_inventory,
        "contention_artifact_inventory_sha256": _canonical_sha256(
            contention_artifact_inventory
        ),
        "stage2_config": _file_receipt(root, config_file),
        "bound_inputs": bound_inputs,
        "bound_inputs_sha256": _canonical_sha256(bound_inputs),
        "measured_at_utc": utc_now(),
    }
    value["payload_sha256"] = _canonical_sha256(value)
    destination = Path(output_path or probe_config["manifest"])
    if not destination.is_absolute():
        destination = root / destination
    atomic_write_json(destination, value)
    return value


def _run_parallel_contention_probe(
    root: Path,
    canonical_source: Path,
    output_root: Path,
    probe: Mapping[str, Any],
) -> dict[str, Any]:
    """Measure 1/2/6-process OmniRetarget slowdown on distinct CPU cores."""

    if probe.get("method") != "omniretarget":
        raise Stage2Error("The registered contention probe must use OmniRetarget")
    levels = tuple(int(value) for value in probe.get("levels", ()))
    if levels != (1, 2, 6):
        raise Stage2Error("Contention probe levels must be exactly 1/2/6")
    if (
        probe.get("cpu_assignment") != "distinct_allowed_cpus"
        or probe.get("apply_to_all_methods_with_workers_gt_one") is not True
    ):
        raise Stage2Error("Contention probe CPU/application policy changed")
    repetitions = int(probe.get("repetitions_per_level", 0))
    frames = int(probe.get("source_frames", 0))
    if repetitions < 1 or frames < 20:
        raise Stage2Error(
            "Contention probe requires positive repetitions and at least 20 frames"
        )
    if not hasattr(os, "sched_getaffinity"):
        raise Stage2Error("Contention probe requires Linux CPU affinity")
    cpus = _physical_cpu_ids(max(levels))
    python = _resolve_python(str(probe["environment"]))
    environment = os.environ.copy()
    environment.update(THREAD_ENVIRONMENT)
    source_root = str(root / "src")
    environment["PYTHONPATH"] = source_root + (
        os.pathsep + environment["PYTHONPATH"]
        if environment.get("PYTHONPATH")
        else ""
    )

    results: list[dict[str, Any]] = []

    def launch(command: list[str], stdout_path: Path, stderr_path: Path) -> dict[str, Any]:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            completed = subprocess.run(
                command,
                cwd=root,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        return {
            "exit_code": completed.returncode,
            "process_wall_s": time.perf_counter() - started,
            "command": list(command),
            "command_sha256": _canonical_sha256(command),
            "stdout_log": _file_receipt(root, stdout_path),
            "stderr_log": _file_receipt(root, stderr_path),
        }

    for level in levels:
        level_walls: list[float] = []
        process_walls: list[list[float]] = []
        steady_walls: list[list[float]] = []
        process_receipts: list[list[dict[str, Any]]] = []
        timing_receipts: list[list[dict[str, Any]]] = []
        for repetition in range(repetitions):
            commands: list[tuple[list[str], Path, Path, Path]] = []
            for worker in range(level):
                directory = output_root / f"level-{level}/rep-{repetition}/worker-{worker}"
                output = directory / "canonical_g1.npz"
                timing = directory / "timing.json"
                command = [
                    "taskset",
                    "--cpu-list",
                    str(cpus[worker]),
                    str(python),
                    "-m",
                    "retargeting_comparison.method_worker",
                    "--method",
                    "omniretarget",
                    "--repo-root",
                    str(root),
                    "--source",
                    str(canonical_source),
                    "--output",
                    str(output),
                    "--work-dir",
                    str(directory / "work"),
                    "--timing-json",
                    str(timing),
                    "--warmup-runs",
                    "0",
                    "--measured-runs",
                    "1",
                    "--max-frames",
                    str(frames),
                ]
                commands.append(
                    (
                        command,
                        directory / "stdout.log",
                        directory / "stderr.log",
                        timing,
                    )
                )
            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=level) as pool:
                futures = [
                    pool.submit(launch, command, stdout, stderr)
                    for command, stdout, stderr, _timing in commands
                ]
                process_results = [future.result() for future in futures]
            level_wall = time.perf_counter() - started
            if any(result["exit_code"] != 0 for result in process_results):
                raise Stage2Error(f"OmniRetarget contention probe failed at level {level}")
            timing_values = [
                json.loads(timing.read_text(encoding="utf-8"))
                for _command, _stdout, _stderr, timing in commands
            ]
            if any(
                len(value.get("repetitions", [])) != 1
                or value["repetitions"][0].get("role") != "measured"
                or value["repetitions"][0].get("timing_boundary")
                != "canonical_source_file_to_canonical_g1_in_memory"
                or int(value["repetitions"][0].get("frame_count", -1)) != frames
                for value in timing_values
            ):
                raise Stage2Error(
                    "Contention probe timing receipts violate the registered protocol"
                )
            repetition_timing_receipts = []
            for value, (_command, _stdout, _stderr, timing) in zip(
                timing_values, commands
            ):
                repetition_timing_receipts.append(
                    {
                        "timing_json": _file_receipt(root, timing),
                        "payload_sha256": _canonical_sha256(value),
                        "payload": value,
                    }
                )
            measured_steady = [
                float(value["repetitions"][0]["steady_end_to_end_total_s"])
                for value in timing_values
            ]
            if not all(value > 0.0 and np.isfinite(value) for value in measured_steady):
                raise Stage2Error("Contention probe produced invalid in-memory timings")
            level_walls.append(level_wall)
            process_walls.append(
                [float(result["process_wall_s"]) for result in process_results]
            )
            process_receipts.append(process_results)
            timing_receipts.append(repetition_timing_receipts)
            steady_walls.append(measured_steady)
        max_individual_process_walls = [max(values) for values in process_walls]
        # The process-level makespan is the externally observed wall interval
        # from launching the complete level until all fresh processes exit.
        # Individual subprocess timers remain a diagnostic lower bound.
        process_makespans = list(level_walls)
        results.append(
            {
                "workers": level,
                "level_wall_s_raw": level_walls,
                "level_wall_s_median": float(np.median(level_walls)),
                "process_wall_s_raw": process_walls,
                "process_makespan_s_raw": process_makespans,
                "process_makespan_s_median": float(np.median(process_makespans)),
                "max_individual_process_wall_s_raw": max_individual_process_walls,
                "steady_in_memory_s_raw": steady_walls,
                "steady_worker_s_median": float(np.median(steady_walls)),
                "process_receipts": process_receipts,
                "timing_receipts": timing_receipts,
                "distinct_cpu_ids": cpus[:level],
            }
        )
    serial_makespan = float(results[0]["process_makespan_s_median"])
    serial_steady = float(results[0]["steady_worker_s_median"])
    for row in results:
        process_slowdown = max(
            1.0, float(row["process_makespan_s_median"]) / serial_makespan
        )
        steady_slowdown = max(
            1.0, float(row["steady_worker_s_median"]) / serial_steady
        )
        combined_slowdown = max(process_slowdown, steady_slowdown)
        row["process_makespan_slowdown_vs_one_process"] = process_slowdown
        row["steady_per_worker_slowdown_vs_one_process"] = steady_slowdown
        row["combined_slowdown_vs_one_process"] = combined_slowdown
        # Backward-readable alias: the projection always uses the conservative
        # maximum of both independently measured slowdown boundaries.
        row["per_worker_slowdown_vs_one_process"] = combined_slowdown
        row["observed_aggregate_speedup"] = float(row["workers"]) / combined_slowdown
    observed = max(float(row["combined_slowdown_vs_one_process"]) for row in results)
    margin = float(probe["conservative_slowdown_multiplier"])
    if margin < 1.0:
        raise Stage2Error("Contention slowdown multiplier must be >= 1")
    return {
        "method": "omniretarget",
        "environment": str(probe["environment"]),
        "levels": list(levels),
        "repetitions_per_level": repetitions,
        "source_frames": frames,
        "command_template": (
            "taskset --cpu-list <distinct-cpu> <stage1-hsretargeting-python> -m "
            "retargeting_comparison.method_worker --method omniretarget "
            "--warmup-runs 0 --measured-runs 1 --max-frames <source_frames>"
        ),
        "timing_boundary": "canonical_source_file_to_canonical_g1_in_memory",
        "cpu_assignment": "distinct_allowed_cpus",
        "results": results,
        "observed_max_combined_slowdown": observed,
        "observed_max_per_worker_slowdown": observed,
        "conservative_slowdown_multiplier": margin,
        "conservative_parallel_contention_factor": observed * margin,
        "apply_to_all_methods_with_workers_gt_one": True,
        "python": _file_receipt(root, python),
        "conda_history": _file_receipt(
            root, python.parent.parent / "conda-meta/history"
        ),
    }


def _resolve_manifest_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _payload_sha256(value: Mapping[str, Any], field: str = "payload_sha256") -> str:
    payload = dict(value)
    payload.pop(field, None)
    return _canonical_sha256(payload)


def _write_job_manifest(path: Path, value: Mapping[str, Any]) -> None:
    payload = dict(value)
    payload["payload_sha256"] = _payload_sha256(payload)
    atomic_write_json(path, payload)


def _load_job_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("payload_sha256") != _payload_sha256(value)
    ):
        raise ValueError("Stage-2 job manifest payload hash is invalid")
    return value


def _path_below(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _attempt_chain_valid(
    manifest_path: Path,
    job: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
    *,
    allow_last_running: bool = False,
) -> bool:
    """Validate every preserved attempt, not merely the latest success."""

    expected_attempt_root = (
        manifest_path.parent.parent / "attempts" / str(job["job_id"])
    )
    planned_job = {key: job[key] for key in sorted(job) if key != "plan_sha256"}
    try:
        for index, attempt in enumerate(attempts, start=1):
            attempt_root = expected_attempt_root / f"attempt-{index:03d}"
            execution_path = Path(str(attempt["execution_contract_path"]))
            expected_output = attempt_root / "canonical_g1.npz"
            if (
                int(attempt.get("attempt", -1)) != index
                or execution_path.resolve()
                != (attempt_root / "execution_contract.json").resolve()
                or Path(str(attempt.get("expected_output_path", ""))).resolve()
                != expected_output.resolve()
                or Path(str(attempt.get("stdout_log", ""))).resolve()
                != (attempt_root / "logs/stdout.log").resolve()
                or Path(str(attempt.get("stderr_log", ""))).resolve()
                != (attempt_root / "logs/stderr.log").resolve()
                or not execution_path.is_file()
                or attempt.get("command_sha256")
                != _canonical_sha256(attempt.get("command", []))
                or attempt.get("thread_environment_sha256")
                != _canonical_sha256(attempt.get("thread_environment", {}))
                or attempt.get("python_sha256")
                != sha256_file(Path(str(attempt["python"])))
                or attempt.get("conda_history_sha256")
                != sha256_file(Path(str(attempt["conda_history"])))
            ):
                return False
            execution = json.loads(execution_path.read_text(encoding="utf-8"))
            execution_payload = dict(execution)
            execution_sha256 = execution_payload.pop(
                "execution_contract_sha256", None
            )
            if (
                execution_sha256 != _canonical_sha256(execution_payload)
                or attempt.get("execution_contract_sha256") != execution_sha256
                or execution.get("plan_sha256") != job.get("plan_sha256")
                or execution.get("planned_job") != planned_job
                or execution.get("job_contract_sha256")
                != job.get("job_contract_sha256")
                or execution.get("method_contract_sha256")
                != job.get("method_contract_sha256")
                or execution.get("policy_sha256") != job.get("policy_sha256")
                or execution.get("command") != attempt.get("command")
                or execution.get("command_sha256")
                != attempt.get("command_sha256")
                or Path(str(execution.get("expected_output_path", ""))).resolve()
                != expected_output.resolve()
                or (
                    execution.get("work_directory") is not None
                    and Path(str(execution["work_directory"])).resolve()
                    != (attempt_root / "work").resolve()
                )
            ):
                return False
            for log_name in ("stdout", "stderr"):
                hash_key = f"{log_name}_log_sha256"
                bytes_key = f"{log_name}_log_bytes"
                if hash_key not in attempt:
                    continue
                log_path = Path(str(attempt[f"{log_name}_log"]))
                if (
                    not log_path.is_file()
                    or attempt[hash_key] != sha256_file(log_path)
                    or (
                        bytes_key in attempt
                        and int(attempt[bytes_key]) != log_path.stat().st_size
                    )
                ):
                    return False
            has_exit_code = "exit_code" in attempt
            if not has_exit_code:
                if not (allow_last_running and index == len(attempts)):
                    return False
            else:
                wall = float(attempt.get("wall_time_s", -1.0))
                if not np.isfinite(wall) or wall < 0.0:
                    return False
            if attempt.get("status") == "abandoned_conservatively_accounted":
                started = _parse_utc(str(attempt["started_at_utc"]))
                recovered = _parse_utc(str(attempt["recovered_at_utc"]))
                elapsed = (recovered - started).total_seconds()
                if (
                    int(attempt.get("exit_code", -1))
                    != ABANDONED_RUNNING_EXIT_CODE
                    or attempt.get("termination_reason")
                    != ABANDONED_RUNNING_REASON
                    or attempt.get("budget_accounting")
                    != "full_elapsed_utc_interval_charged_conservatively"
                    or attempt.get("finished_at_utc")
                    != attempt.get("recovered_at_utc")
                    or elapsed < 0.0
                    or not np.isclose(
                        float(attempt.get("wall_time_s", -1.0)),
                        elapsed,
                        rtol=0.0,
                        atol=1e-6,
                    )
                ):
                    return False
        return True
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return False


def _job_manifest_valid(path: Path, job: Mapping[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        value = _load_job_manifest(path)
        output = Path(value["output_path"]).resolve()
        sequence_manifest = Path(value["sequence_manifest"]).resolve()
        attempts = value.get("attempts", [])
        if (
            not isinstance(attempts, list)
            or not attempts
            or not _attempt_chain_valid(path, job, attempts)
        ):
            return False
        attempt = attempts[-1]
        execution_contract = Path(attempt["execution_contract_path"]).resolve()
        if not execution_contract.is_file():
            return False
        execution_value = json.loads(execution_contract.read_text(encoding="utf-8"))
        execution_sha = execution_value.get("execution_contract_sha256")
        execution_payload = dict(execution_value)
        execution_payload.pop("execution_contract_sha256", None)
        expected_attempt_root = path.parent.parent / "attempts" / str(job["job_id"])
        identity_valid = bool(
            value.get("status") == "succeeded"
            and value.get("job_id") == job["job_id"]
            and value.get("plan_sha256") == job["plan_sha256"]
            and value.get("method") == job.get("method")
            and value.get("sequence_id") == job.get("sequence_id")
            and value.get("stratum") == job.get("stratum")
            and value.get("job_contract_sha256") == job.get("job_contract_sha256")
            and value.get("method_contract_sha256")
            == job.get("method_contract_sha256")
            and value.get("policy_sha256") == job.get("policy_sha256")
            and np.isclose(
                float(value.get("cumulative_accounted_attempt_wall_s", -1.0)),
                sum(float(item["wall_time_s"]) for item in attempts),
                rtol=0.0,
                atol=1e-9,
            )
            and float(value.get("completion_ratio", -1.0)) == 1.0
            and len(attempts) == int(attempt["attempt"])
            and attempt.get("exit_code") == 0
            and attempt.get("command_sha256") == _canonical_sha256(attempt["command"])
            and attempt.get("thread_environment_sha256")
            == _canonical_sha256(attempt["thread_environment"])
            and attempt.get("python_sha256") == sha256_file(Path(attempt["python"]))
            and attempt.get("conda_history_sha256")
            == sha256_file(Path(attempt["conda_history"]))
            and execution_value.get("environment", {}).get("python_sha256")
            == attempt.get("python_sha256")
            and execution_value.get("environment", {}).get("conda_history_sha256")
            == attempt.get("conda_history_sha256")
            and attempt.get("execution_contract_sha256") == execution_sha
            and execution_sha == _canonical_sha256(execution_payload)
            and value.get("execution_contract_sha256") == execution_sha
            and _path_below(output, expected_attempt_root)
            and output == Path(attempt["expected_output_path"]).resolve()
            and execution_value.get("expected_output_path") == str(output)
            and execution_value.get("command") == attempt["command"]
            and execution_value.get("command_sha256") == attempt["command_sha256"]
            and execution_value.get("plan_sha256") == job["plan_sha256"]
            and execution_value.get("job_contract_sha256")
            == job.get("job_contract_sha256")
            and execution_value.get("method_contract_sha256")
            == job.get("method_contract_sha256")
            and execution_value.get("policy_sha256") == job.get("policy_sha256")
            and value.get("repository_contract_sha256")
            == execution_value.get("repository_contract_sha256")
            and value.get("hardware_contract_sha256")
            == execution_value.get("hardware_contract_sha256")
            and execution_value.get("planned_job") == {
                key: job[key] for key in sorted(job) if key != "plan_sha256"
            }
            and output.is_file()
            and value.get("output_sha256") == sha256_file(output)
            and int(value.get("output_bytes", -1)) == output.stat().st_size
            and Path(attempt["stdout_log"]).is_file()
            and Path(attempt["stderr_log"]).is_file()
            and attempt.get("stdout_log_sha256")
            == sha256_file(Path(attempt["stdout_log"]))
            and attempt.get("stderr_log_sha256")
            == sha256_file(Path(attempt["stderr_log"]))
            and sequence_manifest.is_file()
            and value.get("sequence_manifest_sha256") == sha256_file(sequence_manifest)
            and execution_value.get("sequence_manifest_sha256")
            == value.get("sequence_manifest_sha256")
        )
        if not identity_valid:
            return False
        cleanup = attempt.get("rebuildable_work_cleanup", {})
        cleanup_payload = dict(cleanup)
        cleanup_sha256 = cleanup_payload.pop("receipt_sha256", None)
        cleanup_files = cleanup_payload.get("files", [])
        if (
            cleanup_sha256 != _canonical_sha256(cleanup_payload)
            or cleanup_payload.get("role")
            != "rebuildable_native_intermediates_deleted_after_hash_capture"
            or cleanup_payload.get("deleted") is not True
            or cleanup_payload.get("files_sha256")
            != _canonical_sha256(cleanup_files)
            or int(cleanup_payload.get("total_bytes_before_deletion", -1))
            != sum(int(item["bytes"]) for item in cleanup_files)
            or Path(str(execution_value.get("work_directory", ""))).exists()
        ):
            return False
        from .schemas import CanonicalG1

        motion = CanonicalG1.load(output)
        sequence = json.loads(sequence_manifest.read_text(encoding="utf-8"))
        sequence_payload = dict(sequence)
        sequence_payload_hash = sequence_payload.pop("payload_sha256", None)
        if (
            sequence_payload_hash != _canonical_sha256(sequence_payload)
            or value.get("sequence_payload_sha256") != sequence_payload_hash
        ):
            return False
        execution_root = execution_value.get("repo_root")
        _validate_production_motion(
            motion,
            job,
            sequence,
            execution_value,
            repo_root=None if execution_root is None else Path(str(execution_root)),
        )
        from .native_target_capture import tensor_sha256

        if (
            value.get("output_qpos_sha256")
            != tensor_sha256(np.asarray(motion.qpos, dtype=np.float64))
            or motion.metadata.get("stage2_job_contract_sha256")
            != job.get("job_contract_sha256")
            or motion.metadata.get("stage2_execution_contract_sha256")
            != execution_sha
            or motion.metadata.get("stage2_method_contract_sha256")
            != job.get("method_contract_sha256")
            or motion.metadata.get("stage2_policy_sha256")
            != job.get("policy_sha256")
            or motion.metadata.get("stage2_sequence_manifest_sha256")
            != value.get("sequence_manifest_sha256")
        ):
            return False
        return True
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return False


def job_is_complete(manifest_path: str | Path, job: Mapping[str, Any]) -> bool:
    """Public resume predicate: success means both provenance and bytes match."""

    return _job_manifest_valid(Path(manifest_path), job)


def _validate_production_motion(
    motion: Any,
    job: Mapping[str, Any],
    sequence: Mapping[str, Any],
    execution_contract: Mapping[str, Any] | None = None,
    repo_root: Path | None = None,
) -> None:
    """Enforce the exact Stage-2 timeline and method-identity contract."""

    frames = int(sequence["source_frames"])
    fps = float(sequence["fps"])
    qpos = np.asarray(motion.qpos, dtype=np.float64)
    solve = np.asarray(motion.per_frame_solve_time_s, dtype=np.float64)
    if qpos.shape != (frames, 36):
        raise ValueError("Stage-2 output must contain exactly every source frame")
    if not np.isclose(float(motion.fps), fps, rtol=0.0, atol=1e-9):
        raise ValueError("Stage-2 output fps differs from the frozen source timeline")
    if not np.array_equal(np.asarray(motion.source_frame_idx), np.arange(frames)):
        raise ValueError("Stage-2 output source_frame_idx must be contiguous 0..T-1")
    if not np.asarray(motion.valid, dtype=bool).all():
        raise ValueError("Stage-2 output contains invalid frames")
    if not np.isfinite(qpos).all() or not np.isfinite(solve).all():
        raise ValueError("Stage-2 output contains NaN/Inf")
    norms = np.linalg.norm(qpos[:, 3:7], axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("Stage-2 root quaternions are not unit wxyz quaternions")
    metadata = motion.metadata
    if str(metadata.get("stage2_method_id")) != str(job["method"]):
        raise ValueError("Stage-2 output method identity mismatch")
    if str(metadata.get("stage2_sequence_id")) != str(job["sequence_id"]):
        raise ValueError("Stage-2 output sequence identity mismatch")
    if str(metadata.get("experiment_stratum")) != str(job["stratum"]):
        raise ValueError("Stage-2 output evidence-stratum mismatch")
    if str(metadata.get("completion_status")) != "succeeded":
        raise ValueError("Stage-2 output completion_status is not succeeded")
    required_metadata = {
        "stage2_plan_sha256": job.get("plan_sha256"),
        "stage2_job_contract_sha256": job.get("job_contract_sha256"),
        "stage2_method_contract_sha256": job.get("method_contract_sha256"),
        "stage2_policy_sha256": job.get("policy_sha256"),
        "stage2_source_sha256": sequence.get("source_sha256"),
        "stage2_canonical_source_sha256": sequence.get("canonical_sha256"),
    }
    for key, expected in required_metadata.items():
        if expected is not None and metadata.get(key) != expected:
            raise ValueError(f"Stage-2 output metadata binding failed: {key}")
    if execution_contract is not None:
        method_contract = execution_contract.get("method_contract", {})
        exact_upstream = {
            "method": job["method"],
            "stratum": job["stratum"],
            "registered_revision": method_contract.get("registered_revision"),
            "method_contract_sha256": job.get("method_contract_sha256"),
            "policy_sha256": job.get("policy_sha256"),
            "timing_evidence_sha256": method_contract.get(
                "timing_evidence_sha256"
            ),
            "environment_provenance_sha256": method_contract.get(
                "environment_provenance_sha256"
            ),
        }
        execution_metadata = {
            "stage2_design_id": execution_contract.get("design_id"),
            "stage2_config_sha256": execution_contract.get("config_sha256"),
            "stage2_repository_contract_sha256": execution_contract.get(
                "repository_contract_sha256"
            ),
            "stage2_hardware_contract_sha256": execution_contract.get(
                "hardware_contract_sha256"
            ),
            "stage2_execution_contract_sha256": execution_contract.get(
                "execution_contract_sha256"
            ),
            "stage2_sequence_manifest_sha256": execution_contract.get(
                "sequence_manifest_sha256"
            ),
            "stage2_sequence_payload_sha256": execution_contract.get(
                "sequence_payload_sha256"
            ),
            "stage2_reference_sha256": execution_contract.get("reference_sha256"),
            "stage2_unitree_reference_contract_sha256": execution_contract.get(
                "unitree_reference_contract_sha256"
            ),
            "stage2_environment_provenance_sha256": execution_contract.get(
                "environment_provenance_sha256"
            ),
        }
        for key, expected in execution_metadata.items():
            if metadata.get(key) != expected:
                raise ValueError(f"Stage-2 output execution binding failed: {key}")
        if metadata.get(
            "stage2_exact_upstream_config_asset_policy_contract"
        ) != exact_upstream:
            raise ValueError(
                "Stage-2 output upstream/config/asset/policy binding failed"
            )
    if job["stratum"] == "controlled_common_per_sequence_scale":
        calibration = sequence.get("common_scale", {})
        authoritative = metadata.get("stage2_authoritative_scale_anchor", {})
        target = metadata.get("stage2_pre_solver_target_contract", {})
        expected_target: Mapping[str, Any] | None = None
        if repo_root is not None:
            from .schemas import CanonicalHuman

            canonical_path = _resolve_manifest_path(
                repo_root, str(sequence["canonical_path"])
            )
            human = CanonicalHuman.load(canonical_path)
            variant = "sparse" if job["method"] == "sparse-neutral" else "dense"
            expected_target = _controlled_pre_solver_target_contract(
                repo_root, human, sequence, variant
            )
        if (
            authoritative.get("calibration_evidence_sha256")
            != sequence.get("calibration_evidence_sha256")
            or authoritative.get("local_body_scale")
            != calibration.get("local_body_scale")
            or authoritative.get("root_displacement_scale")
            != calibration.get("root_displacement_scale")
            or authoritative.get("root_alignment_translation_m")
            != calibration.get("root_alignment_translation_m")
            or authoritative.get("policy")
            != "controlled_common_per_sequence_scale"
            or authoritative.get("replaces_stage1_pilot_scale_anchor_metadata")
            is not True
            or any(
                key in metadata
                for key in (
                    "scale_policy",
                    "root_alignment_translation_m",
                    "root_displacement_scale",
                    "local_body_scale",
                    "root_anchor_policy",
                )
            )
            or not isinstance(target.get("target_labels"), list)
            or not target["target_labels"]
            or target.get("target_tensor_shape")
            != [frames, len(target["target_labels"]) * 3 + 1]
            or target.get("observed_boundary")
            != "immediately_before_controlled_mink_frame_tasks"
            or any(
                not isinstance(target.get(key), str)
                or re.fullmatch(r"[0-9a-f]{64}", str(target.get(key))) is None
                for key in (
                    "target_tensor_sha256",
                    "position_tensor_sha256",
                    "root_yaw_tensor_sha256",
                )
            )
            or (expected_target is not None and target != expected_target)
        ):
            raise ValueError("Controlled output lacks authoritative scale/anchor/target binding")
    if job["stratum"] == "external_reference" and (
        metadata.get("timeline_alignment_contract")
        != "same_basename_and_frame_indices_only"
        or metadata.get("byte_identical_human_source_verified") is not False
        or metadata.get("exact_timestamp_identity_claimed") is not False
        or metadata.get("verified_official_ground_truth") is not False
        or metadata.get("fps_resampling_performed") is not False
        or metadata.get("upstream_path") != job.get("reference_relative_path")
        or metadata.get("source_and_reference_frame_count_match") is not True
    ):
        raise ValueError("External reference output overclaims its alignment/evidence role")


def _worker_command(
    root: Path,
    python: Path,
    job: Mapping[str, Any],
    sequence_manifest: Path,
    output: Path,
    work_dir: Path,
    execution_contract: Path,
) -> list[str]:
    command = [
        str(python),
        "-m",
        "retargeting_comparison.stage2_worker",
        "--repo-root",
        str(root),
        "--method",
        str(job["method"]),
        "--sequence-manifest",
        str(sequence_manifest),
        "--output",
        str(output),
        "--work-dir",
        str(work_dir),
        "--execution-contract",
        str(execution_contract),
    ]
    if job.get("reference_relative_path") is not None:
        command.extend(
            ["--reference-relative-path", str(job["reference_relative_path"])]
        )
    return command


class _ExecutionGuard:
    """Shared fail-closed wall/storage guard for every concurrent worker lane."""

    def __init__(
        self,
        run_root: Path,
        *,
        deadline_monotonic: float,
        storage_stop_bytes: float,
        external_retained_bytes: int = 0,
    ) -> None:
        self.run_root = run_root
        self.deadline_monotonic = deadline_monotonic
        self.storage_stop_bytes = storage_stop_bytes
        self.external_retained_bytes = max(0, int(external_retained_bytes))
        self.cancel_event = threading.Event()
        self._lock = threading.RLock()
        self._reason: str | None = None
        self._processes: dict[int, subprocess.Popen[Any]] = {}
        self._reservations: dict[str, int] = {}

    @property
    def reason(self) -> str | None:
        with self._lock:
            return self._reason

    def _set_reason(self, reason: str) -> bool:
        with self._lock:
            first = self._reason is None
            if first:
                self._reason = reason
            self.cancel_event.set()
            return first

    def cancel(self, reason: str) -> None:
        self._set_reason(reason)
        with self._lock:
            processes = list(self._processes.values())
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    def poll_reason(self) -> str | None:
        if self.cancel_event.is_set():
            return self.reason or "shared_cancellation"
        if time.monotonic() >= self.deadline_monotonic:
            self.cancel("wall_time_guard")
            return self.reason
        with self._lock:
            reserved = sum(self._reservations.values())
        if (
            self.external_retained_bytes
            + _directory_size(self.run_root)
            + reserved
            >= self.storage_stop_bytes
        ):
            self.cancel("retained_storage_guard")
            return self.reason
        return None

    def reserve(
        self, job_id: str, retained_bytes: int, estimated_runtime_s: float
    ) -> bool:
        failure: str | None = None
        with self._lock:
            if self.cancel_event.is_set():
                return False
            if (
                time.monotonic() + max(0.0, estimated_runtime_s)
                >= self.deadline_monotonic
            ):
                failure = "wall_time_reservation_guard"
            projected = (
                self.external_retained_bytes
                + _directory_size(self.run_root)
                + sum(self._reservations.values())
                + max(0, retained_bytes)
            )
            if failure is None and projected >= self.storage_stop_bytes:
                failure = "retained_storage_reservation_guard"
            if failure is None:
                self._reservations[job_id] = max(0, retained_bytes)
        if failure is not None:
            self.cancel(failure)
            return False
        return True

    def release(self, job_id: str) -> None:
        with self._lock:
            self._reservations.pop(job_id, None)

    def register(self, process: subprocess.Popen[Any]) -> None:
        with self._lock:
            if self.cancel_event.is_set():
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            self._processes[process.pid] = process

    def unregister(self, process: subprocess.Popen[Any]) -> None:
        with self._lock:
            self._processes.pop(process.pid, None)

    def kill_remaining(self) -> None:
        with self._lock:
            processes = list(self._processes.values())
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


def _build_execution_contract(
    root: Path,
    plan: Mapping[str, Any],
    config: Mapping[str, Any],
    job: Mapping[str, Any],
    sequence_manifest: Path,
    output: Path,
    work: Path,
    python: Path,
    command: Sequence[str],
) -> dict[str, Any]:
    sequence = json.loads(sequence_manifest.read_text(encoding="utf-8"))
    payload = dict(sequence)
    sequence_payload_sha256 = payload.pop("payload_sha256", None)
    if sequence_payload_sha256 != _canonical_sha256(payload):
        raise Stage2Error("Sequence manifest payload hash failed before worker launch")
    method_spec = next(
        (
            value
            for value in plan["method_specs"]
            if value.get("method") == job["method"]
        ),
        None,
    )
    if method_spec is None or method_spec.get("method_contract_sha256") != job.get(
        "method_contract_sha256"
    ):
        raise Stage2Error("Planned method/job contract mismatch")
    environment = str(config["methods"][job["method"]]["environment"])
    history = python.parent.parent / "conda-meta/history"
    environment_receipt = {
        "environment": environment,
        "python_path": str(python),
        "python_sha256": sha256_file(python),
        "conda_history_path": str(history),
        "conda_history_sha256": sha256_file(history),
    }
    stage1_method = plan.get("stage1_timing_evidence", {}).get("methods", {}).get(
        job["method"]
    )
    if stage1_method is not None and environment_receipt != stage1_method.get(
        "environment"
    ):
        raise Stage2Error(
            f"Stage-2 worker environment differs from Stage-1 timing: {job['method']}"
        )
    thread_environment = dict(THREAD_ENVIRONMENT)
    production_cpu_ids = list(plan["hardware"]["production_cpu_ids"])
    worker_index = int(job["worker_index"])
    if not 0 <= worker_index < len(production_cpu_ids):
        raise Stage2Error("Planned worker lane lacks a frozen physical CPU")
    cpu_affinity = [int(production_cpu_ids[worker_index])]
    contract: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "repo_root": str(root),
        "plan_sha256": plan["plan_sha256"],
        "design_id": plan["design_id"],
        "config_sha256": plan["config_sha256"],
        "repository_contract_sha256": _canonical_sha256(plan["repository"]),
        "hardware_contract_sha256": _canonical_sha256(plan["hardware"]),
        "planned_job": {key: job[key] for key in sorted(job)},
        "job_contract_sha256": job["job_contract_sha256"],
        "method_contract": method_spec,
        "method_contract_sha256": job["method_contract_sha256"],
        "policy_sha256": job["policy_sha256"],
        "sequence_manifest_path": str(sequence_manifest.resolve()),
        "sequence_manifest_sha256": sha256_file(sequence_manifest),
        "sequence_payload_sha256": sequence_payload_sha256,
        "calibration_evidence_sha256": sequence.get(
            "calibration_evidence_sha256"
        ),
        "source_sha256": sequence["source_sha256"],
        "canonical_source_sha256": sequence["canonical_sha256"],
        "reference_sha256": sequence.get("reference_sha256"),
        "unitree_reference_contract_sha256": plan.get(
            "unitree_reference_contract", {}
        ).get("contract_sha256"),
        "environment": environment_receipt,
        "environment_provenance_sha256": _canonical_sha256(environment_receipt),
        "thread_environment": thread_environment,
        "thread_environment_sha256": _canonical_sha256(thread_environment),
        "cpu_affinity": cpu_affinity,
        "cpu_affinity_policy": "one_distinct_allowed_physical_core_per_worker_lane",
        "command": list(command),
        "command_sha256": _canonical_sha256(list(command)),
        "expected_output_path": str(output.resolve()),
        "work_directory": str(work.resolve()),
        "production_passes": 1,
        "automatic_retry": False,
    }
    contract["execution_contract_sha256"] = _canonical_sha256(contract)
    return contract


@contextmanager
def _signal_cancellation_scope(guard: _ExecutionGuard):
    """Convert launcher SIGINT/SIGTERM into shared process-group cancellation."""

    previous: dict[int, Any] = {}

    def handler(signum: int, _frame: Any) -> None:
        try:
            name = signal.Signals(signum).name
        except ValueError:
            name = str(signum)
        guard.cancel(f"orchestrator_{name.lower()}")

    try:
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous[signum] = signal.getsignal(signum)
                signal.signal(signum, handler)
        try:
            yield
        except BaseException:
            guard.cancel(guard.reason or "orchestrator_exception")
            raise
    finally:
        # Give lane loops a chance to reap TERM, then guarantee no orphaned
        # process group survives an exception or parent interruption.
        guard.kill_remaining()
        for signum, old_handler in previous.items():
            signal.signal(signum, old_handler)


def _close_abandoned_running_attempt(
    attempt: Mapping[str, Any], *, recovered_at: datetime | None = None
) -> dict[str, Any]:
    """Turn a crash-left running attempt into an immutable budget receipt.

    The complete UTC interval is charged deliberately: after an orchestrator
    crash there is no trustworthy narrower process boundary.  Existing launch,
    command, environment, and log receipts are preserved byte-for-byte.
    """

    closed = dict(attempt)
    if "exit_code" in closed:
        return closed
    try:
        started = _parse_utc(str(closed["started_at_utc"]))
    except (KeyError, TypeError, ValueError) as error:
        raise Stage2Error(
            "Crash-left running job attempt lacks a valid start timestamp"
        ) from error
    recovered = recovered_at or datetime.now(timezone.utc)
    recovered = recovered.astimezone(timezone.utc)
    elapsed = (recovered - started).total_seconds()
    if elapsed < 0.0 or not np.isfinite(elapsed):
        raise Stage2Error("Crash-left job attempt has a future/invalid start time")
    recovered_text = recovered.isoformat()
    closed.update(
        {
            "status": "abandoned_conservatively_accounted",
            "recovered_at_utc": recovered_text,
            "finished_at_utc": recovered_text,
            "wall_time_s": elapsed,
            "exit_code": ABANDONED_RUNNING_EXIT_CODE,
            "termination_reason": ABANDONED_RUNNING_REASON,
            "budget_accounting": (
                "full_elapsed_utc_interval_charged_conservatively"
            ),
        }
    )
    for log_name in ("stdout", "stderr"):
        path = Path(str(closed.get(f"{log_name}_log", "")))
        if path.is_file():
            closed.setdefault(f"{log_name}_log_sha256", sha256_file(path))
            closed.setdefault(f"{log_name}_log_bytes", path.stat().st_size)
    return closed


def _run_job(
    root: Path,
    run_root: Path,
    plan: Mapping[str, Any],
    config: Mapping[str, Any],
    job: Mapping[str, Any],
    sequence_manifest: Path,
    guard: _ExecutionGuard,
) -> dict[str, Any]:
    manifest_path = run_root / "job_manifests" / f"{job['job_id']}.json"
    expected = {**job, "plan_sha256": plan["plan_sha256"]}
    if _job_manifest_valid(manifest_path, expected):
        return _load_job_manifest(manifest_path)
    previous: dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            previous = _load_job_manifest(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise Stage2Error(
                f"Existing job manifest is tampered/unreadable: {job['job_id']}"
            ) from error
        if previous.get("status") == "succeeded":
            raise Stage2Error(
                f"Successful job artifact failed hash verification: {job['job_id']}"
            )
        previous_attempts = previous.get("attempts", [])
        expected_repository_sha256 = _canonical_sha256(plan["repository"])
        expected_hardware_sha256 = _canonical_sha256(plan["hardware"])
        if (
            previous.get("status") not in {"running", "failed", "incomplete"}
            or previous.get("job_id") != job["job_id"]
            or previous.get("method") != job["method"]
            or previous.get("sequence_id") != job["sequence_id"]
            or previous.get("stratum") != job["stratum"]
            or previous.get("plan_sha256") != plan["plan_sha256"]
            or previous.get("config_sha256") != plan["config_sha256"]
            or previous.get("repository_contract_sha256")
            != expected_repository_sha256
            or previous.get("hardware_contract_sha256")
            != expected_hardware_sha256
            or previous.get("job_contract_sha256")
            != job["job_contract_sha256"]
            or previous.get("method_contract_sha256")
            != job["method_contract_sha256"]
            or previous.get("policy_sha256") != job["policy_sha256"]
            or Path(str(previous.get("sequence_manifest", ""))).resolve()
            != sequence_manifest.resolve()
            or previous.get("sequence_manifest_sha256")
            != sha256_file(sequence_manifest)
            or not isinstance(previous_attempts, list)
            or not previous_attempts
            or not _attempt_chain_valid(
                manifest_path,
                expected,
                previous_attempts,
                allow_last_running=previous.get("status") == "running",
            )
        ):
            raise Stage2Error(
                f"Existing incomplete job provenance is not resumable: {job['job_id']}"
            )
    attempts = list(previous.get("attempts", []))
    if previous.get("status") == "running":
        attempts[-1] = _close_abandoned_running_attempt(attempts[-1])
        previous = {
            **previous,
            "status": "incomplete",
            "message": (
                "Crash-left running attempt was closed and charged "
                "conservatively before resume"
            ),
            "attempts": attempts,
            "cumulative_accounted_attempt_wall_s": sum(
                float(item["wall_time_s"]) for item in attempts
            ),
            "recovered_at_utc": attempts[-1].get("recovered_at_utc", utc_now()),
        }
        # Persist the normalized chain before constructing or launching the
        # next attempt, so a second crash cannot resurrect an open record.
        _write_job_manifest(manifest_path, previous)
    attempt_number = len(attempts) + 1
    attempt_dir = (
        run_root / "attempts" / str(job["job_id"]) / f"attempt-{attempt_number:03d}"
    )
    output = attempt_dir / "canonical_g1.npz"
    logs = attempt_dir / "logs"
    work = attempt_dir / "work"
    method_entry = config["methods"][job["method"]]
    python = _resolve_python(str(method_entry["environment"]))
    execution_contract_path = attempt_dir / "execution_contract.json"
    command = _worker_command(
        root,
        python,
        job,
        sequence_manifest,
        output,
        work,
        execution_contract_path,
    )
    execution_contract = _build_execution_contract(
        root,
        plan,
        config,
        job,
        sequence_manifest,
        output,
        work,
        python,
        command,
    )
    atomic_write_json(execution_contract_path, execution_contract)
    history = python.parent.parent / "conda-meta/history"
    attempt = {
        "attempt": attempt_number,
        "started_at_utc": utc_now(),
        "command": command,
        "command_sha256": _canonical_sha256(command),
        "environment": f"conda:{method_entry['environment']}",
        "python": str(python),
        "python_sha256": sha256_file(python),
        "conda_history": str(history),
        "conda_history_sha256": sha256_file(history),
        "thread_environment": dict(THREAD_ENVIRONMENT),
        "thread_environment_sha256": _canonical_sha256(THREAD_ENVIRONMENT),
        "execution_contract_path": str(execution_contract_path.resolve()),
        "execution_contract_sha256": execution_contract[
            "execution_contract_sha256"
        ],
        "expected_output_path": str(output.resolve()),
        "stdout_log": str(logs / "stdout.log"),
        "stderr_log": str(logs / "stderr.log"),
    }
    running = {
        "schema_version": SCHEMA_VERSION,
        "job_id": job["job_id"],
        "method": job["method"],
        "sequence_id": job["sequence_id"],
        "stratum": job["stratum"],
        "status": "running",
        "plan_sha256": plan["plan_sha256"],
        "repo_commit": _git_head(root),
        "config_sha256": plan["config_sha256"],
        "repository_contract_sha256": _canonical_sha256(plan["repository"]),
        "hardware_contract_sha256": _canonical_sha256(plan["hardware"]),
        "job_contract_sha256": job["job_contract_sha256"],
        "method_contract_sha256": job["method_contract_sha256"],
        "policy_sha256": job["policy_sha256"],
        "execution_contract_sha256": execution_contract[
            "execution_contract_sha256"
        ],
        "sequence_manifest": str(sequence_manifest),
        "sequence_manifest_sha256": sha256_file(sequence_manifest),
        "sequence_payload_sha256": json.loads(
            sequence_manifest.read_text(encoding="utf-8")
        )["payload_sha256"],
        "prior_accounted_attempt_wall_s": sum(
            float(item["wall_time_s"]) for item in attempts
        ),
        "attempts": attempts + [attempt],
    }
    _write_job_manifest(manifest_path, running)
    logs.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(THREAD_ENVIRONMENT)
    source_path = str(root / "src")
    env["PYTHONPATH"] = source_path + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    started = time.perf_counter()
    stop_reason: str | None = None
    sequence_value = json.loads(sequence_manifest.read_text())
    source_frames = int(sequence_value["source_frames"])
    frozen_method_spec = execution_contract["method_contract"]
    reserved_bytes = int(
        frozen_method_spec["retained_bytes_per_sequence"]
    ) + source_frames * int(frozen_method_spec["retained_bytes_per_frame"])
    estimated_runtime = float(job.get("estimated_runtime_s", 0.0)) * float(
        config["budget"].get("safety_factor", 1.0)
    )
    if not guard.reserve(str(job["job_id"]), reserved_bytes, estimated_runtime):
        stop_reason = guard.reason or "shared_budget_reservation_guard"
        attempt.update(
            {
                "finished_at_utc": utc_now(),
                "wall_time_s": 0.0,
                "exit_code": 125,
                "budget_stop_reason": stop_reason,
            }
        )
        finished = {
            **running,
            "status": "failed",
            "message": f"Production worker not started: {stop_reason}",
            "attempts": attempts + [attempt],
            "cumulative_accounted_attempt_wall_s": sum(
                float(item["wall_time_s"]) for item in attempts + [attempt]
            ),
        }
        _write_job_manifest(manifest_path, finished)
        return finished

    process: subprocess.Popen[Any] | None = None
    try:
        with (
            (logs / "stdout.log").open("w", encoding="utf-8") as stdout,
            (logs / "stderr.log").open("w", encoding="utf-8") as stderr,
        ):
            process = subprocess.Popen(
                command,
                cwd=root,
                env=env,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            guard.register(process)
            while process.poll() is None:
                stop_reason = guard.poll_reason()
                if stop_reason is not None:
                    try:
                        process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait()
                    else:
                        # TERM may reap the direct worker while a descendant
                        # ignores it. The original pid remains the process-group
                        # id, so kill the residual group even after parent exit.
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    break
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    pass
            returncode = int(
                process.returncode if process.returncode is not None else 124
            )
    except BaseException as error:
        stop_reason = stop_reason or guard.reason
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if process.poll() is None:
                process.wait()
        returncode = int(
            process.returncode if process and process.returncode is not None else 126
        )
        wall = time.perf_counter() - started
        attempt.update(
            {
                "finished_at_utc": utc_now(),
                "wall_time_s": wall,
                "exit_code": returncode,
                "budget_stop_reason": stop_reason,
            }
        )
        finished = {
            **running,
            "status": "failed",
            "message": f"Orchestrator/worker exception: {type(error).__name__}: {error}",
            "attempts": attempts + [attempt],
            "cumulative_accounted_attempt_wall_s": sum(
                float(item["wall_time_s"]) for item in attempts + [attempt]
            ),
        }
        _write_job_manifest(manifest_path, finished)
        raise
    finally:
        if process is not None:
            guard.unregister(process)
        guard.release(str(job["job_id"]))

    wall = time.perf_counter() - started
    attempt.update(
        {
            "finished_at_utc": utc_now(),
            "wall_time_s": wall,
            "exit_code": returncode,
            "budget_stop_reason": stop_reason,
            "stdout_log_sha256": sha256_file(logs / "stdout.log"),
            "stdout_log_bytes": (logs / "stdout.log").stat().st_size,
            "stderr_log_sha256": sha256_file(logs / "stderr.log"),
            "stderr_log_bytes": (logs / "stderr.log").stat().st_size,
        }
    )
    finished = {
        **running,
        "attempts": attempts + [attempt],
        "cumulative_accounted_attempt_wall_s": sum(
            float(item["wall_time_s"]) for item in attempts + [attempt]
        ),
    }
    if returncode != 0 or not output.is_file():
        finished.update(
            {
                "status": "failed",
                "message": (
                    f"Production worker stopped by {stop_reason}; preserved for resume"
                    if stop_reason is not None
                    else "Production worker failed; resume creates a new preserved attempt"
                ),
            }
        )
        _write_job_manifest(manifest_path, finished)
        return finished

    try:
        from .schemas import CanonicalG1

        motion = CanonicalG1.load(output)
        _validate_production_motion(
            motion,
            expected,
            sequence_value,
            execution_contract,
            repo_root=root,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        finished.update(
            {
                "status": "incomplete",
                "message": f"Exact Stage-2 output contract failed: {error}",
                "output_path": str(output),
                "output_sha256": sha256_file(output),
                "completion_ratio": len(motion.qpos) / source_frames
                if "motion" in locals()
                else 0.0,
            }
        )
        _write_job_manifest(manifest_path, finished)
        return finished
    if config["storage_projection"].get(
        "delete_rebuildable_native_intermediates_after_hash_capture"
    ) is not True:
        raise Stage2Error(
            "Production storage policy must delete rebuildable native work"
        )
    attempt["rebuildable_work_cleanup"] = _capture_and_delete_rebuildable_work(work)
    finished.update(
        {
            "status": "succeeded",
            "output_path": str(output),
            "output_sha256": sha256_file(output),
            "output_bytes": output.stat().st_size,
            "output_qpos_sha256": __import__(
                "retargeting_comparison.native_target_capture",
                fromlist=["tensor_sha256"],
            ).tensor_sha256(np.asarray(motion.qpos, dtype=np.float64)),
            "completion_ratio": 1.0,
        }
    )
    _write_job_manifest(manifest_path, finished)
    return finished


def _directory_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except FileNotFoundError:
            # Native workers may atomically replace or delete rebuildable
            # intermediates while another lane performs budget accounting.
            continue
    return total


def _capture_and_delete_rebuildable_work(work: Path) -> dict[str, Any]:
    """Hash a successful job's rebuildable native work tree, then remove it."""

    files: list[dict[str, Any]] = []
    if work.exists():
        for path in sorted(work.rglob("*")):
            if path.is_symlink():
                raise Stage2Error(
                    f"Refusing to clean a symlinked native intermediate: {path}"
                )
            if path.is_file():
                files.append(
                    {
                        "relative_path": path.relative_to(work).as_posix(),
                        "sha256": sha256_file(path),
                        "bytes": path.stat().st_size,
                    }
                )
        shutil.rmtree(work)
    receipt = {
        "role": "rebuildable_native_intermediates_deleted_after_hash_capture",
        "files": files,
        "files_sha256": _canonical_sha256(files),
        "total_bytes_before_deletion": sum(int(item["bytes"]) for item in files),
        "deleted": not work.exists(),
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return receipt


@contextmanager
def _exclusive_execution_lock(run_root: Path):
    """Prevent two launchers from consuming the same immutable plan at once."""

    import fcntl

    run_root.mkdir(parents=True, exist_ok=True)
    path = run_root / "execution.lock"
    with path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise Stage2Error(
                "Another Stage-2 launcher holds the execution lock"
            ) from error
        stream.seek(0)
        stream.truncate()
        stream.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "host": platform.node(),
                    "acquired_at_utc": utc_now(),
                },
                sort_keys=True,
            )
            + "\n"
        )
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _write_execution_attempt(path: Path, value: Mapping[str, Any]) -> None:
    payload = dict(value)
    payload["payload_sha256"] = _payload_sha256(payload)
    atomic_write_json(path, payload)


def _account_prior_execution_wall(
    run_root: Path, *, expected_plan_sha256: str | None = None
) -> float:
    """Return cumulative launch wall time, conservatively closing crash records.

    A clean invocation stores its measured active wall duration.  If the
    orchestrator itself died before finalization, elapsed UTC time through the
    next resume is charged to the budget.  This deliberately errs toward an
    early stop instead of silently granting another 48-hour allowance.
    """

    directory = run_root / "execution_attempts"
    directory.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    total = 0.0
    for path in sorted(directory.glob("attempt-*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise Stage2Error(
                f"Unreadable Stage-2 execution attempt: {path}"
            ) from error
        if value.get("schema_version") == SCHEMA_VERSION:
            if value.get("payload_sha256") != _payload_sha256(value):
                raise Stage2Error(f"Execution attempt payload hash failed: {path}")
            if (
                expected_plan_sha256 is not None
                and value.get("plan_sha256") != expected_plan_sha256
            ):
                raise Stage2Error(f"Execution attempt belongs to another plan: {path}")
        elif expected_plan_sha256 is not None:
            raise Stage2Error(f"Legacy unbound execution attempt is not resumable: {path}")
        recorded = value.get("active_wall_s")
        if recorded is not None:
            elapsed = float(recorded)
            if elapsed < 0.0:
                raise Stage2Error(f"Negative Stage-2 execution duration: {path}")
        elif value.get("status") == "running" and value.get("started_at_utc"):
            elapsed = max(
                0.0,
                (now - _parse_utc(str(value["started_at_utc"]))).total_seconds(),
            )
            value.update(
                {
                    "status": "abandoned_conservatively_accounted",
                    "finished_at_utc": now.isoformat(),
                    "active_wall_s": elapsed,
                    "accounting_note": (
                        "orchestrator did not finalize; elapsed UTC interval was "
                        "charged fail-closed at resume"
                    ),
                }
            )
            _write_execution_attempt(path, value)
        else:
            raise Stage2Error(f"Execution attempt lacks auditable wall time: {path}")
        total += elapsed
    return total


def _begin_execution_attempt(
    run_root: Path, prior_wall_s: float, plan: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    directory = run_root / "execution_attempts"
    count = len(list(directory.glob("attempt-*.json")))
    path = directory / f"attempt-{count + 1:03d}.json"
    value = {
        "schema_version": SCHEMA_VERSION,
        "plan_sha256": plan["plan_sha256"],
        "design_id": plan["design_id"],
        "repository_contract_sha256": _canonical_sha256(plan["repository"]),
        "status": "running",
        "started_at_utc": utc_now(),
        "host": platform.node(),
        "pid": os.getpid(),
        "prior_cumulative_active_wall_s": prior_wall_s,
    }
    _write_execution_attempt(path, value)
    return path, value


def validate_plan_for_launch(
    repo_root: str | Path,
    plan_path: str | Path,
    config_path: str | Path = "configs/stage2.yaml",
    *,
    expected_plan_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    root = Path(repo_root).resolve()
    path = Path(plan_path)
    if not path.is_absolute():
        path = root / path
    plan = json.loads(path.read_text())
    if not verify_plan_hash(plan) or plan.get("plan_sha256") != expected_plan_sha256:
        raise Stage2Error(
            "Stage-2 plan hash is invalid or was not acknowledged exactly"
        )
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config = load_yaml(config_file)
    _validate_authorization(config)
    current_bound_verdict, _ = _load_stage1_gate(root, config)
    if plan.get("stage1_bound_verdict") != current_bound_verdict:
        raise Stage2Error("Bound Stage-1 verdict changed after preflight")
    formal_evidence = _validate_stage1_formal_outputs(root, config)
    if plan.get("stage1_formal_evidence", {}) != formal_evidence:
        raise Stage2Error("Stage-1 formal-output evidence changed after preflight")
    timing_evidence = _validate_stage1_timing_evidence(root, config)
    if plan.get("stage1_timing_evidence", {}) != timing_evidence:
        raise Stage2Error("Stage-1 timing campaign changed after preflight")
    cost_probe = _validate_cost_probe(root, config)
    if plan.get("preflight_cost_probe", {}) != cost_probe:
        raise Stage2Error("Stage-2 measured/conservative cost probe changed")
    if cost_probe:
        config["budget_allowances"]["source_preparation_rtf"] = float(
            cost_probe["conservative_source_preparation_rtf"]
        )
        config["budget_allowances"]["evaluator_rtf_per_method"] = float(
            cost_probe["conservative_evaluator_rtf_per_method"]
        )
        config["storage_projection"]["canonical_source_bytes_per_frame"] = int(
            np.ceil(cost_probe["conservative_canonical_source_bytes_per_frame"])
        )
    unitree_contract = _validate_unitree_reference_contract(root, config)
    if plan.get("unitree_reference_contract", {}) != unitree_contract:
        raise Stage2Error("Unitree reference/asset/FK contract changed after preflight")
    if plan.get("config_sha256") != sha256_file(config_file):
        raise Stage2Error("Stage-2 configuration changed after preflight")
    current_repository = _repository_identity(root, config)
    _validate_repository_gate(current_repository, config)
    if plan.get("repository") != current_repository:
        raise Stage2Error("Repository code or tracked patch changed after preflight")
    current_hardware = _hardware_identity(config)
    if plan.get("hardware") != current_hardware:
        raise Stage2Error("Stage-2 launch host hardware changed after preflight")
    if config.get("production_design", {}).get("required") is True and tuple(
        plan.get("selected_production_methods", ())
    ) != PRODUCTION_METHODS:
        raise Stage2Error("Stage-2 plan does not contain the exact five+reference design")
    method_specs = [MethodSpec(**value) for value in plan["method_specs"]]
    current_method_digest = _method_design_sha256(method_specs, config)
    if (
        current_method_digest != plan.get("method_design_sha256")
        or not str(plan.get("design_id", "")).endswith(current_method_digest[:12])
    ):
        raise Stage2Error("Stage-2 method/config design digest changed")
    calibration_contract = plan.get("calibration_contract", {})
    calibration_payload = dict(calibration_contract)
    calibration_digest = calibration_payload.pop("contract_sha256", None)
    if calibration_digest != _canonical_sha256(calibration_payload):
        raise Stage2Error("Stage-2 calibration contract payload hash failed")
    for key in (
        "evaluator_manifest",
        "calibration_implementation",
        "robot_model_implementation",
        "stage2_preparation_implementation",
    ):
        if key not in calibration_contract:
            if config.get("production_design", {}).get("required") is True:
                raise Stage2Error(f"Stage-2 calibration receipt is missing: {key}")
            continue
        receipt = calibration_contract.get(key, {})
        path_value = Path(str(receipt.get("path", "")))
        path_value = path_value if path_value.is_absolute() else root / path_value
        if _file_receipt(root, path_value) != receipt:
            raise Stage2Error(f"Stage-2 calibration input changed: {key}")
    stage1_path = root / plan["stage1_validation_path"]
    if plan.get("stage1_validation_sha256") != sha256_file(stage1_path):
        raise Stage2Error("Stage-1 evidence changed after Stage-2 preflight")
    gates = plan.get("gates", {})
    if not all(
        gates.get(key) is True
        for key in (
            "user_authorization_verified",
            "stage1_go_verified",
            "preflight_within_wall_budget",
            "preflight_within_storage_budget",
            "launchable",
        )
    ):
        raise Stage2Error("Stage-2 plan contains a closed authorization or budget gate")
    for value in plan["selected_sequences"]:
        sequence = SequenceSpec.from_dict(value)
        source = root / config["dataset"]["root"] / sequence.relative_path
        if not source.is_file() or sha256_file(source) != sequence.source_sha256:
            raise Stage2Error(
                f"Stage-2 source inventory changed: {sequence.relative_path}"
            )
        if sequence.reference_relative_path is not None:
            reference = (
                root
                / config["reference_corpus"]["root"]
                / sequence.reference_relative_path
            )
            if (
                not reference.is_file()
                or sha256_file(reference) != sequence.reference_sha256
            ):
                raise Stage2Error(
                    f"Stage-2 reference inventory changed: {sequence.reference_relative_path}"
                )
    # Re-audit the complete 77-source/40-reference inventory, not only the
    # selected subset. This catches added, removed, non-finite, or frame-shifted
    # reference files between planning and launch.
    full_inventory = discover_lafan_sequences(
        root, config, enforce_expected_inventory=True
    )
    if inventory_sha256(full_inventory) != str(plan.get("full_inventory_sha256")):
        raise Stage2Error("Full LAFAN/reference inventory changed after preflight")
    return plan, config, path


def _execute_stage2_locked(
    root: Path,
    run_root: Path,
    plan: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute one exclusive, cumulatively budgeted Stage-2 launch attempt."""

    stop_fraction = float(config["budget"]["stop_before_limit_fraction"])
    total_wall_stop_s = (
        float(config["budget"]["wall_time_hours"]) * 3600.0 * stop_fraction
    )
    prior_wall_s = _account_prior_execution_wall(
        run_root, expected_plan_sha256=str(plan["plan_sha256"])
    )
    projection = plan.get("projection", {})
    safety_factor = float(config["budget"].get("safety_factor", 1.0))
    analysis_reserved_wall_s = (
        float(projection.get("wall_components_s", {}).get("evaluation", 0.0))
        * safety_factor
    )
    remaining_wall_s = total_wall_stop_s - prior_wall_s - analysis_reserved_wall_s
    if remaining_wall_s <= 0.0:
        raise Stage2Error(
            "Stage-2 cumulative runtime guard is already exhausted; refusing a fresh deadline"
        )
    attempt_path, attempt = _begin_execution_attempt(run_root, prior_wall_s, plan)
    start = time.monotonic()
    outcome = "error"
    storage_stop = (
        float(config["budget"]["retained_storage_gb"]) * 1_000_000_000.0 * stop_fraction
    )
    allowances = config.get("budget_allowances", {})
    analysis_reserved_storage_bytes = int(
        (
            int(allowances.get("analysis_fixed_retained_bytes", 0))
            + len(plan.get("projection", {}).get("jobs", []))
            * int(allowances.get("analysis_retained_bytes_per_job", 0))
        )
        * safety_factor
    )
    storage_stop -= analysis_reserved_storage_bytes
    if storage_stop <= 0:
        raise Stage2Error("Stage-2 analysis storage reserve exhausts the hard budget")
    deadline = start + remaining_wall_s
    frozen_raw_input_bytes = 0
    if config.get("storage_projection", {}).get(
        "retain_raw_inputs_in_incremental_budget"
    ):
        frozen_raw_input_bytes = sum(
            int(value["source_size_bytes"]) + int(value["reference_size_bytes"])
            for value in plan["selected_sequences"]
        )
    guard = _ExecutionGuard(
        run_root,
        deadline_monotonic=deadline,
        storage_stop_bytes=storage_stop,
        external_retained_bytes=frozen_raw_input_bytes,
    )
    try:
        with _signal_cancellation_scope(guard):
            sequence_by_id = {
                item.sequence_id: item
                for item in (
                    SequenceSpec.from_dict(value)
                    for value in plan["selected_sequences"]
                )
            }
            sequence_manifests: dict[str, Path] = {}
            for identifier, sequence in sequence_by_id.items():
                reason = guard.poll_reason()
                if reason is not None:
                    raise Stage2Error(f"Stage-2 preparation cancelled: {reason}")
                sequence_manifests[identifier] = prepare_sequence(
                    root, run_root, sequence, config, plan
                )
                reason = guard.poll_reason()
                if reason is not None:
                    raise Stage2Error(f"Stage-2 preparation exceeded guard: {reason}")

            jobs = list(plan["projection"]["jobs"])
            for phase_index in sorted({int(job["phase_index"]) for job in jobs}):
                reason = guard.poll_reason()
                if reason is not None:
                    raise Stage2Error(f"Stage-2 phase cancelled: {reason}")
                phase = [job for job in jobs if int(job["phase_index"]) == phase_index]
                lanes: dict[int, list[dict[str, Any]]] = {}
                for job in phase:
                    lanes.setdefault(int(job["worker_index"]), []).append(job)
                for lane in lanes.values():
                    lane.sort(
                        key=lambda job: (
                            float(job["projected_start_s"]),
                            str(job["job_id"]),
                        )
                    )

                def run_lane(lane: list[dict[str, Any]]) -> list[dict[str, Any]]:
                    results = []
                    for job in lane:
                        if guard.poll_reason() is not None:
                            break
                        results.append(
                            _run_job(
                                root,
                                run_root,
                                plan,
                                config,
                                job,
                                sequence_manifests[str(job["sequence_id"])],
                                guard,
                            )
                        )
                    return results

                with ThreadPoolExecutor(max_workers=max(1, len(lanes))) as pool:
                    # Exhaust every lane result so exceptions are propagated.
                    list(
                        pool.map(
                            run_lane,
                            [lanes[index] for index in sorted(lanes)],
                        )
                    )
                if guard.cancel_event.is_set():
                    raise Stage2Error(
                        f"Stage-2 shared execution guard cancelled all lanes: {guard.reason}"
                    )

        planned_statuses: list[str] = []
        for job in jobs:
            manifest_path = run_root / "job_manifests" / f"{job['job_id']}.json"
            expected = {**job, "plan_sha256": plan["plan_sha256"]}
            if _job_manifest_valid(manifest_path, expected):
                planned_statuses.append("succeeded")
                continue
            if manifest_path.is_file():
                try:
                    planned_statuses.append(
                        str(
                            json.loads(manifest_path.read_text()).get(
                                "status", "unknown"
                            )
                        )
                    )
                except (OSError, json.JSONDecodeError):
                    planned_statuses.append("invalid-manifest")
            else:
                planned_statuses.append("missing")
        status_counts = {
            status: planned_statuses.count(status)
            for status in sorted(
                set(planned_statuses) | {"succeeded", "incomplete", "failed", "missing"}
            )
        }
        all_planned_succeeded = bool(
            len(planned_statuses) == len(jobs)
            and planned_statuses
            and all(status == "succeeded" for status in planned_statuses)
        )
        outcome = "complete" if all_planned_succeeded else "incomplete"
        elapsed = time.monotonic() - start
        summary = {
            "schema_version": SCHEMA_VERSION,
            "status": outcome,
            "design_id": plan["design_id"],
            "plan_sha256": plan["plan_sha256"],
            "planned_job_count": len(jobs),
            "verified_succeeded_job_count": status_counts.get("succeeded", 0),
            "host": platform.node(),
            "elapsed_wall_s": elapsed,
            "prior_cumulative_active_wall_s": prior_wall_s,
            "cumulative_active_wall_s": prior_wall_s + elapsed,
            "cumulative_wall_guard_s": total_wall_stop_s,
            "analysis_reserved_wall_s": analysis_reserved_wall_s,
            "analysis_reserved_storage_bytes": analysis_reserved_storage_bytes,
            "retained_bytes": frozen_raw_input_bytes + _directory_size(run_root),
            "budget_ledger": {
                "hard_wall_limit_s": float(config["budget"]["wall_time_hours"])
                * 3600.0,
                "hard_storage_limit_bytes": float(
                    config["budget"]["retained_storage_gb"]
                )
                * 1_000_000_000.0,
                "stop_fraction": stop_fraction,
                "prior_active_wall_s": prior_wall_s,
                "current_active_wall_s": elapsed,
                "cumulative_active_wall_s": prior_wall_s + elapsed,
                "frozen_raw_input_bytes": frozen_raw_input_bytes,
                "run_directory_retained_bytes": _directory_size(run_root),
                "retained_bytes": frozen_raw_input_bytes + _directory_size(run_root),
                "analysis_reserved_wall_s": analysis_reserved_wall_s,
                "analysis_reserved_storage_bytes": analysis_reserved_storage_bytes,
            },
            "successful_job_manifests": [
                _file_receipt(
                    root,
                    run_root / "job_manifests" / f"{job['job_id']}.json",
                )
                for job in jobs
                if _job_manifest_valid(
                    run_root / "job_manifests" / f"{job['job_id']}.json",
                    {**job, "plan_sha256": plan["plan_sha256"]},
                )
            ],
            "job_status_counts": status_counts,
            "completed_at_utc": utc_now(),
        }
        summary["payload_sha256"] = _payload_sha256(summary)
        atomic_write_json(run_root / "execution_summary.json", summary)
        return summary
    finally:
        elapsed = time.monotonic() - start
        error = sys.exc_info()[1]
        attempt.update(
            {
                "status": f"finished_{outcome}",
                "finished_at_utc": utc_now(),
                "active_wall_s": elapsed,
                "cumulative_active_wall_s": prior_wall_s + elapsed,
                "error": None if error is None else f"{type(error).__name__}: {error}",
            }
        )
        _write_execution_attempt(attempt_path, attempt)


def execute_stage2(
    repo_root: str | Path,
    plan_path: str | Path,
    *,
    expected_plan_sha256: str,
    config_path: str | Path = "configs/stage2.yaml",
    launch: bool = False,
) -> dict[str, Any]:
    """Preview by default; an acknowledged launch executes fixed worker lanes."""

    root = Path(repo_root).resolve()
    plan, config, resolved_plan_path = validate_plan_for_launch(
        root,
        plan_path,
        config_path,
        expected_plan_sha256=expected_plan_sha256,
    )
    if not launch:
        return {
            "status": "preview",
            "launched": False,
            "plan_path": str(resolved_plan_path),
            "plan_sha256": plan["plan_sha256"],
            "design_id": plan["design_id"],
            "projection": plan["projection"],
        }

    run_root = root / config["canonicalization"]["output_root"] / plan["design_id"]
    with _exclusive_execution_lock(run_root):
        return _execute_stage2_locked(root, run_root, plan, config)
