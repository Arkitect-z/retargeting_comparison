"""Official Holosoma two-case Full/No-Hard interaction experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import atomic_write_json, atomic_write_text, sha256_file
from .runner import _aggregate_sha256, _conda_python, _git, utc_now
from .schemas import RunManifest, RunStatus


INTERACTION_CASES = {
    "box": {
        "task_name": "sub3_largebox_003",
        "task_type": "object_interaction",
        "data_format": "smplh",
        "relative_data_path": "demo_data/OMOMO_new",
    },
    "climb": {
        "task_name": "mocap_climb_seq_0",
        "task_type": "climbing",
        "data_format": "mocap",
        "relative_data_path": "demo_data/climb",
    },
}
INTERACTION_METRIC_REVISION = "source-conditioned-intended-contact-v4"
INTERACTION_PROVENANCE_SCHEMA_VERSION = 1
INTERACTION_OUTPUT_LEDGER_SCHEMA_VERSION = 1
HOLOSOMA_INTERACTION_COMMIT = "5f48635a3624656a5f46a07df26d43187e59f855"
HOLOSOMA_INTERACTION_PATCH = Path(
    "patches/holosoma/interaction-hard-constraint-flags.patch"
)
HOLOSOMA_INTERACTION_SOLVER = Path(
    "src/holosoma_retargeting/holosoma_retargeting/src/interaction_mesh_retargeter.py"
)
CONTACT_THRESHOLDS_M = {"2cm": 0.02, "5cm": 0.05, "10cm": 0.10}
CONTACT_PENETRATION_TOLERANCE_M = 0.0011
SOURCE_CONTACT_CONDITION_LABEL = "5cm"

# These are not post-hoc nearest-body choices.  Every source landmark and G1
# body is fixed by the official Holosoma task mapping before looking at either
# variant's result.
INTENDED_CONTACT_SPECS = {
    "box": (
        {
            "semantic": "left_hand_wrist",
            "role": "hand/wrist",
            "source_joint": "L_Wrist",
        },
        {
            "semantic": "right_hand_wrist",
            "role": "hand/wrist",
            "source_joint": "R_Wrist",
        },
    ),
    "climb": (
        {
            "semantic": "left_hand_wrist",
            "role": "hand/wrist",
            "source_joint": "LeftHandMiddle3",
        },
        {
            "semantic": "right_hand_wrist",
            "role": "hand/wrist",
            "source_joint": "RightHandMiddle3",
        },
        {
            "semantic": "left_ankle_foot",
            "role": "foot/ankle",
            "source_joint": "LeftFoot",
        },
        {
            "semantic": "right_ankle_foot",
            "role": "foot/ankle",
            "source_joint": "RightFoot",
        },
        {
            "semantic": "left_toe",
            "role": "toe",
            "source_joint": "LeftToeBase",
        },
        {
            "semantic": "right_toe",
            "role": "toe",
            "source_joint": "RightToeBase",
        },
    ),
}


def _json_bytes(value: Any) -> bytes:
    """Return the exact byte representation used by ``atomic_write_json``."""

    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _path_label(root: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return str(resolved)


def _resolve_path_label(root: Path, label: str) -> Path:
    path = Path(label)
    return path if path.is_absolute() else root / path


def _interaction_file_record(
    root: Path, path: Path, roles: list[str] | tuple[str, ...]
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Frozen interaction input is missing: {path}")
    return {
        "path": _path_label(root, path),
        "roles": sorted(set(roles)),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _interaction_files(directory: Path, pattern: str = "*") -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Frozen interaction input directory is missing: {directory}")
    return sorted(path for path in directory.rglob(pattern) if path.is_file())


def _holosoma_exact_diff(checkout: Path) -> tuple[bytes, list[str]]:
    command = [
        "git",
        "-C",
        str(checkout),
        "diff",
        "--binary",
        "--no-ext-diff",
        "--unified=0",
        "HEAD",
        "--",
    ]
    result = subprocess.run(command, check=True, capture_output=True)
    names = subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "diff",
            "--name-only",
            "HEAD",
            "--",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return result.stdout, sorted(value for value in names if value)


def _interaction_provenance_payload(
    root: Path, case: str, variant: str
) -> tuple[dict[str, Any], bytes]:
    """Build the deterministic, fail-closed input ledger for one ablation arm.

    The file list is deliberately conservative.  In particular, it binds the
    complete upstream ``models/g1`` asset closure so a mesh referenced through
    either URDF or generated MuJoCo XML cannot escape provenance validation.
    """

    if case not in INTERACTION_CASES or variant not in {"full", "no-hard"}:
        raise ValueError("Unknown interaction case or variant")
    root = root.resolve()
    checkout = root / "external" / "holosoma"
    package = checkout / "src" / "holosoma_retargeting" / "holosoma_retargeting"
    commit = _git(checkout, "rev-parse", "HEAD")
    if commit != HOLOSOMA_INTERACTION_COMMIT:
        raise RuntimeError(
            "Holosoma interaction checkout moved away from the frozen commit: "
            f"{commit}"
        )
    exact_diff, changed_paths = _holosoma_exact_diff(checkout)
    expected_changed = [HOLOSOMA_INTERACTION_SOLVER.as_posix()]
    if changed_paths != expected_changed:
        raise RuntimeError(
            "Holosoma interaction worktree must contain exactly the frozen solver "
            f"patch; observed {changed_paths}"
        )
    patch_path = root / HOLOSOMA_INTERACTION_PATCH
    if not patch_path.is_file():
        raise FileNotFoundError(f"Frozen Holosoma patch is missing: {patch_path}")
    tracked_patch = patch_path.read_bytes()
    if exact_diff != tracked_patch:
        raise RuntimeError(
            "The exact Holosoma worktree diff differs from the tracked interaction patch"
        )

    records: dict[str, dict[str, Any]] = {}

    def register(path: Path, *roles: str) -> None:
        label = _path_label(root, path)
        if label in records:
            records[label]["roles"] = sorted(
                set(records[label]["roles"]) | set(roles)
            )
            return
        records[label] = _interaction_file_record(root, path, roles)

    harness_names = (
        "interaction.py",
        "interaction_worker.py",
        "io_utils.py",
        "schemas.py",
        "runner.py",
    )
    for name in harness_names:
        register(root / "src" / "retargeting_comparison" / name, "harness_code")
    register(patch_path, "tracked_holosoma_patch")

    upstream_code = [
        package / "src" / "interaction_mesh_retargeter.py",
        package / "src" / "utils.py",
        package / "src" / "mujoco_utils.py",
        package / "src" / "viser_utils.py",
        package / "examples" / "robot_retarget.py",
        *_interaction_files(package / "config_types", "*.py"),
        *_interaction_files(package / "config_values", "*.py"),
    ]
    for path in sorted(set(upstream_code)):
        register(path, "holosoma_runtime_code")

    spec = INTERACTION_CASES[case]
    data_path = package / spec["relative_data_path"]
    if case == "box":
        primary_input = data_path / f"{spec['task_name']}.pt"
        object_paths = [
            *_interaction_files(package / "models" / "largebox"),
            *_interaction_files(package / "models" / "templates"),
        ]
    else:
        case_directory = data_path / spec["task_name"]
        primary_candidates = sorted(case_directory.glob("*.npy"))
        if len(primary_candidates) != 1:
            raise RuntimeError(
                "Climb interaction provenance requires exactly one primary motion NPY"
            )
        primary_input = primary_candidates[0]
        object_paths = [
            *(
                path
                for path in _interaction_files(case_directory)
                if path != primary_input
            ),
            *_interaction_files(package / "models" / "templates"),
        ]
    register(primary_input, "primary_human_motion", "full_untruncated_input")
    for path in sorted(set(object_paths)):
        register(path, "object_mesh_urdf_xml_or_template")
    for path in _interaction_files(package / "models" / "g1"):
        register(path, "robot_asset_closure")

    python = _conda_python("hsretargeting")
    python_resolved = python.resolve()
    history = python.parent.parent / "conda-meta" / "history"
    register(python_resolved, "hsretargeting_python_binary")
    register(history, "hsretargeting_conda_history")
    version = subprocess.run(
        [str(python), "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    version_text = (version.stdout or version.stderr).strip()
    enabled = variant == "full"
    ordered_records = [records[name] for name in sorted(records)]
    payload = {
        "schema_version": INTERACTION_PROVENANCE_SCHEMA_VERSION,
        "case": case,
        "variant": variant,
        "metric_revision": INTERACTION_METRIC_REVISION,
        "task": {
            "task_name": spec["task_name"],
            "task_type": spec["task_type"],
            "data_format": spec["data_format"],
            "full_primary_human_input": _path_label(root, primary_input),
        },
        "ablation_flags": {
            "activate_obj_non_penetration": enabled,
            "activate_foot_sticking": enabled,
            "activate_joint_limits": True,
        },
        "full_vs_no_hard_only_configuration_difference": (
            "variant plus the two coupled hard-constraint flags"
        ),
        "holosoma": {
            "checkout": _path_label(root, checkout),
            "commit": commit,
            "expected_commit": HOLOSOMA_INTERACTION_COMMIT,
            "changed_paths": changed_paths,
            "exact_worktree_diff_sha256": hashlib.sha256(exact_diff).hexdigest(),
            "tracked_patch_path": HOLOSOMA_INTERACTION_PATCH.as_posix(),
            "tracked_patch_sha256": hashlib.sha256(tracked_patch).hexdigest(),
            "exact_diff_byte_identical_to_tracked_patch": True,
            "exact_diff_artifact": "holosoma_worktree.patch",
        },
        "environment": {
            "name": "conda:hsretargeting",
            "python_requested_path": str(python),
            "python_resolved_path": str(python_resolved),
            "python_version": version_text,
            "conda_history_path": str(history.resolve()),
        },
        "asset_scope": {
            "robot": (
                "complete upstream models/g1 directory; conservative superset "
                "of URDF/XML runtime references"
            ),
            "object": "all case object mesh/URDF/XML/template inputs",
            "human": "one full, untruncated official case motion file",
        },
        "files": ordered_records,
        "file_count": len(ordered_records),
        "file_bundle_sha256": hashlib.sha256(
            json.dumps(
                ordered_records, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
    }
    return payload, exact_diff


def _interaction_config_sha256(root: Path, case: str, variant: str) -> str:
    payload, _ = _interaction_provenance_payload(root.resolve(), case, variant)
    return _json_sha256(payload)


def _verify_provenance_file_records(root: Path, payload: dict[str, Any]) -> None:
    records = payload.get("files")
    if not isinstance(records, list) or len(records) != int(
        payload.get("file_count", -1)
    ):
        raise ValueError("Interaction provenance file ledger is malformed")
    paths = [str(record.get("path", "")) for record in records]
    if paths != sorted(paths) or len(set(paths)) != len(paths):
        raise ValueError("Interaction provenance file ledger is not unique and sorted")
    expected_bundle = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if payload.get("file_bundle_sha256") != expected_bundle:
        raise ValueError("Interaction provenance file-bundle hash is stale")
    for record in records:
        path = _resolve_path_label(root, str(record["path"]))
        if (
            not path.is_file()
            or int(record.get("bytes", -1)) != path.stat().st_size
            or record.get("sha256") != sha256_file(path)
        ):
            raise ValueError(f"Interaction provenance input changed: {path}")


def _write_interaction_provenance_attempt(
    root: Path, case: str, variant: str, run_dir: Path
) -> tuple[Path, str]:
    payload, exact_diff = _interaction_provenance_payload(root, case, variant)
    run_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = run_dir / "provenance_ledger.json"
    atomic_write_json(ledger_path, payload)
    diff_path = run_dir / "holosoma_worktree.patch"
    atomic_write_text(diff_path, exact_diff.decode("utf-8"))
    ledger_hash = sha256_file(ledger_path)
    if ledger_hash != _json_sha256(payload):
        raise RuntimeError("Interaction provenance ledger serialization is unstable")
    if sha256_file(diff_path) != payload["holosoma"]["exact_worktree_diff_sha256"]:
        raise RuntimeError("Interaction exact-diff artifact was not written faithfully")
    return ledger_path, ledger_hash


def _verify_interaction_provenance_attempt(
    root: Path, case: str, variant: str, run_dir: Path
) -> tuple[dict[str, Any], str]:
    ledger_path = run_dir / "provenance_ledger.json"
    diff_path = run_dir / "holosoma_worktree.patch"
    if not ledger_path.is_file() or not diff_path.is_file():
        raise ValueError("Interaction attempt lacks its input provenance artifacts")
    stored = json.loads(ledger_path.read_text(encoding="utf-8"))
    _verify_provenance_file_records(root, stored)
    current, exact_diff = _interaction_provenance_payload(root, case, variant)
    if stored != current:
        raise ValueError("Interaction attempt provenance differs from current frozen inputs")
    if diff_path.read_bytes() != exact_diff:
        raise ValueError("Interaction exact-diff attempt artifact was tampered")
    return stored, sha256_file(ledger_path)


def _mujoco_name(model: Any, object_type: Any, index: int) -> str:
    import mujoco

    return mujoco.mj_id2name(model, object_type, int(index)) or ""


def _array_contract_sha256(values: dict[str, np.ndarray]) -> str:
    """Hash an array bundle independently of NPZ container timestamps."""
    digest = hashlib.sha256()
    for name in sorted(values):
        value = np.ascontiguousarray(values[name])
        digest.update(name.encode("utf-8"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(json.dumps(value.shape, separators=(",", ":")).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _object_collision_geoms(model: Any, object_token: str) -> list[int]:
    import mujoco

    result: list[int] = []
    for geom_id in range(model.ngeom):
        name = _mujoco_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        body_name = _mujoco_name(
            model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])
        )
        if (
            (object_token in name or object_token in body_name)
            and (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id])
        ):
            result.append(geom_id)
    if not result:
        raise RuntimeError(
            f"No enabled object collision geometry matches token {object_token!r}"
        )
    return result


def resolve_intended_contact_mapping(
    model: Any,
    case: str,
    demo_joints: list[str],
    official_mapping: dict[str, str],
    object_token: str,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Resolve the frozen source-landmark -> G1-body -> object-geom contract.

    Only collision geoms attached directly to the officially mapped G1 body
    are admitted.  Falling back to a nearest body or a whole-limb geom would
    make the contact metric outcome-dependent, so missing pairs fail closed.
    """
    import mujoco

    if case not in INTENDED_CONTACT_SPECS:
        raise ValueError(f"No intended-contact specification for {case!r}")
    if len(set(demo_joints)) != len(demo_joints):
        raise RuntimeError("Official demo-joint list contains duplicate names")
    object_geoms = _object_collision_geoms(model, object_token)
    object_names = [
        _mujoco_name(model, mujoco.mjtObj.mjOBJ_GEOM, value)
        for value in object_geoms
    ]
    if any(not name for name in object_names):
        raise RuntimeError("An intended object collision geom has no stable name")
    resolved: list[dict[str, Any]] = []
    for frozen in INTENDED_CONTACT_SPECS[case]:
        source_joint = str(frozen["source_joint"])
        if source_joint not in demo_joints:
            raise RuntimeError(
                f"Required intended-contact source joint {source_joint!r} is absent"
            )
        if source_joint not in official_mapping:
            raise RuntimeError(
                f"Official Holosoma mapping has no G1 body for {source_joint!r}"
            )
        body_name = str(official_mapping[source_joint])
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise RuntimeError(f"Mapped G1 body {body_name!r} is absent from the scene")
        robot_geoms = [
            geom_id
            for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) == body_id
            and (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id])
        ]
        if not robot_geoms:
            raise RuntimeError(
                f"Mapped G1 body {body_name!r} has no enabled collision geometry"
            )
        robot_geom_names = [
            _mujoco_name(model, mujoco.mjtObj.mjOBJ_GEOM, value)
            for value in robot_geoms
        ]
        if any(not name for name in robot_geom_names):
            raise RuntimeError(
                f"Mapped G1 body {body_name!r} has an unnamed collision geometry"
            )
        resolved.append(
            {
                **frozen,
                "source_joint_index": int(demo_joints.index(source_joint)),
                "robot_body_name": body_name,
                "robot_body_id": int(body_id),
                "robot_geom_ids": [int(value) for value in robot_geoms],
                "robot_geom_names": robot_geom_names,
                "object_geom_ids": [int(value) for value in object_geoms],
                "object_geom_names": object_names,
            }
        )
    if len({row["semantic"] for row in resolved}) != len(resolved):
        raise RuntimeError("Intended-contact semantics are not unique")
    return resolved, object_geoms


def _quaternion_wxyz_to_matrix(value: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Object pose contains an invalid quaternion")
    w, x, y, z = quaternion / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _source_surface_distances(
    mesh_path: Path,
    world_points: np.ndarray,
    object_poses_xyz_wxyz: np.ndarray,
    mesh_scale_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Point-to-triangle distances on the actual source object mesh.

    The raw mesh is scaled in its local frame, and each query is transformed
    through the source/demo object pose.  `closest_point_naive` is deliberately
    used: unlike point-cloud or object-origin distances it queries triangles,
    and it does not depend on an optional spatial-index package.
    """
    import trimesh

    points = np.asarray(world_points, dtype=np.float64)
    poses = np.asarray(object_poses_xyz_wxyz, dtype=np.float64)
    scale = np.asarray(mesh_scale_xyz, dtype=np.float64)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError("Source intended-contact points must have shape [T,S,3]")
    if poses.shape != (len(points), 7):
        raise ValueError("Source object poses must have shape [T,7] in xyz+wxyz order")
    if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise ValueError("Source object mesh scale must be three finite positives")
    loaded = trimesh.load(mesh_path, force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh) or not len(loaded.faces):
        raise RuntimeError(f"Source object surface is not a triangle mesh: {mesh_path}")
    mesh = loaded.copy()
    mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) * scale[None, :]
    local = np.empty_like(points)
    for frame, pose in enumerate(poses):
        rotation = _quaternion_wxyz_to_matrix(pose[3:])
        local[frame] = (points[frame] - pose[:3]) @ rotation
    flat = local.reshape(-1, 3)
    unsigned = np.empty(len(flat), dtype=np.float64)
    triangles = np.empty(len(flat), dtype=np.int64)
    # Largebox has 27k triangles.  Batching prevents the naive exact backend
    # from allocating a points x triangles tensor for the entire sequence.
    for begin in range(0, len(flat), 8):
        end = min(begin + 8, len(flat))
        _, distance, triangle = trimesh.proximity.closest_point_naive(
            mesh, flat[begin:end]
        )
        unsigned[begin:end] = np.asarray(distance, dtype=np.float64)
        triangles[begin:end] = np.asarray(triangle, dtype=np.int64)
    if not np.isfinite(unsigned).all():
        raise RuntimeError("Non-finite source mesh/surface distance")

    # A closed source mesh supports an inside/outside sign.  The published
    # largebox mesh is open, so its source distance is explicitly unsigned;
    # robot distances remain signed MuJoCo surface distances in both cases.
    sign_available = bool(mesh.is_watertight and mesh.is_winding_consistent)
    signed = unsigned.copy()
    if sign_available:
        tri = np.asarray(mesh.triangles, dtype=np.float64)
        for index, point in enumerate(flat):
            a = tri[:, 0] - point
            b = tri[:, 1] - point
            c = tri[:, 2] - point
            numerator = np.einsum("ij,ij->i", a, np.cross(b, c))
            na = np.linalg.norm(a, axis=1)
            nb = np.linalg.norm(b, axis=1)
            nc = np.linalg.norm(c, axis=1)
            denominator = (
                na * nb * nc
                + np.einsum("ij,ij->i", a, b) * nc
                + np.einsum("ij,ij->i", b, c) * na
                + np.einsum("ij,ij->i", c, a) * nb
            )
            solid_angle = float(np.sum(2.0 * np.arctan2(numerator, denominator)))
            if abs(solid_angle) > 2.0 * np.pi:
                signed[index] *= -1.0
    shape = points.shape[:2]
    metadata = {
        "backend": "trimesh exact point-to-triangle closest_point_naive",
        "mesh_watertight": bool(mesh.is_watertight),
        "mesh_winding_consistent": bool(mesh.is_winding_consistent),
        "source_signed_distance_available": sign_available,
        "source_sign_policy": (
            "negative inside closed oriented mesh"
            if sign_available
            else "unsigned: published source mesh is not watertight"
        ),
        "vertices": int(len(mesh.vertices)),
        "triangles": int(len(mesh.faces)),
    }
    return (
        unsigned.reshape(shape),
        signed.reshape(shape),
        triangles.reshape(shape),
        metadata,
    )


def _runtime_object_mesh_scale(
    model: Any, object_geoms: list[int]
) -> tuple[np.ndarray, list[str]]:
    """Read the physical mesh scale from the compiled MuJoCo object geoms."""
    import mujoco

    scales: list[np.ndarray] = []
    mesh_names: list[str] = []
    for geom_id in object_geoms:
        mesh_id = int(model.geom_dataid[geom_id])
        if mesh_id < 0:
            raise RuntimeError("Intended object collision geom is not a mesh")
        scales.append(np.asarray(model.mesh_scale[mesh_id], dtype=np.float64))
        mesh_names.append(_mujoco_name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id))
    if not scales or not np.allclose(scales, scales[0], atol=1e-12, rtol=0.0):
        raise RuntimeError("Intended object collision meshes do not share one scale")
    return scales[0], mesh_names


def _paired_sample_scale(
    object_points_local: np.ndarray, object_points_local_demo: np.ndarray
) -> tuple[np.ndarray, float]:
    """Recover the official source/object scale from paired sampled points."""
    source = np.asarray(object_points_local, dtype=np.float64)
    demo = np.asarray(object_points_local_demo, dtype=np.float64)
    if source.shape != demo.shape or source.ndim != 2 or source.shape[1] != 3:
        raise RuntimeError("Official object sample arrays are not paired [N,3] values")
    numerator = np.sum(source * demo, axis=0)
    denominator = np.sum(source * source, axis=0)
    if np.any(denominator < 1e-12):
        raise RuntimeError("Official object samples cannot identify all three scale axes")
    scale = numerator / denominator
    residual = float(np.sqrt(np.mean((source * scale - demo) ** 2)))
    if not np.isfinite(scale).all() or np.any(scale <= 0.0) or residual > 1e-8:
        raise RuntimeError(
            f"Official object samples do not define a pure local scale (rms={residual})"
        )
    return scale, residual


def intended_contact_diagnostics(
    model: Any,
    qpos: np.ndarray,
    case: str,
    object_token: str,
    demo_joints: list[str],
    official_mapping: dict[str, str],
    human_joint_motions: np.ndarray,
    object_poses_xyz_wxyz: np.ndarray,
    object_points_local_demo: np.ndarray,
    object_points_local: np.ndarray,
    source_mesh_path: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate source-conditioned contact preservation on frozen semantic pairs."""
    import mujoco

    trajectory = np.asarray(qpos, dtype=np.float64)
    human = np.asarray(human_joint_motions, dtype=np.float64)
    poses = np.asarray(object_poses_xyz_wxyz, dtype=np.float64)
    if trajectory.ndim != 2 or len(trajectory) == 0:
        raise ValueError("Interaction qpos must be a non-empty [T,nq] array")
    if trajectory.shape[1] != model.nq:
        raise ValueError("Interaction qpos width does not match the runtime scene")
    if human.ndim != 3 or human.shape[2] != 3 or len(human) != len(trajectory):
        raise ValueError("Captured source joint motion is not frame-aligned with qpos")
    if poses.shape != (len(trajectory), 7):
        raise ValueError("Captured source object poses are not frame-aligned xyz+wxyz")
    if not (
        np.isfinite(trajectory).all()
        and np.isfinite(human).all()
        and np.isfinite(poses).all()
    ):
        raise ValueError("Interaction contact inputs contain NaN or Inf")

    mapping, object_geoms = resolve_intended_contact_mapping(
        model, case, demo_joints, official_mapping, object_token
    )
    selected = np.stack(
        [human[:, int(row["source_joint_index"]), :] for row in mapping], axis=1
    )
    if case == "box":
        source_scale, scale_residual = _paired_sample_scale(
            object_points_local, object_points_local_demo
        )
        scale_provenance = "official paired object_points_local -> object_points_local_demo"
    else:
        source_scale, runtime_mesh_names = _runtime_object_mesh_scale(
            model, object_geoms
        )
        scale_residual = 0.0
        scale_provenance = "compiled MuJoCo object collision mesh_scale"
        for row in mapping:
            row["runtime_object_mesh_names"] = runtime_mesh_names
    source_unsigned, source_signed, source_triangles, surface_metadata = (
        _source_surface_distances(
            Path(source_mesh_path), selected, poses, source_scale
        )
    )

    data = mujoco.MjData(model)
    rows: list[dict[str, Any]] = []
    for frame, value in enumerate(trajectory):
        data.qpos[:] = value
        mujoco.mj_forward(model, data)
        for semantic_index, contract in enumerate(mapping):
            fromto = np.zeros(6, dtype=np.float64)
            pairs: list[tuple[float, int, int]] = []
            for robot_geom in contract["robot_geom_ids"]:
                for object_geom in object_geoms:
                    distance = float(
                        mujoco.mj_geomDistance(
                            model, data, robot_geom, object_geom, 10.0, fromto
                        )
                    )
                    pairs.append((distance, int(robot_geom), int(object_geom)))
            if not pairs:
                raise RuntimeError(
                    f"No runtime geom pair for intended semantic {contract['semantic']}"
                )
            robot_distance, robot_geom, object_geom = min(
                pairs, key=lambda item: item[0]
            )
            source_distance = float(source_signed[frame, semantic_index])
            source_abs = float(source_unsigned[frame, semantic_index])
            source_valid = (
                source_distance >= -CONTACT_PENETRATION_TOLERANCE_M
                if surface_metadata["source_signed_distance_available"]
                else True
            )
            robot_valid = robot_distance >= -CONTACT_PENETRATION_TOLERANCE_M
            row: dict[str, Any] = {
                "frame": int(frame),
                "semantic": contract["semantic"],
                "role": contract["role"],
                "source_joint_name": contract["source_joint"],
                "source_joint_index": int(contract["source_joint_index"]),
                "robot_body_name": contract["robot_body_name"],
                "robot_geom_names": "|".join(contract["robot_geom_names"]),
                "object_geom_names": "|".join(contract["object_geom_names"]),
                "source_absolute_surface_distance_m": source_abs,
                "source_reference_surface_distance_m": source_distance,
                "source_signed_distance_available": bool(
                    surface_metadata["source_signed_distance_available"]
                ),
                "source_closest_triangle_id": int(
                    source_triangles[frame, semantic_index]
                ),
                "mapped_robot_object_signed_surface_distance_m": robot_distance,
                "mapped_robot_penetration_depth_m": max(0.0, -robot_distance),
                "mapped_robot_valid_separation": bool(robot_valid),
                "closest_mapped_robot_geom_name": _mujoco_name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, robot_geom
                ),
                "closest_object_geom_name": _mujoco_name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, object_geom
                ),
                "signed_robot_minus_source_reference_distance_m": (
                    robot_distance - source_distance
                ),
                "absolute_robot_source_distance_error_m": abs(
                    robot_distance - source_distance
                ),
            }
            for label, threshold in CONTACT_THRESHOLDS_M.items():
                row[f"source_contact_{label}"] = bool(
                    source_valid and source_distance <= threshold
                )
                row[f"mapped_robot_within_{label}"] = bool(
                    robot_valid and robot_distance <= threshold
                )
                row[f"preserved_{label}_given_source_{label}"] = bool(
                    row[f"source_contact_{label}"]
                    and row[f"mapped_robot_within_{label}"]
                )
            rows.append(row)

    def summarise(values: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {"observations": len(values)}
        for source_label in CONTACT_THRESHOLDS_M:
            conditioned = [row for row in values if row[f"source_contact_{source_label}"]]
            result[f"source_contact_observations_{source_label}"] = len(conditioned)
            for robot_label in CONTACT_THRESHOLDS_M:
                result[
                    f"preservation_{robot_label}_given_source_{source_label}"
                ] = (
                    float(
                        np.mean(
                            [row[f"mapped_robot_within_{robot_label}"] for row in conditioned]
                        )
                    )
                    if conditioned
                    else None
                )
            result[f"penetration_rate_given_source_{source_label}"] = (
                float(
                    np.mean(
                        [
                            row["mapped_robot_penetration_depth_m"]
                            > CONTACT_PENETRATION_TOLERANCE_M
                            for row in conditioned
                        ]
                    )
                )
                if conditioned
                else None
            )
            result[f"mean_signed_distance_error_m_given_source_{source_label}"] = (
                float(
                    np.mean(
                        [
                            row["signed_robot_minus_source_reference_distance_m"]
                            for row in conditioned
                        ]
                    )
                )
                if conditioned
                else None
            )
            result[f"mean_absolute_distance_error_m_given_source_{source_label}"] = (
                float(
                    np.mean(
                        [
                            row["absolute_robot_source_distance_error_m"]
                            for row in conditioned
                        ]
                    )
                )
                if conditioned
                else None
            )
        return result

    per_semantic = {
        str(contract["semantic"]): summarise(
            [row for row in rows if row["semantic"] == contract["semantic"]]
        )
        for contract in mapping
    }
    aggregate = summarise(rows)
    # Human joint centres are landmark proxies, not skin surfaces.  The frozen
    # 5 cm near-contact envelope is therefore the conditioning set; the stricter
    # 2 cm count remains reported and can legitimately be zero.  A case is not
    # publishable if any declared semantic never enters that source envelope.
    missing_source_contact = [
        semantic
        for semantic, values in per_semantic.items()
        if int(
            values[
                f"source_contact_observations_{SOURCE_CONTACT_CONDITION_LABEL}"
            ]
        )
        == 0
    ]
    if missing_source_contact:
        raise RuntimeError(
            "Could not establish the frozen 5 cm source intended-contact envelope for: "
            + ", ".join(missing_source_contact)
        )

    mapping_for_hash = [
        {
            key: value
            for key, value in contract.items()
            if key not in {"robot_body_id", "robot_geom_ids", "object_geom_ids"}
        }
        for contract in mapping
    ]
    mapping_payload = {
        "schema_version": 1,
        "metric_revision": INTERACTION_METRIC_REVISION,
        "case": case,
        "selection_policy": "frozen official source-joint to directly attached G1 collision geoms",
        "object_token": object_token,
        "source_mesh_path": str(source_mesh_path),
        "source_mesh_sha256": sha256_file(source_mesh_path),
        "source_mesh_scale_xyz": source_scale.tolist(),
        "source_mesh_scale_provenance": scale_provenance,
        "source_mesh_scale_fit_rms_m": scale_residual,
        "surface_query": surface_metadata,
        "penetration_tolerance_m": CONTACT_PENETRATION_TOLERANCE_M,
        "thresholds_m": CONTACT_THRESHOLDS_M,
        "source_contact_condition_label": SOURCE_CONTACT_CONDITION_LABEL,
        "source_contact_condition_rationale": (
            "human joint centres are landmark proxies; use the predeclared 5 cm near-contact envelope"
        ),
        "mappings": mapping_for_hash,
    }
    mapping_bytes = json.dumps(
        mapping_payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    summary: dict[str, Any] = {
        "intended_contact_pair_asserted": True,
        "intended_contact_fail_closed": True,
        "intended_contact_semantic_count": len(mapping),
        "intended_contact_observations": len(rows),
        "intended_contact_distance_scope": (
            "frozen official source landmark -> directly mapped G1 body collision geoms "
            "against true object collision surfaces"
        ),
        "source_surface_distance_backend": surface_metadata["backend"],
        "mapped_robot_surface_distance_backend": (
            "mujoco.mj_geomDistance over frozen mapped-body/object-geom pairs"
        ),
        "source_surface_mesh_sha256": sha256_file(source_mesh_path),
        "intended_contact_mapping_canonical_sha256": hashlib.sha256(
            mapping_bytes
        ).hexdigest(),
        "intended_contact_aggregate": aggregate,
        "intended_contact_per_semantic": per_semantic,
        "intended_source_contact_condition": (
            "source landmark has valid true-surface separation <= 5 cm"
        ),
        "intended_source_contact_condition_threshold_m": CONTACT_THRESHOLDS_M[
            SOURCE_CONTACT_CONDITION_LABEL
        ],
        "intended_source_contact_observations": int(
            aggregate[
                f"source_contact_observations_{SOURCE_CONTACT_CONDITION_LABEL}"
            ]
        ),
        "intended_mapped_penetration_rate_given_source_contact_5cm": aggregate[
            f"penetration_rate_given_source_{SOURCE_CONTACT_CONDITION_LABEL}"
        ],
        "intended_robot_signed_minus_source_reference_distance_mean_m_given_source_contact_5cm": (
            aggregate[
                f"mean_signed_distance_error_m_given_source_{SOURCE_CONTACT_CONDITION_LABEL}"
            ]
        ),
        "intended_absolute_distance_error_mean_m_given_source_contact_5cm": aggregate[
            f"mean_absolute_distance_error_m_given_source_{SOURCE_CONTACT_CONDITION_LABEL}"
        ],
    }
    for label in CONTACT_THRESHOLDS_M:
        summary[f"intended_source_contact_observations_{label}"] = int(
            aggregate[f"source_contact_observations_{label}"]
        )
        summary[
            f"intended_contact_preservation_{label}_given_source_contact_5cm"
        ] = aggregate[
            f"preservation_{label}_given_source_{SOURCE_CONTACT_CONDITION_LABEL}"
        ]
    contract_arrays = {
        "semantic_names": np.asarray(
            [row["semantic"] for row in mapping], dtype="U64"
        ),
        "source_joint_names": np.asarray(
            [row["source_joint"] for row in mapping], dtype="U64"
        ),
        "robot_body_names": np.asarray(
            [row["robot_body_name"] for row in mapping], dtype="U128"
        ),
        "source_joint_positions": selected,
        "source_object_poses_xyz_wxyz": poses,
        "source_mesh_scale_xyz": source_scale,
        "object_points_local_demo": np.asarray(
            object_points_local_demo, dtype=np.float64
        ),
        "object_points_local": np.asarray(object_points_local, dtype=np.float64),
    }
    return rows, summary, mapping_payload, contract_arrays


def mesh_surface_diagnostics(
    model: Any,
    qpos: np.ndarray,
    object_token: str,
    foot_body_names: list[str],
    foot_sticking: list[dict[str, bool]],
    foot_tolerance_m: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compute mutually interpretable signed surface-distance diagnostics.

    The minimum is over all enabled robot/object collision pairs.  It is a
    whole-robot proximity diagnostic, not an assertion that the closest body
    is the task-intended contact body.  Penetration deeper than the frozen
    1.1 mm tolerance is never counted as successful contact.
    """
    import mujoco

    data = mujoco.MjData(model)
    object_geoms = _object_collision_geoms(model, object_token)
    object_geom_set = set(object_geoms)
    robot_geoms = []
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        if (
            geom_id not in object_geom_set
            and body_id != 0
            and (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id])
        ):
            robot_geoms.append(geom_id)
    if not object_geoms or not robot_geoms:
        raise RuntimeError("Could not identify both robot and object collision surfaces")
    foot_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in foot_body_names
    ]
    if any(value < 0 for value in foot_ids):
        raise RuntimeError("Interaction scene is missing a configured foot-sticking body")
    rows: list[dict[str, Any]] = []
    previous_feet: np.ndarray | None = None
    for frame, value in enumerate(qpos):
        data.qpos[:] = value
        mujoco.mj_forward(model, data)
        fromto = np.zeros(6, dtype=np.float64)
        candidates = [
            (
                float(mujoco.mj_geomDistance(model, data, robot, obj, 10.0, fromto)),
                robot,
                obj,
            )
            for robot in robot_geoms
            for obj in object_geoms
        ]
        distance, closest_robot_geom, closest_object_geom = min(
            candidates, key=lambda item: item[0]
        )
        penetration_tolerance_m = CONTACT_PENETRATION_TOLERANCE_M
        valid_surface_separation = distance >= -penetration_tolerance_m
        feet = data.xpos[foot_ids].copy()
        displacement = np.zeros(len(feet)) if previous_feet is None else np.linalg.norm(
            feet[:, :2] - previous_feet[:, :2], axis=1
        )
        previous_feet = feet
        flags = foot_sticking[frame] if frame < len(foot_sticking) else {}
        left_stance = any(bool(flag) for key, flag in flags.items() if key.lower().startswith("l"))
        right_stance = any(bool(flag) for key, flag in flags.items() if key.lower().startswith("r"))
        left_motion = float(max((d for d, name in zip(displacement, foot_body_names) if "left" in name), default=0.0))
        right_motion = float(max((d for d, name in zip(displacement, foot_body_names) if "right" in name), default=0.0))
        rows.append(
            {
                "frame": frame,
                "minimum_robot_object_surface_distance_m": distance,
                "closest_robot_geom_id": closest_robot_geom,
                "closest_object_geom_id": closest_object_geom,
                "strict_contact_2cm": valid_surface_separation and distance <= 0.02,
                "near_contact_2_to_5cm": 0.02 < distance <= 0.05,
                "proximity_5_to_10cm": 0.05 < distance <= 0.10,
                "near_contact_5cm": valid_surface_separation and distance <= 0.05,
                "proximity_10cm": valid_surface_separation and distance <= 0.10,
                "penetration_depth_m": max(0.0, -distance),
                "left_stance": left_stance,
                "right_stance": right_stance,
                "left_foot_xy_displacement_m": left_motion,
                "right_foot_xy_displacement_m": right_motion,
                "foot_sticking_violation": (
                    (left_stance and left_motion > np.sqrt(2.0) * foot_tolerance_m + 1e-4)
                    or (right_stance and right_motion > np.sqrt(2.0) * foot_tolerance_m + 1e-4)
                ),
            }
        )
    distances = np.asarray([row["minimum_robot_object_surface_distance_m"] for row in rows])
    tolerance = CONTACT_PENETRATION_TOLERANCE_M
    valid = distances >= -tolerance
    summary = {
        "frames": len(rows),
        "robot_collision_geom_count": len(robot_geoms),
        "object_collision_geom_count": len(object_geoms),
        "distance_backend": "mujoco.mj_geomDistance over collision surfaces",
        "distance_scope": "minimum over all enabled robot-object collision pairs",
        "task_intended_contact_pair_asserted": False,
        "strict_contact_2cm_frame_rate": float(np.mean(valid & (distances <= 0.02))),
        "near_contact_2_to_5cm_frame_rate": float(
            np.mean((distances > 0.02) & (distances <= 0.05))
        ),
        "proximity_5_to_10cm_frame_rate": float(
            np.mean((distances > 0.05) & (distances <= 0.10))
        ),
        "near_contact_5cm_frame_rate": float(np.mean(valid & (distances <= 0.05))),
        "proximity_10cm_frame_rate": float(np.mean(valid & (distances <= 0.10))),
        "penetration_any_frame_rate": float(np.mean(distances < 0.0)),
        "penetration_frame_rate": float(np.mean(distances < -tolerance)),
        "penetration_primary_threshold_m": tolerance,
        "contact_penetration_tolerance_m": tolerance,
        "metric_revision": INTERACTION_METRIC_REVISION,
        "maximum_penetration_depth_m": float(max(0.0, -distances.min())),
        "foot_sticking_violation_frame_rate": float(
            np.mean([row["foot_sticking_violation"] for row in rows])
        ),
    }
    return rows, summary


def _interaction_output_paths(run_dir: Path, case: str) -> dict[str, Path]:
    spec = INTERACTION_CASES[case]
    return {
        "raw_qpos": (
            run_dir
            / "native_output"
            / f"{spec['task_name']}_original.npz"
        ),
        "expanded_scene": run_dir / "expanded_scene.xml",
        "per_frame_metrics": run_dir / "per_frame_metrics.csv",
        "intended_contact_mapping": run_dir / "intended_contact_mapping.json",
        "intended_contact_source_contract": (
            run_dir / "intended_contact_source_contract.npz"
        ),
        "intended_contact_per_semantic": (
            run_dir / "intended_contact_per_semantic.csv"
        ),
    }


def _write_interaction_output_ledger(
    run_dir: Path,
    case: str,
    variant: str,
    provenance_ledger_sha256: str,
) -> tuple[Path, str, dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for name, path in _interaction_output_paths(run_dir, case).items():
        if not path.is_file():
            raise FileNotFoundError(f"Interaction output artifact is missing: {path}")
        artifacts.append(
            {
                "name": name,
                "path": path.relative_to(run_dir).as_posix(),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    payload = {
        "schema_version": INTERACTION_OUTPUT_LEDGER_SCHEMA_VERSION,
        "case": case,
        "variant": variant,
        "metric_revision": INTERACTION_METRIC_REVISION,
        "provenance_ledger_sha256": provenance_ledger_sha256,
        "artifacts": artifacts,
        "artifact_count": len(artifacts),
        "artifact_bundle_sha256": hashlib.sha256(
            json.dumps(artifacts, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest(),
    }
    ledger_path = run_dir / "output_artifacts.json"
    atomic_write_json(ledger_path, payload)
    return ledger_path, sha256_file(ledger_path), payload


def _verify_interaction_output_ledger(
    run_dir: Path,
    case: str,
    variant: str,
    provenance_ledger_sha256: str,
    summary: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    ledger_path = run_dir / "output_artifacts.json"
    if not ledger_path.is_file():
        raise ValueError("Interaction attempt lacks its output-artifact ledger")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger_hash = sha256_file(ledger_path)
    if (
        ledger.get("schema_version") != INTERACTION_OUTPUT_LEDGER_SCHEMA_VERSION
        or ledger.get("case") != case
        or ledger.get("variant") != variant
        or ledger.get("metric_revision") != INTERACTION_METRIC_REVISION
        or ledger.get("provenance_ledger_sha256") != provenance_ledger_sha256
        or summary.get("output_artifact_ledger_sha256") != ledger_hash
        or summary.get("provenance_ledger_sha256") != provenance_ledger_sha256
    ):
        raise ValueError("Interaction output ledger is stale or not bound to its run")
    artifacts = ledger.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != int(
        ledger.get("artifact_count", -1)
    ):
        raise ValueError("Interaction output ledger artifact list is malformed")
    expected = _interaction_output_paths(run_dir, case)
    by_name = {str(record.get("name", "")): record for record in artifacts}
    if set(by_name) != set(expected) or len(by_name) != len(artifacts):
        raise ValueError("Interaction output ledger has missing or unexpected artifacts")
    if [str(record.get("name", "")) for record in artifacts] != list(expected):
        raise ValueError("Interaction output ledger artifact order is not deterministic")
    expected_bundle = hashlib.sha256(
        json.dumps(artifacts, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if ledger.get("artifact_bundle_sha256") != expected_bundle:
        raise ValueError("Interaction output artifact-bundle hash is stale")
    summary_hash_keys = {
        "raw_qpos": "qpos_sha256",
        "expanded_scene": "expanded_scene_sha256",
        "per_frame_metrics": "per_frame_metrics_sha256",
        "intended_contact_mapping": "intended_contact_mapping_artifact_sha256",
        "intended_contact_source_contract": (
            "intended_contact_source_contract_sha256"
        ),
        "intended_contact_per_semantic": (
            "intended_contact_per_semantic_sha256"
        ),
    }
    for name, path in expected.items():
        record = by_name[name]
        if record.get("path") != path.relative_to(run_dir).as_posix():
            raise ValueError(f"Interaction output path changed for {name}")
        if (
            not path.is_file()
            or int(record.get("bytes", -1)) != path.stat().st_size
            or record.get("sha256") != sha256_file(path)
            or summary.get(summary_hash_keys[name]) != record.get("sha256")
        ):
            raise ValueError(f"Interaction output artifact changed: {name}")
    return ledger, ledger_hash


def verify_interaction_attempt(
    root: Path,
    case: str,
    variant: str,
    manifest: RunManifest,
) -> tuple[Path, dict[str, Any]]:
    """Recompute every input and output hash for reuse/publication/validation."""

    root = root.resolve()
    if manifest.status != RunStatus.SUCCEEDED or manifest.output_path is None:
        raise ValueError(f"Interaction {case}/{variant} is not a succeeded run")
    summary_path = Path(str(manifest.output_path))
    if not summary_path.is_absolute():
        summary_path = root / summary_path
    if (
        not summary_path.is_file()
        or manifest.output_sha256 != sha256_file(summary_path)
    ):
        raise ValueError(f"Interaction summary changed for {case}/{variant}")
    run_dir = summary_path.parent
    provenance, provenance_hash = _verify_interaction_provenance_attempt(
        root, case, variant, run_dir
    )
    if manifest.config_sha256 != provenance_hash:
        raise ValueError(f"Interaction config/provenance hash is stale for {case}/{variant}")
    if provenance_hash != _interaction_config_sha256(root, case, variant):
        raise ValueError(f"Current interaction inputs changed for {case}/{variant}")
    primary = next(
        record
        for record in provenance["files"]
        if "primary_human_motion" in record["roles"]
    )
    if manifest.source_sha256 != primary["sha256"]:
        raise ValueError(f"Interaction manifest source hash is stale for {case}/{variant}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("case") != case
        or summary.get("variant") != variant
        or summary.get("provenance_ledger_sha256") != provenance_hash
        or summary.get("provenance_file_bundle_sha256")
        != provenance["file_bundle_sha256"]
        or summary.get("holosoma_exact_worktree_diff_sha256")
        != provenance["holosoma"]["exact_worktree_diff_sha256"]
        or summary.get("holosoma_tracked_patch_sha256")
        != provenance["holosoma"]["tracked_patch_sha256"]
        or summary.get("input_sha256") != primary["sha256"]
    ):
        raise ValueError(f"Interaction summary provenance is stale for {case}/{variant}")
    _verify_interaction_output_ledger(
        run_dir, case, variant, provenance_hash, summary
    )
    return summary_path, summary


def verify_interaction_ablation_pair(root: Path, case: str) -> None:
    """Prove that Full/No-Hard differ only in their two frozen flags."""

    payloads: dict[str, dict[str, Any]] = {}
    for variant in ("full", "no-hard"):
        manifest_path = (
            root
            / "runs"
            / "interaction_manifests"
            / f"interaction__{case}__{variant}.json"
        )
        manifest = RunManifest.load(manifest_path)
        summary_path, _ = verify_interaction_attempt(root, case, variant, manifest)
        payloads[variant] = json.loads(
            (summary_path.parent / "provenance_ledger.json").read_text(
                encoding="utf-8"
            )
        )
    expected_flags = {
        "full": {
            "activate_obj_non_penetration": True,
            "activate_foot_sticking": True,
            "activate_joint_limits": True,
        },
        "no-hard": {
            "activate_obj_non_penetration": False,
            "activate_foot_sticking": False,
            "activate_joint_limits": True,
        },
    }
    for variant, payload in payloads.items():
        if payload.get("ablation_flags") != expected_flags[variant]:
            raise ValueError(f"Unexpected interaction ablation flags for {case}/{variant}")
    normalized: dict[str, dict[str, Any]] = {}
    for variant, payload in payloads.items():
        value = json.loads(json.dumps(payload))
        value["variant"] = "<full-vs-no-hard>"
        value["ablation_flags"] = {
            "activate_obj_non_penetration": "<ablation>",
            "activate_foot_sticking": "<ablation>",
            "activate_joint_limits": True,
        }
        normalized[variant] = value
    if normalized["full"] != normalized["no-hard"]:
        raise ValueError(
            f"Interaction Full/No-Hard provenance differs beyond frozen flags for {case}"
        )


def run_interaction_native(
    repo_root: str | Path,
    case: str,
    variant: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    if case not in INTERACTION_CASES or variant not in {"full", "no-hard"}:
        raise ValueError("Unknown interaction case or variant")
    root = Path(repo_root).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    provenance, provenance_hash = _verify_interaction_provenance_attempt(
        root, case, variant, output
    )
    checkout = root / "external" / "holosoma"
    package_root = checkout / "src" / "holosoma_retargeting"
    module_root = str(package_root)
    sys.path.insert(0, module_root)
    original_cwd = Path.cwd()
    try:
        import mujoco

        from holosoma_retargeting.config_types.retargeter import RetargeterConfig
        from holosoma_retargeting.config_types.retargeting import RetargetingConfig
        from holosoma_retargeting.config_types.robot import RobotConfig
        from holosoma_retargeting.examples import robot_retarget
        from holosoma_retargeting.src.interaction_mesh_retargeter import InteractionMeshRetargeter

        spec = INTERACTION_CASES[case]
        data_path = package_root / "holosoma_retargeting" / spec["relative_data_path"]
        native_output = output / "native_output"
        native_output.mkdir(parents=True, exist_ok=True)
        instances: list[Any] = []
        frame_times: list[float] = []
        steady_times: list[float] = []

        class CapturingRetargeter(InteractionMeshRetargeter):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._rtcmp_foot_sticking: list[dict[str, bool]] = []
                self._rtcmp_source_contract: dict[str, np.ndarray] = {}
                instances.append(self)

            def iterate(self, *args, **kwargs):
                start = time.perf_counter()
                result = super().iterate(*args, **kwargs)
                frame_times.append(time.perf_counter() - start)
                return result

            def retarget_motion(self, *args, **kwargs):
                required = {
                    "human_joint_motions",
                    "object_poses",
                    "object_points_local_demo",
                    "object_points_local",
                    "foot_sticking_sequences",
                }
                missing = required - set(kwargs)
                if missing:
                    raise RuntimeError(
                        "Cannot capture official interaction source contract; missing "
                        + ", ".join(sorted(missing))
                    )
                self._rtcmp_foot_sticking = kwargs["foot_sticking_sequences"]
                self._rtcmp_source_contract = {
                    name: np.asarray(kwargs[name]).copy()
                    for name in required - {"foot_sticking_sequences"}
                }
                start = time.perf_counter()
                result = super().retarget_motion(*args, **kwargs)
                steady_times.append(time.perf_counter() - start)
                return result

        original_symbol = robot_retarget.InteractionMeshRetargeter
        robot_retarget.InteractionMeshRetargeter = CapturingRetargeter
        enabled = variant == "full"
        retargeter_cfg = RetargeterConfig(
            activate_obj_non_penetration=enabled,
            activate_foot_sticking=enabled,
            activate_joint_limits=True,
        )
        robot_cfg = RobotConfig(
            robot_type="g1",
            robot_urdf_file=(
                "models/g1/g1_29dof_spherehand.urdf" if case == "climb" else None
            ),
        )
        cfg = RetargetingConfig(
            task_type=spec["task_type"],
            robot="g1",
            data_format=spec["data_format"],
            task_name=spec["task_name"],
            data_path=data_path,
            save_dir=native_output,
            robot_config=robot_cfg,
            retargeter=retargeter_cfg,
        )
        np.random.seed(0)
        os.chdir(package_root / "holosoma_retargeting")
        start = time.perf_counter()
        try:
            robot_retarget.main(cfg)
        finally:
            robot_retarget.InteractionMeshRetargeter = original_symbol
        wall_time = time.perf_counter() - start
        if len(instances) != 1:
            raise RuntimeError("Expected exactly one captured Holosoma retargeter")
        instance = instances[0]
        if not instance._rtcmp_source_contract:
            raise RuntimeError("Official interaction source contract was not captured")
        result_path = native_output / f"{spec['task_name']}_original.npz"
        with np.load(result_path, allow_pickle=False) as data:
            qpos = np.asarray(data["qpos"], dtype=np.float64)
            fps = float(data["fps"])
        if len(frame_times) != len(qpos):
            raise RuntimeError("Interaction timing count does not match output frames")
        if len(steady_times) != 1:
            raise RuntimeError("Interaction timing adapter did not observe one sequence loop")
        scene_path = output / "expanded_scene.xml"
        mujoco.mj_saveLastXML(str(scene_path), instance.robot_model)
        object_token = "largebox" if case == "box" else "multi_boxes"
        rows, metrics = mesh_surface_diagnostics(
            instance.robot_model,
            qpos,
            object_token,
            list(instance.foot_links),
            instance._rtcmp_foot_sticking,
            instance.foot_sticking_tolerance,
        )
        source_mesh_path = (
            package_root
            / "holosoma_retargeting"
            / "models"
            / "largebox"
            / "largebox.obj"
            if case == "box"
            else data_path / spec["task_name"] / "multi_boxes.obj"
        )
        contact_rows, contact_summary, mapping_payload, contract_arrays = (
            intended_contact_diagnostics(
                instance.robot_model,
                qpos,
                case,
                object_token,
                list(instance.demo_joints),
                dict(instance.laplacian_match_links),
                instance._rtcmp_source_contract["human_joint_motions"],
                instance._rtcmp_source_contract["object_poses"],
                instance._rtcmp_source_contract["object_points_local_demo"],
                instance._rtcmp_source_contract["object_points_local"],
                source_mesh_path,
            )
        )
        per_frame_contacts: dict[int, list[dict[str, Any]]] = {}
        for contact_row in contact_rows:
            per_frame_contacts.setdefault(int(contact_row["frame"]), []).append(
                contact_row
            )
        for row in rows:
            frame_contacts = per_frame_contacts.get(int(row["frame"]), [])
            if len(frame_contacts) != int(contact_summary["intended_contact_semantic_count"]):
                raise RuntimeError("Intended-contact rows are not complete for every frame")
            row.update(
                {
                    "intended_source_contact_count_2cm": sum(
                        bool(value["source_contact_2cm"]) for value in frame_contacts
                    ),
                    "intended_source_contact_count_5cm": sum(
                        bool(value["source_contact_5cm"]) for value in frame_contacts
                    ),
                    "intended_source_contact_count_10cm": sum(
                        bool(value["source_contact_10cm"]) for value in frame_contacts
                    ),
                    "intended_min_source_surface_distance_m": min(
                        float(value["source_absolute_surface_distance_m"])
                        for value in frame_contacts
                    ),
                    "intended_min_mapped_robot_signed_surface_distance_m": min(
                        float(value["mapped_robot_object_signed_surface_distance_m"])
                        for value in frame_contacts
                    ),
                    "intended_max_mapped_robot_penetration_depth_m": max(
                        float(value["mapped_robot_penetration_depth_m"])
                        for value in frame_contacts
                    ),
                }
            )
        mapping_path = output / "intended_contact_mapping.json"
        atomic_write_json(mapping_path, mapping_payload)
        contract_path = output / "intended_contact_source_contract.npz"
        np.savez_compressed(contract_path, **contract_arrays)
        intended_csv_path = output / "intended_contact_per_semantic.csv"
        with intended_csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(contact_rows[0]))
            writer.writeheader()
            writer.writerows(contact_rows)
        contact_summary.update(
            {
                "intended_contact_mapping_artifact": mapping_path.name,
                "intended_contact_mapping_artifact_sha256": sha256_file(mapping_path),
                "intended_contact_source_contract_artifact": contract_path.name,
                "intended_contact_source_contract_sha256": sha256_file(contract_path),
                "intended_contact_source_contract_canonical_sha256": (
                    _array_contract_sha256(contract_arrays)
                ),
                "intended_contact_per_semantic_artifact": intended_csv_path.name,
                "intended_contact_per_semantic_sha256": sha256_file(
                    intended_csv_path
                ),
            }
        )
        metrics.update(contact_summary)
        constraint_trace = list(getattr(instance, "_rtcmp_constraint_audit", []))
        if not constraint_trace:
            raise RuntimeError("Holosoma did not emit the runtime constraint-graph audit")
        component_names = (
            "laplacian_equality",
            "foot_sticking_and_lock",
            "object_non_penetration",
            "self_collision",
            "joint_limits",
            "step_size",
        )
        trace_payload = json.dumps(
            constraint_trace,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        metrics["constraint_graph_audit"] = {
            "solve_calls": len(constraint_trace),
            "trace_sha256": hashlib.sha256(trace_payload).hexdigest(),
            "component_count_min": {
                name: int(
                    min(
                        entry["component_counts"].get(name, 0)
                        for entry in constraint_trace
                    )
                )
                for name in component_names
            },
            "component_count_max": {
                name: int(
                    max(
                        entry["component_counts"].get(name, 0)
                        for entry in constraint_trace
                    )
                )
                for name in component_names
            },
            "component_nonzero_solve_calls": {
                name: int(
                    sum(
                        entry["component_counts"].get(name, 0) > 0
                        for entry in constraint_trace
                    )
                )
                for name in component_names
            },
            "unique_graph_signatures": sorted(
                {
                    json.dumps(
                        {
                            "component_counts": entry["component_counts"],
                            "constraint_types": entry["constraint_types"],
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    for entry in constraint_trace
                }
            ),
            "runtime_graph_not_source_ast_only": True,
        }
        csv_path = output / "per_frame_metrics.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        input_path = (
            data_path / f"{spec['task_name']}.pt"
            if case == "box"
            else next((data_path / spec["task_name"]).glob("*.npy"))
        )
        mesh_paths = (
            [package_root / "holosoma_retargeting" / "models" / "largebox" / "largebox.obj"]
            if case == "box"
            else sorted((data_path / spec["task_name"] / "box_models").glob("*.obj"))
        )
        metrics.update(
            {
                "case": case,
                "task_name": spec["task_name"],
                "variant": variant,
                "status": "succeeded",
                "fps": fps,
                "wall_time_s": wall_time,
                "initialization_and_preprocess_time_s": max(0.0, wall_time - steady_times[0]),
                "steady_end_to_end_time_s": steady_times[0],
                "native_frame_times_s": frame_times,
                "native_median_frame_s": float(np.median(frame_times)),
                "native_core_rtf": float(sum(frame_times) / (len(qpos) / fps)),
                "end_to_end_rtf": float(steady_times[0] / (len(qpos) / fps)),
                "activate_obj_non_penetration": enabled,
                "activate_foot_sticking": enabled,
                "activate_joint_limits": True,
                "input_sha256": sha256_file(input_path),
                "object_mesh_sha256": _aggregate_sha256(mesh_paths),
                "expanded_scene_sha256": sha256_file(scene_path),
                "per_frame_metrics_sha256": sha256_file(csv_path),
                "qpos_sha256": sha256_file(result_path),
                "qpos_width": int(qpos.shape[1]),
                "upstream_commit": HOLOSOMA_INTERACTION_COMMIT,
                "patch": HOLOSOMA_INTERACTION_PATCH.as_posix(),
                "provenance_ledger": "provenance_ledger.json",
                "provenance_ledger_sha256": provenance_hash,
                "provenance_file_bundle_sha256": provenance[
                    "file_bundle_sha256"
                ],
                "holosoma_exact_worktree_diff_sha256": provenance[
                    "holosoma"
                ]["exact_worktree_diff_sha256"],
                "holosoma_tracked_patch_sha256": provenance["holosoma"][
                    "tracked_patch_sha256"
                ],
                "claim_scope": "two-case case-study evidence only",
            }
        )
        current_provenance, current_hash = _verify_interaction_provenance_attempt(
            root, case, variant, output
        )
        if current_provenance != provenance or current_hash != provenance_hash:
            raise RuntimeError("Interaction inputs changed while the solver was running")
        output_ledger_path, output_ledger_hash, output_ledger = (
            _write_interaction_output_ledger(
                output, case, variant, provenance_hash
            )
        )
        metrics.update(
            {
                "output_artifact_ledger": output_ledger_path.name,
                "output_artifact_ledger_sha256": output_ledger_hash,
                "output_artifact_bundle_sha256": output_ledger[
                    "artifact_bundle_sha256"
                ],
            }
        )
        atomic_write_json(output / "summary.json", metrics)
        return metrics
    finally:
        os.chdir(original_cwd)
        if sys.path[0] == module_root:
            sys.path.pop(0)


def run_interaction(
    case: str, variant: str, repo_root: str | Path = "."
) -> RunManifest:
    root = Path(repo_root).resolve()
    config_hash = _interaction_config_sha256(root, case, variant)
    run_id = f"interaction__{case}__{variant}"
    run_dir = root / "runs" / "interaction_v4" / case / variant
    manifest_path = root / "runs" / "interaction_manifests" / f"{run_id}.json"
    summary_path = run_dir / "summary.json"
    if manifest_path.exists():
        existing = RunManifest.load(manifest_path)
        try:
            _, existing_summary = verify_interaction_attempt(
                root, case, variant, existing
            )
            if (
                existing_summary.get("metric_revision")
                == INTERACTION_METRIC_REVISION
                and existing_summary.get("constraint_graph_audit", {}).get(
                    "runtime_graph_not_source_ast_only"
                )
                is True
                and existing_summary.get("intended_contact_pair_asserted") is True
            ):
                return existing
        except (FileNotFoundError, KeyError, StopIteration, ValueError):
            pass
        attempt = datetime.now(timezone.utc).strftime("attempt_%Y%m%dT%H%M%SZ")
        archive = manifest_path.with_name(f"{manifest_path.stem}__{attempt}.json")
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest_path, archive)
        run_dir = run_dir / attempt
        summary_path = run_dir / "summary.json"
    ledger_path, written_config_hash = _write_interaction_provenance_attempt(
        root, case, variant, run_dir
    )
    if written_config_hash != config_hash:
        raise RuntimeError("Interaction config hash differs from written provenance ledger")
    provenance = json.loads(ledger_path.read_text(encoding="utf-8"))
    primary = next(
        record
        for record in provenance["files"]
        if "primary_human_motion" in record["roles"]
    )
    python = _conda_python("hsretargeting")
    command = [
        str(python),
        "-m",
        "retargeting_comparison.interaction_worker",
        "--repo-root",
        str(root),
        "--case",
        case,
        "--variant",
        variant,
        "--output-dir",
        str(run_dir),
    ]
    logs = run_dir / "logs"
    manifest = RunManifest(
        run_id=run_id,
        method=f"interaction-{case}-{variant}",
        status=RunStatus.RUNNING,
        command=command,
        environment="conda:hsretargeting",
        repo_commit=_git(root, "rev-parse", "HEAD"),
        config_sha256=config_hash,
        device="CPU; threads=1",
        started_at=utc_now(),
        stdout_log=str(logs / "stdout.log"),
        stderr_log=str(logs / "stderr.log"),
        output_path=str(summary_path),
        source_sha256=str(primary["sha256"]),
    )
    manifest.save(manifest_path)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(root / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    logs.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with (logs / "stdout.log").open("w", encoding="utf-8") as stdout, (
        logs / "stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        result = subprocess.run(command, cwd=root, env=env, stdout=stdout, stderr=stderr)
    manifest.wall_time_s = time.perf_counter() - start
    manifest.finished_at = utc_now()
    manifest.exit_code = result.returncode
    if result.returncode == 0 and summary_path.is_file():
        manifest.output_sha256 = sha256_file(summary_path)
        manifest.status = RunStatus.SUCCEEDED
        try:
            verify_interaction_attempt(root, case, variant, manifest)
        except (FileNotFoundError, KeyError, StopIteration, ValueError) as error:
            manifest.status = RunStatus.FAILED
            manifest.message = f"Interaction provenance validation failed: {error}"
    else:
        manifest.status = RunStatus.FAILED
        manifest.message = "Interaction worker failed; inspect stderr log"
    manifest.save(manifest_path)
    return manifest
