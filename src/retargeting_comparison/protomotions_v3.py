"""Isolated adapter for ProtoMotions v3's official modified-PyRoki G1 path.

The upstream solver optimizes an entire fixed-length trajectory and emits
``base_frame_pos``, ``base_frame_wxyz`` and ``joint_angles``.  This module does
not reimplement or tune that solver.  It validates the native input and robot
asset, launches the native worker in a separate environment, and converts the
result to the benchmark's canonical ``float64[T, 36]`` contract.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree as ET

import numpy as np

from .constants import G1_JOINT_NAMES, MIN_COMPLETION_RATIO
from .io_utils import atomic_write_json, sha256_file
from .schemas import CanonicalG1


PROTOMOTIONS_V3_COMMIT = "49fe5ad69de67ebbc07ea2b25d41b0f622c15c3c"
PYROKI_COMMIT = "388e43e1fc0d0ee382968d3dd72970fd62a0450c"
JAXLS_COMMIT = "cb89259f87402872485dc83706dc28234f898a82"
TARGET_RAW_FRAMES = 600
N_RETARGET_KEYPOINTS = 15
N_AUX_KEYPOINTS = 3
OFFICIAL_MAX_ITERATIONS = 800


@dataclass(frozen=True)
class UrdfJoint:
    name: str
    joint_type: str
    parent: str
    child: str
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    axis: tuple[float, float, float]
    lower: float | None
    upper: float | None


def _vector(text: str | None, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if text is None:
        return default
    values = tuple(float(value) for value in text.split())
    if len(values) != 3:
        raise ValueError(f"Expected a three-vector, got {text!r}")
    return values  # type: ignore[return-value]


def read_urdf_joints(path: str | Path) -> tuple[UrdfJoint, ...]:
    """Read all URDF joints without importing a simulator-specific parser."""
    root = ET.parse(path).getroot()
    result: list[UrdfJoint] = []
    for element in root.findall("joint"):
        origin = element.find("origin")
        axis = element.find("axis")
        limit = element.find("limit")
        parent = element.find("parent")
        child = element.find("child")
        if parent is None or child is None:
            raise ValueError(f"URDF joint {element.get('name')!r} is missing a link")
        joint_type = str(element.get("type"))
        lower = upper = None
        if joint_type in {"revolute", "prismatic"}:
            if limit is None or limit.get("lower") is None or limit.get("upper") is None:
                raise ValueError(f"Limited joint {element.get('name')!r} is missing limits")
            lower, upper = float(limit.get("lower")), float(limit.get("upper"))
        result.append(
            UrdfJoint(
                name=str(element.get("name")),
                joint_type=joint_type,
                parent=str(parent.get("link")),
                child=str(child.get("link")),
                xyz=_vector(None if origin is None else origin.get("xyz"), (0.0, 0.0, 0.0)),
                rpy=_vector(None if origin is None else origin.get("rpy"), (0.0, 0.0, 0.0)),
                axis=_vector(None if axis is None else axis.get("xyz"), (1.0, 0.0, 0.0)),
                lower=lower,
                upper=upper,
            )
        )
    return tuple(result)


def actuated_urdf_joints(path: str | Path) -> tuple[UrdfJoint, ...]:
    return tuple(
        joint
        for joint in read_urdf_joints(path)
        if joint.joint_type in {"revolute", "continuous", "prismatic"}
    )


def _rotation_rpy(rpy: Iterable[float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.asarray([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.asarray([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.asarray([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def _rotation_axis(axis: Iterable[float], angle: float) -> np.ndarray:
    value = np.asarray(tuple(axis), dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm < 1e-12:
        raise ValueError("URDF joint has a zero rotation axis")
    x, y, z = value / norm
    cross = np.asarray([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
    return np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)


def _transform(rotation: np.ndarray, translation: Iterable[float]) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = rotation
    value[:3, 3] = tuple(translation)
    return value


def urdf_forward_kinematics(
    path: str | Path, joint_values: dict[str, float]
) -> dict[str, np.ndarray]:
    """Compute link poses for a fixed/revolute URDF tree.

    This independent implementation is deliberately small: it is used only to
    prove that two frozen assets assign the same generalized coordinates to the
    same link frames.  It is not part of the retargeting solver.
    """
    root = ET.parse(path).getroot()
    links = {str(link.get("name")) for link in root.findall("link")}
    joints = read_urdf_joints(path)
    children = {joint.child for joint in joints}
    bases = sorted(links - children)
    if len(bases) != 1:
        raise ValueError(f"Expected one URDF base link, found {bases}")
    poses = {bases[0]: np.eye(4, dtype=np.float64)}
    remaining = list(joints)
    while remaining:
        progressed = False
        for joint in remaining[:]:
            if joint.parent not in poses:
                continue
            local = _transform(_rotation_rpy(joint.rpy), joint.xyz)
            position = float(joint_values.get(joint.name, 0.0))
            if joint.joint_type in {"revolute", "continuous"}:
                local = local @ _transform(_rotation_axis(joint.axis, position), (0, 0, 0))
            elif joint.joint_type == "prismatic":
                local = local @ _transform(np.eye(3), np.asarray(joint.axis) * position)
            elif joint.joint_type != "fixed":
                raise ValueError(f"Unsupported URDF joint type: {joint.joint_type}")
            poses[joint.child] = poses[joint.parent] @ local
            remaining.remove(joint)
            progressed = True
        if not progressed:
            unresolved = [joint.name for joint in remaining]
            raise ValueError(f"URDF joint graph is disconnected or cyclic: {unresolved}")
    return poses


def _joint_difference(
    left: UrdfJoint, right: UrdfJoint, *, atol: float = 1e-12
) -> dict[str, bool]:
    return {
        "origin": not (
            np.allclose(left.xyz, right.xyz, atol=atol, rtol=0.0)
            and np.allclose(left.rpy, right.rpy, atol=atol, rtol=0.0)
        ),
        "axis": not np.allclose(left.axis, right.axis, atol=atol, rtol=0.0),
        "limits": not (
            left.lower == right.lower and left.upper == right.upper
        ),
    }


def compare_urdf_contracts(
    proto_urdf: str | Path,
    reference_urdf: str | Path,
    *,
    random_seed: int = 20260722,
) -> dict[str, Any]:
    proto = actuated_urdf_joints(proto_urdf)
    reference = actuated_urdf_joints(reference_urdf)
    proto_by_name = {joint.name: joint for joint in proto}
    reference_by_name = {joint.name: joint for joint in reference}
    if set(proto_by_name) != set(reference_by_name):
        raise ValueError("G1 URDFs do not expose the same actuated joint set")
    origin_differences: list[str] = []
    axis_differences: list[str] = []
    limit_differences: list[dict[str, Any]] = []
    for name in G1_JOINT_NAMES:
        left, right = proto_by_name[name], reference_by_name[name]
        difference = _joint_difference(left, right)
        if difference["origin"]:
            origin_differences.append(name)
        if difference["axis"]:
            axis_differences.append(name)
        if difference["limits"]:
            limit_differences.append(
                {
                    "joint": name,
                    "proto_lower_rad": left.lower,
                    "proto_upper_rad": left.upper,
                    "reference_lower_rad": right.lower,
                    "reference_upper_rad": right.upper,
                }
            )

    rng = np.random.default_rng(random_seed)
    trials = [np.zeros(len(G1_JOINT_NAMES), dtype=np.float64)]
    for _ in range(3):
        values = []
        for name in G1_JOINT_NAMES:
            left, right = proto_by_name[name], reference_by_name[name]
            lower = max(float(left.lower), float(right.lower))
            upper = min(float(left.upper), float(right.upper))
            values.append(rng.uniform(lower * 0.5, upper * 0.5))
        trials.append(np.asarray(values))
    common_child_links = [proto_by_name[name].child for name in G1_JOINT_NAMES]
    maximum_fk_position_difference = 0.0
    for values in trials:
        config = dict(zip(G1_JOINT_NAMES, values, strict=True))
        proto_fk = urdf_forward_kinematics(proto_urdf, config)
        reference_fk = urdf_forward_kinematics(reference_urdf, config)
        maximum_fk_position_difference = max(
            maximum_fk_position_difference,
            max(
                float(np.linalg.norm(proto_fk[link][:3, 3] - reference_fk[link][:3, 3]))
                for link in common_child_links
            ),
        )
    return {
        "proto_urdf": str(Path(proto_urdf).resolve()),
        "reference_urdf": str(Path(reference_urdf).resolve()),
        "proto_sha256": sha256_file(proto_urdf),
        "reference_sha256": sha256_file(reference_urdf),
        "proto_joint_order": [joint.name for joint in proto],
        "reference_joint_order": [joint.name for joint in reference],
        "canonical_joint_order": list(G1_JOINT_NAMES),
        "proto_order_is_canonical": tuple(joint.name for joint in proto) == G1_JOINT_NAMES,
        "reference_order_is_canonical": tuple(joint.name for joint in reference) == G1_JOINT_NAMES,
        "origin_differences": origin_differences,
        "axis_differences": axis_differences,
        "limit_differences": limit_differences,
        "limit_difference_count": len(limit_differences),
        "fk_trials": len(trials),
        "max_actuated_child_fk_position_difference_m": maximum_fk_position_difference,
    }


def audit_robot_assets(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    proto = root / "external/ProtoMotions/protomotions/data/assets/urdf/for_retargeting/g1.urdf"
    holosoma_native = (
        root
        / "external/holosoma/src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof.urdf"
    )
    evaluator = root / "external/holosoma/src/holosoma/holosoma/data/robots/g1/g1_29dof.urdf"
    native = compare_urdf_contracts(proto, holosoma_native)
    canonical = compare_urdf_contracts(proto, evaluator)
    for audit in (native, canonical):
        audit["proto_urdf"] = str(Path(audit["proto_urdf"]).relative_to(root))
        audit["reference_urdf"] = str(Path(audit["reference_urdf"]).relative_to(root))
    if not native["proto_order_is_canonical"] or not native["reference_order_is_canonical"]:
        raise ValueError("ProtoMotions/Holosoma native G1 joint order is not canonical")
    if native["axis_differences"] or native["origin_differences"]:
        raise ValueError("ProtoMotions and Holosoma native retargeting kinematics differ")
    if native["limit_difference_count"] != 15:
        raise ValueError("Expected exactly 15 ProtoMotions/Holosoma limit differences")
    return {
        "native_retargeting_asset": native,
        "canonical_evaluator_asset": canonical,
        "interpretation": (
            "ProtoMotions and Holosoma's retargeting URDF share the 29-coordinate "
            "kinematics, while ProtoMotions narrows 15 joint limits. The separate "
            "Holosoma evaluator asset also shifts four joint origins; canonical "
            "evaluation therefore uses common geometry rather than claiming native-FK identity."
        ),
    }


def validate_keypoint_input(path: str | Path, expected_frames: int = TARGET_RAW_FRAMES) -> dict[str, Any]:
    """Validate the exact native file consumed by the official v3 script."""
    with Path(path).open("rb") as stream:
        value = np.load(stream, allow_pickle=True)
        if not isinstance(value, np.ndarray) or value.shape != ():
            raise ValueError("ProtoMotions keypoint input must be a scalar numpy mapping")
        mapping = value.item()
    if not isinstance(mapping, dict):
        raise ValueError("ProtoMotions keypoint input is not a mapping")
    required = {
        "positions",
        "orientations",
        "left_foot_contacts",
        "right_foot_contacts",
        "fps",
    }
    if not required.issubset(mapping):
        raise ValueError(f"ProtoMotions keypoint input is missing {sorted(required - set(mapping))}")
    positions = np.asarray(mapping["positions"], dtype=np.float64)
    orientations = np.asarray(mapping["orientations"], dtype=np.float64)
    left = np.asarray(mapping["left_foot_contacts"])
    right = np.asarray(mapping["right_foot_contacts"])
    expected_points = N_RETARGET_KEYPOINTS + N_AUX_KEYPOINTS
    if positions.shape != (expected_frames, expected_points, 3):
        raise ValueError(f"positions must have shape [{expected_frames},{expected_points},3]")
    if orientations.shape != (expected_frames, expected_points, 3, 3):
        raise ValueError(
            f"orientations must have shape [{expected_frames},{expected_points},3,3]"
        )
    if left.shape != (expected_frames, 2) or right.shape != (expected_frames, 2):
        raise ValueError(f"foot contacts must each have shape [{expected_frames},2]")
    if not np.isfinite(positions).all() or not np.isfinite(orientations).all():
        raise ValueError("ProtoMotions keypoint input contains NaN/Inf")
    if not np.isin(left, (0, 1)).all() or not np.isin(right, (0, 1)).all():
        raise ValueError("ProtoMotions foot contacts must be binary before native smoothing")
    orthogonality = orientations @ np.swapaxes(orientations, -1, -2)
    max_rotation_error = float(np.max(np.abs(orthogonality - np.eye(3))))
    max_determinant_error = float(np.max(np.abs(np.linalg.det(orientations) - 1.0)))
    if max_rotation_error > 1e-6 or max_determinant_error > 1e-6:
        raise ValueError("ProtoMotions input orientations are not proper rotations")
    fps = float(mapping["fps"])
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("ProtoMotions input fps must be finite and positive")
    root = positions[:, 0]
    extent = np.ptp(positions[..., 2], axis=1)
    radius = np.linalg.norm(positions - root[:, None, :], axis=-1).max(axis=1)
    median_extent = float(np.median(extent))
    median_radius = float(np.median(radius))
    # This catches the common millimetre/metre error without constraining actor height.
    if not (0.25 <= median_extent <= 3.0 and 0.25 <= median_radius <= 3.0):
        raise ValueError("Keypoint magnitudes are inconsistent with the official metre contract")
    return {
        "path": str(Path(path)),
        "sha256": sha256_file(path),
        "frames": expected_frames,
        "keypoints": expected_points,
        "fps": fps,
        "coordinate_unit": "metre",
        "quaternion_convention": "not applicable; input orientations are 3x3 matrices",
        "median_vertical_extent_m": median_extent,
        "median_root_relative_radius_m": median_radius,
        "max_rotation_orthogonality_error": max_rotation_error,
        "max_rotation_determinant_error": max_determinant_error,
    }


def prepare_scale_variant_keypoints(
    source: str | Path,
    output: str | Path,
    *,
    root_scale_multiplier: float = 1.0,
    local_scale_multiplier: float = 1.0,
) -> dict[str, Any]:
    """Apply the registered root/local sensitivity intervention before v3.

    The official v3 lower/upper axis scales remain untouched and are applied by
    its loader afterwards.  Root displacement is perturbed about frame zero;
    root-relative landmarks are perturbed about each frame's pelvis.  Source
    orientations and fixed contact labels are copied exactly.
    """
    if root_scale_multiplier <= 0.0 or local_scale_multiplier <= 0.0:
        raise ValueError("Scale sensitivity multipliers must be positive")
    with Path(source).open("rb") as stream:
        mapping = np.load(stream, allow_pickle=True).item()
    positions = np.asarray(mapping["positions"], dtype=np.float64)
    roots = positions[:, 0]
    anchor = roots[0]
    scaled_roots = anchor + (roots - anchor) * root_scale_multiplier
    scaled_positions = scaled_roots[:, None, :] + (
        positions - roots[:, None, :]
    ) * local_scale_multiplier
    value = dict(mapping)
    value["positions"] = scaled_positions
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".npy", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.save(temporary, value, allow_pickle=True)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    audit = validate_keypoint_input(destination, expected_frames=len(positions))
    audit.update(
        {
            "experiment_role": "native_pipeline_scale_response",
            "root_scale_multiplier": float(root_scale_multiplier),
            "local_scale_multiplier": float(local_scale_multiplier),
            "root_anchor": "frame_zero_pelvis",
            "official_native_axis_scale_preserved": True,
            "orientations_unchanged": bool(
                np.array_equal(value["orientations"], mapping["orientations"])
            ),
            "contact_labels_unchanged": bool(
                np.array_equal(value["left_foot_contacts"], mapping["left_foot_contacts"])
                and np.array_equal(
                    value["right_foot_contacts"], mapping["right_foot_contacts"]
                )
            ),
        }
    )
    return audit


def solver_log_diagnostics(
    path: str | Path, *, configured_max_iterations: int = OFFICIAL_MAX_ITERATIONS
) -> dict[str, Any]:
    """Extract iteration evidence without changing the official JAXLS solve.

    A formal timing process contains four solver calls (one warm-up and three
    measured), each of which restarts its iteration counter at zero.  Recording
    every segment prevents a returned trajectory from being silently described
    as converged when the configured ceiling was reached instead.
    """
    if configured_max_iterations < 1:
        raise ValueError("configured_max_iterations must be positive")
    step_pattern = re.compile(
        r"step #(\d+): cost=([+\-0-9.eE]+)"
    )
    termination_pattern = re.compile(
        r"Terminated @ iteration #(\d+): cost=([+\-0-9.eE]+) "
        r"criteria=\[([^]]+)\], term_deltas=([^\s]+)"
    )
    segments: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        step_match = step_pattern.search(line)
        termination_match = termination_pattern.search(line)
        if termination_match is not None:
            if not segments:
                raise ValueError("JAXLS termination record precedes any logged iteration")
            criteria = [int(value) for value in termination_match.group(3).split()]
            deltas = [float(value) for value in termination_match.group(4).split(",")]
            if len(criteria) != 3 or len(deltas) != 3:
                raise ValueError("JAXLS termination record has an unexpected schema")
            segments[-1].update(
                {
                    "termination_observed": True,
                    "termination_iteration": int(termination_match.group(1)),
                    "termination_cost": float(termination_match.group(2)),
                    "termination_criteria": {
                        "relative_cost": bool(criteria[0]),
                        "gradient": bool(criteria[1]),
                        "parameters": bool(criteria[2]),
                    },
                    "termination_deltas": {
                        "relative_cost": deltas[0],
                        "gradient": deltas[1],
                        "parameters": deltas[2],
                    },
                }
            )
            continue
        if step_match is None:
            continue
        iteration = int(step_match.group(1))
        cost = float(step_match.group(2))
        if iteration == 0 or not segments:
            segments.append(
                {
                    "first_iteration": iteration,
                    "last_iteration": iteration,
                    "first_cost": cost,
                    "final_logged_cost": cost,
                }
            )
        else:
            segment = segments[-1]
            if iteration <= int(segment["last_iteration"]):
                raise ValueError("JAXLS iteration counter regressed without restarting at zero")
            segment["last_iteration"] = iteration
            segment["final_logged_cost"] = cost
    for segment in segments:
        last = int(segment["last_iteration"])
        termination_iteration = int(segment.get("termination_iteration", last + 1))
        segment.setdefault("termination_observed", False)
        segment["logged_iteration_count"] = last + 1
        segment["solver_iteration_count"] = termination_iteration
        segment["final_cost"] = float(
            segment.get("termination_cost", segment["final_logged_cost"])
        )
        segment["reached_configured_iteration_ceiling"] = (
            termination_iteration >= configured_max_iterations
        )
    return {
        "configured_max_iterations": configured_max_iterations,
        "solver_call_count": len(segments),
        "solver_calls": segments,
        "all_calls_have_termination_record": bool(segments)
        and all(segment["termination_observed"] for segment in segments),
        "any_configured_iteration_ceiling_reached": any(
            segment["reached_configured_iteration_ceiling"] for segment in segments
        ),
    }


def _git_commit(path: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _python_probe(python: Path, env: dict[str, str]) -> dict[str, Any]:
    probe = r'''
import importlib.metadata as metadata
import json
import platform
import sys
modules = ["jax", "jaxlib", "jaxlie", "jaxls", "jax_dataclasses", "yourdfpy", "numpy", "pyroki", "protomotions"]
values = {"python": platform.python_version(), "modules": {}}
for name in modules:
    try:
        module = __import__(name)
        distribution = "jax-dataclasses" if name == "jax_dataclasses" else name
        try:
            version = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            version = getattr(module, "__version__", "unknown")
        entry = {"available": True, "version": version}
        try:
            direct_url = metadata.distribution(distribution).read_text("direct_url.json")
            if direct_url:
                entry["direct_url"] = json.loads(direct_url)
        except metadata.PackageNotFoundError:
            pass
        values["modules"][name] = entry
    except Exception as error:
        values["modules"][name] = {"available": False, "error": f"{type(error).__name__}: {error}"}
for name in ["robot_descriptions", "lzfse"]:
    try:
        __import__(name)
        values["modules"][name] = {"available": True}
    except Exception as error:
        values["modules"][name] = {"available": False, "error": f"{type(error).__name__}: {error}"}
try:
    import jax
    values["jax_devices"] = [str(device) for device in jax.devices()]
except Exception as error:
    values["jax_device_error"] = f"{type(error).__name__}: {error}"
print("RTCMP_PROBE=" + json.dumps(values, sort_keys=True))
'''
    result = subprocess.run(
        [str(python), "-c", probe],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    marker = "RTCMP_PROBE="
    line = next((item for item in result.stdout.splitlines() if item.startswith(marker)), None)
    if line is None:
        return {
            "returncode": result.returncode,
            "probe_error": (result.stderr or result.stdout)[-4000:],
        }
    value = json.loads(line[len(marker) :])
    value["returncode"] = result.returncode
    return value


def native_environment(
    repo_root: str | Path, python: str | Path, *, device: str = "cpu"
) -> dict[str, str]:
    if device not in {"cpu", "cuda"}:
        raise ValueError("ProtoMotions v3 device must be 'cpu' or 'cuda'")
    root = Path(repo_root).resolve()
    env = os.environ.copy()
    paths = [
        str(root / "external/pyroki_upstream/src"),
        str(root / "external/ProtoMotions"),
        str(root / "src"),
    ]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(paths),
            "JAX_PLATFORMS": device,
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "RTCMP_PROTOMOTIONS_V3_PYTHON": str(Path(python).resolve()),
        }
    )
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
        # The environment contains an installed CUDA PJRT plugin even on CPU
        # workers.  JAX discovers installed plugins before honoring the
        # requested backend; skip that plugin's CUDA version/device probe while
        # still requiring and verifying the CPU backend in the native worker.
        env["JAX_SKIP_CUDA_CONSTRAINTS_CHECK"] = "1"
    return env


def audit_environment(
    repo_root: str | Path, python: str | Path, *, device: str = "cpu"
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    executable = Path(python).resolve()
    probe = _python_probe(
        executable, native_environment(root, executable, device=device)
    )
    pipeline_modules = (
        "jax",
        "jaxlib",
        "jaxlie",
        "jaxls",
        "jax_dataclasses",
        "yourdfpy",
        "numpy",
        "pyroki",
        "protomotions",
    )
    modules = probe.get("modules", {})
    pipeline_ready = probe.get("returncode") == 0 and all(
        modules.get(name, {}).get("available") is True for name in pipeline_modules
    )
    proto_commit = _git_commit(root / "external/ProtoMotions")
    pyroki_commit = _git_commit(root / "external/pyroki_upstream")
    jaxls_direct = modules.get("jaxls", {}).get("direct_url", {})
    jaxls_commit = jaxls_direct.get("vcs_info", {}).get("commit_id", "unavailable")
    return {
        "python_environment": executable.parent.parent.name,
        "requested_device": device,
        "probe": probe,
        "pipeline_required_modules": list(pipeline_modules),
        "pipeline_ready": pipeline_ready,
        "upstream_declared_but_unused_by_g1_script_missing": [
            name
            for name in ("robot_descriptions", "lzfse")
            if not modules.get(name, {}).get("available", False)
        ],
        "protomotions_commit": proto_commit,
        "protomotions_commit_matches": proto_commit == PROTOMOTIONS_V3_COMMIT,
        "pyroki_commit": pyroki_commit,
        "pyroki_commit_matches": pyroki_commit == PYROKI_COMMIT,
        "jaxls_expected_commit": JAXLS_COMMIT,
        "jaxls_commit": jaxls_commit,
        "jaxls_commit_matches": jaxls_commit == JAXLS_COMMIT,
        "note": (
            "ProtoMotions documents a floating PyRoki install. This benchmark freezes "
            "the checked-out PyRoki source and the installed jaxls direct-url revision."
        ),
    }


def build_native_command(
    *,
    python: str | Path,
    repo_root: str | Path,
    source: str | Path,
    keypoints: str | Path,
    native_output: str | Path,
    timing_json: str | Path,
    target_raw_frames: int = TARGET_RAW_FRAMES,
    warmup_runs: int = 1,
    measured_runs: int = 3,
    root_scale_multiplier: float = 1.0,
    local_scale_multiplier: float = 1.0,
    capture_runtime_witness: bool = False,
) -> list[str]:
    if (
        target_raw_frames < 1
        or warmup_runs < 0
        or measured_runs < 1
        or root_scale_multiplier <= 0.0
        or local_scale_multiplier <= 0.0
    ):
        raise ValueError("Invalid target/timing repetition count")
    command = [
        str(Path(python).resolve()),
        "-m",
        "retargeting_comparison.protomotions_v3_native",
        "--repo-root",
        str(Path(repo_root).resolve()),
        "--source",
        str(Path(source).resolve()),
        "--keypoints",
        str(Path(keypoints).resolve()),
        "--output",
        str(Path(native_output).resolve()),
        "--timing-json",
        str(Path(timing_json).resolve()),
        "--target-raw-frames",
        str(target_raw_frames),
        "--warmup-runs",
        str(warmup_runs),
        "--measured-runs",
        str(measured_runs),
        "--root-scale-multiplier",
        str(root_scale_multiplier),
        "--local-scale-multiplier",
        str(local_scale_multiplier),
    ]
    if capture_runtime_witness:
        command.append("--capture-runtime-witness")
    return command


def _continuous_wxyz(quaternions: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternions, dtype=np.float64).copy()
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("ProtoMotions output contains a zero-norm root quaternion")
    value /= norms
    if value[0, 0] < 0.0:
        value[0] *= -1.0
    for frame in range(1, len(value)):
        if float(np.dot(value[frame - 1], value[frame])) < 0.0:
            value[frame] *= -1.0
    return value


def convert_native_output(
    native_output: str | Path,
    *,
    source_frame_count: int,
    native_joint_names: Iterable[str] = G1_JOINT_NAMES,
    timing_json: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> CanonicalG1:
    """Convert the untouched official NPZ fields to canonical G1 qpos."""
    with np.load(native_output, allow_pickle=False) as data:
        required = {"base_frame_pos", "base_frame_wxyz", "joint_angles", "fps"}
        if not required.issubset(data.files):
            raise ValueError(f"Native ProtoMotions output is missing {sorted(required - set(data.files))}")
        root = np.asarray(data["base_frame_pos"], dtype=np.float64)
        quaternion = np.asarray(data["base_frame_wxyz"], dtype=np.float64)
        joints = np.asarray(data["joint_angles"], dtype=np.float64)
        fps = float(data["fps"])
        source_fps = float(data["source_fps"]) if "source_fps" in data.files else fps
        subsample = int(data["subsample_factor"]) if "subsample_factor" in data.files else 1
    frames = root.shape[0]
    if root.shape != (frames, 3) or quaternion.shape != (frames, 4):
        raise ValueError("Native ProtoMotions root fields have invalid shapes")
    names = tuple(native_joint_names)
    if len(names) != len(set(names)) or set(names) != set(G1_JOINT_NAMES):
        raise ValueError("Native ProtoMotions joint names do not match canonical G1-29")
    if joints.shape != (frames, len(names)):
        raise ValueError("Native ProtoMotions joint_angles must have shape [T,29]")
    if frames < 1 or source_frame_count < 1 or frames > source_frame_count:
        raise ValueError("Native ProtoMotions output has an invalid frame count")
    if not np.isfinite(root).all() or not np.isfinite(quaternion).all() or not np.isfinite(joints).all():
        raise ValueError("Native ProtoMotions output contains NaN/Inf")
    if not np.isfinite(fps) or fps <= 0.0 or subsample != 1:
        raise ValueError("Formal ProtoMotions output must retain positive native fps and stride 1")
    if np.max(np.abs(root)) > 100.0:
        raise ValueError("Native ProtoMotions root translation is inconsistent with metres")
    order = np.asarray([names.index(name) for name in G1_JOINT_NAMES], dtype=np.int64)
    quaternion = _continuous_wxyz(quaternion)
    qpos = np.concatenate((root, quaternion, joints[:, order]), axis=1).astype(np.float64)
    completion = "succeeded" if frames / source_frame_count >= MIN_COMPLETION_RATIO else "incomplete"
    timing: dict[str, Any] = {}
    if timing_json is not None:
        timing = json.loads(Path(timing_json).read_text())
    measured = [
        float(run["wall_time_s"])
        for run in timing.get("repetitions", [])
        if run.get("role") == "measured"
    ]
    selected_total = float(np.median(measured)) if measured else 0.0
    details: dict[str, Any] = {
        "method": "protomotions_v3",
        "method_family": "upstream",
        "implementation": "official whole-trajectory modified-PyRoki/JAXLS",
        "upstream_commit": PROTOMOTIONS_V3_COMMIT,
        "pyroki_commit": PYROKI_COMMIT,
        "completion_status": completion,
        "quaternion_order": "wxyz",
        "translation_unit": "metre",
        "native_joint_order": list(names),
        "canonical_joint_order": list(G1_JOINT_NAMES),
        "source_fps": source_fps,
        "subsample_factor": subsample,
        "native_output_sha256": sha256_file(native_output),
        "timing_granularity": "whole trajectory",
        "per_frame_solve_time_semantics": (
            "median measured whole-trajectory solve time amortized uniformly over frames; "
            "not an independently observed frame time"
        ),
        "steady_end_to_end_total_s": selected_total,
        "native_core_total_s": selected_total,
        "timing_protocol": timing,
        "adapter_changes": "field validation, joint-order assertion/reorder, float64 cast, quaternion normalization",
    }
    if metadata:
        details.update(metadata)
    motion = CanonicalG1(
        qpos=qpos,
        fps=fps,
        source_frame_idx=np.arange(frames, dtype=np.int64),
        valid=np.ones(frames, dtype=bool),
        per_frame_solve_time_s=np.full(frames, selected_total / frames, dtype=np.float64),
        metadata=details,
    )
    motion.validate(source_frame_count=source_frame_count)
    return motion


def validate_canonical_fk(
    motion: CanonicalG1, robot_xml: str | Path, *, sample_count: int = 64
) -> dict[str, Any]:
    """Prove that converted qpos is accepted by the frozen evaluator model."""
    from .robot_model import CanonicalRobotModel

    robot = CanonicalRobotModel(robot_xml)
    indices = np.unique(
        np.linspace(0, len(motion.qpos) - 1, min(sample_count, len(motion.qpos)), dtype=int)
    )
    maximum_limit_violation = 0.0
    minimum = np.full(3, np.inf)
    maximum = np.full(3, -np.inf)
    for index in indices:
        positions = np.stack(list(robot.semantic_positions(motion.qpos[index]).values()))
        if not np.isfinite(positions).all():
            raise ValueError(f"Canonical MuJoCo FK is non-finite at frame {index}")
        minimum = np.minimum(minimum, positions.min(axis=0))
        maximum = np.maximum(maximum, positions.max(axis=0))
        maximum_limit_violation = max(
            maximum_limit_violation, robot.joint_limit_violation(motion.qpos[index])
        )
    return {
        "robot_xml": str(Path(robot_xml).resolve()),
        "robot_xml_sha256": robot.sha256,
        "sampled_frames": indices.tolist(),
        "finite": True,
        "semantic_fk_aabb_min_m": minimum.tolist(),
        "semantic_fk_aabb_max_m": maximum.tolist(),
        "maximum_canonical_joint_limit_violation_rad": maximum_limit_violation,
    }


def joint_limit_diagnostics(
    motion: CanonicalG1, urdf_path: str | Path, *, tolerance_rad: float = 1e-4
) -> dict[str, Any]:
    """Report, but do not hide, violations against one frozen URDF policy."""
    joints = {joint.name: joint for joint in actuated_urdf_joints(urdf_path)}
    if set(joints) != set(G1_JOINT_NAMES):
        raise ValueError("Joint-limit diagnostic URDF is not canonical G1-29")
    angles = np.asarray(motion.qpos[:, 7:], dtype=np.float64)
    lower = np.asarray([joints[name].lower for name in G1_JOINT_NAMES], dtype=np.float64)
    upper = np.asarray([joints[name].upper for name in G1_JOINT_NAMES], dtype=np.float64)
    violation = np.maximum(lower[None, :] - angles, angles - upper[None, :])
    violation = np.maximum(violation - tolerance_rad, 0.0)
    maximum_by_joint = violation.max(axis=0)
    return {
        "urdf": str(Path(urdf_path)),
        "urdf_sha256": sha256_file(urdf_path),
        "tolerance_rad": tolerance_rad,
        "maximum_violation_rad": float(violation.max()),
        "violating_frame_count": int(np.count_nonzero(np.any(violation > 0.0, axis=1))),
        "violating_joint_count": int(np.count_nonzero(maximum_by_joint > 0.0)),
        "maximum_by_joint_rad": {
            name: float(maximum_by_joint[index])
            for index, name in enumerate(G1_JOINT_NAMES)
            if maximum_by_joint[index] > 0.0
        },
    }


def write_failure_evidence(
    path: str | Path,
    *,
    phase: str,
    reason: str,
    environment_audit: dict[str, Any] | None = None,
    command: list[str] | None = None,
    returncode: int | None = None,
    stdout_log: str | None = None,
    stderr_log: str | None = None,
    status: str = "na",
    outcome: str | None = None,
) -> None:
    if status not in {"na", "failed"}:
        raise ValueError("Failure evidence status must be 'na' or 'failed'")
    atomic_write_json(
        path,
        {
            "method": "ProtoMotions v3 / modified PyRoki",
            "status": status,
            "outcome": outcome
            or (
                "N/A — public human→G1 pipeline not integration-ready under the Pilot budget"
                if status == "na"
                else "Failed — official 600-frame solver did not complete; no substitute output"
            ),
            "phase": phase,
            "reason": reason,
            "upstream_commit": PROTOMOTIONS_V3_COMMIT,
            "pyroki_commit": PYROKI_COMMIT,
            "environment_audit": environment_audit,
            "command": command,
            "returncode": returncode,
            "stdout_log": stdout_log,
            "stderr_log": stderr_log,
            "synthetic_or_substitute_output_used": False,
        },
    )
