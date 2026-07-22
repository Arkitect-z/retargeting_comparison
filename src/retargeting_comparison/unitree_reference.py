"""Adapter and provenance checks for the Unitree-attributed LAFAN1 corpus.

The Hugging Face files handled here are an external reference corpus, not a
verified official baseline and not ground truth.  In particular, the dataset
owner stated in a public discussion that the data had been copied and that no
retargeting-method material was available.  The adapter preserves an exact
as-published view and separately exposes one frozen, scale-free coordinate-
convention view for comparison in the benchmark frame.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Literal
import xml.etree.ElementTree as ET

import numpy as np

from .constants import CANONICAL_QPOS_WIDTH, G1_JOINT_NAMES
from .io_utils import atomic_write_json, load_yaml, sha256_file
from .rotations import normalize_quaternion_wxyz, xyzw_to_wxyz
from .schemas import CanonicalG1, CanonicalHuman


UNITREE_REFERENCE_REPOSITORY = "lvhaidong/LAFAN1_Retargeting_Dataset"
UNITREE_REFERENCE_REVISION = "ce1572906efe6157840e8474d5a0d7aa87481e74"
UNITREE_REFERENCE_PILOT_PATH = "g1/dance1_subject1.csv"
UNITREE_REFERENCE_PILOT_SHA256 = (
    "e2a369e92e5ad076c5acffff7bce53157a0e6cd9e7e9f6a08dadfb1eb2928d8f"
)
UNITREE_REFERENCE_PILOT_AS_PUBLISHED_QPOS_SHA256 = (
    "b0755c087c5187453f5fe5eff874af321a303404124e01deba627c0f1ff77f27"
)
UNITREE_REFERENCE_PILOT_CANONICAL_QPOS_SHA256 = (
    "c2ad1eafaf229c52fa1b65baeb0e5318ec61420cb53f03929db2bdf502f8c958"
)
UNITREE_REFERENCE_FPS = 30.0
UNITREE_REFERENCE_LABEL = "Unitree-attributed reference corpus"
UNITREE_REFERENCE_ROLE = "external_reference_not_verified_ground_truth"
AS_PUBLISHED_VIEW = "as_published"
CANONICAL_COORDINATE_VIEW = "canonical_coordinate_view"
REFERENCE_TO_CANONICAL_YAW_RAD = -np.pi / 2.0
CoordinateView = Literal["as_published", "canonical_coordinate_view"]

UPSTREAM_ROOT_FIELDS = ("x", "y", "z", "qx", "qy", "qz", "qw")
CANONICAL_ROOT_FIELDS = ("x", "y", "z", "qw", "qx", "qy", "qz")
UPSTREAM_G1_COLUMNS = UPSTREAM_ROOT_FIELDS + G1_JOINT_NAMES


def _multiply_quaternion_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton product with broadcasting and scalar-first storage."""

    left_value = np.asarray(left, dtype=np.float64)
    right_value = np.asarray(right, dtype=np.float64)
    lw, lx, ly, lz = np.moveaxis(left_value, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right_value, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def transform_reference_coordinate_view(
    qpos: np.ndarray,
    *,
    inverse: bool = False,
) -> np.ndarray:
    """Rotate between the published and benchmark world-frame conventions.

    Published -> canonical is the frozen active world rotation ``Rz(-pi/2)``;
    the inverse uses ``Rz(+pi/2)``.  Absolute root XYZ and root orientation are
    transformed together.  Root anchoring, scale, height, joint values, and
    time are untouched.
    """

    value = np.asarray(qpos, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != CANONICAL_QPOS_WIDTH:
        raise ValueError(
            f"qpos must have shape [T,{CANONICAL_QPOS_WIDTH}], got {value.shape}"
        )
    if not np.isfinite(value).all():
        raise ValueError("qpos contains NaN/Inf")
    result = value.copy()
    angle = (
        -REFERENCE_TO_CANONICAL_YAW_RAD
        if inverse
        else REFERENCE_TO_CANONICAL_YAW_RAD
    )
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    rotation = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    result[:, :3] = result[:, :3] @ rotation.T
    world_rotation = np.asarray(
        [np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)],
        dtype=np.float64,
    )
    result[:, 3:7] = normalize_quaternion_wxyz(
        _multiply_quaternion_wxyz(world_rotation, result[:, 3:7])
    )
    return result


def qpos_content_sha256(qpos: np.ndarray) -> str:
    """Return a stable content hash independent of NPZ container timestamps."""

    value = np.ascontiguousarray(np.asarray(qpos, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(f"shape={value.shape};dtype=<f8;".encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class RootSimilarityAudit:
    """A descriptive 2-D similarity fit; it does not alter either trajectory."""

    scale: float
    rotation_row_major: tuple[tuple[float, float], tuple[float, float]]
    rotation_det: float
    rmse_m: float
    max_error_m: float
    reference_first_xy_m: tuple[float, float]
    source_first_xy_m: tuple[float, float]
    z_slope: float
    z_intercept_m: float
    z_rmse_m: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UrdfJointSpec:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]
    axis_xyz: tuple[float, float, float]
    lower: float | None
    upper: float | None


def _triplet(
    value: str | None, default: tuple[float, float, float]
) -> tuple[float, float, float]:
    if value is None:
        return default
    parsed = tuple(float(item) for item in value.split())
    if len(parsed) != 3:
        raise ValueError(f"Expected three components, got {value!r}")
    return parsed  # type: ignore[return-value]


def read_actuated_urdf_joints(path: str | Path) -> tuple[UrdfJointSpec, ...]:
    """Return non-fixed URDF joints in document order for kinematic auditing."""

    root = ET.parse(path).getroot()
    joints: list[UrdfJointSpec] = []
    for element in root.findall("joint"):
        joint_type = element.attrib.get("type", "")
        if joint_type in {"fixed", "floating"}:
            continue
        parent = element.find("parent")
        child = element.find("child")
        if parent is None or child is None:
            raise ValueError(f"Joint {element.attrib.get('name')} is missing parent/child")
        origin = element.find("origin")
        axis = element.find("axis")
        limit = element.find("limit")
        joints.append(
            UrdfJointSpec(
                name=element.attrib["name"],
                joint_type=joint_type,
                parent=parent.attrib["link"],
                child=child.attrib["link"],
                origin_xyz=_triplet(
                    None if origin is None else origin.attrib.get("xyz"),
                    (0.0, 0.0, 0.0),
                ),
                origin_rpy=_triplet(
                    None if origin is None else origin.attrib.get("rpy"),
                    (0.0, 0.0, 0.0),
                ),
                axis_xyz=_triplet(
                    None if axis is None else axis.attrib.get("xyz"),
                    (1.0, 0.0, 0.0),
                ),
                lower=(
                    None
                    if limit is None or "lower" not in limit.attrib
                    else float(limit.attrib["lower"])
                ),
                upper=(
                    None
                    if limit is None or "upper" not in limit.attrib
                    else float(limit.attrib["upper"])
                ),
            )
        )
    return tuple(joints)


def compare_urdf_kinematics(
    reference_path: str | Path,
    candidate_path: str | Path,
    *,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Compare joint order, tree, axes, origins, and limits of two URDFs.

    Visual, collision, inertial, transmission, and actuator properties are
    intentionally outside this test.  A positive result is therefore
    *kinematic-contract equivalence*, not byte or dynamics equivalence.
    """

    reference = read_actuated_urdf_joints(reference_path)
    candidate = read_actuated_urdf_joints(candidate_path)
    reference_by_name = {joint.name: joint for joint in reference}
    candidate_by_name = {joint.name: joint for joint in candidate}
    missing = sorted(set(reference_by_name) - set(candidate_by_name))
    extra = sorted(set(candidate_by_name) - set(reference_by_name))

    tree_mismatches: list[str] = []
    type_mismatches: list[str] = []
    numeric_mismatches: list[str] = []
    max_origin_xyz = 0.0
    max_origin_rpy = 0.0
    max_axis = 0.0
    max_limit = 0.0
    for name in sorted(set(reference_by_name) & set(candidate_by_name)):
        left = reference_by_name[name]
        right = candidate_by_name[name]
        if (left.parent, left.child) != (right.parent, right.child):
            tree_mismatches.append(name)
        if left.joint_type != right.joint_type:
            type_mismatches.append(name)
        origin_xyz = float(
            np.max(np.abs(np.asarray(left.origin_xyz) - np.asarray(right.origin_xyz)))
        )
        origin_rpy = float(
            np.max(np.abs(np.asarray(left.origin_rpy) - np.asarray(right.origin_rpy)))
        )
        axis = float(np.max(np.abs(np.asarray(left.axis_xyz) - np.asarray(right.axis_xyz))))
        max_origin_xyz = max(max_origin_xyz, origin_xyz)
        max_origin_rpy = max(max_origin_rpy, origin_rpy)
        max_axis = max(max_axis, axis)
        limit_delta = 0.0
        for left_limit, right_limit in ((left.lower, right.lower), (left.upper, right.upper)):
            if left_limit is None and right_limit is None:
                continue
            if left_limit is None or right_limit is None:
                limit_delta = float("inf")
                break
            limit_delta = max(limit_delta, abs(left_limit - right_limit))
        max_limit = max(max_limit, limit_delta)
        if max(origin_xyz, origin_rpy, axis, limit_delta) > tolerance:
            numeric_mismatches.append(name)

    reference_order = tuple(joint.name for joint in reference)
    candidate_order = tuple(joint.name for joint in candidate)
    return {
        "reference_sha256": sha256_file(reference_path),
        "candidate_sha256": sha256_file(candidate_path),
        "reference_joint_count": len(reference),
        "candidate_joint_count": len(candidate),
        "reference_order": list(reference_order),
        "candidate_order": list(candidate_order),
        "joint_order_match": reference_order == candidate_order,
        "canonical_g1_order_match": reference_order == G1_JOINT_NAMES,
        "missing_joints": missing,
        "extra_joints": extra,
        "tree_mismatches": tree_mismatches,
        "type_mismatches": type_mismatches,
        "numeric_mismatches": numeric_mismatches,
        "max_origin_xyz_difference_m": max_origin_xyz,
        "max_origin_rpy_difference_rad": max_origin_rpy,
        "max_axis_component_difference": max_axis,
        "max_joint_limit_difference_rad": max_limit,
        "kinematic_contract_equivalent": not (
            missing
            or extra
            or tree_mismatches
            or type_mismatches
            or numeric_mismatches
            or reference_order != candidate_order
        ),
        "comparison_scope": (
            "actuated joint order/tree/origin/axis/limits only; excludes visual, "
            "collision, inertial, transmission, and actuator properties"
        ),
    }


def _rpy_matrix(rpy: tuple[float, float, float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _axis_angle_matrix(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    direction = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise ValueError("URDF joint axis has zero length")
    x, y, z = direction / norm
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    one_minus = 1.0 - cosine
    return np.asarray(
        [
            [
                cosine + x * x * one_minus,
                x * y * one_minus - z * sine,
                x * z * one_minus + y * sine,
            ],
            [
                y * x * one_minus + z * sine,
                cosine + y * y * one_minus,
                y * z * one_minus - x * sine,
            ],
            [
                z * x * one_minus - y * sine,
                z * y * one_minus + x * sine,
                cosine + z * z * one_minus,
            ],
        ],
        dtype=np.float64,
    )


def _numpy_urdf_fk(
    path: Path,
    *,
    root_xyz: np.ndarray,
    root_rotation: np.ndarray,
    joint_values: dict[str, float],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Evaluate all URDF link frames using only NumPy rigid transforms."""

    xml_root = ET.parse(path).getroot()
    records: list[
        tuple[
            str,
            str,
            str,
            str,
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ]
    ] = []
    child_links: set[str] = set()
    for element in xml_root.findall("joint"):
        parent = element.find("parent")
        child = element.find("child")
        if parent is None or child is None:
            raise ValueError(f"Joint {element.attrib.get('name')} lacks parent/child")
        origin = element.find("origin")
        axis = element.find("axis")
        child_name = child.attrib["link"]
        child_links.add(child_name)
        records.append(
            (
                element.attrib["name"],
                element.attrib.get("type", "fixed"),
                parent.attrib["link"],
                child_name,
                _triplet(
                    None if origin is None else origin.attrib.get("xyz"),
                    (0.0, 0.0, 0.0),
                ),
                _triplet(
                    None if origin is None else origin.attrib.get("rpy"),
                    (0.0, 0.0, 0.0),
                ),
                _triplet(
                    None if axis is None else axis.attrib.get("xyz"),
                    (1.0, 0.0, 0.0),
                ),
            )
        )
    link_names = {element.attrib["name"] for element in xml_root.findall("link")}
    base_links = sorted(link_names - child_links)
    if base_links != ["pelvis"]:
        raise ValueError(f"Expected pelvis as the sole URDF base, got {base_links}")
    poses: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "pelvis": (
            np.asarray(root_xyz, dtype=np.float64),
            np.asarray(root_rotation, dtype=np.float64),
        )
    }
    unresolved = records.copy()
    while unresolved:
        remaining = []
        progressed = False
        for name, joint_type, parent, child, origin_xyz, origin_rpy, axis in unresolved:
            if parent not in poses:
                remaining.append(
                    (name, joint_type, parent, child, origin_xyz, origin_rpy, axis)
                )
                continue
            parent_position, parent_rotation = poses[parent]
            origin_rotation = _rpy_matrix(origin_rpy)
            child_position = parent_position + parent_rotation @ np.asarray(origin_xyz)
            child_rotation = parent_rotation @ origin_rotation
            value = float(joint_values.get(name, 0.0))
            if joint_type in {"revolute", "continuous"}:
                child_rotation = child_rotation @ _axis_angle_matrix(axis, value)
            elif joint_type == "prismatic":
                child_position = child_position + child_rotation @ (
                    np.asarray(axis, dtype=np.float64) * value
                )
            elif joint_type not in {"fixed", "floating"}:
                raise ValueError(f"Unsupported URDF joint type {joint_type!r}")
            poses[child] = (child_position, child_rotation)
            progressed = True
        if not progressed:
            raise ValueError("URDF tree could not be resolved from the pelvis base")
        unresolved = remaining
    return poses


def compare_urdf_to_mujoco_fk(
    reference_urdf_path: str | Path,
    canonical_mjcf_path: str | Path,
    *,
    random_sample_count: int = 10,
    seed: int = 1947,
    position_tolerance_m: float = 1e-5,
    rotation_tolerance_rad: float = 1e-5,
) -> dict[str, Any]:
    """Cross-check the 29 actuated link frames in URDF FK and MuJoCo.

    MuJoCo is imported inside the function so the CSV adapter itself does not
    acquire a heavyweight dependency.  The reference side uses deterministic
    NumPy URDF tree FK, avoiding an optional Pinocchio dependency in the
    fail-closed validator.  The comparison covers the pelvis and the child link
    of every actuated joint at neutral plus deterministic random poses.  It does
    not compare geometry, contact, inertia, transmission, or actuator properties
    and therefore cannot establish dynamics/collision equivalence.
    """

    if random_sample_count < 0:
        raise ValueError("random_sample_count must be non-negative")
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - depends on audit environment
        raise RuntimeError("The cross-engine asset audit requires mujoco") from exc

    reference_path = Path(reference_urdf_path)
    canonical_path = Path(canonical_mjcf_path)
    mj_model = mujoco.MjModel.from_xml_path(str(canonical_path))
    mj_data = mujoco.MjData(mj_model)
    if mj_model.nq != CANONICAL_QPOS_WIDTH:
        raise ValueError("The MuJoCo asset must implement the frozen 36-value G1 contract")

    mj_joint_names = tuple(
        mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, index)
        for index in range(1, mj_model.njnt)
    )
    if mj_joint_names != G1_JOINT_NAMES:
        raise ValueError("Canonical MJCF joint order differs from the G1 contract")
    reference_joints = read_actuated_urdf_joints(reference_path)
    reference_joint_names = tuple(joint.name for joint in reference_joints)
    if reference_joint_names != G1_JOINT_NAMES:
        raise ValueError("Reference URDF joint order differs from the G1 contract")

    link_names = ("pelvis",) + tuple(joint.child for joint in reference_joints)
    for name in link_names:
        body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"Missing cross-engine link frame {name!r}")

    lower = np.asarray(
        [mj_model.jnt_range[index, 0] for index in range(1, mj_model.njnt)],
        dtype=np.float64,
    )
    upper = np.asarray(
        [mj_model.jnt_range[index, 1] for index in range(1, mj_model.njnt)],
        dtype=np.float64,
    )
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("Canonical MJCF must have finite limits for all 29 joints")
    rng = np.random.default_rng(seed)
    joint_samples = [np.zeros(len(G1_JOINT_NAMES), dtype=np.float64)]
    joint_samples.extend(
        rng.uniform(lower, upper) for _ in range(random_sample_count)
    )

    position_errors: list[float] = []
    rotation_errors: list[float] = []
    worst_link = ""
    worst_position = -1.0
    root_xyz = np.asarray([0.17, -0.23, 0.91], dtype=np.float64)
    root_yaw = 0.31
    root_xyzw = np.asarray(
        [0.0, 0.0, np.sin(root_yaw / 2.0), np.cos(root_yaw / 2.0)],
        dtype=np.float64,
    )
    root_rotation = np.asarray(
        [
            [np.cos(root_yaw), -np.sin(root_yaw), 0.0],
            [np.sin(root_yaw), np.cos(root_yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    for joint_values in joint_samples:
        reference_poses = _numpy_urdf_fk(
            reference_path,
            root_xyz=root_xyz,
            root_rotation=root_rotation,
            joint_values=dict(zip(G1_JOINT_NAMES, joint_values)),
        )

        mj_data.qpos[:3] = root_xyz
        mj_data.qpos[3:7] = xyzw_to_wxyz(root_xyzw)
        mj_data.qpos[7:] = joint_values
        mujoco.mj_forward(mj_model, mj_data)

        for name in link_names:
            body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
            reference_position, reference_rotation = reference_poses[name]
            mj_position = mj_data.xpos[body_id]
            mj_rotation = mj_data.xmat[body_id].reshape(3, 3)
            position_error = float(np.linalg.norm(reference_position - mj_position))
            cosine = float(
                np.clip(
                    (np.trace(reference_rotation.T @ mj_rotation) - 1.0) / 2.0,
                    -1.0,
                    1.0,
                )
            )
            rotation_error = float(np.arccos(cosine))
            position_errors.append(position_error)
            rotation_errors.append(rotation_error)
            if position_error > worst_position:
                worst_position = position_error
                worst_link = name

    max_position = max(position_errors)
    max_rotation = max(rotation_errors)
    return {
        "reference_urdf_sha256": sha256_file(reference_path),
        "canonical_mjcf_sha256": sha256_file(canonical_path),
        "seed": seed,
        "neutral_sample_count": 1,
        "random_sample_count": random_sample_count,
        "link_frames_per_sample": len(link_names),
        "max_position_error_m": max_position,
        "rmse_position_error_m": float(
            np.sqrt(np.mean(np.square(position_errors)))
        ),
        "max_rotation_error_rad": max_rotation,
        "worst_position_link": worst_link,
        "kinematic_fk_equivalent": (
            max_position <= position_tolerance_m
            and max_rotation <= rotation_tolerance_rad
        ),
        "position_tolerance_m": position_tolerance_m,
        "rotation_tolerance_rad": rotation_tolerance_rad,
        "comparison_scope": (
            "pelvis plus 29 actuated child-link frames; excludes visual, collision, "
            "inertial, transmission, actuator, and dynamics properties"
        ),
    }


def load_unitree_g1_csv(
    path: str | Path,
    *,
    frame_start: int = 0,
    frame_end: int | None = None,
    expected_sha256: str | None = None,
    expected_source_frames: int | None = None,
    canonical_source_sha256: str | None = None,
    canonical_source_fps: float | None = None,
    coordinate_view: CoordinateView = AS_PUBLISHED_VIEW,
) -> CanonicalG1:
    """Load a revision-pinned G1 CSV and convert root ``xyzw`` to ``wxyz``.

    The as-published view applies no coordinate transform.  The canonical
    coordinate view additionally applies the frozen ``Rz(-pi/2)`` world-frame
    conversion to root XYZ and orientation, with no fitted angle or scale.
    No anchoring, scale, grounding, resampling, interpolation, or joint-angle
    change is applied.  The all-zero solve-time array is an explicit placeholder
    because this is a precomputed corpus with no timing provenance and must not
    enter runtime comparisons.
    """

    csv_path = Path(path)
    if coordinate_view not in (AS_PUBLISHED_VIEW, CANONICAL_COORDINATE_VIEW):
        raise ValueError(f"Unsupported coordinate view {coordinate_view!r}")
    if expected_source_frames is not None and expected_source_frames <= 0:
        raise ValueError("expected_source_frames must be positive")
    if canonical_source_sha256 is not None and len(canonical_source_sha256) != 64:
        raise ValueError("canonical_source_sha256 must be a SHA-256 digest")
    digest = sha256_file(csv_path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(
            f"Reference CSV SHA-256 mismatch: expected {expected_sha256}, got {digest}"
        )
    raw = np.loadtxt(csv_path, delimiter=",", dtype=np.float64, ndmin=2)
    if raw.ndim != 2 or raw.shape[1] != CANONICAL_QPOS_WIDTH:
        raise ValueError(
            f"Unitree reference CSV must have {CANONICAL_QPOS_WIDTH} columns, "
            f"got shape {raw.shape}"
        )
    if not np.isfinite(raw).all():
        raise ValueError("Unitree reference CSV contains NaN/Inf")
    stop = len(raw) if frame_end is None else frame_end
    if frame_start < 0 or stop <= frame_start or stop > len(raw):
        raise ValueError(
            f"Invalid frame interval [{frame_start}, {stop}) for {len(raw)} rows"
        )

    selected = raw[frame_start:stop].copy()
    upstream_norms = np.linalg.norm(selected[:, 3:7], axis=1)
    canonical_quaternion = normalize_quaternion_wxyz(
        xyzw_to_wxyz(selected[:, 3:7])
    )
    selected[:, 3:7] = canonical_quaternion
    if coordinate_view == CANONICAL_COORDINATE_VIEW:
        selected = transform_reference_coordinate_view(selected)
    frames = len(selected)
    source_frames = (
        expected_source_frames if expected_source_frames is not None else frames
    )
    completion_status = "succeeded" if frames / source_frames >= 0.95 else "incomplete"
    motion = CanonicalG1(
        qpos=selected,
        fps=UNITREE_REFERENCE_FPS,
        source_frame_idx=np.arange(frame_start, stop, dtype=np.int64),
        valid=np.ones(frames, dtype=bool),
        per_frame_solve_time_s=np.zeros(frames, dtype=np.float64),
        metadata={
            "method": "unitree_reference",
            "method_family": "external_precomputed_reference",
            "label": UNITREE_REFERENCE_LABEL,
            "provenance_role": UNITREE_REFERENCE_ROLE,
            "verified_official_ground_truth": False,
            "upstream_repository": UNITREE_REFERENCE_REPOSITORY,
            "upstream_revision": UNITREE_REFERENCE_REVISION,
            "upstream_path": UNITREE_REFERENCE_PILOT_PATH,
            "source_csv_sha256": digest,
            "canonical_source_sha256": canonical_source_sha256,
            "canonical_source_fps": canonical_source_fps,
            "timeline_relation": (
                "same frame indices; reference declares nominal 30 Hz while canonical "
                "LAFAN frame time resolves to 30.000300003 Hz"
            ),
            "alignment_basis": (
                "same sequence basename and frame indices only; byte-identical human "
                "source and exact timestamp identity are not claimed"
            ),
            "byte_identical_human_source_verified": False,
            "exact_timestamp_identity_claimed": False,
            "source_csv_rows": int(len(raw)),
            "source_csv_columns": int(raw.shape[1]),
            "frame_start": int(frame_start),
            "frame_end": int(stop),
            "source_root_fields": list(UPSTREAM_ROOT_FIELDS),
            "canonical_root_fields": list(CANONICAL_ROOT_FIELDS),
            "root_quaternion_conversion": "xyzw_to_wxyz_then_unit_normalize",
            "coordinate_view": coordinate_view,
            "as_published_semantics": (
                "coordinate-as-published after required root quaternion xyzw-to-wxyz "
                "conversion and unit normalization; root/joint numeric values otherwise unchanged"
            ),
            "coordinate_view_transform": (
                "identity"
                if coordinate_view == AS_PUBLISHED_VIEW
                else "fixed_active_world_Rz(-pi/2)_on_root_xyz_and_orientation"
            ),
            "coordinate_rotation_yaw_rad": (
                0.0
                if coordinate_view == AS_PUBLISHED_VIEW
                else REFERENCE_TO_CANONICAL_YAW_RAD
            ),
            "coordinate_rotation_fitted_from_results": False,
            "coordinate_transform_scale": 1.0,
            "coordinate_transform_translation_m": [0.0, 0.0, 0.0],
            "absolute_root_anchor_preserved": True,
            "upstream_quaternion_norm_max_abs_error": float(
                np.max(np.abs(upstream_norms - 1.0))
            ),
            "joint_names": list(G1_JOINT_NAMES),
            "fps": UNITREE_REFERENCE_FPS,
            "timing_available": False,
            "per_frame_solve_time_s_semantics": (
                "zero placeholder; excluded from runtime and RTF comparisons"
            ),
            "adapter_translation_scale_ground_resampling": "none",
            "qpos_content_sha256": qpos_content_sha256(selected),
            "completion_status": completion_status,
        },
    )
    motion.validate(source_frame_count=source_frames)
    return motion


def build_unitree_reference_pilot(repo_root: str | Path = ".") -> dict[str, Any]:
    """Rebuild both frozen Pilot coordinate views and a hash-bound manifest."""

    root = Path(repo_root).resolve()
    config_path = root / "configs/unitree_reference.yaml"
    config = load_yaml(config_path)
    sequence_path = root / "manifests/pilot_sequence.yaml"
    sequence = load_yaml(sequence_path)
    human_path = root / str(sequence["canonical_path"])
    human = CanonicalHuman.load(human_path)
    if len(human.timestamps) != 600:
        raise ValueError("Unitree Stage-1 reference is frozen to the 600-frame Pilot")
    revision = str(config["upstream"]["revision"])
    revision_root = root / "data/external/unitree_lafan1_reference" / revision
    raw_path = root / str(config["upstream"]["local_ignored_path"])
    if sha256_file(raw_path) != UNITREE_REFERENCE_PILOT_SHA256:
        raise ValueError("Pinned Unitree reference Pilot CSV changed")

    resource_paths = {
        "readme": revision_root / "README.md",
        "license": revision_root / "LICENSE",
        "metadata": revision_root / "meta_data/info.json",
        "visualizer": revision_root / "rerun_visualize.py",
        "g1_urdf": revision_root / "robot_description/g1/g1_29dof_rev_1_0.urdf",
    }
    resources: dict[str, dict[str, Any]] = {}
    for key, path in resource_paths.items():
        declared = config["upstream"]["pinned_resources"][key]
        if (
            not path.is_file()
            or path.stat().st_size != int(declared["size_bytes"])
            or sha256_file(path) != str(declared["sha256"])
        ):
            raise ValueError(f"Pinned Unitree resource changed: {key}")
        resources[key] = {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    output_dir = (
        root
        / "runs"
        / str(sequence["sequence_id"])
        / "unitree-attributed-reference"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    views = {
        AS_PUBLISHED_VIEW: output_dir / "canonical_g1.as-published.npz",
        CANONICAL_COORDINATE_VIEW: output_dir / "canonical_g1.npz",
    }
    output_rows: dict[str, dict[str, Any]] = {}
    for view, output in views.items():
        motion = load_unitree_g1_csv(
            raw_path,
            frame_start=0,
            frame_end=600,
            expected_sha256=UNITREE_REFERENCE_PILOT_SHA256,
            expected_source_frames=len(human.timestamps),
            canonical_source_sha256=human.source_sha256,
            canonical_source_fps=float(human.fps),
            coordinate_view=view,  # type: ignore[arg-type]
        )
        motion.metadata["canonical_source_path"] = human_path.relative_to(root).as_posix()
        expected_content = (
            UNITREE_REFERENCE_PILOT_AS_PUBLISHED_QPOS_SHA256
            if view == AS_PUBLISHED_VIEW
            else UNITREE_REFERENCE_PILOT_CANONICAL_QPOS_SHA256
        )
        if motion.metadata["qpos_content_sha256"] != expected_content:
            raise RuntimeError(f"Unitree {view} qpos content changed")
        motion.save(output, source_frame_count=len(human.timestamps))
        output_rows[view] = {
            "path": output.relative_to(root).as_posix(),
            "sha256": sha256_file(output),
            "qpos_content_sha256": expected_content,
            "frames": len(motion.qpos),
            "fps": float(motion.fps),
            "frame_start": 0,
            "frame_end_exclusive": 600,
        }

    csv_files = sorted((revision_root / "g1").glob("*.csv"))
    inventory_digest = hashlib.sha256()
    for path in csv_files:
        inventory_digest.update(path.name.encode("utf-8"))
        inventory_digest.update(bytes.fromhex(sha256_file(path)))

    stage = load_yaml(root / "configs/stage1.yaml")
    evaluator_path = root / "manifests/evaluator.yaml"
    canonical_urdf = root / str(stage["canonical_robot"]["urdf"])
    canonical_scene = root / str(stage["canonical_robot"]["xml"])
    bound_research = [
        root / "research/unitree_reference_provenance.csv",
        root / "research/unitree_reference_pilot_audit.csv",
        root / "research/UNITREE_REFERENCE_CORPUS_AUDIT.md",
    ]
    manifest = {
        "schema_version": 2,
        "label": UNITREE_REFERENCE_LABEL,
        "role": UNITREE_REFERENCE_ROLE,
        "verified_official_ground_truth": False,
        "eligible_for_runtime_comparison": False,
        "upstream_repository": UNITREE_REFERENCE_REPOSITORY,
        "upstream_revision": revision,
        "raw_pilot": {
            "path": raw_path.relative_to(root).as_posix(),
            "sha256": sha256_file(raw_path),
            "size_bytes": raw_path.stat().st_size,
            "rows": 3945,
            "columns": 36,
        },
        "pinned_resources": resources,
        "g1_csv_inventory": {
            "files": len(csv_files),
            "aggregate_sha256": inventory_digest.hexdigest(),
        },
        "source_binding": {
            "sequence_manifest": sequence_path.relative_to(root).as_posix(),
            "sequence_manifest_sha256": sha256_file(sequence_path),
            "canonical_source": human_path.relative_to(root).as_posix(),
            "canonical_source_file_sha256": sha256_file(human_path),
            "canonical_source_content_sha256": human.source_sha256,
            "canonical_source_fps": float(human.fps),
            "alignment_basis": "same basename and frame indices 0:600",
            "byte_identical_human_source_verified": False,
            "exact_timestamp_identity_claimed": False,
        },
        "coordinate_views": output_rows,
        "asset_binding": {
            "reference_urdf": resources["g1_urdf"],
            "canonical_evaluator_urdf_path": canonical_urdf.relative_to(root).as_posix(),
            "canonical_evaluator_urdf_sha256": sha256_file(canonical_urdf),
            "canonical_evaluator_scene_path": canonical_scene.relative_to(root).as_posix(),
            "canonical_evaluator_scene_sha256": sha256_file(canonical_scene),
            "evaluator_manifest_path": evaluator_path.relative_to(root).as_posix(),
            "evaluator_manifest_sha256": sha256_file(evaluator_path),
        },
        "adapter": {
            "implementation_path": Path(__file__).resolve().relative_to(root).as_posix(),
            "implementation_sha256": sha256_file(Path(__file__)),
            "operations": (
                "xyzw-to-wxyz and quaternion normalization for both views; canonical "
                "view additionally applies fixed Rz(-pi/2); no scale/translation/ground/resample"
            ),
        },
        "research_evidence": [
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
            }
            for path in bound_research
        ],
    }
    manifest_path = root / "manifests/unitree_reference.json"
    atomic_write_json(manifest_path, manifest)
    return manifest


def fit_root_xy_similarity(
    source_root_m: np.ndarray,
    reference_root_m: np.ndarray,
) -> RootSimilarityAudit:
    """Fit a proper row-vector 2-D similarity to root displacements.

    This descriptive diagnostic separates the reference corpus's observed
    root anchor/heading/scale.  The fit itself never alters either input.
    """

    source = np.asarray(source_root_m, dtype=np.float64)
    reference = np.asarray(reference_root_m, dtype=np.float64)
    if source.shape != reference.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source_root_m and reference_root_m must both have shape [T,3]")
    if (
        len(source) < 2
        or not np.isfinite(source).all()
        or not np.isfinite(reference).all()
    ):
        raise ValueError("Root trajectories must contain at least two finite frames")

    source_xy = source[:, :2] - source[0, :2]
    reference_xy = reference[:, :2] - reference[0, :2]
    left, _, right_t = np.linalg.svd(source_xy.T @ reference_xy)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right_t
    denominator = float(np.sum(source_xy * source_xy))
    if denominator < 1e-12:
        raise ValueError("Source root trajectory has no measurable XY displacement")
    scale = float(np.sum((source_xy @ rotation) * reference_xy) / denominator)
    residual = reference_xy - scale * (source_xy @ rotation)
    error = np.linalg.norm(residual, axis=1)

    z_design = np.column_stack((source[:, 2], np.ones(len(source))))
    z_slope, z_intercept = np.linalg.lstsq(z_design, reference[:, 2], rcond=None)[0]
    z_residual = reference[:, 2] - z_design @ np.asarray([z_slope, z_intercept])
    return RootSimilarityAudit(
        scale=scale,
        rotation_row_major=(
            (float(rotation[0, 0]), float(rotation[0, 1])),
            (float(rotation[1, 0]), float(rotation[1, 1])),
        ),
        rotation_det=float(np.linalg.det(rotation)),
        rmse_m=float(np.sqrt(np.mean(error**2))),
        max_error_m=float(np.max(error)),
        reference_first_xy_m=(float(reference[0, 0]), float(reference[0, 1])),
        source_first_xy_m=(float(source[0, 0]), float(source[0, 1])),
        z_slope=float(z_slope),
        z_intercept_m=float(z_intercept),
        z_rmse_m=float(np.sqrt(np.mean(z_residual**2))),
    )
