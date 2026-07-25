"""Rerun visualization for the frozen Stage 1 human-to-G1 comparison.

The viewer intentionally reads only the canonical on-disk contracts. Robot
forward kinematics uses the frozen G1 URDF so the ``vis`` environment does not
need MuJoCo or any method-specific dependencies.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .calibration import human_heading_yaw, load_evaluator_protocol
from .constants import G1_JOINT_NAMES, STAGE1_RUN_DIRECTORIES
from .io_utils import atomic_write_json, load_yaml, sha256_file
from .rotations import quaternion_wxyz_to_matrix, yaw_from_matrix
from .schemas import CanonicalG1, CanonicalHuman
from .smpl_skinning import (
    SmplSkinMotion,
    default_skin_cache,
    smpl_mesh_sequence,
)


SHARED_COMPARISON_FRAMES = 450


@dataclass(frozen=True)
class MethodStyle:
    key: str
    display_name: str
    color: tuple[int, int, int]
    operating_point: bool
    external_reference: bool = False
    annotate_artifact_causes: bool = True


METHOD_STYLES = (
    MethodStyle("sparse-neutral", "Sparse · neutral", (238, 106, 73), True),
    MethodStyle("dense", "Dense", (42, 157, 111), True),
    MethodStyle("gmr", "GMR", (225, 174, 49), True),
    MethodStyle("omniretarget", "OmniRetarget", (132, 113, 238), True),
    MethodStyle(
        "protomotions-v2.3",
        "ProtoMotions v2.3 · Mink",
        (243, 139, 45),
        True,
    ),
    MethodStyle(
        "protomotions-v3",
        "ProtoMotions v3 · modified PyRoki",
        (73, 191, 171),
        True,
    ),
    MethodStyle(
        "unitree-reference",
        "Unitree-attributed external reference",
        (245, 245, 245),
        False,
        external_reference=True,
        annotate_artifact_causes=False,
    ),
    MethodStyle("sparse-a", "Sparse · seed A", (47, 169, 209), False),
    MethodStyle("sparse-b", "Sparse · seed B", (214, 91, 156), False),
)

SOURCE_STYLE = MethodStyle(
    "source-human",
    "Source human · fitted SMPL skin",
    (230, 234, 241),
    True,
)

LANE_OFFSETS = {
    "source-human": (-8.0, 3.0, 0.0),
    "sparse-neutral": (-4.0, 3.0, 0.0),
    "dense": (0.0, 3.0, 0.0),
    "gmr": (4.0, 3.0, 0.0),
    "omniretarget": (8.0, 3.0, 0.0),
    "protomotions-v2.3": (-8.0, -3.0, 0.0),
    "protomotions-v3": (-4.0, -3.0, 0.0),
    "unitree-reference": (0.0, -3.0, 0.0),
    "sparse-a": (4.0, -3.0, 0.0),
    "sparse-b": (8.0, -3.0, 0.0),
}

EXPECTED_TRAJECTORY_KEYS = tuple(style.key for style in METHOD_STYLES)

# There is exactly one admissible run directory and one admissible evaluator
# table for every trajectory.  These paths deliberately have no legacy
# fallback: a stale output must fail closed rather than silently enter a new
# recording.
if set(EXPECTED_TRAJECTORY_KEYS) != set(STAGE1_RUN_DIRECTORIES):
    raise RuntimeError(
        "Rerun trajectory registry must exactly match STAGE1_RUN_DIRECTORIES"
    )

PUBLICATION_SUMMARY_FILES = (
    "metrics/stage1_core_summary.csv",
    "metrics/stage1_sparse_seed_evaluations.csv",
    "metrics/stage1_reference_summary.csv",
)

VIEW_METHOD_KEYS = {
    "grid": tuple(style.key for style in METHOD_STYLES),
    "world": tuple(
        style.key
        for style in METHOD_STYLES
        if style.operating_point or style.external_reference
    ),
    "root_frame": tuple(
        style.key
        for style in METHOD_STYLES
        if style.operating_point or style.external_reference
    ),
    "seeds": tuple(style.key for style in METHOD_STYLES if style.key.startswith("sparse-")),
}

CLOSEUP_METHOD_KEYS = EXPECTED_TRAJECTORY_KEYS
EXPECTED_VIEW_INSTANCE_COUNTS = {
    **{view: len(keys) for view, keys in VIEW_METHOD_KEYS.items()},
    "closeups": len(CLOSEUP_METHOD_KEYS),
}

ROBOT_SEMANTIC_LINKS = {
    "root": "pelvis",
    "torso": "torso_link",
    "head": "mid360_link",
    "left_shoulder": "left_shoulder_roll_link",
    "right_shoulder": "right_shoulder_roll_link",
    "left_elbow": "left_elbow_link",
    "right_elbow": "right_elbow_link",
    "left_wrist": "left_wrist_yaw_link",
    "right_wrist": "right_wrist_yaw_link",
    "left_hip": "left_hip_roll_link",
    "right_hip": "right_hip_roll_link",
    "left_knee": "left_knee_link",
    "right_knee": "right_knee_link",
    "left_ankle": "left_ankle_roll_link",
    "right_ankle": "right_ankle_roll_link",
    "left_toe": "left_foot_contact_point",
    "right_toe": "right_foot_contact_point",
}

ROBOT_SEMANTICS = tuple(ROBOT_SEMANTIC_LINKS)
ROBOT_SEMANTIC_INDEX = {name: index for index, name in enumerate(ROBOT_SEMANTICS)}
ROBOT_BONES = (
    ("root", "torso"),
    ("torso", "head"),
    ("torso", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("torso", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("root", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_toe"),
    ("root", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_toe"),
)
ROBOT_BONE_INDICES = tuple(
    (ROBOT_SEMANTIC_INDEX[parent], ROBOT_SEMANTIC_INDEX[child])
    for parent, child in ROBOT_BONES
)

METRIC_FIELDS = (
    "rf_kpe_all_m",
    "rf_kpe_targeted_m",
    "rf_kpe_untracked_m",
    "root_translation_error_m",
    "root_yaw_error_rad",
    "ground_penetration_depth_m",
    "left_foot_speed_m_s",
    "right_foot_speed_m_s",
    "left_source_stance",
    "right_source_stance",
    "left_foot_skating",
    "right_foot_skating",
    "ground_penetration_artifact",
    "joint_limit_artifact",
    "invalid_artifact",
    "artifact",
    "solve_time_s",
)


def _parse_vector(value: str) -> np.ndarray:
    vector = np.fromstring(value, sep=" ", dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"Expected a three-vector in URDF, got {value!r}")
    return vector


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    """URDF fixed-axis roll/pitch/yaw rotation matrix."""

    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.asarray([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.asarray([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def _origin_transform(element: ET.Element | None) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    if element is None:
        return transform
    transform[:3, :3] = _rpy_matrix(
        _parse_vector(element.attrib.get("rpy", "0 0 0"))
    )
    transform[:3, 3] = _parse_vector(element.attrib.get("xyz", "0 0 0"))
    return transform


def _rgba32(value: str | None) -> tuple[int, int, int, int]:
    if value is None:
        return (178, 178, 178, 255)
    rgba = np.fromstring(value, sep=" ", dtype=np.float64)
    if rgba.shape != (4,) or not np.isfinite(rgba).all():
        raise ValueError(f"Expected a finite URDF RGBA four-vector, got {value!r}")
    return tuple(int(round(component * 255.0)) for component in np.clip(rgba, 0.0, 1.0))


def _axis_rotation(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Vectorized homogeneous Rodrigues rotation for one URDF joint axis."""

    direction = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        raise ValueError("URDF revolute joint has a zero axis")
    x, y, z = direction / norm
    values = np.asarray(angle, dtype=np.float64)
    cosine = np.cos(values)
    sine = np.sin(values)
    one_minus = 1.0 - cosine
    result = np.zeros((len(values), 4, 4), dtype=np.float64)
    result[:, 0, 0] = cosine + x * x * one_minus
    result[:, 0, 1] = x * y * one_minus - z * sine
    result[:, 0, 2] = x * z * one_minus + y * sine
    result[:, 1, 0] = y * x * one_minus + z * sine
    result[:, 1, 1] = cosine + y * y * one_minus
    result[:, 1, 2] = y * z * one_minus - x * sine
    result[:, 2, 0] = z * x * one_minus - y * sine
    result[:, 2, 1] = z * y * one_minus + x * sine
    result[:, 2, 2] = cosine + z * z * one_minus
    result[:, 3, 3] = 1.0
    return result


@dataclass(frozen=True)
class UrdfVisual:
    """One link-local visual mesh from the frozen G1 URDF."""

    link_name: str
    mesh_path: Path
    origin: np.ndarray
    scale: np.ndarray
    rgba: tuple[int, int, int, int]


class UrdfSemanticKinematics:
    """Compute the evaluator's semantic G1 points from canonical qpos."""

    def __init__(self, urdf_path: str | Path):
        self.urdf_path = Path(urdf_path).resolve()
        if not self.urdf_path.is_file():
            raise FileNotFoundError(f"Canonical G1 URDF is missing: {self.urdf_path}")
        robot = ET.parse(self.urdf_path).getroot()
        links = {element.attrib["name"] for element in robot.findall("link")}
        children = {
            element.find("child").attrib["link"]  # type: ignore[union-attr]
            for element in robot.findall("joint")
        }
        roots = links - children
        if roots != {"pelvis"}:
            raise ValueError(f"Expected G1 base link 'pelvis', got roots {sorted(roots)}")

        named_materials: dict[str, tuple[int, int, int, int]] = {}
        for material in robot.findall("material"):
            color = material.find("color")
            if color is not None:
                named_materials[material.attrib["name"]] = _rgba32(
                    color.attrib.get("rgba")
                )
        visuals: list[UrdfVisual] = []
        for link in robot.findall("link"):
            link_name = link.attrib["name"]
            for visual in link.findall("visual"):
                mesh = visual.find("geometry/mesh")
                if mesh is None:
                    continue
                filename = mesh.attrib["filename"]
                if filename.startswith("package://"):
                    raise ValueError(
                        f"Package-relative G1 visual mesh is unsupported: {filename}"
                    )
                mesh_path = (self.urdf_path.parent / filename).resolve()
                if not mesh_path.is_file():
                    raise FileNotFoundError(
                        f"G1 URDF visual mesh is missing for {link_name}: {mesh_path}"
                    )
                material = visual.find("material")
                rgba = (178, 178, 178, 255)
                if material is not None:
                    color = material.find("color")
                    if color is not None:
                        rgba = _rgba32(color.attrib.get("rgba"))
                    elif material.attrib.get("name") in named_materials:
                        rgba = named_materials[material.attrib["name"]]
                scale = _parse_vector(mesh.attrib.get("scale", "1 1 1"))
                if np.any(scale <= 0.0):
                    raise ValueError(f"Non-positive G1 visual scale for {link_name}")
                visuals.append(
                    UrdfVisual(
                        link_name=link_name,
                        mesh_path=mesh_path,
                        origin=_origin_transform(visual.find("origin")),
                        scale=scale,
                        rgba=rgba,
                    )
                )

        joints = []
        actuated_names = []
        for element in robot.findall("joint"):
            name = element.attrib["name"]
            joint_type = element.attrib["type"]
            parent = element.find("parent").attrib["link"]  # type: ignore[union-attr]
            child = element.find("child").attrib["link"]  # type: ignore[union-attr]
            origin = _origin_transform(element.find("origin"))
            axis_element = element.find("axis")
            axis = _parse_vector(
                axis_element.attrib.get("xyz", "1 0 0") if axis_element is not None else "1 0 0"
            )
            if joint_type in {"revolute", "continuous"}:
                actuated_names.append(name)
            elif joint_type != "fixed":
                raise ValueError(f"Unsupported G1 URDF joint type {joint_type!r}")
            joints.append((name, joint_type, parent, child, origin, axis))
        if tuple(actuated_names) != G1_JOINT_NAMES:
            raise ValueError("G1 URDF joint order differs from the canonical 29-DoF contract")
        self.joints = tuple(joints)
        self.visuals = tuple(visuals)

    def semantic_positions(self, qpos: np.ndarray) -> dict[str, np.ndarray]:
        value = np.asarray(qpos, dtype=np.float64)
        if value.shape != (36,):
            raise ValueError("Canonical G1 qpos must have shape [36]")
        points = self.motion_positions(value[None])[0]
        return {name: points[index] for index, name in enumerate(ROBOT_SEMANTICS)}

    def motion_positions(self, qpos: np.ndarray) -> np.ndarray:
        return self._positions_from_link_transforms(self.motion_link_transforms(qpos))

    def motion_link_transforms(self, qpos: np.ndarray) -> dict[str, np.ndarray]:
        """Return batched world-from-link transforms for every URDF link."""

        values = np.asarray(qpos, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 36:
            raise ValueError("Canonical G1 motion qpos must have shape [T,36]")
        frames = len(values)
        root_transform = np.zeros((frames, 4, 4), dtype=np.float64)
        root_transform[:, :3, :3] = quaternion_wxyz_to_matrix(values[:, 3:7])
        root_transform[:, :3, 3] = values[:, :3]
        root_transform[:, 3, 3] = 1.0
        transforms: dict[str, np.ndarray] = {"pelvis": root_transform}
        joint_index = {name: index for index, name in enumerate(G1_JOINT_NAMES)}
        for name, joint_type, parent, child, origin, axis in self.joints:
            local = np.broadcast_to(origin, (frames, 4, 4))
            if joint_type in {"revolute", "continuous"}:
                rotation = _axis_rotation(axis, values[:, 7 + joint_index[name]])
                local = origin @ rotation
            transforms[child] = transforms[parent] @ local

        return transforms

    @staticmethod
    def _positions_from_link_transforms(
        transforms: dict[str, np.ndarray],
    ) -> np.ndarray:
        frames = len(transforms["pelvis"])
        result = np.empty((frames, len(ROBOT_SEMANTICS), 3), dtype=np.float64)
        for semantic, link in ROBOT_SEMANTIC_LINKS.items():
            result[:, ROBOT_SEMANTIC_INDEX[semantic]] = transforms[link][:, :3, 3]
        return result


@dataclass
class MethodVisualization:
    style: MethodStyle
    motion: CanonicalG1
    positions: np.ndarray
    link_transforms: dict[str, np.ndarray]
    metrics: dict[str, np.ndarray]
    output_path: Path
    metrics_path: Path
    metrics_summary_path: Path
    evidence_binding: dict[str, Any]


@dataclass
class HumanSkinVisualization:
    """Visualization-only SMPL surface fitted to the original LAFAN BVH."""

    motion: SmplSkinMotion
    vertices: np.ndarray
    faces: np.ndarray
    cache_path: Path
    evidence_path: Path


@dataclass
class Stage1Visualization:
    repo_root: Path
    sequence: dict[str, Any]
    human: CanonicalHuman
    human_scale: float
    methods: dict[str, MethodVisualization]
    urdf_path: Path
    robot_visuals: tuple[UrdfVisual, ...]
    missing_methods: dict[str, str]
    source_path: Path
    evaluator_path: Path
    evaluator_robot_path: Path
    human_skin: HumanSkinVisualization
    acceptance_evidence: bool
    snapshot_role: str

    @property
    def frame_count(self) -> int:
        return min(
            SHARED_COMPARISON_FRAMES,
            len(self.human.timestamps),
            *(len(method.motion.qpos) for method in self.methods.values()),
        )


def _parse_metric(value: str) -> float:
    normalized = value.strip().lower()
    if normalized in {"true", "false"}:
        return float(normalized == "true")
    return float(value)


def _load_metrics(
    path: Path,
    frame_count: int,
    *,
    allow_trailing_frames: bool = False,
) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Per-frame evaluation metrics are missing: {path}")
    result = {name: np.full(frame_count, np.nan, dtype=np.float64) for name in METRIC_FIELDS}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            frame = int(row["source_frame_idx"])
            if not 0 <= frame < frame_count:
                if allow_trailing_frames and frame >= frame_count:
                    continue
                raise ValueError(f"Metric frame {frame} is outside the source timeline")
            for field in METRIC_FIELDS:
                result[field][frame] = _parse_metric(row[field])
    return result


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _run_output_path(root: Path, sequence_id: str, key: str) -> Path:
    """Resolve one output exclusively through the live Stage 1 registry."""

    return (
        root
        / "runs"
        / sequence_id
        / STAGE1_RUN_DIRECTORIES[key]
        / "canonical_g1.npz"
    )


def _metric_path(root: Path, key: str) -> Path:
    """Return the sole publication-grade per-frame evaluator table."""

    return root / "metrics" / "stage1_publication" / "runs" / f"{key}_per_frame.csv"


def _metric_summary_path(root: Path, key: str) -> Path:
    return root / "metrics" / "stage1_publication" / "runs" / f"{key}_summary.json"


def _publication_binding_rows(
    root: Path,
) -> tuple[dict[str, tuple[dict[str, str], Path]], tuple[Path, ...]]:
    rows: dict[str, tuple[dict[str, str], Path]] = {}
    missing: list[Path] = []
    for relative in PUBLICATION_SUMMARY_FILES:
        path = root / relative
        if not path.is_file():
            missing.append(path)
            continue
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                key = str(row.get("key", ""))
                if key not in EXPECTED_TRAJECTORY_KEYS:
                    continue
                if key in rows:
                    raise ValueError(f"Duplicate publication binding row for {key}")
                rows[key] = (row, path)
    return rows, tuple(missing)


def _load_human_skin(
    root: Path,
    human: CanonicalHuman,
    source_path: Path,
) -> HumanSkinVisualization:
    skin_cache_path = default_skin_cache(root)
    skin_evidence_path = skin_cache_path.with_name(
        "lafan_bvh_fitted_smpl.evidence.json"
    )
    if not skin_cache_path.is_file() or not skin_evidence_path.is_file():
        raise FileNotFoundError(
            "The fitted source-human SMPL skin is missing. Run "
            "`rtcmp prepare-smpl-skin` before building the Rerun recording."
        )
    skin_motion = SmplSkinMotion.load(skin_cache_path)
    if (
        len(skin_motion.pose_aa_zup) != len(human.timestamps)
        or not np.isclose(skin_motion.fps, human.fps)
    ):
        raise ValueError("The fitted SMPL skin has a stale source timeline")
    if skin_motion.metadata.get("canonical_source_file_sha256") != sha256_file(
        source_path
    ):
        raise ValueError("The fitted SMPL skin is bound to a different source")
    skin_vertices, skin_faces, _ = smpl_mesh_sequence(
        skin_motion,
        root,
        frame_limit=min(SHARED_COMPARISON_FRAMES, len(human.timestamps)),
        batch_size=64,
        device="cpu",
    )
    return HumanSkinVisualization(
        motion=skin_motion,
        vertices=skin_vertices,
        faces=skin_faces,
        cache_path=skin_cache_path,
        evidence_path=skin_evidence_path,
    )


def _build_evidence_binding(
    *,
    root: Path,
    key: str,
    output_path: Path,
    metrics_path: Path,
    metrics_summary_path: Path,
    publication_row: dict[str, str],
    publication_path: Path,
    human: CanonicalHuman,
    source_path: Path,
    evaluator_path: Path,
    evaluator: dict[str, Any],
    urdf_path: Path,
) -> dict[str, Any]:
    """Verify and describe the output↔metric↔source↔robot evidence chain."""

    output_rel = output_path.relative_to(root).as_posix()
    output_sha = sha256_file(output_path)
    evaluator_sha = sha256_file(evaluator_path)
    evaluator_robot_path = root / str(evaluator["robot_xml"])
    declared_output = str(publication_row.get("output_path", ""))
    if declared_output != output_rel:
        raise ValueError(
            f"{key} publication row points to {declared_output!r}, expected {output_rel!r}"
        )
    if publication_row.get("output_sha256") != output_sha:
        raise ValueError(f"{key} publication output hash is stale")

    metric_summary = json.loads(metrics_summary_path.read_text(encoding="utf-8"))
    if not isinstance(metric_summary, dict):
        raise ValueError(f"{key} evaluator summary is not a mapping")
    for owner, value in (
        ("publication row", publication_row),
        ("metric summary", metric_summary),
    ):
        if str(value.get("canonical_source_sha256", "")) != human.source_sha256:
            raise ValueError(f"{key} {owner} is bound to a different source")
        if str(value.get("evaluator_protocol_sha256", "")) != evaluator_sha:
            raise ValueError(f"{key} {owner} is bound to a different evaluator")
        if str(value.get("robot_model_sha256", "")) != str(
            evaluator["robot_xml_sha256"]
        ):
            raise ValueError(f"{key} {owner} is bound to a different evaluator robot")
        expected_comparison_frames = min(
            SHARED_COMPARISON_FRAMES, len(human.timestamps)
        )
        if int(float(value.get("frames", -1))) != expected_comparison_frames:
            raise ValueError(f"{key} {owner} has the wrong frame count")

    if sha256_file(evaluator_robot_path) != str(evaluator["robot_xml_sha256"]):
        raise ValueError("Evaluator robot XML hash differs from its frozen manifest")
    if str(evaluator.get("source_sha256", "")) != human.source_sha256:
        raise ValueError("Evaluator manifest is bound to a different source")

    binding: dict[str, Any] = {
        "run_directory_registry_key": key,
        "run_directory": STAGE1_RUN_DIRECTORIES[key],
        "output": {
            "path": output_rel,
            "size_bytes": output_path.stat().st_size,
            "sha256": output_sha,
        },
        "per_frame_metrics": {
            "path": metrics_path.relative_to(root).as_posix(),
            "size_bytes": metrics_path.stat().st_size,
            "sha256": sha256_file(metrics_path),
        },
        "metrics_summary": {
            "path": metrics_summary_path.relative_to(root).as_posix(),
            "size_bytes": metrics_summary_path.stat().st_size,
            "sha256": sha256_file(metrics_summary_path),
        },
        "publication_binding_table": {
            "path": publication_path.relative_to(root).as_posix(),
            "sha256": sha256_file(publication_path),
        },
        "source": {
            "path": source_path.relative_to(root).as_posix(),
            "canonical_package_sha256": sha256_file(source_path),
            "declared_source_sha256": human.source_sha256,
        },
        "evaluator": {
            "path": evaluator_path.relative_to(root).as_posix(),
            "sha256": evaluator_sha,
            "schema_version": int(evaluator["schema_version"]),
        },
        "robot": {
            "canonical_urdf_path": urdf_path.relative_to(root).as_posix(),
            "canonical_urdf_sha256": sha256_file(urdf_path),
            "evaluator_xml_path": evaluator_robot_path.relative_to(root).as_posix(),
            "evaluator_xml_sha256": sha256_file(evaluator_robot_path),
        },
        "verified": True,
    }
    binding["binding_sha256"] = _canonical_json_sha256(binding)
    return binding


def load_stage1_visualization(
    repo_root: str | Path = ".",
    sequence_manifest: str | Path = "manifests/pilot_sequence.yaml",
    *,
    skip_missing_methods: bool = False,
) -> Stage1Visualization:
    root = Path(repo_root).resolve()
    sequence_path = Path(sequence_manifest)
    if not sequence_path.is_absolute():
        sequence_path = root / sequence_path
    sequence = load_yaml(sequence_path)
    source_path = root / sequence["canonical_path"]
    human = CanonicalHuman.load(source_path)
    stage = load_yaml(root / "configs" / "stage1.yaml")
    urdf_value = stage["canonical_robot"].get("urdf")
    if not urdf_value:
        raise ValueError("configs/stage1.yaml must declare canonical_robot.urdf")
    urdf_path = root / urdf_value
    kinematics = UrdfSemanticKinematics(urdf_path)
    evaluator_path = root / "manifests" / "evaluator.yaml"
    evaluator = load_yaml(evaluator_path)
    evaluator_robot_path = root / str(evaluator["robot_xml"])
    publication_rows, missing_publication_tables = _publication_binding_rows(root)
    human_skin = _load_human_skin(root, human, source_path)

    sequence_id = sequence["sequence_id"]
    methods: dict[str, MethodVisualization] = {}
    missing_methods: dict[str, str] = {}
    for style in METHOD_STYLES:
        path = _run_output_path(root, sequence_id, style.key)
        metrics_path = _metric_path(root, style.key)
        metrics_summary_path = _metric_summary_path(root, style.key)
        publication_binding = publication_rows.get(style.key)
        missing = []
        if not path.is_file():
            missing.append(f"registered output {path}")
        if not metrics_path.is_file():
            missing.append(f"publication metrics {metrics_path}")
        if not metrics_summary_path.is_file():
            missing.append(f"publication metric summary {metrics_summary_path}")
        if publication_binding is None:
            missing.append(
                "publication output binding row"
                + (
                    " (tables missing: "
                    + ", ".join(str(value) for value in missing_publication_tables)
                    + ")"
                    if missing_publication_tables
                    else ""
                )
            )
        if missing:
            missing_methods[style.key] = "; ".join(missing)
            continue
        assert publication_binding is not None
        motion = CanonicalG1.load(path)
        motion.validate(source_frame_count=len(human.timestamps))
        expected_frames = (
            SHARED_COMPARISON_FRAMES
            if style.key == "protomotions-v3"
            else len(human.timestamps)
        )
        if (
            len(motion.qpos) != expected_frames
            or not np.array_equal(
                motion.source_frame_idx, np.arange(expected_frames)
            )
        ):
            raise ValueError(
                f"{style.key} does not cover its registered source-frame contract"
            )
        link_transforms = kinematics.motion_link_transforms(motion.qpos)
        positions = kinematics._positions_from_link_transforms(link_transforms)
        metrics = _load_metrics(metrics_path, SHARED_COMPARISON_FRAMES)
        if not all(np.isfinite(value).all() for value in metrics.values()):
            raise ValueError(f"{style.key} visualization metrics contain missing or non-finite values")
        publication_row, publication_path = publication_binding
        evidence_binding = _build_evidence_binding(
            root=root,
            key=style.key,
            output_path=path,
            metrics_path=metrics_path,
            metrics_summary_path=metrics_summary_path,
            publication_row=publication_row,
            publication_path=publication_path,
            human=human,
            source_path=source_path,
            evaluator_path=evaluator_path,
            evaluator=evaluator,
            urdf_path=urdf_path,
        )
        methods[style.key] = MethodVisualization(
            style=style,
            motion=motion,
            positions=positions,
            link_transforms=link_transforms,
            metrics=metrics,
            output_path=path,
            metrics_path=metrics_path,
            metrics_summary_path=metrics_summary_path,
            evidence_binding=evidence_binding,
        )

    if missing_methods and not skip_missing_methods:
        details = "\n".join(
            f"- {key}: {reason}" for key, reason in missing_methods.items()
        )
        raise FileNotFoundError(
            "Stage 1 Rerun inputs are incomplete:\n"
            f"{details}\n"
            "Complete/evaluate these methods, or set skip_missing_methods=True "
            "for a diagnostic recording. A skipped recording is not Stage 1 evidence."
        )
    if not methods:
        raise FileNotFoundError("No complete method output/metric pairs are available to visualize")

    protocol = load_evaluator_protocol(root / "manifests" / "evaluator.yaml")
    if not missing_methods and tuple(methods) != EXPECTED_TRAJECTORY_KEYS:
        raise ValueError("Visualization did not load the exact registered trajectory order")
    return Stage1Visualization(
        repo_root=root,
        sequence=sequence,
        human=human,
        human_scale=float(protocol["scale"]["common_static_scale"]),
        methods=methods,
        urdf_path=urdf_path,
        robot_visuals=kinematics.visuals,
        missing_methods=missing_methods,
        source_path=source_path,
        evaluator_path=evaluator_path,
        evaluator_robot_path=evaluator_robot_path,
        human_skin=human_skin,
        acceptance_evidence=True,
        snapshot_role="stage1_publication",
    )


def load_current_diagnostic_visualization(
    repo_root: str | Path = ".",
    snapshot_manifest: str | Path = (
        "artifacts/visualization/"
        "stage1_current_completed_9methods.manifest.json"
    ),
) -> Stage1Visualization:
    """Load the prior nine-method snapshot without granting acceptance status."""

    root = Path(repo_root).resolve()
    manifest_path = Path(snapshot_manifest)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    snapshot = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        snapshot.get("acceptance_evidence") is not False
        or snapshot.get("snapshot_role")
        != "diagnostic_current_completed_methods"
        or tuple(snapshot.get("methods", ())) != EXPECTED_TRAJECTORY_KEYS
    ):
        raise ValueError("The current diagnostic snapshot manifest is stale")
    sequence = load_yaml(root / "manifests" / "pilot_sequence.yaml")
    if snapshot.get("sequence_id") != sequence["sequence_id"]:
        raise ValueError("The diagnostic snapshot is bound to a different sequence")
    source_path = root / str(sequence["canonical_path"])
    human = CanonicalHuman.load(source_path)
    human_skin = _load_human_skin(root, human, source_path)
    stage = load_yaml(root / "configs" / "stage1.yaml")
    urdf_path = root / str(stage["canonical_robot"]["urdf"])
    kinematics = UrdfSemanticKinematics(urdf_path)
    evaluator_path = root / "manifests" / "evaluator.yaml"
    evaluator = load_yaml(evaluator_path)
    evaluator_robot_path = root / str(evaluator["robot_xml"])
    protocol = load_evaluator_protocol(evaluator_path)

    selected = snapshot.get("selected_outputs", {})
    if set(selected) != set(EXPECTED_TRAJECTORY_KEYS):
        raise ValueError("The diagnostic snapshot output set is incomplete")
    methods: dict[str, MethodVisualization] = {}
    for style in METHOD_STYLES:
        output_entry = selected[style.key]
        output_path = root / str(output_entry["path"])
        if (
            not output_path.is_file()
            or sha256_file(output_path) != str(output_entry["sha256"])
        ):
            raise ValueError(f"Diagnostic output hash is stale for {style.key}")
        metric_path = (
            root
            / "metrics"
            / "diagnostic_current"
            / f"{style.key}_per_frame.csv"
        )
        metric_summary_path = metric_path.with_name(f"{style.key}_summary.json")
        if not metric_path.is_file() or not metric_summary_path.is_file():
            raise FileNotFoundError(
                f"Diagnostic evaluator evidence is missing for {style.key}"
            )
        motion = CanonicalG1.load(output_path)
        motion.validate(source_frame_count=len(human.timestamps))
        if len(motion.qpos) < SHARED_COMPARISON_FRAMES:
            raise ValueError(f"Diagnostic output is too short for {style.key}")
        link_transforms = kinematics.motion_link_transforms(motion.qpos)
        metrics = _load_metrics(
            metric_path,
            SHARED_COMPARISON_FRAMES,
            allow_trailing_frames=True,
        )
        if not all(np.isfinite(value).all() for value in metrics.values()):
            raise ValueError(f"Diagnostic metrics are non-finite for {style.key}")
        binding: dict[str, Any] = {
            "mode": "diagnostic_snapshot",
            "acceptance_evidence": False,
            "output": {
                "path": output_path.relative_to(root).as_posix(),
                "sha256": sha256_file(output_path),
            },
            "per_frame_metrics": {
                "path": metric_path.relative_to(root).as_posix(),
                "sha256": sha256_file(metric_path),
            },
            "metrics_summary": {
                "path": metric_summary_path.relative_to(root).as_posix(),
                "sha256": sha256_file(metric_summary_path),
            },
            "source": {
                "path": source_path.relative_to(root).as_posix(),
                "sha256": sha256_file(source_path),
            },
            "verified": True,
        }
        binding["binding_sha256"] = _canonical_json_sha256(binding)
        methods[style.key] = MethodVisualization(
            style=style,
            motion=motion,
            positions=kinematics._positions_from_link_transforms(link_transforms),
            link_transforms=link_transforms,
            metrics=metrics,
            output_path=output_path,
            metrics_path=metric_path,
            metrics_summary_path=metric_summary_path,
            evidence_binding=binding,
        )
    return Stage1Visualization(
        repo_root=root,
        sequence=sequence,
        human=human,
        human_scale=float(protocol["scale"]["common_static_scale"]),
        methods=methods,
        urdf_path=urdf_path,
        robot_visuals=kinematics.visuals,
        missing_methods={},
        source_path=source_path,
        evaluator_path=evaluator_path,
        evaluator_robot_path=evaluator_robot_path,
        human_skin=human_skin,
        acceptance_evidence=False,
        snapshot_role="diagnostic_current_completed_methods",
    )


def root_frame_points(points: np.ndarray, roots: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Express world points in each frame's root/yaw coordinate system."""

    values = np.asarray(points, dtype=np.float64)
    origins = np.asarray(roots, dtype=np.float64)
    angles = np.asarray(yaw, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] != len(origins) or len(angles) != len(origins):
        raise ValueError("points, roots, and yaw must share the same frame count")
    local = values - origins[:, None, :]
    cosine = np.cos(-angles)[:, None]
    sine = np.sin(-angles)[:, None]
    x = cosine * local[..., 0] - sine * local[..., 1]
    y = sine * local[..., 0] + cosine * local[..., 1]
    return np.stack((x, y, local[..., 2]), axis=-1)


def _world_aligned_robot(method: MethodVisualization) -> np.ndarray:
    points = method.positions.copy()
    initial_root = points[0, ROBOT_SEMANTIC_INDEX["root"]]
    points[..., :2] -= initial_root[:2]
    return points


def _world_aligned_link_transforms(
    method: MethodVisualization,
) -> dict[str, np.ndarray]:
    initial_root = method.positions[0, ROBOT_SEMANTIC_INDEX["root"]]
    offset = np.asarray([-initial_root[0], -initial_root[1], 0.0])
    result: dict[str, np.ndarray] = {}
    for link, transforms in method.link_transforms.items():
        aligned = transforms.copy()
        aligned[:, :3, 3] += offset
        result[link] = aligned
    return result


def _root_frame_link_transforms(
    method: MethodVisualization,
) -> dict[str, np.ndarray]:
    roots = method.positions[:, ROBOT_SEMANTIC_INDEX["root"]]
    yaw = yaw_from_matrix(quaternion_wxyz_to_matrix(method.motion.qpos[:, 3:7]))
    cosine = np.cos(-yaw)
    sine = np.sin(-yaw)
    world_to_root = np.zeros((len(roots), 4, 4), dtype=np.float64)
    world_to_root[:, 0, 0] = cosine
    world_to_root[:, 0, 1] = -sine
    world_to_root[:, 1, 0] = sine
    world_to_root[:, 1, 1] = cosine
    world_to_root[:, 2, 2] = 1.0
    world_to_root[:, 3, 3] = 1.0
    world_to_root[:, :3, 3] = -np.einsum(
        "tij,tj->ti", world_to_root[:, :3, :3], roots
    )
    return {
        link: world_to_root @ transforms
        for link, transforms in method.link_transforms.items()
    }


def visual_instance_poses(
    visual: UrdfVisual,
    method_keys: tuple[str, ...],
    transforms: dict[str, dict[str, np.ndarray]],
    frame: int,
    offsets: dict[str, tuple[float, float, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build world-space poses for one articulated visual link across methods."""

    poses = []
    for key in method_keys:
        pose = transforms[key][visual.link_name][frame] @ visual.origin
        if offsets is not None:
            pose = pose.copy()
            pose[:3, 3] += np.asarray(offsets[key], dtype=np.float64)
        poses.append(pose)
    stacked = np.asarray(poses, dtype=np.float64)
    scales = np.broadcast_to(visual.scale, (len(method_keys), 3)).copy()
    return stacked[:, :3, 3], stacked[:, :3, :3], scales


def _world_aligned_human(data: Stage1Visualization) -> np.ndarray:
    points = (
        data.human.world_positions - data.human.root_translation[0, None, :]
    ) * data.human_scale
    points[..., 2] += _human_display_ground_shift(data)
    return points


def _human_display_ground_shift(data: Stage1Visualization) -> float:
    points = (
        data.human.world_positions - data.human.root_translation[0, None, :]
    ) * data.human_scale
    names = {name: index for index, name in enumerate(data.human.joint_names.astype(str))}
    feet = [names[name] for name in ("LeftFoot", "LeftToe", "RightFoot", "RightToe")]
    # Display alignment is method-independent: source feet start on z=0.  It
    # cannot inherit the root height or scale of a comparison method.
    return -float(np.min(points[0, feet, 2]))


def _world_aligned_human_skin(data: Stage1Visualization) -> np.ndarray:
    vertices = (
        data.human_skin.vertices
        - data.human.root_translation[0, None, :]
    ) * data.human_scale
    vertices[..., 2] += _human_display_ground_shift(data)
    return vertices


def _root_frame_human_skin(data: Stage1Visualization) -> np.ndarray:
    return root_frame_points(
        data.human_skin.vertices,
        data.human.root_translation[: len(data.human_skin.vertices)],
        human_heading_yaw(data.human)[: len(data.human_skin.vertices)],
    ) * data.human_scale


def _human_edges(human: CanonicalHuman) -> tuple[tuple[int, int], ...]:
    return tuple((int(parent), child) for child, parent in enumerate(human.parent_indices) if parent >= 0)


def _segments(points: np.ndarray, edges: tuple[tuple[int, int], ...]) -> list[np.ndarray]:
    return [points[[parent, child]] for parent, child in edges]


def _active_styles(data: Stage1Visualization) -> tuple[MethodStyle, ...]:
    return tuple(style for style in METHOD_STYLES if style.key in data.methods)


def _active_view_method_keys(
    data: Stage1Visualization, view: str
) -> tuple[str, ...]:
    return tuple(key for key in VIEW_METHOD_KEYS[view] if key in data.methods)


def _recording_id(data: Stage1Visualization) -> str:
    digest = hashlib.sha256(
        b"articulated_g1_visual_meshes_v6_fitted_smpl_skin"
    )
    digest.update(data.human.source_sha256.encode())
    digest.update(data.snapshot_role.encode())
    digest.update(str(data.acceptance_evidence).encode())
    digest.update(bytes.fromhex(sha256_file(data.human_skin.cache_path)))
    digest.update(bytes.fromhex(sha256_file(data.human_skin.evidence_path)))
    digest.update(bytes.fromhex(sha256_file(data.urdf_path)))
    digest.update(
        bytes.fromhex(sha256_file(data.repo_root / "manifests" / "evaluator.yaml"))
    )
    for style in _active_styles(data):
        method = data.methods[style.key]
        digest.update(style.key.encode())
        digest.update(bytes.fromhex(sha256_file(method.output_path)))
        digest.update(bytes.fromhex(sha256_file(method.metrics_path)))
    for visual in data.robot_visuals:
        digest.update(bytes.fromhex(sha256_file(visual.mesh_path)))
    return str(uuid.UUID(bytes=digest.digest()[:16]))


def _method_metric(data: Stage1Visualization, key: str, field: str, frame: int) -> float:
    return float(data.methods[key].metrics[field][frame])


def _hidden_robot_debug_overrides(
    rrb: Any, root: str, method_keys: tuple[str, ...]
) -> dict[str, Any]:
    """Keep quantitative skeleton helpers available but hidden by default."""

    return {
        f"/{root}/{key}/{leaf}": rrb.EntityBehavior(visible=False)
        for key in method_keys
        for leaf in ("bones", "joints")
    }


def _spatial_overrides(
    rrb: Any, root: str, method_keys: tuple[str, ...]
) -> dict[str, Any]:
    overrides = _hidden_robot_debug_overrides(rrb, root, method_keys)
    if root in {"grid", "world", "root_frame"}:
        overrides.update(
            {
                f"/{root}/source-human/bones": rrb.EntityBehavior(visible=False),
                f"/{root}/source-human/joints": rrb.EntityBehavior(visible=False),
            }
        )
    return overrides


def _blueprint(rrb: Any, fps: float) -> Any:
    main_spatial = rrb.Tabs(
        rrb.Spatial3DView(
            origin="/grid",
            name="Side-by-side · articulated G1",
            line_grid=True,
            eye_controls=rrb.EyeControls3D(
                position=(0.0, -25.0, 8.0),
                look_target=(0.0, 0.0, 1.0),
                eye_up=(0.0, 0.0, 1.0),
            ),
            overrides=_spatial_overrides(
                rrb, "grid", VIEW_METHOD_KEYS["grid"]
            ),
        ),
        rrb.Spatial3DView(
            origin="/world",
            name="World overlay · root tracking",
            line_grid=True,
            eye_controls=rrb.EyeControls3D(
                position=(3.5, -6.5, 3.2),
                look_target=(0.0, 0.0, 0.9),
                eye_up=(0.0, 0.0, 1.0),
            ),
            overrides=_spatial_overrides(
                rrb, "world", VIEW_METHOD_KEYS["world"]
            ),
        ),
        rrb.Spatial3DView(
            origin="/root_frame",
            name="Root-frame pose overlay",
            line_grid=False,
            eye_controls=rrb.EyeControls3D(
                position=(2.8, -4.2, 2.0),
                look_target=(0.0, 0.0, 0.8),
                eye_up=(0.0, 0.0, 1.0),
            ),
            overrides=_spatial_overrides(
                rrb, "root_frame", VIEW_METHOD_KEYS["root_frame"]
            ),
        ),
        rrb.Spatial3DView(
            origin="/seeds",
            name="Sparse seed ambiguity",
            line_grid=False,
            eye_controls=rrb.EyeControls3D(
                position=(2.8, -4.2, 2.0),
                look_target=(0.0, 0.0, 0.8),
                eye_up=(0.0, 0.0, 1.0),
            ),
            overrides=_spatial_overrides(
                rrb, "seeds", VIEW_METHOD_KEYS["seeds"]
            ),
        ),
        name="Overlay views",
    )
    closeups = rrb.Grid(
        *(
            rrb.Spatial3DView(
                origin=f"/closeups/{style.key}",
                name=style.display_name,
                line_grid=False,
                eye_controls=rrb.EyeControls3D(
                    position=(2.5, -3.4, 1.7),
                    look_target=(0.0, 0.0, 0.75),
                    eye_up=(0.0, 0.0, 1.0),
                ),
                overrides=_hidden_robot_debug_overrides(
                    rrb, f"closeups/{style.key}", (style.key,)
                ),
            )
            for style in METHOD_STYLES
        ),
        grid_columns=3,
        name="Nine synchronized G1 close-ups",
    )
    spatial = rrb.Tabs(main_spatial, closeups, name="Articulated G1 motion")
    metrics = rrb.Tabs(
        rrb.TimeSeriesView(origin="/metrics/rf_kpe_all", name="RF-KPE all"),
        rrb.TimeSeriesView(
            origin="/metrics/root_translation",
            name="Root translation · common scale",
        ),
        rrb.TimeSeriesView(origin="/metrics/root_yaw", name="Root yaw"),
        rrb.TimeSeriesView(origin="/metrics/ground_penetration", name="Ground penetration"),
        rrb.TimeSeriesView(origin="/metrics/artifact", name="Cause-triggered flags"),
        rrb.TimeSeriesView(origin="/metrics/solve_time", name="Per-frame solve time"),
        name="Measured metrics",
    )
    return rrb.Blueprint(
        rrb.Horizontal(spatial, metrics, column_shares=[3.0, 2.0]),
        rrb.TimePanel(expanded=True, timeline="time", fps=fps),
        rrb.SelectionPanel(expanded=False),
        collapse_panels=False,
    )


def _log_pose(
    rr: Any,
    entity_root: str,
    points: np.ndarray,
    edges: tuple[tuple[int, int], ...],
    style: MethodStyle,
    label: str,
    foot_indices: tuple[int, int] | None = None,
    foot_colors: list[tuple[int, int, int]] | None = None,
) -> None:
    rr.log(
        f"{entity_root}/bones",
        rr.LineStrips3D(_segments(points, edges), colors=style.color, radii=0.012),
    )
    rr.log(
        f"{entity_root}/joints",
        rr.Points3D(points, colors=style.color, radii=0.022),
    )
    label_position = points[0]
    rr.log(
        f"{entity_root}/label",
        rr.Points3D(
            [label_position + np.asarray([0.0, 0.0, 0.12])],
            colors=style.color,
            radii=0.001,
            labels=[label],
            show_labels=True,
        ),
    )
    if foot_indices is not None and foot_colors is not None:
        rr.log(
            f"{entity_root}/foot_state",
            rr.Points3D(points[list(foot_indices)], colors=foot_colors, radii=0.052),
        )


def _foot_colors(method: MethodVisualization, frame: int) -> list[tuple[int, int, int]]:
    colors: list[tuple[int, int, int]] = []
    for side in ("left", "right"):
        stance = bool(method.metrics[f"{side}_source_stance"][frame])
        speed = float(method.metrics[f"{side}_foot_speed_m_s"][frame])
        if stance and speed > 0.01:
            colors.append((244, 83, 72))
        elif stance:
            colors.append((64, 196, 117))
        else:
            colors.append((126, 136, 151))
    return colors


def _artifact_label(method: MethodVisualization, frame: int) -> str:
    causes = []
    if bool(method.metrics["invalid_artifact"][frame]):
        causes.append("INVALID")
    if bool(method.metrics["ground_penetration_artifact"][frame]):
        causes.append("PENETRATION")
    if bool(method.metrics["joint_limit_artifact"][frame]):
        causes.append("JOINT-LIMIT")
    if bool(method.metrics["left_foot_skating"][frame]):
        causes.append("SKATING-L")
    if bool(method.metrics["right_foot_skating"][frame]):
        causes.append("SKATING-R")
    return "+".join(causes)


def _frame_label(
    style: MethodStyle, method: MethodVisualization, frame: int
) -> str:
    """Return an entity label without assigning artifact status to references."""

    if not style.annotate_artifact_causes:
        return style.display_name
    cause = _artifact_label(method, frame)
    return style.display_name + (f" · {cause}" if cause else "")


def _log_series_styles(rr: Any, data: Stage1Visualization) -> None:
    series = (
        ("rf_kpe_all", "rf_kpe_all_m", False),
        ("root_translation", "root_translation_error_m", False),
        ("root_yaw", "root_yaw_error_rad", False),
        ("ground_penetration", "ground_penetration_depth_m", False),
        ("artifact", "artifact", True),
        ("solve_time", "solve_time_s", False),
    )
    for root, _, step in series:
        for style in _active_styles(data):
            if style.external_reference and root == "solve_time":
                continue
            rr.log(
                f"metrics/{root}/{style.key}",
                rr.SeriesLines(
                    colors=style.color,
                    names=style.display_name,
                    widths=2.0,
                    interpolation_mode="StepAfter" if step else "Linear",
                ),
                static=True,
            )


def _mesh_entity(view: str, index: int, visual: UrdfVisual) -> str:
    return f"{view}/g1_visual_meshes/{index:02d}_{visual.link_name}"


def _closeup_mesh_entity(key: str, index: int, visual: UrdfVisual) -> str:
    return f"closeups/{key}/g1_visual_meshes/{index:02d}_{visual.link_name}"


def _human_skin_entity(view: str) -> str:
    return f"{view}/source-human/skin"


def _log_human_skin_topology(rr: Any, data: Stage1Visualization) -> None:
    """Log immutable SMPL topology once; frame rows update only vertices."""

    for view in ("grid", "world", "root_frame"):
        rr.log(
            _human_skin_entity(view),
            rr.Mesh3D.from_fields(
                triangle_indices=data.human_skin.faces,
                albedo_factor=(205, 177, 151, 255),
            ),
            static=True,
        )


def _log_human_skin_frame(
    rr: Any,
    view: str,
    vertices: np.ndarray,
) -> None:
    rr.log(
        _human_skin_entity(view),
        rr.Mesh3D.from_fields(vertex_positions=vertices),
    )


def _log_g1_mesh_assets(rr: Any, data: Stage1Visualization) -> None:
    """Embed one copy of each G1 visual mesh per comparison coordinate space."""

    for view in VIEW_METHOD_KEYS:
        for index, visual in enumerate(data.robot_visuals):
            rr.log(
                _mesh_entity(view, index, visual),
                rr.Asset3D(path=visual.mesh_path, albedo_factor=visual.rgba),
                static=True,
            )
    for style in _active_styles(data):
        for index, visual in enumerate(data.robot_visuals):
            rr.log(
                _closeup_mesh_entity(style.key, index, visual),
                rr.Asset3D(path=visual.mesh_path, albedo_factor=visual.rgba),
                static=True,
            )


def _log_g1_mesh_instances(
    rr: Any,
    data: Stage1Visualization,
    view: str,
    transforms: dict[str, dict[str, np.ndarray]],
    frame: int,
    offsets: dict[str, tuple[float, float, float]] | None = None,
) -> None:
    method_keys = _active_view_method_keys(data, view)
    if not method_keys:
        return
    for index, visual in enumerate(data.robot_visuals):
        translations, rotations, scales = visual_instance_poses(
            visual,
            method_keys,
            transforms,
            frame,
            offsets,
        )
        rr.log(
            _mesh_entity(view, index, visual),
            rr.InstancePoses3D(
                translations=translations,
                mat3x3=rotations,
                scales=scales,
            ),
        )


def _log_closeup_mesh_instances(
    rr: Any,
    data: Stage1Visualization,
    key: str,
    transforms: dict[str, np.ndarray],
    frame: int,
) -> None:
    for index, visual in enumerate(data.robot_visuals):
        pose = transforms[visual.link_name][frame] @ visual.origin
        rr.log(
            _closeup_mesh_entity(key, index, visual),
            rr.InstancePoses3D(
                translations=[pose[:3, 3]],
                mat3x3=[pose[:3, :3]],
                scales=[visual.scale],
            ),
        )


def _log_static_scene(rr: Any, data: Stage1Visualization) -> None:
    for root in ("grid", "world", "root_frame", "seeds"):
        rr.log(root, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    for style in _active_styles(data):
        rr.log(
            f"closeups/{style.key}",
            rr.ViewCoordinates.RIGHT_HAND_Z_UP,
            static=True,
        )

    world_human = _world_aligned_human(data)
    human_root_path = world_human[: data.frame_count, 0]
    rr.log(
        "world/source-human/root_path",
        rr.LineStrips3D([human_root_path], colors=SOURCE_STYLE.color, radii=0.008),
        static=True,
    )
    rr.log(
        "grid/source-human/root_path",
        rr.LineStrips3D(
            [human_root_path + np.asarray(LANE_OFFSETS["source-human"])],
            colors=SOURCE_STYLE.color,
            radii=0.008,
        ),
        static=True,
    )
    for style in _active_styles(data):
        method = data.methods[style.key]
        world = _world_aligned_robot(method)
        root_path = world[:, ROBOT_SEMANTIC_INDEX["root"]]
        rr.log(
            f"grid/{style.key}/root_path",
            rr.LineStrips3D(
                [root_path + np.asarray(LANE_OFFSETS[style.key])],
                colors=style.color,
                radii=0.008,
            ),
            static=True,
        )
        if style.operating_point or style.external_reference:
            rr.log(
                f"world/{style.key}/root_path",
                rr.LineStrips3D([root_path], colors=style.color, radii=0.008),
                static=True,
            )

    summary = "\n".join(
        [
            "# Human-to-G1 Stage 1 · Rerun comparison",
            "",
            f"- Sequence: `{data.sequence['sequence_id']}`",
            f"- Frames: {data.frame_count}",
            "- Shared comparison window: source frames [0, 450)",
            "- ProtoMotions v3 official native contract: 450/450; full-source coverage: 450/600",
            "- No visualization padding, interpolation, or trajectory stitching",
            f"- FPS: {data.human.fps:.6f}",
            f"- Human display scale: {data.human_scale:.6f}",
            "- LAFAN1 source format: 22-joint BVH (not native SMPL/SMPL-X)",
            (
                "- Human surface: visualization-only neutral SMPL fit; "
                f"root-aligned MPJPE {float(data.human_skin.motion.metadata['root_aligned_mpjpe_m']) * 1000.0:.1f} mm"
            ),
            f"- Loaded G1 trajectories: {len(data.methods)}",
            "- External reference: Unitree-attributed corpus (not verified ground truth)",
            "- Green feet: frozen source stance without skating",
            "- Red feet: source stance with target foot speed > 1 cm/s",
            f"- G1 appearance: {len(data.robot_visuals)} articulated Holosoma URDF visual meshes",
            "- Side-by-side view preserves root displacement and ground height",
            "- Root-frame view removes each motion's root translation and yaw",
            "- Nine close-ups show one articulated G1 trajectory per panel",
            "- Robot bones/joints are quantitative helpers hidden by default",
            "- Source BVH bones/joints remain available but are hidden by default",
            "",
            "The visualization replays measured canonical outputs; it does not rerun a retargeter.",
        ]
    )
    rr.log("metadata/readme", rr.TextDocument(summary, media_type="text/markdown"), static=True)
    if not data.acceptance_evidence:
        rr.log(
            "metadata/diagnostic_snapshot",
            rr.TextDocument(
                "# Diagnostic snapshot\n\n"
                "This recording visualizes the explicitly frozen current-completed "
                "outputs. It is not Stage 1 acceptance evidence and does not imply "
                "that publication-grade reruns or timing repetitions are complete.",
                media_type="text/markdown",
            ),
            static=True,
        )
    if data.missing_methods:
        skipped = "\n".join(
            [
                "# Diagnostic-only incomplete recording",
                "",
                "The following registered methods were explicitly skipped:",
                *[f"- `{key}`: {reason}" for key, reason in data.missing_methods.items()],
                "",
                "This recording is not complete Stage 1 visualization evidence.",
            ]
        )
        rr.log(
            "metadata/missing_methods",
            rr.TextDocument(skipped, media_type="text/markdown"),
            static=True,
        )
    if "unitree-reference" in data.methods:
        rr.log(
            "metadata/unitree_reference",
            rr.TextDocument(
                "# Unitree-attributed external reference\n\n"
                "This trajectory is a coordinate-canonicalized external reference corpus "
                "entry. It is not treated as verified ground truth, an optimization "
                "method, or a runtime operating point. Per-frame evaluator signals remain "
                "available in the metric tabs, but the robot label never calls the "
                "reference an artifact.",
                media_type="text/markdown",
            ),
            static=True,
        )
    _log_human_skin_topology(rr, data)
    _log_g1_mesh_assets(rr, data)


def _rerun_cli_prefix() -> tuple[str, ...]:
    configured = os.environ.get("RTCMP_RERUN_CLI")
    if configured:
        prefix = tuple(shlex.split(configured))
        if not prefix:
            raise RuntimeError("RTCMP_RERUN_CLI is empty")
        return prefix
    executable = shutil.which("rerun")
    if executable:
        return (executable,)
    for candidate in (
        Path.home() / "anaconda3/envs/vis/bin/rerun",
        Path.home() / "miniconda3/envs/vis/bin/rerun",
    ):
        if candidate.is_file():
            return (str(candidate),)
    conda = shutil.which("conda")
    if conda:
        return (conda, "run", "-n", "vis", "rerun")
    raise RuntimeError(
        "Rerun CLI is required for `rerun rrd verify`; activate the vis environment "
        "or set RTCMP_RERUN_CLI"
    )


def _run_rerun_cli(arguments: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    command = (*_rerun_cli_prefix(), *arguments)
    environment = dict(os.environ)
    # Analytics are irrelevant to local artifact verification and can attempt
    # to write outside a restricted workspace.
    environment.setdefault("RERUN_ANALYTICS", "disabled")
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def _number_from_rerun(value: str) -> int:
    digits = "".join(character for character in value if character.isdigit())
    if not digits:
        raise ValueError(f"Rerun statistic is not an integer: {value!r}")
    return int(digits)


def _parse_rerun_stats(text: str) -> dict[str, Any]:
    scalars: dict[str, int] = {}
    entities: dict[str, int] = {}
    components: dict[str, int] = {}
    section = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "Num chunks per entity":
            section = "entities"
            continue
        if stripped == "Num chunks per component":
            section = "components"
            continue
        if stripped and set(stripped) == {"-"}:
            continue
        match = re.fullmatch(r"(num_[a-z_]+)\s*=\s*([\d\s,_\u2009]+)(?:\s.*)?", stripped)
        if match:
            scalars.setdefault(
                match.group(1), _number_from_rerun(match.group(2))
            )
            continue
        pair = re.fullmatch(r"(.+):\s*([\d\s,._\u2009]+)", stripped)
        if not pair:
            if stripped and not stripped.startswith("("):
                section = ""
            continue
        if section == "entities" and pair.group(1).startswith("/"):
            entities[pair.group(1)] = _number_from_rerun(pair.group(2))
        elif section == "components":
            components[pair.group(1)] = _number_from_rerun(pair.group(2))
    return {"scalars": scalars, "entities": entities, "components": components}


def _rrd_entity_rows_and_components(recording: Path, entity: str) -> tuple[int, str]:
    printed = _run_rerun_cli(("rrd", "print", "-vv", "--entity", entity, str(recording)))
    if printed.returncode != 0:
        raise RuntimeError(
            f"Rerun could not inspect {entity}: {printed.stderr.strip()}"
        )
    row_pattern = re.compile(
        r"Chunk\([^\n]+\) with\s+([\d\s,._\u2009]+)\s+rows[^\n]*"
        + r"-\s*"
        + re.escape(entity)
        + r"(?:\s+-)?\s*$",
        flags=re.MULTILINE,
    )
    rows = sum(_number_from_rerun(value) for value in row_pattern.findall(printed.stdout))
    if rows <= 0:
        raise ValueError(f"Rerun entity {entity} has no inspectable rows")
    return rows, printed.stdout


def inspect_rerun_recording(
    recording: str | Path,
    *,
    frames: int,
    methods: tuple[str, ...] = EXPECTED_TRAJECTORY_KEYS,
    robot_visual_asset_count: int = 35,
) -> dict[str, Any]:
    """Run the Rerun CLI verifier and assert the scientific view structure."""

    path = Path(recording).resolve()
    if not path.is_file() or path.stat().st_size < 1_000_000:
        raise ValueError("Rerun recording is missing or implausibly small")
    verified = _run_rerun_cli(
        ("rrd", "verify", "--check-footers", "true", str(path))
    )
    if verified.returncode != 0:
        raise ValueError(
            "`rerun rrd verify` rejected the recording: "
            + (verified.stderr.strip() or verified.stdout.strip())
        )
    stats_process = _run_rerun_cli(("rrd", "stats", str(path)))
    if stats_process.returncode != 0:
        raise ValueError(
            "`rerun rrd stats` rejected the recording: "
            + (stats_process.stderr.strip() or stats_process.stdout.strip())
        )
    parsed = _parse_rerun_stats(stats_process.stdout)
    entities: dict[str, int] = parsed["entities"]
    components: dict[str, int] = parsed["components"]

    exact_grid_methods = {
        match.group(1)
        for entity in entities
        if (match := re.fullmatch(r"/grid/([^/]+)/bones", entity)) is not None
        and match.group(1) != "source-human"
    }
    if exact_grid_methods != set(methods):
        raise ValueError(
            f"Rerun grid trajectory set is {sorted(exact_grid_methods)}, "
            f"expected {sorted(methods)}"
        )
    exact_closeups = {
        match.group(1)
        for entity in entities
        if (
            match := re.fullmatch(
                r"/closeups/([^/]+)/g1_visual_meshes/00_[^/]+", entity
            )
        )
        is not None
    }
    if exact_closeups != set(methods):
        raise ValueError("Rerun does not contain exactly one G1 close-up per trajectory")

    # Asset link names come from the frozen URDF, so verify their exact count by
    # coordinate-space prefix rather than duplicating that asset list here.
    required_entities = {
        "/metadata/frame_marker",
        "/grid/source-human/skin",
        "/world/source-human/skin",
        "/root_frame/source-human/skin",
    }
    for view in VIEW_METHOD_KEYS:
        count = sum(
            entity.startswith(f"/{view}/g1_visual_meshes/") for entity in entities
        )
        if count != robot_visual_asset_count:
            raise ValueError(f"Rerun {view} contains {count} G1 mesh entities")
    for key in methods:
        count = sum(
            entity.startswith(f"/closeups/{key}/g1_visual_meshes/")
            for entity in entities
        )
        if count != robot_visual_asset_count:
            raise ValueError(f"Rerun close-up {key} contains {count} G1 mesh entities")
        required_entities.update(
            {
                f"/grid/{key}/bones",
                f"/grid/{key}/joints",
                f"/closeups/{key}/{key}/bones",
                f"/closeups/{key}/{key}/joints",
                f"/metrics/rf_kpe_all/{key}",
            }
        )
    missing = sorted(required_entities - set(entities))
    if missing:
        raise ValueError(f"Rerun required entities are missing: {missing}")

    required_components = {
        "Asset3D:blob",
        "Asset3D:media_type",
        "InstancePoses3D:translations",
        "InstancePoses3D:mat3x3",
        "InstancePoses3D:scales",
        "LineStrips3D:strips",
        "Mesh3D:triangle_indices",
        "Mesh3D:vertex_positions",
        "Points3D:positions",
        "Scalars:scalars",
    }
    if not required_components.issubset(components):
        raise ValueError(
            "Rerun component set is incomplete: "
            + ", ".join(sorted(required_components - set(components)))
        )

    marker_rows, marker_dump = _rrd_entity_rows_and_components(
        path, "/metadata/frame_marker"
    )
    if marker_rows != frames or "Scalars:scalars" not in marker_dump:
        raise ValueError(
            f"Rerun frame marker has {marker_rows} rows, expected {frames}"
        )
    human_skin_rows: dict[str, int] = {}
    for view in ("grid", "world", "root_frame"):
        entity = f"/{view}/source-human/skin"
        rows, dump = _rrd_entity_rows_and_components(path, entity)
        if rows != frames + 1 or not {
            "Mesh3D:triangle_indices",
            "Mesh3D:vertex_positions",
        }.issubset(set(re.findall(r"Mesh3D:[a-z0-9_]+", dump))):
            raise ValueError(
                f"Rerun {view} source-human skin does not contain static "
                "SMPL topology plus one vertex row per frame"
            )
        human_skin_rows[view] = rows
    view_instance_counts = {
        view: sum(key in methods for key in keys)
        for view, keys in VIEW_METHOD_KEYS.items()
    }
    representative_mesh_rows: dict[str, int] = {}
    for view, instance_count in view_instance_counts.items():
        representative_mesh = next(
            entity
            for entity in entities
            if entity.startswith(f"/{view}/g1_visual_meshes/00_")
        )
        mesh_rows, mesh_dump = _rrd_entity_rows_and_components(
            path, representative_mesh
        )
        representative_mesh_rows[view] = mesh_rows
        # InstancePoses3D is logged once per frame with one transform array
        # containing every robot in the view.  RRD rows therefore count
        # frames, not individual instances: one static Asset3D row plus one
        # batched pose row per frame.  The exact batch width is fixed by
        # ``view_instance_counts`` and the writer's active method registry.
        if mesh_rows != 1 + frames or not {
            "Asset3D:blob",
            "InstancePoses3D:translations",
        }.issubset(
            set(
                re.findall(
                    r"(?:Asset3D|InstancePoses3D):[a-z0-9_]+", mesh_dump
                )
            )
        ):
            raise ValueError(
                f"Rerun {view} mesh rows do not encode one batched "
                f"{instance_count}-robot pose per frame"
            )
    representative_closeup = next(
        entity
        for entity in entities
        if entity.startswith(f"/closeups/{methods[0]}/g1_visual_meshes/00_")
    )
    closeup_rows, closeup_dump = _rrd_entity_rows_and_components(
        path, representative_closeup
    )
    if closeup_rows != frames + 1 or "InstancePoses3D:translations" not in closeup_dump:
        raise ValueError("Rerun close-up does not contain one G1 instance per frame")
    representative_mesh_rows["closeup_per_method"] = closeup_rows
    metric_entity = f"/metrics/rf_kpe_all/{methods[0]}"
    metric_rows, metric_dump = _rrd_entity_rows_and_components(path, metric_entity)
    if metric_rows != frames + 1 or not {
        "SeriesLines:names",
        "Scalars:scalars",
    }.issubset(set(re.findall(r"(?:SeriesLines|Scalars):[a-z0-9_]+", metric_dump))):
        raise ValueError("Rerun metric entity lacks its style or full frame series")

    return {
        "result": "verified",
        "verify_command": "rerun rrd verify --check-footers true",
        "stats_command": "rerun rrd stats",
        "stats_sha256": hashlib.sha256(stats_process.stdout.encode()).hexdigest(),
        "num_chunks": parsed["scalars"].get("num_chunks"),
        "num_entity_paths": parsed["scalars"].get("num_entity_paths"),
        "num_rows": parsed["scalars"].get("num_rows"),
        "num_static": parsed["scalars"].get("num_static"),
        "frame_marker_rows": marker_rows,
        "representative_mesh_rows": representative_mesh_rows,
        "source_human_skin_rows": human_skin_rows,
        "representative_metric_rows": metric_rows,
        "verified_view_instance_counts": {
            **view_instance_counts,
            "closeups": len(methods),
        },
        "required_component_counts": {
            key: components[key] for key in sorted(required_components)
        },
        "exact_grid_trajectory_set": sorted(exact_grid_methods),
        "exact_closeup_trajectory_set": sorted(exact_closeups),
    }


def validate_rerun_manifest_contract(
    repo_root: str | Path,
    manifest: str | Path = "manifests/rerun_visualization.json",
    *,
    verify_recording: bool = True,
) -> dict[str, Any]:
    """Fail closed on stale paths, tampered bytes, or a fake/incomplete RRD."""

    root = Path(repo_root).resolve()
    manifest_path = Path(manifest)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or int(value.get("schema_version", 0)) != 6:
        raise ValueError("Rerun manifest schema v6 is required")
    if value.get("methods") != list(EXPECTED_TRAJECTORY_KEYS):
        raise ValueError("Rerun manifest does not contain the exact ordered trajectory set")
    expected_roles = {
        key: "external_reference" if key == "unitree-reference" else "retargeter"
        for key in EXPECTED_TRAJECTORY_KEYS
    }
    if value.get("method_roles") != expected_roles:
        raise ValueError("Rerun method/reference roles are stale")
    if (
        value.get("missing_methods") != {}
        or value.get("complete_registered_method_set") is not True
        or value.get("acceptance_evidence") is not True
        or value.get("snapshot_role") != "stage1_publication"
        or value.get("stage1_visualization_acceptance_eligible") is not True
    ):
        raise ValueError("Rerun manifest is diagnostic or incomplete")
    if value.get("rendering") != "articulated_g1_visual_meshes":
        raise ValueError("Rerun manifest does not declare articulated G1 meshes")
    if value.get("default_robot_rendering") != "full_articulated_g1_mesh":
        raise ValueError("Rerun default rendering is not the G1 mesh")
    if value.get("robot_debug_bones_and_joints_default_visible") is not False:
        raise ValueError("Rerun robot helper skeletons must be hidden by default")
    if value.get("source_human_rendering") != (
        "visualization_only_fitted_smpl_skin"
    ):
        raise ValueError("Rerun source human is not rendered as fitted SMPL")
    if value.get("source_human_debug_bones_and_joints_default_visible") is not False:
        raise ValueError("Rerun source helper skeleton must be hidden by default")
    if (
        value.get("canonical_source", {}).get("representation") != "LAFAN1_BVH"
        or value.get("canonical_source", {}).get("dataset_native_smpl") is not False
    ):
        raise ValueError("Rerun manifest misstates the original LAFAN representation")
    source_skin = value.get("source_human_skin", {})
    if (
        source_skin.get("dataset_native_smpl") is not False
        or int(source_skin.get("vertices", -1)) != 6890
        or int(source_skin.get("triangles", -1)) != 13776
        or not 0.0 < float(source_skin.get("root_aligned_mpjpe_m", -1.0)) < 0.2
    ):
        raise ValueError("Rerun fitted-SMPL skin evidence is incomplete")

    expected_view_keys = {
        **{view: list(keys) for view, keys in VIEW_METHOD_KEYS.items()},
        "closeups": list(CLOSEUP_METHOD_KEYS),
    }
    if value.get("view_method_keys") != expected_view_keys:
        raise ValueError("Rerun view trajectory membership is stale")
    if value.get("view_robot_instance_counts") != EXPECTED_VIEW_INSTANCE_COUNTS:
        raise ValueError("Rerun view instance counts are stale")
    expected_instances = sum(EXPECTED_VIEW_INSTANCE_COUNTS.values())
    if int(value.get("robot_instances_per_frame", -1)) != expected_instances:
        raise ValueError("Rerun logical robot instance count is incorrect")
    visual_count = int(value.get("robot_visual_asset_count", -1))
    if visual_count != 35 or len(value.get("robot_visual_assets", {})) != visual_count:
        raise ValueError("Rerun visual asset inventory is incomplete")
    if int(value.get("robot_mesh_entity_count", -1)) != visual_count * (
        len(VIEW_METHOD_KEYS) + len(CLOSEUP_METHOD_KEYS)
    ):
        raise ValueError("Rerun mesh entity inventory is inconsistent")

    def checked_path(relative: Any, digest: Any) -> Path:
        candidate = (root / str(relative)).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError(f"Rerun evidence path escapes the repository: {relative}")
        if not candidate.is_file() or sha256_file(candidate) != str(digest):
            raise ValueError(f"Rerun evidence hash mismatch: {relative}")
        return candidate

    output = checked_path(value.get("output"), value.get("output_sha256"))
    if output.stat().st_size != int(value.get("output_size_bytes", -1)):
        raise ValueError("Rerun recording size is stale")
    checked_path(
        value.get("canonical_source", {}).get("path"),
        value.get("canonical_source", {}).get("sha256"),
    )
    checked_path(source_skin.get("cache_path"), source_skin.get("cache_sha256"))
    checked_path(
        source_skin.get("evidence_path"), source_skin.get("evidence_sha256")
    )
    checked_path(value.get("canonical_urdf"), value.get("canonical_urdf_sha256"))
    checked_path(
        value.get("evaluator_robot", {}).get("path"),
        value.get("evaluator_robot", {}).get("sha256"),
    )
    evaluator = root / "manifests/evaluator.yaml"
    if value.get("evaluator_protocol_sha256") != sha256_file(evaluator):
        raise ValueError("Rerun evaluator hash is stale")
    for path, digest in value["robot_visual_assets"].items():
        checked_path(path, digest)

    sequence_id = str(value.get("sequence_id"))
    bindings = value.get("method_evidence_bindings", {})
    if set(bindings) != set(EXPECTED_TRAJECTORY_KEYS):
        raise ValueError("Rerun method evidence binding set is incomplete")
    for key in EXPECTED_TRAJECTORY_KEYS:
        binding = bindings[key]
        if (
            binding.get("verified") is not True
            or binding.get("run_directory_registry_key") != key
            or binding.get("run_directory") != STAGE1_RUN_DIRECTORIES[key]
        ):
            raise ValueError(f"Rerun binding registry is stale for {key}")
        without_digest = dict(binding)
        digest = without_digest.pop("binding_sha256", None)
        if digest != _canonical_json_sha256(without_digest):
            raise ValueError(f"Rerun binding digest is stale for {key}")
        output_binding = binding["output"]
        expected_output = _run_output_path(root, sequence_id, key)
        if (root / output_binding["path"]).resolve() != expected_output.resolve():
            raise ValueError(f"Rerun binding resolves a stale output for {key}")
        checked_path(output_binding["path"], output_binding["sha256"])
        if output_binding["sha256"] != value["method_outputs"][key]:
            raise ValueError(f"Rerun output digest maps disagree for {key}")
        metric_binding = binding["per_frame_metrics"]
        if (root / metric_binding["path"]).resolve() != _metric_path(root, key).resolve():
            raise ValueError(f"Rerun binding resolves stale metrics for {key}")
        checked_path(metric_binding["path"], metric_binding["sha256"])
        if metric_binding["sha256"] != value["method_metrics"][key]:
            raise ValueError(f"Rerun metric digest maps disagree for {key}")
        for section in (
            "metrics_summary",
            "publication_binding_table",
            "source",
            "evaluator",
            "robot",
        ):
            entry = binding[section]
            for path_key, digest_key in (
                ("path", "sha256"),
                ("canonical_urdf_path", "canonical_urdf_sha256"),
                ("evaluator_xml_path", "evaluator_xml_sha256"),
            ):
                if path_key in entry and digest_key in entry:
                    checked_path(entry[path_key], entry[digest_key])
        if binding["source"]["canonical_package_sha256"] != sha256_file(
            root / binding["source"]["path"]
        ):
            raise ValueError(f"Rerun source package binding is stale for {key}")
    if value.get("method_evidence_bundle_sha256") != _canonical_json_sha256(bindings):
        raise ValueError("Rerun evidence bundle digest is stale")

    frames = int(value.get("frames_logged", -1))
    if frames != SHARED_COMPARISON_FRAMES or int(value.get("stride", -1)) != 1:
        raise ValueError("Rerun timeline is incomplete")
    recorded_verification = value.get("rrd_verification", {})
    if recorded_verification.get("result") != "verified":
        raise ValueError("Rerun manifest lacks CLI verification evidence")
    if recorded_verification.get("verified_view_instance_counts") != (
        EXPECTED_VIEW_INSTANCE_COUNTS
    ):
        raise ValueError("Rerun CLI evidence does not prove exact view instance counts")
    if not verify_recording:
        return recorded_verification
    current = inspect_rerun_recording(
        output,
        frames=frames,
        methods=EXPECTED_TRAJECTORY_KEYS,
        robot_visual_asset_count=visual_count,
    )
    for key in (
        "frame_marker_rows",
        "representative_mesh_rows",
        "source_human_skin_rows",
        "representative_metric_rows",
        "verified_view_instance_counts",
        "exact_grid_trajectory_set",
        "exact_closeup_trajectory_set",
    ):
        if current.get(key) != recorded_verification.get(key):
            raise ValueError(f"Rerun verification evidence changed for {key}")
    return current


def write_rerun_recording(
    data: Stage1Visualization,
    output: str | Path | None,
    *,
    spawn: bool = False,
    max_frames: int | None = None,
    stride: int = 1,
) -> dict[str, Any]:
    """Write and optionally open a synchronized Rerun Stage 1 recording."""

    if output is None and not spawn:
        raise ValueError("At least one of output or spawn must be requested")
    if stride < 1:
        raise ValueError("stride must be at least one")
    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except ImportError as error:  # pragma: no cover - environment error path
        raise RuntimeError("Rerun SDK is required; use conda environment 'vis'") from error
    version = tuple(int(part) for part in rr.__version__.split(".")[:2])
    if version < (0, 34):
        raise RuntimeError(
            f"Rerun >= 0.34 is required; environment has {rr.__version__}. Use conda env 'vis'."
        )

    output_path = Path(output).resolve() if output is not None else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    blueprint = _blueprint(rrb, data.human.fps)
    rr.init(
        "human_to_g1_stage1_comparison",
        recording_id=_recording_id(data),
        strict=True,
        default_blueprint=blueprint,
    )
    if output_path is not None:
        rr.save(output_path, default_blueprint=blueprint)
    if spawn:
        rr.spawn(default_blueprint=blueprint)
    rr.send_blueprint(blueprint)
    _log_static_scene(rr, data)
    _log_series_styles(rr, data)

    human_edges = _human_edges(data.human)
    world_human = _world_aligned_human(data)
    world_human_skin = _world_aligned_human_skin(data)
    human_yaw = human_heading_yaw(data.human)
    root_human = root_frame_points(
        data.human.world_positions,
        data.human.root_translation,
        human_yaw,
    ) * data.human_scale
    root_human_skin = _root_frame_human_skin(data)
    world_methods = {key: _world_aligned_robot(method) for key, method in data.methods.items()}
    root_methods = {
        key: root_frame_points(
            method.positions,
            method.positions[:, ROBOT_SEMANTIC_INDEX["root"]],
            yaw_from_matrix(quaternion_wxyz_to_matrix(method.motion.qpos[:, 3:7])),
        )
        for key, method in data.methods.items()
    }
    world_link_transforms = {
        key: _world_aligned_link_transforms(method)
        for key, method in data.methods.items()
    }
    root_link_transforms = {
        key: _root_frame_link_transforms(method)
        for key, method in data.methods.items()
    }
    foot_indices = (
        ROBOT_SEMANTIC_INDEX["left_toe"],
        ROBOT_SEMANTIC_INDEX["right_toe"],
    )
    source_names = {name: index for index, name in enumerate(data.human.joint_names.astype(str))}
    source_feet = (source_names["LeftToe"], source_names["RightToe"])

    stop = data.frame_count if max_frames is None else min(max_frames, data.frame_count)
    frames = list(range(0, stop, stride))
    for frame in frames:
        rr.set_time("frame", sequence=frame)
        rr.set_time("time", duration=float(data.human.timestamps[frame]))
        rr.log("metadata/frame_marker", rr.Scalars(float(frame)))
        source_foot_colors = [
            (64, 196, 117) if bool(data.human.foot_contact_labels[frame, side]) else (126, 136, 151)
            for side in range(2)
        ]
        _log_pose(
            rr,
            "grid/source-human",
            world_human[frame] + np.asarray(LANE_OFFSETS["source-human"]),
            human_edges,
            SOURCE_STYLE,
            SOURCE_STYLE.display_name,
            source_feet,
            source_foot_colors,
        )
        _log_pose(
            rr,
            "world/source-human",
            world_human[frame],
            human_edges,
            SOURCE_STYLE,
            SOURCE_STYLE.display_name,
            source_feet,
            source_foot_colors,
        )
        _log_pose(
            rr,
            "root_frame/source-human",
            root_human[frame],
            human_edges,
            SOURCE_STYLE,
            SOURCE_STYLE.display_name,
            source_feet,
            source_foot_colors,
        )
        _log_human_skin_frame(
            rr,
            "grid",
            world_human_skin[frame] + np.asarray(LANE_OFFSETS["source-human"]),
        )
        _log_human_skin_frame(
            rr,
            "world",
            world_human_skin[frame],
        )
        _log_human_skin_frame(
            rr,
            "root_frame",
            root_human_skin[frame],
        )

        for style in _active_styles(data):
            method = data.methods[style.key]
            label = _frame_label(style, method, frame)
            feet = _foot_colors(method, frame)
            _log_pose(
                rr,
                f"grid/{style.key}",
                world_methods[style.key][frame] + np.asarray(LANE_OFFSETS[style.key]),
                ROBOT_BONE_INDICES,
                style,
                label,
                foot_indices,
                feet,
            )
            if style.operating_point or style.external_reference:
                _log_pose(
                    rr,
                    f"world/{style.key}",
                    world_methods[style.key][frame],
                    ROBOT_BONE_INDICES,
                    style,
                    label,
                    foot_indices,
                    feet,
                )
                _log_pose(
                    rr,
                    f"root_frame/{style.key}",
                    root_methods[style.key][frame],
                    ROBOT_BONE_INDICES,
                    style,
                    label,
                    foot_indices,
                    feet,
                )
            if style.key.startswith("sparse-"):
                _log_pose(
                    rr,
                    f"seeds/{style.key}",
                    root_methods[style.key][frame],
                    ROBOT_BONE_INDICES,
                    style,
                    label,
                    foot_indices,
                    feet,
                )
            _log_pose(
                rr,
                f"closeups/{style.key}/{style.key}",
                root_methods[style.key][frame],
                ROBOT_BONE_INDICES,
                style,
                label,
                foot_indices,
                feet,
            )
            _log_closeup_mesh_instances(
                rr,
                data,
                style.key,
                root_link_transforms[style.key],
                frame,
            )

            metric_paths = {
                "rf_kpe_all": "rf_kpe_all_m",
                "root_translation": "root_translation_error_m",
                "root_yaw": "root_yaw_error_rad",
                "ground_penetration": "ground_penetration_depth_m",
                "artifact": "artifact",
                "solve_time": "solve_time_s",
            }
            for metric_root, field in metric_paths.items():
                if style.external_reference and metric_root == "solve_time":
                    continue
                rr.log(
                    f"metrics/{metric_root}/{style.key}",
                    rr.Scalars(_method_metric(data, style.key, field, frame)),
                )

        _log_g1_mesh_instances(
            rr,
            data,
            "grid",
            world_link_transforms,
            frame,
            LANE_OFFSETS,
        )
        _log_g1_mesh_instances(
            rr,
            data,
            "world",
            world_link_transforms,
            frame,
        )
        _log_g1_mesh_instances(
            rr,
            data,
            "root_frame",
            root_link_transforms,
            frame,
        )
        _log_g1_mesh_instances(
            rr,
            data,
            "seeds",
            root_link_transforms,
            frame,
        )

    rr.disconnect()
    active_styles = _active_styles(data)
    view_method_keys = {
        view: _active_view_method_keys(data, view) for view in VIEW_METHOD_KEYS
    }
    closeup_method_keys = tuple(
        key for key in CLOSEUP_METHOD_KEYS if key in data.methods
    )
    result: dict[str, Any] = {
        "schema_version": 6,
        "sequence_id": data.sequence["sequence_id"],
        "source_sha256": data.human.source_sha256,
        "canonical_source": {
            "path": data.source_path.relative_to(data.repo_root).as_posix(),
            "sha256": sha256_file(data.source_path),
            "representation": "LAFAN1_BVH",
            "dataset_native_smpl": False,
        },
        "source_human_rendering": "visualization_only_fitted_smpl_skin",
        "source_human_debug_bones_and_joints_default_visible": False,
        "source_human_skin": {
            "cache_path": data.human_skin.cache_path.relative_to(
                data.repo_root
            ).as_posix(),
            "cache_sha256": sha256_file(data.human_skin.cache_path),
            "evidence_path": data.human_skin.evidence_path.relative_to(
                data.repo_root
            ).as_posix(),
            "evidence_sha256": sha256_file(data.human_skin.evidence_path),
            "body_model_sha256": data.human_skin.motion.metadata[
                "body_model_sha256"
            ],
            "vertices": int(data.human_skin.vertices.shape[1]),
            "triangles": int(len(data.human_skin.faces)),
            "root_aligned_mpjpe_m": float(
                data.human_skin.motion.metadata["root_aligned_mpjpe_m"]
            ),
            "dataset_native_smpl": False,
        },
        "evaluator_protocol_sha256": sha256_file(
            data.evaluator_path
        ),
        "evaluator_robot": {
            "path": data.evaluator_robot_path.relative_to(data.repo_root).as_posix(),
            "sha256": sha256_file(data.evaluator_robot_path),
        },
        "rerun_version": rr.__version__,
        "recording_id": _recording_id(data),
        "frames_logged": len(frames),
        "frame_start": frames[0] if frames else None,
        "frame_end_inclusive": frames[-1] if frames else None,
        "stride": stride,
        "methods": [style.key for style in active_styles],
        "method_roles": {
            style.key: "external_reference" if style.external_reference else "retargeter"
            for style in active_styles
        },
        "missing_methods": data.missing_methods,
        "complete_registered_method_set": not data.missing_methods,
        "acceptance_evidence": data.acceptance_evidence,
        "snapshot_role": data.snapshot_role,
        "stage1_visualization_acceptance_eligible": (
            data.acceptance_evidence
            and
            not data.missing_methods
            and tuple(style.key for style in active_styles)
            == EXPECTED_TRAJECTORY_KEYS
            and frames == list(range(data.frame_count))
            and stride == 1
        ),
        "method_evidence_bindings": {
            style.key: data.methods[style.key].evidence_binding
            for style in active_styles
        },
        "method_outputs": {
            style.key: sha256_file(data.methods[style.key].output_path)
            for style in active_styles
        },
        "method_metrics": {
            style.key: sha256_file(data.methods[style.key].metrics_path)
            for style in active_styles
        },
        "canonical_urdf": str(data.urdf_path.relative_to(data.repo_root)),
        "canonical_urdf_sha256": sha256_file(data.urdf_path),
        "rendering": "articulated_g1_visual_meshes",
        "default_robot_rendering": "full_articulated_g1_mesh",
        "robot_debug_bones_and_joints_default_visible": False,
        "robot_visual_asset_count": len(data.robot_visuals),
        "robot_visual_assets": {
            visual.mesh_path.relative_to(data.repo_root).as_posix(): sha256_file(
                visual.mesh_path
            )
            for visual in data.robot_visuals
        },
        "view_method_keys": {
            **{view: list(keys) for view, keys in view_method_keys.items()},
            "closeups": list(closeup_method_keys),
        },
        "view_robot_instance_counts": {
            **{view: len(keys) for view, keys in view_method_keys.items()},
            "closeups": len(closeup_method_keys),
        },
        "robot_instances_per_frame": (
            sum(len(keys) for keys in view_method_keys.values())
            + len(closeup_method_keys)
        ),
        "robot_mesh_entity_count": len(data.robot_visuals)
        * (len(VIEW_METHOD_KEYS) + len(closeup_method_keys)),
        "output": str(output_path) if output_path is not None else None,
    }
    result["method_evidence_bundle_sha256"] = _canonical_json_sha256(
        result["method_evidence_bindings"]
    )
    if output_path is not None:
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError(f"Rerun recording was not written: {output_path}")
        result["output_size_bytes"] = output_path.stat().st_size
        result["output_sha256"] = sha256_file(output_path)
        result["rrd_verification"] = inspect_rerun_recording(
            output_path,
            frames=len(frames),
            methods=tuple(style.key for style in active_styles),
            robot_visual_asset_count=len(data.robot_visuals),
        )
        atomic_write_json(output_path.with_suffix(output_path.suffix + ".json"), result)
    return result


def visualize_stage1(
    repo_root: str | Path = ".",
    sequence_manifest: str | Path = "manifests/pilot_sequence.yaml",
    output: str | Path | None = "artifacts/visualization/stage1_comparison.rrd",
    manifest: str | Path | None = "manifests/rerun_visualization.json",
    *,
    spawn: bool = False,
    max_frames: int | None = None,
    stride: int = 1,
    skip_missing_methods: bool = False,
) -> dict[str, Any]:
    data = load_stage1_visualization(
        repo_root,
        sequence_manifest,
        skip_missing_methods=skip_missing_methods,
    )
    root = Path(repo_root).resolve()
    if data.missing_methods and manifest is not None:
        requested_manifest = Path(manifest)
        if not requested_manifest.is_absolute():
            requested_manifest = root / requested_manifest
        acceptance_manifest = root / "manifests" / "rerun_visualization.json"
        if requested_manifest.resolve() == acceptance_manifest.resolve():
            raise ValueError(
                "A diagnostic recording with skipped methods cannot write the "
                "Stage 1 acceptance manifest. Set manifest=None or choose a "
                "separate diagnostic manifest path."
            )
    output_path: Path | None = None
    if output is not None:
        output_path = Path(output)
        if not output_path.is_absolute():
            output_path = root / output_path
    result = write_rerun_recording(
        data,
        output_path,
        spawn=spawn,
        max_frames=max_frames,
        stride=stride,
    )
    if manifest is not None:
        manifest_path = Path(manifest)
        if not manifest_path.is_absolute():
            manifest_path = root / manifest_path
        portable = dict(result)
        if output_path is not None:
            try:
                portable["output"] = output_path.relative_to(root).as_posix()
            except ValueError:
                portable["output"] = str(output_path)
        atomic_write_json(manifest_path, portable)
    return result


def visualize_current_diagnostic(
    repo_root: str | Path = ".",
    snapshot_manifest: str | Path = (
        "artifacts/visualization/"
        "stage1_current_completed_9methods.manifest.json"
    ),
    output: str | Path = (
        "artifacts/visualization/"
        "stage1_current_completed_9methods_smpl.rrd"
    ),
    manifest: str | Path = (
        "artifacts/visualization/"
        "stage1_current_completed_9methods_smpl.manifest.json"
    ),
    *,
    spawn: bool = False,
    max_frames: int | None = None,
    stride: int = 1,
) -> dict[str, Any]:
    """Rebuild the prior nine-method diagnostic with a fitted SMPL source skin."""

    data = load_current_diagnostic_visualization(repo_root, snapshot_manifest)
    root = Path(repo_root).resolve()
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = root / output_path
    manifest_path = Path(manifest)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    acceptance_manifest = root / "manifests" / "rerun_visualization.json"
    if manifest_path.resolve() == acceptance_manifest.resolve():
        raise ValueError(
            "A diagnostic snapshot cannot write the Stage 1 acceptance manifest"
        )
    result = write_rerun_recording(
        data,
        output_path,
        spawn=spawn,
        max_frames=max_frames,
        stride=stride,
    )
    portable = dict(result)
    try:
        portable["output"] = output_path.relative_to(root).as_posix()
    except ValueError:
        portable["output"] = str(output_path)
    atomic_write_json(manifest_path, portable)
    return result
