"""Rerun visualization for the frozen Stage 1 human-to-G1 comparison.

The viewer intentionally reads only the canonical on-disk contracts. Robot
forward kinematics uses the frozen G1 URDF so the ``vis`` environment does not
need MuJoCo or any method-specific dependencies.
"""

from __future__ import annotations

import csv
import hashlib
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .constants import G1_JOINT_NAMES
from .io_utils import atomic_write_json, load_yaml, sha256_file
from .rotations import quaternion_wxyz_to_matrix, yaw_from_matrix
from .schemas import CanonicalG1, CanonicalHuman


@dataclass(frozen=True)
class MethodStyle:
    key: str
    display_name: str
    color: tuple[int, int, int]
    operating_point: bool


METHOD_STYLES = (
    MethodStyle("sparse-neutral", "Sparse · neutral", (238, 106, 73), True),
    MethodStyle("dense", "Dense", (42, 157, 111), True),
    MethodStyle("gmr", "GMR", (225, 174, 49), True),
    MethodStyle("omniretarget", "OmniRetarget", (132, 113, 238), True),
    MethodStyle("sparse-a", "Sparse · seed A", (47, 169, 209), False),
    MethodStyle("sparse-b", "Sparse · seed B", (214, 91, 156), False),
)

SOURCE_STYLE = MethodStyle("source-human", "Source human", (230, 234, 241), True)

ROBOT_SEMANTIC_LINKS = {
    "root": "pelvis",
    "torso": "torso_link",
    "head": "head_link",
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

        joints = []
        actuated_names = []
        for element in robot.findall("joint"):
            name = element.attrib["name"]
            joint_type = element.attrib["type"]
            parent = element.find("parent").attrib["link"]  # type: ignore[union-attr]
            child = element.find("child").attrib["link"]  # type: ignore[union-attr]
            origin_element = element.find("origin")
            xyz = _parse_vector(
                origin_element.attrib.get("xyz", "0 0 0") if origin_element is not None else "0 0 0"
            )
            rpy = _parse_vector(
                origin_element.attrib.get("rpy", "0 0 0") if origin_element is not None else "0 0 0"
            )
            origin = np.eye(4, dtype=np.float64)
            origin[:3, :3] = _rpy_matrix(rpy)
            origin[:3, 3] = xyz
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

    def semantic_positions(self, qpos: np.ndarray) -> dict[str, np.ndarray]:
        value = np.asarray(qpos, dtype=np.float64)
        if value.shape != (36,):
            raise ValueError("Canonical G1 qpos must have shape [36]")
        points = self.motion_positions(value[None])[0]
        return {name: points[index] for index, name in enumerate(ROBOT_SEMANTICS)}

    def motion_positions(self, qpos: np.ndarray) -> np.ndarray:
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

        result = np.empty((frames, len(ROBOT_SEMANTICS), 3), dtype=np.float64)
        for semantic, link in ROBOT_SEMANTIC_LINKS.items():
            if semantic == "head":
                continue
            result[:, ROBOT_SEMANTIC_INDEX[semantic]] = transforms[link][:, :3, 3]
        torso = transforms["torso_link"]
        result[:, ROBOT_SEMANTIC_INDEX["head"]] = (
            torso[:, :3, 3] + torso[:, :3, :3] @ np.asarray([0.0, 0.0, 0.35])
        )
        return result


@dataclass
class MethodVisualization:
    style: MethodStyle
    motion: CanonicalG1
    positions: np.ndarray
    metrics: dict[str, np.ndarray]


@dataclass
class Stage1Visualization:
    repo_root: Path
    sequence: dict[str, Any]
    human: CanonicalHuman
    human_scale: float
    methods: dict[str, MethodVisualization]
    urdf_path: Path

    @property
    def frame_count(self) -> int:
        return len(self.human.timestamps)


def _parse_metric(value: str) -> float:
    normalized = value.strip().lower()
    if normalized in {"true", "false"}:
        return float(normalized == "true")
    return float(value)


def _load_metrics(path: Path, frame_count: int) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Per-frame evaluation metrics are missing: {path}")
    result = {name: np.full(frame_count, np.nan, dtype=np.float64) for name in METRIC_FIELDS}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            frame = int(row["source_frame_idx"])
            if not 0 <= frame < frame_count:
                raise ValueError(f"Metric frame {frame} is outside the source timeline")
            for field in METRIC_FIELDS:
                result[field][frame] = _parse_metric(row[field])
    return result


def _height_scale(human: CanonicalHuman, robot_points: np.ndarray) -> float:
    names = {name: index for index, name in enumerate(human.joint_names.astype(str))}
    required = ("Head", "LeftToe", "RightToe")
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f"Canonical human is missing visualization joints: {missing}")
    human_foot = 0.5 * (
        human.world_positions[0, names["LeftToe"]]
        + human.world_positions[0, names["RightToe"]]
    )
    robot_foot = 0.5 * (
        robot_points[0, ROBOT_SEMANTIC_INDEX["left_toe"]]
        + robot_points[0, ROBOT_SEMANTIC_INDEX["right_toe"]]
    )
    human_height = np.linalg.norm(human.world_positions[0, names["Head"]] - human_foot)
    robot_height = np.linalg.norm(
        robot_points[0, ROBOT_SEMANTIC_INDEX["head"]] - robot_foot
    )
    if human_height < 0.5 or robot_height < 0.5:
        raise ValueError("Implausible source or robot height in visualization")
    return float(robot_height / human_height)


def load_stage1_visualization(
    repo_root: str | Path = ".",
    sequence_manifest: str | Path = "manifests/pilot_sequence.yaml",
) -> Stage1Visualization:
    root = Path(repo_root).resolve()
    sequence_path = Path(sequence_manifest)
    if not sequence_path.is_absolute():
        sequence_path = root / sequence_path
    sequence = load_yaml(sequence_path)
    if sequence.get("full_lafan_authorized") is not False:
        raise RuntimeError("Visualization requires the frozen Stage 1 Full-LAFAN hard stop")
    human = CanonicalHuman.load(root / sequence["canonical_path"])
    stage = load_yaml(root / "configs" / "stage1.yaml")
    urdf_value = stage["canonical_robot"].get("urdf")
    if not urdf_value:
        raise ValueError("configs/stage1.yaml must declare canonical_robot.urdf")
    urdf_path = root / urdf_value
    kinematics = UrdfSemanticKinematics(urdf_path)

    sequence_id = sequence["sequence_id"]
    methods: dict[str, MethodVisualization] = {}
    for style in METHOD_STYLES:
        path = root / "runs" / sequence_id / style.key / "canonical_g1.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Required Stage 1 visualization output is missing: {path}")
        motion = CanonicalG1.load(path)
        motion.validate(source_frame_count=len(human.timestamps))
        if not np.array_equal(motion.source_frame_idx, np.arange(len(human.timestamps))):
            raise ValueError(f"{style.key} does not cover the complete source frame timeline")
        positions = kinematics.motion_positions(motion.qpos)
        metrics = _load_metrics(root / "metrics" / "runs" / f"{style.key}_per_frame.csv", len(human.timestamps))
        if not all(np.isfinite(value).all() for value in metrics.values()):
            raise ValueError(f"{style.key} visualization metrics contain missing or non-finite values")
        methods[style.key] = MethodVisualization(style, motion, positions, metrics)

    reference = methods["sparse-neutral"].positions
    return Stage1Visualization(
        repo_root=root,
        sequence=sequence,
        human=human,
        human_scale=_height_scale(human, reference),
        methods=methods,
        urdf_path=urdf_path,
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


def _world_aligned_human(data: Stage1Visualization) -> np.ndarray:
    reference_root = data.methods["sparse-neutral"].positions[0, ROBOT_SEMANTIC_INDEX["root"]]
    aligned_origin = np.asarray([0.0, 0.0, reference_root[2]])
    return (
        (data.human.world_positions - data.human.root_translation[0, None, :])
        * data.human_scale
        + aligned_origin[None, None, :]
    )


def _human_edges(human: CanonicalHuman) -> tuple[tuple[int, int], ...]:
    return tuple((int(parent), child) for child, parent in enumerate(human.parent_indices) if parent >= 0)


def _segments(points: np.ndarray, edges: tuple[tuple[int, int], ...]) -> list[np.ndarray]:
    return [points[[parent, child]] for parent, child in edges]


def _recording_id(data: Stage1Visualization) -> str:
    digest = hashlib.sha256(data.human.source_sha256.encode())
    for style in METHOD_STYLES:
        path = data.repo_root / "runs" / data.sequence["sequence_id"] / style.key / "canonical_g1.npz"
        digest.update(bytes.fromhex(sha256_file(path)))
    return str(uuid.UUID(bytes=digest.digest()[:16]))


def _method_metric(data: Stage1Visualization, key: str, field: str, frame: int) -> float:
    return float(data.methods[key].metrics[field][frame])


def _blueprint(rrb: Any, fps: float) -> Any:
    spatial = rrb.Tabs(
        rrb.Spatial3DView(origin="/grid", name="Side-by-side · world motion", line_grid=True),
        rrb.Spatial3DView(origin="/world", name="World overlay · root tracking", line_grid=True),
        rrb.Spatial3DView(origin="/root_frame", name="Root-frame pose overlay", line_grid=False),
        rrb.Spatial3DView(origin="/seeds", name="Sparse seed ambiguity", line_grid=False),
        name="Motion views",
    )
    metrics = rrb.Tabs(
        rrb.TimeSeriesView(origin="/metrics/rf_kpe_all", name="RF-KPE all"),
        rrb.TimeSeriesView(origin="/metrics/root_translation", name="Root translation"),
        rrb.TimeSeriesView(origin="/metrics/root_yaw", name="Root yaw"),
        rrb.TimeSeriesView(origin="/metrics/ground_penetration", name="Ground penetration"),
        rrb.TimeSeriesView(origin="/metrics/artifact", name="Artifact flags"),
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
        for style in METHOD_STYLES:
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


def _log_static_scene(rr: Any, data: Stage1Visualization) -> None:
    for root in ("grid", "world", "root_frame", "seeds"):
        rr.log(root, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    world_human = _world_aligned_human(data)
    human_root_path = world_human[:, 0]
    rr.log(
        "world/source-human/root_path",
        rr.LineStrips3D([human_root_path], colors=SOURCE_STYLE.color, radii=0.008),
        static=True,
    )
    lane_offsets = {
        "source-human": np.asarray([-6.0, 3.0, 0.0]),
        "sparse-neutral": np.asarray([-2.0, 3.0, 0.0]),
        "dense": np.asarray([2.0, 3.0, 0.0]),
        "gmr": np.asarray([6.0, 3.0, 0.0]),
        "omniretarget": np.asarray([-4.0, -3.0, 0.0]),
        "sparse-a": np.asarray([0.0, -3.0, 0.0]),
        "sparse-b": np.asarray([4.0, -3.0, 0.0]),
    }
    rr.log(
        "grid/source-human/root_path",
        rr.LineStrips3D(
            [human_root_path + lane_offsets["source-human"]],
            colors=SOURCE_STYLE.color,
            radii=0.008,
        ),
        static=True,
    )
    for style in METHOD_STYLES:
        method = data.methods[style.key]
        world = _world_aligned_robot(method)
        root_path = world[:, ROBOT_SEMANTIC_INDEX["root"]]
        rr.log(
            f"grid/{style.key}/root_path",
            rr.LineStrips3D(
                [root_path + lane_offsets[style.key]], colors=style.color, radii=0.008
            ),
            static=True,
        )
        if style.operating_point:
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
            f"- FPS: {data.human.fps:.6f}",
            f"- Human display scale: {data.human_scale:.6f}",
            "- Green feet: frozen source stance without skating",
            "- Red feet: source stance with target foot speed > 1 cm/s",
            "- Side-by-side view preserves root displacement and ground height",
            "- Root-frame view removes each motion's root translation and yaw",
            "",
            "The visualization replays measured canonical outputs; it does not rerun a retargeter.",
        ]
    )
    rr.log("metadata/readme", rr.TextDocument(summary, media_type="text/markdown"), static=True)


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
    human_yaw = yaw_from_matrix(quaternion_wxyz_to_matrix(data.human.world_rotations[:, 0]))
    root_human = root_frame_points(
        data.human.world_positions,
        data.human.root_translation,
        human_yaw,
    ) * data.human_scale
    world_methods = {key: _world_aligned_robot(method) for key, method in data.methods.items()}
    root_methods = {
        key: root_frame_points(
            method.positions,
            method.positions[:, ROBOT_SEMANTIC_INDEX["root"]],
            yaw_from_matrix(quaternion_wxyz_to_matrix(method.motion.qpos[:, 3:7])),
        )
        for key, method in data.methods.items()
    }
    lane_offsets = {
        "source-human": np.asarray([-6.0, 3.0, 0.0]),
        "sparse-neutral": np.asarray([-2.0, 3.0, 0.0]),
        "dense": np.asarray([2.0, 3.0, 0.0]),
        "gmr": np.asarray([6.0, 3.0, 0.0]),
        "omniretarget": np.asarray([-4.0, -3.0, 0.0]),
        "sparse-a": np.asarray([0.0, -3.0, 0.0]),
        "sparse-b": np.asarray([4.0, -3.0, 0.0]),
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
        source_foot_colors = [
            (64, 196, 117) if bool(data.human.foot_contact_labels[frame, side]) else (126, 136, 151)
            for side in range(2)
        ]
        _log_pose(
            rr,
            "grid/source-human",
            world_human[frame] + lane_offsets["source-human"],
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

        for style in METHOD_STYLES:
            method = data.methods[style.key]
            artifact = bool(method.metrics["artifact"][frame])
            label = style.display_name + (" · ARTIFACT" if artifact else "")
            feet = _foot_colors(method, frame)
            _log_pose(
                rr,
                f"grid/{style.key}",
                world_methods[style.key][frame] + lane_offsets[style.key],
                ROBOT_BONE_INDICES,
                style,
                label,
                foot_indices,
                feet,
            )
            if style.operating_point:
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

            metric_paths = {
                "rf_kpe_all": "rf_kpe_all_m",
                "root_translation": "root_translation_error_m",
                "root_yaw": "root_yaw_error_rad",
                "ground_penetration": "ground_penetration_depth_m",
                "artifact": "artifact",
                "solve_time": "solve_time_s",
            }
            for metric_root, field in metric_paths.items():
                rr.log(
                    f"metrics/{metric_root}/{style.key}",
                    rr.Scalars(_method_metric(data, style.key, field, frame)),
                )

    rr.disconnect()
    result: dict[str, Any] = {
        "schema_version": 1,
        "sequence_id": data.sequence["sequence_id"],
        "source_sha256": data.human.source_sha256,
        "rerun_version": rr.__version__,
        "recording_id": _recording_id(data),
        "frames_logged": len(frames),
        "frame_start": frames[0] if frames else None,
        "frame_end_inclusive": frames[-1] if frames else None,
        "stride": stride,
        "methods": [style.key for style in METHOD_STYLES],
        "method_outputs": {
            style.key: sha256_file(
                data.repo_root
                / "runs"
                / data.sequence["sequence_id"]
                / style.key
                / "canonical_g1.npz"
            )
            for style in METHOD_STYLES
        },
        "canonical_urdf": str(data.urdf_path.relative_to(data.repo_root)),
        "canonical_urdf_sha256": sha256_file(data.urdf_path),
        "output": str(output_path) if output_path is not None else None,
    }
    if output_path is not None:
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError(f"Rerun recording was not written: {output_path}")
        result["output_size_bytes"] = output_path.stat().st_size
        result["output_sha256"] = sha256_file(output_path)
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
) -> dict[str, Any]:
    data = load_stage1_visualization(repo_root, sequence_manifest)
    output_path: Path | None = None
    if output is not None:
        output_path = Path(output)
        if not output_path.is_absolute():
            output_path = Path(repo_root).resolve() / output_path
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
            manifest_path = Path(repo_root).resolve() / manifest_path
        portable = dict(result)
        if output_path is not None:
            try:
                portable["output"] = output_path.relative_to(Path(repo_root).resolve()).as_posix()
            except ValueError:
                portable["output"] = str(output_path)
        atomic_write_json(manifest_path, portable)
    return result
