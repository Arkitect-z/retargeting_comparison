"""Capture frozen native pre-solver targets without invoking any solver.

This module exists to keep two scientifically different objects separate:

* an ``exact_solver_input`` must be observed at the formal solver boundary and
  is the tensor handed to (or immediately consumed by) the optimizer;
* a ``normalized_projection`` is a benchmark-side semantic projection used for
  controlled comparisons and is not evidence of a public method's native input.

Offline replay/reconstruction remains useful, but it is accepted as exact only
when its tensor hash is equal to the hash recorded during a completed formal
solver run.  Reconstruction alone is never presented as runtime observation.

The worker entry points below only execute loaders, deterministic preprocessing,
task-target setters, and interaction-mesh construction.  They never call
``solve_ik``, ``retarget``, ``retarget_motion``, or ``solve_retargeting``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from .io_utils import atomic_write_json, load_yaml, sha256_file
from .rotations import quaternion_wxyz_to_matrix
from .schemas import CanonicalHuman


CAPTURE_SCHEMA_VERSION = 2
CAPTURE_CLASS = "exact_solver_input"
ACQUISITION_MODES = frozenset(
    {
        "native_runtime_target_setter",
        "official_loader_return_value",
        "native_runtime_target_setter_replay_verified",
        "runtime_hash_verified_reconstruction",
        "official_loader_runtime_hash_verified",
    }
)
GEOMETRY_CLASSES = frozenset(
    {"exact_solver_input", "exact_pre_solver_position_source"}
)
SEQUENCE_MANIFEST = "manifests/pilot_sequence.yaml"

HOLOSOMA_LAFAN_ORDER = (
    "Hips",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
    "Spine",
    "Spine1",
    "Spine2",
    "Neck",
    "Head",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
)

PROTOMOTIONS_V3_LABELS = (
    "pelvis",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_foot",
    "right_foot",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_wrist_aux",
    "right_wrist_aux",
    "pelvis_aux",
)


def tensor_sha256(value: np.ndarray) -> str:
    """Hash an array's numeric contract, independent of container metadata."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _npy_bytes(value: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, np.asarray(value), allow_pickle=False)
    return stream.getvalue()


def write_deterministic_npz(path: str | Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write a byte-reproducible, pickle-free compressed NPZ archive."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for key in sorted(arrays):
                if not key or "/" in key or "\\" in key:
                    raise ValueError(f"Invalid NPZ key: {key!r}")
                array = np.asarray(arrays[key])
                if array.dtype.hasobject:
                    raise ValueError(f"Object arrays are forbidden in capture NPZ: {key}")
                info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, _npy_bytes(array), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def gmr_scaled_positions(
    positions: np.ndarray,
    root_positions: np.ndarray,
    scale: np.ndarray,
    root_scale: np.ndarray,
) -> np.ndarray:
    """Apply GMR's root-global/body-root-relative scale formula."""

    value = np.asarray(positions, dtype=np.float64)
    root = np.asarray(root_positions, dtype=np.float64)
    local_scale = np.asarray(scale, dtype=np.float64)
    global_scale = np.asarray(root_scale, dtype=np.float64)
    return root * global_scale + (value - root) * local_scale


def holosoma_lafan_preprocess(
    positions: np.ndarray,
    joint_names: Iterable[str] = HOLOSOMA_LAFAN_ORDER,
    *,
    scale: float = 1.27 / 1.7,
    mat_height: float = 0.1,
) -> tuple[np.ndarray, float]:
    """Reproduce frozen Holosoma LAFAN spine, grounding, and scale operations."""

    labels = tuple(joint_names)
    value = np.asarray(positions, dtype=np.float64).copy()
    if value.ndim != 3 or value.shape[1:] != (len(labels), 3):
        raise ValueError("Holosoma LAFAN positions have an invalid shape")
    value[:, labels.index("Spine1"), 2] -= 0.06
    toe_indices = [labels.index("LeftToeBase"), labels.index("RightToeBase")]
    ground = float(value[:, toe_indices, 2].min())
    if ground >= mat_height:
        ground -= mat_height
    value[:, :, 2] -= ground
    value *= float(scale)
    return value, ground


def protomotions_v2_positions(
    positions: np.ndarray, scale_xyz: Iterable[float] = (0.75, 1.0, 0.8)
) -> np.ndarray:
    """Apply the frozen v2.3 adapter's world-axis position scaling."""

    value = np.asarray(positions, dtype=np.float64)
    scale = np.asarray(tuple(scale_xyz), dtype=np.float64)
    if scale.shape != (3,) or np.any(scale <= 0.0):
        raise ValueError("ProtoMotions v2.3 scale must be three positive values")
    return value * scale


def protomotions_v3_positions(positions: np.ndarray) -> np.ndarray:
    """Apply official v3 SMPL lower/upper/root axis scaling exactly."""

    value = np.asarray(positions, dtype=np.float64)
    if value.ndim != 3 or value.shape[1:] != (18, 3):
        raise ValueError("ProtoMotions v3 positions must have shape [T,18,3]")
    roots = value[:, 0]
    local = value - roots[:, None]
    lower = local[:, 1:9] * np.asarray([0.9, 0.9, 0.85])[None, None]
    upper = local[:, 9:18] * np.asarray([0.9, 0.9, 0.8])[None, None]
    scaled_roots = roots * np.asarray([0.9, 0.9, 0.85])[None]
    scaled_local = np.concatenate((lower, upper), axis=1)
    return np.concatenate(
        (scaled_roots[:, None], scaled_roots[:, None] + scaled_local), axis=1
    )


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path.resolve())


def _capture_paths(root: Path, sequence_id: str, method: str) -> tuple[Path, Path]:
    directory = root / "runs" / sequence_id / "native-pre-solver-targets" / method
    return directory / "targets.npz", directory / "capture.json"


def _formal_runtime_capture(
    root: Path,
    sequence: Mapping[str, Any],
    method: str,
) -> tuple[dict[str, Any], Path]:
    """Load the target hash observed by a completed formal solver run."""

    from .constants import STAGE1_RUN_DIRECTORIES
    from .schemas import CanonicalG1

    if method not in STAGE1_RUN_DIRECTORIES:
        raise KeyError(f"No frozen Stage-1 run directory for {method}")
    output = (
        root
        / "runs"
        / str(sequence["sequence_id"])
        / STAGE1_RUN_DIRECTORIES[method]
        / "canonical_g1.npz"
    )
    if not output.is_file():
        raise FileNotFoundError(f"Formal runtime witness is missing for {method}: {output}")
    motion = CanonicalG1.load(output)
    if motion.metadata.get("completion_status") != "succeeded":
        raise RuntimeError(f"Formal runtime witness is not complete for {method}")
    capture = motion.metadata.get("runtime_pre_solver_capture")
    if not isinstance(capture, dict) or capture.get("observed_during_solver_run") is not True:
        raise RuntimeError(f"Formal output lacks runtime target observation for {method}")
    return capture, output


def _save_capture(
    root: Path,
    sequence: dict[str, Any],
    method: str,
    arrays: Mapping[str, np.ndarray],
    *,
    tensor_key: str,
    tensor_label_key: str,
    geometry_tensor_key: str,
    geometry_label_key: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    if (
        tensor_key not in arrays
        or tensor_label_key not in arrays
        or geometry_tensor_key not in arrays
        or geometry_label_key not in arrays
    ):
        raise ValueError("Capture tensor/geometry keys are missing")
    tensor = np.asarray(arrays[tensor_key])
    tensor_labels = np.asarray(arrays[tensor_label_key]).astype(str)
    geometry = np.asarray(arrays[geometry_tensor_key])
    labels = np.asarray(arrays[geometry_label_key]).astype(str)
    if tensor.ndim != 3 or tensor.shape[-1] != 3 or not np.isfinite(tensor).all():
        raise ValueError(f"{method} target tensor must be finite [T,N,3]")
    if geometry.ndim != 3 or geometry.shape[-1] != 3 or not np.isfinite(geometry).all():
        raise ValueError(f"{method} geometry tensor must be finite [T,N,3]")
    if len(tensor_labels) != tensor.shape[1]:
        raise ValueError(f"{method} primary target labels do not match its tensor")
    if len(labels) != geometry.shape[1] or tensor.shape[0] != int(sequence["num_frames"]):
        raise ValueError(f"{method} target labels/frame count do not match the Pilot")
    if metadata.get("acquisition_mode") not in ACQUISITION_MODES:
        raise ValueError(f"{method} acquisition mode is missing or invalid")
    if metadata.get("geometry_class") not in GEOMETRY_CLASSES:
        raise ValueError(f"{method} geometry class is missing or invalid")
    for field in (
        "boundary_observed_during_formal_run",
        "reconstruction_matches_runtime",
        "runtime_witness_solver_invoked",
    ):
        if metadata.get(field) is not True:
            raise ValueError(f"{method} runtime witness field is not true: {field}")
    if metadata.get("runtime_tensor_sha256") != tensor_sha256(tensor):
        raise ValueError(f"{method} runtime/replayed target hash mismatch")
    witness_path = root / str(metadata["runtime_witness_output_path"])
    if (
        not witness_path.is_file()
        or sha256_file(witness_path) != metadata.get("runtime_witness_output_sha256")
    ):
        raise ValueError(f"{method} formal runtime witness output is missing or changed")
    artifact, metadata_path = _capture_paths(root, str(sequence["sequence_id"]), method)
    write_deterministic_npz(artifact, arrays)
    details = {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "method": method,
        "sequence_id": sequence["sequence_id"],
        "capture_class": CAPTURE_CLASS,
        "exact_solver_input": True,
        "normalized_projection": False,
        "capture_generation_solver_invoked": False,
        "frames": int(tensor.shape[0]),
        "target_count": int(tensor.shape[1]),
        "tensor_key": tensor_key,
        "tensor_label_key": tensor_label_key,
        "tensor_shape": list(tensor.shape),
        "tensor_dtype": tensor.dtype.str,
        "tensor_sha256": tensor_sha256(tensor),
        "geometry_tensor_key": geometry_tensor_key,
        "geometry_label_key": geometry_label_key,
        "geometry_tensor_sha256": tensor_sha256(geometry),
        "target_artifact": _relative(root, artifact),
        "target_artifact_sha256": sha256_file(artifact),
        "source_units": "metre",
        "target_units": "metre",
        "coordinate_axes": "right-handed xyz, +z up",
        **metadata,
    }
    atomic_write_json(metadata_path, details)
    return details


def _load_sequence(root: Path) -> dict[str, Any]:
    path = root / SEQUENCE_MANIFEST
    try:
        sequence = load_yaml(path)
    except ModuleNotFoundError as error:
        if error.name != "yaml":
            raise
        # Some frozen native environments intentionally contain only their
        # method dependencies.  The Pilot manifest is flat, so parse its
        # scalar fields without changing those environments.
        sequence = {}
        for line in path.read_text().splitlines():
            if not line or line[0].isspace() or line.lstrip().startswith("#"):
                continue
            key, separator, raw = line.partition(":")
            if not separator:
                continue
            value = raw.strip()
            if value.lower() in {"true", "false"}:
                parsed: Any = value.lower() == "true"
            else:
                try:
                    parsed = int(value)
                except ValueError:
                    try:
                        parsed = float(value)
                    except ValueError:
                        parsed = value
            sequence[key.strip()] = parsed
    if int(sequence["num_frames"]) != 600:
        raise ValueError("Native Stage 1 capture is frozen to the 600-frame Pilot")
    return sequence


def capture_gmr(repo_root: str | Path = ".") -> dict[str, Any]:
    """Capture targets set by GMR ``update_targets``; never call ``retarget``."""

    root = Path(repo_root).resolve()
    sequence = _load_sequence(root)
    checkout = root / "external" / "GMR"
    source = root / str(sequence["cropped_source_file"])
    config_path = checkout / "general_motion_retargeting/ik_configs/bvh_lafan1_to_g1.json"
    implementation = checkout / "general_motion_retargeting/motion_retarget.py"
    loader_path = checkout / "general_motion_retargeting/utils/lafan1.py"
    asset = checkout / "assets/unitree_g1/g1_mocap_29dof.xml"
    sys.path.insert(0, str(checkout))
    try:
        from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting
        from general_motion_retargeting.utils.lafan1 import load_bvh_file

        frames, actual_human_height = load_bvh_file(str(source), format="lafan1")
        retargeter = GeneralMotionRetargeting(
            src_human="bvh_lafan1",
            tgt_robot="unitree_g1",
            actual_human_height=actual_human_height,
            solver="daqp",
            damping=0.5,
            verbose=False,
        )
        config = json.loads(config_path.read_text())
        entries: list[tuple[str, str, str, float, float]] = []
        for table_name, table in (
            ("table1", config["ik_match_table1"]),
            ("table2", config["ik_match_table2"]),
        ):
            for robot_body, entry in table.items():
                human_body, position_cost, orientation_cost, _, _ = entry
                if float(position_cost) != 0.0 or float(orientation_cost) != 0.0:
                    entries.append(
                        (
                            table_name,
                            str(robot_body),
                            str(human_body),
                            float(position_cost),
                            float(orientation_cost),
                        )
                    )
        positions = np.empty((len(frames), len(entries), 3), dtype=np.float64)
        orientations = np.empty((len(frames), len(entries), 4), dtype=np.float64)
        for frame_index, frame in enumerate(frames):
            # This is the exact target-setting half of ``retarget``.  No IK
            # error, velocity, integration, or solve call follows it.
            retargeter.update_targets(frame, offset_to_ground=False)
            scaled = retargeter.scaled_human_data
            for target_index, (_, _, human_body, _, _) in enumerate(entries):
                positions[frame_index, target_index] = scaled[human_body][0]
                orientations[frame_index, target_index] = scaled[human_body][1]
        labels = np.asarray(
            [f"{table}:{robot}<-{human}" for table, robot, human, _, _ in entries]
        )
        position_costs = np.asarray([entry[3] for entry in entries], dtype=np.float64)
        orientation_costs = np.asarray([entry[4] for entry in entries], dtype=np.float64)
        scale_labels = np.asarray(list(retargeter.human_scale_table))
        scale_values = np.asarray(
            [retargeter.human_scale_table[name] for name in scale_labels], dtype=np.float64
        )
        arrays = {
            "position_targets": positions,
            "orientation_targets_wxyz": orientations,
            "target_labels": labels,
            "position_costs": position_costs,
            "orientation_costs": orientation_costs,
            "human_scale_labels": scale_labels,
            "human_scale_values": scale_values,
        }
        runtime_capture, runtime_output = _formal_runtime_capture(
            root, sequence, "gmr"
        )
        runtime_hash = str(runtime_capture["position_tensor_sha256"])
        runtime_orientation_hash = str(
            runtime_capture["orientation_wxyz_tensor_sha256"]
        )
        replay_hash = tensor_sha256(positions)
        replay_orientation_hash = tensor_sha256(orientations)
        if (
            runtime_hash != replay_hash
            or runtime_orientation_hash != replay_orientation_hash
        ):
            raise RuntimeError(
                "GMR position/orientation target replay differs from the completed formal run"
            )
        return _save_capture(
            root,
            sequence,
            "gmr",
            arrays,
            tensor_key="position_targets",
            tensor_label_key="target_labels",
            geometry_tensor_key="position_targets",
            geometry_label_key="target_labels",
            metadata={
                "acquisition_mode": "native_runtime_target_setter_replay_verified",
                "geometry_class": "exact_solver_input",
                "capture_stage": (
                    "official update_targets replay, hash-equal to scaled_human_data "
                    "observed during the completed formal solver run"
                ),
                "target_representation": "world-frame FrameTask translations for both staged task tables",
                "formula": (
                    "scaled_root=s_Hips*root; target=scaled_root+"
                    "s_body*(body-root); then configured local pose offset and zero ground offset"
                ),
                "root_target_index": int(next(i for i, value in enumerate(labels) if "<-Hips" in value)),
                "actual_human_height_m": float(actual_human_height),
                "human_height_assumption_m": float(config["human_height_assumption"]),
                "height_ratio": float(actual_human_height / config["human_height_assumption"]),
                "source_path": _relative(root, source),
                "source_sha256": sha256_file(source),
                "config_path": _relative(root, config_path),
                "config_sha256": sha256_file(config_path),
                "robot_asset_path": _relative(root, asset),
                "robot_asset_sha256": sha256_file(asset),
                "implementation_path": _relative(root, implementation),
                "implementation_sha256": sha256_file(implementation),
                "loader_path": _relative(root, loader_path),
                "loader_sha256": sha256_file(loader_path),
                "upstream_commit": "bb1bbe40774794fceb2a7c579a3464a28e68c844",
                "ground_policy": "offset_to_ground=False and ground_offset=0",
                "task_labels": labels.tolist(),
                "position_costs": position_costs.tolist(),
                "orientation_costs": orientation_costs.tolist(),
                "human_scale_table": {
                    str(name): float(value) for name, value in zip(scale_labels, scale_values)
                },
                "boundary_observed_during_formal_run": True,
                "reconstruction_matches_runtime": True,
                "runtime_witness_solver_invoked": True,
                "runtime_boundary": runtime_capture["boundary"],
                "runtime_tensor_sha256": runtime_hash,
                "runtime_orientation_wxyz_tensor_sha256": runtime_orientation_hash,
                "replay_orientation_wxyz_tensor_sha256": replay_orientation_hash,
                "runtime_witness_output_path": _relative(root, runtime_output),
                "runtime_witness_output_sha256": sha256_file(runtime_output),
            },
        )
    finally:
        if sys.path and sys.path[0] == str(checkout):
            sys.path.pop(0)


def capture_holosoma(repo_root: str | Path = ".") -> dict[str, Any]:
    """Capture Holosoma's exact interaction-mesh Laplacian target tensor."""

    root = Path(repo_root).resolve()
    sequence = _load_sequence(root)
    checkout = root / "external" / "holosoma"
    package_root = checkout / "src/holosoma_retargeting"
    source = (
        root
        / "source_adapters/omniretarget"
        / str(sequence["sequence_id"])
        / "pilot_canonical.npy"
    )
    implementation = (
        package_root
        / "holosoma_retargeting/src/interaction_mesh_retargeter.py"
    )
    utility_path = package_root / "holosoma_retargeting/src/utils.py"
    data_config_path = package_root / "holosoma_retargeting/config_types/data_type.py"
    pipeline_path = package_root / "holosoma_retargeting/examples/robot_retarget.py"
    asset = package_root / "holosoma_retargeting/models/g1/g1_29dof.urdf"
    sys.path.insert(0, str(package_root))
    try:
        from holosoma_retargeting.config_types.data_type import MotionDataConfig
        from holosoma_retargeting.examples.robot_retarget import (
            create_ground_points,
            extract_foot_sticking_sequence_velocity,
            load_motion_data,
        )
        from holosoma_retargeting.src.utils import (
            calculate_laplacian_coordinates,
            create_interaction_mesh,
            get_adjacency_list,
            preprocess_motion_data,
        )

        motion_config = MotionDataConfig(data_format="lafan", robot_type="g1")
        constants = SimpleNamespace(DEMO_JOINTS=list(HOLOSOMA_LAFAN_ORDER))
        loaded, _, native_scale = load_motion_data(
            "robot_only",
            "lafan",
            source.parent,
            source.stem,
            constants,
            motion_config,
        )
        expected_loaded = np.load(source)[..., [0, 2, 1]]
        expected_preprocessed, ground_offset = holosoma_lafan_preprocess(
            expected_loaded,
            scale=float(native_scale),
        )
        preprocess_proxy = SimpleNamespace(demo_joints=list(HOLOSOMA_LAFAN_ORDER))
        preprocessed = preprocess_motion_data(
            loaded.copy(),
            preprocess_proxy,
            motion_config.toe_names,
            scale=float(native_scale),
        )
        if not np.array_equal(preprocessed, expected_preprocessed):
            raise RuntimeError("Holosoma official preprocessing differs from frozen formula")
        mapping = motion_config.resolved_joints_mapping
        mapped_labels = np.asarray(list(mapping))
        mapped_indices = [HOLOSOMA_LAFAN_ORDER.index(name) for name in mapped_labels]
        mapped_positions = preprocessed[:, mapped_indices]
        ground_points = create_ground_points((-10.0, 10.0), (-10.0, 10.0), 15)
        vertex_labels = np.asarray(
            list(mapped_labels)
            + [f"ground_{index:03d}" for index in range(len(ground_points))]
        )
        laplacians = np.empty(
            (len(preprocessed), len(vertex_labels), 3), dtype=np.float64
        )
        topology_hashes: list[str] = []
        adjacency_hashes: list[str] = []
        for frame_index, positions in enumerate(mapped_positions):
            vertices, tetrahedra = create_interaction_mesh(
                np.vstack((positions, ground_points))
            )
            adjacency = get_adjacency_list(tetrahedra, len(vertices))
            laplacians[frame_index] = calculate_laplacian_coordinates(
                vertices, adjacency
            )
            topology_hashes.append(tensor_sha256(np.asarray(tetrahedra, dtype=np.int64)))
            adjacency_text = json.dumps(
                [sorted(int(item) for item in neighbors) for neighbors in adjacency],
                separators=(",", ":"),
            )
            adjacency_hashes.append(hashlib.sha256(adjacency_text.encode()).hexdigest())
        contacts_sequence = extract_foot_sticking_sequence_velocity(
            preprocessed,
            list(HOLOSOMA_LAFAN_ORDER),
            motion_config.toe_names,
        )
        contacts = np.asarray(
            [
                [entry["L_Toe"], entry["R_Toe"]]
                for entry in contacts_sequence
            ],
            dtype=bool,
        )
        arrays = {
            "solver_target_laplacian": laplacians,
            "solver_vertex_labels": vertex_labels,
            "mapped_position_targets": mapped_positions,
            "mapped_position_labels": mapped_labels,
            "human_joint_positions": preprocessed,
            "human_joint_labels": np.asarray(HOLOSOMA_LAFAN_ORDER),
            "ground_points": ground_points,
            "foot_sticking": contacts,
            "topology_sha256": np.asarray(topology_hashes),
            "adjacency_sha256": np.asarray(adjacency_hashes),
        }
        runtime_capture, runtime_output = _formal_runtime_capture(
            root, sequence, "omniretarget"
        )
        runtime_hash = str(runtime_capture["target_tensor_sha256"])
        reconstruction_hash = tensor_sha256(laplacians)
        mapped_runtime_hash = str(
            runtime_capture["mapped_position_tensor_sha256"]
        )
        runtime_contact_hash = str(runtime_capture["foot_sticking_tensor_sha256"])
        runtime_adjacency = list(runtime_capture["adjacency_sha256"])
        constraint_contract = runtime_capture.get("constraint_contract")
        if (
            runtime_hash != reconstruction_hash
            or mapped_runtime_hash != tensor_sha256(mapped_positions)
            or runtime_contact_hash != tensor_sha256(contacts)
            or runtime_adjacency != adjacency_hashes
            or not isinstance(constraint_contract, dict)
            or constraint_contract.get("activate_foot_sticking") is not True
            or constraint_contract.get("activate_obj_non_penetration") is not True
            or constraint_contract.get("activate_joint_limits") is not True
            or not np.isclose(
                float(constraint_contract.get("foot_sticking_tolerance_m", -1.0)),
                0.02,
                atol=0.0,
                rtol=0.0,
            )
        ):
            raise RuntimeError(
                "Holosoma target/contact/topology/constraint replay differs from runtime"
            )
        return _save_capture(
            root,
            sequence,
            "omniretarget",
            arrays,
            tensor_key="solver_target_laplacian",
            tensor_label_key="solver_vertex_labels",
            geometry_tensor_key="mapped_position_targets",
            geometry_label_key="mapped_position_labels",
            metadata={
                "acquisition_mode": "runtime_hash_verified_reconstruction",
                "geometry_class": "exact_pre_solver_position_source",
                "capture_stage": (
                    "offline official-function reconstruction, hash-equal to every "
                    "target_laplacian argument observed at iterate() in the formal run"
                ),
                "target_representation": (
                    "uniform-weight Laplacian coordinates of 15 mapped human positions "
                    "plus the frozen 15x15 ground grid"
                ),
                "formula": (
                    "canonical adapter -> y-up loader round-trip -> Spine1.z-=0.06 -> "
                    "global toe minimum grounding -> uniform scale 1.27/1.7 -> 15-joint "
                    "mapping -> Delaunay interaction mesh -> uniform Laplacian coordinates"
                ),
                "root_target_index": 0,
                "source_path": _relative(root, source),
                "source_sha256": sha256_file(source),
                "config_path": _relative(root, data_config_path),
                "config_sha256": sha256_file(data_config_path),
                "robot_asset_path": _relative(root, asset),
                "robot_asset_sha256": sha256_file(asset),
                "implementation_path": _relative(root, implementation),
                "implementation_sha256": sha256_file(implementation),
                "pipeline_path": _relative(root, pipeline_path),
                "pipeline_sha256": sha256_file(pipeline_path),
                "utility_path": _relative(root, utility_path),
                "utility_sha256": sha256_file(utility_path),
                "upstream_commit": "5f48635a3624656a5f46a07df26d43187e59f855",
                "native_uniform_scale": float(native_scale),
                "source_ground_offset_before_scale_m": float(ground_offset),
                "human_joint_order_22": list(HOLOSOMA_LAFAN_ORDER),
                "mapped_position_order_15": mapped_labels.tolist(),
                "mapped_robot_links_15": list(mapping.values()),
                "ground_grid": {"range_xy_m": [-10.0, 10.0], "size_per_axis": 15},
                "position_source_tensor_key": "mapped_position_targets",
                "solver_vertex_count": int(len(vertex_labels)),
                "boundary_observed_during_formal_run": True,
                "reconstruction_matches_runtime": True,
                "runtime_witness_solver_invoked": True,
                "runtime_boundary": runtime_capture["boundary"],
                "runtime_tensor_sha256": runtime_hash,
                "runtime_geometry_tensor_sha256": mapped_runtime_hash,
                "runtime_foot_sticking_tensor_sha256": runtime_contact_hash,
                "runtime_adjacency_sequence_sha256": runtime_capture[
                    "adjacency_sequence_sha256"
                ],
                "runtime_constraint_contract_sha256": runtime_capture[
                    "constraint_contract_sha256"
                ],
                "runtime_witness_output_path": _relative(root, runtime_output),
                "runtime_witness_output_sha256": sha256_file(runtime_output),
            },
        )
    finally:
        if sys.path and sys.path[0] == str(package_root):
            sys.path.pop(0)


def capture_protomotions_v2(repo_root: str | Path = ".") -> dict[str, Any]:
    """Capture the 14 exact FrameTask translations used by the v2.3 adapter."""

    root = Path(repo_root).resolve()
    sequence = _load_sequence(root)
    source = root / str(sequence["canonical_path"])
    config_path = root / "configs/protomotions_v2.yaml"
    config = load_yaml(config_path)
    implementation = root / "src/retargeting_comparison/protomotions_v2.py"
    upstream_implementation = root / str(config["upstream"]["worktree"]) / str(
        config["upstream"]["implementation"]
    )
    asset = root / str(config["native_robot"]["xml"])
    # Import and instantiate the exact formal adapter.  Its constructor creates
    # task objects and audits assets, but it does not run any optimization.
    from .protomotions_v2 import ProtoMotionsV2Retargeter

    human = CanonicalHuman.load(source)
    retargeter = ProtoMotionsV2Retargeter(root, strict_dependencies=False)
    indices = {
        name: index for index, name in enumerate(human.joint_names.astype(str))
    }
    targets = list(config["source_adapter"]["targets"])
    positions = np.empty((len(human.timestamps), len(targets), 3), dtype=np.float64)
    orientations = np.empty(
        (len(human.timestamps), len(targets), 3, 3), dtype=np.float64
    )
    root_index = indices["Hips"]
    for frame in range(len(human.timestamps)):
        for target_index, target in enumerate(targets):
            joint_index = indices[str(target["human_joint"])]
            positions[frame, target_index] = retargeter._target_position(
                human,
                frame,
                joint_index,
                root_index,
            )
            orientations[frame, target_index] = quaternion_wxyz_to_matrix(
                human.world_rotations[frame, joint_index]
            )
    source_positions = human.world_positions[
        :, [indices[str(target["human_joint"])] for target in targets]
    ]
    expected = protomotions_v2_positions(
        source_positions, config["algorithm"]["position_scale_xyz"]
    )
    if not np.array_equal(positions, expected):
        raise RuntimeError("ProtoMotions v2.3 formal adapter target formula changed")
    labels = np.asarray(
        [
            f"{target['robot_body']}<-{target['human_joint']}"
            for target in targets
        ]
    )
    arrays = {
        "position_targets": positions,
        "orientation_targets": orientations,
        "target_labels": labels,
        "robot_body_labels": np.asarray([target["robot_body"] for target in targets]),
        "human_joint_labels": np.asarray([target["human_joint"] for target in targets]),
    }
    runtime_capture, runtime_output = _formal_runtime_capture(
        root, sequence, "protomotions-v2.3"
    )
    runtime_hash = str(runtime_capture["position_tensor_sha256"])
    runtime_orientation_hash = str(
        runtime_capture["orientation_matrix_tensor_sha256"]
    )
    reconstruction_hash = tensor_sha256(positions)
    reconstruction_orientation_hash = tensor_sha256(orientations)
    if (
        runtime_hash != reconstruction_hash
        or runtime_orientation_hash != reconstruction_orientation_hash
    ):
        raise RuntimeError(
            "ProtoMotions v2.3 position/orientation replay differs from formal FrameTask setters"
        )
    return _save_capture(
        root,
        sequence,
        "protomotions-v2.3",
        arrays,
        tensor_key="position_targets",
        tensor_label_key="target_labels",
        geometry_tensor_key="position_targets",
        geometry_label_key="target_labels",
        metadata={
            "acquisition_mode": "runtime_hash_verified_reconstruction",
            "geometry_class": "exact_solver_input",
            "capture_stage": (
                "formula replay, hash-equal to values observed at every formal "
                "Mink FrameTask.set_target call"
            ),
            "target_representation": "14 world-frame semantic FrameTask translations",
            "formula": "target_xyz=canonical_LAFAN_world_xyz*[0.75,1.0,0.8]",
            "root_target_index": 0,
            "source_path": _relative(root, source),
            "source_sha256": sha256_file(source),
            "config_path": _relative(root, config_path),
            "config_sha256": sha256_file(config_path),
            "robot_asset_path": _relative(root, asset),
            "robot_asset_sha256": sha256_file(asset),
            "implementation_path": _relative(root, implementation),
            "implementation_sha256": sha256_file(implementation),
            "upstream_implementation_path": _relative(root, upstream_implementation),
            "upstream_implementation_sha256": sha256_file(upstream_implementation),
            "upstream_commit": str(config["upstream"]["commit"]),
            "official_position_scale_xyz": [
                float(value) for value in config["algorithm"]["position_scale_xyz"]
            ],
            "position_cost": float(config["algorithm"]["frame_position_cost"]),
            "orientation_cost": float(config["algorithm"]["frame_orientation_cost"]),
            "task_labels": labels.tolist(),
            "adapter_scope": (
                "canonical LAFAN port of the official v2.3 G1 Mink task/scale/solver; "
                "not an upstream official LAFAN entry point"
            ),
            "boundary_observed_during_formal_run": True,
            "reconstruction_matches_runtime": True,
            "runtime_witness_solver_invoked": True,
            "runtime_boundary": runtime_capture["boundary"],
            "runtime_tensor_sha256": runtime_hash,
            "runtime_orientation_matrix_tensor_sha256": runtime_orientation_hash,
            "replay_orientation_matrix_tensor_sha256": reconstruction_orientation_hash,
            "runtime_witness_output_path": _relative(root, runtime_output),
            "runtime_witness_output_sha256": sha256_file(runtime_output),
        },
    )


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import frozen module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def capture_protomotions_v3(repo_root: str | Path = ".") -> dict[str, Any]:
    """Capture official ``load_motion_data`` outputs without compiling a solver."""

    root = Path(repo_root).resolve()
    sequence = _load_sequence(root)
    source = (
        root
        / "source_adapters/protomotions_v3"
        / str(sequence["sequence_id"])
        / "keypoints.npy"
    )
    config_path = root / "configs/protomotions_v3.yaml"
    try:
        config = load_yaml(config_path)
    except ModuleNotFoundError as error:
        if error.name != "yaml":
            raise
        config = {
            "upstream": {
                "script": "pyroki/batch_retarget_to_g1_from_keypoints.py",
                "commit": "49fe5ad69de67ebbc07ea2b25d41b0f622c15c3c",
            },
            "pyroki": {"commit": "388e43e1fc0d0ee382968d3dd72970fd62a0450c"},
            "native_environment": {"conda_environment": "egoallo"},
        }
    script = root / str(config["upstream"]["script"])
    if not script.is_file():
        script = root / "external/ProtoMotions" / str(config["upstream"]["script"])
    asset = root / "external/ProtoMotions/protomotions/data/assets/urdf/for_retargeting/g1.urdf"
    official = _load_module(script, "rtcmp_native_target_capture_protomotions_v3")
    (
        target_keypoints,
        target_orientations,
        left_contact,
        right_contact,
        display_frames,
        input_fps,
    ) = official.load_motion_data(
        str(source),
        "smpl",
        1,
        int(sequence["num_frames"]),
        30.0,
    )
    positions = np.asarray(target_keypoints, dtype=np.float64)
    orientations = np.asarray(target_orientations, dtype=np.float64)
    left = np.asarray(left_contact, dtype=np.float64)
    right = np.asarray(right_contact, dtype=np.float64)
    raw = np.load(source, allow_pickle=True).item()
    expected = protomotions_v3_positions(
        np.asarray(raw["positions"], dtype=np.float64)
    )
    if not np.array_equal(positions, expected):
        raise RuntimeError("ProtoMotions v3 official loader target formula changed")
    if int(display_frames) != int(sequence["num_frames"]):
        raise RuntimeError("ProtoMotions v3 loader did not preserve the full Pilot")
    labels = np.asarray(PROTOMOTIONS_V3_LABELS)
    arrays = {
        "target_keypoints": positions,
        "target_orientations": orientations,
        "target_labels": labels,
        "left_foot_contact_smoothed": left,
        "right_foot_contact_smoothed": right,
    }
    runtime_capture, runtime_output = _formal_runtime_capture(
        root, sequence, "protomotions-v3"
    )
    runtime_hash = str(runtime_capture["target_keypoints_sha256"])
    runtime_orientation_hash = str(
        runtime_capture["target_orientations_sha256"]
    )
    runtime_contact_hash = str(runtime_capture["foot_contacts_sha256"])
    loader_hash = tensor_sha256(positions)
    orientation_hash = tensor_sha256(orientations)
    contacts_hash = tensor_sha256(np.stack((left, right), axis=-1))
    if (
        runtime_hash != loader_hash
        or runtime_orientation_hash != orientation_hash
        or runtime_contact_hash != contacts_hash
    ):
        raise RuntimeError(
            "ProtoMotions v3 keypoint/orientation/contact replay differs from formal solve input"
        )
    return _save_capture(
        root,
        sequence,
        "protomotions-v3",
        arrays,
        tensor_key="target_keypoints",
        tensor_label_key="target_labels",
        geometry_tensor_key="target_keypoints",
        geometry_label_key="target_labels",
        metadata={
            "acquisition_mode": "official_loader_runtime_hash_verified",
            "geometry_class": "exact_solver_input",
            "capture_stage": (
                "official load_motion_data replay, hash-equal to target_keypoints "
                "observed immediately before formal solve_retargeting"
            ),
            "target_representation": "15 semantic plus 3 auxiliary world-frame keypoints",
            "formula": (
                "root*= [0.9,0.9,0.85]; keypoints[1:9]-root *= "
                "[0.9,0.9,0.85]; keypoints[9:18]-root *= [0.9,0.9,0.8]"
            ),
            "root_target_index": 0,
            "source_path": _relative(root, source),
            "source_sha256": sha256_file(source),
            "config_path": _relative(root, config_path),
            "config_sha256": sha256_file(config_path),
            "robot_asset_path": _relative(root, asset),
            "robot_asset_sha256": sha256_file(asset),
            "implementation_path": _relative(root, script),
            "implementation_sha256": sha256_file(script),
            "upstream_commit": str(config["upstream"]["commit"]),
            "pyroki_commit": str(config["pyroki"]["commit"]),
            "input_fps": float(input_fps),
            "subsample_factor": 1,
            "task_labels": labels.tolist(),
            "contact_capture": "official 5-frame cross-faded ankle/toe mean",
            "capture_environment": str(config["native_environment"]["conda_environment"]),
            "boundary_observed_during_formal_run": True,
            "reconstruction_matches_runtime": True,
            "runtime_witness_solver_invoked": True,
            "runtime_boundary": runtime_capture["boundary"],
            "runtime_tensor_sha256": runtime_hash,
            "runtime_orientation_tensor_sha256": runtime_orientation_hash,
            "runtime_contact_tensor_sha256": runtime_contact_hash,
            "replay_orientation_tensor_sha256": orientation_hash,
            "replay_contact_tensor_sha256": contacts_hash,
            "runtime_witness_output_path": _relative(root, runtime_output),
            "runtime_witness_output_sha256": sha256_file(runtime_output),
        },
    )


def _geometry_rows(
    method: str,
    capture_class: str,
    geometry_class: str,
    tensor_key: str,
    positions: np.ndarray,
    labels: np.ndarray,
    root_index: int,
) -> list[dict[str, Any]]:
    value = np.asarray(positions, dtype=np.float64)
    names = np.asarray(labels).astype(str)
    if not 0 <= root_index < value.shape[1]:
        raise ValueError(f"{method} root target index is invalid")
    roots = value[:, root_index]
    rows: list[dict[str, Any]] = []
    for index, label in enumerate(names):
        trajectory = value[:, index]
        displacement = np.diff(trajectory, axis=0)
        relative = trajectory - roots
        span = np.ptp(trajectory, axis=0)
        rows.append(
            {
                "method": method,
                "capture_class": capture_class,
                "geometry_class": geometry_class,
                "tensor_key": tensor_key,
                "target_index": index,
                "target_label": label,
                "frames": len(value),
                "mean_x_m": float(trajectory[:, 0].mean()),
                "mean_y_m": float(trajectory[:, 1].mean()),
                "mean_z_m": float(trajectory[:, 2].mean()),
                "span_x_m": float(span[0]),
                "span_y_m": float(span[1]),
                "span_z_m": float(span[2]),
                "trajectory_length_m": float(np.linalg.norm(displacement, axis=1).sum()),
                "mean_root_relative_radius_m": float(np.linalg.norm(relative, axis=1).mean()),
                "max_root_relative_radius_m": float(np.linalg.norm(relative, axis=1).max()),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def aggregate_captures(repo_root: str | Path = ".") -> dict[str, Any]:
    """Validate all exact captures and publish small manifest/geometry tables."""

    root = Path(repo_root).resolve()
    sequence = _load_sequence(root)
    methods = ("gmr", "omniretarget", "protomotions-v2.3", "protomotions-v3")
    manifest_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    for method in methods:
        artifact, metadata_path = _capture_paths(
            root, str(sequence["sequence_id"]), method
        )
        if not artifact.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"Native capture is missing for {method}")
        metadata = json.loads(metadata_path.read_text())
        if (
            metadata.get("capture_class") != CAPTURE_CLASS
            or metadata.get("acquisition_mode") not in ACQUISITION_MODES
            or metadata.get("geometry_class") not in GEOMETRY_CLASSES
            or metadata.get("exact_solver_input") is not True
            or metadata.get("normalized_projection") is not False
            or metadata.get("capture_generation_solver_invoked") is not False
            or metadata.get("boundary_observed_during_formal_run") is not True
            or metadata.get("reconstruction_matches_runtime") is not True
            or metadata.get("runtime_witness_solver_invoked") is not True
        ):
            raise ValueError(f"{method} capture class/solver guard is invalid")
        if sha256_file(artifact) != metadata["target_artifact_sha256"]:
            raise ValueError(f"{method} deterministic NPZ hash mismatch")
        with np.load(artifact, allow_pickle=False) as archive:
            tensor = np.asarray(archive[metadata["tensor_key"]])
            tensor_labels = np.asarray(archive[metadata["tensor_label_key"]])
            geometry = np.asarray(archive[metadata["geometry_tensor_key"]])
            labels = np.asarray(archive[metadata["geometry_label_key"]])
        if len(tensor_labels) != tensor.shape[1]:
            raise ValueError(f"{method} primary target labels do not match its tensor")
        if tensor_sha256(tensor) != metadata["tensor_sha256"]:
            raise ValueError(f"{method} target tensor hash mismatch")
        if tensor_sha256(geometry) != metadata["geometry_tensor_sha256"]:
            raise ValueError(f"{method} geometry tensor hash mismatch")
        geometry_rows.extend(
            _geometry_rows(
                method,
                str(metadata["capture_class"]),
                str(metadata["geometry_class"]),
                str(metadata["geometry_tensor_key"]),
                geometry,
                labels,
                int(metadata["root_target_index"]),
            )
        )
        manifest_rows.append(
            {
                "method": method,
                "sequence_id": metadata["sequence_id"],
                "capture_class": metadata["capture_class"],
                "acquisition_mode": metadata["acquisition_mode"],
                "exact_solver_input": str(metadata["exact_solver_input"]).lower(),
                "normalized_projection": str(metadata["normalized_projection"]).lower(),
                "capture_generation_solver_invoked": str(
                    metadata["capture_generation_solver_invoked"]
                ).lower(),
                "boundary_observed_during_formal_run": str(
                    metadata["boundary_observed_during_formal_run"]
                ).lower(),
                "reconstruction_matches_runtime": str(
                    metadata["reconstruction_matches_runtime"]
                ).lower(),
                "runtime_witness_solver_invoked": str(
                    metadata["runtime_witness_solver_invoked"]
                ).lower(),
                "runtime_boundary": metadata["runtime_boundary"],
                "runtime_tensor_sha256": metadata["runtime_tensor_sha256"],
                "runtime_witness_output_path": metadata[
                    "runtime_witness_output_path"
                ],
                "runtime_witness_output_sha256": metadata[
                    "runtime_witness_output_sha256"
                ],
                "capture_stage": metadata["capture_stage"],
                "target_representation": metadata["target_representation"],
                "formula": metadata["formula"],
                "frames": metadata["frames"],
                "target_count": metadata["target_count"],
                "tensor_key": metadata["tensor_key"],
                "tensor_label_key": metadata["tensor_label_key"],
                "tensor_shape": json.dumps(metadata["tensor_shape"], separators=(",", ":")),
                "tensor_dtype": metadata["tensor_dtype"],
                "tensor_sha256": metadata["tensor_sha256"],
                "geometry_tensor_key": metadata["geometry_tensor_key"],
                "geometry_class": metadata["geometry_class"],
                "geometry_tensor_sha256": metadata["geometry_tensor_sha256"],
                "target_artifact": metadata["target_artifact"],
                "target_artifact_sha256": metadata["target_artifact_sha256"],
                "source_path": metadata["source_path"],
                "source_sha256": metadata["source_sha256"],
                "config_path": metadata["config_path"],
                "config_sha256": metadata["config_sha256"],
                "robot_asset_path": metadata["robot_asset_path"],
                "robot_asset_sha256": metadata["robot_asset_sha256"],
                "implementation_path": metadata["implementation_path"],
                "implementation_sha256": metadata["implementation_sha256"],
                "upstream_commit": metadata["upstream_commit"],
                "source_units": metadata["source_units"],
                "target_units": metadata["target_units"],
                "coordinate_axes": metadata["coordinate_axes"],
            }
        )
    manifest_fields = [
        "method",
        "sequence_id",
        "capture_class",
        "acquisition_mode",
        "exact_solver_input",
        "normalized_projection",
        "capture_generation_solver_invoked",
        "boundary_observed_during_formal_run",
        "reconstruction_matches_runtime",
        "runtime_witness_solver_invoked",
        "runtime_boundary",
        "runtime_tensor_sha256",
        "runtime_witness_output_path",
        "runtime_witness_output_sha256",
        "capture_stage",
        "target_representation",
        "formula",
        "frames",
        "target_count",
        "tensor_key",
        "tensor_label_key",
        "tensor_shape",
        "tensor_dtype",
        "tensor_sha256",
        "geometry_tensor_key",
        "geometry_class",
        "geometry_tensor_sha256",
        "target_artifact",
        "target_artifact_sha256",
        "source_path",
        "source_sha256",
        "config_path",
        "config_sha256",
        "robot_asset_path",
        "robot_asset_sha256",
        "implementation_path",
        "implementation_sha256",
        "upstream_commit",
        "source_units",
        "target_units",
        "coordinate_axes",
    ]
    geometry_fields = [
        "method",
        "capture_class",
        "geometry_class",
        "tensor_key",
        "target_index",
        "target_label",
        "frames",
        "mean_x_m",
        "mean_y_m",
        "mean_z_m",
        "span_x_m",
        "span_y_m",
        "span_z_m",
        "trajectory_length_m",
        "mean_root_relative_radius_m",
        "max_root_relative_radius_m",
    ]
    manifest_path = root / "manifests/native_pre_solver_targets.csv"
    geometry_path = root / "metrics/native_pre_solver_target_geometry.csv"
    _write_csv(manifest_path, manifest_rows, manifest_fields)
    _write_csv(geometry_path, geometry_rows, geometry_fields)
    return {
        "methods": list(methods),
        "manifest": _relative(root, manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "geometry": _relative(root, geometry_path),
        "geometry_sha256": sha256_file(geometry_path),
        "geometry_rows": len(geometry_rows),
        "all_exact_solver_input": True,
        "all_formal_boundaries_observed": True,
        "all_reconstructions_match_runtime": True,
        "normalized_projection_rows": 0,
        "capture_generation_solver_invocations": 0,
        "runtime_witness_solver_invocations": len(methods),
    }


CAPTURE_FUNCTIONS: dict[str, Callable[[str | Path], dict[str, Any]]] = {
    "gmr": capture_gmr,
    "omniretarget": capture_holosoma,
    "protomotions-v2.3": capture_protomotions_v2,
    "protomotions-v3": capture_protomotions_v3,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("capture", "aggregate"))
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--method", choices=tuple(CAPTURE_FUNCTIONS))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action == "capture":
        if args.method is None:
            raise SystemExit("--method is required for capture")
        result = CAPTURE_FUNCTIONS[args.method](args.repo_root)
    else:
        if args.method is not None:
            raise SystemExit("--method is invalid for aggregate")
        result = aggregate_captures(args.repo_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
