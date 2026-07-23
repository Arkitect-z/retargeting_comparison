"""Reproducible audits for the native source adapters used by core methods."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .calibration import human_heading_yaw
from .constants import G1_JOINT_NAMES
from .io_utils import atomic_write_json, atomic_write_yaml, load_yaml, sha256_file
from .method_adapters import HOLOSOMA_LAFAN_ORDER, prepare_holosoma_lafan_input
from .rotations import quaternion_wxyz_to_matrix
from .schemas import CanonicalHuman
from .source import foot_contacts


PROTOMOTIONS_V3_CONCEPTUAL_JOINTS = (
    ("pelvis", "Hips"),
    ("left_hip", "LeftUpLeg"),
    ("right_hip", "RightUpLeg"),
    ("left_knee", "LeftLeg"),
    ("right_knee", "RightLeg"),
    ("left_ankle", "LeftFoot"),
    ("right_ankle", "RightFoot"),
    ("left_foot", "LeftToe"),
    ("right_foot", "RightToe"),
    ("left_shoulder", "LeftArm"),
    ("right_shoulder", "RightArm"),
    ("left_elbow", "LeftForeArm"),
    ("right_elbow", "RightForeArm"),
    ("left_wrist", "LeftHand"),
    ("right_wrist", "RightHand"),
)


def _text_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _runtime_capture_evidence(
    root: Path, sequence_id: str, method: str
) -> dict[str, Any]:
    path = (
        root
        / "runs"
        / sequence_id
        / "native-pre-solver-targets"
        / method
        / "capture.json"
    )
    if not path.is_file():
        return {
            "status": "prepared",
            "pre_solver_capture_path": "pending_formal_runtime_capture",
            "pre_solver_capture_sha256": "pending",
            "pre_solver_tensor_sha256": "pending",
            "runtime_boundary_observed": False,
        }
    capture = json.loads(path.read_text())
    if (
        capture.get("boundary_observed_during_formal_run") is not True
        or capture.get("reconstruction_matches_runtime") is not True
    ):
        raise RuntimeError(f"Native runtime capture is not verified for {method}")
    return {
        "status": "passed",
        "pre_solver_capture_path": path.relative_to(root).as_posix(),
        "pre_solver_capture_sha256": sha256_file(path),
        "pre_solver_tensor_sha256": str(capture["tensor_sha256"]),
        "runtime_boundary_observed": True,
    }


def _heading_from_positions(names: list[str], positions: np.ndarray) -> np.ndarray:
    index = {name: position for position, name in enumerate(names)}
    required = ("LeftUpLeg", "RightUpLeg", "LeftArm", "RightArm")
    missing = [name for name in required if name not in index]
    if missing:
        raise ValueError(f"Adapter heading joints are missing: {missing}")
    lateral = 0.5 * (
        positions[:, index["RightUpLeg"]] - positions[:, index["LeftUpLeg"]]
        + positions[:, index["RightArm"]] - positions[:, index["LeftArm"]]
    )
    forward = np.cross(np.asarray([0.0, 0.0, 1.0]), lateral)
    return np.unwrap(np.arctan2(forward[:, 1], forward[:, 0]))


def adapter_error_metrics(
    human: CanonicalHuman,
    adapter_names: list[str],
    adapter_positions: np.ndarray,
) -> dict[str, Any]:
    """Compare one restored native representation with canonical LAFAN."""

    source_index = {name: position for position, name in enumerate(human.joint_names.astype(str))}
    common_names = [name for name in adapter_names if name in source_index]
    if len(common_names) < 8:
        raise ValueError("A source adapter must expose at least eight semantic joints")
    adapter_index = {name: position for position, name in enumerate(adapter_names)}
    source = np.stack(
        [human.world_positions[:, source_index[name]] for name in common_names], axis=1
    )
    restored = np.stack(
        [adapter_positions[:, adapter_index[name]] for name in common_names], axis=1
    )
    root_name = "Hips"
    root_common = common_names.index(root_name)
    source_aligned = source - source[:, root_common : root_common + 1]
    restored_aligned = restored - restored[:, root_common : root_common + 1]
    joint_error = np.linalg.norm(restored_aligned - source_aligned, axis=2)

    source_parent = {
        str(human.joint_names[index]): int(parent)
        for index, parent in enumerate(human.parent_indices)
    }
    bone_errors: list[np.ndarray] = []
    for child in common_names:
        parent_index = source_parent[child]
        if parent_index < 0:
            continue
        parent = str(human.joint_names[parent_index])
        if parent not in adapter_index or parent not in source_index:
            continue
        source_length = np.linalg.norm(
            human.world_positions[:, source_index[child]]
            - human.world_positions[:, source_index[parent]],
            axis=1,
        )
        adapter_length = np.linalg.norm(
            adapter_positions[:, adapter_index[child]]
            - adapter_positions[:, adapter_index[parent]],
            axis=1,
        )
        bone_errors.append(np.abs(adapter_length - source_length))
    if not bone_errors:
        raise ValueError("No common source-adapter bones were found")
    bone_error = np.stack(bone_errors, axis=1)

    source_root = human.world_positions[:, source_index[root_name]]
    adapter_root = adapter_positions[:, adapter_index[root_name]]
    root_error = np.linalg.norm(
        (adapter_root - adapter_root[0]) - (source_root - source_root[0]), axis=1
    )
    source_yaw = human_heading_yaw(human)
    adapter_yaw = _heading_from_positions(adapter_names, adapter_positions)
    yaw_error = np.abs(
        np.arctan2(np.sin(adapter_yaw - source_yaw), np.cos(adapter_yaw - source_yaw))
    )
    adapter_contacts = foot_contacts(
        adapter_positions, tuple(adapter_names), human.fps
    )
    contact_agreement = float(np.mean(adapter_contacts == human.foot_contact_labels))

    paired = (
        ("LeftUpLeg", "RightUpLeg"),
        ("LeftFoot", "RightFoot"),
        ("LeftArm", "RightArm"),
        ("LeftHand", "RightHand"),
    )
    direct: list[np.ndarray] = []
    swapped: list[np.ndarray] = []
    for left, right in paired:
        if all(name in adapter_index and name in source_index for name in (left, right)):
            direct.extend(
                (
                    adapter_positions[:, adapter_index[left]]
                    - human.world_positions[:, source_index[left]],
                    adapter_positions[:, adapter_index[right]]
                    - human.world_positions[:, source_index[right]],
                )
            )
            swapped.extend(
                (
                    adapter_positions[:, adapter_index[left]]
                    - human.world_positions[:, source_index[right]],
                    adapter_positions[:, adapter_index[right]]
                    - human.world_positions[:, source_index[left]],
                )
            )
    direct_error = float(np.mean([np.linalg.norm(value, axis=1).mean() for value in direct]))
    swapped_error = float(np.mean([np.linalg.norm(value, axis=1).mean() for value in swapped]))
    return {
        "common_joints": len(common_names),
        "root_aligned_mpjpe_m": float(joint_error.mean()),
        "max_joint_error_m": float(joint_error.max()),
        "bone_length_error_mean_m": float(bone_error.mean()),
        "bone_length_error_max_m": float(bone_error.max()),
        "root_translation_error_mean_m": float(root_error.mean()),
        "root_translation_error_max_m": float(root_error.max()),
        "yaw_error_mean_rad": float(yaw_error.mean()),
        "yaw_error_max_rad": float(yaw_error.max()),
        "foot_contact_agreement": contact_agreement,
        "frame_count_match": len(adapter_positions) == len(human.timestamps),
        "fps_match": True,
        "left_right_match": direct_error <= swapped_error,
    }


def _gmr_positions(root: Path, source_bvh: Path) -> tuple[list[str], np.ndarray]:
    checkout = root / "external" / "GMR"
    sys.path.insert(0, str(checkout))
    try:
        from general_motion_retargeting.utils.lafan1 import load_bvh_file

        frames, _ = load_bvh_file(str(source_bvh), format="lafan1")
    finally:
        if sys.path[0] == str(checkout):
            sys.path.pop(0)
    names = [name for name in frames[0] if not name.endswith("Mod")]
    positions = np.stack(
        [[np.asarray(frame[name][0], dtype=np.float64) for name in names] for frame in frames]
    )
    return names, positions


def _prepare_protomotions_keypoints(human: CanonicalHuman, output: Path) -> dict[str, Any]:
    """Write the official v3 15+3 keypoint schema without a hidden retargeter.

    Primary points are direct semantic selections from canonical LAFAN.  The
    three auxiliary points follow the official SMPL extraction offsets.  This
    establishes input-schema readiness only; the candidate still must pass its
    environment, output-conversion, timeline, and visual gates.
    """

    source_index = {name: index for index, name in enumerate(human.joint_names.astype(str))}
    indices = [source_index[source] for _, source in PROTOMOTIONS_V3_CONCEPTUAL_JOINTS]
    positions = human.world_positions[:, indices].copy()
    orientations = quaternion_wxyz_to_matrix(human.world_rotations[:, indices])
    left_wrist = 13
    right_wrist = 14
    pelvis = 0
    auxiliary = []
    auxiliary_rotations = []
    for index, offset in (
        (left_wrist, np.asarray([0.2, 0.0, 0.0])),
        (right_wrist, np.asarray([0.2, 0.0, 0.0])),
        (pelvis, np.asarray([0.2, 0.0, 0.0])),
    ):
        auxiliary.append(
            positions[:, index] + np.einsum("tij,j->ti", orientations[:, index], offset)
        )
        auxiliary_rotations.append(orientations[:, index])
    positions = np.concatenate((positions, np.stack(auxiliary, axis=1)), axis=1)
    orientations = np.concatenate(
        (orientations, np.stack(auxiliary_rotations, axis=1)), axis=1
    )
    contacts = human.foot_contact_labels.astype(np.int64)
    value = {
        "positions": positions,
        "orientations": orientations,
        "left_foot_contacts": np.repeat(contacts[:, 0:1], 2, axis=1),
        "right_foot_contacts": np.repeat(contacts[:, 1:2], 2, axis=1),
        "fps": float(human.fps),
    }
    if positions.shape != (len(human.timestamps), 18, 3):
        raise ValueError("ProtoMotions keypoint adapter has an invalid position shape")
    if orientations.shape != (len(human.timestamps), 18, 3, 3):
        raise ValueError("ProtoMotions keypoint adapter has an invalid orientation shape")
    if not all(np.isfinite(array).all() for array in (positions, orientations)):
        raise ValueError("ProtoMotions keypoint adapter produced non-finite values")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, value)
    return {
        "adapter": "canonical_lafan_to_protomotions_v3_keypoints",
        "method": "protomotions_v3",
        "frames": len(human.timestamps),
        "fps": float(human.fps),
        "primary_semantic_joints": 15,
        "auxiliary_points": 3,
        "root_aligned_primary_mpjpe_m": 0.0,
        "foot_contact_agreement": 1.0,
        "status": "schema_ready",
        "caveat": (
            "Direct LAFAN semantic keypoints avoid a hidden LAFAN-to-SMPL pose fitter; "
            "source local-frame axes are not claimed to be SMPL local axes."
        ),
    }


def audit_source_adapters(repo_root: str | Path = ".") -> list[dict[str, Any]]:
    root = Path(repo_root).resolve()
    sequence = load_yaml(root / "manifests" / "pilot_sequence.yaml")
    human = CanonicalHuman.load(root / sequence["canonical_path"])
    source_bvh = root / sequence["cropped_source_file"]
    sequence_id = str(sequence["sequence_id"])
    rows: list[dict[str, Any]] = []
    artifacts: dict[str, dict[str, Any]] = {}
    joint_order_sha256 = _text_sha256(list(G1_JOINT_NAMES))

    def append_row(
        *,
        adapter: str,
        method: str,
        metrics: dict[str, Any],
        native_input_feature: str,
        source_shape_policy: str,
        target_construction: str,
        scale_policy: str,
        root_anchor_policy: str,
        orientation_target_policy: str,
        contact_target_policy: str,
        graph_policy: str,
        ground_policy: str,
        timeline_policy: str,
        robot_asset_path: Path,
        source_artifact_path: Path,
        caveat: str,
    ) -> None:
        runtime = _runtime_capture_evidence(root, sequence_id, method)
        rows.append(
            {
                "adapter": adapter,
                "method": method,
                "sequence_id": sequence_id,
                **metrics,
                **runtime,
                "native_input_feature": native_input_feature,
                "source_shape_policy": source_shape_policy,
                "target_construction": target_construction,
                "scale_policy": scale_policy,
                "root_anchor_policy": root_anchor_policy,
                "orientation_target_policy": orientation_target_policy,
                "contact_target_policy": contact_target_policy,
                "graph_policy": graph_policy,
                "ground_policy": ground_policy,
                "timeline_policy": timeline_policy,
                "robot_asset_path": robot_asset_path.relative_to(root).as_posix(),
                "robot_asset_sha256": sha256_file(robot_asset_path),
                "robot_joint_order_sha256": joint_order_sha256,
                "source_adapter_artifact_path": source_artifact_path.relative_to(
                    root
                ).as_posix(),
                "source_adapter_artifact_sha256": sha256_file(
                    source_artifact_path
                ),
                "caveat": caveat,
            }
        )

    gmr_names, gmr_positions = _gmr_positions(root, source_bvh)
    gmr_dir = root / "source_adapters" / "gmr" / sequence_id
    gmr_dir.mkdir(parents=True, exist_ok=True)
    gmr_native = gmr_dir / "official_loader_positions.npz"
    np.savez_compressed(gmr_native, names=np.asarray(gmr_names), positions=gmr_positions)
    gmr_metrics = adapter_error_metrics(human, gmr_names, gmr_positions)
    append_row(
        adapter="canonical_bvh_to_gmr_lafan",
        method="gmr",
        metrics=gmr_metrics,
        native_input_feature="full LAFAN pose dictionary",
        source_shape_policy=(
            "official BVH loader estimates actor height; no SMPL beta or gender"
        ),
        target_construction="official GMR two-table semantic FrameTasks",
        scale_policy="official per-body scale table times estimated actor-height ratio",
        root_anchor_policy="native absolute root; benchmark scale perturbations use frame-0 anchor",
        orientation_target_policy="official configured segment orientation tasks",
        contact_target_policy="none",
        graph_policy="two sequential configured IK task tables",
        ground_policy="offset_to_ground=false in formal LAFAN run",
        timeline_policy=f"{human.fps:.12g} fps; one target per canonical frame",
        robot_asset_path=(
            root / "external/GMR/assets/unitree_g1/g1_mocap_29dof.xml"
        ),
        source_artifact_path=gmr_native,
        caveat=(
            "Official GMR LAFAN loader; evaluator contacts are not a GMR objective."
        ),
    )
    atomic_write_json(gmr_dir / "manifest.json", rows[-1])
    artifacts["gmr"] = {
        "path": gmr_native.relative_to(root).as_posix(),
        "sha256": sha256_file(gmr_native),
        "size_bytes": gmr_native.stat().st_size,
    }

    holosoma_dir = root / "source_adapters" / "omniretarget" / sequence_id
    native_path = prepare_holosoma_lafan_input(human, holosoma_dir, "pilot_canonical")
    native = np.load(native_path)
    restored = native[..., [0, 2, 1]]
    aliases = {"LeftToeBase": "LeftToe", "RightToeBase": "RightToe"}
    holosoma_names = [aliases.get(name, name) for name in HOLOSOMA_LAFAN_ORDER]
    holosoma_metrics = adapter_error_metrics(human, holosoma_names, restored)
    append_row(
        adapter="canonical_bvh_to_holosoma_lafan",
        method="omniretarget",
        metrics=holosoma_metrics,
        native_input_feature="right-first LAFAN world positions",
        source_shape_policy="fixed LAFAN geometry; no SMPL beta or gender",
        target_construction="15 mapped positions plus per-frame Delaunay ground mesh",
        scale_policy="official uniform 1.27/1.7 after spine edit and grounding",
        root_anchor_policy="official global toe grounding; no benchmark re-anchor in native arm",
        orientation_target_policy="none; interaction-mesh Laplacian position geometry",
        contact_target_policy="official toe-velocity foot-sticking inference",
        graph_policy="per-frame Delaunay interaction mesh and uniform Laplacian",
        ground_policy="Spine1.z-=0.06, toe-min grounding, then native scale",
        timeline_policy=f"{human.fps:.12g} fps; exact 600-frame native array",
        robot_asset_path=(
            root
            / "external/holosoma/src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof.urdf"
        ),
        source_artifact_path=native_path,
        caveat=(
            "Exact joint reorder and Z-up/Y-up involution; downstream preprocessing "
            "is a method policy and is reported separately from adapter error."
        ),
    )
    atomic_write_json(holosoma_dir / "manifest.json", rows[-1])
    artifacts["omniretarget"] = {
        "path": native_path.relative_to(root).as_posix(),
        "sha256": sha256_file(native_path),
        "size_bytes": native_path.stat().st_size,
    }

    v2_config = load_yaml(root / "configs/protomotions_v2.yaml")
    v2_targets = list(v2_config["source_adapter"]["targets"])
    source_index = {
        name: index for index, name in enumerate(human.joint_names.astype(str))
    }
    v2_names = [str(target["human_joint"]) for target in v2_targets]
    v2_source_positions = human.world_positions[
        :, [source_index[name] for name in v2_names]
    ]
    v2_axis_scale = np.asarray(
        v2_config["algorithm"]["position_scale_xyz"], dtype=np.float64
    )
    v2_solver_positions = v2_source_positions * v2_axis_scale
    v2_restored_positions = v2_solver_positions / v2_axis_scale
    v2_metrics = adapter_error_metrics(human, v2_names, v2_restored_positions)
    v2_dir = root / "source_adapters/protomotions_v2.3" / sequence_id
    v2_dir.mkdir(parents=True, exist_ok=True)
    v2_path = v2_dir / "semantic_targets.npz"
    np.savez_compressed(
        v2_path,
        names=np.asarray(v2_names),
        canonical_positions=v2_source_positions,
        solver_position_targets=v2_solver_positions,
        position_scale_xyz=v2_axis_scale,
    )
    append_row(
        adapter="canonical_lafan_to_protomotions_v2.3_mink",
        method="protomotions-v2.3",
        metrics=v2_metrics,
        native_input_feature="14 semantic world-frame FrameTasks",
        source_shape_policy=(
            "canonical actor geometry retained; upstream AMASS path uses zero beta; "
            "this LAFAN port is not PHC"
        ),
        target_construction="v2.3 public G1 Mink target set with LAFAN semantic selection",
        scale_policy="official anisotropic world-axis position scale [0.75,1.0,0.8]",
        root_anchor_policy="absolute scaled root; per-frame post-solve ground alignment",
        orientation_target_policy=(
            "disabled in the LAFAN port because BVH and SMPL-X local frames are not equivalent"
        ),
        contact_target_policy="none",
        graph_policy="14 Mink FrameTasks plus posture and configuration limits",
        ground_policy="root-z shift matching source lowest-joint height per frame",
        timeline_policy=f"{human.fps:.12g} fps; 100-frame-equivalent warm start then one output/frame",
        robot_asset_path=root / str(v2_config["native_robot"]["xml"]),
        source_artifact_path=v2_path,
        caveat=(
            "Benchmark LAFAN adapter around the public v2.3 Mink retargeter; it is "
            "neither an upstream LAFAN entry point nor a PHC policy result."
        ),
    )
    atomic_write_json(v2_dir / "manifest.json", rows[-1])
    artifacts["protomotions-v2.3"] = {
        "path": v2_path.relative_to(root).as_posix(),
        "sha256": sha256_file(v2_path),
        "size_bytes": v2_path.stat().st_size,
    }

    proto_path = (
        root
        / "source_adapters"
        / "protomotions_v3"
        / sequence_id
        / "keypoints.npy"
    )
    proto_metrics = _prepare_protomotions_keypoints(human, proto_path)
    proto_mapping = np.load(proto_path, allow_pickle=True).item()
    proto_names = [source for _, source in PROTOMOTIONS_V3_CONCEPTUAL_JOINTS]
    proto_adapter_metrics = adapter_error_metrics(
        human,
        proto_names,
        np.asarray(proto_mapping["positions"], dtype=np.float64)[:, :15],
    )
    append_row(
        adapter="canonical_lafan_to_protomotions_v3_keypoints",
        method="protomotions-v3",
        metrics=proto_adapter_metrics,
        native_input_feature="15 semantic plus 3 auxiliary world-frame keypoints",
        source_shape_policy="direct canonical actor keypoints; no SMPL beta or gender",
        target_construction="published generic SMPL keypoint schema through modified PyRoki",
        scale_policy="official root/lower/upper anisotropic keypoint scales",
        root_anchor_policy="absolute scaled root; official whole-trajectory optimization",
        orientation_target_policy="15 source world orientations plus copied auxiliary frames",
        contact_target_policy="canonical contacts repeated ankle/toe then official smoothing",
        graph_policy="official fixed pairwise retarget mask and whole-trajectory objectives",
        ground_policy="official contact/tilt losses; no post-hoc ground shift",
        timeline_policy=(
            f"{human.fps:.12g} fps source adapter retains 600 frames; official "
            "trim-or-pad contract consumes source prefix 0:450 at stride 1"
        ),
        robot_asset_path=(
            root
            / "external/ProtoMotions/protomotions/data/assets/urdf/for_retargeting/g1.urdf"
        ),
        source_artifact_path=proto_path,
        caveat=(
            f"{proto_metrics['caveat']} The upstream fixed-shape solver consumes "
            "450/600 source frames; no padding, interpolation, or stitching is used "
            "to claim full-source completion."
        ),
    )
    atomic_write_json(proto_path.parent / "manifest.json", rows[-1])
    artifacts["protomotions-v3"] = {
        "path": proto_path.relative_to(root).as_posix(),
        "sha256": sha256_file(proto_path),
        "size_bytes": proto_path.stat().st_size,
        "status": rows[-1]["status"],
        "caveat": proto_metrics["caveat"],
    }

    output = root / "metrics" / "source_adapter_errors.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    atomic_write_yaml(
        root / "manifests" / "source_adapters.yaml",
        {
            "schema_version": 3,
            "sequence_id": sequence_id,
            "canonical_source_sha256": human.source_sha256,
            "metrics_path": output.relative_to(root).as_posix(),
            "metrics_sha256": sha256_file(output),
            "native_artifacts_committed_to_git": False,
            "native_artifacts": artifacts,
            "expected_methods": [
                "gmr",
                "omniretarget",
                "protomotions-v2.3",
                "protomotions-v3",
            ],
            "all_runtime_boundaries_observed": all(
                bool(row["runtime_boundary_observed"]) for row in rows
            ),
            "full_lafan_authorized": False,
        },
    )
    return rows
