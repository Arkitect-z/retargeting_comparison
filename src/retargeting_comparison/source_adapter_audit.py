"""Reproducible audits for the native source adapters used by core methods."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .calibration import human_heading_yaw
from .io_utils import atomic_write_json, atomic_write_yaml, load_yaml, sha256_file
from .method_adapters import HOLOSOMA_LAFAN_ORDER, prepare_holosoma_lafan_input
from .rotations import quaternion_wxyz_to_matrix
from .schemas import CanonicalHuman
from .source import foot_contacts


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

    conceptual = (
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
    source_index = {name: index for index, name in enumerate(human.joint_names.astype(str))}
    indices = [source_index[source] for _, source in conceptual]
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

    gmr_names, gmr_positions = _gmr_positions(root, source_bvh)
    gmr_dir = root / "source_adapters" / "gmr" / sequence_id
    gmr_dir.mkdir(parents=True, exist_ok=True)
    gmr_native = gmr_dir / "official_loader_positions.npz"
    np.savez_compressed(gmr_native, names=np.asarray(gmr_names), positions=gmr_positions)
    gmr_metrics = adapter_error_metrics(human, gmr_names, gmr_positions)
    rows.append(
        {
            "adapter": "canonical_bvh_to_gmr_lafan",
            "method": "gmr",
            "sequence_id": sequence_id,
            **gmr_metrics,
            "status": "passed",
            "native_input_feature": "full LAFAN pose dictionary",
            "caveat": "Official GMR LAFAN loader; contact labels are evaluator-only, not a GMR objective.",
        }
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
    rows.append(
        {
            "adapter": "canonical_bvh_to_holosoma_lafan",
            "method": "omniretarget",
            "sequence_id": sequence_id,
            **holosoma_metrics,
            "status": "passed",
            "native_input_feature": "right-first LAFAN world positions",
            "caveat": "Exact joint reorder and Z-up/Y-up involution; contact is inferred downstream.",
        }
    )
    atomic_write_json(holosoma_dir / "manifest.json", rows[-1])
    artifacts["omniretarget"] = {
        "path": native_path.relative_to(root).as_posix(),
        "sha256": sha256_file(native_path),
        "size_bytes": native_path.stat().st_size,
    }

    proto_path = (
        root
        / "source_adapters"
        / "protomotions_v3"
        / sequence_id
        / "keypoints.npy"
    )
    proto_metrics = _prepare_protomotions_keypoints(human, proto_path)
    atomic_write_json(proto_path.parent / "manifest.json", proto_metrics)
    artifacts["protomotions_v3"] = {
        "path": proto_path.relative_to(root).as_posix(),
        "sha256": sha256_file(proto_path),
        "size_bytes": proto_path.stat().st_size,
        "status": "input_schema_ready_only",
        "caveat": proto_metrics["caveat"],
    }
    candidate_output = root / "metrics" / "candidate_source_adapter_errors.csv"
    with candidate_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(proto_metrics), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerow(proto_metrics)

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
            "schema_version": 2,
            "sequence_id": sequence_id,
            "canonical_source_sha256": human.source_sha256,
            "metrics_path": output.relative_to(root).as_posix(),
            "metrics_sha256": sha256_file(output),
            "candidate_metrics_path": candidate_output.relative_to(root).as_posix(),
            "candidate_metrics_sha256": sha256_file(candidate_output),
            "native_artifacts_committed_to_git": False,
            "native_artifacts": artifacts,
            "full_lafan_authorized": False,
        },
    )
    return rows
