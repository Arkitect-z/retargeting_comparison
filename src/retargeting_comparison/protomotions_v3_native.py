"""Native-environment runner for ProtoMotions v3's unchanged G1 solver."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .constants import G1_JOINT_NAMES
from .io_utils import atomic_write_json
from .native_target_capture import tensor_sha256
from .protomotions_v3 import actuated_urdf_joints
from .rotations import quaternion_wxyz_to_matrix
from .schemas import CanonicalHuman
from .stage1_timing_campaign import assert_formal_campaign_ownership


PROTOMOTIONS_V3_SOURCE_JOINTS = (
    "Hips",
    "LeftUpLeg",
    "RightUpLeg",
    "LeftLeg",
    "RightLeg",
    "LeftFoot",
    "RightFoot",
    "LeftToe",
    "RightToe",
    "LeftArm",
    "RightArm",
    "LeftForeArm",
    "RightForeArm",
    "LeftHand",
    "RightHand",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--keypoints", required=True)
    parser.add_argument(
        "--source",
        required=True,
        help=(
            "Canonical human NPZ. Formal timing requires this so the timed "
            "boundary starts at the canonical source file instead of a cached adapter."
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--timing-json", required=True)
    parser.add_argument("--target-raw-frames", type=int, required=True)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=3)
    parser.add_argument("--root-scale-multiplier", type=float, default=1.0)
    parser.add_argument("--local-scale-multiplier", type=float, default=1.0)
    parser.add_argument("--capture-runtime-witness", action="store_true")
    return parser


def _load_official(script: Path):
    spec = importlib.util.spec_from_file_location("rtcmp_protomotions_v3_official", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import official ProtoMotions script: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pin_to_one_cpu() -> list[int]:
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("Formal ProtoMotions timing requires Linux CPU affinity")
    allowed = sorted(os.sched_getaffinity(0))
    if not allowed:
        raise RuntimeError("Formal ProtoMotions timing has no allowed CPU")
    selected = [int(allowed[0])]
    os.sched_setaffinity(0, set(selected))
    if sorted(os.sched_getaffinity(0)) != selected:
        raise RuntimeError("Could not freeze ProtoMotions CPU affinity")
    return selected


def _canonical_qpos_in_memory(
    transforms_wxyz_xyz: np.ndarray,
    joints: np.ndarray,
    native_joint_names: list[str],
) -> np.ndarray:
    order = np.asarray(
        [native_joint_names.index(name) for name in G1_JOINT_NAMES],
        dtype=np.int64,
    )
    quaternion = np.asarray(transforms_wxyz_xyz[:, :4], dtype=np.float64).copy()
    norms = np.linalg.norm(quaternion, axis=1)
    if np.any(norms < 1e-12):
        raise RuntimeError("ProtoMotions produced a zero root quaternion")
    quaternion /= norms[:, None]
    for index in range(1, len(quaternion)):
        if float(np.dot(quaternion[index - 1], quaternion[index])) < 0.0:
            quaternion[index] *= -1.0
    return np.concatenate(
        (
            np.asarray(transforms_wxyz_xyz[:, 4:], dtype=np.float64),
            quaternion,
            np.asarray(joints[:, order], dtype=np.float64),
        ),
        axis=1,
    )


def _canonical_human_mapping(human: CanonicalHuman) -> dict[str, object]:
    """Construct the audited 15+3 native schema without an intermediate file."""

    source_index = {
        str(name): index for index, name in enumerate(human.joint_names.astype(str))
    }
    missing = [name for name in PROTOMOTIONS_V3_SOURCE_JOINTS if name not in source_index]
    if missing:
        raise ValueError(f"Canonical source lacks ProtoMotions joints: {missing}")
    indices = [source_index[name] for name in PROTOMOTIONS_V3_SOURCE_JOINTS]
    positions = np.asarray(human.world_positions[:, indices], dtype=np.float64).copy()
    orientations = quaternion_wxyz_to_matrix(human.world_rotations[:, indices])
    auxiliary = []
    auxiliary_orientations = []
    for index in (13, 14, 0):
        offset = np.asarray([0.2, 0.0, 0.0], dtype=np.float64)
        auxiliary.append(
            positions[:, index]
            + np.einsum("tij,j->ti", orientations[:, index], offset)
        )
        auxiliary_orientations.append(orientations[:, index])
    positions = np.concatenate((positions, np.stack(auxiliary, axis=1)), axis=1)
    orientations = np.concatenate(
        (orientations, np.stack(auxiliary_orientations, axis=1)), axis=1
    )
    contacts = np.asarray(human.foot_contact_labels, dtype=np.int64)
    return {
        "positions": positions,
        "orientations": orientations,
        "left_foot_contacts": np.repeat(contacts[:, 0:1], 2, axis=1),
        "right_foot_contacts": np.repeat(contacts[:, 1:2], 2, axis=1),
        "fps": float(human.fps),
    }


def _scale_mapping(
    mapping: dict[str, object],
    *,
    root_multiplier: float,
    local_multiplier: float,
) -> dict[str, object]:
    if root_multiplier <= 0.0 or local_multiplier <= 0.0:
        raise ValueError("ProtoMotions scale multipliers must be positive")
    value = dict(mapping)
    # Preserve the declared adapter witness bit-for-bit for the native policy.
    # Reconstructing ``root + (point - root)`` at 1.0 introduces harmless
    # floating-point roundoff, but that breaks the intentionally exact witness
    # equality check before the solver.
    if root_multiplier == 1.0 and local_multiplier == 1.0:
        return value
    positions = np.asarray(mapping["positions"], dtype=np.float64)
    root = positions[:, 0]
    scaled_root = root[0] + (root - root[0]) * root_multiplier
    value["positions"] = scaled_root[:, None, :] + (
        positions - root[:, None, :]
    ) * local_multiplier
    return value


def _mapping_hashes(mapping: dict[str, object]) -> dict[str, str]:
    contacts = np.stack(
        (
            np.asarray(mapping["left_foot_contacts"], dtype=np.float64),
            np.asarray(mapping["right_foot_contacts"], dtype=np.float64),
        ),
        axis=-1,
    )
    return {
        "positions_sha256": tensor_sha256(
            np.asarray(mapping["positions"], dtype=np.float64)
        ),
        "orientations_sha256": tensor_sha256(
            np.asarray(mapping["orientations"], dtype=np.float64)
        ),
        "contacts_sha256": tensor_sha256(contacts),
    }


class _MappingProxy:
    def __init__(self, value: dict[str, object]):
        self.value = value

    def item(self) -> dict[str, object]:
        return self.value


def _official_load_from_memory(
    official: object,
    mapping: dict[str, object],
    target_raw_frames: int,
):
    """Call the unchanged official loader while replacing only its file read."""

    marker = "__rtcmp_canonical_human_in_memory__"
    original_load = official.onp.load

    def load(path: object, *args: object, **kwargs: object):
        if str(path) == marker:
            return _MappingProxy(mapping)
        return original_load(path, *args, **kwargs)

    official.onp.load = load
    try:
        return official.load_motion_data(marker, "smpl", 1, target_raw_frames, 30.0)
    finally:
        official.onp.load = original_load


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.target_raw_frames < 1 or args.warmup_runs < 0 or args.measured_runs < 1:
        raise SystemExit("Invalid target or timing repetitions")
    root = Path(args.repo_root).resolve()
    assert_formal_campaign_ownership(root)
    keypoints = Path(args.keypoints).resolve()
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    timing_path = Path(args.timing_json).resolve()
    cpu_affinity = _pin_to_one_cpu()
    requested_platform = os.environ.get("JAX_PLATFORMS", "")
    if requested_platform not in {"cpu", "cuda"}:
        raise RuntimeError(
            "Native ProtoMotions requires an explicit JAX_PLATFORMS=cpu|cuda contract"
        )
    proto = root / "external/ProtoMotions"
    script = proto / "pyroki/batch_retarget_to_g1_from_keypoints.py"
    urdf_path = proto / "protomotions/data/assets/urdf/for_retargeting/g1.urdf"
    mesh_dir = proto / "protomotions/data/assets/mesh/G1"

    initialization_started = time.perf_counter()
    official = _load_official(script)
    import jax
    import jax.numpy as jnp
    import pyroki as pk
    import yourdfpy

    actual_backend = jax.default_backend()
    backend_matches_request = (
        actual_backend == "cpu"
        if requested_platform == "cpu"
        else actual_backend in {"cuda", "gpu"}
    )
    if not backend_matches_request:
        raise RuntimeError(
            "ProtoMotions JAX backend differs from the requested native environment"
        )
    runtime_environment_contract = {
        "requested_jax_platform": requested_platform,
        "actual_jax_backend": actual_backend,
        "jax_devices": [str(device) for device in jax.devices()],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "jax_skip_cuda_constraints_check": os.environ.get(
            "JAX_SKIP_CUDA_CONSTRAINTS_CHECK"
        ),
        "contract_satisfied": True,
    }

    urdf = yourdfpy.URDF.load(str(urdf_path), mesh_dir=str(mesh_dir))
    robot = pk.Robot.from_urdf(urdf)
    native_joint_names = [
        joint.name for joint in actuated_urdf_joints(urdf_path)
    ]
    if set(native_joint_names) != set(G1_JOINT_NAMES):
        raise RuntimeError("ProtoMotions native joint set is not canonical G1-29")
    official.G1_LINK_NAMES = list(robot.links.names)
    official.human_retarget_names, official.g1_joint_retarget_indices = (
        official.get_humanoid_retarget_indices()
    )
    n_retarget = len(official.g1_joint_retarget_indices)
    retarget_mask = jnp.zeros((n_retarget, n_retarget))
    for link_a, link_b, weight in official.direct_pairs:
        index_a = official.human_retarget_names.index(link_a)
        index_b = official.human_retarget_names.index(link_b)
        retarget_mask = retarget_mask.at[index_a, index_b].set(weight)
        retarget_mask = retarget_mask.at[index_b, index_a].set(weight)
    weights = official.RetargetingWeights(
        local_alignment=1.0,
        global_alignment=4.0,
        root_smoothness=1.0,
        joint_smoothness=4.0,
        self_collision=0.0,
        joint_rest_penalty=1.0,
        joint_vel_limit=50.0,
        foot_contact=30.0,
        foot_tilt=1.0,
    )
    # The declared adapter artifact remains an independently hashed witness,
    # but is never the formal end-to-end timing start.  Prove once that the
    # in-memory constructor is byte-equivalent before entering repetitions.
    audit_human = CanonicalHuman.load(source)
    audit_mapping = _canonical_human_mapping(audit_human)
    audit_mapping = _scale_mapping(
        audit_mapping,
        root_multiplier=args.root_scale_multiplier,
        local_multiplier=args.local_scale_multiplier,
    )
    with keypoints.open("rb") as stream:
        declared_mapping = np.load(stream, allow_pickle=True).item()
    for key in (
        "positions",
        "orientations",
        "left_foot_contacts",
        "right_foot_contacts",
        "fps",
    ):
        if not np.array_equal(
            np.asarray(audit_mapping[key]), np.asarray(declared_mapping[key])
        ):
            raise RuntimeError(
                f"In-memory canonical adapter differs from declared witness: {key}"
            )
    declared_mapping_hashes = _mapping_hashes(audit_mapping)
    initialization_time = time.perf_counter() - initialization_started

    repetitions = []
    selected = None
    total = args.warmup_runs + args.measured_runs
    for repetition in range(total):
        role = "warmup" if repetition < args.warmup_runs else "measured"
        started_at_utc = datetime.now(timezone.utc).isoformat()
        started = time.perf_counter()
        source_load_started = time.perf_counter()
        human = CanonicalHuman.load(source)
        source_load_time = time.perf_counter() - source_load_started
        adapter_started = time.perf_counter()
        mapping = _canonical_human_mapping(human)
        mapping = _scale_mapping(
            mapping,
            root_multiplier=args.root_scale_multiplier,
            local_multiplier=args.local_scale_multiplier,
        )
        adapter_time = time.perf_counter() - adapter_started
        official_preprocess_started = time.perf_counter()
        (
            target_keypoints,
            target_orientations,
            left_contact,
            right_contact,
            frames,
            input_fps,
        ) = _official_load_from_memory(
            official,
            mapping,
            args.target_raw_frames,
        )
        official_preprocess_time = time.perf_counter() - official_preprocess_started
        native_input_ready_at_utc = datetime.now(timezone.utc).isoformat()
        solve_started = time.perf_counter()
        transforms, joints = official.solve_retargeting(
            robot=robot,
            robot_coll=None,
            target_keypoints=target_keypoints,
            target_orientations=target_orientations,
            left_foot_contact=left_contact,
            right_foot_contact=right_contact,
            g1_joint_retarget_indices=official.g1_joint_retarget_indices,
            g1_retarget_mask=retarget_mask,
            weights=weights,
            subsample_factor=1,
            input_fps=input_fps,
        )
        # Converting to NumPy is the explicit synchronization boundary for JAX.
        transforms_np = np.asarray(transforms.wxyz_xyz[:frames])
        joints_np = np.asarray(joints[:frames])
        native_core_time = time.perf_counter() - solve_started
        canonical_conversion_started = time.perf_counter()
        canonical_qpos = _canonical_qpos_in_memory(
            transforms_np, joints_np, native_joint_names
        )
        canonical_conversion_time = time.perf_counter() - canonical_conversion_started
        if canonical_qpos.shape != (int(frames), 36):
            raise RuntimeError("ProtoMotions canonical in-memory output is not [T,36]")
        output_in_memory_wall = time.perf_counter() - started
        output_in_memory_at_utc = datetime.now(timezone.utc).isoformat()
        qpos_sha256 = tensor_sha256(canonical_qpos)
        content_sha256 = hashlib.sha256(
            json.dumps(
                {
                    "qpos_sha256": qpos_sha256,
                    "frames": int(frames),
                    "fps_hex": float(
                        official.subsampled_fps(input_fps, 1)
                    ).hex(),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        witness_hash_started = time.perf_counter()
        runtime_capture = None
        if args.capture_runtime_witness:
            runtime_capture = {
                "boundary": (
                    "official load_motion_data return values passed unchanged "
                    "to solve_retargeting"
                ),
                "target_keypoints_sha256": tensor_sha256(
                    np.asarray(target_keypoints, dtype=np.float64)
                ),
                "target_keypoints_shape": list(np.asarray(target_keypoints).shape),
                "target_orientations_sha256": tensor_sha256(
                    np.asarray(target_orientations, dtype=np.float64)
                ),
                "foot_contacts_sha256": tensor_sha256(
                    np.stack(
                        (
                            np.asarray(left_contact, dtype=np.float64),
                            np.asarray(right_contact, dtype=np.float64),
                        ),
                        axis=-1,
                    )
                ),
                "observed_during_solver_run": True,
                "capture_execution_role": "independent_cold_process",
            }
        witness_hash_time = time.perf_counter() - witness_hash_started
        repetition_output = (
            output.parent / "timing_artifacts" / f"{role}_{repetition:02d}.npz"
        )
        repetition_output.parent.mkdir(parents=True, exist_ok=True)
        artifact_write_started = time.perf_counter()
        np.savez_compressed(
            repetition_output,
            base_frame_pos=transforms_np[:, 4:],
            base_frame_wxyz=transforms_np[:, :4],
            joint_angles=joints_np,
            fps=official.subsampled_fps(input_fps, 1),
            source_fps=input_fps,
            subsample_factor=1,
        )
        artifact_write_time = time.perf_counter() - artifact_write_started
        repetitions.append(
            {
                "index": repetition,
                "role": role,
                "started_at_utc": started_at_utc,
                "output_in_memory_at_utc": output_in_memory_at_utc,
                "wall_time_s": output_in_memory_wall,
                "frame_count": int(frames),
                "timing_granularity": "whole trajectory",
                "canonical_source_load_time_s": source_load_time,
                "canonical_to_native_in_memory_time_s": adapter_time,
                "official_preprocess_time_s": official_preprocess_time,
                "native_core_s": native_core_time,
                "native_total_s": native_core_time,
                "canonical_conversion_time_s": canonical_conversion_time,
                "steady_end_to_end_total_s": output_in_memory_wall,
                "timing_boundary": "canonical_source_file_to_canonical_g1_in_memory",
                "timing_boundary_detail": (
                    "canonical NPZ load, audited in-memory 15+3 adapter, unchanged "
                    "official preprocessing, solver/JAX synchronization, and canonical "
                    "float64/order/quaternion conversion"
                ),
                "native_boundary": "native_input_ready_to_native_g1_in_memory",
                "native_input_ready_at_utc": native_input_ready_at_utc,
                "runtime_pre_solver_capture": runtime_capture,
                "runtime_witness_enabled": bool(args.capture_runtime_witness),
                "runtime_witness_hash_time_s_excluded": witness_hash_time,
                "canonical_qpos_sha256": qpos_sha256,
                "canonical_g1_content_sha256": content_sha256,
                "intermediate_artifact_write_time_s_excluded": artifact_write_time,
                "timing_artifact": str(repetition_output),
                "cpu_affinity": cpu_affinity,
                "thread_limit": 1,
            }
        )
        if role == "measured":
            selected = (transforms_np, joints_np, frames, input_fps)
    assert selected is not None
    transforms_np, joints_np, frames, input_fps = selected
    if transforms_np.shape != (frames, 7) or joints_np.shape != (frames, 29):
        raise RuntimeError("Official ProtoMotions output has an unexpected shape")
    if not np.isfinite(transforms_np).all() or not np.isfinite(joints_np).all():
        raise RuntimeError("Official ProtoMotions output contains NaN/Inf")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f".{output.name}.tmp.npz")
    final_artifact_write_started = time.perf_counter()
    np.savez_compressed(
        temporary_output,
        base_frame_pos=transforms_np[:, 4:],
        base_frame_wxyz=transforms_np[:, :4],
        joint_angles=joints_np,
        fps=official.subsampled_fps(input_fps, 1),
        source_fps=input_fps,
        subsample_factor=1,
    )
    os.replace(temporary_output, output)
    final_artifact_write_time = time.perf_counter() - final_artifact_write_started
    atomic_write_json(
        timing_path,
        {
            "initialization_time_s": initialization_time,
            "canonical_source_load_time_s": [
                item["canonical_source_load_time_s"] for item in repetitions
            ],
            "warmup_runs": args.warmup_runs,
            "measured_runs": args.measured_runs,
            "repetitions": repetitions,
            "jax_sync_boundary": "numpy conversion of full solution",
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
            "runtime_environment_contract": runtime_environment_contract,
            "cpu_affinity": cpu_affinity,
            "thread_limit": 1,
            "final_native_artifact_write_time_s_excluded": final_artifact_write_time,
            "formal_boundary": "canonical_source_file_to_canonical_g1_in_memory",
            "native_boundary": "native_input_ready_to_native_g1_in_memory",
            "artifact_writes_excluded": True,
            "runtime_witness_enabled": bool(args.capture_runtime_witness),
            "declared_adapter_mapping_hashes": declared_mapping_hashes,
            "declared_adapter_file": str(keypoints),
            "canonical_source_file": str(source),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
