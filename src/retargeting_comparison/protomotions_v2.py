"""Frozen ProtoMotions v2.3 Mink adapter for the Stage-1 LAFAN pilot.

ProtoMotions v2.3 exposes its Mink retargeter through an AMASS/SMPL-X loader.
This module keeps the official G1 asset, target set, scaling, costs, solver,
limits, warm-up, and post-retarget ground alignment while adapting the frozen
canonical LAFAN world-joint package to those same semantic targets.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import mink
import mujoco
import numpy as np

from .constants import G1_JOINT_NAMES, MIN_COMPLETION_RATIO
from .io_utils import atomic_write_json, load_yaml, sha256_file
from .rotations import quaternion_wxyz_to_matrix
from .schemas import CanonicalG1, CanonicalHuman


UPSTREAM_MINK_RETARGET_SHA256 = (
    "356087a34d5e68088db0bfa0c15561f506b158efcf013083b19a7bdc011e5f31"
)
UPSTREAM_SMPLX_POSE_INPUT_LINE = 611
UPSTREAM_SMPLX_POSE_CONTRACT_LINES = (637, 644)
SMPLX_AMASS_POSE_DIM = 165
SMPLX_MINK_POSE_DIM = 156
SMPLX_MINK_JOINT_COUNT = 52


def smplx_165d_to_mink_52x3(poses: np.ndarray) -> np.ndarray:
    """Apply the frozen v2.3 SMPL-X pose-width contract exactly.

    The upstream AMASS path consumes ``poses[T,165]``, removes the nine values
    at ``[66:75]``, concatenates ``[:66]`` with ``[75:]``, and reshapes the
    resulting 156 values to ``[T,52,3]`` before its SMPL-to-MuJoCo reorder.
    This function intentionally stops before that separate name-based reorder.
    """
    value = np.asarray(poses)
    if value.ndim != 2 or value.shape[1] != SMPLX_AMASS_POSE_DIM:
        raise ValueError("ProtoMotions v2.3 requires SMPL-X poses with shape [T,165]")
    compact = np.concatenate((value[:, :66], value[:, 75:]), axis=-1)
    if compact.shape != (len(value), SMPLX_MINK_POSE_DIM):
        raise RuntimeError("ProtoMotions v2.3 SMPL-X slicing did not produce 156 values")
    return compact.reshape(len(value), SMPLX_MINK_JOINT_COUNT, 3)


def audit_smplx_pose_contract(
    repo_root: str | Path,
    config_path: str | Path = "configs/protomotions_v2.yaml",
) -> dict[str, Any]:
    """Bind the 165D contract to the frozen upstream file and source lines."""
    root = Path(repo_root).resolve()
    config = load_yaml(root / config_path)
    implementation = (
        root
        / config["upstream"]["worktree"]
        / config["upstream"]["implementation"]
    )
    actual_sha256 = sha256_file(implementation)
    expected_sha256 = str(config["upstream"]["implementation_sha256"])
    if actual_sha256 != expected_sha256 or actual_sha256 != UPSTREAM_MINK_RETARGET_SHA256:
        raise RuntimeError("Frozen v2.3 Mink file does not match the SMPL-X contract audit")
    lines = implementation.read_text(encoding="utf-8").splitlines()
    line_start, line_end = UPSTREAM_SMPLX_POSE_CONTRACT_LINES
    if len(lines) < line_end:
        raise RuntimeError("Frozen v2.3 Mink file is shorter than the contract evidence")
    evidence_lines = {
        "amass_pose_field": {
            "line": UPSTREAM_SMPLX_POSE_INPUT_LINE,
            "text": lines[UPSTREAM_SMPLX_POSE_INPUT_LINE - 1].strip(),
        },
        "slice_and_reshape": {
            "line_start": line_start,
            "line_end": line_end,
            "text": "\n".join(lines[line_start - 1 : line_end]),
        },
    }
    if evidence_lines["amass_pose_field"]["text"] != 'amass_pose = motion_data["poses"]':
        raise RuntimeError("Frozen v2.3 AMASS pose-field evidence changed")
    contract_text = str(evidence_lines["slice_and_reshape"]["text"])
    required_tokens = (
        'motion_data["pose_aa"][:, :66]',
        'motion_data["pose_aa"][:, 75:]',
        "pose_aa.reshape(batch_size, 52, 3)",
    )
    if not all(token in contract_text for token in required_tokens):
        raise RuntimeError("Frozen v2.3 165D slice/reshape evidence changed")
    return {
        "contract": "165 -> concatenate [:66] and [75:] -> 156 -> [T,52,3]",
        "input_shape": ["T", SMPLX_AMASS_POSE_DIM],
        "removed_component_slice": [66, 75],
        "compact_shape": ["T", SMPLX_MINK_POSE_DIM],
        "reshaped_output": ["T", SMPLX_MINK_JOINT_COUNT, 3],
        "reorder_boundary": "before upstream smpl_2_mujoco name-based reorder",
        "upstream_path": str(implementation.relative_to(root)),
        "upstream_sha256": actual_sha256,
        "upstream_commit": str(config["upstream"]["commit"]),
        "line_evidence": evidence_lines,
        "line_evidence_sha256": hashlib.sha256(
            (
                str(evidence_lines["amass_pose_field"]["text"])
                + "\n"
                + contract_text
            ).encode("utf-8")
        ).hexdigest(),
    }


def _joint_names(model: mujoco.MjModel) -> tuple[str, ...]:
    return tuple(
        str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id))
        for joint_id in range(1, model.njnt)
    )


def _body_position(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise ValueError(f"Robot body is missing: {name}")
    return np.asarray(data.xpos[body_id], dtype=np.float64).copy()


def _git_head(worktree: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(worktree), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"Cannot verify ProtoMotions worktree {worktree}") from error


def _package_versions(names: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for name in names:
        try:
            values[name] = version(name)
        except PackageNotFoundError:
            values[name] = "not-installed"
    return values


def verify_frozen_installation(
    repo_root: str | Path,
    config_path: str | Path = "configs/protomotions_v2.yaml",
    *,
    strict_dependencies: bool = True,
) -> dict[str, Any]:
    """Verify commit, files, dimensions, joint order, and frozen dependencies."""
    root = Path(repo_root).resolve()
    path = root / config_path
    config = load_yaml(path)
    worktree = root / config["upstream"]["worktree"]
    implementation = worktree / config["upstream"]["implementation"]
    native_xml = root / config["native_robot"]["xml"]
    canonical_xml = root / config["canonical_robot"]["xml"]
    required = (worktree, implementation, native_xml, canonical_xml)
    missing = [str(value) for value in required if not value.exists()]
    if missing:
        raise FileNotFoundError(f"Frozen ProtoMotions integration files missing: {missing}")

    head = _git_head(worktree)
    if head != config["upstream"]["commit"]:
        raise RuntimeError(f"ProtoMotions commit mismatch: {head}")
    hashes = {
        "implementation": sha256_file(implementation),
        "native_robot": sha256_file(native_xml),
        "canonical_robot": sha256_file(canonical_xml),
    }
    expected_hashes = {
        "implementation": config["upstream"]["implementation_sha256"],
        "native_robot": config["native_robot"]["xml_sha256"],
        "canonical_robot": config["canonical_robot"]["xml_sha256"],
    }
    if hashes != expected_hashes:
        raise RuntimeError(f"Frozen ProtoMotions file hash mismatch: {hashes}")

    model = mujoco.MjModel.from_xml_path(str(native_xml))
    dimensions = {"nq": model.nq, "nv": model.nv, "nu": model.nu}
    expected_dimensions = {
        name: int(config["native_robot"][name]) for name in ("nq", "nv", "nu")
    }
    if dimensions != expected_dimensions:
        raise RuntimeError(f"Native G1 dimensions mismatch: {dimensions}")
    joints = _joint_names(model)
    configured_joints = tuple(config["native_robot"]["joint_order"])
    if joints != configured_joints or joints != tuple(G1_JOINT_NAMES):
        raise RuntimeError("Native G1 joint order is not canonical G1-29")

    package_names = list(config["frozen_environment"]["packages"])
    actual_packages = _package_versions(package_names)
    expected_packages = {
        name: str(value)
        for name, value in config["frozen_environment"]["packages"].items()
    }
    python_version = platform.python_version()
    dependency_mismatches = {
        name: {"expected": expected_packages[name], "actual": actual_packages[name]}
        for name in package_names
        if actual_packages[name] != expected_packages[name]
    }
    expected_python = str(config["frozen_environment"]["python"])
    if python_version != expected_python:
        dependency_mismatches["python"] = {
            "expected": expected_python,
            "actual": python_version,
        }
    if strict_dependencies and dependency_mismatches:
        raise RuntimeError(f"Frozen dependency mismatch: {dependency_mismatches}")
    smplx_pose_contract = audit_smplx_pose_contract(root, config_path)
    return {
        "upstream_commit": head,
        "hashes": hashes,
        "dimensions": dimensions,
        "joint_order": list(joints),
        "python": python_version,
        "packages": actual_packages,
        "dependency_mismatches": dependency_mismatches,
        "smplx_pose_contract": smplx_pose_contract,
    }


def audit_robot_compatibility(
    repo_root: str | Path,
    config_path: str | Path = "configs/protomotions_v2.yaml",
    *,
    random_samples: int = 8,
    seed: int = 20260722,
) -> dict[str, Any]:
    """Quantify native-vs-canonical G1 limit and forward-kinematics mismatch."""
    root = Path(repo_root).resolve()
    config = load_yaml(root / config_path)
    native_path = root / config["native_robot"]["xml"]
    canonical_path = root / config["canonical_robot"]["xml"]
    native = mujoco.MjModel.from_xml_path(str(native_path))
    canonical = mujoco.MjModel.from_xml_path(str(canonical_path))
    native_order = _joint_names(native)
    canonical_order = _joint_names(canonical)
    order_matches = native_order == canonical_order == tuple(G1_JOINT_NAMES)

    native_ranges = np.asarray(native.jnt_range[1:], dtype=np.float64)
    canonical_ranges = np.asarray(canonical.jnt_range[1:], dtype=np.float64)
    range_delta = np.abs(native_ranges - canonical_ranges)
    lower = np.maximum(native_ranges[:, 0], canonical_ranges[:, 0])
    upper = np.minimum(native_ranges[:, 1], canonical_ranges[:, 1])
    if np.any(lower > upper):
        raise RuntimeError("Native and canonical G1 have disjoint joint limits")

    # Native v2.3 adds an artificial point body named ``head``.  The canonical
    # scene has no matching body, so compare only bodies with the same semantics.
    requested = [target["robot_body"] for target in config["source_adapter"]["targets"]]
    common_bodies = [
        name
        for name in requested
        if name != "head"
        and mujoco.mj_name2id(native, mujoco.mjtObj.mjOBJ_BODY, name) >= 0
        and mujoco.mj_name2id(canonical, mujoco.mjtObj.mjOBJ_BODY, name) >= 0
    ]
    rng = np.random.default_rng(seed)
    joint_samples = [np.zeros(len(G1_JOINT_NAMES), dtype=np.float64)]
    joint_samples.extend(rng.uniform(lower, upper) for _ in range(random_samples))
    errors: list[float] = []
    per_sample_max: list[float] = []
    native_data = mujoco.MjData(native)
    canonical_data = mujoco.MjData(canonical)
    for joints in joint_samples:
        for model, data in ((native, native_data), (canonical, canonical_data)):
            data.qpos[:] = model.qpos0
            data.qpos[:3] = 0.0
            data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
            data.qpos[7:] = joints
            mujoco.mj_forward(model, data)
        native_root = _body_position(native, native_data, "pelvis")
        canonical_root = _body_position(canonical, canonical_data, "pelvis")
        sample_errors = []
        for body in common_bodies:
            native_point = _body_position(native, native_data, body) - native_root
            canonical_point = _body_position(canonical, canonical_data, body) - canonical_root
            sample_errors.append(float(np.linalg.norm(native_point - canonical_point)))
        errors.extend(sample_errors)
        per_sample_max.append(max(sample_errors, default=0.0))

    return {
        "native_xml_sha256": sha256_file(native_path),
        "canonical_xml_sha256": sha256_file(canonical_path),
        "joint_order_matches": order_matches,
        "native_joint_order": list(native_order),
        "canonical_joint_order": list(canonical_order),
        "joint_limit_exact_match_count": int(np.all(range_delta <= 1e-12, axis=1).sum()),
        "joint_limit_mismatch_count": int(np.any(range_delta > 1e-12, axis=1).sum()),
        "joint_limit_max_abs_difference_rad": float(range_delta.max(initial=0.0)),
        "fk_common_bodies": common_bodies,
        "fk_sample_count": len(joint_samples),
        "fk_root_aligned_mean_error_m": float(np.mean(errors)) if errors else 0.0,
        "fk_root_aligned_max_error_m": float(max(per_sample_max, default=0.0)),
        "excluded_body_mismatch": {
            "native_only": "head",
            "reason": "ProtoMotions v2.3 adds a 0.1-mm marker body; canonical Holosoma has no body named head",
        },
    }


class ProtoMotionsV2Retargeter:
    """Run the frozen v2.3 Mink operating point on canonical LAFAN targets."""

    def __init__(
        self,
        repo_root: str | Path,
        config_path: str | Path = "configs/protomotions_v2.yaml",
        *,
        strict_dependencies: bool = True,
        root_scale_multiplier: float = 1.0,
        local_scale_multiplier: float = 1.0,
    ):
        start = time.perf_counter()
        self.repo_root = Path(repo_root).resolve()
        self.config_path = self.repo_root / config_path
        self.config = load_yaml(self.config_path)
        if root_scale_multiplier <= 0.0 or local_scale_multiplier <= 0.0:
            raise ValueError("Scale multipliers must be positive")
        self.root_scale_multiplier = float(root_scale_multiplier)
        self.local_scale_multiplier = float(local_scale_multiplier)
        self.installation = verify_frozen_installation(
            self.repo_root, config_path, strict_dependencies=strict_dependencies
        )
        self.robot_audit = audit_robot_compatibility(self.repo_root, config_path)
        native_path = self.repo_root / self.config["native_robot"]["xml"]
        self.model = mujoco.MjModel.from_xml_path(str(native_path))
        self.algorithm = self.config["algorithm"]
        qpos = np.zeros(self.model.nq, dtype=np.float64)
        qpos[3] = 1.0
        self.configuration = mink.Configuration(self.model, q=qpos)
        self.frame_tasks: list[tuple[dict[str, Any], mink.FrameTask]] = []
        for target in self.config["source_adapter"]["targets"]:
            task = mink.FrameTask(
                frame_name=target["robot_body"],
                frame_type="body",
                position_cost=float(self.algorithm["frame_position_cost"]),
                orientation_cost=float(self.algorithm["frame_orientation_cost"]),
                lm_damping=float(self.algorithm["frame_lm_damping"]),
            )
            self.frame_tasks.append((target, task))
        self.posture = mink.PostureTask(
            self.model,
            cost=float(self.algorithm["posture_cost"]),
            lm_damping=float(self.algorithm["posture_lm_damping"]),
        )
        self.tasks = [task for _, task in self.frame_tasks] + [self.posture]
        self.limits = [mink.ConfigurationLimit(self.model)]
        self.initialization_time_s = time.perf_counter() - start

    def _initialize_from_source(
        self, human: CanonicalHuman, indices: dict[str, int]
    ) -> None:
        qpos = np.zeros(self.model.nq, dtype=np.float64)
        qpos[:3] = human.world_positions[0, indices["Hips"]]
        # Canonical and MuJoCo storage are both wxyz.  Do not reproduce the
        # apparent upstream Poselib-xyzw initialization assignment.
        qpos[3:7] = human.world_rotations[0, indices["Hips"]]
        mujoco.mj_normalizeQuat(self.model, qpos)
        self.configuration.update(qpos)
        self.posture.set_target_from_configuration(self.configuration)

    def _target_position(
        self,
        human: CanonicalHuman,
        frame: int,
        joint: int,
        root_joint: int,
    ) -> np.ndarray:
        """Apply the experimental perturbation before official v2 XYZ scaling."""
        official_axis_scale = np.asarray(
            self.algorithm["position_scale_xyz"], dtype=np.float64
        )
        if self.root_scale_multiplier == 1.0 and self.local_scale_multiplier == 1.0:
            # Preserve the native path bit-for-bit instead of introducing a
            # subtract/add round-off merely to express identity multipliers.
            return human.world_positions[frame, joint] * official_axis_scale
        source_root = human.world_positions[frame, root_joint]
        source_anchor = human.world_positions[0, root_joint]
        perturbed_root = source_anchor + (
            source_root - source_anchor
        ) * self.root_scale_multiplier
        root_relative = human.world_positions[frame, joint] - source_root
        perturbed_target = (
            perturbed_root + root_relative * self.local_scale_multiplier
        )
        return perturbed_target * official_axis_scale

    def _set_targets(
        self,
        human: CanonicalHuman,
        frame: int,
        indices: dict[str, int],
        *,
        capture_runtime_witness: bool,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        root_joint = indices["Hips"]
        observed_positions: list[np.ndarray] = []
        observed_orientations: list[np.ndarray] = []
        for target, task in self.frame_tasks:
            joint = indices[target["human_joint"]]
            position = self._target_position(human, frame, joint, root_joint)
            rotation_matrix = quaternion_wxyz_to_matrix(human.world_rotations[frame, joint])
            rotation = mink.SO3.from_matrix(rotation_matrix)
            task.set_target(mink.SE3.from_rotation_and_translation(rotation, position))
            if capture_runtime_witness:
                observed_positions.append(
                    np.asarray(position, dtype=np.float64).copy()
                )
                observed_orientations.append(
                    np.asarray(rotation_matrix, dtype=np.float64).copy()
                )
        if not capture_runtime_witness:
            return None
        return (
            np.asarray(observed_positions, dtype=np.float64),
            np.asarray(observed_orientations, dtype=np.float64),
        )

    def _ground_align(
        self, raw_qpos: np.ndarray, human: CanonicalHuman
    ) -> tuple[np.ndarray, np.ndarray]:
        output = np.asarray(raw_qpos, dtype=np.float64).copy()
        offsets = np.empty(len(output), dtype=np.float64)
        data = mujoco.MjData(self.model)
        source_lowest = np.min(human.world_positions[: len(output), :, 2], axis=1)
        for frame, value in enumerate(raw_qpos):
            data.qpos[:] = value
            mujoco.mj_forward(self.model, data)
            robot_lowest = float(np.min(data.xpos[1:, 2]))
            offsets[frame] = source_lowest[frame] - robot_lowest
            output[frame, 2] += offsets[frame]
        return output, offsets

    def run(
        self,
        human: CanonicalHuman,
        max_frames: int | None = None,
        *,
        capture_runtime_witness: bool = True,
    ) -> CanonicalG1:
        method_pipeline_started = time.perf_counter()
        human.validate()
        indices = {name: index for index, name in enumerate(human.joint_names.astype(str))}
        required = {target["human_joint"] for target, _ in self.frame_tasks}
        missing = sorted(required - set(indices))
        if missing:
            raise ValueError(f"Canonical source lacks ProtoMotions targets: {missing}")
        source_frames = len(human.timestamps)
        frames = source_frames if max_frames is None else min(int(max_frames), source_frames)
        if frames < 1:
            raise ValueError("max_frames must select at least one frame")
        self._initialize_from_source(human, indices)

        fps = float(human.fps)
        dt = 1.0 / (fps * int(self.algorithm["optimization_steps_per_frame"]))
        warmup_start = int(
            np.ceil(
                -float(self.algorithm["warmup_reference_frames"])
                * fps
                / float(self.algorithm["warmup_reference_fps"])
            )
        )
        raw_qpos: list[np.ndarray] = []
        solve_times: list[float] = []
        end_to_end_times: list[float] = []
        runtime_target_frames: list[np.ndarray] = []
        runtime_warmup_targets: list[np.ndarray] = []
        runtime_orientation_frames: list[np.ndarray] = []
        runtime_warmup_orientations: list[np.ndarray] = []
        native_core_started = time.perf_counter()
        for t in range(warmup_start, frames):
            source_frame = max(0, t)
            frame_start = time.perf_counter()
            observed = self._set_targets(
                human,
                source_frame,
                indices,
                capture_runtime_witness=capture_runtime_witness,
            )
            if observed is not None:
                observed_targets, observed_orientations = observed
                if t < 0:
                    runtime_warmup_targets.append(observed_targets)
                    runtime_warmup_orientations.append(observed_orientations)
                else:
                    runtime_target_frames.append(observed_targets)
                    runtime_orientation_frames.append(observed_orientations)
            solve_start = time.perf_counter()
            for _ in range(int(self.algorithm["optimization_steps_per_frame"])):
                velocity = mink.solve_ik(
                    self.configuration,
                    self.tasks,
                    dt,
                    str(self.algorithm["solver"]),
                    float(self.algorithm["solve_damping"]),
                    limits=self.limits,
                )
                self.configuration.integrate_inplace(velocity, dt)
            solve_elapsed = time.perf_counter() - solve_start
            if t >= 0:
                value = self.configuration.data.qpos.copy()
                if not np.isfinite(value).all():
                    break
                raw_qpos.append(value)
                solve_times.append(solve_elapsed)
                end_to_end_times.append(time.perf_counter() - frame_start)
        if not raw_qpos:
            raise RuntimeError("ProtoMotions v2.3 Mink produced no finite frames")
        aligned_qpos, ground_offsets = self._ground_align(np.asarray(raw_qpos), human)
        native_core_total_s = time.perf_counter() - native_core_started
        produced = len(aligned_qpos)
        canonical_qpos = np.asarray(aligned_qpos, dtype=np.float64)
        solve_times_array = np.asarray(solve_times, dtype=np.float64)
        # The scientific boundary stops as soon as the canonical qpos and its
        # timing vector exist in memory.  Witness hashing and metadata assembly
        # below are deliberately outside it.
        steady_method_pipeline_s = time.perf_counter() - method_pipeline_started
        witness_processing_started = time.perf_counter()
        from .native_target_capture import tensor_sha256

        runtime_labels = np.asarray(
            [
                f"{target['robot_body']}<-{target['human_joint']}"
                for target, _ in self.frame_tasks
            ]
        )
        runtime_capture: dict[str, Any] | None = None
        if capture_runtime_witness:
            runtime_targets = np.asarray(runtime_target_frames, dtype=np.float64)
            runtime_warmup = np.asarray(runtime_warmup_targets, dtype=np.float64)
            runtime_orientations = np.asarray(
                runtime_orientation_frames, dtype=np.float64
            )
            runtime_warmup_orientations = np.asarray(
                runtime_warmup_orientations, dtype=np.float64
            )
            if runtime_targets.shape != (produced, len(self.frame_tasks), 3):
                raise RuntimeError(
                    "ProtoMotions v2.3 runtime target capture does not match output frames"
                )
            if runtime_orientations.shape != (
                produced,
                len(self.frame_tasks),
                3,
                3,
            ):
                raise RuntimeError(
                    "ProtoMotions v2.3 runtime orientation capture is incomplete"
                )
            runtime_capture = {
                "boundary": (
                    "values passed to every formal Mink FrameTask.set_target "
                    "immediately before solve_ik"
                ),
                "position_tensor_sha256": tensor_sha256(runtime_targets),
                "position_shape": list(runtime_targets.shape),
                "orientation_matrix_tensor_sha256": tensor_sha256(
                    runtime_orientations
                ),
                "orientation_matrix_shape": list(runtime_orientations.shape),
                "labels": runtime_labels.tolist(),
                "labels_sha256": tensor_sha256(runtime_labels),
                "warmup_position_tensor_sha256": tensor_sha256(runtime_warmup),
                "warmup_position_shape": list(runtime_warmup.shape),
                "warmup_orientation_matrix_tensor_sha256": tensor_sha256(
                    runtime_warmup_orientations
                ),
                "warmup_orientation_matrix_shape": list(
                    runtime_warmup_orientations.shape
                ),
                "observed_during_solver_run": True,
                "capture_execution_role": "independent_untimed_or_cold_witness",
            }
        witness_processing_time_s = time.perf_counter() - witness_processing_started
        completion = (
            "succeeded" if produced / source_frames >= MIN_COMPLETION_RATIO else "incomplete"
        )
        metadata = {
            "method": self.config["method"],
            "method_family": self.config["method_family"],
            "experiment_identity": (
                "benchmark LAFAN port of the official ProtoMotions v2.3 "
                "Mink task/solver"
            ),
            "is_phc_result": False,
            "phc_relationship": (
                "lineage/preprocessing context only; the reported trajectory is "
                "produced by the upstream v2.3 Mink task/solver"
            ),
            "upstream_repository": self.config["upstream"]["repository"],
            "upstream_tag": self.config["upstream"]["tag"],
            "upstream_commit": self.config["upstream"]["commit"],
            "upstream_implementation_sha256": self.installation["hashes"]["implementation"],
            "completion_status": completion,
            "canonical_source_sha256": human.source_sha256,
            "config_path": str(self.config_path.relative_to(self.repo_root)),
            "config_sha256": sha256_file(self.config_path),
            "native_robot_xml_sha256": self.installation["hashes"]["native_robot"],
            "canonical_robot_xml_sha256": self.installation["hashes"]["canonical_robot"],
            "joint_order": list(G1_JOINT_NAMES),
            "input_units": "meters",
            "input_up_axis": "z",
            "quaternion_order": "wxyz",
            "position_scale_xyz": list(map(float, self.algorithm["position_scale_xyz"])),
            "root_scale_multiplier": self.root_scale_multiplier,
            "local_scale_multiplier": self.local_scale_multiplier,
            "scale_intervention": "pre_solver_root_path_and_root_relative_target_multipliers",
            "warmup_frames": int(-warmup_start),
            "optimization_steps_per_frame": int(self.algorithm["optimization_steps_per_frame"]),
            "solver": self.algorithm["solver"],
            "solve_damping": float(self.algorithm["solve_damping"]),
            "ground_alignment": self.algorithm["ground_alignment"],
            "ground_offset_min_m": float(np.min(ground_offsets)),
            "ground_offset_max_m": float(np.max(ground_offsets)),
            "initialization_time_s": self.initialization_time_s,
            "steady_end_to_end_total_s": float(steady_method_pipeline_s),
            "steady_method_pipeline_s": float(steady_method_pipeline_s),
            "native_core_total_s": float(native_core_total_s),
            "native_core_boundary": (
                "canonical LAFAN semantic targets ready to grounded native G1 qpos in memory"
            ),
            "runtime_pre_solver_capture": runtime_capture,
            "runtime_witness_enabled": bool(capture_runtime_witness),
            "runtime_witness_processing_time_s_excluded": float(
                witness_processing_time_s
            ),
            "adapter_deviations": list(self.config["source_adapter"]["deviations"]),
            "robot_compatibility_audit": self.robot_audit,
            "frozen_environment": self.installation,
            "upstream_smplx_pose_contract": self.installation[
                "smplx_pose_contract"
            ],
        }
        result = CanonicalG1(
            qpos=canonical_qpos,
            fps=fps,
            source_frame_idx=np.arange(produced, dtype=np.int64),
            valid=np.ones(produced, dtype=bool),
            per_frame_solve_time_s=solve_times_array,
            metadata=metadata,
        )
        result.validate(source_frames)
        return result


def run_scale_variant(
    repo_root: str | Path,
    canonical_source: str | Path,
    output_dir: str | Path,
    root_scale_multiplier: float,
    local_scale_multiplier: float,
) -> CanonicalG1:
    """Run one scale-sensitivity arm through the actual v2.3 Mink solver.

    Root displacement from the frozen frame-zero source anchor and every
    root-relative semantic target are perturbed independently.  The operation
    occurs before v2.3's published absolute XYZ scale ``[0.75, 1.0, 0.8]``;
    rotations, costs, solver, warm-up, limits, and ground alignment are fixed.
    """
    root = Path(repo_root).resolve()
    source_path = Path(canonical_source)
    if not source_path.is_absolute():
        source_path = root / source_path
    work = Path(output_dir)
    if not work.is_absolute():
        work = root / work
    work.mkdir(parents=True, exist_ok=True)
    human = CanonicalHuman.load(source_path)
    retargeter = ProtoMotionsV2Retargeter(
        root,
        root_scale_multiplier=root_scale_multiplier,
        local_scale_multiplier=local_scale_multiplier,
    )
    atomic_write_json(work / "robot_compatibility_audit.json", retargeter.robot_audit)
    result = retargeter.run(human)
    result.metadata.update(
        {
            "experiment_role": "native_pipeline_scale_response",
            "scale_variant_work_dir": str(work),
            "root_scale_multiplier": float(root_scale_multiplier),
            "local_scale_multiplier": float(local_scale_multiplier),
            "scale_intervention": "pre_solver_root_path_and_root_relative_target_multipliers",
        }
    )
    result.validate(source_frame_count=len(human.timestamps))
    return result
