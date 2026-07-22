"""Adapters for frozen upstream GMR and OmniRetarget/Holosoma implementations."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import sha256_file
from .schemas import CanonicalG1, CanonicalHuman


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


def _completion(frames: int, source_frames: int) -> str:
    return "succeeded" if frames / source_frames >= 0.95 else "incomplete"


def _bvh_header_contract(path: Path) -> tuple[int, float, float]:
    """Read the declared BVH frame count/time without relying on an adapter."""

    frame_count: int | None = None
    frame_time_s: float | None = None
    with path.open("r", encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if line.startswith("Frames:"):
                frame_count = int(line.split(":", 1)[1].strip())
            elif line.startswith("Frame Time:"):
                frame_time_s = float(line.split(":", 1)[1].strip())
                break
    if frame_count is None or frame_time_s is None or frame_time_s <= 0.0:
        raise ValueError(f"Invalid or incomplete BVH motion header: {path}")
    return frame_count, frame_time_s, 1.0 / frame_time_s


class PreparedGMRRetargeter:
    """Reusable GMR model/tasks with per-repetition BVH parsing and reset."""

    def __init__(
        self,
        repo_root: str | Path,
        source_bvh: str | Path,
        *,
        root_scale_multiplier: float = 1.0,
        local_scale_multiplier: float = 1.0,
    ):
        initialization_start = time.perf_counter()
        self.root = Path(repo_root).resolve()
        self.source_bvh = Path(source_bvh).resolve()
        self.gmr_root = self.root / "external" / "GMR"
        if str(self.gmr_root) not in sys.path:
            sys.path.insert(0, str(self.gmr_root))
        from general_motion_retargeting.motion_retarget import (
            GeneralMotionRetargeting,
        )
        from general_motion_retargeting.utils.lafan1 import load_bvh_file

        self.load_bvh_file = load_bvh_file
        (
            self.native_header_frames,
            self.native_header_frame_time_s,
            self.native_header_fps,
        ) = _bvh_header_contract(self.source_bvh)
        # Height is a constructor parameter in upstream GMR.  This one-time
        # parse is initialization evidence only; every formal repetition parses
        # the declared BVH again inside the end-to-end boundary.
        _, human_height = load_bvh_file(str(self.source_bvh), format="lafan1")
        self.human_height = float(human_height)
        self.retargeter = GeneralMotionRetargeting(
            src_human="bvh_lafan1",
            tgt_robot="unitree_g1",
            actual_human_height=self.human_height,
            solver="daqp",
            damping=0.5,
            verbose=False,
        )
        if root_scale_multiplier <= 0.0 or local_scale_multiplier <= 0.0:
            raise ValueError("Scale multipliers must be positive")
        self.root_scale_multiplier = float(root_scale_multiplier)
        self.local_scale_multiplier = float(local_scale_multiplier)
        # This is a benchmark-only, pre-solver sensitivity intervention.  It
        # preserves GMR's native region ratios and changes only the root-path
        # or root-relative target scale, respectively.
        for name in self.retargeter.human_scale_table:
            if name != self.retargeter.human_root_name:
                self.retargeter.human_scale_table[name] *= self.local_scale_multiplier
        self.root_name = self.retargeter.human_root_name
        self.config_path = (
            self.gmr_root
            / "general_motion_retargeting"
            / "ik_configs"
            / "bvh_lafan1_to_g1.json"
        )
        config = json.loads(self.config_path.read_text())
        self.target_entries: list[tuple[str, str, str]] = []
        for table_name, table in (
            ("table1", config["ik_match_table1"]),
            ("table2", config["ik_match_table2"]),
        ):
            for robot_body, entry in table.items():
                human_body, position_cost, orientation_cost, _, _ = entry
                if float(position_cost) != 0.0 or float(orientation_cost) != 0.0:
                    self.target_entries.append(
                        (table_name, str(robot_body), str(human_body))
                    )
        self.xml_path = (
            self.gmr_root / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"
        )
        self.initialization_time_s = time.perf_counter() - initialization_start

    def _reset(self) -> None:
        self.retargeter.configuration.update(
            self.retargeter.model.qpos0.copy()
        )
        self.retargeter.ground_offset = 0.0

    def run(
        self,
        human: CanonicalHuman,
        max_frames: int | None = None,
        *,
        capture_runtime_witness: bool = True,
    ) -> CanonicalG1:
        method_pipeline_started = time.perf_counter()
        frames, parsed_height = self.load_bvh_file(
            str(self.source_bvh), format="lafan1"
        )
        if len(frames) != self.native_header_frames:
            raise RuntimeError(
                "GMR loader frame count differs from the cropped BVH header"
            )
        if not np.isclose(parsed_height, self.human_height, atol=0.0, rtol=0.0):
            raise RuntimeError("GMR BVH height changed after method initialization")
        count = len(frames) if max_frames is None else min(max_frames, len(frames))
        if count < 1:
            raise ValueError("max_frames must select at least one GMR frame")
        self._reset()
        first_root = np.asarray(frames[0][self.root_name][0], dtype=np.float64)
        qpos: list[np.ndarray] = []
        solve_times: list[float] = []
        runtime_position_targets: list[np.ndarray] = []
        runtime_orientation_targets: list[np.ndarray] = []
        for frame in frames[:count]:
            if self.root_scale_multiplier != 1.0:
                current_root = np.asarray(frame[self.root_name][0], dtype=np.float64)
                scaled_root = first_root + (
                    current_root - first_root
                ) * self.root_scale_multiplier
                translation = scaled_root - current_root
                frame = {
                    name: [np.asarray(value[0], dtype=np.float64) + translation, value[1]]
                    for name, value in frame.items()
                }
            start = time.perf_counter()
            value = self.retargeter.retarget(frame, offset_to_ground=False)
            solve_times.append(time.perf_counter() - start)
            if not np.isfinite(value).all():
                break
            qpos.append(np.asarray(value, dtype=np.float64))
            if capture_runtime_witness:
                runtime_position_targets.append(
                    np.stack(
                        [
                            np.asarray(
                                self.retargeter.scaled_human_data[human_body][0]
                            )
                            for _, _, human_body in self.target_entries
                        ]
                    )
                )
                runtime_orientation_targets.append(
                    np.stack(
                        [
                            np.asarray(
                                self.retargeter.scaled_human_data[human_body][1]
                            )
                            for _, _, human_body in self.target_entries
                        ]
                    )
                )
        if not qpos:
            raise RuntimeError("GMR produced no finite frames")
        canonical_qpos = np.asarray(qpos, dtype=np.float64)
        solve_times_array = np.asarray(solve_times[: len(qpos)], dtype=np.float64)
        steady_method_pipeline_s = time.perf_counter() - method_pipeline_started
        witness_processing_started = time.perf_counter()
        from .native_target_capture import tensor_sha256

        runtime_labels = [
            f"{table}:{robot}<-{human}"
            for table, robot, human in self.target_entries
        ]
        runtime_capture: dict[str, Any] | None = None
        if capture_runtime_witness:
            runtime_targets = np.asarray(runtime_position_targets, dtype=np.float64)
            runtime_orientations = np.asarray(
                runtime_orientation_targets, dtype=np.float64
            )
            runtime_capture = {
                "boundary": (
                    "GeneralMotionRetargeting.retarget after update_targets; "
                    "scaled_human_data translations and wxyz orientations consumed "
                    "by both configured FrameTask stages"
                ),
                "position_tensor_sha256": tensor_sha256(runtime_targets),
                "position_shape": list(runtime_targets.shape),
                "orientation_wxyz_tensor_sha256": tensor_sha256(
                    runtime_orientations
                ),
                "orientation_wxyz_shape": list(runtime_orientations.shape),
                "labels": runtime_labels,
                "labels_sha256": tensor_sha256(np.asarray(runtime_labels)),
                "observed_during_solver_run": True,
                "capture_execution_role": "independent_untimed_or_cold_witness",
            }
        witness_processing_time_s = time.perf_counter() - witness_processing_started
        return CanonicalG1(
            qpos=canonical_qpos,
            fps=human.fps,
            source_frame_idx=np.arange(len(qpos), dtype=np.int64),
            valid=np.ones(len(qpos), dtype=bool),
            per_frame_solve_time_s=solve_times_array,
            metadata={
                "method": "gmr",
                "method_family": "upstream",
                "upstream_commit": "bb1bbe40774794fceb2a7c579a3464a28e68c844",
                "completion_status": _completion(len(qpos), len(human.timestamps)),
                "canonical_source_sha256": human.source_sha256,
                "native_source_sha256": sha256_file(self.source_bvh),
                "native_source_parsed_frames": len(frames),
                "native_source_header_frames": self.native_header_frames,
                "native_source_header_frame_time_s": self.native_header_frame_time_s,
                "native_source_parsed_fps": self.native_header_fps,
                "native_source_timeline_mapping": (
                    "exact cropped-BVH header order; one parsed BVH frame per "
                    "canonical frame; frame-index aligned"
                ),
                "actual_human_height_m": self.human_height,
                "config_sha256": sha256_file(self.config_path),
                "robot_xml_sha256": sha256_file(self.xml_path),
                "adapter_changes": "headless I/O, provenance, and per-frame timing only",
                "quaternion_order": "wxyz",
                "initialization_time_s": self.initialization_time_s,
                "steady_end_to_end_total_s": float(steady_method_pipeline_s),
                "steady_method_pipeline_s": float(steady_method_pipeline_s),
                "native_core_total_s": float(sum(solve_times[: len(qpos)])),
                "native_core_boundary": (
                    "official parsed LAFAN frame dictionary ready to native G1 qpos in memory"
                ),
                "runtime_pre_solver_capture": runtime_capture,
                "runtime_witness_enabled": bool(capture_runtime_witness),
                "runtime_witness_processing_time_s_excluded": float(
                    witness_processing_time_s
                ),
                "root_scale_multiplier": self.root_scale_multiplier,
                "local_scale_multiplier": self.local_scale_multiplier,
                "scale_intervention": (
                    "pre-solver frame-zero-root path transform plus root-relative "
                    "human_scale_table multiplier"
                ),
            },
        )


def run_gmr(
    repo_root: str | Path,
    source_bvh: str | Path,
    canonical_source: str | Path,
    max_frames: int | None = None,
    root_scale_multiplier: float = 1.0,
    local_scale_multiplier: float = 1.0,
    *,
    capture_runtime_witness: bool = True,
) -> CanonicalG1:
    """Run the clean official GMR LAFAN→G1 path without visualization."""

    prepared = PreparedGMRRetargeter(
        repo_root,
        source_bvh,
        root_scale_multiplier=root_scale_multiplier,
        local_scale_multiplier=local_scale_multiplier,
    )
    human = CanonicalHuman.load(canonical_source)
    return prepared.run(
        human,
        max_frames=max_frames,
        capture_runtime_witness=capture_runtime_witness,
    )


def holosoma_lafan_array(human: CanonicalHuman) -> np.ndarray:
    """Construct Holosoma's native Y-up array without filesystem I/O."""

    names = {name: index for index, name in enumerate(human.joint_names.astype(str))}
    aliases = {"LeftToeBase": "LeftToe", "RightToeBase": "RightToe"}
    indices = []
    for name in HOLOSOMA_LAFAN_ORDER:
        source_name = aliases.get(name, name)
        if source_name not in names:
            raise ValueError(f"Holosoma adapter source joint missing: {source_name}")
        indices.append(names[source_name])
    canonical = human.world_positions[:, indices]
    y_up = canonical[..., [0, 2, 1]]
    restored = y_up[..., [0, 2, 1]]
    if not np.array_equal(restored, canonical):
        raise RuntimeError("Holosoma coordinate adapter failed its exact round-trip")
    return y_up


def prepare_holosoma_lafan_input(
    human: CanonicalHuman, output_dir: str | Path, task_name: str
) -> Path:
    """Write a precomputed Holosoma Y-up adapter array.

    Holosoma's loader applies [x,y,z]→[x,z,y].  Applying that involution here
    preserves the benchmark's canonical Z-up coordinates exactly.  Toe aliases
    and the upstream right-first joint order are made explicit rather than
    relying on the left-first order of the Ubisoft BVH files.
    """
    y_up = holosoma_lafan_array(human)
    output = Path(output_dir) / f"{task_name}.npy"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, y_up)
    return output


def run_holosoma(
    repo_root: str | Path,
    canonical_source: str | Path | CanonicalHuman,
    work_dir: str | Path,
    max_frames: int | None = None,
    root_scale_multiplier: float = 1.0,
    local_scale_multiplier: float = 1.0,
    fixed_contact_labels: bool = False,
    capture_runtime_witness: bool = True,
) -> CanonicalG1:
    """Run clean Holosoma LAFAN→G1 with a canonical file adapter."""
    root = Path(repo_root).resolve()
    canonical_source_path = (
        None
        if isinstance(canonical_source, CanonicalHuman)
        else Path(canonical_source).resolve()
    )
    checkout = root / "external" / "holosoma"
    package_root = checkout / "src" / "holosoma_retargeting"
    clean_module_root = str(package_root)
    sys.path.insert(0, clean_module_root)
    original_cwd = Path.cwd()
    try:
        # This worker is always launched in a fresh method-specific process.
        from holosoma_retargeting.config_types.retargeter import RetargeterConfig
        from holosoma_retargeting.config_types.retargeting import RetargetingConfig
        from holosoma_retargeting.config_types.robot import RobotConfig
        from holosoma_retargeting.config_types.task import TaskConfig
        from holosoma_retargeting.examples import robot_retarget
        from holosoma_retargeting.src.interaction_mesh_retargeter import InteractionMeshRetargeter
        from .native_target_capture import tensor_sha256

        human = (
            canonical_source
            if isinstance(canonical_source, CanonicalHuman)
            else CanonicalHuman.load(canonical_source_path)
        )
        source_count = len(human.timestamps)
        if max_frames is not None:
            end = min(max_frames, len(human.timestamps))
            human = CanonicalHuman(
                joint_names=human.joint_names,
                parent_indices=human.parent_indices,
                local_rotations=human.local_rotations[:end],
                world_rotations=human.world_rotations[:end],
                world_positions=human.world_positions[:end],
                root_translation=human.root_translation[:end],
                fps=human.fps,
                timestamps=np.arange(end) / human.fps,
                foot_contact_labels=human.foot_contact_labels[:end],
                source_sha256=human.source_sha256,
            )
        if root_scale_multiplier <= 0.0 or local_scale_multiplier <= 0.0:
            raise ValueError("Scale multipliers must be positive")
        run_root = Path(work_dir).resolve()
        input_dir = run_root / "native_input"
        save_dir = run_root / "native_output"
        task_name = "pilot_canonical"
        adapter_started = time.perf_counter()
        native_y_up = holosoma_lafan_array(human)
        adapter_in_memory_time_s = time.perf_counter() - adapter_started
        native_input_path = input_dir / f"{task_name}.npy"
        native_input_path.parent.mkdir(parents=True, exist_ok=True)
        native_input_write_started = time.perf_counter()
        np.save(native_input_path, native_y_up)
        native_input_write_time_s = time.perf_counter() - native_input_write_started
        save_dir.mkdir(parents=True, exist_ok=True)

        frame_times: list[float] = []
        steady_times: list[float] = []
        native_core_times: list[float] = []
        native_save_times: list[float] = []
        retarget_starts: list[float] = []
        runtime_laplacian_targets: list[np.ndarray] = []
        runtime_adjacency_hashes: list[str] = []
        runtime_mapped_positions: np.ndarray | None = None
        runtime_mapped_labels: np.ndarray | None = None
        runtime_vertex_labels: np.ndarray | None = None
        runtime_contacts: np.ndarray | None = None
        runtime_object_points_local: np.ndarray | None = None
        runtime_object_points_demo: np.ndarray | None = None
        runtime_object_poses: np.ndarray | None = None
        runtime_object_poses_augmented: np.ndarray | None = None
        runtime_q_init: np.ndarray | None = None
        runtime_contact_labels: list[str] | None = None
        runtime_constraint_contract: dict[str, Any] | None = None
        captured_native_payload: dict[str, np.ndarray] | None = None
        witness_collection_time_s = 0.0
        retargeter_initialization_times: list[float] = []
        original_iterate = InteractionMeshRetargeter.iterate
        original_retargeter_init = InteractionMeshRetargeter.__init__
        original_retarget_motion = InteractionMeshRetargeter.retarget_motion
        original_contact_extractor = robot_retarget.extract_foot_sticking_sequence_velocity
        original_np_savez = np.savez

        def timed_retargeter_init(self, *args, **kwargs):
            start = time.perf_counter()
            result = original_retargeter_init(self, *args, **kwargs)
            retargeter_initialization_times.append(time.perf_counter() - start)
            return result

        def scale_native_targets(
            positions: np.ndarray, demo_joints: list[str] | tuple[str, ...]
        ) -> np.ndarray:
            """Perturb the official post-preprocess, pre-solver target package."""

            value = np.asarray(positions, dtype=np.float64)
            if root_scale_multiplier == 1.0 and local_scale_multiplier == 1.0:
                return value
            if "Hips" not in demo_joints:
                raise RuntimeError("Holosoma LAFAN target package lacks Hips")
            root_index = list(demo_joints).index("Hips")
            roots = value[:, root_index]
            anchor = roots[0]
            scaled_roots = anchor + (roots - anchor) * root_scale_multiplier
            return scaled_roots[:, None, :] + (
                value - roots[:, None, :]
            ) * local_scale_multiplier

        def timed_iterate(self, *args, **kwargs):
            nonlocal witness_collection_time_s
            if "target_laplacian" not in kwargs or "adj_list" not in kwargs:
                raise RuntimeError(
                    "Holosoma runtime capture requires keyword target_laplacian/adj_list"
                )
            if capture_runtime_witness:
                witness_started = time.perf_counter()
                runtime_laplacian_targets.append(
                    np.asarray(kwargs["target_laplacian"], dtype=np.float64).copy()
                )
                adjacency = [
                    sorted(int(item) for item in neighbors)
                    for neighbors in kwargs["adj_list"]
                ]
                encoded = json.dumps(adjacency, separators=(",", ":")).encode()
                runtime_adjacency_hashes.append(
                    hashlib.sha256(encoded).hexdigest()
                )
                witness_collection_time_s += time.perf_counter() - witness_started
            start = time.perf_counter()
            result = original_iterate(self, *args, **kwargs)
            frame_times.append(time.perf_counter() - start)
            return result

        def timed_retarget_motion(self, *args, **kwargs):
            nonlocal runtime_mapped_positions, runtime_mapped_labels
            nonlocal runtime_vertex_labels, runtime_contacts
            nonlocal runtime_object_points_local, runtime_object_points_demo
            nonlocal runtime_object_poses, runtime_object_poses_augmented
            nonlocal runtime_q_init, witness_collection_time_s
            nonlocal runtime_contact_labels, runtime_constraint_contract
            start = time.perf_counter()
            retarget_starts.append(start)
            scaled_kwargs = dict(kwargs)
            if "human_joint_motions" not in scaled_kwargs:
                raise RuntimeError("Holosoma timing hook requires keyword target input")
            scaled_kwargs["human_joint_motions"] = scale_native_targets(
                scaled_kwargs["human_joint_motions"], self.demo_joints
            )
            if capture_runtime_witness:
                witness_started = time.perf_counter()
                mapped_indices = np.asarray(
                    self.smplh_mapped_joint_indices, dtype=np.int64
                )
                runtime_mapped_positions = np.asarray(
                    scaled_kwargs["human_joint_motions"][:, mapped_indices],
                    dtype=np.float64,
                ).copy()
                runtime_mapped_labels = np.asarray(
                    [self.demo_joints[index] for index in mapped_indices]
                )
                runtime_object_points_local = np.asarray(
                    scaled_kwargs["object_points_local"], dtype=np.float64
                ).copy()
                runtime_object_points_demo = np.asarray(
                    scaled_kwargs["object_points_local_demo"], dtype=np.float64
                ).copy()
                runtime_object_poses = np.asarray(
                    scaled_kwargs["object_poses"], dtype=np.float64
                ).copy()
                runtime_object_poses_augmented = np.asarray(
                    scaled_kwargs["object_poses_augmented"], dtype=np.float64
                ).copy()
                runtime_q_init = np.asarray(
                    scaled_kwargs["q_a_init"], dtype=np.float64
                ).copy()
                frames = scaled_kwargs["foot_sticking_sequences"]
                if not frames:
                    raise RuntimeError("Holosoma contact sequence is empty")
                foot_order = tuple(str(name) for name in frames[0])
                if any(tuple(str(name) for name in frame) != foot_order for frame in frames):
                    raise RuntimeError("Holosoma contact label order changed by frame")
                runtime_contact_labels = list(foot_order)
                runtime_contacts = np.asarray(
                    [
                        [bool(frame[name]) for name in foot_order]
                        for frame in frames
                    ],
                    dtype=bool,
                )
                runtime_constraint_contract = {
                    "activate_foot_sticking": bool(self.activate_foot_sticking),
                    "activate_obj_non_penetration": bool(
                        self.activate_obj_non_penetration
                    ),
                    "activate_joint_limits": bool(self.activate_joint_limits),
                    "foot_sticking_tolerance_m": float(
                        self.foot_sticking_tolerance
                    ),
                    "penetration_tolerance_m": float(self.penetration_tolerance),
                    "foot_constraint_links": sorted(self.foot_links),
                    "q_a_indices_sha256": tensor_sha256(
                        np.asarray(self.q_a_indices, dtype=np.int64)
                    ),
                    "q_a_lower_limits_sha256": tensor_sha256(
                        np.asarray(self.q_a_lb, dtype=np.float64)
                    ),
                    "q_a_upper_limits_sha256": tensor_sha256(
                        np.asarray(self.q_a_ub, dtype=np.float64)
                    ),
                }
                object_count = len(runtime_object_points_local)
                runtime_vertex_labels = np.asarray(
                    runtime_mapped_labels.tolist()
                    + [f"ground_{index:03d}" for index in range(object_count)]
                )
                witness_collection_time_s += time.perf_counter() - witness_started
            result = original_retarget_motion(self, *args, **scaled_kwargs)
            if len(native_core_times) < len(retarget_starts):
                native_core_times.append(time.perf_counter() - start)
            steady_times.append(time.perf_counter() - start)
            return result

        def timed_np_savez(*args, **kwargs):
            nonlocal captured_native_payload
            if retarget_starts and len(native_core_times) < len(retarget_starts):
                native_core_times.append(time.perf_counter() - retarget_starts[-1])
            if captured_native_payload is not None:
                raise RuntimeError("Holosoma attempted more than one native save")
            captured_native_payload = {
                str(key): np.asarray(value).copy() for key, value in kwargs.items()
            }
            # The official function exposes qpos only through np.savez.  Capture
            # that in-memory payload and elide disk I/O; the harness persists a
            # canonical timing witness after its end-to-end boundary instead.
            native_save_times.append(0.0)
            return None

        InteractionMeshRetargeter.iterate = timed_iterate
        InteractionMeshRetargeter.retarget_motion = timed_retarget_motion
        InteractionMeshRetargeter.__init__ = timed_retargeter_init
        np.savez = timed_np_savez
        if fixed_contact_labels:
            contacts = np.asarray(human.foot_contact_labels, dtype=bool).copy()

            def fixed_contact_extractor(smpl_joints, demo_joints, foot_names, *args, **kwargs):
                if len(smpl_joints) != len(contacts):
                    raise RuntimeError("Fixed contact labels do not match Holosoma frames")
                return [
                    {
                        foot_names[0]: bool(contacts[index, 0]),
                        foot_names[1]: bool(contacts[index, 1]),
                    }
                    for index in range(len(contacts))
                ]

            robot_retarget.extract_foot_sticking_sequence_velocity = fixed_contact_extractor
        elif root_scale_multiplier != 1.0 or local_scale_multiplier != 1.0:

            def scale_aware_contact_extractor(
                smpl_joints, demo_joints, foot_names, *args, **kwargs
            ):
                scaled = scale_native_targets(smpl_joints, demo_joints)
                return original_contact_extractor(
                    scaled, demo_joints, foot_names, *args, **kwargs
                )

            robot_retarget.extract_foot_sticking_sequence_velocity = (
                scale_aware_contact_extractor
            )
        cfg = RetargetingConfig(
            task_type="robot_only",
            robot="g1",
            data_format="lafan",
            task_name=task_name,
            data_path=input_dir,
            save_dir=save_dir,
            robot_config=RobotConfig(
                robot_type="g1",
                robot_urdf_file="models/g1/g1_29dof.urdf",
            ),
            task_config=TaskConfig(ground_range=(-10.0, 10.0)),
            retargeter=RetargeterConfig(foot_sticking_tolerance=0.02),
        )
        os.chdir(package_root / "holosoma_retargeting")
        try:
            main_start = time.perf_counter()
            robot_retarget.main(cfg)
        finally:
            InteractionMeshRetargeter.iterate = original_iterate
            InteractionMeshRetargeter.retarget_motion = original_retarget_motion
            InteractionMeshRetargeter.__init__ = original_retargeter_init
            robot_retarget.extract_foot_sticking_sequence_velocity = original_contact_extractor
            np.savez = original_np_savez
        main_wall = time.perf_counter() - main_start
        canonical_conversion_started = time.perf_counter()
        if captured_native_payload is None or "qpos" not in captured_native_payload:
            raise RuntimeError("Holosoma did not expose its native qpos payload")
        qpos = np.asarray(captured_native_payload["qpos"], dtype=np.float64)
        canonical_conversion_time_s = time.perf_counter() - canonical_conversion_started
        if qpos.shape[1] != 36:
            raise ValueError(f"Holosoma Pilot output has width {qpos.shape[1]}, expected 36")
        if len(frame_times) != len(qpos):
            raise RuntimeError("Holosoma timing adapter did not observe exactly one solve per frame")
        if len(steady_times) != 1:
            raise RuntimeError("Holosoma timing adapter did not observe one sequence loop")
        if len(native_core_times) != 1 or len(native_save_times) != 1:
            raise RuntimeError(
                "Holosoma timing adapter did not isolate its in-memory/native-save boundary"
            )
        if len(retargeter_initialization_times) != 1:
            raise RuntimeError("Holosoma retargeter initialization was not isolated")
        steady_method_pipeline_s = (
            adapter_in_memory_time_s
            + main_wall
            - retargeter_initialization_times[0]
            - witness_collection_time_s
            + canonical_conversion_time_s
        )
        witness_hash_started = time.perf_counter()
        data_config_path = (
            package_root / "holosoma_retargeting" / "config_types" / "data_type.py"
        )
        implementation_path = (
            package_root
            / "holosoma_retargeting"
            / "src"
            / "interaction_mesh_retargeter.py"
        )
        native_robot_urdf_path = (
            package_root
            / "holosoma_retargeting"
            / "models"
            / "g1"
            / "g1_29dof.urdf"
        )
        native_robot_xml_path = native_robot_urdf_path.with_suffix(".xml")
        canonical_robot_scene_path = (
            checkout
            / "src"
            / "holosoma"
            / "holosoma"
            / "data"
            / "robots"
            / "g1"
            / "scenes"
            / "scene_g1_29dof_wbt_plane.xml"
        )
        runtime_capture: dict[str, Any] | None = None
        if capture_runtime_witness:
            if (
                runtime_mapped_positions is None
                or runtime_mapped_labels is None
                or runtime_vertex_labels is None
                or runtime_contacts is None
                or runtime_object_points_local is None
                or runtime_object_points_demo is None
                or runtime_object_poses is None
                or runtime_object_poses_augmented is None
                or runtime_q_init is None
                or runtime_contact_labels is None
                or runtime_constraint_contract is None
                or len(runtime_laplacian_targets) != len(qpos)
            ):
                raise RuntimeError("Holosoma runtime pre-solver capture is incomplete")
            runtime_laplacian = np.asarray(
                runtime_laplacian_targets, dtype=np.float64
            )
            if runtime_laplacian.shape[1] != len(runtime_vertex_labels):
                raise RuntimeError(
                    "Holosoma runtime target labels do not match its tensor"
                )
            constraint_contract = dict(runtime_constraint_contract)
            constraint_contract["adjacency_sha256"] = runtime_adjacency_hashes
            constraint_sha256 = hashlib.sha256(
                json.dumps(
                    constraint_contract, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()
            runtime_capture = {
                "boundary": (
                    "InteractionMeshRetargeter.iterate target_laplacian/adj_list "
                    "and retarget_motion contact/constraint inputs observed for every frame"
                ),
                "target_tensor_sha256": tensor_sha256(runtime_laplacian),
                "target_shape": list(runtime_laplacian.shape),
                "target_labels": runtime_vertex_labels.tolist(),
                "target_labels_sha256": tensor_sha256(runtime_vertex_labels),
                "mapped_position_tensor_sha256": tensor_sha256(
                    runtime_mapped_positions
                ),
                "mapped_position_shape": list(runtime_mapped_positions.shape),
                "mapped_position_labels": runtime_mapped_labels.tolist(),
                "mapped_position_labels_sha256": tensor_sha256(
                    runtime_mapped_labels
                ),
                "foot_sticking_tensor_sha256": tensor_sha256(runtime_contacts),
                "foot_sticking_shape": list(runtime_contacts.shape),
                "foot_sticking_labels": runtime_contact_labels,
                "foot_sticking_labels_sha256": tensor_sha256(
                    np.asarray(runtime_contact_labels)
                ),
                "object_points_local_sha256": tensor_sha256(
                    runtime_object_points_local
                ),
                "object_points_demo_sha256": tensor_sha256(
                    runtime_object_points_demo
                ),
                "object_poses_sha256": tensor_sha256(runtime_object_poses),
                "object_poses_augmented_sha256": tensor_sha256(
                    runtime_object_poses_augmented
                ),
                "q_init_sha256": tensor_sha256(runtime_q_init),
                "adjacency_sha256": runtime_adjacency_hashes,
                "adjacency_sequence_sha256": tensor_sha256(
                    np.asarray(runtime_adjacency_hashes)
                ),
                "constraint_contract": constraint_contract,
                "constraint_contract_sha256": constraint_sha256,
                "observed_during_solver_run": True,
                "capture_execution_role": "independent_untimed_or_cold_witness",
            }
        witness_hash_time_s = time.perf_counter() - witness_hash_started
        return CanonicalG1(
            qpos=qpos,
            fps=human.fps,
            source_frame_idx=np.arange(len(qpos), dtype=np.int64),
            valid=np.isfinite(qpos).all(axis=1),
            per_frame_solve_time_s=np.asarray(frame_times),
            metadata={
                "method": "omniretarget",
                "method_family": "upstream",
                "upstream_project": "Holosoma",
                "upstream_commit": "5f48635a3624656a5f46a07df26d43187e59f855",
                "config_sha256": sha256_file(data_config_path),
                "implementation_sha256": sha256_file(implementation_path),
                # Holosoma solves against the retargeting package's URDF/XML
                # pair.  The benchmark evaluates the returned qpos in a
                # separate canonical scene.  Keep these identities explicit:
                # neither hash is evidence that the two geometries are equal.
                "solver_robot_urdf_path": str(
                    native_robot_urdf_path.relative_to(root)
                ),
                "solver_robot_urdf_sha256": sha256_file(
                    native_robot_urdf_path
                ),
                "solver_robot_xml_path": str(
                    native_robot_xml_path.relative_to(root)
                ),
                "solver_robot_xml_sha256": sha256_file(native_robot_xml_path),
                "native_robot_urdf_path": str(native_robot_urdf_path.relative_to(root)),
                "native_robot_urdf_sha256": sha256_file(native_robot_urdf_path),
                "native_robot_xml_path": str(native_robot_xml_path.relative_to(root)),
                "native_robot_xml_sha256": sha256_file(native_robot_xml_path),
                "canonical_evaluator_robot_scene_path": str(
                    canonical_robot_scene_path.relative_to(root)
                ),
                "canonical_evaluator_robot_scene_sha256": sha256_file(
                    canonical_robot_scene_path
                ),
                "solver_and_evaluator_assets_identical": False,
                "native_to_canonical_asset_transfer_floor": {
                    "maximum_joint_origin_component_difference_m": 0.019,
                    "sampled_same_qpos_max_fk_difference_m": 0.012011018631222517,
                    "evidence": "research/unitree_reference_pilot_audit.csv",
                },
                "completion_status": _completion(len(qpos), source_count),
                "canonical_source_sha256": human.source_sha256,
                "native_adapter": "explicit right-first joint reorder; exact canonical Z-up↔native Y-up swap",
                "native_adapter_roundtrip_max_m": 0.0,
                "adapter_changes": "input/output, provenance, and per-frame timing only",
                "quaternion_order": "wxyz",
                "initialization_time_s": retargeter_initialization_times[0],
                "steady_end_to_end_total_s": steady_method_pipeline_s,
                "steady_method_pipeline_s": steady_method_pipeline_s,
                "native_core_total_s": native_core_times[0],
                "native_artifact_save_time_s_excluded_from_core": native_save_times[0],
                "native_artifact_save_elided_during_timing": True,
                "native_input_adapter_in_memory_time_s": adapter_in_memory_time_s,
                "native_input_write_time_s_excluded": native_input_write_time_s,
                "native_core_boundary": (
                    "official post-preprocess LAFAN targets ready to native G1 qpos in memory"
                ),
                "root_scale_multiplier": float(root_scale_multiplier),
                "local_scale_multiplier": float(local_scale_multiplier),
                "scale_intervention": (
                    "post-native-preprocess pre-solver root-path/local-target transform; "
                    "official grounding and frame-zero anchor held fixed"
                ),
                "contact_label_policy": (
                    "fixed canonical source labels"
                    if fixed_contact_labels
                    else "native recomputation after preprocessing"
                ),
                "fixed_contact_labels": bool(fixed_contact_labels),
                "runtime_pre_solver_capture": runtime_capture,
                "runtime_witness_enabled": bool(capture_runtime_witness),
                "runtime_witness_collection_time_s_excluded": witness_collection_time_s,
                "runtime_witness_hash_time_s_excluded": witness_hash_time_s,
            },
        )
    finally:
        os.chdir(original_cwd)
        if sys.path[0] == clean_module_root:
            sys.path.pop(0)
