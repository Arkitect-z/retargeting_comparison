"""Adapters for frozen upstream GMR and OmniRetarget/Holosoma implementations."""

from __future__ import annotations

import os
import sys
import time
from dataclasses import replace
from pathlib import Path

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


def run_gmr(
    repo_root: str | Path,
    source_bvh: str | Path,
    canonical_source: str | Path,
    max_frames: int | None = None,
) -> CanonicalG1:
    """Run the clean official GMR LAFAN→G1 path without visualization."""
    root = Path(repo_root).resolve()
    gmr_root = root / "external" / "GMR"
    sys.path.insert(0, str(gmr_root))
    try:
        from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting
        from general_motion_retargeting.utils.lafan1 import load_bvh_file

        human = CanonicalHuman.load(canonical_source)
        frames, human_height = load_bvh_file(str(source_bvh), format="lafan1")
        count = len(frames) if max_frames is None else min(max_frames, len(frames))
        retargeter = GeneralMotionRetargeting(
            src_human="bvh_lafan1",
            tgt_robot="unitree_g1",
            actual_human_height=human_height,
            solver="daqp",
            damping=0.5,
            verbose=False,
        )
        qpos: list[np.ndarray] = []
        solve_times: list[float] = []
        for frame in frames[:count]:
            start = time.perf_counter()
            value = retargeter.retarget(frame, offset_to_ground=False)
            solve_times.append(time.perf_counter() - start)
            if not np.isfinite(value).all():
                break
            qpos.append(np.asarray(value, dtype=np.float64))
        if not qpos:
            raise RuntimeError("GMR produced no finite frames")
        config_path = gmr_root / "general_motion_retargeting" / "ik_configs" / "bvh_lafan1_to_g1.json"
        xml_path = gmr_root / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"
        return CanonicalG1(
            qpos=np.asarray(qpos),
            fps=human.fps,
            source_frame_idx=np.arange(len(qpos), dtype=np.int64),
            valid=np.ones(len(qpos), dtype=bool),
            per_frame_solve_time_s=np.asarray(solve_times[: len(qpos)]),
            metadata={
                "method": "gmr",
                "method_family": "upstream",
                "upstream_commit": "bb1bbe40774794fceb2a7c579a3464a28e68c844",
                "completion_status": _completion(len(qpos), len(human.timestamps)),
                "canonical_source_sha256": human.source_sha256,
                "native_source_sha256": sha256_file(source_bvh),
                "config_sha256": sha256_file(config_path),
                "robot_xml_sha256": sha256_file(xml_path),
                "adapter_changes": "headless I/O, provenance, and per-frame timing only",
                "quaternion_order": "wxyz",
            },
        )
    finally:
        if sys.path[0] == str(gmr_root):
            sys.path.pop(0)


def prepare_holosoma_lafan_input(
    human: CanonicalHuman, output_dir: str | Path, task_name: str
) -> Path:
    """Write Holosoma's native Y-up position array with explicit joint reordering.

    Holosoma's loader applies [x,y,z]→[x,z,y].  Applying that involution here
    preserves the benchmark's canonical Z-up coordinates exactly.  Toe aliases
    and the upstream right-first joint order are made explicit rather than
    relying on the left-first order of the Ubisoft BVH files.
    """
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
    output = Path(output_dir) / f"{task_name}.npy"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, y_up)
    restored = y_up[..., [0, 2, 1]]
    if not np.array_equal(restored, canonical):
        raise RuntimeError("Holosoma coordinate adapter failed its exact round-trip")
    return output


def run_holosoma(
    repo_root: str | Path,
    canonical_source: str | Path,
    work_dir: str | Path,
    max_frames: int | None = None,
) -> CanonicalG1:
    """Run clean Holosoma LAFAN→G1 with a canonical file adapter."""
    root = Path(repo_root).resolve()
    canonical_source = Path(canonical_source).resolve()
    checkout = root / "external" / "holosoma"
    package_root = checkout / "src" / "holosoma_retargeting"
    clean_module_root = str(package_root)
    sys.path.insert(0, clean_module_root)
    original_cwd = Path.cwd()
    try:
        # This worker is always launched in a fresh method-specific process.
        from holosoma_retargeting.config_types.retargeter import RetargeterConfig
        from holosoma_retargeting.config_types.retargeting import RetargetingConfig
        from holosoma_retargeting.config_types.task import TaskConfig
        from holosoma_retargeting.examples import robot_retarget
        from holosoma_retargeting.src.interaction_mesh_retargeter import InteractionMeshRetargeter

        human = CanonicalHuman.load(canonical_source)
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
        run_root = Path(work_dir).resolve()
        input_dir = run_root / "native_input"
        save_dir = run_root / "native_output"
        task_name = "pilot_canonical"
        prepare_holosoma_lafan_input(human, input_dir, task_name)
        save_dir.mkdir(parents=True, exist_ok=True)

        frame_times: list[float] = []
        original_iterate = InteractionMeshRetargeter.iterate

        def timed_iterate(self, *args, **kwargs):
            start = time.perf_counter()
            result = original_iterate(self, *args, **kwargs)
            frame_times.append(time.perf_counter() - start)
            return result

        InteractionMeshRetargeter.iterate = timed_iterate
        cfg = RetargetingConfig(
            task_type="robot_only",
            robot="g1",
            data_format="lafan",
            task_name=task_name,
            data_path=input_dir,
            save_dir=save_dir,
            task_config=TaskConfig(ground_range=(-10.0, 10.0)),
            retargeter=RetargeterConfig(foot_sticking_tolerance=0.02),
        )
        os.chdir(package_root / "holosoma_retargeting")
        try:
            robot_retarget.main(cfg)
        finally:
            InteractionMeshRetargeter.iterate = original_iterate
        native_output = save_dir / f"{task_name}.npz"
        with np.load(native_output, allow_pickle=False) as data:
            qpos = np.asarray(data["qpos"], dtype=np.float64)
        if qpos.shape[1] != 36:
            raise ValueError(f"Holosoma Pilot output has width {qpos.shape[1]}, expected 36")
        if len(frame_times) != len(qpos):
            raise RuntimeError("Holosoma timing adapter did not observe exactly one solve per frame")
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
                "completion_status": _completion(len(qpos), source_count),
                "canonical_source_sha256": human.source_sha256,
                "native_adapter": "explicit right-first joint reorder; exact canonical Z-up↔native Y-up swap",
                "native_adapter_roundtrip_max_m": 0.0,
                "adapter_changes": "input/output, provenance, and per-frame timing only",
                "quaternion_order": "wxyz",
            },
        )
    finally:
        os.chdir(original_cwd)
        if sys.path[0] == clean_module_root:
            sys.path.pop(0)
