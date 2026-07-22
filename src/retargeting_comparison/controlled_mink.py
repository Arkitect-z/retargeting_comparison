"""Controlled Sparse and Dense Mink baselines with one frozen configuration."""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import mink
import mujoco
import numpy as np

from .io_utils import load_yaml, sha256_file
from .rotations import quaternion_wxyz_to_matrix
from .schemas import CanonicalG1, CanonicalHuman


def common_baseline_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return all controlled parameters except the intentionally varied task set."""
    return copy.deepcopy(config["common"])


def seed_qpos(model: mujoco.MjModel, config: dict[str, Any], seed_name: str) -> np.ndarray:
    seeds = config["common"]["seed_random_values"]
    if seed_name not in seeds:
        raise ValueError(f"Unknown Sparse seed {seed_name!r}")
    qpos = model.qpos0.copy()
    if seed_name == "neutral":
        return qpos
    rng = np.random.default_rng(int(seeds[seed_name]))
    fraction = float(config["common"]["seed_perturbation_fraction"])
    for joint_id in range(1, model.njnt):
        address = model.jnt_qposadr[joint_id]
        if model.jnt_limited[joint_id]:
            lower, upper = model.jnt_range[joint_id]
            qpos[address] = np.clip(
                qpos[address] + rng.uniform(-fraction, fraction) * (upper - lower),
                lower,
                upper,
            )
    return qpos


class ControlledMinkRetargeter:
    def __init__(
        self,
        repo_root: str | Path,
        variant: str,
        seed_name: str = "neutral",
        config_path: str | Path = "configs/controlled_mink.yaml",
    ):
        initialization_start = time.perf_counter()
        if variant not in {"sparse", "dense"}:
            raise ValueError("Controlled baseline must be 'sparse' or 'dense'")
        self.repo_root = Path(repo_root).resolve()
        self.config_path = self.repo_root / config_path
        self.config = load_yaml(self.config_path)
        self.variant = variant
        self.seed_name = seed_name
        self.model = mujoco.MjModel.from_xml_path(
            str(self.repo_root / self.config["robot_xml"])
        )
        self.configuration = mink.Configuration(
            self.model, q=seed_qpos(self.model, self.config, seed_name)
        )
        self.tasks: list[Any] = []
        self.frame_tasks: list[tuple[dict[str, Any], Any]] = []
        common = self.config["common"]
        for spec in self.config["target_sets"][variant]:
            root = spec["semantic"] == "root"
            task = mink.FrameTask(
                frame_name=spec["robot_body"],
                frame_type="body",
                position_cost=float(spec["position_cost"]),
                orientation_cost=(
                    np.asarray([0.0, 0.0, float(spec["yaw_cost"])])
                    if root
                    else 0.0
                ),
                lm_damping=float(common["lm_damping"]),
            )
            self.tasks.append(task)
            self.frame_tasks.append((spec, task))
        self.posture = mink.PostureTask(
            self.model,
            cost=float(common["posture_cost"]),
            lm_damping=float(common["lm_damping"]),
        )
        self.posture.set_target_from_configuration(self.configuration)
        self.tasks.append(self.posture)
        self.limits = [mink.ConfigurationLimit(self.model)]
        self.initialization_time_s = time.perf_counter() - initialization_start

    def _target_position(
        self, human: CanonicalHuman, frame: int, joint_index: int, scale_group: str
    ) -> np.ndarray:
        common = self.config["common"]
        root = human.world_positions[frame, 0]
        scale = float(common[f"position_scale_{scale_group}"])
        scaled_root = root * float(common["position_scale_root_torso_legs"])
        return scaled_root + (human.world_positions[frame, joint_index] - root) * scale

    def _set_targets(self, human: CanonicalHuman, frame: int, indices: dict[str, int]) -> None:
        root_rotation = quaternion_wxyz_to_matrix(human.world_rotations[frame, 0])
        # Frozen GMR pelvis alignment quaternion [0.5, 0.5, 0.5, 0.5].
        pelvis_offset = quaternion_wxyz_to_matrix(np.asarray([0.5, 0.5, 0.5, 0.5]))
        aligned = root_rotation @ pelvis_offset
        yaw = float(np.arctan2(aligned[1, 0], aligned[0, 0]))
        yaw_rotation = mink.SO3.from_z_radians(yaw)
        identity = mink.SO3.identity()
        for spec, task in self.frame_tasks:
            position = self._target_position(
                human, frame, indices[spec["human_joint"]], spec["scale_group"]
            )
            rotation = yaw_rotation if spec["semantic"] == "root" else identity
            task.set_target(mink.SE3.from_rotation_and_translation(rotation, position))

    def run(self, human: CanonicalHuman, max_frames: int | None = None) -> CanonicalG1:
        human.validate()
        indices = {name: index for index, name in enumerate(human.joint_names.astype(str))}
        required = {spec["human_joint"] for spec, _ in self.frame_tasks}
        missing = sorted(required - set(indices))
        if missing:
            raise ValueError(f"Canonical source lacks controlled targets: {missing}")
        frames = len(human.timestamps) if max_frames is None else min(max_frames, len(human.timestamps))
        qpos: list[np.ndarray] = []
        solve_times: list[float] = []
        end_to_end_times: list[float] = []
        common = self.config["common"]
        dt = self.model.opt.timestep
        for frame in range(frames):
            frame_start = time.perf_counter()
            self._set_targets(human, frame, indices)
            start = time.perf_counter()
            current_error = float(
                np.linalg.norm(
                    np.concatenate([task.compute_error(self.configuration) for task in self.tasks])
                )
            )
            velocity = mink.solve_ik(
                self.configuration,
                self.tasks,
                dt,
                common["solver"],
                float(common["damping"]),
                self.limits,
            )
            self.configuration.integrate_inplace(velocity, dt)
            next_error = float(
                np.linalg.norm(
                    np.concatenate([task.compute_error(self.configuration) for task in self.tasks])
                )
            )
            iteration = 0
            while (
                current_error - next_error > float(common["improvement_tolerance"])
                and iteration < int(common["max_improvement_iterations"])
            ):
                current_error = next_error
                velocity = mink.solve_ik(
                    self.configuration,
                    self.tasks,
                    dt,
                    common["solver"],
                    float(common["damping"]),
                    self.limits,
                )
                self.configuration.integrate_inplace(velocity, dt)
                next_error = float(
                    np.linalg.norm(
                        np.concatenate([task.compute_error(self.configuration) for task in self.tasks])
                    )
                )
                iteration += 1
            solve_times.append(time.perf_counter() - start)
            value = self.configuration.data.qpos.copy()
            if not np.isfinite(value).all():
                break
            qpos.append(value)
            end_to_end_times.append(time.perf_counter() - frame_start)
        if not qpos:
            raise RuntimeError("Controlled Mink produced no valid frames")
        completion = "succeeded" if len(qpos) / len(human.timestamps) >= 0.95 else "incomplete"
        return CanonicalG1(
            qpos=np.asarray(qpos, dtype=np.float64),
            fps=human.fps,
            source_frame_idx=np.arange(len(qpos), dtype=np.int64),
            valid=np.ones(len(qpos), dtype=bool),
            per_frame_solve_time_s=np.asarray(solve_times[: len(qpos)]),
            metadata={
                "method": variant_label(self.variant, self.seed_name),
                "method_family": "controlled_mink",
                "variant": self.variant,
                "seed": self.seed_name,
                "completion_status": completion,
                "canonical_source_sha256": human.source_sha256,
                "config_path": str(self.config_path.relative_to(self.repo_root)),
                "config_sha256": sha256_file(self.config_path),
                "sequential_warm_start": True,
                "orientation_targets": ["root_yaw"],
                "initialization_time_s": self.initialization_time_s,
                "steady_end_to_end_total_s": float(sum(end_to_end_times)),
            },
        )


def variant_label(variant: str, seed_name: str) -> str:
    return f"{variant}-{seed_name.lower()}" if variant == "sparse" else "dense"
