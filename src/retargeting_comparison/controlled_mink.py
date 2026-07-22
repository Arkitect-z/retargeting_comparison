"""Controlled Sparse and Dense Mink baselines with one frozen configuration."""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import mink
import mujoco
import numpy as np

from .calibration import human_heading_yaw, load_evaluator_protocol
from .io_utils import load_yaml, sha256_file
from .robot_model import CanonicalRobotModel, SEMANTIC_FRAMES
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
        scale_policy: dict[str, Any] | None = None,
        method_label: str | None = None,
    ):
        initialization_start = time.perf_counter()
        if variant not in {"sparse", "dense"}:
            raise ValueError("Controlled baseline must be 'sparse' or 'dense'")
        self.repo_root = Path(repo_root).resolve()
        self.config_path = self.repo_root / config_path
        self.config = load_yaml(self.config_path)
        self.evaluator_path = self.repo_root / "manifests" / "evaluator.yaml"
        self.evaluator = load_evaluator_protocol(self.evaluator_path)
        self.variant = variant
        self.seed_name = seed_name
        self.scale_policy = copy.deepcopy(scale_policy)
        self.method_label = method_label
        robot_xml = (self.repo_root / self.config["robot_xml"]).resolve()
        self.robot_xml = robot_xml
        evaluator_xml = Path(str(self.evaluator["robot_xml"]))
        if not evaluator_xml.is_absolute():
            evaluator_xml = (self.repo_root / evaluator_xml).resolve()
        if robot_xml != evaluator_xml:
            raise ValueError(
                "Controlled solver and evaluator must use the same canonical robot scene"
            )
        self.robot = CanonicalRobotModel(robot_xml)
        if self.robot.sha256 != str(self.evaluator["robot_xml_sha256"]):
            raise ValueError("Controlled canonical robot hash differs from evaluator")
        if self.robot.joint_order_sha256 != str(
            self.evaluator["robot_joint_order_sha256"]
        ):
            raise ValueError("Controlled canonical robot joint order differs from evaluator")
        self.model = self.robot.model
        self.configuration = mink.Configuration(
            self.model, q=seed_qpos(self.model, self.config, seed_name)
        )
        self.tasks: list[Any] = []
        self.frame_tasks: list[tuple[dict[str, Any], Any]] = []
        common = self.config["common"]
        self.root_displacement_scale = float(common["root_displacement_scale"])
        self.local_body_scale = float(common["local_body_scale"])
        evaluator_root_scale = float(
            self.evaluator["scale"]["common_root_displacement_scale"]
        )
        evaluator_local_scale = float(
            self.evaluator["scale"]["common_local_body_scale"]
        )
        if not np.isclose(
            self.root_displacement_scale,
            evaluator_root_scale,
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError(
                "Controlled root-displacement scale must equal the frozen evaluator value"
            )
        if not np.isclose(
            self.local_body_scale,
            evaluator_local_scale,
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError(
                "Controlled local/body scale must equal the frozen evaluator value"
            )
        for compatibility_key in (
            "position_scale_root_torso_legs",
            "position_scale_arms",
        ):
            if not np.isclose(
                float(common[compatibility_key]),
                self.local_body_scale,
                atol=1e-12,
                rtol=0.0,
            ):
                raise ValueError(
                    f"{compatibility_key} must be a local/body scale alias"
                )
        self.root_alignment_translation = np.asarray(
            self.evaluator["scale"]["common_root_alignment_translation_m"],
            dtype=np.float64,
        )
        for spec in self.config["target_sets"][variant]:
            expected_type, expected_name = SEMANTIC_FRAMES[spec["semantic"]]
            if (
                spec["robot_frame_type"] != expected_type
                or spec["robot_frame"] != expected_name
            ):
                raise ValueError(
                    "Controlled target frame differs from canonical semantic FK: "
                    f"{spec['semantic']}"
                )
            root = spec["semantic"] == "root"
            task = mink.FrameTask(
                frame_name=spec["robot_frame"],
                frame_type=spec["robot_frame_type"],
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
        # The A/B intervention is initial configuration only.  All seeds share
        # the exact same neutral posture-prior centre; otherwise the diagnostic
        # would mix basin sensitivity with a changed objective.
        self.neutral_posture_configuration = mink.Configuration(
            self.model, q=self.model.qpos0.copy()
        )
        self.posture.set_target_from_configuration(self.neutral_posture_configuration)
        self.tasks.append(self.posture)
        self.temporal = mink.PostureTask(
            self.model,
            cost=float(common["temporal_smoothness_cost"]),
            lm_damping=float(common["lm_damping"]),
        )
        self.temporal.set_target_from_configuration(self.configuration)
        self.tasks.append(self.temporal)
        self.limits = [mink.ConfigurationLimit(self.model)]
        self.initialization_time_s = time.perf_counter() - initialization_start

    def reset(self) -> None:
        """Restore the frozen seed before an independent trajectory solve.

        A formal warm process reuses the parsed MuJoCo model, Mink tasks, and
        limits, but each repetition must start from the same initial condition.
        Resetting only mutable solver state keeps initialization outside the
        steady-state boundary without turning sequential warm start into an
        accidental cross-repetition warm start.
        """

        self.configuration.update(
            seed_qpos(self.model, self.config, self.seed_name).copy()
        )
        self.posture.set_target_from_configuration(
            self.neutral_posture_configuration
        )
        self.temporal.set_target_from_configuration(self.configuration)

    def _target_position(
        self, human: CanonicalHuman, frame: int, joint_index: int, scale_group: str
    ) -> np.ndarray:
        root = human.world_positions[frame, 0]
        if self.scale_policy is not None:
            root_axis = np.asarray(self.scale_policy["root_axis"], dtype=np.float64)
            local_axes = self.scale_policy["local_axes"]
            local_axis = np.asarray(local_axes[scale_group], dtype=np.float64)
            if root_axis.shape != (3,) or local_axis.shape != (3,):
                raise ValueError("Scale-policy axes must contain exactly three values")
            source_root0 = human.world_positions[0, 0]
            robot_anchor = (
                source_root0 * self.root_displacement_scale
                + self.root_alignment_translation
            )
            scaled_root = robot_anchor + (root - source_root0) * root_axis
            return scaled_root + (human.world_positions[frame, joint_index] - root) * local_axis
        source_root0 = human.world_positions[0, 0]
        robot_anchor = (
            source_root0 * self.root_displacement_scale
            + self.root_alignment_translation
        )
        scaled_root = robot_anchor + (
            root - source_root0
        ) * self.root_displacement_scale
        return scaled_root + (
            human.world_positions[frame, joint_index] - root
        ) * self.local_body_scale

    def _set_targets(self, human: CanonicalHuman, frame: int, indices: dict[str, int]) -> None:
        # The evaluator-v3 heading is derived from source geometry, avoiding a
        # BVH-root/G1-pelvis axis convention hidden inside an upstream method.
        yaw = float(human_heading_yaw(human, np.asarray([frame]))[0])
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
        self.reset()
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
            # Freeze q[t-1] as a weak temporal target for all iterations of
            # frame t.  Sequential warm start alone is not a smoothness cost.
            self.temporal.set_target_from_configuration(self.configuration)
            self._set_targets(human, frame, indices)
            start = time.perf_counter()
            maximum_iterations = (
                int(common["first_frame_maximum_iterations"])
                if frame == 0
                else int(common["max_improvement_iterations"]) + 1
            )
            minimum_iterations = (
                int(common["first_frame_minimum_iterations"])
                if frame == 0
                else 1
            )
            current_error = float("inf")
            for iteration in range(maximum_iterations):
                before = float(
                    np.linalg.norm(
                        np.concatenate(
                            [task.compute_error(self.configuration) for task in self.tasks]
                        )
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
                current_error = float(
                    np.linalg.norm(
                        np.concatenate([task.compute_error(self.configuration) for task in self.tasks])
                    )
                )
                if (
                    iteration + 1 >= minimum_iterations
                    and before - current_error
                    <= float(common["improvement_tolerance"])
                ):
                    break
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
                "method": self.method_label or variant_label(self.variant, self.seed_name),
                "method_family": "controlled_mink",
                "variant": self.variant,
                "seed": self.seed_name,
                "benchmark_revision": self.config["benchmark_revision"],
                "completion_status": completion,
                "canonical_source_sha256": human.source_sha256,
                "config_path": str(self.config_path.relative_to(self.repo_root)),
                "config_sha256": sha256_file(self.config_path),
                "evaluator_sha256": sha256_file(self.evaluator_path),
                "sequential_warm_start": True,
                "initialization_intervention_only": True,
                "posture_prior_target": "neutral_model_qpos0_for_all_seeds",
                "temporal_smoothness_cost": float(common["temporal_smoothness_cost"]),
                "root_displacement_scale": self.root_displacement_scale,
                "local_body_scale": self.local_body_scale,
                "position_scale_root_torso_legs": float(
                    common["position_scale_root_torso_legs"]
                ),
                "position_scale_arms": float(common["position_scale_arms"]),
                "scale_policy": self.scale_policy,
                "root_alignment_translation_m": self.root_alignment_translation.tolist(),
                "root_anchor_policy": self.evaluator["scale"]["root_anchor_policy"],
                "root_local_and_anchor_frozen_separately": True,
                "canonical_robot_xml": str(
                    self.robot_xml.relative_to(self.repo_root)
                ),
                "canonical_robot_xml_sha256": self.robot.sha256,
                "canonical_joint_order": list(self.robot.joint_names),
                "canonical_joint_order_sha256": self.robot.joint_order_sha256,
                "canonical_fk_contract": "CanonicalRobotModel.semantic_positions/v1",
                "orientation_targets": ["root_yaw"],
                "initialization_time_s": self.initialization_time_s,
                "steady_end_to_end_total_s": float(sum(end_to_end_times)),
                "native_core_total_s": float(sum(end_to_end_times)),
                "native_core_timing_boundary": (
                    "per-frame target construction start -> finite canonical qpos in memory; "
                    "sum excludes model/config initialization and serialization"
                ),
            },
        )


def variant_label(variant: str, seed_name: str) -> str:
    return f"{variant}-{seed_name.lower()}" if variant == "sparse" else "dense"
