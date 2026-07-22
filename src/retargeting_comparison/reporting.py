"""Build evidence tables, figures, Stage 2 projection, and Markdown deliverables."""

from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .constants import FULL_LAFAN_STOP_MESSAGE, STAGE1_RUN_DIRECTORIES
from .calibration import load_evaluator_protocol
from .evaluator import HUMAN_SEMANTIC_JOINTS, evaluate_motion, save_evaluation
from .io_utils import atomic_write_json, atomic_write_text, load_yaml, sha256_file
from .robot_model import CanonicalRobotModel, default_robot_scene
from .rotations import quaternion_wxyz_to_matrix, yaw_from_matrix
from .schemas import CanonicalG1, CanonicalHuman, RunManifest


CORE_LABELS = (
    "sparse-neutral",
    "sparse-a",
    "sparse-b",
    "dense",
    "gmr",
    "omniretarget",
)
OPERATING_POINTS = ("sparse-neutral", "dense", "gmr", "omniretarget")
REPORTS = (
    "PILOT_REPORT.md",
    "EXECUTIVE_SUMMARY.md",
    "GO_NO_GO.md",
    "METHOD_SCOPE.md",
    "SPARSE_IK_ANALYSIS.md",
    "INTERACTION_CASE_STUDY.md",
    "REPRODUCE_PILOT.md",
    "PRESENTATION.md",
    "SCALE_POLICY_SENSITIVITY.md",
    "UNITREE_REFERENCE_COMPARISON.md",
    "STAGE1_REVIEW.md",
)


def _sequence(root: Path) -> dict[str, Any]:
    return load_yaml(root / "manifests" / "pilot_sequence.yaml")


def _run_paths(root: Path) -> dict[str, Path]:
    sequence_id = _sequence(root)["sequence_id"]
    return {
        label: root
        / "runs"
        / sequence_id
        / STAGE1_RUN_DIRECTORIES[label]
        / "canonical_g1.npz"
        for label in CORE_LABELS
    }


def evaluate_core(root: Path) -> pd.DataFrame:
    sequence = _sequence(root)
    human = CanonicalHuman.load(root / sequence["canonical_path"])
    robot = CanonicalRobotModel(default_robot_scene(root))
    protocol_path = root / "manifests" / "evaluator.yaml"
    protocol = load_evaluator_protocol(protocol_path)
    protocol["manifest_sha256"] = sha256_file(protocol_path)
    rows = []
    for label, path in _run_paths(root).items():
        if not path.is_file():
            raise FileNotFoundError(f"Required core output is missing: {path}")
        motion = CanonicalG1.load(path)
        motion.validate(source_frame_count=len(human.timestamps))
        table, summary = evaluate_motion(human, motion, robot, protocol)
        save_evaluation(table, summary, root / "metrics" / "runs", label)
        summary["label"] = label
        summary["output_sha256"] = sha256_file(path)
        rows.append(summary)
    frame = pd.DataFrame(rows).sort_values("label")
    frame.to_csv(root / "metrics" / "core_summary.csv", index=False)
    frame.to_parquet(root / "metrics" / "core_summary.parquet", index=False)
    return frame


def collect_timing(root: Path, core: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sequence_id = _sequence(root)["sequence_id"]
    raw_rows = []
    summary_rows = []
    for label in CORE_LABELS:
        run_dir = root / "runs" / sequence_id / STAGE1_RUN_DIRECTORIES[label]
        timing_path = run_dir / "timing_refined.json"
        if not timing_path.is_file():
            timing_path = run_dir / "timing.json"
        timing = json.loads(timing_path.read_text())
        repetitions = [timing["cold"], *timing["warmup"], *timing["measured_warm"]]
        for sequence_index, repetition in enumerate(repetitions):
            duration = repetition["frame_count"] / float(_sequence(root)["fps"])
            raw_rows.append(
                {
                    "method": label,
                    "sequence_index": sequence_index,
                    "role": repetition["role"] if sequence_index else "cold",
                    "wall_time_s": repetition["wall_time_s"],
                    "native_total_s": repetition["native_total_s"],
                    "frame_count": repetition["frame_count"],
                    "end_to_end_rtf": repetition["wall_time_s"] / duration,
                    "native_core_rtf": repetition["native_total_s"] / duration,
                }
            )
        warm_rtf = np.asarray(timing["end_to_end_rtf_raw"], dtype=float)
        summary_rows.append(
            {
                "label": label,
                "end_to_end_rtf_median": timing["end_to_end_rtf_median"],
                "native_core_rtf_median": timing["native_core_rtf_median"],
                "end_to_end_rtf_cv": float(np.std(warm_rtf) / max(np.mean(warm_rtf), 1e-12)),
                "cold_process_wall_s": timing["cold_process_wall_s"],
                "cold_import_startup_s": timing["cold_import_startup_s"],
                "initialization_and_adapter_s_cold": timing["initialization_and_adapter_s_cold"],
            }
        )
    raw = pd.DataFrame(raw_rows)
    summary = pd.DataFrame(summary_rows)
    raw.to_csv(root / "metrics" / "timing_raw.csv", index=False)
    raw.to_parquet(root / "metrics" / "timing_raw.parquet", index=False)
    summary.to_csv(root / "metrics" / "timing_summary.csv", index=False)
    return raw, core.merge(summary, on="label", how="left")


def _robot_root_frame(robot: CanonicalRobotModel, motion: CanonicalG1) -> np.ndarray:
    semantics = [name for name in HUMAN_SEMANTIC_JOINTS if name != "root"]
    frames = [robot.semantic_positions(qpos) for qpos in motion.qpos]
    root = np.stack([frame["root"] for frame in frames])
    points = np.stack([[frame[name] for name in semantics] for frame in frames]) - root[:, None]
    yaw = yaw_from_matrix(quaternion_wxyz_to_matrix(motion.qpos[:, 3:7]))
    cosine, sine = np.cos(-yaw), np.sin(-yaw)
    x = cosine[:, None] * points[..., 0] - sine[:, None] * points[..., 1]
    y = sine[:, None] * points[..., 0] + cosine[:, None] * points[..., 1]
    return np.stack([x, y, points[..., 2]], axis=-1)


def sparse_seed_divergence(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    robot = CanonicalRobotModel(default_robot_scene(root))
    motions = {
        label: CanonicalG1.load(_run_paths(root)[label])
        for label in ("sparse-neutral", "sparse-a", "sparse-b")
    }
    root_frames = {label: _robot_root_frame(robot, motion) for label, motion in motions.items()}
    rows = []
    for left, right in combinations(motions, 2):
        joint_delta = np.arctan2(
            np.sin(motions[left].qpos[:, 7:] - motions[right].qpos[:, 7:]),
            np.cos(motions[left].qpos[:, 7:] - motions[right].qpos[:, 7:]),
        )
        joint_rms = np.sqrt(np.mean(joint_delta**2, axis=1))
        point_rms = np.sqrt(np.mean(np.sum((root_frames[left] - root_frames[right]) ** 2, axis=2), axis=1))
        rows.extend(
            {
                "pair": f"{left}__{right}",
                "frame": frame,
                "joint_angle_rms_rad": float(joint_rms[frame]),
                "robot_rf_point_rms_m": float(point_rms[frame]),
            }
            for frame in range(len(joint_rms))
        )
    per_frame = pd.DataFrame(rows)
    summary = (
        per_frame.groupby("pair", as_index=False)
        .agg(
            joint_angle_rms_mean_rad=("joint_angle_rms_rad", "mean"),
            joint_angle_rms_max_rad=("joint_angle_rms_rad", "max"),
            robot_rf_point_rms_mean_m=("robot_rf_point_rms_m", "mean"),
            robot_rf_point_rms_max_m=("robot_rf_point_rms_m", "max"),
        )
    )
    per_frame.to_csv(root / "metrics" / "sparse_seed_divergence_per_frame.csv", index=False)
    summary.to_csv(root / "metrics" / "sparse_seed_divergence.csv", index=False)
    semantics = [name for name in HUMAN_SEMANTIC_JOINTS if name != "root"]
    targeted_indices = [semantics.index(name) for name in ("left_wrist", "right_wrist", "left_ankle", "right_ankle")]
    untracked_indices = [index for index in range(len(semantics)) if index not in targeted_indices]
    stacked_points = np.stack([root_frames[label] for label in motions])
    point_variance = np.var(stacked_points, axis=0)
    point_variance_norm = np.sum(point_variance, axis=2)
    joint_angles = np.stack([motion.qpos[:, 7:] for motion in motions.values()])
    circular_mean = np.arctan2(np.mean(np.sin(joint_angles), axis=0), np.mean(np.cos(joint_angles), axis=0))
    circular_delta = np.arctan2(
        np.sin(joint_angles - circular_mean[None]),
        np.cos(joint_angles - circular_mean[None]),
    )
    qpos_variance = np.mean(circular_delta**2, axis=(0, 2))
    variance = pd.DataFrame(
        {
            "frame": np.arange(stacked_points.shape[1]),
            "targeted_point_variance_m2": np.mean(
                point_variance_norm[:, targeted_indices], axis=1
            ),
            "untracked_point_variance_m2": np.mean(
                point_variance_norm[:, untracked_indices], axis=1
            ),
            "all_point_variance_m2": np.mean(point_variance_norm, axis=1),
            "qpos_circular_variance_rad2": qpos_variance,
        }
    )
    variance.to_csv(root / "metrics" / "sparse_seed_variance_per_frame.csv", index=False)
    maximum_index = int(variance.all_point_variance_m2.idxmax())
    variance_summary = pd.DataFrame(
        [
            {
                "targeted_point_variance_mean_m2": variance.targeted_point_variance_m2.mean(),
                "untracked_point_variance_mean_m2": variance.untracked_point_variance_m2.mean(),
                "all_point_variance_mean_m2": variance.all_point_variance_m2.mean(),
                "qpos_circular_variance_mean_rad2": variance.qpos_circular_variance_rad2.mean(),
                "maximum_divergence_frame": maximum_index,
                "maximum_divergence_time_s": maximum_index / motions["sparse-neutral"].fps,
                "maximum_all_point_variance_m2": variance.loc[
                    maximum_index, "all_point_variance_m2"
                ],
            }
        ]
    )
    variance_summary.to_csv(root / "metrics" / "sparse_seed_variance.csv", index=False)
    return per_frame, summary


def scale_diagnostics(root: Path, core: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "label",
        "common_static_scale",
        "common_local_body_scale",
        "common_root_displacement_scale",
        "native_root_scale",
        "effective_root_xy_scale",
        "root_scale_bias_fraction",
        "root_translation_common_scale_mean_m",
        "root_translation_native_scale_mean_m",
        "root_translation_scale_invariant_mean_m",
    ]
    frame = core[columns].copy()
    frame["effective_minus_native_scale"] = (
        frame.effective_root_xy_scale - frame.native_root_scale
    )
    frame.to_csv(root / "metrics" / "root_scale_diagnostics.csv", index=False)
    frame.to_parquet(root / "metrics" / "root_scale_diagnostics.parquet", index=False)
    return frame


def controlled_task_residuals(root: Path) -> pd.DataFrame:
    """Measure the actual Sparse/Dense v3 world-position objectives.

    RF-KPE is a morphology-referenced fidelity metric.  It must not be
    mislabeled as the residual minimized by the controlled optimizer, so the
    latter is exported independently here.
    """

    sequence = _sequence(root)
    human = CanonicalHuman.load(root / sequence["canonical_path"])
    robot = CanonicalRobotModel(default_robot_scene(root))
    config = load_yaml(root / "configs" / "controlled_mink.yaml")
    protocol = load_evaluator_protocol(root / "manifests" / "evaluator.yaml")
    root_alignment = np.asarray(
        protocol["scale"]["common_root_alignment_translation_m"], dtype=np.float64
    )
    indices = {name: index for index, name in enumerate(human.joint_names.astype(str))}
    per_frame_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for label in ("sparse-neutral", "sparse-a", "sparse-b", "dense"):
        variant = "dense" if label == "dense" else "sparse"
        motion = CanonicalG1.load(_run_paths(root)[label])
        robot_frames = [robot.semantic_positions(qpos) for qpos in motion.qpos]
        task_errors: dict[str, np.ndarray] = {}
        for spec in config["target_sets"][variant]:
            root_position = human.world_positions[:, 0]
            root_scale = float(config["common"]["root_displacement_scale"])
            local_scale = float(config["common"]["local_body_scale"])
            robot_anchor = (
                human.world_positions[0, 0] * root_scale + root_alignment
            )
            scaled_root = robot_anchor + (
                root_position - human.world_positions[0, 0]
            ) * root_scale
            target = scaled_root + (
                human.world_positions[:, indices[spec["human_joint"]]]
                - root_position
            ) * local_scale
            actual = np.stack([frame[spec["semantic"]] for frame in robot_frames])
            task_errors[spec["semantic"]] = np.linalg.norm(actual - target, axis=1)
        root_error = task_errors["root"]
        ee_error = np.mean(
            np.stack([task_errors[name] for name in ("left_wrist", "right_wrist", "left_ankle", "right_ankle")]),
            axis=0,
        )
        all_error = np.mean(np.stack(list(task_errors.values())), axis=0)
        added_names = [name for name in task_errors if name not in {"root", "left_wrist", "right_wrist", "left_ankle", "right_ankle"}]
        added_error = (
            np.mean(np.stack([task_errors[name] for name in added_names]), axis=0)
            if added_names
            else np.zeros(len(motion.qpos))
        )
        for frame in range(len(motion.qpos)):
            row: dict[str, Any] = {
                "label": label,
                "frame": frame,
                "root_position_residual_m": root_error[frame],
                "four_ee_position_residual_m": ee_error[frame],
                "dense_added_position_residual_m": added_error[frame],
                "all_declared_position_residual_m": all_error[frame],
            }
            row.update(
                {
                    f"{semantic}_position_residual_m": values[frame]
                    for semantic, values in task_errors.items()
                }
            )
            per_frame_rows.append(row)
        summary_rows.append(
            {
                "label": label,
                "root_position_residual_mean_m": root_error.mean(),
                "root_position_residual_p95_m": np.percentile(root_error, 95),
                "four_ee_position_residual_mean_m": ee_error.mean(),
                "four_ee_position_residual_p95_m": np.percentile(ee_error, 95),
                "dense_added_position_residual_mean_m": added_error.mean(),
                "all_declared_position_residual_mean_m": all_error.mean(),
            }
        )
    per_frame = pd.DataFrame(per_frame_rows)
    summary = pd.DataFrame(summary_rows)
    per_frame.to_csv(root / "metrics" / "controlled_task_residuals_per_frame.csv", index=False)
    per_frame.to_parquet(
        root / "metrics" / "controlled_task_residuals_per_frame.parquet", index=False
    )
    summary.to_csv(root / "metrics" / "controlled_task_residuals.csv", index=False)
    return summary


def collect_interaction(root: Path) -> pd.DataFrame:
    rows = []
    for case in ("box", "climb"):
        for variant in ("full", "no-hard"):
            manifest_path = (
                root / "runs" / "interaction_manifests" / f"interaction__{case}__{variant}.json"
            )
            manifest = RunManifest.load(manifest_path)
            path = Path(manifest.output_path or "")
            if not path.is_absolute():
                path = root / path
            if not path.is_file():
                raise FileNotFoundError(f"Required interaction output is missing: {path}")
            value = json.loads(path.read_text())
            value.pop("native_frame_times_s", None)
            per_frame = pd.read_csv(path.parent / "per_frame_metrics.csv")
            depth = per_frame.penetration_depth_m.to_numpy(dtype=float)
            left_stance = (
                per_frame.left_stance
                if per_frame.left_stance.dtype == bool
                else per_frame.left_stance.astype(str).str.lower().eq("true")
            )
            right_stance = (
                per_frame.right_stance
                if per_frame.right_stance.dtype == bool
                else per_frame.right_stance.astype(str).str.lower().eq("true")
            )
            left_violation = left_stance & (
                per_frame.left_foot_xy_displacement_m > np.sqrt(2.0) * 0.001 + 1e-4
            )
            right_violation = right_stance & (
                per_frame.right_foot_xy_displacement_m > np.sqrt(2.0) * 0.001 + 1e-4
            )
            value["penetration_any_frame_rate"] = float(np.mean(depth > 0.0))
            value["penetration_frame_rate"] = float(np.mean(depth > 0.0011))
            value["penetration_primary_threshold_m"] = 0.0011
            value["foot_sticking_violation_frame_rate"] = float(
                np.mean(left_violation | right_violation)
            )
            value["reported_metric_revision"] = "tolerance-aware-v2"
            rows.append(value)
    frame = pd.DataFrame(rows).sort_values(["case", "variant"])
    frame.to_csv(root / "metrics" / "interaction_summary.csv", index=False)
    frame.to_parquet(root / "metrics" / "interaction_summary.parquet", index=False)
    return frame


def stage2_projection(root: Path, timing: pd.DataFrame) -> pd.DataFrame:
    dataset = load_yaml(root / "manifests" / "dataset.yaml")
    total_frames = int(dataset["validation"]["total_frames"])
    sequence_count = int(dataset["validation"]["bvh_file_count"])
    fps = float(dataset["validation"]["unique_nominal_fps"])
    duration_s = total_frames / fps
    retry_rate_observed = 0.0
    rows = []
    for label in OPERATING_POINTS:
        row = timing.loc[timing.label == label].iloc[0]
        steady_hours = float(row.end_to_end_rtf_median * duration_s / 3600.0)
        startup_adapter_s = float(
            row.cold_import_startup_s + row.initialization_and_adapter_s_cold
        )
        startup_adapter_hours = startup_adapter_s * sequence_count / 3600.0
        raw_hours = (steady_hours + startup_adapter_hours) * (1.0 + retry_rate_observed)
        rows.append(
            {
                "method": label,
                "lafan_frames": total_frames,
                "lafan_sequences": sequence_count,
                "measured_pilot_end_to_end_rtf": row.end_to_end_rtf_median,
                "projected_steady_wall_hours": steady_hours,
                "measured_startup_and_adapter_s_per_sequence": startup_adapter_s,
                "projected_startup_and_adapter_hours": startup_adapter_hours,
                "observed_retry_rate": retry_rate_observed,
                "raw_projected_wall_hours": raw_hours,
                "safety_factor": 1.5,
                "safe_projected_wall_hours": raw_hours * 1.5,
            }
        )
    frame = pd.DataFrame(rows)
    retained_paths = [*_run_paths(root).values()]
    retained_paths.extend(
        path
        for path in (root / "metrics" / "runs").glob("*")
        if path.is_file() and ("per_frame" in path.name or "summary" in path.name)
    )
    pilot_bytes = sum(path.stat().st_size for path in retained_paths)
    projected_storage_gb = pilot_bytes / 600.0 * total_frames * 1.5 / 1e9
    total_safe_hours = float(frame.safe_projected_wall_hours.sum())
    projection = {
        "lafan_frames": total_frames,
        "lafan_sequences": sequence_count,
        "lafan_duration_hours": duration_s / 3600.0,
        "runtime_safety_factor": 1.5,
        "retry_rate_observed": retry_rate_observed,
        "startup_and_adapter_assumed_per_sequence": True,
        "safe_projected_wall_hours_serial": total_safe_hours,
        "safe_projected_storage_gb": projected_storage_gb,
        "retained_pilot_artifact_bytes": pilot_bytes,
        "storage_excludes_rebuildable_logs_and_timing_worker_trajectories": True,
        "wall_budget_hours": 48,
        "storage_budget_gb": 200,
        "within_wall_budget": total_safe_hours <= 48,
        "within_storage_budget": projected_storage_gb <= 200,
        "recommendation": (
            "Full-LAFAN remains gated; use a deterministically named reduced-LAFAN design or optimize/parallelize OmniRetarget before approval."
            if total_safe_hours > 48
            else "Projection fits both budgets; explicit user approval is still required."
        ),
    }
    frame.to_csv(root / "metrics" / "stage2_projection.csv", index=False)
    atomic_write_json(root / "metrics" / "stage2_projection.json", projection)
    return frame


def _save_plot(root: Path, name: str, source: pd.DataFrame, draw) -> None:
    source_dir = root / "figures" / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    source.to_csv(source_dir / f"{name}.csv", index=False)
    figure, axis = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    draw(axis, source)
    for suffix, kwargs in (("svg", {}), ("pdf", {}), ("png", {"dpi": 300})):
        figure.savefig(root / "figures" / f"{name}.{suffix}", **kwargs)
    plt.close(figure)
    svg_path = root / "figures" / f"{name}.svg"
    atomic_write_text(
        svg_path,
        "\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n",
    )


def build_figures(
    root: Path,
    core_timing: pd.DataFrame,
    seed_summary: pd.DataFrame,
    interaction: pd.DataFrame,
    scales: pd.DataFrame,
) -> None:
    points = core_timing[core_timing.label.isin(OPERATING_POINTS)].copy()

    def scatter(axis, data, x, y, xlabel, ylabel):
        for row in data.itertuples():
            axis.scatter(getattr(row, x), getattr(row, y), s=55)
            axis.annotate(row.label, (getattr(row, x), getattr(row, y)), xytext=(4, 4), textcoords="offset points")
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)

    def rtf_scatter(axis, data, y, ylabel):
        scatter(
            axis,
            data,
            "end_to_end_rtf_median",
            y,
            "End-to-end steady RTF (log scale; lower is faster)",
            ylabel,
        )
        axis.set_xscale("log")

    _save_plot(
        root,
        "rtf_vs_rf_kpe_all",
        points[["label", "end_to_end_rtf_median", "rf_kpe_all_mean_m"]],
        lambda ax, data: rtf_scatter(ax, data, "rf_kpe_all_mean_m", "RF-KPE-all mean (m, lower is better)"),
    )
    _save_plot(
        root,
        "rtf_vs_artifact_rate",
        points[["label", "end_to_end_rtf_median", "artifact_rate"]],
        lambda ax, data: rtf_scatter(ax, data, "artifact_rate", "Artifact frame rate"),
    )
    _save_plot(
        root,
        "targeted_vs_untracked",
        points[["label", "rf_kpe_targeted_mean_m", "rf_kpe_untracked_mean_m"]],
        lambda ax, data: scatter(ax, data, "rf_kpe_targeted_mean_m", "rf_kpe_untracked_mean_m", "Targeted hand/foot RF-KPE (m)", "Untracked-body RF-KPE (m)"),
    )

    def seed_bar(axis, data):
        axis.bar(data.pair, data.robot_rf_point_rms_mean_m)
        axis.set_ylabel("Mean pairwise RF point RMS (m)")
        axis.tick_params(axis="x", rotation=18)
        axis.grid(axis="y", alpha=0.25)

    _save_plot(root, "sparse_seed_divergence", seed_summary, seed_bar)

    def scale_bar(axis, data):
        x = np.arange(len(data))
        width = 0.35
        axis.bar(x - width / 2, data.native_root_scale, width, label="Declared native")
        axis.bar(x + width / 2, data.effective_root_xy_scale, width, label="Measured output")
        axis.axhline(
            float(data.common_root_displacement_scale.iloc[0]),
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="Common root-displacement gain",
        )
        axis.set_xticks(x, data.label)
        axis.set_ylabel("Root translation scale")
        axis.legend()
        axis.grid(axis="y", alpha=0.25)

    _save_plot(root, "root_scale_policies", scales, scale_bar)

    root_errors = scales[
        [
            "label",
            "root_translation_common_scale_mean_m",
            "root_translation_native_scale_mean_m",
            "root_translation_scale_invariant_mean_m",
        ]
    ]

    def root_error_bar(axis, data):
        x = np.arange(len(data))
        width = 0.24
        axis.bar(
            x - width,
            data.root_translation_common_scale_mean_m,
            width,
            label="Common-scale",
        )
        axis.bar(
            x,
            data.root_translation_native_scale_mean_m,
            width,
            label="Native-target",
        )
        axis.bar(
            x + width,
            data.root_translation_scale_invariant_mean_m,
            width,
            label="Scale-invariant shape",
        )
        axis.set_xticks(x, data.label)
        axis.set_ylabel("Mean root trajectory error (m)")
        axis.legend()
        axis.grid(axis="y", alpha=0.25)

    _save_plot(root, "root_error_decomposition", root_errors, root_error_bar)

    artifact_components = points[
        [
            "label",
            "foot_skating_frame_rate",
            "ground_penetration_frame_rate",
            "joint_limit_violation_frame_rate",
            "invalid_frame_rate",
        ]
    ]

    def artifact_bar(axis, data):
        x = np.arange(len(data))
        bottom = np.zeros(len(data))
        for column, label in (
            ("foot_skating_frame_rate", "foot skating"),
            ("ground_penetration_frame_rate", "ground penetration"),
            ("joint_limit_violation_frame_rate", "joint limit"),
            ("invalid_frame_rate", "invalid"),
        ):
            values = data[column].to_numpy()
            axis.bar(x, values, bottom=bottom, label=label)
            bottom += values
        axis.set_xticks(x, data.label)
        axis.set_ylabel("Component frame rates (stacked, overlaps possible)")
        axis.legend()
        axis.grid(axis="y", alpha=0.25)

    _save_plot(root, "artifact_components", artifact_components, artifact_bar)
    interaction_source = interaction[
        [
            "case",
            "variant",
            "strict_contact_2cm_frame_rate",
            "near_contact_5cm_frame_rate",
            "proximity_10cm_frame_rate",
            "penetration_frame_rate",
            "foot_sticking_violation_frame_rate",
        ]
    ]

    def interaction_bar(axis, data):
        labels = [f"{row.case}\n{row.variant}" for row in data.itertuples()]
        x = np.arange(len(labels))
        axis.bar(x - 0.18, data.strict_contact_2cm_frame_rate, width=0.36, label="≤2 cm")
        axis.bar(x + 0.18, data.penetration_frame_rate, width=0.36, label="penetration")
        axis.set_xticks(x, labels)
        axis.set_ylabel("Frame rate")
        axis.legend()
        axis.grid(axis="y", alpha=0.25)

    _save_plot(root, "interaction_full_vs_no_hard", interaction_source, interaction_bar)


def _fmt(value: Any, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _write_report(path: Path, body: str) -> None:
    atomic_write_text(path, body.rstrip() + "\n\n" + FULL_LAFAN_STOP_MESSAGE + "\n")


def build_markdown(
    root: Path,
    core: pd.DataFrame,
    timing: pd.DataFrame,
    seeds: pd.DataFrame,
    interaction: pd.DataFrame,
    projection: pd.DataFrame,
) -> None:
    operating = timing[timing.label.isin(OPERATING_POINTS)].set_index("label")
    best_quality = operating.rf_kpe_all_mean_m.idxmin()
    fastest = operating.end_to_end_rtf_median.idxmin()
    projection_info = json.loads((root / "metrics" / "stage2_projection.json").read_text())
    evaluator = load_evaluator_protocol(root / "manifests" / "evaluator.yaml")
    scales = pd.read_csv(root / "metrics" / "root_scale_diagnostics.csv").set_index("label")
    residuals = pd.read_csv(root / "metrics" / "controlled_task_residuals.csv")
    candidates = pd.read_csv(root / "metrics" / "conditional_candidates.csv")
    adapters = pd.read_csv(root / "metrics" / "source_adapter_errors.csv")
    seed_variance = pd.read_csv(root / "metrics" / "sparse_seed_variance.csv")
    stage1_config = load_yaml(root / "configs" / "stage1.yaml")
    revised_stage1_complete = (
        stage1_config.get("completion_status") == "complete_revised_scope"
    )
    outcome = (
        (
            "GO"
            if projection_info["within_wall_budget"]
            and projection_info["within_storage_budget"]
            else "GO WITH CHANGES"
        )
        if revised_stage1_complete
        else "NO-GO"
    )
    quality_table = operating[
        [
            "rf_kpe_all_mean_m",
            "rf_kpe_targeted_mean_m",
            "rf_kpe_untracked_mean_m",
            "root_translation_common_scale_mean_m",
            "root_yaw_mean_rad",
            "artifact_rate",
        ]
    ].to_markdown(floatfmt=".4f")
    timing_table = operating[
        [
            "end_to_end_rtf_median",
            "native_core_rtf_median",
            "end_to_end_rtf_cv",
        ]
    ].to_markdown(floatfmt=".4f")
    scale_table = scales[
        [
            "common_static_scale",
            "common_local_body_scale",
            "common_root_displacement_scale",
            "native_root_scale",
            "effective_root_xy_scale",
            "root_translation_common_scale_mean_m",
            "root_translation_native_scale_mean_m",
            "root_translation_scale_invariant_mean_m",
        ]
    ].to_markdown(floatfmt=".4f")
    artifact_table = operating[
        [
            "foot_skating_frame_rate",
            "ground_penetration_frame_rate",
            "joint_limit_violation_frame_rate",
            "invalid_frame_rate",
            "artifact_rate",
        ]
    ].to_markdown(floatfmt=".4f")
    temporal_table = operating[
        [
            "joint_velocity_rms_mean_rad_s",
            "joint_acceleration_rms_mean_rad_s2",
            "joint_jerk_rms_p95_rad_s3",
            "pose_jump_p95_m",
        ]
    ].to_markdown(floatfmt=".4f")
    residual_table = residuals.to_markdown(index=False, floatfmt=".5f")
    candidate_table = candidates[
        [
            "candidate",
            "status",
            "input_ready",
            "environment_ready",
            "canonical_output_ready",
            "elapsed_s",
            "outcome",
            "reason",
        ]
    ].to_markdown(index=False, floatfmt=".2f")
    adapter_table = adapters[
        [
            "method",
            "common_joints",
            "root_aligned_mpjpe_m",
            "bone_length_error_mean_m",
            "root_translation_error_mean_m",
            "yaw_error_mean_rad",
            "foot_contact_agreement",
            "status",
        ]
    ].to_markdown(index=False, floatfmt=".6f")
    interaction_table = interaction[
        [
            "case",
            "variant",
            "strict_contact_2cm_frame_rate",
            "near_contact_5cm_frame_rate",
            "proximity_10cm_frame_rate",
            "penetration_any_frame_rate",
            "penetration_frame_rate",
            "foot_sticking_violation_frame_rate",
            "end_to_end_rtf",
        ]
    ].to_markdown(index=False, floatfmt=".4f")
    projection_table = projection[
        [
            "method",
            "measured_pilot_end_to_end_rtf",
            "projected_steady_wall_hours",
            "projected_startup_and_adapter_hours",
            "raw_projected_wall_hours",
            "safety_factor",
            "safe_projected_wall_hours",
        ]
    ].to_markdown(index=False, floatfmt=".3f")

    _write_report(
        root / "PILOT_REPORT.md",
        f"""# Human-to-G1 Retargeting Pilot Report

## Abstract

**Revision status:** the legacy four-method execution below is complete, but
the revised Stage 1 is not. ProtoMotions v2.3/v3, pre-solver policy capture,
the controlled scale-policy transplant, and registered root/local sensitivity
runs remain mandatory. The current decision is `NO-GO — work in progress`.

This Stage 1 Pilot compares controlled Sparse and Dense Mink retargeting, official GMR, and official OmniRetarget/Holosoma on the source-only-selected 600-frame (`19.9998 s`) LAFAN1 window `dance1_subject1_f000000_000600`. The original presentation made several methods look nearly identical because it mixed method-specific root scales with a method-dependent evaluator scale and used root-frame plots that intentionally remove global translation. Evaluator v3 fixes that confound without changing the sequence or thresholds: one registered shared-landmark least-squares scale is frozen for local/body quality, root-displacement gain and root anchor are explicit separate parameters, native scale policy and scale-invariant path shape are reported separately, and controlled baseline v6 uses the canonical Holosoma G1 scene with an explicit weak temporal cost.

## Frozen design and scope

The legacy evidence contains one LAFAN Pilot, three Sparse seeds, four completed operating points, and the two official box/climb interaction cases in Full and No-Hard form. Revised Stage 1 additionally requires ProtoMotions v2.3 and v3 plus the registered preprocessing/scale study. It is not a Full-LAFAN ranking. The official public methods retain their native scaling policies; they are not silently rescaled or retuned. The evaluator uses the Holosoma G1 29-DoF model only as common robot geometry and joint order.

## Main results

{quality_table}

`{best_quality}` has the lowest RF-KPE-all and `{fastest}` has the lowest median end-to-end RTF at this one operating point. Neither observation is a dataset-level ranking. The visual similarity is expected: every output is the same G1 morphology, all methods track overlapping major body landmarks, and RF-KPE removes root position and heading before comparing pose. Differences are most visible in the disaggregated root-scale, task-residual, temporal, artifact, and seed-sensitivity evidence below.

## Scale audit: why root translation looked inconsistent

The common local/body scale is `{float(evaluator['scale']['common_local_body_scale']):.9f}`, obtained by the registered root-relative, heading-aligned shared-semantic-landmark scalar least-squares fit. The separately frozen root-displacement gain is `{float(evaluator['scale']['common_root_displacement_scale']):.9f}`; both start at the same value but remain independent intervention parameters. The legacy `head→mean(toes)` ratio is `{float(evaluator['scale']['diagnostics']['head_to_toe_scale']):.9f}` and is diagnostic only. Controlled v6 additionally applies the rigid translation `{np.asarray(evaluator['scale']['common_root_alignment_translation_m']).round(6).tolist()} m`, defined as neutral-G1 pelvis minus root-scaled source frame-0 pelvis. This anchor changes only world placement; it does not alter scale, root-path deltas, or RF-KPE.

{scale_table}

Three quantities answer three different questions:

- **Common-scale error** asks whether the output follows the morphology-referenced benchmark trajectory.
- **Native-scale error** asks whether the public solver follows the trajectory implied by its own declared policy.
- **Scale-invariant error** fits one scalar to the root XY path and asks only whether path shape is preserved.

GMR's declared root/leg policy is `0.9 × 1.75 / 1.8 = 0.875`; Holosoma's LAFAN default is `1.27 / 1.7 = 0.7470588`. The old evaluator also estimated robot height from each method's first output pose, which made the reference itself method-dependent. These are the reasons identical source and target robot did not produce identical root scales. A large common-scale error together with a small native or scale-invariant error is scale-policy mismatch, not necessarily solver tracking failure.

![Declared, measured, and common root scales](figures/root_scale_policies.svg)

![Root error decomposition](figures/root_error_decomposition.svg)

## RF-KPE: what it measures and why values cluster

RF-KPE is **root-frame keypoint position error**. Human semantic joints are scaled with the one common scale, translated relative to the human root, and rotated into the geometry-derived human heading; G1 semantic points are treated the same way using the robot root. It therefore evaluates relative whole-body pose while deliberately excluding root translation and root yaw. `targeted` covers wrists and ankles; `untracked` covers the remaining semantic body joints. Similar RF-KPE values do not imply identical trajectories: common robot morphology imposes a shared error floor, and the metric cannot expose the scale differences that were removed by root alignment. Root translation/yaw, declared task residuals, and artifacts must be read beside it.

## Sparse versus Dense design

For the neutral-seed operating-point comparison, Sparse and Dense v3 share the same G1 model, common uniform scale, rigid frame-0 root anchor, DAQP solver, damping, joint limits, iteration budget, first-frame convergence budget, posture cost, weak `q[t-1]` temporal cost, root weights, and sequential warm start. Only the declared task set differs. Sparse tracks root translation/yaw plus left/right wrists and ankles. Dense adds torso, head, shoulders, elbows, hips, knees, and toes. No hand/foot orientation, contact prior, learned prior, or independent-frame variant enters the main comparison.

{residual_table}

These are the actual world-position residuals minimized by the controlled solver. They are reported separately from RF-KPE, which is an evaluator-side morphology metric. Dense has more mutually competing position targets on a robot with different segment proportions, so lower full-body RF-KPE can coexist with higher declared-task residual and root-path error; that trade-off is part of the result, not a scale inconsistency.

## Temporal and artifact evidence

{temporal_table}

{artifact_table}

`artifact` is an aggregate per-frame flag, not a method label or a statement that the whole motion is invalid. A frame is flagged only for one or more named causes: source-stance foot skating over `0.01 m/s`, ground penetration over `0.01 m`, joint-limit violation, or invalid numeric output. The rebuilt Rerun overlay shows these exact causes (`SKATING-L/R`, `PENETRATION`, `JOINT-LIMIT`, `INVALID`) instead of the ambiguous word “Artifact.” Component rates can overlap and are not summed into a score.

![Artifact causes](figures/artifact_components.svg)

## Sparse null-space sensitivity

{seeds.to_markdown(index=False, floatfmt='.5f')}

Mean targeted point variance is `{float(seed_variance.targeted_point_variance_mean_m2.iloc[0]):.6g} m²`; mean untracked point variance is `{float(seed_variance.untracked_point_variance_mean_m2.iloc[0]):.6g} m²`. Three deterministic first-frame seeds do not sample the entire null space, but they directly test whether similar sparse task satisfaction hides different full-body solutions.

## Native source adapter audit

{adapter_table}

Adapter errors are not attributed to the retargeter. The GMR row is produced by its official LAFAN loader. The Holosoma row checks the explicit right-first reorder and exact Z-up/Y-up involution. Native adapter files remain ignored licensed/generated artifacts; hashes are frozen in `manifests/source_adapters.yaml`.

## Synchronized visual inspection

The Rerun recording synchronizes the source human with articulated G1 meshes for all core operating points and Sparse seeds A/B. World, overlay, root-frame, and seed views deliberately answer different questions. The viewer replays canonical outputs and is excluded from timing; see `docs/RERUN_VISUALIZATION.md` and `manifests/rerun_visualization.json`.

## Interaction case study

{interaction_table}

Distances are computed with MuJoCo geometry-surface queries over the actual collision meshes. `penetration_any_frame_rate` records every negative signed distance; the primary `penetration_frame_rate` records depth over 1.1 mm (the frozen 1 mm constraint tolerance plus 0.1 mm numerical margin). The evidence covers exactly one box sequence and one climbing sequence and must not be generalized to a dataset.

## Timing protocol

{timing_table}

Each core run uses one fresh cold process, one warm-up, and three measured warm repetitions with one CPU thread and no visualization. End-to-end and native-core values remain separate; initialization/JIT/import costs are excluded from steady-state RTF and retained in the raw records.

## Legacy conditional candidate gates (superseded)

{candidate_table}

These rows record the former gate and are retained for provenance. The revised design promotes ProtoMotions v3 to a required full run and replaces PHC's experimental slot with ProtoMotions v2.3/Mink. PHC's official fitting asset is excluded from the canonical plot because it is 37-motor, not G1-29. An `N/A` row is not a negative quality result and cannot satisfy the revised required run.

## Stage 2 gate

{projection_table}

After the required 1.5× safety factor, the serial projection is {_fmt(projection_info['safe_projected_wall_hours_serial'], 2)} h and {_fmt(projection_info['safe_projected_storage_gb'], 2)} GB. It {'fits' if projection_info['within_wall_budget'] else 'does not fit'} the 48-hour runtime gate and {'fits' if projection_info['within_storage_budget'] else 'does not fit'} the 200 GB storage gate. Stage 2 remains stopped pending a separate design decision and explicit approval.
""",
    )
    _write_report(
        root / "EXECUTIVE_SUMMARY.md",
        f"""# Executive Summary

The legacy four-core execution completes four operating points, three Sparse seeds, the earlier evaluator scale correction, controlled root anchoring, synchronized articulated-G1 Rerun evidence, and both Full/No-Hard interaction ablations. Revised Stage 1 remains incomplete pending corrected evaluator-v3/controlled-v6 reruns, ProtoMotions v2.3/v3, pre-solver policy capture, controlled scale-policy transplantation, and registered root/local sensitivity runs. The fastest observed legacy operating point is `{fastest}` and the lowest RF-KPE-all is `{best_quality}` on this one Pilot only.

The apparent lack of visual separation was primarily a measurement-presentation issue: all outputs share G1 morphology, root-frame pose plots remove global trajectory, and the old root reference used inconsistent scales. The corrected report separates common-scale fidelity, native solver tracking, and scale-invariant path shape, and decomposes artifacts by cause.

Decision: **{outcome} — revised Stage 1 work in progress**. This completeness decision supersedes the legacy budget-only decision. The old four-core 1.5× serial Stage 2 runtime projection is {_fmt(projection_info['safe_projected_wall_hours_serial'], 2)} hours and must be recomputed after the revised method set is complete. Stage 2 was not started.
""",
    )
    _write_report(
        root / "GO_NO_GO.md",
        f"""# Stage 1 Decision: {outcome} — Revised Scope Incomplete

The legacy Sparse, Dense, GMR, and OmniRetarget runs complete 600/600 frames, and the interaction evidence is retained. They do not satisfy revised Stage 1 by themselves. ProtoMotions v2.3 and v3 must become full canonical operating points; PHC remains lineage-only because its official fitting asset is not canonical G1-29. Native preprocessing, controlled policy transplantation, and root/local ±5% sensitivity are mandatory.

The old projection is not the revised Stage 2 estimate. Recompute runtime and storage only after the expanded Stage 1 finishes; no Full-LAFAN execution is authorized.
""",
    )
    _write_report(
        root / "METHOD_SCOPE.md",
        """# Method Scope

The revised required set is controlled Sparse Mink, controlled Dense Mink, official GMR at `bb1bbe40774794fceb2a7c579a3464a28e68c844`, official OmniRetarget/Holosoma at `5f48635a3624656a5f46a07df26d43187e59f855`, ProtoMotions v2.3/Mink at `4a905b998101333a2fb91f2de8e2cab4bd0db68e`, and ProtoMotions v3/modified-PyRoki at `49fe5ad69de67ebbc07ea2b25d41b0f622c15c3c`. Sparse and Dense v3 share robot, scale, anchor, solver, limits, warm start, iteration budget, damping, posture/temporal costs, and neutral initialization; only their task sets differ.

ProtoMotions v2.3 is labelled `PHC-derived preprocessing/FK infrastructure + sequential Mink`; it is not a PHC result. ProtoMotions v2/v3 are a native pipeline lineage pair, not a pure backend ablation. PHC is lineage/AMASS-policy evidence because its official fitting asset has 37 motors rather than canonical G1-29. Native official results and controlled preprocessing/scale ablations must remain separate. SOMA/cuRobo remain conditional input-compatibility candidates. Mink/PyRoki are backends; controllers and benchmarks remain outside the retargeter scatter.
""",
    )
    _write_report(
        root / "SPARSE_IK_ANALYSIS.md",
        f"""# Sparse IK Analysis

Sparse tracks root translation/yaw plus both wrists and ankles. Dense adds torso, head, shoulders, elbows, hips, knees, and toes. Both use local/body scale `{float(evaluator['scale']['common_local_body_scale']):.9f}`, separately declared root-displacement gain `{float(evaluator['scale']['common_root_displacement_scale']):.9f}`, the same neutral-pelvis frame-0 anchor, canonical Holosoma G1 joint order/limits, sequential warm start, weak fixed-posture regularization, and the same explicit weak `q[t-1]` temporal cost. Sparse contains no torso/elbow/knee task, contact objective, or learned prior.

{seeds.to_markdown(index=False, floatfmt='.5f')}

{residual_table}

The pairwise table quantifies hidden-state sensitivity in joint space and root-frame robot point space. Main comparisons use neutral; A/B are diagnostics and are never averaged into the operating point. Target residual and untracked-pose divergence are separate claims.
""",
    )
    _write_report(
        root / "INTERACTION_CASE_STUDY.md",
        f"""# Interaction Case Study

{interaction_table}

Full enables object non-penetration, foot sticking, and joint limits. No-Hard disables only the first two flags; initialization, input, object sampling seed, solver, iteration budget, and joint limits remain unchanged. A regression test checks that the patched non-penetration gate and the upstream foot-sticking gate both alter constraint construction.

The 2 cm, 5 cm, and 10 cm thresholds use signed geometry-surface distances from `mujoco.mj_geomDistance`, never distance to the object origin. `penetration_any_frame_rate` records every negative distance, while the primary `penetration_frame_rate` applies the frozen 1.1 mm tolerance-aware threshold. Results are two-case case-study evidence only.
""",
    )
    _write_report(
        root / "REPRODUCE_PILOT.md",
        """# Reproduce the Pilot

1. Follow `docs/UPSTREAM_SETUP.md`, verify frozen commits, and place licensed assets outside Git as recorded in `manifests/`.
2. Run `rtcmp audit`, `rtcmp prepare-source`, `rtcmp validate-models`, and `rtcmp freeze-evaluator` before any formal method.
3. Freeze the preprocessing-policy manifest and pre-solver target contract from `research/OFFICIAL_SCALE_AND_PREPROCESSING_AUDIT.md` and `configs/scale_policy_sensitivity.yaml`.
4. Run the legacy Sparse/Dense/GMR/OmniRetarget commands, then the required ProtoMotions v2.3 and v3 (`target_raw_frames=600`) adapters and full Pilot runs.
5. Capture every native pre-solver target; run the controlled policy transplant, registered root/local ±5% variants, contact-label diagnostics, and the hash-bound neutral-SMPL-X actor-shape policy formula probe (not runtime-constructor or ranking evidence).
6. Run both variants of `rtcmp run-interaction` for box and climb.
7. In conda env `vis`, rebuild all articulated-G1 views; then run `rtcmp build-report` and `rtcmp validate-stage1`. Validation must remain `NO-GO` while any revised requirement is absent.

Every run has an atomic status manifest and immutable output hash. Existing successful output is reused; a failed retry receives a new attempt directory. Raw datasets, body models, upstream history, trajectories, logs, and caches remain ignored.
""",
    )
    slides = f"""# Human-to-G1 Retargeting Pilot

## 1. Decision

**{outcome} — revised Stage 1 work in progress.** The legacy four-core Pilot is complete, but ProtoMotions v2.3/v3 and the registered preprocessing/scale study are still mandatory.

## 2. Question

How do information density, public retargeter design, and official preprocessing/scale policy trade fidelity, artifacts, and speed on one frozen human-motion Pilot?

## 3. Frozen source

One source-only selected LAFAN1 window: 600 frames, 19.9998 seconds, selected before any retargeter ran.

## 4. Controlled baselines

Sparse and Dense v3 share robot, scale, rigid root anchor, solver, limits, initialization, posture and temporal costs. Only the task set changes; Sparse additionally exposes three deterministic diagnostic seeds.

## 5. Scale was a confound

![Root scale policies](figures/root_scale_policies.svg)

GMR uses `0.875`; Holosoma uses `0.7471`; evaluator v3 freezes one shared-landmark LS local/body scale plus a separately declared root gain for every method and reports native tracking separately.

## 6. Public methods

GMR and OmniRetarget have completed legacy runs. ProtoMotions v2.3/Mink and v3/modified-PyRoki are required by the revision and are not yet complete; PHC remains lineage-only because its fitting asset is not canonical G1-29.

## 7. RF-KPE and quality vs speed

![RTF versus RF-KPE](figures/rtf_vs_rf_kpe_all.svg)

RF-KPE is root-frame semantic pose error; it intentionally excludes root translation and heading.

## 8. Targeted vs untracked

![Targeted versus untracked](figures/targeted_vs_untracked.svg)

## 9. Artifacts are cause-specific

![Artifact components](figures/artifact_components.svg)

## 10. Sparse sensitivity

![Sparse seed divergence](figures/sparse_seed_divergence.svg)

## 11. Interaction ablation

![Full versus No-Hard](figures/interaction_full_vs_no_hard.svg)

## 12. Evidence limits

This is one LAFAN operating point and two interaction cases. No dataset-level ranking or controller claim is made.

## 13. Synchronized G1 evidence

The Rerun recording provides full articulated G1 meshes in side-by-side world, world overlay, root-frame pose, and Sparse seed views, together with foot-state and per-frame metrics for every canonical output.

## 14. Stage 2 budget

The old four-core 1.5× serial projection is {_fmt(projection_info['safe_projected_wall_hours_serial'], 2)} h and is not the revised estimate. Recompute it after all revised Stage 1 runs; Stage 2 remains stopped.
"""
    _write_report(root / "PRESENTATION.md", slides)


def publish_manifests(root: Path) -> None:
    destination = root / "manifests" / "runs"
    destination.mkdir(parents=True, exist_ok=True)
    sources = [
        *(root / "runs" / "manifests").glob("*.json"),
        *(root / "runs" / "interaction_manifests").glob("*.json"),
    ]
    for source in sources:
        if "smoke" in source.name:
            continue
        value = json.loads(source.read_text())
        environment = str(value.get("environment", "")).removeprefix("conda:")
        if value.get("command") and environment:
            value["command"][0] = f"${{RTCMP_{environment.upper()}_PYTHON}}"

        def relative(item):
            if isinstance(item, str):
                return item.replace(str(root), ".")
            if isinstance(item, list):
                return [relative(part) for part in item]
            return item

        atomic_write_json(destination / source.name, {key: relative(value) for key, value in value.items()})


def artifact_manifest(root: Path) -> None:
    candidates = []
    for folder in ("metrics", "figures", "manifests", "docs", "research", "configs"):
        candidates.extend(path for path in (root / folder).rglob("*") if path.is_file())
    candidates.extend(root / report for report in REPORTS)
    interactive_report = root / "INTERACTIVE_REPORT.html"
    if interactive_report.is_file():
        candidates.append(interactive_report)
        delivery_manifest = interactive_report.with_suffix(
            interactive_report.suffix + ".manifest.json"
        )
        if delivery_manifest.is_file():
            candidates.append(delivery_manifest)
    output = root / "manifests" / "artifacts.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("path", "size_bytes", "sha256"),
            lineterminator="\n",
        )
        writer.writeheader()
        for path in sorted(set(candidates)):
            if path == output:
                continue
            writer.writerow(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )


def build_report(repo_root: str | Path = ".") -> None:
    """Build the expanded six-method Stage 1 publication.

    Legacy helper functions remain importable for audit compatibility, but the
    public CLI must never silently rebuild the obsolete four-method report.
    """

    root = Path(repo_root).resolve()
    from .stage1_publication import (
        build_stage1_publication,
        finalize_stage1_publication,
    )

    build_stage1_publication(root)
    from .interactive_report import build_interactive_report
    from .validation import validate_stage1

    # Phase 1 is evidence-only and cannot issue GO.  Render a PENDING browser
    # artifact so the independent validator can check report structure without
    # consuming its own conclusion.  Phase 2 binds one fail-closed verdict to
    # the immutable PENDING evidence ledger; final Markdown/HTML only consume
    # that bound verdict.  There is deliberately no validate/render fixed point.
    build_interactive_report(root, force_pending=True)
    validate_stage1(root)
    finalize_stage1_publication(root)
    build_interactive_report(root)
    artifact_manifest(root)
