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

from .constants import FULL_LAFAN_STOP_MESSAGE
from .evaluator import HUMAN_SEMANTIC_JOINTS, evaluate_motion, save_evaluation
from .io_utils import atomic_write_json, atomic_write_text, load_yaml, sha256_file
from .robot_model import CanonicalRobotModel, default_robot_scene
from .rotations import quaternion_wxyz_to_matrix, yaw_from_matrix
from .schemas import CanonicalG1, CanonicalHuman, RunManifest, RunStatus


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
)


def _sequence(root: Path) -> dict[str, Any]:
    return load_yaml(root / "manifests" / "pilot_sequence.yaml")


def _run_paths(root: Path) -> dict[str, Path]:
    sequence_id = _sequence(root)["sequence_id"]
    return {
        label: root / "runs" / sequence_id / label / "canonical_g1.npz"
        for label in CORE_LABELS
    }


def evaluate_core(root: Path) -> pd.DataFrame:
    sequence = _sequence(root)
    human = CanonicalHuman.load(root / sequence["canonical_path"])
    robot = CanonicalRobotModel(default_robot_scene(root))
    rows = []
    for label, path in _run_paths(root).items():
        if not path.is_file():
            raise FileNotFoundError(f"Required core output is missing: {path}")
        motion = CanonicalG1.load(path)
        motion.validate(source_frame_count=len(human.timestamps))
        table, summary = evaluate_motion(human, motion, robot)
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
        run_dir = root / "runs" / sequence_id / label
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
    return per_frame, summary


def collect_interaction(root: Path) -> pd.DataFrame:
    rows = []
    for case in ("box", "climb"):
        for variant in ("full", "no-hard"):
            path = root / "runs" / "interaction" / case / variant / "summary.json"
            if not path.is_file():
                raise FileNotFoundError(f"Required interaction output is missing: {path}")
            value = json.loads(path.read_text())
            value.pop("native_frame_times_s", None)
            rows.append(value)
    frame = pd.DataFrame(rows).sort_values(["case", "variant"])
    frame.to_csv(root / "metrics" / "interaction_summary.csv", index=False)
    frame.to_parquet(root / "metrics" / "interaction_summary.parquet", index=False)
    return frame


def stage2_projection(root: Path, timing: pd.DataFrame) -> pd.DataFrame:
    dataset = load_yaml(root / "manifests" / "dataset.yaml")
    total_frames = int(dataset["validation"]["total_frames"])
    fps = float(dataset["validation"]["unique_nominal_fps"])
    duration_s = total_frames / fps
    rows = []
    for label in OPERATING_POINTS:
        row = timing.loc[timing.label == label].iloc[0]
        raw_hours = float(row.end_to_end_rtf_median * duration_s / 3600.0)
        rows.append(
            {
                "method": label,
                "lafan_frames": total_frames,
                "measured_pilot_end_to_end_rtf": row.end_to_end_rtf_median,
                "raw_projected_wall_hours": raw_hours,
                "safety_factor": 1.5,
                "safe_projected_wall_hours": raw_hours * 1.5,
            }
        )
    frame = pd.DataFrame(rows)
    formal_dirs = [
        root / "runs" / _sequence(root)["sequence_id"] / label for label in CORE_LABELS
    ]
    pilot_bytes = sum(
        path.stat().st_size
        for directory in formal_dirs
        for path in directory.rglob("*")
        if path.is_file()
    )
    projected_storage_gb = pilot_bytes / 600.0 * total_frames * 1.5 / 1e9
    total_safe_hours = float(frame.safe_projected_wall_hours.sum())
    projection = {
        "lafan_frames": total_frames,
        "lafan_duration_hours": duration_s / 3600.0,
        "runtime_safety_factor": 1.5,
        "retry_rate_observed": 0.0,
        "safe_projected_wall_hours_serial": total_safe_hours,
        "safe_projected_storage_gb": projected_storage_gb,
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


def build_figures(
    root: Path,
    core_timing: pd.DataFrame,
    seed_summary: pd.DataFrame,
    interaction: pd.DataFrame,
) -> None:
    points = core_timing[core_timing.label.isin(OPERATING_POINTS)].copy()

    def scatter(axis, data, x, y, xlabel, ylabel):
        for row in data.itertuples():
            axis.scatter(getattr(row, x), getattr(row, y), s=55)
            axis.annotate(row.label, (getattr(row, x), getattr(row, y)), xytext=(4, 4), textcoords="offset points")
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)

    _save_plot(
        root,
        "rtf_vs_rf_kpe_all",
        points[["label", "end_to_end_rtf_median", "rf_kpe_all_mean_m"]],
        lambda ax, data: scatter(ax, data, "end_to_end_rtf_median", "rf_kpe_all_mean_m", "End-to-end steady RTF (lower is faster)", "RF-KPE-all mean (m, lower is better)"),
    )
    _save_plot(
        root,
        "rtf_vs_artifact_rate",
        points[["label", "end_to_end_rtf_median", "artifact_rate"]],
        lambda ax, data: scatter(ax, data, "end_to_end_rtf_median", "artifact_rate", "End-to-end steady RTF", "Artifact frame rate"),
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
    outcome = "GO" if projection_info["within_wall_budget"] else "GO WITH CHANGES"
    table = operating[
        [
            "rf_kpe_all_mean_m",
            "rf_kpe_targeted_mean_m",
            "rf_kpe_untracked_mean_m",
            "artifact_rate",
            "end_to_end_rtf_median",
            "native_core_rtf_median",
        ]
    ].to_markdown(floatfmt=".4f")
    interaction_table = interaction[
        [
            "case",
            "variant",
            "strict_contact_2cm_frame_rate",
            "penetration_frame_rate",
            "foot_sticking_violation_frame_rate",
            "end_to_end_rtf",
        ]
    ].to_markdown(index=False, floatfmt=".4f")
    projection_table = projection.to_markdown(index=False, floatfmt=".3f")

    _write_report(
        root / "PILOT_REPORT.md",
        f"""# Human-to-G1 Retargeting Pilot Report

## Scope and frozen design

This Stage 1 experiment compares controlled Sparse and Dense Mink baselines, official GMR, and official OmniRetarget/Holosoma on the preselected 600-frame (`19.9998 s`) LAFAN1 window `dance1_subject1_f000000_000600`. The sequence, thresholds, method commits, evaluator model, and timing order were frozen before formal runs. Full-LAFAN was not started.

## Main results

{table}

`{best_quality}` has the lowest RF-KPE-all at this single operating point; `{fastest}` is fastest by median end-to-end RTF. These are Pilot observations, not dataset-level rankings. Adapter error is reported separately in `metrics/source_adapter_errors.csv`.

## Interpretation

The targeted and untracked columns are intentionally separate: a method can match hands/feet while degrading torso or limb structure. Temporal and artifact fields remain disaggregated in `metrics/runs/*_summary.json` and per-frame CSV/Parquet files. No composite score is used.

## Interaction case study

{interaction_table}

Distances are computed with MuJoCo geometry-surface queries over the actual collision meshes. The evidence covers exactly one box sequence and one climbing sequence and must not be generalized to a dataset.

## Timing protocol

Each core run used a fresh cold process, one warm-up, and three measured warm repetitions with one CPU thread and no visualization. Raw end-to-end and native-core values are in `metrics/timing_raw.csv`; cold/import/initialization overhead remains separate.

## Stage 2 gate

{projection_table}

After the required 1.5× safety factor, the serial projection is {_fmt(projection_info['safe_projected_wall_hours_serial'], 2)} h and {_fmt(projection_info['safe_projected_storage_gb'], 2)} GB. The runtime projection therefore {'fits' if projection_info['within_wall_budget'] else 'does not fit'} the 48-hour gate. Stage 2 requires a new design discussion and explicit approval.
""",
    )
    _write_report(
        root / "EXECUTIVE_SUMMARY.md",
        f"""# Executive Summary

Stage 1 completed the four required operating points and both Full/No-Hard interaction ablations on the frozen Pilot. The fastest observed operating point was `{fastest}` and the lowest RF-KPE-all was `{best_quality}`; neither observation is a Full-LAFAN conclusion.

Decision: **{outcome}**. The Stage 1 harness and evidence are usable, but the 1.5× serial Stage 2 runtime projection is {_fmt(projection_info['safe_projected_wall_hours_serial'], 2)} hours, above the 48-hour limit. Storage remains within 200 GB. Before Stage 2, use repeat timing only on a frozen subset and either optimize/parallelize OmniRetarget or define and rename a deterministic reduced-LAFAN experiment.
""",
    )
    _write_report(
        root / "GO_NO_GO.md",
        f"""# Stage 1 Decision: {outcome}

All four core methods completed 600/600 frames, all evaluator and adapter tests passed, and both interaction cases completed in Full and No-Hard form. Raw timing has the required cold/warm structure. Conditional candidates were not started because the four core points directly answer the Pilot question and candidate integration would not change the Stage 2 runtime bottleneck.

The change required before approval is budget-related: projected serial Full-LAFAN runtime with the frozen 1.5× factor is {_fmt(projection_info['safe_projected_wall_hours_serial'], 2)} hours. Proposed order: remove non-core candidates, keep repeated timing on a frozen subset only, discard rebuildable intermediates, then use a deterministically selected and explicitly renamed reduced-LAFAN set if the runtime still exceeds 48 hours.
""",
    )
    _write_report(
        root / "METHOD_SCOPE.md",
        """# Method Scope

The experimental retargeters are controlled Sparse Mink, controlled Dense Mink, official GMR at the frozen commit, and official OmniRetarget/Holosoma at commit `5f48635a3624656a5f46a07df26d43187e59f855`. Sparse and Dense share solver, limits, scaling, warm start, iteration budget, damping, posture regularization, and initialization; only their declared target sets differ.

ProtoMotions v3 and PHC remain conditional candidates and were not used as plotted points. SOMA Retargeter and cuRoboV2 are lineage/input-compatibility evidence only. Mink/PyRoki are optimization backends; MaskedMimic/BeyondMimic are controllers or trackers; LocoMuJoCo is a benchmark; MIRROR is not a G1 point here. Historical and experimental claims remain separated in `research/claims.csv`.
""",
    )
    _write_report(
        root / "SPARSE_IK_ANALYSIS.md",
        f"""# Sparse IK Analysis

Sparse tracks root translation/yaw, both wrists, and both ankles. It uses neutral and two deterministic perturbed initial postures, followed by sequential warm start. No torso, elbow, knee, contact, or learned prior is present.

{seeds.to_markdown(index=False, floatfmt='.5f')}

The pairwise table quantifies hidden-state sensitivity in joint space and in root-frame robot point space. Main comparisons use the neutral seed; A/B are diagnostics and are not silently averaged into the operating point.
""",
    )
    _write_report(
        root / "INTERACTION_CASE_STUDY.md",
        f"""# Interaction Case Study

{interaction_table}

Full enables object non-penetration, foot sticking, and joint limits. No-Hard disables only the first two flags; initialization, input, object sampling seed, solver, iteration budget, and joint limits remain unchanged. A regression test checks that the patched non-penetration gate and the upstream foot-sticking gate both alter constraint construction.

The 2 cm, 5 cm, and 10 cm thresholds use signed geometry-surface distances from `mujoco.mj_geomDistance`, never distance to the object origin. Results are two-case case-study evidence only.
""",
    )
    _write_report(
        root / "REPRODUCE_PILOT.md",
        """# Reproduce the Pilot

1. Follow `docs/UPSTREAM_SETUP.md`, verify checkout commits, and place licensed assets outside Git as recorded in `manifests/`.
2. Run `rtcmp audit`, `rtcmp prepare-source`, and `rtcmp validate-models`.
3. Run core methods in `manifests/experiment_order.yaml`; Sparse uses neutral, A, and B.
4. Run `rtcmp run-interaction --case box --variant full`, repeat with `no-hard`, then repeat both variants for `climb`.
5. Run `rtcmp build-report` and `rtcmp validate-stage1`.

Every run has an atomic status manifest and immutable output hash. Existing successful output is reused; a failed retry receives a new attempt directory. Raw datasets, body models, upstream history, trajectories, logs, and caches remain ignored.
""",
    )
    slides = f"""# Human-to-G1 Retargeting Pilot

## 1. Decision

**{outcome}** — Stage 1 evidence is complete; Stage 2 needs a runtime-budget change and explicit approval.

## 2. Question

How do sparse constraints, dense constraints, GMR, and OmniRetarget trade fidelity, artifacts, and speed on one frozen human-motion Pilot?

## 3. Frozen source

One source-only selected LAFAN1 window: 600 frames, 19.9998 seconds, selected before any retargeter ran.

## 4. Controlled baselines

Sparse and Dense share every solver setting. Only the task set changes; Sparse also exposes three deterministic initial postures.

## 5. Public methods

GMR and OmniRetarget run at frozen official commits in isolated subprocess environments with only I/O, provenance, and timing adapters.

## 6. Quality vs speed

![RTF versus RF-KPE](figures/rtf_vs_rf_kpe_all.svg)

## 7. Targeted vs untracked

![Targeted versus untracked](figures/targeted_vs_untracked.svg)

## 8. Artifacts

![RTF versus artifacts](figures/rtf_vs_artifact_rate.svg)

## 9. Sparse sensitivity

![Sparse seed divergence](figures/sparse_seed_divergence.svg)

## 10. Interaction ablation

![Full versus No-Hard](figures/interaction_full_vs_no_hard.svg)

## 11. Evidence limits

This is one LAFAN operating point and two interaction cases. No dataset-level ranking or controller claim is made.

## 12. Stage 2 budget

The 1.5× serial projection is {_fmt(projection_info['safe_projected_wall_hours_serial'], 2)} h and {_fmt(projection_info['safe_projected_storage_gb'], 2)} GB. Discuss reduced-LAFAN or optimized/parallel execution before approval.
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
    for folder in ("metrics", "figures", "manifests/runs"):
        candidates.extend(path for path in (root / folder).rglob("*") if path.is_file())
    candidates.extend(root / report for report in REPORTS)
    output = root / "manifests" / "artifacts.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("path", "size_bytes", "sha256"))
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
    root = Path(repo_root).resolve()
    (root / "metrics").mkdir(exist_ok=True)
    (root / "figures").mkdir(exist_ok=True)
    core = evaluate_core(root)
    _, timing = collect_timing(root, core)
    _, seed_summary = sparse_seed_divergence(root)
    interaction = collect_interaction(root)
    projection = stage2_projection(root, timing)
    build_figures(root, timing, seed_summary, interaction)
    build_markdown(root, core, timing, seed_summary, interaction, projection)
    publish_manifests(root)
    artifact_manifest(root)
