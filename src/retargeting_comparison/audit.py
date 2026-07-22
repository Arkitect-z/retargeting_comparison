"""Stage 0 method-scope and provenance audit generation."""

from __future__ import annotations

import csv
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .constants import FULL_LAFAN_STOP_MESSAGE
from .io_utils import atomic_write_text, atomic_write_yaml, sha256_file

METHOD_FIELDS = (
    "name", "version_or_commit", "publication_year", "open_source_status",
    "source_representation", "target_robot", "g1_29dof_supported",
    "outputs_g1_reference_trajectory", "retargeting_family", "optimizer",
    "kinematics_backend", "collision_backend", "temporal_scope",
    "contact_handling", "object_or_terrain_support", "requires_training",
    "requires_rl_training", "requires_pre_retargeted_input", "experiment_status",
    "exclusion_reason", "official_source", "category", "stage1_role",
)


def _method(
    name: str,
    version: str,
    year: str,
    source: str,
    target: str,
    g1: str,
    output: str,
    family: str,
    optimizer: str,
    kinematics: str,
    collision: str,
    temporal: str,
    contact: str,
    interaction: str,
    training: str,
    rl: str,
    pretargeted: str,
    status: str,
    exclusion: str,
    url: str,
    category: str,
    role: str,
    open_source: str = "yes",
) -> dict[str, str]:
    return dict(zip(METHOD_FIELDS, (
        name, version, year, open_source, source, target, g1, output, family,
        optimizer, kinematics, collision, temporal, contact, interaction,
        training, rl, pretargeted, status, exclusion, url, category, role,
    ), strict=True))


METHODS: tuple[dict[str, str], ...] = (
    _method("Controlled Sparse-IK", "benchmark v2", "2026", "canonical LAFAN BVH root + 4 end-effectors", "Unitree G1 29-DoF", "yes", "yes", "controlled sparse-task retargeting", "Mink differential IK / DAQP", "MuJoCo + Mink", "joint limits only", "sequential framewise + weak q[t-1] cost", "none", "none", "no", "no", "no", "required; completed", "", "https://github.com/kevinzakka/mink", "controlled_baseline", "required"),
    _method("Controlled Dense-KeyBody IK", "benchmark v2", "2026", "canonical LAFAN BVH dense key bodies", "Unitree G1 29-DoF", "yes", "yes", "controlled dense-task retargeting", "Mink differential IK / DAQP", "MuJoCo + Mink", "joint limits only", "sequential framewise + weak q[t-1] cost", "none", "none", "no", "no", "no", "required; completed", "", "https://github.com/kevinzakka/mink", "controlled_baseline", "required"),
    _method("GMR", "bb1bbe40774794fceb2a7c579a3464a28e68c844", "2025", "LAFAN BVH", "Unitree G1 29-DoF", "yes", "yes", "body-level non-uniform retargeting", "two-stage Mink differential IK / DAQP", "MuJoCo + Mink", "MuJoCo geometry", "sequential framewise", "no explicit source contact objective", "no", "no", "no", "no", "required; completed", "", "https://github.com/YanjieZe/GMR", "retargeter", "required"),
    _method("OmniRetarget / Holosoma", "5f48635a3624656a5f46a07df26d43187e59f855", "2025", "LAFAN positions or SMPL-family interaction motion", "Unitree G1 29-DoF", "yes", "yes", "interaction-mesh constrained retargeting", "Sequential SOCP", "Holosoma kinematics", "MuJoCo surface distance in evaluator", "sequential trajectory", "foot sticking", "object and terrain interaction", "no", "no", "no", "required; completed", "", "https://github.com/amazon-far/holosoma", "retargeter", "required"),
    _method("ProtoMotions v3", "49fe5ad69de67ebbc07ea2b25d41b0f622c15c3c", "2026", "SMPL MotionLib keypoints", "Unitree G1 29-DoF", "yes", "yes", "trajectory-level PyRoki retargeting", "JAX least squares", "PyRoki", "self-collision term present but disabled in frozen G1 script", "whole trajectory; configurable fixed buffer", "foot contact and foot tilt", "no object in retargeting script", "no", "no", "no", "conditional gate", "reported by bounded gate when not integration-ready", "https://github.com/NVlabs/ProtoMotions", "retargeter_and_framework", "conditional_1"),
    _method("PHC retargeter", "846988d433ce1f341e85ac6fbd2cd51911bb3341", "2023", "AMASS/SMPL pose parameters", "Unitree G1 (repository fitting config)", "yes", "yes", "SMPL shape + robot motion fitting", "PyTorch Adam/Adadelta fitting", "SMPLSim / MuJoCo model", "not an explicit fitting objective", "whole sequence parameter tensor with smoothing postprocess", "not in documented fitting loss", "no", "no for fitting; yes for PHC controller", "no for fitting", "no", "conditional gate", "reported by bounded gate when LAFAN-to-SMPL adapter/environment is not ready", "https://github.com/ZhengyiLuo/PHC", "retargeter_component", "conditional_2"),
    _method("ProtoMotions v2", "historical ref not frozen", "2024", "SMPL-family motion", "humanoids including historical G1 path", "requires historical verification", "historical pipeline", "Mink-based data retargeting", "Mink differential IK", "Mink", "backend-dependent", "sequential framewise", "configuration-dependent", "no", "no", "no", "no", "lineage only", "no stable historical ref frozen within Pilot", "https://github.com/NVlabs/ProtoMotions", "historical_retargeter", "lineage_only"),
    _method("SOMA Retargeter", "accessed 2026-07-22", "2025", "SOMA BVH/keypoints", "Unitree G1", "yes", "yes", "GPU IK", "Warp/Newton optimization", "Warp/Newton", "robot collision model", "trajectory/batched", "method-specific", "limited", "no", "no", "no", "lineage only", "no verified lossless LAFAN-to-SOMA adapter", "https://github.com/NVIDIA/soma-retargeter", "retargeter", "lineage_input_incompatible"),
    _method("cuRoboV2 MotionRetargeter", "accessed 2026-07-22", "2025", "SOMA/keypoints", "Unitree G1", "documented", "yes", "GPU kinematic optimization", "cuRobo optimization", "cuRobo", "cuRobo collision", "batched/online", "method-specific", "limited", "no", "no", "no", "lineage only", "no verified lossless LAFAN-to-SOMA adapter", "https://nvlabs.github.io/curobo/latest/getting-started/humanoid_retargeting.html", "retargeter", "lineage_input_incompatible"),
    _method("PhySINK / PHUMA", "paper/project evidence", "2025", "human motion", "Unitree G1", "paper claims", "paper-level", "physics-constrained retargeting", "physics-constrained optimization", "project-specific", "physics-aware", "trajectory", "physics contact", "physics/terrain", "method dependent", "no large RL run in Pilot", "no", "literature only", "no frozen public end-to-end Pilot pipeline verified", "https://davian-robotics.github.io/PHUMA/", "physics_aware_retargeting", "literature_only"),
    _method("MaskedMimic", "official project", "2024", "partial intent/reference", "simulated humanoids", "not an independent G1 retargeter", "controller output", "motion inpainting controller", "learned policy", "simulator", "physics contacts", "temporal learned controller", "physics", "objects/keyframes/text conditions", "yes", "yes", "yes", "pipeline context", "consumes conditions as a trained controller; not offline human-to-G1 retargeting", "https://research.nvidia.com/labs/par/project/maskedmimic.html", "controller_tracker", "pipeline_context_only"),
    _method("BeyondMimic", "official repository", "2025", "pre-retargeted robot reference", "Unitree G1", "tracker supports G1", "no new human-to-G1 reference", "whole-body tracking", "learned control policy", "Isaac Lab", "physics contacts", "temporal controller", "physics", "terrain/control", "yes", "yes", "yes", "pipeline context", "requires an existing generalized-coordinate robot reference", "https://github.com/HybridRobotics/whole_body_tracking", "controller_tracker", "pipeline_context_only"),
    _method("LocoMuJoCo", "official continuous project", "2023", "preprocessed motion datasets", "benchmark robots", "not a standalone public human-to-G1 method", "benchmark trajectories", "imitation benchmark", "N/A", "MuJoCo", "MuJoCo", "dataset", "dataset-dependent", "benchmark tasks", "no", "no", "yes", "pipeline context", "benchmark/data rather than an independent raw-human-to-G1 algorithm", "https://github.com/robfiras/loco-mujoco", "dataset_benchmark", "pipeline_context_only"),
    _method("Mink", "1.1.1 in benchmark env", "2024", "caller-defined tasks", "caller-defined robot", "backend only", "only through caller", "differential IK backend", "QP differential IK", "MuJoCo", "caller-defined", "single solve", "caller-defined", "caller-defined", "no", "no", "no", "backend only", "not a complete input-to-output retargeter", "https://github.com/kevinzakka/mink", "solver_backend", "backend_only"),
    _method("PyRoki", "ProtoMotions frozen dependency", "2025", "caller-defined keypoints", "caller-defined robot", "backend only", "only through caller", "JAX robot optimization backend", "nonlinear least squares", "PyRoki", "optional", "trajectory", "caller-defined", "caller-defined", "no", "no", "no", "backend only", "not a complete input-to-output retargeter", "https://github.com/chungmin99/pyroki", "solver_backend", "backend_only"),
    _method("MIRROR", "official THEMIS implementation", "2022", "human motion", "THEMIS", "no", "non-G1 reference", "real-time differential IK", "differential IK", "iDynTree/YARP", "method-specific", "online", "method-specific", "no", "no", "no", "no", "excluded", "target robot is not Unitree G1", "https://github.com/ami-iit/paper_ramadoss-2022-ral-humanoid-retargeting", "retargeter_non_g1", "excluded"),
    _method("ReActor", "paper evidence", "2026", "human motion/control task", "humanoid robot", "method-level", "reference plus policy", "bilevel physics-aware retargeting", "retargeting + RL", "simulator", "physics", "trajectory and policy", "physics", "physics", "yes", "yes", "no", "literature only", "RL training and dynamics rollout are outside Stage 1", "https://arxiv.org/abs/2605.06593", "physics_aware_controller", "literature_only", open_source="not verified for Pilot"),
)


def _run(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _git_status(root: Path) -> str:
    return "dirty" if _run(["git", "-C", str(root), "status", "--porcelain"]) else "clean"


def _hardware() -> dict[str, Any]:
    gpu_query = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "gpu": gpu_query,
        "full_lafan_authorized": False,
    }


def generate_audit(repo_root: str | Path) -> None:
    root = Path(repo_root).resolve()
    research = root / "research"
    manifests = root / "manifests"
    research.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)
    accessed = datetime.now(timezone.utc).date().isoformat()

    matrix = research / "human_to_g1_method_matrix.csv"
    with matrix.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=METHOD_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(METHODS)

    source_lines = [
        "# First-party sources",
        "",
        f"Accessed: {accessed}",
        "",
        "Every Stage 0 classification below is anchored to an official paper, project page, repository, or documentation site.",
        "Issues and third-party summaries are not primary evidence.",
        "",
    ]
    source_lines.extend(
        f"- [{item['name']}]({item['official_source']}) — frozen evidence: `{item['version_or_commit']}`"
        for item in METHODS
    )
    source_lines.extend(
        [
            "- [GMR paper](https://arxiv.org/abs/2510.02252)",
            "- [OmniRetarget paper](https://arxiv.org/abs/2509.26633)",
            "- [H2O / PHC project](https://human2humanoid.com/)",
            "- [PHUMA project](https://davian-robotics.github.io/PHUMA/)",
            "- [Gleicher 1998](https://graphics.cs.wisc.edu/Papers/1998/Gle98/)",
            "- [Gleicher 1997](https://graphics.cs.wisc.edu/Papers/1997/Gle97a/)",
            "",
            "Exact repository revisions and local license observations are frozen in `manifests/repositories.csv`.",
            "",
            "## LAFAN1 availability note",
            "",
            "The official Ubisoft repository was accessed first, but its Git LFS endpoint reported an exhausted budget for the official archive (SHA-256 `ea918082b500a5d158e9d3aa39039df04cd42e25f5c02fe8f7e88e8e9365a977`).",
            "The Pilot uses the per-file mirror frozen in `manifests/dataset.yaml`; it matches the official 77 filenames, 496,672 frames, and nominal 30 fps and is never represented as an official Ubisoft host.",
        ]
    )
    atomic_write_text(research / "sources.md", "\n".join(source_lines) + "\n")

    taxonomy_rows = [
        "# Method scope and taxonomy",
        "",
        "The experimental boundary is a complete public path from a human motion to a canonical Unitree G1 29-DoF reference trajectory.",
        "Solvers, controllers, trackers, datasets, and benchmarks are retained as lineage or pipeline context but never promoted to retargeter results.",
        "",
        "| System | Classification | Stage 1 role | Rationale |",
        "|---|---|---|---|",
    ]
    taxonomy_rows.extend(
        f"| {item['name']} | {item['category']} | {item['stage1_role']} | {item['exclusion_reason'] or item['experiment_status']} |"
        for item in METHODS
    )
    atomic_write_text(research / "method_taxonomy.md", "\n".join(taxonomy_rows) + "\n")

    lineage = """# Lineage evidence

The graph separates three axes: computational backend, optimization abstraction,
and information content. Edges mean a documented implementation or conceptual
dependency; they do not imply identical evaluation conditions.

```mermaid
flowchart LR
  Mink[Mink backend] --> GMR[GMR]
  Mink --> PM2[ProtoMotions v2 retargeting]
  PyRoki[PyRoki backend] --> PM3[ProtoMotions v3 retargeting]
  PHC[PHC SMPL fitting] --> PM2
  SOCP[Sequential SOCP] --> Omni[OmniRetarget / Holosoma]
  Sparse[Sparse task tracking] --> Dense[Dense body preservation]
  Dense --> Interaction[Interaction preservation]
  Interaction --> Physics[Physics-aware adaptation / control]
  GMR --> Dense
  PM3 --> Dense
  Omni --> Interaction
  MaskedMimic[MaskedMimic controller] --> Physics
  BeyondMimic[BeyondMimic tracker] --> Physics
```

The graph is an evidence map, not a performance ranking.
"""
    atomic_write_text(research / "lineage_evidence.md", lineage)
    atomic_write_text(research / "lineage_graph.mmd", lineage.split("```mermaid\n", 1)[1].split("```", 1)[0])

    graph_nodes = {
        "Human motion": (0.0, 2.2),
        "Sparse tasks": (1.7, 3.2),
        "Dense body": (3.6, 3.2),
        "Interaction": (5.5, 3.2),
        "Physics control": (7.4, 3.2),
        "Mink": (1.7, 1.2),
        "PyRoki": (3.6, 1.2),
        "Sequential SOCP": (5.5, 1.2),
        "G1 reference": (7.4, 2.2),
    }
    graph_edges = (
        ("Human motion", "Sparse tasks", "information"),
        ("Sparse tasks", "Dense body", "adds body targets"),
        ("Dense body", "Interaction", "adds contact/object state"),
        ("Interaction", "Physics control", "adds dynamics"),
        ("Mink", "Sparse tasks", "backend"),
        ("Mink", "Dense body", "GMR / controlled"),
        ("PyRoki", "Dense body", "ProtoMotions v3"),
        ("Sequential SOCP", "Interaction", "OmniRetarget"),
        ("Sparse tasks", "G1 reference", "outputs"),
        ("Dense body", "G1 reference", "outputs"),
        ("Interaction", "G1 reference", "outputs"),
        ("G1 reference", "Physics control", "consumed by tracker"),
    )
    with (research / "lineage_graph_source.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("source", "target", "relation"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(
            {"source": source, "target": target, "relation": relation}
            for source, target, relation in graph_edges
        )
    # Imported only for the audit command so run-method and validation do not
    # create Matplotlib cache/config side effects at module import time.
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(12, 5.5))
    colors = {
        "Human motion": "#e76f51",
        "G1 reference": "#2a9d8f",
        "Mink": "#457b9d",
        "PyRoki": "#457b9d",
        "Sequential SOCP": "#457b9d",
    }
    for source, target, relation in graph_edges:
        start = graph_nodes[source]
        end = graph_nodes[target]
        axis.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "->", "color": "#65727e", "lw": 1.25},
        )
        midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        axis.text(*midpoint, relation, fontsize=7, color="#52606d", ha="center")
    for node, (x, y) in graph_nodes.items():
        axis.scatter(x, y, s=1250, color=colors.get(node, "#f4d35e"), zorder=3)
        axis.text(x, y, node, ha="center", va="center", fontsize=8, weight="bold", zorder=4)
    axis.set_xlim(-0.7, 8.1)
    axis.set_ylim(0.35, 4.05)
    axis.axis("off")
    axis.set_title("Human-to-G1 method lineage and pipeline boundary", loc="left", weight="bold")
    figure.tight_layout()
    for extension, kwargs in (("svg", {}), ("pdf", {}), ("png", {"dpi": 300})):
        figure.savefig(research / f"lineage_graph.{extension}", bbox_inches="tight", **kwargs)
    plt.close(figure)

    claims = research / "claims.csv"
    claim_fields = [
        "claim_id",
        "claim_text",
        "claim_type",
        "source_or_experiment",
        "method",
        "metric",
        "result_file",
        "figure",
        "scope",
        "caveat",
        "status",
    ]
    with claims.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=claim_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            [
                {
                    "claim_id": "scope-001",
                    "claim_text": "A solver, tracker, controller, or dataset alone is not a complete human-to-G1 retargeter.",
                    "claim_type": "scope",
                    "source_or_experiment": "first-party documented I/O",
                    "method": "multiple",
                    "result_file": "research/human_to_g1_method_matrix.csv",
                    "scope": "Stage 0 taxonomy",
                    "caveat": "Classification is revision-specific.",
                    "status": "supported",
                },
                {
                    "claim_id": "hist-gmr-scale",
                    "claim_text": "Frozen GMR uses a 0.9 Hips scale multiplied by 1.75/1.8, yielding root scale 0.875.",
                    "claim_type": "historical",
                    "source_or_experiment": "official GMR loader and configuration",
                    "method": "GMR",
                    "metric": "declared native root scale",
                    "result_file": "manifests/evaluator.yaml",
                    "scope": "frozen commit",
                    "caveat": "This is a policy choice, not robot geometry calibration.",
                    "status": "supported",
                },
                {
                    "claim_id": "hist-omni-scale",
                    "claim_text": "Frozen Holosoma LAFAN uses default_scale_factor 1.27/1.7.",
                    "claim_type": "historical",
                    "source_or_experiment": "official Holosoma configuration",
                    "method": "OmniRetarget",
                    "metric": "declared native root scale",
                    "result_file": "manifests/evaluator.yaml",
                    "scope": "frozen commit",
                    "caveat": "This is separate from the common evaluator scale.",
                    "status": "supported",
                },
                {
                    "claim_id": "hist-proto-v3",
                    "claim_text": "ProtoMotions v3 replaces the earlier Mink lineage with a trajectory-level PyRoki/JAX retargeter.",
                    "claim_type": "historical",
                    "source_or_experiment": "official ProtoMotions README and retargeting workflow",
                    "method": "ProtoMotions v3",
                    "result_file": "research/sources.md",
                    "scope": "frozen v3 checkout",
                    "caveat": "Historical v2 is not treated as an experimental point.",
                    "status": "supported",
                },
                {
                    "claim_id": "hist-phc-boundary",
                    "claim_text": "PHC repository retargeting preprocessing is distinct from its learned physics controller.",
                    "claim_type": "historical",
                    "source_or_experiment": "official PHC retargeting documentation and scripts",
                    "method": "PHC",
                    "result_file": "research/method_taxonomy.md",
                    "scope": "frozen checkout",
                    "caveat": "The controller is excluded from the retargeter scatter.",
                    "status": "supported",
                },
                {
                    "claim_id": "exp-common-scale",
                    "claim_text": "All primary quality metrics use one method-independent neutral-G1/source landmark scale.",
                    "claim_type": "experimental",
                    "source_or_experiment": "frozen evaluator protocol",
                    "method": "all core",
                    "metric": "common_static_scale",
                    "result_file": "metrics/root_scale_diagnostics.csv",
                    "figure": "figures/root_scale_policies.svg",
                    "scope": "single frozen Pilot",
                    "caveat": "Native method scales remain reported separately.",
                    "status": "supported",
                },
                {
                    "claim_id": "exp-root-decomposition",
                    "claim_text": "Root translation error is decomposed into common-scale, native-scale, and scale-invariant path-shape components.",
                    "claim_type": "experimental",
                    "source_or_experiment": "evaluator v2",
                    "method": "all core",
                    "metric": "root translation decomposition",
                    "result_file": "metrics/root_scale_diagnostics.csv",
                    "figure": "figures/root_error_decomposition.svg",
                    "scope": "single frozen Pilot",
                    "caveat": "No component is a dataset-level ranking.",
                    "status": "supported",
                },
                {
                    "claim_id": "exp-rfkpe",
                    "claim_text": "RF-KPE reports root-frame semantic keypoint preservation after common morphology scaling.",
                    "claim_type": "experimental",
                    "source_or_experiment": "evaluator v2",
                    "method": "all core",
                    "metric": "RF-KPE",
                    "result_file": "metrics/core_summary.csv",
                    "figure": "figures/targeted_vs_untracked.svg",
                    "scope": "single frozen Pilot",
                    "caveat": "It removes global root translation and heading, so root tracking is separate.",
                    "status": "supported",
                },
                {
                    "claim_id": "exp-sparse-seeds",
                    "claim_text": "Sparse null-space sensitivity is evaluated across three deterministic first-frame seeds.",
                    "claim_type": "experimental",
                    "source_or_experiment": "controlled baseline runs",
                    "method": "Controlled Sparse-IK",
                    "metric": "seed variance and pairwise divergence",
                    "result_file": "metrics/sparse_seed_variance.csv",
                    "figure": "figures/sparse_seed_divergence.svg",
                    "scope": "single frozen Pilot",
                    "caveat": "Three seeds diagnose sensitivity; they do not sample the full solution space.",
                    "status": "supported",
                },
                {
                    "claim_id": "exp-interaction",
                    "claim_text": "Full versus No-Hard is reported only as a two-case interaction ablation.",
                    "claim_type": "experimental",
                    "source_or_experiment": "official box and climb cases",
                    "method": "OmniRetarget",
                    "metric": "surface contact, penetration, foot sticking",
                    "result_file": "metrics/interaction_summary.csv",
                    "figure": "figures/interaction_full_vs_no_hard.svg",
                    "scope": "two cases",
                    "caveat": "Not dataset-level evidence.",
                    "status": "supported",
                },
            ]
        )

    atomic_write_yaml(manifests / "hardware.yaml", _hardware())
    atomic_write_yaml(
        manifests / "methods.yaml",
        {
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "required": [
                {
                    "name": "sparse",
                    "implementation": "controlled Mink baseline",
                    "config": "configs/controlled_mink.yaml",
                    "seeds": ["neutral", "A", "B"],
                },
                {
                    "name": "dense",
                    "implementation": "controlled Mink baseline",
                    "config": "configs/controlled_mink.yaml",
                    "seeds": ["neutral"],
                },
                {
                    "name": "gmr",
                    "repository": "https://github.com/YanjieZe/GMR.git",
                    "commit": "bb1bbe40774794fceb2a7c579a3464a28e68c844",
                    "license": "MIT",
                    "native_environment": "robot",
                },
                {
                    "name": "omniretarget",
                    "upstream_project": "Holosoma",
                    "repository": "https://github.com/amazon-far/holosoma.git",
                    "commit": "5f48635a3624656a5f46a07df26d43187e59f855",
                    "license": "Apache-2.0",
                    "native_environment": "hsretargeting",
                    "patch": "patches/holosoma/interaction-hard-constraint-flags.patch",
                },
            ],
            "conditional_order": ["protomotions_v3", "phc"],
            "conditional_per_method_limit_s": 7200,
            "conditional_total_limit_s": 14400,
            "full_lafan_authorized": False,
            "hard_stop_message": FULL_LAFAN_STOP_MESSAGE,
        },
    )
    repositories = manifests / "repositories.csv"
    with repositories.open("w", newline="", encoding="utf-8") as stream:
        fields = ["name", "url", "commit", "status", "license", "role"]
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerow(
            {
                "name": "retargeting_comparison",
                "url": "https://github.com/Arkitect-z/retargeting_comparison.git",
                "commit": _run(["git", "-C", str(root), "rev-parse", "HEAD"]),
                "status": _git_status(root),
                "license": "Apache-2.0",
                "role": "original harness",
            }
        )
        writer.writerow(
            {
                "name": "holosoma",
                "url": "https://github.com/amazon-far/holosoma.git",
                "commit": "5f48635a3624656a5f46a07df26d43187e59f855",
                "status": "pinned with one recorded constraint-flag patch",
                "license": "Apache-2.0",
                "role": "required retargeter and canonical G1 model",
            }
        )
        writer.writerow(
            {
                "name": "GMR",
                "url": "https://github.com/YanjieZe/GMR.git",
                "commit": _run(
                    ["git", "-C", str(root / "external" / "GMR"), "rev-parse", "HEAD"]
                ),
                "status": (
                    _git_status(root / "external" / "GMR")
                    if (root / "external" / "GMR" / ".git").is_dir()
                    else "pending checkout"
                ),
                "license": "MIT",
                "role": "required retargeter",
            }
        )
        for name, url, folder, license_name, role in (
            (
                "ProtoMotions v3",
                "https://github.com/NVlabs/ProtoMotions.git",
                "ProtoMotions",
                "Apache-2.0",
                "conditional retargeter",
            ),
            (
                "PHC",
                "https://github.com/ZhengyiLuo/PHC.git",
                "PHC",
                "MIT",
                "conditional retargeter component",
            ),
        ):
            checkout = root / "external" / folder
            writer.writerow(
                {
                    "name": name,
                    "url": url,
                    "commit": _run(["git", "-C", str(checkout), "rev-parse", "HEAD"]),
                    "status": (
                        _git_status(checkout)
                        if (checkout / ".git").is_dir()
                        else "pending checkout"
                    ),
                    "license": license_name,
                    "role": role,
                }
            )
        writer.writerow(
            {
                "name": "LAFAN1 official metadata",
                "url": "https://github.com/ubisoft/ubisoft-laforge-animation-dataset.git",
                "commit": "94084601bacdf9cc3764b5c73daaeccae6035fac",
                "status": "LFS object unavailable: upstream quota exceeded",
                "license": "CC-BY-NC-ND-4.0",
                "role": "canonical dataset origin and archive SHA",
            }
        )
        writer.writerow(
            {
                "name": "LAFAN1 per-file mirror fallback",
                "url": "https://huggingface.co/datasets/johnny095212/lafan1",
                "commit": "10542a0e0c983464b566ebf5c49e78a250279ec4",
                "status": "validated against official sequence/frame/fps invariants",
                "license": "inherits original dataset terms",
                "role": "download fallback; not represented as an official host",
            }
        )
    atomic_write_yaml(
        manifests / "versions.yaml",
        {
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "harness_commit": _run(["git", "-C", str(root), "rev-parse", "HEAD"]),
            "stage_config_sha256": sha256_file(root / "configs" / "stage1.yaml"),
            "python": platform.python_version(),
            "full_lafan_authorized": False,
        },
    )
