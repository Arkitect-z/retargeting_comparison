"""Stage 0 method-scope and provenance audit generation."""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .constants import FULL_LAFAN_STOP_MESSAGE
from .io_utils import atomic_write_text, atomic_write_yaml, sha256_file

METHODS: tuple[dict[str, str], ...] = (
    {
        "method": "Controlled Sparse-IK",
        "category": "controlled_baseline",
        "family": "Mink differential IK",
        "human_input": "canonical LAFAN BVH targets",
        "g1_29dof_output": "yes",
        "stage1_role": "required",
        "reason": "Tests the root+4EE sufficiency hypothesis under a controlled backend.",
        "first_party_url": "https://github.com/kevinzakka/mink",
    },
    {
        "method": "Controlled Dense-KeyBody IK",
        "category": "controlled_baseline",
        "family": "Mink differential IK",
        "human_input": "canonical LAFAN BVH targets",
        "g1_29dof_output": "yes",
        "stage1_role": "required",
        "reason": "Task-density control sharing all Sparse solver settings.",
        "first_party_url": "https://github.com/kevinzakka/mink",
    },
    {
        "method": "GMR",
        "category": "retargeter",
        "family": "Mink + non-uniform scaling + two-stage IK",
        "human_input": "LAFAN/BVH",
        "g1_29dof_output": "yes",
        "stage1_role": "required",
        "reason": "Public direct human-to-G1 pipeline and controlled-baseline anchor.",
        "first_party_url": "https://github.com/YanjieZe/GMR",
    },
    {
        "method": "OmniRetarget / Holosoma",
        "category": "retargeter",
        "family": "interaction mesh + sequential SOCP",
        "human_input": "BVH/SMPL-family interaction motion",
        "g1_29dof_output": "yes",
        "stage1_role": "required",
        "reason": "Primary interaction-aware method and official case-study source.",
        "first_party_url": "https://github.com/amazon-far/holosoma",
    },
    {
        "method": "ProtoMotions v3",
        "category": "retargeter_and_framework",
        "family": "PyRoki retargeting pipeline",
        "human_input": "SMPL/SMPL-X",
        "g1_29dof_output": "verify at frozen revision",
        "stage1_role": "conditional_1",
        "reason": "Current public generation; two-hour native-adapter gate.",
        "first_party_url": "https://github.com/NVlabs/ProtoMotions",
    },
    {
        "method": "PHC retargeter",
        "category": "retargeter_component",
        "family": "SMPL-to-humanoid fitting",
        "human_input": "SMPL",
        "g1_29dof_output": "verify at frozen revision",
        "stage1_role": "conditional_2",
        "reason": "Historically important learned-humanoid lineage; two-hour gate.",
        "first_party_url": "https://github.com/ZhengyiLuo/PHC",
    },
    {
        "method": "ProtoMotions v2",
        "category": "historical_retargeter",
        "family": "Mink differential IK",
        "human_input": "SMPL-family",
        "g1_29dof_output": "historical revision requires verification",
        "stage1_role": "lineage_only",
        "reason": "Historical v2/v3 backend evidence; not core-first execution.",
        "first_party_url": "https://github.com/NVlabs/ProtoMotions",
    },
    {
        "method": "SOMA Retargeter",
        "category": "retargeter",
        "family": "Warp/Newton GPU IK",
        "human_input": "SOMA BVH/keypoints",
        "g1_29dof_output": "yes",
        "stage1_role": "lineage_unless_native_adapter",
        "reason": "No unverified LAFAN-to-SOMA hidden retargeter is permitted.",
        "first_party_url": "https://github.com/NVIDIA/soma-retargeter",
    },
    {
        "method": "cuRoboV2 MotionRetargeter",
        "category": "retargeter",
        "family": "GPU kinematics/optimization",
        "human_input": "SOMA/keypoints",
        "g1_29dof_output": "documented",
        "stage1_role": "lineage_unless_native_adapter",
        "reason": "Input compatibility must be demonstrated without a hidden retargeter.",
        "first_party_url": "https://nvlabs.github.io/curobo/latest/getting-started/humanoid_retargeting.html",
    },
    {
        "method": "MaskedMimic",
        "category": "controller_tracker",
        "family": "masked motion imitation",
        "human_input": "reference/partial observations",
        "g1_29dof_output": "not an independent retargeter",
        "stage1_role": "pipeline_context_only",
        "reason": "Downstream tracking/control must not become an experimental retargeter point.",
        "first_party_url": "https://research.nvidia.com/labs/par/project/maskedmimic.html",
    },
    {
        "method": "BeyondMimic",
        "category": "controller_tracker",
        "family": "whole-body tracking",
        "human_input": "robot reference",
        "g1_29dof_output": "no human-to-G1 retargeting stage",
        "stage1_role": "pipeline_context_only",
        "reason": "Consumes references rather than generating them from human motion.",
        "first_party_url": "https://github.com/HybridRobotics/whole_body_tracking",
    },
    {
        "method": "LocoMuJoCo",
        "category": "dataset_benchmark",
        "family": "simulation benchmark",
        "human_input": "preprocessed datasets",
        "g1_29dof_output": "not an independent public retargeter",
        "stage1_role": "pipeline_context_only",
        "reason": "Benchmark/data must not be represented as a retargeting algorithm.",
        "first_party_url": "https://github.com/robfiras/loco-mujoco",
    },
    {
        "method": "Mink",
        "category": "solver_backend",
        "family": "differential IK library",
        "human_input": "tasks supplied by caller",
        "g1_29dof_output": "only through configured callers",
        "stage1_role": "backend_only",
        "reason": "A naked solver is not an independent full retargeter.",
        "first_party_url": "https://github.com/kevinzakka/mink",
    },
    {
        "method": "PyRoki",
        "category": "solver_backend",
        "family": "robot kinematics optimization",
        "human_input": "tasks supplied by caller",
        "g1_29dof_output": "only through configured callers",
        "stage1_role": "backend_only",
        "reason": "A backend is lineage evidence, not an experimental system point.",
        "first_party_url": "https://github.com/chungmin99/pyroki",
    },
    {
        "method": "MIRROR",
        "category": "retargeter_non_g1",
        "family": "real-time differential IK",
        "human_input": "human motion",
        "g1_29dof_output": "no (THEMIS target)",
        "stage1_role": "excluded",
        "reason": "Target robot does not satisfy the frozen Unitree G1 scope.",
        "first_party_url": "https://github.com/ami-iit/paper_ramadoss-2022-ral-humanoid-retargeting",
    },
    {
        "method": "ReActor",
        "category": "physics_aware_controller",
        "family": "reinforcement learning",
        "human_input": "motion/control task",
        "g1_29dof_output": "outside no-training Pilot boundary",
        "stage1_role": "literature_only",
        "reason": "RL training/dynamics rollout is explicitly excluded from Stage 1.",
        "first_party_url": "https://arxiv.org/abs/2605.06593",
    },
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
        writer = csv.DictWriter(stream, fieldnames=list(METHODS[0]))
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
    source_lines.extend(f"- [{item['method']}]({item['first_party_url']})" for item in METHODS)
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
        f"| {item['method']} | {item['category']} | {item['stage1_role']} | {item['reason']} |"
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
        writer = csv.DictWriter(stream, fieldnames=claim_fields)
        writer.writeheader()
        writer.writerow(
            {
                "claim_id": "scope-001",
                "claim_text": "A solver or tracker alone is not a complete human-to-G1 retargeter.",
                "claim_type": "scope",
                "source_or_experiment": "first-party code and documented I/O",
                "method": "multiple",
                "scope": "Stage 0 taxonomy",
                "caveat": "Classification may change at a future frozen revision.",
                "status": "supported",
            }
        )

    atomic_write_yaml(manifests / "hardware.yaml", _hardware())
    atomic_write_yaml(
        manifests / "methods.yaml",
        {
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "required": ["sparse", "dense", "gmr", "omniretarget"],
            "conditional_order": ["protomotions_v3", "phc"],
            "conditional_per_method_limit_s": 7200,
            "full_lafan_authorized": False,
            "hard_stop_message": FULL_LAFAN_STOP_MESSAGE,
        },
    )
    repositories = manifests / "repositories.csv"
    with repositories.open("w", newline="", encoding="utf-8") as stream:
        fields = ["name", "url", "commit", "status", "license", "role"]
        writer = csv.DictWriter(stream, fieldnames=fields)
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
                "status": "pinned",
                "license": "verify upstream checkout",
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
