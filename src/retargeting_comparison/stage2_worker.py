"""One-pass worker for a single frozen Stage-2 method/sequence job."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--sequence-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--execution-contract", required=True)
    parser.add_argument("--reference-relative-path")
    return parser


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _save_atomic(motion, output: Path, source_frames: int) -> None:
    temporary = output.with_name(f".{output.stem}.partial.npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    motion.save(temporary, source_frame_count=source_frames)
    os.replace(temporary, output)


def _canonical_sha256(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_execution_contract(args, root: Path, sequence_path: Path) -> dict:
    from .io_utils import sha256_file

    path = Path(args.execution_contract).resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    payload = dict(value)
    recorded = payload.pop("execution_contract_sha256", None)
    if value.get("schema_version") != 2 or recorded != _canonical_sha256(payload):
        raise SystemExit("Execution contract payload SHA-256 failed")
    planned = value.get("planned_job", {})
    if (
        planned.get("method") != args.method
        or value.get("repo_root") != str(root)
        or value.get("sequence_manifest_path") != str(sequence_path)
        or value.get("sequence_manifest_sha256") != sha256_file(sequence_path)
        or value.get("expected_output_path") != str(Path(args.output).resolve())
        or value.get("work_directory") != str(Path(args.work_dir).resolve())
        or value.get("production_passes") != 1
        or value.get("automatic_retry") is not False
    ):
        raise SystemExit("Worker arguments differ from the frozen execution contract")
    command = value.get("command", [])
    try:
        observed = [
            item.decode("utf-8")
            for item in Path("/proc/self/cmdline").read_bytes().split(b"\0")
            if item
        ]
    except OSError:
        observed = [str(Path(sys.executable).resolve()), *sys.argv]
    if observed:
        observed[0] = str(Path(observed[0]).resolve())
    declared = list(command)
    if declared:
        declared[0] = str(Path(declared[0]).resolve())
    if observed != declared or value.get("command_sha256") != _canonical_sha256(command):
        raise SystemExit("Worker command differs from the frozen execution contract")
    environment = value.get("environment", {})
    cpu_affinity = value.get("cpu_affinity")
    if (
        value.get("cpu_affinity_policy")
        != "one_distinct_allowed_physical_core_per_worker_lane"
        or not isinstance(cpu_affinity, list)
        or len(cpu_affinity) != 1
        or not hasattr(os, "sched_setaffinity")
    ):
        raise SystemExit("Worker CPU-affinity contract is invalid")
    os.sched_setaffinity(0, {int(cpu_affinity[0])})
    if sorted(os.sched_getaffinity(0)) != [int(cpu_affinity[0])]:
        raise SystemExit("Worker did not enter its frozen physical-CPU lane")
    python = Path(sys.executable).resolve()
    history = python.parent.parent / "conda-meta/history"
    if (
        environment.get("python_path") != str(python)
        or environment.get("python_sha256") != sha256_file(python)
        or environment.get("conda_history_path") != str(history)
        or environment.get("conda_history_sha256") != sha256_file(history)
        or value.get("environment_provenance_sha256")
        != _canonical_sha256(environment)
    ):
        raise SystemExit("Worker conda environment differs from Stage-1 provenance")
    for key, expected in value.get("thread_environment", {}).items():
        if os.environ.get(key) != expected:
            raise SystemExit(f"Worker thread environment differs: {key}")
    if value.get("thread_environment_sha256") != _canonical_sha256(
        value.get("thread_environment", {})
    ):
        raise SystemExit("Thread environment contract hash failed")
    return value


def _controlled(root: Path, human, manifest: dict, variant: str):
    import numpy as np

    from .controlled_mink import ControlledMinkRetargeter

    calibration = manifest["common_scale"]
    local_scale = float(calibration["local_body_scale"])
    root_scale = float(calibration["root_displacement_scale"])
    policy = {
        "root_axis": [root_scale, root_scale, root_scale],
        "local_axes": {
            "root_torso_legs": [local_scale, local_scale, local_scale],
            "arms": [local_scale, local_scale, local_scale],
        },
        "evidence_role": "stage2_controlled_common_per_sequence_scale",
    }
    retargeter = ControlledMinkRetargeter(
        root,
        variant,
        "neutral",
        scale_policy=policy,
        method_label=f"{variant}/controlled-common-per-sequence-scale",
    )
    # ControlledMinkRetargeter validates its frozen Stage-1 config at startup.
    # Its target-policy branch permits a new scale but still derives the anchor
    # from the Pilot evaluator; replace only that translation so its internal
    # expression evaluates to this sequence's frozen neutral-G1 anchor.
    configured_root_scale = retargeter.root_displacement_scale
    source_root0 = human.world_positions[0, 0]
    desired_alignment = np.asarray(
        calibration["root_alignment_translation_m"], dtype=np.float64
    )
    desired_robot_anchor = source_root0 * root_scale + desired_alignment
    retargeter.root_alignment_translation = (
        desired_robot_anchor - source_root0 * configured_root_scale
    )
    from .stage2 import _controlled_pre_solver_target_contract

    target_contract = _controlled_pre_solver_target_contract(
        root, human, manifest, variant
    )
    if target_contract["target_labels"] != [
        str(spec["semantic"]) for spec, _task in retargeter.frame_tasks
    ]:
        raise RuntimeError("Controlled solver target order differs from target receipt")
    motion = retargeter.run(human)
    for key in (
        "scale_policy",
        "root_alignment_translation_m",
        "root_displacement_scale",
        "local_body_scale",
        "root_anchor_policy",
    ):
        motion.metadata.pop(key, None)
    motion.metadata.update(
        {
            "experiment_stratum": "controlled_common_per_sequence_scale",
            "common_per_sequence_scale": local_scale,
            "common_per_sequence_local_body_scale": local_scale,
            "common_per_sequence_root_displacement_scale": root_scale,
            "common_scale_definition": calibration["definition"],
            "common_root_alignment_translation_m": desired_alignment.tolist(),
            "stage2_authoritative_scale_anchor": {
                "policy": "controlled_common_per_sequence_scale",
                "local_body_scale": local_scale,
                "root_displacement_scale": root_scale,
                "root_alignment_translation_m": desired_alignment.tolist(),
                "calibration_evidence_sha256": manifest[
                    "calibration_evidence_sha256"
                ],
                "replaces_stage1_pilot_scale_anchor_metadata": True,
            },
            "stage2_pre_solver_target_contract": target_contract,
            "stage2_production_passes": 1,
        }
    )
    return motion


def _bind_output_metadata(
    motion,
    args: argparse.Namespace,
    manifest: dict,
    contract: dict,
    sequence_path: Path,
) -> None:
    from .io_utils import sha256_file

    planned = contract["planned_job"]
    method_contract = contract["method_contract"]
    motion.metadata.update(
        {
            "stage2_plan_sha256": contract["plan_sha256"],
            "stage2_design_id": contract["design_id"],
            "stage2_config_sha256": contract["config_sha256"],
            "stage2_repository_contract_sha256": contract[
                "repository_contract_sha256"
            ],
            "stage2_hardware_contract_sha256": contract[
                "hardware_contract_sha256"
            ],
            "stage2_job_contract_sha256": contract["job_contract_sha256"],
            "stage2_execution_contract_sha256": contract[
                "execution_contract_sha256"
            ],
            "stage2_method_contract_sha256": contract[
                "method_contract_sha256"
            ],
            "stage2_policy_sha256": contract["policy_sha256"],
            "stage2_sequence_manifest": str(sequence_path),
            "stage2_sequence_manifest_sha256": sha256_file(sequence_path),
            "stage2_sequence_payload_sha256": manifest["payload_sha256"],
            "stage2_source_sha256": manifest["source_sha256"],
            "stage2_canonical_source_sha256": manifest["canonical_sha256"],
            "stage2_reference_sha256": manifest.get("reference_sha256"),
            "stage2_unitree_reference_contract_sha256": contract.get(
                "unitree_reference_contract_sha256"
            ),
            "stage2_environment_provenance_sha256": contract[
                "environment_provenance_sha256"
            ],
            "stage2_exact_upstream_config_asset_policy_contract": {
                "method": args.method,
                "stratum": planned["stratum"],
                "registered_revision": method_contract.get("registered_revision"),
                "method_contract_sha256": contract["method_contract_sha256"],
                "policy_sha256": contract["policy_sha256"],
                "timing_evidence_sha256": method_contract.get(
                    "timing_evidence_sha256"
                ),
                "environment_provenance_sha256": method_contract.get(
                    "environment_provenance_sha256"
                ),
            },
            "stage2_production_passes": 1,
        }
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.repo_root).resolve()
    sequence_path = Path(args.sequence_manifest).resolve()
    execution_contract = _load_execution_contract(args, root, sequence_path)
    manifest = json.loads(sequence_path.read_text())
    manifest_payload = dict(manifest)
    manifest_recorded = manifest_payload.pop("payload_sha256", None)
    from .io_utils import sha256_file

    if (
        manifest.get("schema_version") != 2
        or manifest_recorded != _canonical_sha256(manifest_payload)
        or manifest_recorded != execution_contract.get("sequence_payload_sha256")
        or manifest.get("plan_sha256") != execution_contract.get("plan_sha256")
        or manifest.get("calibration_evidence_sha256")
        != execution_contract.get("calibration_evidence_sha256")
    ):
        raise SystemExit("Sequence/calibration manifest contract failed")
    source_frames = int(manifest["source_frames"])
    canonical_path = _resolve(root, manifest["canonical_path"])
    source_path = _resolve(root, manifest["source_path"])
    if (
        sha256_file(canonical_path) != manifest["canonical_sha256"]
        or sha256_file(source_path) != manifest["source_sha256"]
        or execution_contract.get("source_sha256") != manifest["source_sha256"]
        or execution_contract.get("canonical_source_sha256")
        != manifest["canonical_sha256"]
    ):
        raise SystemExit("Stage-2 source bytes differ from the frozen contract")
    output = Path(args.output).resolve()
    work = Path(args.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)

    from .schemas import CanonicalHuman

    human = CanonicalHuman.load(canonical_path)
    started = time.perf_counter()
    if args.method == "sparse-neutral":
        motion = _controlled(root, human, manifest, "sparse")
    elif args.method == "dense":
        motion = _controlled(root, human, manifest, "dense")
    elif args.method == "gmr":
        from .method_adapters import run_gmr

        motion = run_gmr(root, source_path, canonical_path)
        motion.metadata["experiment_stratum"] = "native_public_pipeline"
    elif args.method == "omniretarget":
        from .method_adapters import run_holosoma

        motion = run_holosoma(root, canonical_path, work / "holosoma")
        motion.metadata["experiment_stratum"] = "native_public_pipeline"
    elif args.method == "protomotions-v2.3":
        from .protomotions_v2 import ProtoMotionsV2Retargeter

        motion = ProtoMotionsV2Retargeter(root).run(human)
        motion.metadata["experiment_stratum"] = "benchmark_public_retargeter_port"
    elif args.method == "protomotions-v3":
        from .protomotions_v3_worker import main as run_v3
        from .source_adapter_audit import _prepare_protomotions_keypoints

        keypoints = work / "source_adapter/keypoints.npy"
        keypoints.parent.mkdir(parents=True, exist_ok=True)
        _prepare_protomotions_keypoints(human, keypoints)
        native_canonical = work / "protomotions-v3/stage2_canonical.npz"
        code = run_v3(
            [
                "--repo-root",
                str(root),
                "--python",
                sys.executable,
                "--source",
                str(canonical_path),
                "--keypoints",
                str(keypoints),
                "--output",
                str(native_canonical),
                "--work-dir",
                str(work / "protomotions-v3"),
                "--evidence-json",
                str(work / "protomotions-v3/evidence.json"),
                "--target-raw-frames",
                str(source_frames),
                "--warmup-runs",
                "0",
                "--measured-runs",
                "1",
            ]
        )
        if code != 0:
            return code
        from .schemas import CanonicalG1

        motion = CanonicalG1.load(native_canonical)
        motion.metadata.update(
            {
                "experiment_stratum": "benchmark_public_retargeter_port",
                "stage2_method_id": args.method,
                "stage2_sequence_id": manifest["sequence_id"],
                "stage2_sequence_manifest": str(sequence_path),
                "stage2_production_passes": 1,
                "stage2_worker_wall_time_s": time.perf_counter() - started,
            }
        )
        _bind_output_metadata(
            motion, args, manifest, execution_contract, sequence_path
        )
        _save_atomic(motion, output, source_frames)
        return 0
    elif args.method == "unitree-reference":
        from .unitree_reference import (
            CANONICAL_COORDINATE_VIEW,
            load_unitree_g1_csv,
        )

        if not args.reference_relative_path:
            raise SystemExit("Reference job lacks a filename-intersection path")
        if args.reference_relative_path != manifest.get("reference_relative_path"):
            raise SystemExit(
                "Reference argument differs from the frozen per-sequence binding"
            )
        reference_path = _resolve(root, manifest["reference_path"])
        reference_frames = int(manifest["reference_frames"])
        motion = load_unitree_g1_csv(
            reference_path,
            frame_start=0,
            frame_end=min(source_frames, reference_frames),
            expected_sha256=manifest["reference_sha256"],
            expected_source_frames=source_frames,
            coordinate_view=CANONICAL_COORDINATE_VIEW,
        )
        native_reference_fps = float(motion.fps)
        source_fps = float(manifest["fps"])
        relative_fps_difference = abs(native_reference_fps - source_fps) / source_fps
        fps_tolerance = float(
            manifest.get("reference_native_fps_relative_tolerance", 2.0e-5)
        )
        if relative_fps_difference > fps_tolerance:
            raise SystemExit(
                "Reference/source fps differ beyond the frozen 2e-5 relative tolerance"
            )
        # The CSV contains one row per exact source frame. Use the source clock
        # for every downstream temporal metric while retaining the native 30 Hz
        # label as provenance; no frame interpolation is performed.
        motion.fps = source_fps
        motion.metadata.update(
            {
                "experiment_stratum": "external_reference",
                "stage2_filename_intersection": True,
                "upstream_path": args.reference_relative_path,
                "source_and_reference_frame_count_match": reference_frames
                == source_frames,
                "native_reference_fps": native_reference_fps,
                "source_timeline_fps": source_fps,
                "native_source_fps_relative_difference": relative_fps_difference,
                "temporal_scoring_uses_source_fps": True,
                "fps_resampling_performed": False,
                "timeline_alignment_contract": "same_basename_and_frame_indices_only",
                "byte_identical_human_source_verified": False,
                "exact_timestamp_identity_claimed": False,
                "verified_official_ground_truth": False,
            }
        )
    else:
        raise SystemExit(f"Unknown Stage-2 method: {args.method}")

    motion.metadata.update(
        {
            "stage2_method_id": args.method,
            "stage2_sequence_id": manifest["sequence_id"],
            "stage2_sequence_manifest": str(sequence_path),
            "stage2_production_passes": 1,
            "stage2_worker_wall_time_s": time.perf_counter() - started,
        }
    )
    _bind_output_metadata(motion, args, manifest, execution_contract, sequence_path)
    _save_atomic(motion, output, source_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
