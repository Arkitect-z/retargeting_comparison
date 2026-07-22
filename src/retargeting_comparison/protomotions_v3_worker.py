"""End-to-end isolated launcher and canonicalizer for ProtoMotions v3."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .io_utils import atomic_write_json, sha256_file
from .protomotions_v3 import (
    PROTOMOTIONS_V3_COMMIT,
    PYROKI_COMMIT,
    TARGET_RAW_FRAMES,
    actuated_urdf_joints,
    audit_environment,
    audit_robot_assets,
    build_native_command,
    convert_native_output,
    joint_limit_diagnostics,
    native_environment,
    prepare_scale_variant_keypoints,
    solver_log_diagnostics,
    validate_canonical_fk,
    validate_keypoint_input,
    write_failure_evidence,
)
from .schemas import CanonicalHuman
from .scale_worker import (
    _atomic_save_motion,
    scale_execution_contract,
)
from .source_adapter_audit import _prepare_protomotions_keypoints
from .stage1_timing_campaign import assert_formal_campaign_ownership


def _pin_to_one_cpu() -> list[int]:
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("Formal ProtoMotions timing requires Linux CPU affinity")
    allowed = sorted(os.sched_getaffinity(0))
    if not allowed:
        raise RuntimeError("Formal ProtoMotions timing has no allowed CPU")
    selected = [int(allowed[0])]
    os.sched_setaffinity(0, set(selected))
    if sorted(os.sched_getaffinity(0)) != selected:
        raise RuntimeError("Could not freeze ProtoMotions CPU affinity")
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--source", required=True, help="Canonical 600-frame human NPZ")
    parser.add_argument("--keypoints", required=True, help="Audited native keypoints.npy")
    parser.add_argument("--output", required=True, help="Canonical G1 output NPZ")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--evidence-json", required=True)
    parser.add_argument("--target-raw-frames", type=int, default=TARGET_RAW_FRAMES)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=3)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--root-scale-multiplier", type=float, default=1.0)
    parser.add_argument("--local-scale-multiplier", type=float, default=1.0)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--capture-runtime-witness", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    worker_started_at = datetime.now(timezone.utc).isoformat()
    cpu_affinity = _pin_to_one_cpu()
    root = Path(args.repo_root).resolve()
    assert_formal_campaign_ownership(root)
    scale_contract = scale_execution_contract(root, "protomotions_v3")
    work = Path(args.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    evidence_path = Path(args.evidence_json).resolve()
    implementation_hashes = {
        "wrapper": sha256_file(Path(__file__)),
        "native_worker": sha256_file(
            root / "src/retargeting_comparison/protomotions_v3_native.py"
        ),
        "adapter": sha256_file(
            root / "src/retargeting_comparison/protomotions_v3.py"
        ),
        "source_adapter": sha256_file(
            root / "src/retargeting_comparison/source_adapter_audit.py"
        ),
        "campaign": sha256_file(
            root / "src/retargeting_comparison/protomotions_v3_campaign.py"
        ),
        "official_solver": sha256_file(
            root / "external/ProtoMotions/pyroki/batch_retarget_to_g1_from_keypoints.py"
        ),
    }
    environment = audit_environment(root, args.python, device=args.device)
    native_environment_observation = {
        "environment_name": str(environment["python_environment"]),
        "requested_device": str(environment["requested_device"]),
        "pipeline_ready": bool(environment["pipeline_ready"]),
        "protomotions_commit": str(environment["protomotions_commit"]),
        "pyroki_commit": str(environment["pyroki_commit"]),
        "jaxls_commit": str(environment["jaxls_commit"]),
    }
    try:
        preflight_started = time.perf_counter()
        source_load_started = time.perf_counter()
        human = CanonicalHuman.load(args.source)
        source_load_time_s = time.perf_counter() - source_load_started
        if not 1 <= args.target_raw_frames <= len(human.timestamps):
            raise ValueError(
                "ProtoMotions target_raw_frames must select a non-empty source prefix"
            )
        declared_keypoints = Path(args.keypoints).resolve()
        declared_audit = validate_keypoint_input(
            declared_keypoints, len(human.timestamps)
        )
        base_keypoints = work / "native_input/canonical_lafan_keypoints.npy"
        adapter_transform_started = time.perf_counter()
        _prepare_protomotions_keypoints(human, base_keypoints)
        adapter_transform_time_s = time.perf_counter() - adapter_transform_started
        base_audit = validate_keypoint_input(base_keypoints, len(human.timestamps))
        with declared_keypoints.open("rb") as declared_stream:
            declared_mapping = np.load(declared_stream, allow_pickle=True).item()
        with base_keypoints.open("rb") as base_stream:
            base_mapping = np.load(base_stream, allow_pickle=True).item()
        for key in (
            "positions", "orientations", "left_foot_contacts",
            "right_foot_contacts", "fps",
        ):
            if not np.array_equal(np.asarray(declared_mapping[key]), np.asarray(base_mapping[key])):
                raise ValueError(
                    f"Declared ProtoMotions keypoint input differs from the reconstructed canonical adapter: {key}"
                )
        input_keypoints = base_keypoints
        scale_intervention_time_s = 0.0
        if args.root_scale_multiplier != 1.0 or args.local_scale_multiplier != 1.0:
            input_keypoints = work / "native_input/scale_variant_keypoints.npy"
            scale_intervention_started = time.perf_counter()
            keypoint_audit = prepare_scale_variant_keypoints(
                base_keypoints,
                input_keypoints,
                root_scale_multiplier=args.root_scale_multiplier,
                local_scale_multiplier=args.local_scale_multiplier,
            )
            scale_intervention_time_s = (
                time.perf_counter() - scale_intervention_started
            )
        else:
            keypoint_audit = base_audit
        source_adapter_time_s = (
            source_load_time_s
            + adapter_transform_time_s
            + scale_intervention_time_s
        )
        try:
            keypoint_audit["path"] = str(input_keypoints.relative_to(root))
        except ValueError:
            # Scale-variant work directories are expected to remain inside the
            # repository, but retain an auditable path if a caller overrides it.
            keypoint_audit["path"] = str(input_keypoints)
        robot_audit = audit_robot_assets(root)
        if not environment["pipeline_ready"]:
            raise RuntimeError("native environment is missing a G1-script dependency")
        if not (
            environment["protomotions_commit_matches"]
            and environment["pyroki_commit_matches"]
            and environment["jaxls_commit_matches"]
        ):
            raise RuntimeError("native checkout does not match the frozen revisions")
        preflight_audit_time_s_excluded = max(
            0.0,
            time.perf_counter() - preflight_started - source_adapter_time_s,
        )
    except Exception as error:
        write_failure_evidence(
            evidence_path,
            phase="preflight",
            reason=f"{type(error).__name__}: {error}",
            environment_audit=environment,
        )
        return 2

    preflight = {
        "method": "ProtoMotions v3 / modified PyRoki",
        "status": "audit_passed" if args.audit_only else "running",
        "upstream_commit": PROTOMOTIONS_V3_COMMIT,
        "pyroki_commit": PYROKI_COMMIT,
        "environment": environment,
        "input": keypoint_audit,
        "declared_input": declared_audit,
        "source_adapter_audit_time_s_excluded": source_adapter_time_s,
        "source_adapter_audit_components_s_excluded": {
            "canonical_source_load": source_load_time_s,
            "canonical_to_native_transform_and_file_contract": adapter_transform_time_s,
            "scale_intervention": scale_intervention_time_s,
        },
        "preflight_audit_time_s_excluded": preflight_audit_time_s_excluded,
        "source_adapter_reconstructed_exactly": True,
        "robot_assets": robot_audit,
        "implementation_hashes": implementation_hashes,
        "scale_execution_contract": scale_contract,
        "scale_runtime_environment": native_environment_observation,
        "scale_variant": {
            (1.0, 1.0): "native",
            (0.95, 1.0): "root_minus_5",
            (1.05, 1.0): "root_plus_5",
            (1.0, 0.95): "local_minus_5",
            (1.0, 1.05): "local_plus_5",
        }.get((args.root_scale_multiplier, args.local_scale_multiplier)),
        "root_scale_multiplier": args.root_scale_multiplier,
        "local_scale_multiplier": args.local_scale_multiplier,
        "canonical_source_file_sha256": sha256_file(args.source),
        "canonical_source_embedded_sha256": human.source_sha256,
        "config_sha256": sha256_file(root / "configs/protomotions_v3.yaml"),
        "scale_protocol_sha256": sha256_file(
            root / "configs/scale_policy_sensitivity.yaml"
        ),
        "started_at_utc": worker_started_at,
        "cpu_affinity": cpu_affinity,
        "thread_limit": 1,
        "synthetic_or_substitute_output_used": False,
    }
    atomic_write_json(evidence_path, preflight)
    if args.audit_only:
        return 0

    attempt = f"{args.device}-{args.target_raw_frames}f"
    native_output = work / "native" / attempt / "protomotions_v3_retargeted.npz"
    timing_json = work / "native" / attempt / "timing.json"
    stdout_log = work / "logs" / f"native.{attempt}.stdout.log"
    stderr_log = work / "logs" / f"native.{attempt}.stderr.log"
    native_output.parent.mkdir(parents=True, exist_ok=True)
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    command = build_native_command(
        python=args.python,
        repo_root=root,
        source=args.source,
        keypoints=input_keypoints,
        native_output=native_output,
        timing_json=timing_json,
        target_raw_frames=args.target_raw_frames,
        warmup_runs=args.warmup_runs,
        measured_runs=args.measured_runs,
        root_scale_multiplier=args.root_scale_multiplier,
        local_scale_multiplier=args.local_scale_multiplier,
        capture_runtime_witness=args.capture_runtime_witness,
    )
    started = time.perf_counter()
    with stdout_log.open("w", encoding="utf-8") as stdout, stderr_log.open(
        "w", encoding="utf-8"
    ) as stderr:
        result = subprocess.run(
            command,
            cwd=root,
            env=native_environment(root, args.python, device=args.device),
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    process_wall = time.perf_counter() - started
    if result.returncode != 0 or not native_output.is_file() or not timing_json.is_file():
        write_failure_evidence(
            evidence_path,
            phase="official_solver",
            reason="official native worker failed or did not produce both artifacts",
            environment_audit=environment,
            command=command,
            returncode=result.returncode,
            stdout_log=str(stdout_log),
            stderr_log=str(stderr_log),
            status="failed",
        )
        return result.returncode or 3

    solve_diagnostics = solver_log_diagnostics(stderr_log)
    expected_solver_calls = args.warmup_runs + args.measured_runs
    if (
        solve_diagnostics["solver_call_count"] != expected_solver_calls
        or not solve_diagnostics["all_calls_have_termination_record"]
    ):
        write_failure_evidence(
            evidence_path,
            phase="solver_log_validation",
            reason=(
                "official solver log does not contain the expected number of "
                f"whole-trajectory calls ({expected_solver_calls}) with explicit "
                "termination records"
            ),
            environment_audit=environment,
            command=command,
            returncode=result.returncode,
            stdout_log=str(stdout_log),
            stderr_log=str(stderr_log),
            status="failed",
        )
        return 5

    native_names = [joint.name for joint in actuated_urdf_joints(
        root / "external/ProtoMotions/protomotions/data/assets/urdf/for_retargeting/g1.urdf"
    )]
    try:
        motion = convert_native_output(
            native_output,
            source_frame_count=len(human.timestamps),
            native_joint_names=native_names,
            timing_json=timing_json,
            metadata={
                "canonical_source_sha256": human.source_sha256,
                "canonical_source_file_sha256": sha256_file(args.source),
                "method_family": "benchmark_lafan_port_of_public_solver",
                "source_adapter_class": "audited canonical-LAFAN port; no upstream LAFAN entry point",
                "experiment_role": "native_response_fixed_canonical_contact",
                "config_sha256": sha256_file(root / "configs/protomotions_v3.yaml"),
                "scale_protocol_sha256": sha256_file(
                    root / "configs/scale_policy_sensitivity.yaml"
                ),
                "native_robot_xml_sha256": sha256_file(
                    root / "external/ProtoMotions/protomotions/data/assets/urdf/for_retargeting/g1.urdf"
                ),
                "native_input_sha256": sha256_file(input_keypoints),
                "process_wall_time_s": process_wall,
                "robot_asset_audit": robot_audit,
                "device": args.device,
                "solver_log_diagnostics": solve_diagnostics,
                "implementation_hashes": implementation_hashes,
                "scale_execution_contract": scale_contract,
                "scale_runtime_environment": native_environment_observation,
                "scale_variant": {
                    (1.0, 1.0): "native",
                    (0.95, 1.0): "root_minus_5",
                    (1.05, 1.0): "root_plus_5",
                    (1.0, 0.95): "local_minus_5",
                    (1.0, 1.05): "local_plus_5",
                }[(args.root_scale_multiplier, args.local_scale_multiplier)],
                "scale_intervention": {
                    "root_scale_multiplier": args.root_scale_multiplier,
                    "local_scale_multiplier": args.local_scale_multiplier,
                    "apply_before_official_native_scale": True,
                },
                "root_scale_multiplier": args.root_scale_multiplier,
                "local_scale_multiplier": args.local_scale_multiplier,
                "fixed_contact_labels": True,
                "fixed_contact_labels_for_pure_scale": True,
                "contact_label_policy": "source contacts constructed before target-scale intervention",
            },
        )
        expected_output_frames = min(
            int(args.target_raw_frames), len(human.timestamps)
        )
        if len(motion.qpos) != expected_output_frames:
            raise ValueError(
                "ProtoMotions output frame count differs from the requested source prefix"
            )
        robot_xml = (
            root
            / "external/holosoma/src/holosoma/holosoma/data/robots/g1/scenes/scene_g1_29dof_wbt_plane.xml"
        )
        fk = validate_canonical_fk(motion, robot_xml)
        proto_urdf = (
            root
            / "external/ProtoMotions/protomotions/data/assets/urdf/for_retargeting/g1.urdf"
        )
        reference_urdf = (
            root
            / "external/holosoma/src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof.urdf"
        )
        limit_diagnostics = {
            "protomotions_native": joint_limit_diagnostics(motion, proto_urdf),
            "holosoma_reference": joint_limit_diagnostics(motion, reference_urdf),
        }
        motion.metadata["canonical_fk_validation"] = fk
        motion.metadata["joint_limit_diagnostics"] = limit_diagnostics
        timing_protocol = motion.metadata["timing_protocol"]
        repetitions = timing_protocol.get("repetitions", [])
        measured_native = [
            float(item["native_core_s"])
            for item in repetitions
            if item.get("role") == "measured"
        ]
        if not measured_native:
            raise ValueError("ProtoMotions timing lacks measured native-core durations")
        conversion_times: list[float] = []
        save_times: list[float] = []
        canonical_timing_dir = work / "canonical_timing_artifacts"
        canonical_timing_dir.mkdir(parents=True, exist_ok=True)
        for item in repetitions:
            native_timing_artifact = Path(str(item["timing_artifact"])).resolve()
            if not native_timing_artifact.is_file():
                raise FileNotFoundError(
                    f"Missing ProtoMotions repetition artifact: {native_timing_artifact}"
                )
            conversion_started = time.perf_counter()
            repetition_motion = convert_native_output(
                native_timing_artifact,
                source_frame_count=len(human.timestamps),
                native_joint_names=native_names,
                metadata={"canonical_source_sha256": human.source_sha256},
            )
            verification_conversion_time_s = time.perf_counter() - conversion_started
            canonical_timing_artifact = canonical_timing_dir / (
                f"{item['role']}_{int(item['index']):02d}.canonical_g1.npz"
            )
            save_started = time.perf_counter()
            repetition_motion.save(
                canonical_timing_artifact,
                source_frame_count=len(human.timestamps),
            )
            save_time_s = time.perf_counter() - save_started
            conversion_times.append(verification_conversion_time_s)
            save_times.append(save_time_s)
            item["verification_conversion_time_s_excluded"] = (
                verification_conversion_time_s
            )
            item["canonical_artifact_save_time_s_excluded"] = save_time_s
            item["native_total_s"] = float(item["native_core_s"])
            if item.get("timing_boundary") != (
                "canonical_source_file_to_canonical_g1_in_memory"
            ):
                raise RuntimeError("ProtoMotions native worker used an obsolete boundary")
            if item.get("native_boundary") != (
                "native_input_ready_to_native_g1_in_memory"
            ):
                raise RuntimeError("ProtoMotions native-core boundary is missing")
            item["steady_end_to_end_total_s"] = float(item["wall_time_s"])
            item["source_adapter_audit_time_s_excluded"] = source_adapter_time_s
            item["native_timing_artifact_path"] = str(native_timing_artifact)
            item["native_timing_artifact_sha256"] = sha256_file(
                native_timing_artifact
            )
            item["timing_artifact_path"] = str(
                canonical_timing_artifact.relative_to(root)
            )
            item["timing_artifact_sha256"] = sha256_file(
                canonical_timing_artifact
            )
        motion.per_frame_solve_time_s[:] = float(np.median(measured_native)) / len(
            motion.qpos
        )
        motion.metadata["native_core_total_s"] = float(np.median(measured_native))
        motion.metadata["source_adapter_audit_time_s_excluded"] = source_adapter_time_s
        motion.metadata["canonical_conversion_time_s"] = float(
            np.median(
                [
                    float(item["canonical_conversion_time_s"])
                    for item in repetitions
                    if item.get("role") == "measured"
                ]
            )
        )
        motion.metadata["verification_conversion_time_s_excluded"] = float(
            np.median(conversion_times)
        )
        motion.metadata["canonical_artifact_save_time_s_excluded"] = float(
            np.median(save_times)
        )
        measured_end_to_end = [
            float(item["steady_end_to_end_total_s"])
            for item in repetitions
            if item.get("role") == "measured"
        ]
        motion.metadata["steady_end_to_end_total_s"] = float(
            np.median(measured_end_to_end)
        )
        captures = [
            item["runtime_pre_solver_capture"]
            for item in repetitions
            if isinstance(item.get("runtime_pre_solver_capture"), dict)
        ]
        if args.capture_runtime_witness:
            target_hashes = {
                str(item["target_keypoints_sha256"]) for item in captures
            }
            if len(captures) != len(repetitions) or len(target_hashes) != 1:
                raise RuntimeError(
                    "ProtoMotions cold runtime pre-solver target capture is incomplete"
                )
            motion.metadata["runtime_pre_solver_capture"] = {
                **captures[0],
                "witness_repetitions": len(captures),
            }
        elif captures:
            raise RuntimeError(
                "ProtoMotions warm timing unexpectedly enabled runtime capture"
            )
        else:
            motion.metadata["runtime_pre_solver_capture"] = None
        motion.metadata["runtime_witness_enabled"] = bool(
            args.capture_runtime_witness
        )
        motion.metadata["cpu_affinity"] = cpu_affinity
        motion.metadata["thread_limit"] = 1
        motion.metadata["formal_timing_boundary"] = (
            "canonical_source_file_to_canonical_g1_in_memory"
        )
        _atomic_save_motion(
            motion,
            Path(args.output).resolve(),
            source_frame_count=len(human.timestamps),
        )
    except Exception as error:
        write_failure_evidence(
            evidence_path,
            phase="canonicalization",
            reason=f"{type(error).__name__}: {error}",
            environment_audit=environment,
            command=command,
            returncode=result.returncode,
            stdout_log=str(stdout_log),
            stderr_log=str(stderr_log),
            status="failed",
        )
        return 4
    evidence = {
        **preflight,
        "status": "succeeded",
        "command": command,
        "returncode": result.returncode,
        "process_wall_time_s": process_wall,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "native_output": str(native_output),
        "native_output_sha256": sha256_file(native_output),
        "timing_json": str(timing_json),
        "canonical_output": str(Path(args.output).resolve()),
        "canonical_output_sha256": sha256_file(args.output),
        "frames": len(motion.qpos),
        "completion_status": motion.metadata["completion_status"],
        "canonical_fk_validation": fk,
        "joint_limit_diagnostics": limit_diagnostics,
        "solver_log_diagnostics": solve_diagnostics,
    }
    atomic_write_json(evidence_path, evidence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
