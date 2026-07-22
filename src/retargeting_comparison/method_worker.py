"""Fresh-process worker used by the resumable method runner."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from .io_utils import atomic_write_json, sha256_file
from .schemas import CanonicalHuman
from .stage1_timing_campaign import assert_formal_campaign_ownership


def _pin_to_one_cpu() -> list[int]:
    """Pin formal timing to the first CPU allowed by the host cpuset."""

    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("Formal timing requires Linux CPU-affinity support")
    allowed = sorted(os.sched_getaffinity(0))
    if not allowed:
        raise RuntimeError("Formal timing process has an empty CPU affinity set")
    selected = [int(allowed[0])]
    os.sched_setaffinity(0, set(selected))
    if sorted(os.sched_getaffinity(0)) != selected:
        raise RuntimeError("Could not freeze the formal timing CPU affinity")
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--native-source")
    parser.add_argument("--output", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--seed", default="neutral")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--measured-runs", type=int, default=1)
    parser.add_argument("--timing-json", required=True)
    parser.add_argument("--capture-runtime-witness", action="store_true")
    parser.add_argument("--native-source-declared-sha256")
    parser.add_argument("--native-source-declared-fps", type=float)
    parser.add_argument("--native-source-declared-frames", type=int)
    return parser


def _canonical_content_hash(motion: object) -> tuple[str, str]:
    from .native_target_capture import tensor_sha256

    qpos_hash = tensor_sha256(np.asarray(motion.qpos, dtype=np.float64))
    contract = {
        "qpos_sha256": qpos_hash,
        "source_frame_idx_sha256": tensor_sha256(
            np.asarray(motion.source_frame_idx, dtype=np.int64)
        ),
        "valid_sha256": tensor_sha256(np.asarray(motion.valid, dtype=bool)),
        "fps_hex": float(motion.fps).hex(),
    }
    content = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return qpos_hash, content


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.repo_root).resolve()
    assert_formal_campaign_ownership(root)
    cpu_affinity = _pin_to_one_cpu()
    initialization_started = time.perf_counter()
    prepared = None
    if args.method in {"sparse", "dense"}:
        from .controlled_mink import ControlledMinkRetargeter

        prepared = ControlledMinkRetargeter(root, args.method, args.seed)
    elif args.method == "gmr":
        from .method_adapters import PreparedGMRRetargeter

        if not args.native_source:
            raise SystemExit("GMR requires --native-source BVH")
        prepared = PreparedGMRRetargeter(root, args.native_source)
    elif args.method == "protomotions_v2_3":
        from .protomotions_v2 import ProtoMotionsV2Retargeter

        prepared = ProtoMotionsV2Retargeter(root)
    elif args.method not in {"omniretarget", "holosoma"}:
        raise SystemExit(f"Unknown method: {args.method}")
    reusable_initialization_time_s = time.perf_counter() - initialization_started

    def execute(repetition: int, human: CanonicalHuman):
        if args.method in {"sparse", "dense"}:
            return prepared.run(human, max_frames=args.max_frames)
        if args.method == "gmr":
            motion = prepared.run(
                human,
                max_frames=args.max_frames,
                capture_runtime_witness=args.capture_runtime_witness,
            )
            actual_sha256 = sha256_file(args.native_source)
            declared_sha256 = args.native_source_declared_sha256
            declared_frames = args.native_source_declared_frames
            declared_fps = args.native_source_declared_fps
            if declared_sha256 is None or actual_sha256 != declared_sha256:
                raise RuntimeError("GMR cropped BVH actual/declared SHA-256 mismatch")
            if declared_frames is None or int(
                motion.metadata["native_source_parsed_frames"]
            ) != int(declared_frames):
                raise RuntimeError("GMR cropped BVH actual/declared frame mismatch")
            if len(human.timestamps) != int(declared_frames):
                raise RuntimeError("GMR canonical/native source frame-count mismatch")
            if declared_fps is None or not np.isclose(
                float(motion.metadata["native_source_parsed_fps"]),
                float(declared_fps),
                atol=1e-12,
                rtol=0.0,
            ):
                raise RuntimeError("GMR cropped BVH actual/declared fps mismatch")
            if not np.isclose(
                float(human.fps), float(declared_fps), atol=1e-12, rtol=0.0
            ):
                raise RuntimeError("GMR canonical/native source fps mismatch")
            motion.metadata["native_source_contract"] = {
                "path": str(Path(args.native_source).resolve()),
                "actual_sha256": actual_sha256,
                "declared_sha256": declared_sha256,
                "actual_frames": int(
                    motion.metadata["native_source_parsed_frames"]
                ),
                "declared_frames": int(declared_frames),
                "canonical_frames": int(len(human.timestamps)),
                "actual_fps": float(
                    motion.metadata["native_source_parsed_fps"]
                ),
                "canonical_fps": float(human.fps),
                "declared_fps": float(declared_fps),
                "timeline": (
                    "exact full cropped-BVH frame order; one native frame maps "
                    "to the same canonical source_frame_idx"
                ),
                "all_equal": True,
            }
            return motion
        if args.method in {"omniretarget", "holosoma"}:
            from .method_adapters import run_holosoma

            return run_holosoma(
                root,
                human,
                Path(args.work_dir) / f"repetition_{repetition:02d}",
                args.max_frames,
                capture_runtime_witness=args.capture_runtime_witness,
            )
        return prepared.run(
            human,
            max_frames=args.max_frames,
            capture_runtime_witness=args.capture_runtime_witness,
        )

    if args.warmup_runs < 0 or args.measured_runs < 1:
        raise SystemExit("warmup-runs must be nonnegative and measured-runs must be positive")
    repetitions = []
    selected = None
    selected_source_count = None
    total = args.warmup_runs + args.measured_runs
    measured_qpos_hashes: list[str] = []
    measured_content_hashes: list[str] = []
    for repetition in range(total):
        role = "warmup" if repetition < args.warmup_runs else "measured"
        started_at_utc = datetime.now(timezone.utc).isoformat()
        start = time.perf_counter()
        source_load_started = time.perf_counter()
        human = CanonicalHuman.load(args.source)
        source_load_time_s = time.perf_counter() - source_load_started
        source_count = len(human.timestamps)
        method_call_started = time.perf_counter()
        motion = execute(repetition, human)
        observed_method_call_wall_s = time.perf_counter() - method_call_started
        # The frozen end-to-end boundary ends when canonical G1 exists in
        # memory.  Persisting a timing witness is measured separately and is
        # deliberately excluded from steady-state RTF.
        observed_output_in_memory_wall_s = time.perf_counter() - start
        method_pipeline_s = float(
            motion.metadata.get(
                "steady_method_pipeline_s",
                motion.metadata.get(
                    "steady_end_to_end_total_s", observed_method_call_wall_s
                ),
            )
        )
        output_in_memory_wall_s = source_load_time_s + method_pipeline_s
        output_in_memory_at_utc = datetime.now(timezone.utc).isoformat()
        qpos_sha256, content_sha256 = _canonical_content_hash(motion)
        repetition_output = (
            Path(args.work_dir)
            / "timing_artifacts"
            / f"{role}_{repetition:02d}.canonical_g1.npz"
        )
        artifact_write_started = time.perf_counter()
        motion.save(repetition_output, source_frame_count=source_count)
        artifact_write_time_s = time.perf_counter() - artifact_write_started
        native_total_s = float(
            motion.metadata.get(
                "native_core_total_s",
                np.sum(motion.per_frame_solve_time_s),
            )
        )
        repetitions.append(
            {
                "index": repetition,
                "role": role,
                "started_at_utc": started_at_utc,
                "output_in_memory_at_utc": output_in_memory_at_utc,
                "wall_time_s": output_in_memory_wall_s,
                "observed_worker_output_in_memory_wall_s": float(
                    observed_output_in_memory_wall_s
                ),
                "canonical_source_load_time_s": float(source_load_time_s),
                "adapter_preprocess_solve_conversion_time_s": method_pipeline_s,
                "post_boundary_and_excluded_processing_time_s": max(
                    0.0,
                    observed_output_in_memory_wall_s
                    - source_load_time_s
                    - method_pipeline_s,
                ),
                "frame_count": int(len(motion.qpos)),
                "native_frame_times_s": motion.per_frame_solve_time_s.tolist(),
                "native_total_s": native_total_s,
                "native_median_frame_s": float(np.median(motion.per_frame_solve_time_s)),
                "steady_end_to_end_total_s": float(output_in_memory_wall_s),
                "initialization_time_s": float(
                    reusable_initialization_time_s
                    + motion.metadata.get("initialization_time_s", 0.0)
                ),
                "reusable_process_initialization_time_s": float(
                    reusable_initialization_time_s
                ),
                "timing_boundary": "canonical_source_file_to_canonical_g1_in_memory",
                "native_boundary": "native_input_ready_to_native_g1_in_memory",
                "canonical_artifact_write_time_s_excluded": float(
                    artifact_write_time_s
                ),
                "cpu_affinity": cpu_affinity,
                "thread_limit": 1,
                "timing_artifact_path": str(repetition_output),
                "timing_artifact_sha256": sha256_file(repetition_output),
                "canonical_qpos_sha256": qpos_sha256,
                "canonical_g1_content_sha256": content_sha256,
                "runtime_witness_enabled": bool(args.capture_runtime_witness),
            }
        )
        if role == "measured":
            selected = motion
            selected_source_count = source_count
            measured_qpos_hashes.append(qpos_sha256)
            measured_content_hashes.append(content_sha256)
    assert selected is not None
    if len(set(measured_qpos_hashes)) != 1 or len(set(measured_content_hashes)) != 1:
        raise RuntimeError(
            "Measured warm repetitions produced non-deterministic canonical G1 content"
        )
    selected.metadata["canonical_source_path"] = str(Path(args.source).resolve())
    selected.metadata["timing_protocol"] = {
        "warmup_runs": args.warmup_runs,
        "measured_runs": args.measured_runs,
        "measured_qpos_sha256": measured_qpos_hashes,
        "measured_content_sha256": measured_content_hashes,
        "all_measured_qpos_identical": True,
        "runtime_witness_enabled": bool(args.capture_runtime_witness),
    }
    assert selected_source_count is not None
    selected.save(args.output, source_frame_count=selected_source_count)
    timing_path = Path(args.timing_json)
    atomic_write_json(timing_path, {"repetitions": repetitions})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
