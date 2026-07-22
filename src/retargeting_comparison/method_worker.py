"""Fresh-process worker used by the resumable method runner."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .io_utils import atomic_write_json
from .schemas import CanonicalHuman


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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.repo_root).resolve()
    human = CanonicalHuman.load(args.source)

    def execute(repetition: int):
        if args.method in {"sparse", "dense"}:
            from .controlled_mink import ControlledMinkRetargeter

            return ControlledMinkRetargeter(root, args.method, args.seed).run(
                human, max_frames=args.max_frames
            )
        if args.method == "gmr":
            from .method_adapters import run_gmr

            if not args.native_source:
                raise SystemExit("GMR requires --native-source BVH")
            return run_gmr(root, args.native_source, args.source, args.max_frames)
        if args.method in {"omniretarget", "holosoma"}:
            from .method_adapters import run_holosoma

            return run_holosoma(
                root,
                args.source,
                Path(args.work_dir) / f"repetition_{repetition:02d}",
                args.max_frames,
            )
        raise SystemExit(f"Unknown method: {args.method}")

    if args.warmup_runs < 0 or args.measured_runs < 1:
        raise SystemExit("warmup-runs must be nonnegative and measured-runs must be positive")
    repetitions = []
    selected = None
    total = args.warmup_runs + args.measured_runs
    for repetition in range(total):
        role = "warmup" if repetition < args.warmup_runs else "measured"
        start = time.perf_counter()
        motion = execute(repetition)
        wall = time.perf_counter() - start
        repetitions.append(
            {
                "index": repetition,
                "role": role,
                "wall_time_s": wall,
                "frame_count": int(len(motion.qpos)),
                "native_frame_times_s": motion.per_frame_solve_time_s.tolist(),
                "native_total_s": float(motion.per_frame_solve_time_s.sum()),
                "native_median_frame_s": float(np.median(motion.per_frame_solve_time_s)),
            }
        )
        if role == "measured":
            selected = motion
    assert selected is not None
    selected.metadata["canonical_source_path"] = str(Path(args.source).resolve())
    selected.metadata["timing_protocol"] = {
        "warmup_runs": args.warmup_runs,
        "measured_runs": args.measured_runs,
    }
    source_count = len(human.timestamps)
    selected.save(args.output, source_frame_count=source_count)
    timing_path = Path(args.timing_json)
    atomic_write_json(timing_path, {"repetitions": repetitions})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
