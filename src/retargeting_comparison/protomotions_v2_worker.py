"""Isolated command-line worker for the frozen ProtoMotions v2.3 adapter."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .io_utils import atomic_write_json, sha256_file
from .protomotions_v2 import ProtoMotionsV2Retargeter, audit_robot_compatibility
from .schemas import CanonicalHuman


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--allow-dependency-drift",
        action="store_true",
        help="Record but do not reject versions differing from the frozen capture environment.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.repo_root.resolve()
    source_path = args.source if args.source.is_absolute() else root / args.source
    output_path = args.output if args.output.is_absolute() else root / args.output
    audit_path = (
        args.audit_output
        if args.audit_output is None or args.audit_output.is_absolute()
        else root / args.audit_output
    )
    started = time.perf_counter()
    source = CanonicalHuman.load(source_path)
    retargeter = ProtoMotionsV2Retargeter(
        root, strict_dependencies=not args.allow_dependency_drift
    )
    result = retargeter.run(source, max_frames=args.max_frames)
    result.save(output_path, source_frame_count=len(source.timestamps))
    if audit_path is not None:
        atomic_write_json(audit_path, audit_robot_compatibility(root))
    summary = {
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "frames": len(result.qpos),
        "source_frames": len(source.timestamps),
        "completion_status": result.metadata["completion_status"],
        "wall_time_s": time.perf_counter() - started,
        "steady_solve_total_s": float(result.per_frame_solve_time_s.sum()),
        "steady_solve_median_s_per_frame": float(
            np.median(result.per_frame_solve_time_s)
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
