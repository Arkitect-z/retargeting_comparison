"""Command-line entrypoint for the Stage 0–1 harness."""

from __future__ import annotations

import argparse

from .constants import FULL_LAFAN_STOP_MESSAGE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rtcmp")
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit", help="freeze first-party research and repository evidence")
    sub.add_parser("prepare-source", help="select and canonicalize the Pilot source")
    sub.add_parser("validate-models", help="validate external SMPL and SMPL-X assets")
    run_method = sub.add_parser("run-method", help="run one frozen retargeting method")
    run_method.add_argument("--method", required=True)
    run_method.add_argument("--sequence", required=True)
    evaluate = sub.add_parser("evaluate", help="evaluate one canonical run")
    evaluate.add_argument("--run", required=True)
    interaction = sub.add_parser("run-interaction", help="run an interaction ablation")
    interaction.add_argument("--case", choices=("box", "climb"), required=True)
    interaction.add_argument("--variant", choices=("full", "no-hard"), required=True)
    sub.add_parser("build-report", help="build Stage 1 figures and Markdown reports")
    sub.add_parser("validate-stage1", help="validate Stage 1 and its hard stop")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-stage1":
        print(FULL_LAFAN_STOP_MESSAGE)
        return 0
    raise SystemExit(f"Command implementation pending: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

