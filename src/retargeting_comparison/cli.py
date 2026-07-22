"""Command-line entrypoint for the Stage 0–1 harness."""

from __future__ import annotations

import argparse
from pathlib import Path

from .audit import generate_audit
from .constants import FULL_LAFAN_STOP_MESSAGE
from .io_utils import load_yaml
from .model_validation import validate_body_models
from .source import canonicalize_source, crop_bvh, select_pilot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rtcmp")
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit", help="freeze first-party research and repository evidence")
    audit.add_argument("--repo-root", default=".")
    prepare = sub.add_parser("prepare-source", help="select and canonicalize the Pilot source")
    prepare.add_argument("--lafan-root")
    prepare.add_argument("--config", default="configs/stage1.yaml")
    prepare.add_argument("--repo-root", default=".")
    models = sub.add_parser("validate-models", help="validate external SMPL and SMPL-X assets")
    models.add_argument("--body-models-root")
    models.add_argument("--config", default="configs/stage1.yaml")
    models.add_argument("--output", default="manifests/body_models.yaml")
    run_method = sub.add_parser("run-method", help="run one frozen retargeting method")
    run_method.add_argument(
        "--method", choices=("sparse", "dense", "gmr", "omniretarget", "holosoma"), required=True
    )
    run_method.add_argument("--sequence", required=True)
    run_method.add_argument("--seed", choices=("neutral", "A", "B"), default="neutral")
    run_method.add_argument("--max-frames", type=int)
    run_method.add_argument("--repo-root", default=".")
    evaluate = sub.add_parser("evaluate", help="evaluate one canonical run")
    evaluate.add_argument("--run", required=True)
    evaluate.add_argument("--source")
    evaluate.add_argument("--robot-xml")
    evaluate.add_argument("--output-dir", default="metrics/runs")
    interaction = sub.add_parser("run-interaction", help="run an interaction ablation")
    interaction.add_argument("--case", choices=("box", "climb"), required=True)
    interaction.add_argument("--variant", choices=("full", "no-hard"), required=True)
    sub.add_parser("build-report", help="build Stage 1 figures and Markdown reports")
    sub.add_parser("validate-stage1", help="validate Stage 1 and its hard stop")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        generate_audit(args.repo_root)
        return 0
    if args.command == "validate-models":
        config = load_yaml(args.config)
        body_root = args.body_models_root or config["body_models_root"]
        validate_body_models(body_root, args.output)
        return 0
    if args.command == "prepare-source":
        root = Path(args.repo_root).resolve()
        config = load_yaml(args.config)
        lafan_root = Path(args.lafan_root or config["lafan_root"])
        selected = select_pilot(
            lafan_root,
            root / "metrics" / "pilot_source_selection.csv",
            float(config["source_position_scale"]),
        )
        cropped = root / "data" / "pilot" / f"{selected.sequence_id}.bvh"
        crop_bvh(selected.source_file, cropped, selected.frame_start, selected.frame_end)
        output = root / "source" / "canonical_human" / f"{selected.sequence_id}.npz"
        canonicalize_source(
            cropped,
            output,
            root / "manifests" / "pilot_sequence.yaml",
            float(config["source_position_scale"]),
            origin_path=selected.source_file,
            origin_frame_start=selected.frame_start,
            origin_frame_end=selected.frame_end,
            repo_root=root,
        )
        return 0
    if args.command == "evaluate":
        from .evaluator import evaluate_motion, save_evaluation
        from .robot_model import CanonicalRobotModel, default_robot_scene
        from .schemas import CanonicalG1, CanonicalHuman

        run_path = Path(args.run)
        motion = CanonicalG1.load(run_path)
        source_path = args.source or motion.metadata.get("canonical_source_path")
        if not source_path:
            raise SystemExit("--source is required when the run metadata has no canonical_source_path")
        human = CanonicalHuman.load(source_path)
        robot = CanonicalRobotModel(args.robot_xml or default_robot_scene())
        table, summary = evaluate_motion(human, motion, robot)
        save_evaluation(table, summary, args.output_dir, run_path.stem)
        return 0
    if args.command == "run-method":
        from .runner import run_method

        manifest = run_method(
            args.method,
            args.sequence,
            repo_root=args.repo_root,
            seed=args.seed,
            max_frames=args.max_frames,
        )
        print(f"{manifest.run_id}: {manifest.status.value}")
        from .schemas import RunStatus

        return 0 if manifest.status in {RunStatus.SUCCEEDED, RunStatus.INCOMPLETE} else 1
    if args.command == "validate-stage1":
        print(FULL_LAFAN_STOP_MESSAGE)
        return 0
    raise SystemExit(f"Command implementation pending: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
