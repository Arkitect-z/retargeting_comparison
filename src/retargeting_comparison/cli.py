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
    evaluator = sub.add_parser(
        "freeze-evaluator",
        help="freeze the method-independent Stage 1 scale and heading protocol",
    )
    evaluator.add_argument("--repo-root", default=".")
    evaluator.add_argument("--sequence", default="manifests/pilot_sequence.yaml")
    evaluator.add_argument("--output", default="manifests/evaluator.yaml")
    candidates = sub.add_parser(
        "gate-candidates",
        help="run bounded integration-readiness gates for conditional candidates",
    )
    candidates.add_argument("--repo-root", default=".")
    adapters = sub.add_parser(
        "audit-source-adapters",
        help="audit core native source conversions against canonical LAFAN",
    )
    adapters.add_argument("--repo-root", default=".")
    skin = sub.add_parser(
        "prepare-smpl-skin",
        help="fit a visualization-only neutral SMPL skin to canonical LAFAN BVH",
    )
    skin.add_argument("--repo-root", default=".")
    skin.add_argument("--sequence", default="manifests/pilot_sequence.yaml")
    skin.add_argument("--output")
    skin.add_argument("--evidence")
    skin.add_argument("--iterations", type=int, default=240)
    skin.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    phc_prepare = sub.add_parser(
        "prepare-phc-visualization",
        help="run the frozen public PHC SMPL-to-G1 fitting path for visualization",
    )
    phc_prepare.add_argument("--repo-root", default=".")
    phc_prepare.add_argument("--sequence", default="manifests/pilot_sequence.yaml")
    phc_prepare.add_argument("--fitting-iterations", type=int, default=500)
    phc_prepare.add_argument("--force", action="store_true")
    phc_visualize = sub.add_parser(
        "visualize-phc",
        help="build the PHC original/scaled-human/G1 Rerun recording",
    )
    phc_visualize.add_argument("--repo-root", default=".")
    phc_visualize.add_argument(
        "--output", default="artifacts/visualization/phc_scale_three_way.rrd"
    )
    phc_visualize.add_argument(
        "--manifest", default="manifests/phc_rerun_visualization.json"
    )
    phc_visualize.add_argument("--spawn", action="store_true")
    phc_visualize.add_argument("--max-frames", type=int)
    run_method = sub.add_parser("run-method", help="run one frozen retargeting method")
    run_method.add_argument(
        "--method",
        choices=("sparse", "dense", "gmr", "omniretarget", "holosoma", "protomotions_v2_3", "protomotions_v3"),
        required=True,
    )
    run_method.add_argument("--sequence", required=True)
    run_method.add_argument("--seed", choices=("neutral", "A", "B"), default="neutral")
    run_method.add_argument("--max-frames", type=int)
    run_method.add_argument("--repo-root", default=".")
    run_method.add_argument("--refresh-timing", action="store_true")
    run_method.add_argument(
        "--revision",
        help="immutable run suffix used when a protocol/config revision must preserve prior evidence",
    )
    evaluate = sub.add_parser("evaluate", help="evaluate one canonical run")
    evaluate.add_argument("--run", required=True)
    evaluate.add_argument("--source")
    evaluate.add_argument("--robot-xml")
    evaluate.add_argument("--evaluator", default="manifests/evaluator.yaml")
    evaluate.add_argument("--output-dir", default="metrics/runs")
    interaction = sub.add_parser("run-interaction", help="run an interaction ablation")
    interaction.add_argument("--case", choices=("box", "climb"), required=True)
    interaction.add_argument("--variant", choices=("full", "no-hard"), required=True)
    interaction.add_argument("--repo-root", default=".")
    report = sub.add_parser("build-report", help="build Stage 1 figures and Markdown reports")
    report.add_argument("--repo-root", default=".")
    interactive = sub.add_parser(
        "build-interactive-report",
        help="build the standalone interactive Stage 1 research report",
    )
    interactive.add_argument("--repo-root", default=".")
    interactive.add_argument("--output", default="INTERACTIVE_REPORT.html")
    visualization = sub.add_parser(
        "visualize-results",
        help="visualize all Stage 1 canonical results with Rerun",
    )
    visualization.add_argument("--repo-root", default=".")
    visualization.add_argument("--sequence", default="manifests/pilot_sequence.yaml")
    visualization.add_argument(
        "--output", default="artifacts/visualization/stage1_comparison.rrd"
    )
    visualization.add_argument(
        "--manifest", default="manifests/rerun_visualization.json"
    )
    visualization.add_argument("--spawn", action="store_true")
    visualization.add_argument("--max-frames", type=int)
    visualization.add_argument("--stride", type=int, default=1)
    diagnostic_visualization = sub.add_parser(
        "visualize-current-results",
        help="rebuild the frozen current nine-method diagnostic with native BVH keypoints",
    )
    diagnostic_visualization.add_argument("--repo-root", default=".")
    diagnostic_visualization.add_argument(
        "--snapshot-manifest",
        default=(
            "artifacts/visualization/"
            "stage1_current_completed_9methods.manifest.json"
        ),
    )
    diagnostic_visualization.add_argument(
        "--output",
        default=(
            "artifacts/visualization/"
            "stage1_current_completed_9methods_bvh.rrd"
        ),
    )
    diagnostic_visualization.add_argument(
        "--manifest",
        default=(
            "artifacts/visualization/"
            "stage1_current_completed_9methods_bvh.manifest.json"
        ),
    )
    diagnostic_visualization.add_argument("--spawn", action="store_true")
    diagnostic_visualization.add_argument("--max-frames", type=int)
    diagnostic_visualization.add_argument("--stride", type=int, default=1)
    scale = sub.add_parser(
        "run-scale-sensitivity",
        help="run the registered pre-solver scale-policy experiments",
    )
    scale.add_argument("--repo-root", default=".")
    scale.add_argument(
        "--phase",
        choices=(
            "targets",
            "controlled",
            "contacts",
            "shape",
            "native",
            "native-contact",
            "summarize",
            "all",
        ),
        default="all",
    )
    scale.add_argument(
        "--method",
        choices=("gmr", "omniretarget", "protomotions_v2_3", "protomotions_v3"),
        action="append",
    )
    scale.add_argument(
        "--variant",
        choices=("native", "root_minus_5", "root_plus_5", "local_minus_5", "local_plus_5"),
        action="append",
        help="limit the native-response phase to one or more registered variants",
    )
    validate = sub.add_parser("validate-stage1", help="validate Stage 1 and its hard stop")
    validate.add_argument("--repo-root", default=".")
    timing = sub.add_parser(
        "run-stage1-timing",
        help="preview or execute the frozen sequential Stage 1 timing campaign",
    )
    timing.add_argument("--repo-root", default=".")
    timing.add_argument("--config", default="configs/stage1_timing_campaign.yaml")
    timing.add_argument("--state", default="manifests/stage1_timing_campaign.json")
    timing.add_argument("--launch", action="store_true")
    stage2_plan = sub.add_parser(
        "plan-stage2", help="build the immutable budget-gated Stage 2 plan"
    )
    stage2_plan.add_argument("--repo-root", default=".")
    stage2_plan.add_argument("--config", default="configs/stage2.yaml")
    stage2_plan.add_argument("--output")
    stage2_run = sub.add_parser(
        "run-stage2", help="preview or execute an exact acknowledged Stage 2 plan"
    )
    stage2_run.add_argument("--repo-root", default=".")
    stage2_run.add_argument("--config", default="configs/stage2.yaml")
    stage2_run.add_argument("--plan", required=True)
    stage2_run.add_argument("--plan-sha256", required=True)
    stage2_run.add_argument("--launch", action="store_true")
    stage2_report = sub.add_parser(
        "build-stage2-report", help="aggregate succeeded Stage 2 jobs and publish evidence"
    )
    stage2_report.add_argument("--repo-root", default=".")
    stage2_report.add_argument("--plan", required=True)
    stage2_validate = sub.add_parser(
        "validate-stage2", help="validate tracked Stage 2 evidence without launching jobs"
    )
    stage2_validate.add_argument("--repo-root", default=".")
    stage2_validate.add_argument("--plan", required=True)
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
    if args.command == "freeze-evaluator":
        from .calibration import freeze_evaluator_protocol
        from .robot_model import CanonicalRobotModel, default_robot_scene
        from .schemas import CanonicalHuman

        root = Path(args.repo_root).resolve()
        sequence_path = Path(args.sequence)
        if not sequence_path.is_absolute():
            sequence_path = root / sequence_path
        sequence = load_yaml(sequence_path)
        human = CanonicalHuman.load(root / sequence["canonical_path"])
        robot = CanonicalRobotModel(default_robot_scene(root))
        output = Path(args.output)
        if not output.is_absolute():
            output = root / output
        freeze_evaluator_protocol(
            human,
            robot,
            output,
            source_path=sequence["canonical_path"],
        )
        print(output)
        return 0
    if args.command == "gate-candidates":
        from .candidate_gate import gate_candidates

        rows = gate_candidates(args.repo_root)
        for row in rows:
            print(f"{row['candidate']}: {row['status']} — {row['reason']}")
        return 0
    if args.command == "audit-source-adapters":
        from .source_adapter_audit import audit_source_adapters

        rows = audit_source_adapters(args.repo_root)
        for row in rows:
            print(f"{row['adapter']}: {row['status']}")
        return 0
    if args.command == "prepare-smpl-skin":
        from .smpl_skinning import fit_lafan_to_smpl_skin

        result = fit_lafan_to_smpl_skin(
            args.repo_root,
            sequence_manifest=args.sequence,
            output=args.output,
            evidence=args.evidence,
            iterations=args.iterations,
            device=args.device,
        )
        print(result["output"])
        return 0
    if args.command == "prepare-phc-visualization":
        from .phc_visualization import prepare_phc_visualization

        result = prepare_phc_visualization(
            args.repo_root,
            sequence_manifest=args.sequence,
            fitting_iterations=args.fitting_iterations,
            force=args.force,
        )
        print(result["prepared_motion"]["path"])
        return 0
    if args.command == "visualize-phc":
        from .phc_visualization import write_phc_rerun_visualization

        result = write_phc_rerun_visualization(
            args.repo_root,
            output=args.output,
            manifest=args.manifest,
            spawn=args.spawn,
            max_frames=args.max_frames,
        )
        print(result["output"])
        return 0
    if args.command == "evaluate":
        from .calibration import load_evaluator_protocol
        from .evaluator import evaluate_motion, save_evaluation
        from .io_utils import sha256_file
        from .robot_model import CanonicalRobotModel, default_robot_scene
        from .schemas import CanonicalG1, CanonicalHuman

        run_path = Path(args.run)
        motion = CanonicalG1.load(run_path)
        source_path = args.source or motion.metadata.get("canonical_source_path")
        if not source_path:
            raise SystemExit("--source is required when the run metadata has no canonical_source_path")
        human = CanonicalHuman.load(source_path)
        robot = CanonicalRobotModel(args.robot_xml or default_robot_scene())
        protocol_path = Path(args.evaluator)
        if not protocol_path.is_absolute():
            protocol_path = Path.cwd() / protocol_path
        protocol = load_evaluator_protocol(protocol_path)
        protocol["manifest_sha256"] = sha256_file(protocol_path)
        table, summary = evaluate_motion(human, motion, robot, protocol)
        run_id = str(motion.metadata.get("method") or run_path.parent.name).replace("/", "-")
        save_evaluation(table, summary, args.output_dir, run_id)
        return 0
    if args.command == "run-method":
        from .runner import refresh_method_timing, run_method

        if args.refresh_timing:
            path = refresh_method_timing(
                args.method,
                args.sequence,
                repo_root=args.repo_root,
                seed=args.seed,
                revision=args.revision,
            )
            print(path)
            return 0

        manifest = run_method(
            args.method,
            args.sequence,
            repo_root=args.repo_root,
            seed=args.seed,
            max_frames=args.max_frames,
            revision=args.revision,
        )
        print(f"{manifest.run_id}: {manifest.status.value}")
        from .schemas import RunStatus

        return 0 if manifest.status in {RunStatus.SUCCEEDED, RunStatus.INCOMPLETE} else 1
    if args.command == "run-interaction":
        from .interaction import run_interaction
        from .schemas import RunStatus

        manifest = run_interaction(args.case, args.variant, args.repo_root)
        print(f"{manifest.run_id}: {manifest.status.value}")
        return 0 if manifest.status == RunStatus.SUCCEEDED else 1
    if args.command == "validate-stage1":
        from .validation import validate_stage1

        result = validate_stage1(args.repo_root)
        print(result["decision"])
        print(FULL_LAFAN_STOP_MESSAGE)
        return 0 if result["decision"] in {"GO", "GO WITH CHANGES"} else 1
    if args.command == "run-stage1-timing":
        from .stage1_timing_campaign import run_stage1_timing_campaign

        result = run_stage1_timing_campaign(
            args.repo_root,
            args.config,
            args.state,
            launch=args.launch,
        )
        print(result["status"])
        return 0 if result["status"] in {"preview", "succeeded"} else 1
    if args.command == "build-report":
        from .reporting import build_report

        build_report(args.repo_root)
        return 0
    if args.command == "build-interactive-report":
        from .interactive_report import build_interactive_report

        print(build_interactive_report(args.repo_root, args.output))
        return 0
    if args.command == "visualize-results":
        from .rerun_visualization import visualize_stage1

        result = visualize_stage1(
            repo_root=args.repo_root,
            sequence_manifest=args.sequence,
            output=args.output,
            manifest=args.manifest,
            spawn=args.spawn,
            max_frames=args.max_frames,
            stride=args.stride,
        )
        print(result["output"] or "Rerun viewer spawned")
        return 0
    if args.command == "visualize-current-results":
        from .rerun_visualization import visualize_current_diagnostic

        result = visualize_current_diagnostic(
            repo_root=args.repo_root,
            snapshot_manifest=args.snapshot_manifest,
            output=args.output,
            manifest=args.manifest,
            spawn=args.spawn,
            max_frames=args.max_frames,
            stride=args.stride,
        )
        print(result["output"] or "Rerun viewer spawned")
        return 0
    if args.command == "run-scale-sensitivity":
        # Native workers deliberately avoid importing the pandas/Mink/report
        # stack.  This branch must remain above the rich analysis-module
        # import so the exact command is runnable inside clean method envs.
        if args.phase == "native":
            from .scale_worker import run_native_response

            for method in args.method or [
                "gmr",
                "omniretarget",
                "protomotions_v2_3",
                "protomotions_v3",
            ]:
                print(
                    "\n".join(
                        map(
                            str,
                            run_native_response(
                                method, args.repo_root, args.variant
                            ),
                        )
                    )
                )
            return 0
        if args.phase == "native-contact":
            from .scale_worker import run_holosoma_native_contact_robustness

            methods = args.method or ["omniretarget"]
            if methods != ["omniretarget"] or args.variant:
                raise SystemExit(
                    "native-contact is the registered five-row Holosoma arm; "
                    "use --method omniretarget and do not select variants"
                )
            print(
                "\n".join(
                    map(
                        str,
                        run_holosoma_native_contact_robustness(args.repo_root),
                    )
                )
            )
            return 0
        from .scale_sensitivity import (
            build_actor_shape_probe,
            build_contact_flip_diagnostics,
            build_pre_solver_target_contract,
            run_controlled_policy_transplants,
            run_scale_stage,
            summarize_scale_sensitivity,
        )

        if args.phase == "targets":
            print(len(build_pre_solver_target_contract(args.repo_root)))
        elif args.phase == "controlled":
            print("\n".join(map(str, run_controlled_policy_transplants(args.repo_root))))
        elif args.phase == "contacts":
            print(len(build_contact_flip_diagnostics(args.repo_root)))
        elif args.phase == "shape":
            print(len(build_actor_shape_probe(args.repo_root)))
        elif args.phase == "summarize":
            print(len(summarize_scale_sensitivity(args.repo_root)))
        else:
            print(run_scale_stage(args.repo_root, args.method))
        return 0
    if args.command == "plan-stage2":
        from .stage2 import build_stage2_plan

        plan = build_stage2_plan(
            args.repo_root,
            args.config,
            output_path=args.output,
        )
        print(plan["design_id"])
        print(plan["plan_sha256"])
        return 0
    if args.command == "run-stage2":
        from .stage2 import execute_stage2

        result = execute_stage2(
            args.repo_root,
            args.plan,
            config_path=args.config,
            expected_plan_sha256=args.plan_sha256,
            launch=args.launch,
        )
        print(result["status"])
        return 0 if result["status"] in {"preview", "complete"} else 1
    if args.command == "build-stage2-report":
        from .stage2_analysis import build_stage2_analysis

        result = build_stage2_analysis(args.repo_root, args.plan)
        print(result["status"])
        return 0 if result["status"] == "complete" else 1
    if args.command == "validate-stage2":
        from .stage2_analysis import validate_stage2

        result = validate_stage2(args.repo_root, args.plan)
        print(result["decision"])
        return 0 if result["decision"] == "GO" else 1
    raise SystemExit(f"Command implementation pending: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
