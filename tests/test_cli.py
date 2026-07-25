from retargeting_comparison.cli import build_parser
from retargeting_comparison.constants import FULL_LAFAN_STOP_MESSAGE


def test_cli_exposes_stage1_commands() -> None:
    parser = build_parser()
    for command in (
        "audit",
        "prepare-source",
        "prepare-smpl-skin",
        "prepare-phc-visualization",
        "validate-models",
        "run-method",
        "evaluate",
        "run-interaction",
        "build-report",
        "build-interactive-report",
        "visualize-results",
        "visualize-current-results",
        "visualize-phc",
        "run-stage1-timing",
        "validate-stage1",
    ):
        assert command in parser.format_help()


def test_hard_stop_message_is_exact() -> None:
    assert FULL_LAFAN_STOP_MESSAGE == (
        "FULL-LAFAN EXPERIMENTS NOT STARTED — WAITING FOR USER APPROVAL."
    )
