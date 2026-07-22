import json
import re
from pathlib import Path

from retargeting_comparison.constants import FULL_LAFAN_STOP_MESSAGE
from retargeting_comparison.interactive_report import (
    OPERATING_POINTS,
    build_interactive_report,
    collect_interactive_data,
)
from retargeting_comparison.io_utils import load_yaml, sha256_file


ROOT = Path(__file__).resolve().parents[1]


def test_interactive_data_is_complete_and_finite() -> None:
    data = collect_interactive_data(ROOT)

    assert [row["label"] for row in data["core"]] == list(OPERATING_POINTS)
    assert len(data["interaction"]) == 4
    assert data["decision"] == "NO-GO"
    assert data["hard_stop_message"] == FULL_LAFAN_STOP_MESSAGE
    required_names = {item["name"] for item in data["methods"]["required"]}
    assert {"protomotions_v2_3", "protomotions_v3"}.issubset(required_names)
    assert "37-motor" in data["methods"]["lineage_only"]["phc"]
    sensitivity = load_yaml(ROOT / "configs" / "scale_policy_sensitivity.yaml")
    assert len(sensitivity["within_method_variants"]) == 5
    assert sensitivity["within_method_protocol"]["apply_before_solver"] is True
    assert sensitivity["within_method_protocol"]["posthoc_qpos_rescale_forbidden"] is True
    for method in OPERATING_POINTS:
        series = data["frame_series"][method]
        assert len(series["source_frame_idx"]) == 600
        assert all(value == value for value in series["rf_kpe_all_m"])
    assert all(item["sha256"] == sha256_file(ROOT / item["path"]) for item in data["input_hashes"])


def test_standalone_report_build_is_deterministic(tmp_path) -> None:
    output = tmp_path / "interactive.html"
    build_interactive_report(ROOT, output)
    first_hash = sha256_file(output)
    build_interactive_report(ROOT, output)
    text = output.read_text(encoding="utf-8")

    assert sha256_file(output) == first_hash
    assert "__RTCMP_INLINE_" not in text
    assert FULL_LAFAN_STOP_MESSAGE in text
    assert "https://cdn" not in text
    assert not re.search(r"<(?:script|link)[^>]+(?:src|href)=[\"']https?://", text)
    for section_id in ("frontier", "frame-story", "sparse", "interaction", "timing", "budget", "evidence"):
        assert f'id="{section_id}"' in text

    match = re.search(
        r'<script id="rtcmp-data" type="application/json">(.*?)</script>',
        text,
        flags=re.DOTALL,
    )
    assert match is not None
    embedded = json.loads(match.group(1))
    assert embedded["pilot"]["selected_before_any_retarget_run"] is True
    assert embedded["stage2_projection"]["within_wall_budget"] is False
