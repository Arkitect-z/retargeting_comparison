import json
import hashlib
import os
import re
from pathlib import Path

import pytest

from retargeting_comparison.constants import FULL_LAFAN_STOP_MESSAGE
from retargeting_comparison.interactive_report import (
    OPERATING_POINTS,
    _apply_decision_copy,
    audit_interactive_delivery,
    build_interactive_report,
    collect_interactive_data,
    validate_interactive_delivery_manifest,
)
from retargeting_comparison.io_utils import load_yaml, sha256_file


ROOT = Path(__file__).resolve().parents[1]
FINAL_MODE = os.environ.get("RTCMP_FINAL_TESTS") == "1"
requires_built_stage1 = pytest.mark.skipif(
    not FINAL_MODE and not (ROOT / "metrics" / "stage1_core_summary.csv").is_file(),
    reason="interactive integration tests require a built Stage 1 evidence package",
)


@requires_built_stage1
def test_interactive_data_is_complete_and_finite() -> None:
    data = collect_interactive_data(ROOT)

    assert [row["label"] for row in data["core"]] == list(OPERATING_POINTS)
    assert [row["label"] for row in data["reference"]] == ["unitree-reference"]
    assert len(data["interaction"]) == 4
    assert data["decision"] in {"PENDING", "GO", "NO-GO"}
    assert data["decision_authority"]["path"] == "manifests/stage1_validation.json"
    if data["decision"] != "PENDING":
        assert data["decision_authority"]["binding_status"] == "verified"
    assert data["hard_stop_message"] == FULL_LAFAN_STOP_MESSAGE
    required_names = {item["name"] for item in data["methods"]["required"]}
    assert {"protomotions_v2_3", "protomotions_v3"}.issubset(required_names)
    assert "37-motor" in data["methods"]["lineage_only"]["phc"]
    sensitivity = load_yaml(ROOT / "configs" / "scale_policy_sensitivity.yaml")
    assert len(sensitivity["within_method_variants"]) == 5
    assert sensitivity["within_method_protocol"]["apply_before_solver"] is True
    assert sensitivity["within_method_protocol"]["posthoc_qpos_rescale_forbidden"] is True
    assert data["schema_version"] == 5
    assert len(data["scale_summary"]) >= 30
    assert data["reference_provenance"]
    direct = [
        row
        for row in data["reference_comparison"]
        if row.get("comparison_type")
        == "direct_same_g1_frame_index_aligned_trajectory_disagreement"
    ]
    assert len(direct) == len(OPERATING_POINTS) * 11
    assert {
        row["timeline_alignment"] for row in direct
    } == {"frame_index_only_not_exact_timestamp"}
    for method in (*OPERATING_POINTS, "unitree-reference"):
        series = data["frame_series"][method]
        assert len(series["source_frame_idx"]) == 450
        assert all(value == value for value in series["rf_kpe_all_m"])
    assert all(item["sha256"] == sha256_file(ROOT / item["path"]) for item in data["input_hashes"])
    expected_bundle = hashlib.sha256(
        json.dumps(
            data["input_hashes"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    assert data["evidence_bundle_sha256"] == expected_bundle


@requires_built_stage1
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
    for section_id in ("frontier", "scale", "reference", "frame-story", "sparse", "interaction", "timing", "budget", "evidence"):
        assert f'id="{section_id}"' in text

    match = re.search(
        r'<script id="rtcmp-data" type="application/json">(.*?)</script>',
        text,
        flags=re.DOTALL,
    )
    assert match is not None
    embedded = json.loads(match.group(1))
    assert embedded["pilot"]["selected_before_any_retarget_run"] is True
    assert embedded["stage2_projection"].get("within_budget", embedded["stage2_projection"].get("within_wall_budget")) is True
    audit = audit_interactive_delivery(ROOT, output)
    assert audit["result"] == "verified"
    assert audit["embedded_data_sha256"] in text
    assert validate_interactive_delivery_manifest(ROOT, output) == audit


@requires_built_stage1
def test_bootstrap_interactive_report_is_explicitly_pending(tmp_path) -> None:
    output = tmp_path / "interactive-pending.html"
    build_interactive_report(ROOT, output, force_pending=True)
    match = re.search(
        r'<script id="rtcmp-data" type="application/json">(.*?)</script>',
        output.read_text(encoding="utf-8"),
        flags=re.DOTALL,
    )
    assert match is not None
    embedded = json.loads(match.group(1))
    assert embedded["decision"] == "PENDING"
    assert embedded["decision_authority"]["binding_status"] == "bootstrap_pending"
    assert embedded["validation"]["provisional"] is True


@pytest.mark.parametrize(
    ("decision", "expected"),
    [
        ("GO", "hash-bound Stage 1 evidence supports a Pilot-level GO"),
        ("NO-GO", "does not authorize a GO"),
        ("PENDING", "publication builder cannot issue GO"),
    ],
)
def test_static_interactive_go_copy_is_replaced_by_bound_decision(
    decision: str, expected: str
) -> None:
    template = (
        "<p>Six complete operating points, an explicitly delimited external reference, "
        "exact pre-solver target capture, causal scale interventions, repeated timing, "
        "and two interaction cases support a Pilot-level GO. Dataset-level ranking "
        "remains deliberately deferred to the budget-gated Stage 2 run.</p>"
    )
    javascript = (
        'x : decision === "GO WITH CHANGES" ? "WITH CHANGES" : "GATE CLOSED";\n'
        'y : "The frozen validation manifest contains at least one unmet Stage 1 '
        'acceptance condition.";'
    )

    rendered, script = _apply_decision_copy(template, javascript, decision)

    assert expected in rendered
    assert "AWAITING VALIDATION" in script
    assert "Independent Stage 1 validation has not yet bound" in script


@requires_built_stage1
def test_embedded_data_tamper_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "interactive-pending.html"
    build_interactive_report(ROOT, output, force_pending=True)
    text = output.read_text(encoding="utf-8")
    text = text.replace('"schema_version":5', '"schema_version":4', 1)
    output.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="modified after rendering"):
        audit_interactive_delivery(ROOT, output, require_rerun=False)


@requires_built_stage1
def test_evidence_hash_tamper_survives_meta_rehash_but_is_rejected(
    tmp_path: Path,
) -> None:
    output = tmp_path / "interactive-pending.html"
    build_interactive_report(ROOT, output, force_pending=True)
    text = output.read_text(encoding="utf-8")
    match = re.search(
        r'<script id="rtcmp-data" type="application/json">(.*?)</script>',
        text,
        flags=re.DOTALL,
    )
    assert match is not None
    data = json.loads(match.group(1))
    data["input_hashes"][0]["sha256"] = "0" * 64
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    text = text[: match.start(1)] + encoded + text[match.end(1) :]
    text = re.sub(
        r'(<meta name="rtcmp-data-sha256" content=")[0-9a-f]{64}(\">)',
        rf"\g<1>{digest}\g<2>",
        text,
        count=1,
    )
    output.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="evidence hash/size mismatch"):
        audit_interactive_delivery(ROOT, output, require_rerun=False)


def test_timeline_reserves_nine_rows_and_reads_all_trajectories() -> None:
    script = (ROOT / "interactive_report/report.js").read_text(encoding="utf-8")
    css = (ROOT / "interactive_report/report.css").read_text(encoding="utf-8")
    assert "const margin = { top: 25, right: 28, bottom: 150, left: 65 };" in script
    assert "TRAJECTORIES.filter((method) => timelineState.active.has(method))" in script
    assert "METHODS.filter((method) => timelineState.active.has(method))" not in script
    assert ".timeline-stage { min-height: 560px;" in css
    assert '["TIMELINE ALIGNMENT", "Frame index only"' in script
