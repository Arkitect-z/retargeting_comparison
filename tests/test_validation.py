import json
from pathlib import Path

import numpy as np
import pytest

from retargeting_comparison.constants import FULL_LAFAN_STOP_MESSAGE
from retargeting_comparison.io_utils import sha256_file
from retargeting_comparison.reporting import REPORTS
from retargeting_comparison.schemas import CanonicalG1
from retargeting_comparison.stage1_publication import (
    PUBLICATION_EVIDENCE_SCHEMA_VERSION,
    VALIDATION_SCHEMA_VERSION,
    _canonical_sha256,
    resolve_bound_stage1_validation,
)
from retargeting_comparison.validation import _core_motion_check, validate_stage1
import retargeting_comparison.validation as validation_module
import retargeting_comparison.reporting as reporting_module
import retargeting_comparison.stage1_publication as publication_module
import retargeting_comparison.interactive_report as interactive_module


def test_delivery_contract_lists_all_required_markdown() -> None:
    assert len(REPORTS) == 11
    assert "PRESENTATION.md" in REPORTS
    assert "SCALE_POLICY_SENSITIVITY.md" in REPORTS
    assert "UNITREE_REFERENCE_COMPARISON.md" in REPORTS
    assert "STAGE1_REVIEW.md" in REPORTS
    assert FULL_LAFAN_STOP_MESSAGE.endswith("USER APPROVAL.")


def test_core_motion_check_returns_json_native_bool() -> None:
    qpos = np.zeros((2, 36), dtype=np.float64)
    qpos[:, 3] = 1.0
    motion = CanonicalG1(
        qpos=qpos,
        fps=30.0,
        source_frame_idx=np.arange(2),
        valid=np.ones(2, dtype=bool),
        per_frame_solve_time_s=np.zeros(2),
        metadata={"completion_status": "succeeded"},
    )

    result = _core_motion_check(motion, source_frame_count=2)

    assert result is True
    assert type(result) is bool


def _minimal_pending_publication(root: Path) -> None:
    (root / "manifests").mkdir(parents=True)
    (root / "metrics").mkdir()
    source = root / "metrics" / "evidence.csv"
    source.write_text("value\n1\n", encoding="utf-8")
    row = {
        "path": "metrics/evidence.csv",
        "sha256": sha256_file(source),
        "bytes": source.stat().st_size,
    }
    manifest = {
        "schema_version": PUBLICATION_EVIDENCE_SCHEMA_VERSION,
        "decision": "PENDING",
        "state": "evidence_ready_for_independent_validation",
        "decision_authority": "manifests/stage1_validation.json",
        "input_hashes": [row],
        "input_bundle_sha256": _canonical_sha256([row]),
        "outputs": [row],
        "output_bundle_sha256": _canonical_sha256([row]),
    }
    (root / "manifests" / "stage1_publication.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )


@pytest.mark.parametrize(
    ("full_lafan_authorized", "expected"),
    [(False, "GO"), (True, "NO-GO")],
)
def test_validator_issues_only_hash_bound_authoritative_decisions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    full_lafan_authorized: bool,
    expected: str,
) -> None:
    _minimal_pending_publication(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "runs" / "stage2").mkdir(parents=True)
    (tmp_path / "configs" / "stage1.yaml").write_text(
        f"full_lafan_authorized: {str(full_lafan_authorized).lower()}\n",
        encoding="utf-8",
    )
    (tmp_path / "manifests" / "pilot_sequence.yaml").write_text(
        "sequence_id: pilot\nfull_lafan_authorized: false\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(validation_module, "_attempt", lambda check: True)
    monkeypatch.setattr(
        validation_module,
        "_stage2_projection",
        lambda root: (True, {"within_budget": True}),
    )
    monkeypatch.setattr(validation_module, "_write_artifact_manifest", lambda root: None)

    result = validate_stage1(tmp_path)

    assert result["schema_version"] == VALIDATION_SCHEMA_VERSION
    assert result["authority"] == "independent_fail_closed_stage1_validator"
    assert result["decision"] == expected
    assert result["publication_evidence_binding"]["input_bundle_sha256"]
    assert result["decision_basis_sha256"]
    resolved = resolve_bound_stage1_validation(tmp_path)
    assert resolved["decision"] == expected
    assert resolved["binding_status"] == "verified"


def test_report_orchestration_has_one_acyclic_validation_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        publication_module,
        "build_stage1_publication",
        lambda root: calls.append("build-pending"),
    )
    monkeypatch.setattr(
        publication_module,
        "finalize_stage1_publication",
        lambda root: calls.append("finalize-from-validation"),
    )
    monkeypatch.setattr(
        interactive_module,
        "build_interactive_report",
        lambda root, force_pending=False: calls.append(
            "interactive-pending" if force_pending else "interactive-final"
        ),
    )
    monkeypatch.setattr(
        validation_module,
        "validate_stage1",
        lambda root: calls.append("validate-once"),
    )
    monkeypatch.setattr(
        reporting_module,
        "artifact_manifest",
        lambda root: calls.append("artifact-manifest"),
    )

    reporting_module.build_report(tmp_path)

    assert calls == [
        "build-pending",
        "interactive-pending",
        "validate-once",
        "finalize-from-validation",
        "interactive-final",
        "artifact-manifest",
    ]
