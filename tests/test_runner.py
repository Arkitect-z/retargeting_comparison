from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from retargeting_comparison.method_adapters import _bvh_header_contract
from retargeting_comparison.runner import _timing_summary
from retargeting_comparison.schemas import RunManifest, RunStatus


def test_run_manifest_roundtrip(tmp_path) -> None:
    value = RunManifest(
        run_id="test",
        method="sparse-neutral",
        status=RunStatus.RUNNING,
        command=["python", "worker"],
        environment="test",
        repo_commit="a" * 40,
        config_sha256="b" * 64,
        device="cpu",
    )
    path = tmp_path / "manifest.json"
    value.save(path)
    assert RunManifest.load(path) == value


def test_timing_summary_preserves_all_raw_runs() -> None:
    repetition = {
        "index": 0,
        "role": "measured",
        "wall_time_s": 2.0,
        "frame_count": 20,
        "native_total_s": 1.0,
        "native_frame_times_s": [0.05] * 20,
        "canonical_qpos_sha256": "c" * 64,
        "canonical_g1_content_sha256": "d" * 64,
    }
    cold = {"repetitions": [repetition]}
    warmup = {**repetition, "role": "warmup"}
    warm = {"repetitions": [warmup, repetition, repetition, repetition]}
    summary = _timing_summary(cold, warm, 3.0, 7.0, fps=20.0)
    assert summary["end_to_end_rtf_raw"] == [2.0, 2.0, 2.0]
    assert summary["native_core_rtf_raw"] == [1.0, 1.0, 1.0]


def test_timing_summary_rejects_cold_warm_content_drift() -> None:
    repetition = {
        "index": 0,
        "role": "measured",
        "wall_time_s": 2.0,
        "frame_count": 20,
        "native_total_s": 1.0,
        "native_frame_times_s": [0.05] * 20,
        "canonical_qpos_sha256": "c" * 64,
        "canonical_g1_content_sha256": "d" * 64,
    }
    cold = {"repetitions": [repetition]}
    warm_repetition = deepcopy(repetition)
    warm_repetition["canonical_qpos_sha256"] = "e" * 64
    warm = {"repetitions": [warm_repetition, warm_repetition, warm_repetition]}
    with pytest.raises(RuntimeError, match="not deterministic"):
        _timing_summary(cold, warm, 3.0, 7.0, fps=20.0)


def test_bvh_header_contract_reports_native_timeline(tmp_path: Path) -> None:
    path = tmp_path / "cropped.bvh"
    path.write_text(
        "HIERARCHY\nMOTION\nFrames: 600\nFrame Time: 0.033333\n",
        encoding="utf-8",
    )
    frames, frame_time_s, fps = _bvh_header_contract(path)
    assert frames == 600
    assert frame_time_s == 0.033333
    assert fps == pytest.approx(30.000300003000028)
