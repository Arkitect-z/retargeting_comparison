from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from retargeting_comparison.stage1_timing_campaign import (
    END_TO_END_BOUNDARY,
    EXPECTED_FROZEN_ORDER,
    NATIVE_BOUNDARY,
    CampaignLock,
    _load_state,
    _new_state,
    _state_for_plan,
    _write_state,
    assert_formal_campaign_ownership,
    build_campaign_plan,
    load_frozen_campaign,
    validate_standard_timing,
)


ROOT = Path(__file__).resolve().parents[1]


def _repetition(index: int, start_second: int) -> dict[str, object]:
    return {
        "index": index,
        "role": "measured",
        "started_at_utc": f"2026-07-22T12:00:{start_second:02d}+00:00",
        "output_in_memory_at_utc": (f"2026-07-22T12:00:{start_second + 1:02d}+00:00"),
        "wall_time_s": 1.0,
        "steady_end_to_end_total_s": 1.0,
        "native_total_s": 0.75,
        "frame_count": 600,
        "timing_boundary": END_TO_END_BOUNDARY,
        "native_boundary": NATIVE_BOUNDARY,
        "canonical_artifact_write_time_s_excluded": 0.05,
        "cpu_affinity": [17],
        "thread_limit": 1,
        "timing_artifact_sha256": "a" * 64,
        "canonical_qpos_sha256": "b" * 64,
        "canonical_g1_content_sha256": "c" * 64,
        "runtime_witness_enabled": index == 0 and start_second == 0,
    }


def _timing() -> dict[str, object]:
    cold = _repetition(0, 0)
    warmup = _repetition(0, 2)
    warmup["role"] = "warmup"
    measured = [_repetition(index, 4 + 2 * index) for index in range(3)]
    return {
        "protocol": {
            "cold_processes": 1,
            "warmup_runs": 1,
            "measured_warm_runs": 3,
            "threads": 1,
            "visualization": False,
            "end_to_end_boundary": END_TO_END_BOUNDARY,
            "native_core_boundary": NATIVE_BOUNDARY,
            "intermediate_and_final_artifact_writes_excluded": True,
            "canonical_qpos_hash_required_per_repetition": True,
            "cold_warm_qpos_determinism_required": True,
            "runtime_witness_policy": (
                "independent cold solver observation; disabled for warmup/measured"
            ),
        },
        "cold": cold,
        "warmup": [warmup],
        "measured_warm": measured,
        "end_to_end_rtf_raw": [0.05, 0.05, 0.05],
        "native_core_rtf_raw": [0.0375, 0.0375, 0.0375],
    }


def test_frozen_numpy_permutation_and_registered_revisions() -> None:
    campaign = load_frozen_campaign(ROOT)
    assert campaign.order == EXPECTED_FROZEN_ORDER
    assert tuple(campaign.outputs) == (
        "sparse-neutral",
        "dense",
        "protomotions-v3",
        "omniretarget",
        "protomotions-v2.3",
        "gmr",
    )


def test_campaign_rejects_a_claimed_order_not_generated_by_seed(tmp_path: Path) -> None:
    config = yaml.safe_load((ROOT / "configs/stage1_timing_campaign.yaml").read_text())
    config["randomization"]["frozen_order"] = list(reversed(EXPECTED_FROZEN_ORDER))
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="reproducible PCG64 permutation"):
        load_frozen_campaign(ROOT, path)


def test_plan_is_exactly_sequential_and_proto_requires_fresh_timing() -> None:
    plan = build_campaign_plan(ROOT)
    assert [job["name"] for job in plan["jobs"]] == list(EXPECTED_FROZEN_ORDER)
    assert len({job["index"] for job in plan["jobs"]}) == 6
    assert plan["protocol"]["overlapping_formal_jobs_forbidden"] is True
    proto = plan["jobs"][2]
    assert proto["registered_output"] == "protomotions-v3-v3"
    assert proto["command"][proto["command"].index("--summary-json") + 1] == (
        "manifests/protomotions_v3_campaign.formal_timing_v3.json"
    )
    assert "--run-fresh-cold" in proto["command"]
    assert "--timing-only" in proto["command"]
    assert plan["jobs"][0]["command"][-4:] == [
        "--revision",
        "v6",
        "--repo-root",
        str(ROOT),
    ]


def test_campaign_state_payload_hash_detects_mutation(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    value = {"schema_version": 1, "status": "pending"}
    _write_state(path, value)
    assert _load_state(path)["status"] == "pending"
    tampered = json.loads(path.read_text())
    tampered["status"] = "succeeded"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        _load_state(path)


def test_resume_identity_excludes_mutable_git_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import retargeting_comparison.stage1_timing_campaign as campaign_module

    monkeypatch.setattr(
        campaign_module,
        "_repo_provenance",
        lambda root: {
            "git_commit": "a" * 40,
            "git_status_porcelain_sha256": "1" * 64,
            "git_worktree_clean": True,
        },
    )
    before = build_campaign_plan(ROOT)
    state_path = tmp_path / "interrupted_state.json"
    state = _new_state(before)
    state["status"] = "interrupted"
    _write_state(state_path, state)

    monkeypatch.setattr(
        campaign_module,
        "_repo_provenance",
        lambda root: {
            "git_commit": "a" * 40,
            "git_status_porcelain_sha256": "2" * 64,
            "git_worktree_clean": False,
        },
    )
    after = build_campaign_plan(ROOT)
    assert before["repo_provenance"] != after["repo_provenance"]
    assert before["plan_sha256"] == after["plan_sha256"]
    resumed = _state_for_plan(ROOT, state_path, after)
    assert resumed["status"] == "interrupted"
    assert resumed["plan_sha256"] == before["plan_sha256"]


def test_campaign_lock_refuses_overlap(tmp_path: Path) -> None:
    lock = tmp_path / "campaign.lock"
    with CampaignLock(lock):
        with pytest.raises(RuntimeError, match="campaign is active"):
            with CampaignLock(lock):
                pass


def test_formal_worker_requires_active_campaign_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    lock = root / "runs/.stage1_formal_timing.lock"
    with CampaignLock(lock) as campaign_lock:
        monkeypatch.delenv("RTCMP_FORMAL_CAMPAIGN_TOKEN", raising=False)
        with pytest.raises(RuntimeError, match="cannot overlap"):
            assert_formal_campaign_ownership(root)
        assert campaign_lock.token is not None
        monkeypatch.setenv("RTCMP_FORMAL_CAMPAIGN_TOKEN", campaign_lock.token)
        assert_formal_campaign_ownership(root)


def test_standard_timing_accepts_only_exact_in_memory_boundaries() -> None:
    timing = _timing()
    receipt = validate_standard_timing(
        timing, source_frames=600, expected_cpu_affinity=[17]
    )
    assert receipt["measured"] == 3
    assert receipt["cpu_affinity"] == [17]

    stale = deepcopy(timing)
    stale["measured_warm"][0]["timing_boundary"] = "component-summed boundary"
    with pytest.raises(ValueError, match="obsolete or ambiguous"):
        validate_standard_timing(stale, source_frames=600)

    unpinned = deepcopy(timing)
    unpinned["warmup"][0]["cpu_affinity"] = [17, 18]
    with pytest.raises(ValueError, match="exactly one CPU"):
        validate_standard_timing(unpinned, source_frames=600)


def test_standard_timing_rejects_overlap_and_count_drift() -> None:
    overlap = _timing()
    overlap["measured_warm"][0]["started_at_utc"] = "2026-07-22T12:00:02.500000+00:00"
    with pytest.raises(ValueError, match="overlap"):
        validate_standard_timing(overlap, source_frames=600)

    short = _timing()
    short["measured_warm"] = short["measured_warm"][:2]
    with pytest.raises(ValueError, match="exactly one cold"):
        validate_standard_timing(short, source_frames=600)


def test_standard_timing_rejects_warm_capture_and_content_drift() -> None:
    captured_warm = _timing()
    captured_warm["measured_warm"][0]["runtime_witness_enabled"] = True
    with pytest.raises(ValueError, match="cold-only"):
        validate_standard_timing(captured_warm, source_frames=600)

    drifted = _timing()
    drifted["warmup"][0]["canonical_g1_content_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="not deterministic"):
        validate_standard_timing(drifted, source_frames=600)
