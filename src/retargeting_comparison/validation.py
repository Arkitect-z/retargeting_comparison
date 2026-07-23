"""Fail-closed acceptance validator for the expanded Stage 1 Pilot."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .calibration import load_evaluator_protocol
from .constants import FULL_LAFAN_STOP_MESSAGE, G1_JOINT_NAMES
from .io_utils import atomic_write_json, load_yaml, sha256_file
from .interaction import (
    INTENDED_CONTACT_SPECS,
    INTERACTION_METRIC_REVISION,
    _array_contract_sha256,
    verify_interaction_ablation_pair,
    verify_interaction_attempt,
)
from .schemas import CanonicalG1, RunManifest, RunStatus
from .scale_metric_registry import (
    SCALE_RANK_METRICS,
    SCALE_RESPONSE_FACTORS,
    SCALE_RESPONSE_METRIC_BY_NAME,
    SCALE_RESPONSE_METRIC_NAMES,
)
from .stage1_publication import (
    CORE_METHODS,
    EXPECTED_FRAMES,
    SHARED_COMPARISON_FRAMES,
    SOURCE_FRAMES,
    FORMAL_NATIVE_SCALE_ROLE,
    CONTACT_ROBUSTNESS_ROLE,
    NATIVE_SCALE_METHODS,
    REFERENCE_METHOD,
    REPORT_FILES,
    SCALE_VARIANTS,
    SPARSE_DIAGNOSTICS,
    TRANSPLANT_POLICIES,
    VALIDATION_SCHEMA_VERSION,
    inspect_publication_evidence,
    method_output_paths,
    presentation_slide_count,
    validation_decision_basis_sha256,
    verify_scale_publication_input_ledger,
)


def _core_motion_check(
    motion: CanonicalG1,
    source_frame_count: int,
    *,
    expected_output_frames: int | None = None,
) -> bool:
    """Return a JSON-native acceptance value for one canonical trajectory."""

    expected_frames = expected_output_frames or source_frame_count
    expected_status = (
        "succeeded"
        if expected_frames / source_frame_count >= 0.95
        else "incomplete"
    )
    return bool(
        len(motion.qpos) == expected_frames
        and motion.metadata.get("completion_status") == expected_status
        and np.isfinite(motion.qpos).all()
        and np.isfinite(motion.per_frame_solve_time_s).all()
        and np.asarray(motion.valid, dtype=bool).all()
        and np.array_equal(motion.source_frame_idx, np.arange(expected_frames))
    )


def _attempt(check: Callable[[], bool]) -> bool:
    try:
        return bool(check())
    except Exception:
        return False


def _boolean_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def _sequence(root: Path) -> dict[str, Any]:
    return load_yaml(root / "manifests/pilot_sequence.yaml")


def _all_motion_outputs(root: Path) -> bool:
    sequence = _sequence(root)
    if int(sequence.get("num_frames", -1)) != EXPECTED_FRAMES:
        return False
    paths = method_output_paths(root)
    protocol = load_evaluator_protocol(root / "manifests/evaluator.yaml")
    controlled_config_hash = sha256_file(root / "configs/controlled_mink.yaml")
    evaluator_hash = sha256_file(root / "manifests/evaluator.yaml")
    for spec in (*CORE_METHODS, *SPARSE_DIAGNOSTICS, REFERENCE_METHOD):
        motion = CanonicalG1.load(paths[spec.key])
        motion.validate(source_frame_count=EXPECTED_FRAMES)
        expected_output_frames = (
            SHARED_COMPARISON_FRAMES
            if spec.key == "protomotions-v3"
            else SOURCE_FRAMES
        )
        if not _core_motion_check(
            motion,
            EXPECTED_FRAMES,
            expected_output_frames=expected_output_frames,
        ):
            return False
        if spec.key == "protomotions-v3" and not (
            motion.metadata.get("full_source_completion_ratio") == 0.75
            and motion.metadata.get("native_contract_completion_status")
            == "succeeded"
            and motion.metadata.get("native_contract_frame_count")
            == SHARED_COMPARISON_FRAMES
        ):
            return False
        if str(motion.metadata.get("method", "")) not in spec.metadata_methods:
            return False
        if spec.key in {"sparse-neutral", "sparse-a", "sparse-b", "dense"} and (
            motion.metadata.get("config_sha256") != controlled_config_hash
            or motion.metadata.get("evaluator_sha256") != evaluator_hash
            or motion.metadata.get("canonical_robot_xml_sha256")
            != protocol["robot_xml_sha256"]
            or motion.metadata.get("canonical_joint_order_sha256")
            != protocol["robot_joint_order_sha256"]
            or motion.metadata.get("root_local_and_anchor_frozen_separately")
            is not True
            or not np.isclose(
                float(motion.metadata.get("root_displacement_scale", np.nan)),
                float(protocol["scale"]["common_root_displacement_scale"]),
                atol=1e-12,
                rtol=0.0,
            )
            or not np.isclose(
                float(motion.metadata.get("local_body_scale", np.nan)),
                float(protocol["scale"]["common_local_body_scale"]),
                atol=1e-12,
                rtol=0.0,
            )
            or not np.isfinite(
                float(motion.metadata.get("native_core_total_s", np.nan))
            )
        ):
            return False
    reference = CanonicalG1.load(paths[REFERENCE_METHOD.key])
    return bool(
        reference.metadata.get("timing_available") is False
        and reference.metadata.get("verified_official_ground_truth") is False
    )


def _formal_run_manifests(root: Path) -> bool:
    sequence_id = str(_sequence(root)["sequence_id"])
    paths = method_output_paths(root)
    for spec in CORE_METHODS:
        candidates = (
            root / "runs/manifests" / f"{sequence_id}__{spec.run_directory}.json",
            root / "manifests/runs" / f"{sequence_id}__{spec.run_directory}.json",
        )
        manifest_path = next((path for path in candidates if path.is_file()), None)
        if manifest_path is None:
            return False
        manifest = RunManifest.load(manifest_path)
        if (
            manifest.status != RunStatus.SUCCEEDED
            or manifest.output_sha256 != sha256_file(paths[spec.key])
            or not manifest.command
            or not manifest.environment
            or not manifest.config_sha256
            or manifest.exit_code != 0
        ):
            return False
    return True


def _smoke_runs(root: Path) -> bool:
    from .native_target_capture import tensor_sha256
    from .runner import METHOD_ENVIRONMENTS, _method_config_hash

    sequence = _sequence(root)
    sequence_id = str(sequence["sequence_id"])
    canonical_source = (root / str(sequence["canonical_path"])).resolve()
    generic = (
        ("sparse-neutral-v6-smoke2", "sparse", "sparse-neutral"),
        ("dense-v6-smoke2", "dense", "dense"),
        ("gmr-v3-smoke2", "gmr", "gmr"),
        ("omniretarget-v3-smoke2", "omniretarget", "omniretarget"),
        (
            "protomotions-v2.3-v3-smoke2",
            "protomotions_v2_3",
            "protomotions_v2_3_mink",
        ),
    )
    for label, method, metadata_method in generic:
        manifest_path = root / "runs/manifests" / f"{sequence_id}__{label}.json"
        manifest = RunManifest.load(manifest_path)
        output = Path(str(manifest.output_path))
        if not output.is_absolute():
            output = root / output
        output = output.resolve()
        run_root = (root / "runs" / sequence_id / label).resolve()
        try:
            output.relative_to(run_root)
        except ValueError:
            return False
        timing_path = Path(str(manifest.timing_path))
        if not timing_path.is_absolute():
            timing_path = root / timing_path
        timing_path = timing_path.resolve()
        provenance_path = output.parent / "provenance.json"
        if (
            not output.is_file()
            or timing_path != output.parent / "timing.json"
            or not timing_path.is_file()
            or not provenance_path.is_file()
        ):
            return False
        motion = CanonicalG1.load(output)
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        current_config = _method_config_hash(root, method)
        expected_environment = f"conda:{METHOD_ENVIRONMENTS[method]}"
        expected_qpos_hash = tensor_sha256(
            np.asarray(motion.qpos, dtype=np.float64)
        )
        protocol = timing.get("protocol", {})
        repetitions = [
            timing.get("cold"),
            *timing.get("warmup", []),
            *timing.get("measured_warm", []),
        ]
        command = list(manifest.command)
        try:
            command_method = command[command.index("--method") + 1]
            command_output = Path(command[command.index("--output") + 1]).resolve()
            command_source = Path(command[command.index("--source") + 1]).resolve()
            command_max_frames = int(command[command.index("--max-frames") + 1])
        except (ValueError, IndexError):
            return False
        history = provenance.get("history")
        timing_provenance = Path(str(timing.get("provenance_path", "")))
        if not timing_provenance.is_absolute():
            timing_provenance = root / timing_provenance
        if (
            len(motion.qpos) != 2
            or not np.isfinite(motion.qpos).all()
            or not np.isfinite(motion.per_frame_solve_time_s).all()
            or not np.array_equal(
                motion.source_frame_idx, np.arange(2, dtype=np.int64)
            )
            or not np.asarray(motion.valid, dtype=bool).all()
            or motion.metadata.get("method") != metadata_method
            or motion.metadata.get("completion_status") != "incomplete"
            or motion.metadata.get("canonical_source_sha256")
            != str(sequence["cropped_sha256"])
            or Path(str(motion.metadata.get("canonical_source_path", ""))).resolve()
            != canonical_source
            or manifest.status != RunStatus.INCOMPLETE
            or manifest.run_id != f"{sequence_id}__{label}"
            or manifest.method != label
            or manifest.output_sha256 != sha256_file(output)
            or manifest.config_sha256 != current_config
            or manifest.source_sha256 != str(sequence["cropped_sha256"])
            or manifest.environment != expected_environment
            or manifest.exit_code != 0
            or command_method != method
            or command_output != output
            or command_source != canonical_source
            or command_max_frames != 2
            or timing.get("implementation_receipt_sha256") != current_config
            or timing_provenance.resolve() != provenance_path.resolve()
            or timing.get("provenance_sha256") != sha256_file(provenance_path)
            or protocol.get("cold_processes") != 1
            or protocol.get("warmup_runs") != 1
            or protocol.get("measured_warm_runs") != 3
            or protocol.get("threads") != 1
            or protocol.get("visualization") is not False
            or protocol.get("end_to_end_boundary")
            != "canonical_source_file_to_canonical_g1_in_memory"
            or protocol.get("native_core_boundary")
            != "native_input_ready_to_native_g1_in_memory"
            or protocol.get("runtime_witness_policy")
            != "independent cold solver observation; disabled for warmup/measured"
            or len(repetitions) != 5
            or any(not isinstance(item, dict) for item in repetitions)
            or any(int(item.get("frame_count", -1)) != 2 for item in repetitions)
            or any(item.get("canonical_qpos_sha256") != expected_qpos_hash for item in repetitions)
            or timing.get("canonical_qpos_sha256") != expected_qpos_hash
            or timing.get("all_cold_warm_qpos_identical") is not True
            or not isinstance(history, list)
            or len(history) < 4
            or any(
                not isinstance(item, dict)
                or not isinstance(item.get("receipt"), dict)
                or item["receipt"].get("aggregate_sha256") != current_config
                for item in history
            )
        ):
            return False
        for item in repetitions:
            artifact = Path(str(item.get("timing_artifact_path", "")))
            if not artifact.is_absolute():
                artifact = root / artifact
            artifact = artifact.resolve()
            try:
                artifact.relative_to(output.parent)
            except ValueError:
                return False
            if (
                not artifact.is_file()
                or sha256_file(artifact) != str(item.get("timing_artifact_sha256"))
                or item.get("timing_boundary")
                != "canonical_source_file_to_canonical_g1_in_memory"
                or item.get("native_boundary")
                != "native_input_ready_to_native_g1_in_memory"
                or item.get("thread_limit") != 1
            ):
                return False
        if method in {"gmr", "omniretarget", "protomotions_v2_3"}:
            capture = motion.metadata.get("runtime_pre_solver_capture")
            binding = motion.metadata.get("runtime_witness_binding")
            if not isinstance(capture, dict):
                return False
            cold_output = Path(str(capture.get("witness_output_path", "")))
            if not cold_output.is_absolute():
                cold_output = root / cold_output
            cold_output = cold_output.resolve()
            try:
                cold_output.relative_to(output.parent)
            except ValueError:
                return False
            if not cold_output.is_file():
                return False
            cold_motion = CanonicalG1.load(cold_output)
            cold_qpos_hash = tensor_sha256(
                np.asarray(cold_motion.qpos, dtype=np.float64)
            )
            if (
                capture.get("observed_during_solver_run") is not True
                or capture.get("witness_and_timed_qpos_identical") is not True
                or capture.get("witness_output_sha256") != sha256_file(cold_output)
                or capture.get("witness_qpos_sha256") != cold_qpos_hash
                or not isinstance(binding, dict)
                or binding.get("identical") is not True
                or binding.get("cold_output_sha256") != sha256_file(cold_output)
                or binding.get("warm_qpos_sha256") != expected_qpos_hash
                or binding.get("cold_qpos_sha256") != cold_qpos_hash
                or cold_qpos_hash != expected_qpos_hash
            ):
                return False

    from .stage1_timing_campaign import _proto_provenance_receipt

    proto_dir = root / "runs" / sequence_id / "protomotions-v3-v3-smoke2"
    proto_output = proto_dir / "canonical_g1.npz"
    proto_evidence = proto_dir / "evidence.json"
    proto_receipt = proto_dir / "smoke_receipt.json"
    if not all(path.is_file() for path in (proto_output, proto_evidence, proto_receipt)):
        return False
    receipt = json.loads(proto_receipt.read_text(encoding="utf-8"))
    motion = CanonicalG1.load(proto_output)
    sequence = _sequence(root)
    source = root / str(sequence["canonical_path"])
    current_implementation = _proto_provenance_receipt(root)
    if (
        receipt.get("schema_version") != 1
        or receipt.get("status") != "succeeded"
        or receipt.get("solver_invoked") is not True
        or int(receipt.get("frames", -1)) != 2
        or receipt.get("output_path")
        != str(proto_output.relative_to(root))
        or receipt.get("output_sha256") != sha256_file(proto_output)
        or receipt.get("evidence_sha256") != sha256_file(proto_evidence)
        or receipt.get("source_sha256") != sha256_file(source)
        or receipt.get("implementation_receipt") != current_implementation
        or receipt.get("implementation_sha256")
        != current_implementation["aggregate_sha256"]
        or receipt.get("qpos_sha256")
        != tensor_sha256(np.asarray(motion.qpos, dtype=np.float64))
        or len(motion.qpos) != 2
        or motion.metadata.get("completion_status") != "incomplete"
    ):
        return False
    return True


def _body_models(root: Path) -> bool:
    value = load_yaml(root / "manifests/body_models.yaml")
    for name in ("smpl", "smplx"):
        entry = value[name]
        asset = (root / str(entry["path"])).resolve()
        if (
            entry.get("finite_forward") is not True
            or not asset.is_file()
            or entry.get("sha256") != sha256_file(asset)
        ):
            return False
    return value.get("original_smpl_pickle_used") is False


def _source_adapters(root: Path) -> bool:
    table_path = root / "metrics/source_adapter_errors.csv"
    manifest_path = root / "manifests/source_adapters.yaml"
    table = pd.read_csv(table_path)
    expected = {
        "gmr",
        "omniretarget",
        "protomotions-v2.3",
        "protomotions-v3",
    }
    if set(table.method.astype(str)) != expected or len(table) != len(expected):
        return False
    for column in (
        "frame_count_match",
        "fps_match",
        "left_right_match",
        "runtime_boundary_observed",
    ):
        values = table[column]
        valid = values if values.dtype == bool else values.astype(str).str.lower().eq("true")
        if not bool(valid.all()):
            return False
    manifest = load_yaml(manifest_path)
    numeric = table.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    if (
        not table.status.astype(str).str.lower().eq("passed").all()
        or not np.isfinite(numeric).all()
        or int(manifest.get("schema_version", 0)) != 3
        or manifest.get("metrics_sha256") != sha256_file(table_path)
        or manifest.get("native_artifacts_committed_to_git") is not False
        or set(map(str, manifest.get("expected_methods", []))) != expected
        or manifest.get("all_runtime_boundaries_observed") is not True
        or set(map(str, manifest.get("native_artifacts", {}))) != expected
    ):
        return False
    expected_joint_order_hash = hashlib.sha256(
        json.dumps(
            list(G1_JOINT_NAMES), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    if not table.robot_joint_order_sha256.astype(str).eq(
        expected_joint_order_hash
    ).all():
        return False
    for row in table.itertuples():
        for path_column, hash_column in (
            ("source_adapter_artifact_path", "source_adapter_artifact_sha256"),
            ("robot_asset_path", "robot_asset_sha256"),
            ("pre_solver_capture_path", "pre_solver_capture_sha256"),
        ):
            path = root / str(getattr(row, path_column))
            if (
                not path.is_file()
                or sha256_file(path) != str(getattr(row, hash_column))
            ):
                return False
        capture_path = root / str(row.pre_solver_capture_path)
        capture = json.loads(capture_path.read_text(encoding="utf-8"))
        if (
            int(capture.get("schema_version", 0)) != 2
            or capture.get("method") != str(row.method)
            or int(capture.get("frames", -1))
            != (
                SHARED_COMPARISON_FRAMES
                if str(row.method) == "protomotions_v3"
                else SOURCE_FRAMES
            )
            or capture.get("tensor_sha256")
            != str(row.pre_solver_tensor_sha256)
            or capture.get("boundary_observed_during_formal_run") is not True
            or capture.get("reconstruction_matches_runtime") is not True
        ):
            return False
        artifact_entry = manifest["native_artifacts"][str(row.method)]
        if (
            str(artifact_entry.get("path"))
            != str(row.source_adapter_artifact_path)
            or str(artifact_entry.get("sha256"))
            != str(row.source_adapter_artifact_sha256)
        ):
            return False
    return True


def _evaluator_outputs(root: Path) -> bool:
    protocol_path = root / "manifests/evaluator.yaml"
    protocol = load_evaluator_protocol(protocol_path)
    sequence = _sequence(root)
    if (
        protocol.get("source_sha256") != sequence.get("cropped_sha256")
        or protocol.get("full_lafan_authorized") is not False
    ):
        return False
    scale = protocol["scale"]
    controlled = load_yaml(root / "configs/controlled_mink.yaml")
    controlled_robot = root / str(controlled["robot_xml"])
    if (
        not controlled_robot.is_file()
        or sha256_file(controlled_robot) != str(protocol["robot_xml_sha256"])
        or str(controlled["robot_xml"]) != str(protocol["robot_xml"])
        or scale.get("root_local_and_anchor_frozen_separately") is not True
        or scale.get("diagnostics", {}).get("head_to_toe_role")
        != "diagnostic_only_not_common_policy"
        or "common_frame0_head_toe_span" in TRANSPLANT_POLICIES
        or len(scale.get("source_landmarks", [])) != 11
        or len(scale.get("robot_landmarks", [])) != 11
        or not np.isclose(
            float(controlled["common"]["root_displacement_scale"]),
            float(scale["common_root_displacement_scale"]),
            atol=1e-12,
            rtol=0.0,
        )
        or not np.isclose(
            float(controlled["common"]["local_body_scale"]),
            float(scale["common_local_body_scale"]),
            atol=1e-12,
            rtol=0.0,
        )
    ):
        return False
    digest = sha256_file(protocol_path)
    core = pd.read_csv(root / "metrics/stage1_core_summary.csv")
    reference = pd.read_csv(root / "metrics/stage1_reference_summary.csv")
    if set(core.key.astype(str)) != {spec.key for spec in CORE_METHODS}:
        return False
    if set(reference.key.astype(str)) != {REFERENCE_METHOD.key}:
        return False
    comparison = pd.read_csv(root / "metrics/stage1_reference_comparison.csv")
    direct_metrics = {
        "direct_root_translation_mean_m",
        "direct_root_translation_p95_m",
        "direct_root_anchor_frame0_m",
        "direct_root_displacement_mean_m",
        "direct_root_displacement_p95_m",
        "direct_root_yaw_mean_rad",
        "direct_root_orientation_mean_rad",
        "direct_joint_angle_rmse_rad",
        "direct_joint_velocity_rmse_rad_s",
        "direct_fk_world_semantic_mean_m",
        "direct_fk_root_frame_semantic_mean_m",
    }
    direct = comparison.loc[
        comparison.comparison_type.astype(str).eq(
            "direct_same_g1_frame_index_aligned_trajectory_disagreement"
        )
    ]
    if (
        len(direct) != len(CORE_METHODS) * len(direct_metrics)
        or set(direct.key.astype(str)) != {spec.key for spec in CORE_METHODS}
        or set(direct.metric.astype(str)) != direct_metrics
        or not (
            direct.frames.astype(int) == SHARED_COMPARISON_FRAMES
        ).all()
        or not direct.timeline_alignment.astype(str).eq(
            "frame_index_only_not_exact_timestamp"
        ).all()
        or not np.allclose(direct.reference_value.astype(float), 0.0)
        or not np.isfinite(direct.method_value.to_numpy(dtype=float)).all()
        or (direct.method_value.astype(float) < 0.0).any()
    ):
        return False
    output_paths = method_output_paths(root)
    reference_hash = sha256_file(output_paths[REFERENCE_METHOD.key])
    for spec in CORE_METHODS:
        rows_for_method = direct.loc[direct.key.astype(str).eq(spec.key)]
        if (
            not rows_for_method.method_output_sha256.astype(str).eq(
                sha256_file(output_paths[spec.key])
            ).all()
            or not rows_for_method.reference_output_sha256.astype(str).eq(
                reference_hash
            ).all()
        ):
            return False
    rows = pd.concat((core, reference), ignore_index=True)
    required = (
        "rf_kpe_all_mean_m",
        "rf_kpe_targeted_mean_m",
        "rf_kpe_untracked_mean_m",
        "root_translation_common_scale_mean_m",
        "root_translation_scale_invariant_mean_m",
        "root_yaw_mean_rad",
        "artifact_rate",
    )
    return bool(
        rows.evaluator_protocol_sha256.astype(str).eq(digest).all()
        and (rows.frames.astype(int) == SHARED_COMPARISON_FRAMES).all()
        and np.allclose(
            rows.comparison_window_completion_ratio.astype(float), 1.0
        )
        and np.allclose(
            rows.completion_ratio.astype(float),
            SHARED_COMPARISON_FRAMES / SOURCE_FRAMES,
        )
        and np.allclose(
            rows.full_source_coverage_ratio.astype(float),
            rows.key.astype(str).map(
                lambda key: 0.75 if key == "protomotions-v3" else 1.0
            ),
        )
        and np.isfinite(rows.loc[:, required].to_numpy(dtype=float)).all()
    )


def _timing(root: Path) -> bool:
    raw = pd.read_csv(root / "metrics/stage1_timing_raw.csv")
    summary = pd.read_csv(root / "metrics/stage1_timing_summary.csv")
    expected = {spec.key for spec in CORE_METHODS}
    if set(summary.key.astype(str)) != expected or set(raw.key.astype(str)) != expected:
        return False
    measured = raw.loc[raw.role.astype(str).eq("measured")]
    warmup = raw.loc[raw.role.astype(str).eq("warmup")]
    cold = raw.loc[raw.role.astype(str).eq("cold")]
    counts = measured.groupby("key").size().to_dict()
    warmups = warmup.groupby("key").size().to_dict()
    colds = cold.groupby("key").size().to_dict()
    numeric = raw[["end_to_end_rtf", "native_core_rtf", "steady_end_to_end_s"]]
    paths = method_output_paths(root)
    fps_by_key: dict[str, float] = {}
    for spec in CORE_METHODS:
        motion = CanonicalG1.load(paths[spec.key])
        fps_by_key[spec.key] = float(motion.fps)
    for row in raw.itertuples(index=False):
        artifact = Path(str(row.timing_artifact_path))
        if not artifact.is_absolute():
            artifact = root / artifact
        if (
            not artifact.is_file()
            or sha256_file(artifact) != str(row.timing_artifact_sha256)
            or str(row.timing_artifact_hash_verified).lower() != "true"
            or int(row.frame_count)
            != (
                SHARED_COMPARISON_FRAMES
                if str(row.key) == "protomotions-v3"
                else SOURCE_FRAMES
            )
        ):
            return False
        duration = int(row.frame_count) / fps_by_key[str(row.key)]
        if not np.isclose(
            float(row.end_to_end_rtf),
            float(row.steady_end_to_end_s) / duration,
            rtol=1e-12,
            atol=1e-12,
        ) or not np.isclose(
            float(row.native_core_rtf),
            float(row.native_total_s) / duration,
            rtol=1e-12,
            atol=1e-12,
        ):
            return False
        if (
            str(row.timing_boundary)
            != "canonical_source_file_to_canonical_g1_in_memory"
        ):
            return False
    for row in summary.itertuples(index=False):
        timing_source = root / str(row.timing_source)
        if (
            not timing_source.is_file()
            or sha256_file(timing_source) != str(row.timing_source_sha256)
            or abs(float(row.reported_minus_recomputed_end_to_end_rtf)) > 1e-12
            or abs(float(row.reported_minus_recomputed_native_core_rtf)) > 1e-12
        ):
            return False
    return bool(
        all(
            counts.get(key) == 3
            and warmups.get(key) == 1
            and colds.get(key) == 1
            for key in expected
        )
        and (summary.measured_repetitions.astype(int) == 3).all()
        and (summary.warmup_repetitions.astype(int) == 1).all()
        and summary.timing_evidence_grade.astype(str).eq("formal_repeated").all()
        and np.isfinite(numeric.to_numpy(dtype=float)).all()
        and (numeric.to_numpy(dtype=float) >= 0.0).all()
        and (raw.steady_end_to_end_s >= raw.native_total_s).all()
    )


def _native_target_capture(root: Path) -> bool:
    from .native_target_capture import CAPTURE_SCHEMA_VERSION, tensor_sha256

    manifest = pd.read_csv(root / "manifests/native_pre_solver_targets.csv")
    geometry = pd.read_csv(root / "metrics/native_pre_solver_target_geometry.csv")
    expected = {"gmr", "omniretarget", "protomotions-v2.3", "protomotions-v3"}
    if set(manifest.method.astype(str)) != expected:
        return False
    for column, value in (
        ("capture_class", "exact_solver_input"),
        ("exact_solver_input", "true"),
        ("normalized_projection", "false"),
        ("capture_generation_solver_invoked", "false"),
        ("boundary_observed_during_formal_run", "true"),
        ("reconstruction_matches_runtime", "true"),
        ("runtime_witness_solver_invoked", "true"),
    ):
        if not manifest[column].astype(str).str.lower().eq(value).all():
            return False
    witness_method_key = {
        "gmr": "gmr",
        "omniretarget": "omniretarget",
        "protomotions-v2.3": "protomotions-v2.3",
        "protomotions-v3": "protomotions-v3",
    }
    runtime_hash_key = {
        "gmr": "position_tensor_sha256",
        "omniretarget": "target_tensor_sha256",
        "protomotions-v2.3": "position_tensor_sha256",
        "protomotions-v3": "target_keypoints_sha256",
    }
    expected_outputs = method_output_paths(root)
    for row in manifest.itertuples():
        artifact = root / str(row.target_artifact)
        if not artifact.is_file() or sha256_file(artifact) != str(row.target_artifact_sha256):
            return False
        capture_path = artifact.with_name("capture.json")
        if not capture_path.is_file():
            return False
        capture_details = json.loads(capture_path.read_text(encoding="utf-8"))
        if (
            int(capture_details.get("schema_version", 0))
            != CAPTURE_SCHEMA_VERSION
            or capture_details.get("method") != str(row.method)
            or capture_details.get("target_artifact_sha256")
            != str(row.target_artifact_sha256)
            or capture_details.get("capture_generation_solver_invoked") is not False
            or capture_details.get("boundary_observed_during_formal_run") is not True
            or capture_details.get("reconstruction_matches_runtime") is not True
            or capture_details.get("runtime_witness_solver_invoked") is not True
        ):
            return False
        with np.load(artifact, allow_pickle=False) as archive:
            tensor_key = str(row.tensor_key)
            if tensor_key not in archive:
                return False
            primary_tensor_hash = tensor_sha256(np.asarray(archive[tensor_key]))
        if (
            primary_tensor_hash != str(row.tensor_sha256)
            or primary_tensor_hash != str(row.runtime_tensor_sha256)
            or primary_tensor_hash != capture_details.get("tensor_sha256")
            or primary_tensor_hash != capture_details.get("runtime_tensor_sha256")
        ):
            return False
        witness = root / str(row.runtime_witness_output_path)
        expected_witness = expected_outputs[witness_method_key[str(row.method)]]
        if (
            witness.resolve() != expected_witness.resolve()
            or not witness.is_file()
            or sha256_file(witness) != str(row.runtime_witness_output_sha256)
            or capture_details.get("runtime_witness_output_sha256")
            != str(row.runtime_witness_output_sha256)
            or capture_details.get("runtime_witness_output_path")
            != str(row.runtime_witness_output_path)
        ):
            return False
        witness_motion = CanonicalG1.load(witness)
        runtime_capture = witness_motion.metadata.get("runtime_pre_solver_capture")
        if (
            not isinstance(runtime_capture, dict)
            or runtime_capture.get("observed_during_solver_run") is not True
            or runtime_capture.get(runtime_hash_key[str(row.method)])
            != primary_tensor_hash
        ):
            return False
        for path_column, hash_column in (
            ("source_path", "source_sha256"),
            ("config_path", "config_sha256"),
            ("robot_asset_path", "robot_asset_sha256"),
            ("implementation_path", "implementation_sha256"),
        ):
            provenance_path = root / str(getattr(row, path_column))
            if (
                not provenance_path.is_file()
                or sha256_file(provenance_path) != str(getattr(row, hash_column))
            ):
                return False
    numeric = geometry.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    return bool(
        (
            manifest.frames.astype(int)
            == manifest.method.astype(str).map(
                lambda method: (
                    SHARED_COMPARISON_FRAMES
                    if method == "protomotions_v3"
                    else SOURCE_FRAMES
                )
            )
        ).all()
        and set(geometry.method.astype(str)) == expected
        and geometry.capture_class.astype(str).eq("exact_solver_input").all()
        and np.isfinite(numeric).all()
    )


def _scale_evidence(root: Path) -> bool:
    # Rebuild all 43 registered metrics from hash-bound per-frame data, target
    # artifacts and canonical qpos before consulting any derived summary.
    if not verify_scale_publication_input_ledger(root, recompute_metrics=True):
        return False
    table = pd.read_csv(root / "metrics/stage1_scale_policy_summary.csv")
    provenance = pd.read_csv(root / "manifests/scale_run_provenance.csv")
    transplant = table.loc[table.experiment_role.eq("controlled_policy")]
    native = table.loc[table.experiment_role.eq(FORMAL_NATIVE_SCALE_ROLE)]
    robustness = table.loc[table.experiment_role.eq(CONTACT_ROBUSTNESS_ROLE)]
    if set(transplant.variant.astype(str)) != set(TRANSPLANT_POLICIES):
        return False
    for method in NATIVE_SCALE_METHODS:
        if set(native.loc[native.method.eq(method), "variant"].astype(str)) != set(SCALE_VARIANTS):
            return False
    if (
        set(robustness.method.astype(str)) != {"omniretarget"}
        or set(robustness.variant.astype(str)) != set(SCALE_VARIANTS)
    ):
        return False
    selected = pd.concat((transplant, native, robustness), ignore_index=True)
    if len(provenance) != 30:
        return False
    for column in (
        "source_match",
        "config_match",
        "upstream_match",
        "method_match",
        "robot_match",
        "role_match",
        "scale_protocol_match",
        "variant_match",
        "contact_policy_match",
        "exact_registered_frames",
        "accepted",
    ):
        if not _boolean_series(provenance[column]).all():
            return False
    for row in selected.itertuples():
        output = root / str(row.output_path)
        if not output.is_file() or sha256_file(output) != str(row.output_sha256):
            return False
    joined = selected.merge(
        provenance,
        on=["experiment_role", "method", "variant", "output_path", "output_sha256"],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if not joined["_merge"].astype(str).eq("both").all():
        return False
    slopes = pd.read_csv(root / "metrics/stage1_scale_sensitivity_slopes.csv")
    ranks = pd.read_csv(root / "metrics/stage1_method_rank_stability.csv")
    registered_numeric = pd.concat((native, robustness), ignore_index=True).loc[
        :, list(SCALE_RESPONSE_METRIC_NAMES)
    ].to_numpy(dtype=float)
    if not np.isfinite(registered_numeric).all():
        return False
    expected_slope_keys = {
        (method, factor, metric)
        for method in NATIVE_SCALE_METHODS
        for factor, _, _ in SCALE_RESPONSE_FACTORS
        for metric in SCALE_RESPONSE_METRIC_NAMES
    }
    actual_slope_keys = set(
        zip(
            slopes.method.astype(str),
            slopes.factor.astype(str),
            slopes.metric.astype(str),
        )
    )
    if actual_slope_keys != expected_slope_keys or len(slopes) != len(
        expected_slope_keys
    ):
        return False
    expected_rank_keys = {
        (metric, variant, method)
        for metric in SCALE_RANK_METRICS
        for variant in SCALE_VARIANTS
        for method in NATIVE_SCALE_METHODS
    }
    actual_rank_keys = set(
        zip(
            ranks.metric.astype(str),
            ranks.variant.astype(str),
            ranks.method.astype(str),
        )
    )
    if actual_rank_keys != expected_rank_keys or len(ranks) != len(
        expected_rank_keys
    ):
        return False
    for frame in (slopes, ranks):
        metric_names = frame.metric.astype(str)
        expected_direction = metric_names.map(
            {
                name: spec.preferred_direction
                for name, spec in SCALE_RESPONSE_METRIC_BY_NAME.items()
            }
        )
        expected_family = metric_names.map(
            {
                name: spec.family
                for name, spec in SCALE_RESPONSE_METRIC_BY_NAME.items()
            }
        )
        if (
            not frame.preferred_direction.astype(str).eq(expected_direction).all()
            or not frame.metric_family.astype(str).eq(expected_family).all()
        ):
            return False
        if _boolean_series(frame.composite_score_used).any():
            return False
    slope_numeric_columns = (
        "native_value",
        "minus_value",
        "plus_value",
        "minus_delta_from_native",
        "plus_delta_from_native",
        "response_range",
        "central_difference_per_unit_multiplier",
        "elasticity_at_native",
    )
    rank_numeric_columns = (
        "rank",
        "native_rank",
        "spearman_vs_native",
    )
    if (
        not np.isfinite(
            slopes.loc[:, list(slope_numeric_columns)].to_numpy(dtype=float)
        ).all()
        or not np.isfinite(
            ranks.loc[:, list(rank_numeric_columns)].to_numpy(dtype=float)
        ).all()
        or not _boolean_series(ranks.rank_eligible).all()
    ):
        return False
    expected_slope_rank_eligibility = slopes.metric.astype(str).map(
        {
            name: spec.rank_eligible
            for name, spec in SCALE_RESPONSE_METRIC_BY_NAME.items()
        }
    )
    expected_higher_is_better = ranks.metric.astype(str).map(
        {
            name: spec.preferred_direction == "maximize"
            for name, spec in SCALE_RESPONSE_METRIC_BY_NAME.items()
        }
    )
    if (
        not _boolean_series(slopes.rank_eligible).eq(
            expected_slope_rank_eligibility
        ).all()
        or not _boolean_series(ranks.higher_is_better).eq(
            expected_higher_is_better
        ).all()
    ):
        return False
    contact = pd.read_csv(root / "metrics/contact_constraint_flip.csv")
    contact_methods = set(
        contact.method.astype(str).replace(
            {"protomotions_v2_3": "protomotions_v2_3_mink"}
        )
    )
    shape = pd.read_csv(root / "metrics/actor_shape_policy_formula_probe.csv")
    expected_shape_methods = {
        "gmr",
        "omniretarget",
        "protomotions_v2_3",
        "protomotions_v3",
    }
    if (
        set(shape.method.astype(str)) != expected_shape_methods
        or set(shape.profile.astype(str)) != {"short", "zero", "tall"}
        or set(zip(shape.profile.astype(str), shape.beta0.astype(float)))
        != {("short", -2.0), ("zero", 0.0), ("tall", 2.0)}
        or not _boolean_series(shape.motion_pose_and_root_path_frozen).all()
        or not np.allclose(
            shape.contact_label_change_rate_vs_zero.astype(float), 0.0
        )
        or shape.body_model_sha256.astype(str).nunique() != 1
        or not shape.constructor_evidence_grade.astype(str).eq(
            "formula_reconstruction_bound_to_official_source_hash"
        ).all()
        or _boolean_series(shape.runtime_constructor_observed).any()
        or not _boolean_series(
            shape.not_used_as_accuracy_or_ranking_evidence
        ).all()
        or _boolean_series(shape.exact_native_pre_solver_target_claimed).any()
    ):
        return False
    for method in expected_shape_methods:
        rows = shape.loc[shape.method.astype(str).eq(method)].sort_values("beta0")
        if len(rows) != 3:
            return False
        consumes_shape = method in {"gmr", "omniretarget"}
        if not _boolean_series(rows.actor_shape_consumed).eq(consumes_shape).all():
            return False
        if consumes_shape:
            if rows.target_sha256.astype(str).nunique() != 3:
                return False
        elif (
            rows.target_sha256.astype(str).nunique() != 1
            or not rows.geometry_profile_used.astype(str).eq("zero").all()
            or not _boolean_series(rows.target_identical_to_zero_profile).all()
        ):
            return False
        for row in rows.itertuples(index=False):
            artifact = root / str(row.target_artifact)
            if (
                not artifact.is_file()
                or sha256_file(artifact) != str(row.target_artifact_sha256)
            ):
                return False
            source_hashes = json.loads(str(row.official_formula_source_sha256s_json))
            source_paths = json.loads(str(row.official_formula_source_paths_json))
            if (
                not isinstance(source_hashes, dict)
                or not isinstance(source_paths, list)
                or set(source_paths) != set(source_hashes)
                or not source_paths
            ):
                return False
            for relative in source_paths:
                source = root / str(relative)
                if (
                    not source.is_file()
                    or sha256_file(source) != str(source_hashes[relative])
                ):
                    return False
            source_bundle = hashlib.sha256(
                json.dumps(
                    source_hashes,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if source_bundle != str(row.official_formula_source_bundle_sha256):
                return False
    shape_numeric = shape.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    return bool(
        len(transplant) == 5
        and len(native) == 20
        and len(robustness) == 5
        and (
            selected.frames.astype(int) == SHARED_COMPARISON_FRAMES
        ).all()
        and np.allclose(
            selected.comparison_window_completion_ratio.astype(float), 1.0
        )
        and np.allclose(
            selected.completion_ratio.astype(float),
            SHARED_COMPARISON_FRAMES / SOURCE_FRAMES,
        )
        and set(slopes.method.astype(str)) == set(NATIVE_SCALE_METHODS)
        and set(ranks.method.astype(str)) == set(NATIVE_SCALE_METHODS)
        and contact_methods == set(NATIVE_SCALE_METHODS)
        and len(shape) == 12
        and np.isfinite(shape_numeric).all()
    )


def _interaction(root: Path) -> bool:
    summary = pd.read_csv(root / "metrics/stage1_interaction_summary.csv")
    if set(zip(summary.case.astype(str), summary.variant.astype(str))) != {
        (case, variant) for case in ("box", "climb") for variant in ("full", "no-hard")
    }:
        return False
    case_invariants: dict[str, list[tuple[str, str, str, str]]] = {}
    for case in ("box", "climb"):
        for variant in ("full", "no-hard"):
            path = root / "runs/interaction_manifests" / f"interaction__{case}__{variant}.json"
            manifest = RunManifest.load(path)
            try:
                output, value = verify_interaction_attempt(
                    root, case, variant, manifest
                )
            except (FileNotFoundError, KeyError, StopIteration, ValueError):
                return False
            graph = value.get("constraint_graph_audit", {})
            minimum = graph.get("component_count_min", {})
            maximum = graph.get("component_count_max", {})
            hard = variant == "full"
            if (
                value.get("metric_revision")
                != INTERACTION_METRIC_REVISION
                or graph.get("runtime_graph_not_source_ast_only") is not True
                or int(minimum.get("joint_limits", -1)) != 2
                or bool(value.get("activate_obj_non_penetration")) != hard
                or bool(value.get("activate_foot_sticking")) != hard
                or value.get("activate_joint_limits") is not True
                or value.get("intended_contact_pair_asserted") is not True
                or value.get("intended_contact_fail_closed") is not True
                or "point-to-triangle"
                not in str(value.get("source_surface_distance_backend", ""))
                or "mj_geomDistance"
                not in str(value.get("mapped_robot_surface_distance_backend", ""))
            ):
                return False
            if hard:
                if (
                    int(maximum.get("object_non_penetration", 0)) <= 0
                    or int(maximum.get("foot_sticking_and_lock", 0)) <= 0
                ):
                    return False
            elif (
                int(maximum.get("object_non_penetration", -1)) != 0
                or int(maximum.get("foot_sticking_and_lock", -1)) != 0
            ):
                return False
            row = summary.loc[
                summary.case.astype(str).eq(case)
                & summary.variant.astype(str).eq(variant)
            ].iloc[0]
            per_frame = output.parent / "per_frame_metrics.csv"
            intended = output.parent / "intended_contact_per_semantic.csv"
            mapping_path = output.parent / "intended_contact_mapping.json"
            contract_path = output.parent / "intended_contact_source_contract.npz"
            if (
                str(row.summary_sha256) != sha256_file(output)
                or not per_frame.is_file()
                or str(row.per_frame_sha256) != sha256_file(per_frame)
                or not intended.is_file()
                or not mapping_path.is_file()
                or not contract_path.is_file()
                or str(row.intended_contact_per_semantic_sha256)
                != sha256_file(intended)
                or str(row.intended_contact_mapping_artifact_sha256)
                != sha256_file(mapping_path)
                or str(row.intended_contact_source_contract_sha256)
                != sha256_file(contract_path)
                or value.get("intended_contact_per_semantic_sha256")
                != sha256_file(intended)
                or value.get("intended_contact_mapping_artifact_sha256")
                != sha256_file(mapping_path)
                or value.get("intended_contact_source_contract_sha256")
                != sha256_file(contract_path)
            ):
                return False
            mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
            mapping_canonical = hashlib.sha256(
                json.dumps(
                    mapping, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()
            expected_semantics = {
                str(item["semantic"]) for item in INTENDED_CONTACT_SPECS[case]
            }
            mappings = mapping.get("mappings", [])
            if (
                mapping.get("metric_revision") != INTERACTION_METRIC_REVISION
                or mapping.get("case") != case
                or value.get("intended_contact_mapping_canonical_sha256")
                != mapping_canonical
                or len(mappings) != len(expected_semantics)
                or {str(item.get("semantic")) for item in mappings}
                != expected_semantics
                or int(value.get("intended_contact_semantic_count", -1))
                != len(expected_semantics)
            ):
                return False
            for item in mappings:
                if (
                    not str(item.get("source_joint", ""))
                    or not str(item.get("robot_body_name", ""))
                    or not item.get("robot_geom_names")
                    or not item.get("object_geom_names")
                ):
                    return False
            source_mesh = Path(str(mapping.get("source_mesh_path", "")))
            if not source_mesh.is_absolute():
                source_mesh = root / source_mesh
            if (
                not source_mesh.is_file()
                or mapping.get("source_mesh_sha256") != sha256_file(source_mesh)
                or value.get("source_surface_mesh_sha256")
                != sha256_file(source_mesh)
            ):
                return False
            contact = pd.read_csv(intended)
            required_contact_columns = {
                "frame",
                "semantic",
                "source_joint_name",
                "robot_body_name",
                "robot_geom_names",
                "object_geom_names",
                "source_absolute_surface_distance_m",
                "source_reference_surface_distance_m",
                "mapped_robot_object_signed_surface_distance_m",
                "mapped_robot_penetration_depth_m",
                "signed_robot_minus_source_reference_distance_m",
                "absolute_robot_source_distance_error_m",
                *{
                    f"source_contact_{label}" for label in ("2cm", "5cm", "10cm")
                },
                *{
                    f"mapped_robot_within_{label}"
                    for label in ("2cm", "5cm", "10cm")
                },
            }
            frames = int(value.get("frames", -1))
            if (
                required_contact_columns - set(contact)
                or len(contact) != frames * len(expected_semantics)
                or set(contact.semantic.astype(str)) != expected_semantics
                or contact.duplicated(["frame", "semantic"]).any()
                or set(contact.frame.astype(int)) != set(range(frames))
                or not contact.groupby("frame").size().eq(len(expected_semantics)).all()
            ):
                return False
            contact_numeric = contact[
                [
                    "source_absolute_surface_distance_m",
                    "source_reference_surface_distance_m",
                    "mapped_robot_object_signed_surface_distance_m",
                    "mapped_robot_penetration_depth_m",
                    "signed_robot_minus_source_reference_distance_m",
                    "absolute_robot_source_distance_error_m",
                ]
            ].to_numpy(dtype=float)
            if not np.isfinite(contact_numeric).all():
                return False
            source_condition = _boolean_series(contact.source_contact_5cm)
            if (
                not contact.assign(_source_condition=source_condition)
                .groupby("semantic")
                ["_source_condition"]
                .sum()
                .gt(0)
                .all()
            ):
                return False
            counts = [
                int(value.get(f"intended_source_contact_observations_{label}", -1))
                for label in ("2cm", "5cm", "10cm")
            ]
            if not (0 <= counts[0] <= counts[1] <= counts[2] <= len(contact)):
                return False
            if (
                counts[1] <= 0
                or int(value.get("intended_source_contact_observations", -1))
                != counts[1]
                or float(
                    value.get("intended_source_contact_condition_threshold_m", np.nan)
                )
                != 0.05
            ):
                return False
            for label in ("2cm", "5cm", "10cm"):
                preservation = float(
                    value.get(
                        f"intended_contact_preservation_{label}_given_source_contact_5cm",
                        np.nan,
                    )
                )
                if (
                    not np.isfinite(preservation)
                    or not 0.0 <= preservation <= 1.0
                ):
                    return False
            penetration = float(
                value.get(
                    "intended_mapped_penetration_rate_given_source_contact_5cm",
                    np.nan,
                )
            )
            signed_error = float(
                value.get(
                    "intended_robot_signed_minus_source_reference_distance_mean_m_given_source_contact_5cm",
                    np.nan,
                )
            )
            absolute_error = float(
                value.get(
                    "intended_absolute_distance_error_mean_m_given_source_contact_5cm",
                    np.nan,
                )
            )
            if (
                not np.isfinite([penetration, signed_error, absolute_error]).all()
                or not 0.0 <= penetration <= 1.0
                or absolute_error < 0.0
            ):
                return False
            with np.load(contract_path, allow_pickle=False) as contract:
                contract_values = {
                    name: np.asarray(contract[name]) for name in contract.files
                }
                if (
                    set(contract.files)
                    != {
                        "semantic_names",
                        "source_joint_names",
                        "robot_body_names",
                        "source_joint_positions",
                        "source_object_poses_xyz_wxyz",
                        "source_mesh_scale_xyz",
                        "object_points_local_demo",
                        "object_points_local",
                    }
                    or set(contract["semantic_names"].astype(str))
                    != expected_semantics
                    or contract["source_joint_positions"].shape
                    != (frames, len(expected_semantics), 3)
                    or contract["source_object_poses_xyz_wxyz"].shape != (frames, 7)
                    or not np.isfinite(contract["source_joint_positions"]).all()
                    or not np.isfinite(contract["source_object_poses_xyz_wxyz"]).all()
                    or value.get(
                        "intended_contact_source_contract_canonical_sha256"
                    )
                    != _array_contract_sha256(contract_values)
                ):
                    return False
            case_invariants.setdefault(case, []).append(
                (
                    str(value.get("input_sha256", "")),
                    str(value.get("source_surface_mesh_sha256", "")),
                    str(value.get("intended_contact_mapping_canonical_sha256", "")),
                    str(
                        value.get(
                            "intended_contact_source_contract_canonical_sha256", ""
                        )
                    ),
                )
            )
        try:
            verify_interaction_ablation_pair(root, case)
        except (FileNotFoundError, KeyError, StopIteration, ValueError):
            return False
    numeric = summary.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    rate_columns = [column for column in summary if column.endswith("_frame_rate")]
    rates = summary[rate_columns].to_numpy(dtype=float)
    return bool(
        len(summary) == 4
        and all(len(values) == 2 and len(set(values)) == 1 for values in case_invariants.values())
        and np.isfinite(numeric).all()
        and np.all((rates >= 0.0) & (rates <= 1.0))
    )


def _rerun(root: Path) -> bool:
    from .rerun_visualization import validate_rerun_manifest_contract

    result = validate_rerun_manifest_contract(root, verify_recording=True)
    return bool(
        result.get("result") == "verified"
        and int(result.get("frame_marker_rows", -1))
        == SHARED_COMPARISON_FRAMES
    )


def _publication(root: Path) -> bool:
    manifest = json.loads((root / "manifests/stage1_publication.json").read_text())
    if (
        manifest.get("decision") != "PENDING"
        or manifest.get("decision_authority") != "manifests/stage1_validation.json"
        or manifest.get("core_rows") != 6
        or manifest.get("reference_rows") != 1
        or manifest.get("interaction_rows") != 4
        or manifest.get("presentation_slides", 99) > 15
        or manifest.get("report_files") != list(REPORT_FILES)
    ):
        return False
    inspect_publication_evidence(root)
    figure_root = root / "figures/stage1_publication"
    stems = {path.stem for path in figure_root.glob("*.svg")}
    if len(stems) != 11:
        return False
    for stem in stems:
        if not all(
            path.is_file() and path.stat().st_size > 0
            for path in (
                figure_root / f"{stem}.svg",
                figure_root / f"{stem}.pdf",
                figure_root / f"{stem}.png",
                figure_root / "source_data" / f"{stem}.csv",
            )
        ):
            return False
    return True


def _reports(root: Path) -> bool:
    for name in REPORT_FILES:
        path = root / name
        if not path.is_file() or not path.read_text(encoding="utf-8").rstrip().endswith(
            FULL_LAFAN_STOP_MESSAGE
        ):
            return False
    return presentation_slide_count((root / "PRESENTATION.md").read_text()) <= 15


def _interactive(root: Path) -> bool:
    from .interactive_report import audit_interactive_delivery

    path = root / "INTERACTIVE_REPORT.html"
    delivery = audit_interactive_delivery(root, path, require_rerun=True)
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r'<script id="rtcmp-data" type="application/json">(.*?)</script>',
        text,
        flags=re.DOTALL,
    )
    if match is None or "https://cdn" in text or "__RTCMP_INLINE_" in text:
        return False
    data = json.loads(match.group(1))
    if (
        data.get("decision") != "GO"
        and "support a Pilot-level GO" in text
    ):
        return False
    return bool(
        delivery.get("result") == "verified"
        and delivery.get("embedded_data_sha256") in text
        and data.get("schema_version") == 5
        and data.get("decision") in {"PENDING", "GO", "NO-GO"}
        and data.get("decision_authority", {}).get("path")
        == "manifests/stage1_validation.json"
        and len(data.get("operating_points", [])) == 6
        and data.get("external_references") == ["unitree-reference"]
        and all(f'id="{section}"' in text for section in ("scale", "reference", "budget"))
    )


def _stage0(root: Path) -> bool:
    matrix = pd.read_csv(root / "research/human_to_g1_method_matrix.csv").set_index("name")
    required = {
        "Controlled Sparse-IK", "Controlled Dense-KeyBody IK", "GMR",
        "OmniRetarget / Holosoma", "ProtoMotions v2.3", "ProtoMotions v3",
    }
    if not required.issubset(matrix.index):
        return False
    if not matrix.loc[list(required), "experiment_status"].astype(str).str.contains("completed").all():
        return False
    claims = pd.read_csv(root / "research/claims.csv")
    return bool(
        len(matrix) >= 15
        and len(claims) >= 8
        and {"historical", "experimental", "scope"}.issubset(set(claims.claim_type.astype(str)))
        and (root / "research/lineage_graph.svg").is_file()
        and (root / "research/UNITREE_REFERENCE_CORPUS_AUDIT.md").is_file()
    )


def _test_evidence(root: Path) -> bool:
    value = json.loads((root / "manifests/test_evidence.json").read_text())
    capture = value.get("capture_suite", {})
    critical = value.get("critical_integration_suite", {})
    rerun = value.get("rerun_recording", {})
    return bool(
        int(value.get("schema_version", 0)) >= 5
        and capture.get("result") == "passed"
        and int(capture.get("passed", 0)) >= 55
        and int(capture.get("failed", 1)) == 0
        and critical.get("result") == "passed"
        and critical.get("environment", {}).get("RTCMP_FINAL_TESTS") == "1"
        and int(critical.get("failed", 1)) == 0
        and int(critical.get("skipped", 1)) == 0
        and rerun.get("result") == "verified"
        and int(rerun.get("frames", 0)) == SHARED_COMPARISON_FRAMES
        and int(rerun.get("methods", 0)) == 9
        and int(rerun.get("visual_asset_count", 0)) == 35
        and int(rerun.get("manifest_schema_version", 0)) == 5
        and rerun.get("rrd_cli_verification") == "verified"
    )


def _unitree_reference(root: Path) -> bool:
    from .unitree_reference import (
        AS_PUBLISHED_VIEW,
        CANONICAL_COORDINATE_VIEW,
        UNITREE_REFERENCE_FPS,
        UNITREE_REFERENCE_LABEL,
        UNITREE_REFERENCE_PILOT_AS_PUBLISHED_QPOS_SHA256,
        UNITREE_REFERENCE_PILOT_CANONICAL_QPOS_SHA256,
        UNITREE_REFERENCE_PILOT_SHA256,
        UNITREE_REFERENCE_REPOSITORY,
        UNITREE_REFERENCE_REVISION,
        UNITREE_REFERENCE_ROLE,
        compare_urdf_kinematics,
        compare_urdf_to_mujoco_fk,
        load_unitree_g1_csv,
        qpos_content_sha256,
        transform_reference_coordinate_view,
    )

    manifest_path = root / "manifests/unitree_reference.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = load_yaml(root / "configs/unitree_reference.yaml")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("label") != UNITREE_REFERENCE_LABEL
        or manifest.get("role") != UNITREE_REFERENCE_ROLE
        or manifest.get("verified_official_ground_truth") is not False
        or manifest.get("eligible_for_runtime_comparison") is not False
        or manifest.get("upstream_repository") != UNITREE_REFERENCE_REPOSITORY
        or manifest.get("upstream_revision") != UNITREE_REFERENCE_REVISION
        or config.get("eligible_as_ground_truth") is not False
        or config.get("eligible_for_runtime_comparison") is not False
        or config.get("role") != UNITREE_REFERENCE_ROLE
        or config.get("upstream", {}).get("revision") != UNITREE_REFERENCE_REVISION
    ):
        return False

    raw_entry = manifest.get("raw_pilot", {})
    raw_path = root / str(raw_entry.get("path", ""))
    if (
        str(raw_entry.get("path"))
        != str(config["upstream"]["local_ignored_path"])
        or not raw_path.is_file()
        or raw_path.stat().st_size != int(raw_entry.get("size_bytes", -1))
        or raw_path.stat().st_size != int(config["upstream"]["pilot_size_bytes"])
        or sha256_file(raw_path) != str(raw_entry.get("sha256"))
        or str(raw_entry.get("sha256")) != UNITREE_REFERENCE_PILOT_SHA256
        or str(raw_entry.get("sha256")) != str(config["upstream"]["pilot_sha256"])
    ):
        return False
    raw = np.loadtxt(raw_path, delimiter=",", dtype=np.float64, ndmin=2)
    if (
        raw.shape
        != (int(raw_entry.get("rows", -1)), int(raw_entry.get("columns", -1)))
        or raw.shape != (3945, 36)
        or not np.isfinite(raw).all()
    ):
        return False

    revision_root = root / "data/external/unitree_lafan1_reference" / UNITREE_REFERENCE_REVISION
    expected_resource_paths = {
        "readme": revision_root / "README.md",
        "license": revision_root / "LICENSE",
        "metadata": revision_root / "meta_data/info.json",
        "visualizer": revision_root / "rerun_visualize.py",
        "g1_urdf": revision_root / "robot_description/g1/g1_29dof_rev_1_0.urdf",
    }
    resources = manifest.get("pinned_resources", {})
    if set(resources) != set(expected_resource_paths):
        return False
    for name, expected_path in expected_resource_paths.items():
        entry = resources[name]
        declared = config["upstream"]["pinned_resources"][name]
        if (
            root / str(entry.get("path", "")) != expected_path
            or not expected_path.is_file()
            or expected_path.stat().st_size != int(entry.get("size_bytes", -1))
            or expected_path.stat().st_size != int(declared["size_bytes"])
            or sha256_file(expected_path) != str(entry.get("sha256"))
            or str(entry.get("sha256")) != str(declared["sha256"])
        ):
            return False

    csv_files = sorted((revision_root / "g1").glob("*.csv"))
    inventory_digest = hashlib.sha256()
    for path in csv_files:
        inventory_digest.update(path.name.encode("utf-8"))
        inventory_digest.update(bytes.fromhex(sha256_file(path)))
    inventory = manifest.get("g1_csv_inventory", {})
    if (
        len(csv_files) != 40
        or int(inventory.get("files", -1)) != len(csv_files)
        or inventory.get("aggregate_sha256") != inventory_digest.hexdigest()
    ):
        return False

    source = manifest.get("source_binding", {})
    sequence_path = root / str(source.get("sequence_manifest", ""))
    human_path = root / str(source.get("canonical_source", ""))
    sequence = load_yaml(sequence_path)
    from .schemas import CanonicalHuman

    human = CanonicalHuman.load(human_path)
    if (
        sequence_path != root / "manifests/pilot_sequence.yaml"
        or sha256_file(sequence_path) != str(source.get("sequence_manifest_sha256"))
        or human_path != root / str(sequence.get("canonical_path"))
        or sha256_file(human_path) != str(source.get("canonical_source_file_sha256"))
        or human.source_sha256 != str(source.get("canonical_source_content_sha256"))
        or human.source_sha256 != str(sequence.get("cropped_sha256"))
        or not np.isclose(
            float(source.get("canonical_source_fps", np.nan)),
            float(human.fps),
            rtol=0.0,
            atol=1e-12,
        )
        or source.get("alignment_basis") != "same basename and frame indices 0:600"
        or source.get("byte_identical_human_source_verified") is not False
        or source.get("exact_timestamp_identity_claimed") is not False
        or len(human.timestamps) != EXPECTED_FRAMES
    ):
        return False

    expected_qpos_hash = {
        AS_PUBLISHED_VIEW: UNITREE_REFERENCE_PILOT_AS_PUBLISHED_QPOS_SHA256,
        CANONICAL_COORDINATE_VIEW: UNITREE_REFERENCE_PILOT_CANONICAL_QPOS_SHA256,
    }
    coordinate_views = manifest.get("coordinate_views", {})
    if set(coordinate_views) != set(expected_qpos_hash):
        return False
    loaded_views: dict[str, CanonicalG1] = {}
    for view, content_hash in expected_qpos_hash.items():
        entry = coordinate_views[view]
        output = root / str(entry.get("path", ""))
        if (
            not output.is_file()
            or sha256_file(output) != str(entry.get("sha256"))
            or str(entry.get("qpos_content_sha256")) != content_hash
            or int(entry.get("frames", -1)) != EXPECTED_FRAMES
            or int(entry.get("frame_start", -1)) != 0
            or int(entry.get("frame_end_exclusive", -1)) != EXPECTED_FRAMES
            or float(entry.get("fps", np.nan)) != UNITREE_REFERENCE_FPS
        ):
            return False
        motion = CanonicalG1.load(output)
        motion.validate(source_frame_count=EXPECTED_FRAMES)
        if (
            len(motion.qpos) != EXPECTED_FRAMES
            or not np.array_equal(
                motion.source_frame_idx, np.arange(EXPECTED_FRAMES, dtype=np.int64)
            )
            or float(motion.fps) != UNITREE_REFERENCE_FPS
            or qpos_content_sha256(motion.qpos) != content_hash
            or motion.metadata.get("qpos_content_sha256") != content_hash
            or motion.metadata.get("coordinate_view") != view
            or motion.metadata.get("canonical_source_sha256") != human.source_sha256
            or motion.metadata.get("exact_timestamp_identity_claimed") is not False
            or motion.metadata.get("byte_identical_human_source_verified") is not False
            or motion.metadata.get("timing_available") is not False
            or not np.allclose(motion.per_frame_solve_time_s, 0.0)
        ):
            return False
        reconstructed = load_unitree_g1_csv(
            raw_path,
            frame_start=0,
            frame_end=EXPECTED_FRAMES,
            expected_sha256=UNITREE_REFERENCE_PILOT_SHA256,
            expected_source_frames=EXPECTED_FRAMES,
            canonical_source_sha256=human.source_sha256,
            canonical_source_fps=float(human.fps),
            coordinate_view=view,
        )
        if not np.array_equal(reconstructed.qpos, motion.qpos):
            return False
        loaded_views[view] = motion
    if (
        not np.allclose(
            transform_reference_coordinate_view(loaded_views[AS_PUBLISHED_VIEW].qpos),
            loaded_views[CANONICAL_COORDINATE_VIEW].qpos,
            rtol=0.0,
            atol=1e-12,
        )
        or not np.array_equal(
            loaded_views[AS_PUBLISHED_VIEW].qpos[:, 7:],
            loaded_views[CANONICAL_COORDINATE_VIEW].qpos[:, 7:],
        )
    ):
        return False

    assets = manifest.get("asset_binding", {})
    reference_urdf = root / str(assets.get("reference_urdf", {}).get("path", ""))
    canonical_urdf = root / str(assets.get("canonical_evaluator_urdf_path", ""))
    canonical_scene = root / str(assets.get("canonical_evaluator_scene_path", ""))
    evaluator_path = root / str(assets.get("evaluator_manifest_path", ""))
    stage = load_yaml(root / "configs/stage1.yaml")
    evaluator = load_evaluator_protocol(evaluator_path)
    if (
        assets.get("reference_urdf") != resources["g1_urdf"]
        or canonical_urdf != root / str(stage["canonical_robot"]["urdf"])
        or canonical_scene != root / str(stage["canonical_robot"]["xml"])
        or not canonical_urdf.is_file()
        or sha256_file(canonical_urdf)
        != str(assets.get("canonical_evaluator_urdf_sha256"))
        or not canonical_scene.is_file()
        or sha256_file(canonical_scene)
        != str(assets.get("canonical_evaluator_scene_sha256"))
        or evaluator_path != root / "manifests/evaluator.yaml"
        or sha256_file(evaluator_path) != str(assets.get("evaluator_manifest_sha256"))
        or evaluator.get("robot_xml_sha256") != sha256_file(canonical_scene)
        or evaluator.get("robot_xml") != canonical_scene.relative_to(root).as_posix()
    ):
        return False
    urdf_audit = compare_urdf_kinematics(reference_urdf, canonical_urdf)
    if (
        urdf_audit.get("kinematic_contract_equivalent") is not True
        or urdf_audit.get("canonical_g1_order_match") is not True
        or urdf_audit.get("joint_order_match") is not True
    ):
        return False
    fk_audit = compare_urdf_to_mujoco_fk(
        reference_urdf,
        canonical_scene,
        random_sample_count=10,
        seed=1947,
        position_tolerance_m=1e-5,
        rotation_tolerance_rad=1e-5,
    )
    if (
        fk_audit.get("kinematic_fk_equivalent") is not True
        or int(fk_audit.get("neutral_sample_count", -1)) != 1
        or int(fk_audit.get("random_sample_count", -1)) != 10
        or int(fk_audit.get("link_frames_per_sample", -1)) != 30
    ):
        return False

    adapter = manifest.get("adapter", {})
    adapter_path = root / str(adapter.get("implementation_path", ""))
    if (
        adapter_path != root / "src/retargeting_comparison/unitree_reference.py"
        or not adapter_path.is_file()
        or sha256_file(adapter_path) != str(adapter.get("implementation_sha256"))
        or "no scale/translation/ground/resample" not in str(adapter.get("operations"))
    ):
        return False
    evidence_entries = manifest.get("research_evidence", [])
    expected_evidence = {
        "research/unitree_reference_provenance.csv",
        "research/unitree_reference_pilot_audit.csv",
        "research/UNITREE_REFERENCE_CORPUS_AUDIT.md",
    }
    if (
        not isinstance(evidence_entries, list)
        or {str(entry.get("path")) for entry in evidence_entries}
        != expected_evidence
    ):
        return False
    for entry in evidence_entries:
        path = root / str(entry["path"])
        if not path.is_file() or sha256_file(path) != str(entry.get("sha256")):
            return False
    provenance = pd.read_csv(root / "research/unitree_reference_provenance.csv")
    audit_table = pd.read_csv(root / "research/unitree_reference_pilot_audit.csv")
    audit_text = (root / "research/UNITREE_REFERENCE_CORPUS_AUDIT.md").read_text()

    def evidence_value(category: str, metric: str) -> str:
        row = audit_table.loc[
            audit_table.category.astype(str).eq(category)
            & audit_table.metric.astype(str).eq(metric)
        ]
        if len(row) != 1:
            raise ValueError(f"Missing Unitree audit evidence {category}/{metric}")
        return str(row.iloc[0].value)

    return bool(
        not provenance.empty
        and set(provenance.field.astype(str)).issuperset(
            {"revision", "pilot_sha256", "pilot_rows", "pilot_columns", "fps"}
        )
        and evidence_value("provenance", "revision") == UNITREE_REFERENCE_REVISION
        and evidence_value("provenance", "pilot_csv_sha256")
        == UNITREE_REFERENCE_PILOT_SHA256
        and evidence_value("asset", "actuated_joint_contract_equivalent") == "true"
        and np.isclose(
            float(evidence_value("asset", "cross_engine_fk_max_position_error")),
            float(fk_audit["max_position_error_m"]),
            rtol=0.0,
            atol=1e-12,
        )
        and np.isclose(
            float(evidence_value("asset", "cross_engine_fk_max_rotation_error")),
            float(fk_audit["max_rotation_error_rad"]),
            rtol=0.0,
            atol=1e-12,
        )
        and "Unitree-attributed reference corpus" in audit_text
        and "frame-index aligned" in audit_text
        and audit_text.rstrip().endswith(FULL_LAFAN_STOP_MESSAGE)
    )


def _stage2_projection(root: Path) -> tuple[bool, dict[str, Any]]:
    from .stage2 import (
        discover_lafan_sequences,
        inventory_sha256,
        resolve_methods,
        select_budgeted_design,
        simplify_methods_before_dataset_reduction,
    )

    config = load_yaml(root / "configs/stage2.yaml")
    sequences = discover_lafan_sequences(root, config)
    ready, readiness_exclusions = resolve_methods(root, config)
    active, budget_exclusions, all_method_projection = (
        simplify_methods_before_dataset_reduction(sequences, ready, config)
    )
    design_id, selected, projection, dataset_fallback = select_budgeted_design(
        sequences, active, config
    )
    summary = {
        **{key: value for key, value in projection.items() if key != "jobs"},
        "design_id": design_id,
        "dataset_kind": "full_lafan1" if dataset_fallback is None else "reduced_lafan1",
        "full_inventory_sequence_count": len(sequences),
        "full_inventory_sha256": inventory_sha256(sequences),
        "selected_sequence_count": len(selected),
        "selected_inventory_sha256": inventory_sha256(selected),
        "all_ready_methods": [method.method for method in ready],
        "selected_methods": [method.method for method in active],
        "readiness_exclusions": readiness_exclusions,
        "budget_method_exclusions": budget_exclusions,
        "dataset_fallback": dataset_fallback,
        "all_method_full_projection": {
            key: value for key, value in all_method_projection.items() if key != "jobs"
        },
        "wall_budget_hours": float(config["budget"]["wall_time_hours"]),
        "storage_budget_gb": float(config["budget"]["retained_storage_gb"]),
        "user_authorized": config["authorization"]["explicitly_authorized"],
    }
    atomic_write_json(root / "metrics/stage2_projection.json", summary)
    factor = float(config["budget"]["safety_factor"])
    rows = [
        {
            "method": phase["method"],
            "workers": phase["workers"],
            "jobs": phase["jobs"],
            "raw_projected_wall_hours": phase["projected_makespan_s"] / 3600.0,
            "safe_projected_wall_hours": phase["projected_makespan_s"] * factor / 3600.0,
            "selected_for_stage2": True,
        }
        for phase in projection["phase_summaries"]
    ]
    pd.DataFrame(rows).to_csv(root / "metrics/stage2_projection.csv", index=False)
    acceptable = bool(
        not readiness_exclusions
        and projection["within_budget"]
        and len(selected) == len(sequences)
        and len(sequences) == 77
        and summary["user_authorized"] is True
        and all(
            item.get("result_metrics_used_for_selection") is False
            for item in budget_exclusions
        )
    )
    return acceptable, summary


def _write_artifact_manifest(root: Path) -> None:
    candidates: list[Path] = []
    for folder in ("metrics", "figures", "manifests", "docs", "research", "configs"):
        candidates.extend(path for path in (root / folder).rglob("*") if path.is_file())
    candidates.extend(root / name for name in REPORT_FILES if (root / name).is_file())
    if (root / "INTERACTIVE_REPORT.html").is_file():
        candidates.append(root / "INTERACTIVE_REPORT.html")
        delivery_manifest = root / "INTERACTIVE_REPORT.html.manifest.json"
        if delivery_manifest.is_file():
            candidates.append(delivery_manifest)
    output = root / "manifests/artifacts.csv"
    excluded = {output, root / "manifests/stage1_validation.json"}
    rows = []
    for path in sorted(set(candidates)):
        if path in excluded:
            continue
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("path", "size_bytes", "sha256"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _artifact_manifest(root: Path) -> bool:
    path = root / "manifests/artifacts.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    if len(rows) < 100:
        return False
    for row in rows:
        artifact = root / row["path"]
        if (
            not artifact.is_file()
            or artifact.stat().st_size != int(row["size_bytes"])
            or sha256_file(artifact) != row["sha256"]
        ):
            return False
    return True


def validate_stage1(repo_root: str | Path = ".") -> dict[str, Any]:
    """Validate the frozen expanded Pilot and publish one JSON-native decision."""

    root = Path(repo_root).resolve()
    stage = load_yaml(root / "configs/stage1.yaml")
    sequence = _sequence(root)
    checks: dict[str, bool] = {
        "historical_stage1_hard_stop_preserved": bool(
            stage.get("full_lafan_authorized") is False
            and sequence.get("full_lafan_authorized") is False
            and not list((root / "runs/stage2").glob("*/execution_summary.json"))
        ),
        "nine_canonical_trajectories_complete": _attempt(lambda: _all_motion_outputs(root)),
        "six_formal_run_manifests_hashed": _attempt(lambda: _formal_run_manifests(root)),
        "four_short_smoke_runs_passed": _attempt(lambda: _smoke_runs(root)),
        "body_models_finite_and_hashed": _attempt(lambda: _body_models(root)),
        "source_adapters_passed": _attempt(lambda: _source_adapters(root)),
        "evaluator_protocol_and_metrics_frozen": _attempt(lambda: _evaluator_outputs(root)),
        "six_method_timing_protocol_complete": _attempt(lambda: _timing(root)),
        "exact_native_pre_solver_targets_captured": _attempt(lambda: _native_target_capture(root)),
        "scale_policy_causal_matrix_complete": _attempt(lambda: _scale_evidence(root)),
        "interaction_full_no_hard_complete": _attempt(lambda: _interaction(root)),
        "rerun_nine_robot_comparison_complete": _attempt(lambda: _rerun(root)),
        "publication_metrics_figures_hashed": _attempt(lambda: _publication(root)),
        "eleven_markdown_reports_complete": _attempt(lambda: _reports(root)),
        "interactive_visual_narrative_complete": _attempt(lambda: _interactive(root)),
        "stage0_taxonomy_and_claims_complete": _attempt(lambda: _stage0(root)),
        "unitree_reference_delimited_and_pinned": _attempt(lambda: _unitree_reference(root)),
        "test_evidence_complete": _attempt(lambda: _test_evidence(root)),
    }
    projection_ok, projection = (False, {})
    try:
        projection_ok, projection = _stage2_projection(root)
    except Exception as error:
        projection = {"projection_error": f"{type(error).__name__}: {error}"}
    checks["stage2_design_within_48h_200gb"] = projection_ok

    for document in (
        "research/OFFICIAL_SCALE_AND_PREPROCESSING_AUDIT.md",
        "research/UNITREE_REFERENCE_CORPUS_AUDIT.md",
        "docs/STAGE1_COMPLETION_AUDIT.md",
    ):
        checks[f"historical_stop_marker_{Path(document).stem.lower()}"] = _attempt(
            lambda document=document: (root / document)
            .read_text(encoding="utf-8")
            .rstrip()
            .endswith(FULL_LAFAN_STOP_MESSAGE)
        )

    _write_artifact_manifest(root)
    checks["artifact_manifest_complete_and_hashed"] = _attempt(
        lambda: _artifact_manifest(root)
    )
    decision = "GO" if all(checks.values()) else "NO-GO"
    try:
        publication_binding = inspect_publication_evidence(root)
    except Exception as error:
        publication_binding = {
            "binding_error": f"{type(error).__name__}: {error}"
        }
    result = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "authority": "independent_fail_closed_stage1_validator",
        "decision": decision,
        "checks": checks,
        "check_count": len(checks),
        "passed_check_count": sum(checks.values()),
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "claim_scope": "one frozen 600-frame LAFAN1 Pilot",
        "stage2_projection": projection,
        "stage2_user_authorized_after_this_historical_snapshot": True,
        "hard_stop_message": FULL_LAFAN_STOP_MESSAGE,
        "publication_evidence_binding": publication_binding,
    }
    result["decision_basis_sha256"] = validation_decision_basis_sha256(
        decision, checks, publication_binding
    )
    atomic_write_json(root / "manifests/stage1_validation.json", result)
    return result
