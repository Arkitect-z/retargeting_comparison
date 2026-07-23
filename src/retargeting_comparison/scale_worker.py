"""Dependency-light execution path for native scale-response runs.

This module is intentionally importable in the clean GMR and Holosoma
environments.  In particular, it must not import pandas, Mink, MuJoCo, or the
report/evaluator stack merely to launch a public method.  The richer
``scale_sensitivity`` module delegates native execution here.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .io_utils import sha256_file
from .method_adapters import run_gmr, run_holosoma
from .schemas import CanonicalG1, CanonicalHuman


WITHIN_METHOD_VARIANTS = {
    "native": (1.0, 1.0),
    "root_minus_5": (0.95, 1.0),
    "root_plus_5": (1.05, 1.0),
    "local_minus_5": (1.0, 0.95),
    "local_plus_5": (1.0, 1.05),
}

NATIVE_POLICY_METHODS = (
    "gmr",
    "omniretarget",
    "protomotions_v2_3",
    "protomotions_v3",
)


def _pilot_fields(root: Path) -> dict[str, str]:
    """Read the three scalar Pilot fields needed by a native worker.

    PyYAML is a core harness dependency but is deliberately absent from the
    clean Holosoma environment.  The frozen Pilot manifest uses plain
    top-level scalar fields, so parsing only the registered keys keeps this
    method launcher independent of that unrelated reporting dependency.  The
    canonical file and embedded source hash are validated again below.
    """

    path = root / "manifests/pilot_sequence.yaml"
    required = {"sequence_id", "canonical_path", "cropped_source_file"}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line or raw_line[0].isspace() or ":" not in raw_line:
            continue
        key, raw_value = raw_line.split(":", 1)
        if key in required:
            value = raw_value.strip()
            if not value or value[0] in "[{&*!|>":
                raise ValueError(f"Pilot field {key!r} is not a plain scalar")
            if key in values:
                raise ValueError(f"Pilot field {key!r} is duplicated")
            values[key] = value.strip("'\"")
    missing = sorted(required - set(values))
    if missing:
        raise ValueError(f"Pilot manifest is missing scalar fields: {missing}")
    return values


def native_output_path(
    root: Path, sequence_id: str, method: str, variant: str
) -> Path:
    return (
        root
        / "runs"
        / sequence_id
        / "scale-sensitivity"
        / "native-response"
        / method
        / variant
        / "canonical_g1.npz"
    )


def native_expected_provenance(root: Path, method: str) -> dict[str, Any]:
    """Return immutable upstream/config and robot-asset identities."""

    if method == "gmr":
        config = (
            root
            / "external/GMR/general_motion_retargeting/ik_configs"
            / "bvh_lafan1_to_g1.json"
        )
        solver_xml = root / "external/GMR/assets/unitree_g1/g1_mocap_29dof.xml"
        return {
            "method": "gmr",
            "upstream_commit": "bb1bbe40774794fceb2a7c579a3464a28e68c844",
            "config_sha256": sha256_file(config),
            "solver_robot_asset_field": "robot_xml_sha256",
            "solver_robot_asset_sha256": sha256_file(solver_xml),
        }
    if method in {"omniretarget", "holosoma"}:
        package = (
            root
            / "external/holosoma/src/holosoma_retargeting"
            / "holosoma_retargeting"
        )
        config = package / "config_types/data_type.py"
        solver_urdf = package / "models/g1/g1_29dof.urdf"
        solver_xml = solver_urdf.with_suffix(".xml")
        evaluator_scene = (
            root
            / "external/holosoma/src/holosoma/holosoma/data/robots/g1/scenes"
            / "scene_g1_29dof_wbt_plane.xml"
        )
        return {
            "method": "omniretarget",
            "upstream_commit": "5f48635a3624656a5f46a07df26d43187e59f855",
            "config_sha256": sha256_file(config),
            "solver_robot_urdf_sha256": sha256_file(solver_urdf),
            "solver_robot_asset_field": "solver_robot_xml_sha256",
            "solver_robot_asset_sha256": sha256_file(solver_xml),
            "canonical_evaluator_asset_field": (
                "canonical_evaluator_robot_scene_sha256"
            ),
            "canonical_evaluator_asset_sha256": sha256_file(evaluator_scene),
        }
    if method == "protomotions_v2_3":
        solver_xml = (
            root
            / "external/ProtoMotions-v2.3/protomotions/data/assets/mjcf/g1.xml"
        )
        return {
            "method": "protomotions_v2_3_mink",
            "upstream_commit": "4a905b998101333a2fb91f2de8e2cab4bd0db68e",
            "config_sha256": sha256_file(root / "configs/protomotions_v2.yaml"),
            "solver_robot_asset_field": "native_robot_xml_sha256",
            "solver_robot_asset_sha256": sha256_file(solver_xml),
        }
    if method == "protomotions_v3":
        solver_urdf = (
            root
            / "external/ProtoMotions/protomotions/data/assets/urdf/for_retargeting"
            / "g1.urdf"
        )
        return {
            "method": "protomotions_v3",
            "upstream_commit": "49fe5ad69de67ebbc07ea2b25d41b0f622c15c3c",
            "config_sha256": sha256_file(root / "configs/protomotions_v3.yaml"),
            "solver_robot_asset_field": "native_robot_xml_sha256",
            "solver_robot_asset_sha256": sha256_file(solver_urdf),
        }
    raise ValueError(f"Unsupported scale-provenance method {method!r}")


def scale_execution_contract(root: Path, method: str) -> dict[str, Any]:
    """Hash-bind reusable scale artifacts to code, evaluator, and env lock."""

    adapter = {
        "gmr": root / "src/retargeting_comparison/method_adapters.py",
        "omniretarget": root / "src/retargeting_comparison/method_adapters.py",
        "protomotions_v2_3": root / "src/retargeting_comparison/protomotions_v2.py",
        "protomotions_v3": root / "src/retargeting_comparison/protomotions_v3_worker.py",
    }[method]
    environment_name = {
        "gmr": "robot",
        "omniretarget": "hsretargeting",
        "protomotions_v2_3": "capture",
        "protomotions_v3": "egoallo",
    }[method]
    environment_lock = (
        root / "configs/protomotions_v2.yaml"
        if method == "protomotions_v2_3"
        else root / "configs/protomotions_v3.yaml"
        if method == "protomotions_v3"
        else root / "environments/environment-locks.yaml"
    )
    return {
        "schema_version": 1,
        "harness_path": "src/retargeting_comparison/scale_worker.py",
        "harness_sha256": sha256_file(Path(__file__)),
        "method_adapter_path": str(adapter.relative_to(root)),
        "method_adapter_sha256": sha256_file(adapter),
        "evaluator_implementation_path": "src/retargeting_comparison/evaluator.py",
        "evaluator_implementation_sha256": sha256_file(
            root / "src/retargeting_comparison/evaluator.py"
        ),
        "evaluator_protocol_path": "manifests/evaluator.yaml",
        "evaluator_protocol_sha256": sha256_file(
            root / "manifests/evaluator.yaml"
        ),
        "environment_name": environment_name,
        "environment_lock_path": str(environment_lock.relative_to(root)),
        "environment_lock_sha256": sha256_file(environment_lock),
    }


def runtime_environment_observation() -> dict[str, str]:
    return {
        "environment_name": Path(sys.prefix).name,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }


def _atomic_save_motion(
    motion: CanonicalG1, output: Path, *, source_frame_count: int
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.stem}.", suffix=".npz", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        motion.save(temporary, source_frame_count=source_frame_count)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def native_output_is_current(
    root: Path,
    motion: CanonicalG1,
    human: CanonicalHuman,
    method: str,
    variant: str,
    *,
    experiment_role: str = "native_response_fixed_canonical_contact",
) -> bool:
    """Fail closed unless every registered identity still matches."""

    root_multiplier, local_multiplier = WITHIN_METHOD_VARIANTS[variant]
    expected = native_expected_provenance(root, method)
    expected_frames = 450 if method == "protomotions_v3" else len(human.timestamps)
    metadata = motion.metadata
    solver_field = str(expected["solver_robot_asset_field"])
    asset_match = (
        metadata.get(solver_field) == expected["solver_robot_asset_sha256"]
    )
    if method in {"omniretarget", "holosoma"}:
        asset_match = bool(
            asset_match
            and metadata.get("solver_robot_urdf_sha256")
            == expected["solver_robot_urdf_sha256"]
            and metadata.get(str(expected["canonical_evaluator_asset_field"]))
            == expected["canonical_evaluator_asset_sha256"]
            and metadata.get("solver_and_evaluator_assets_identical") is False
        )
    return bool(
        metadata.get("method") == expected["method"]
        and metadata.get("upstream_commit") == expected["upstream_commit"]
        and metadata.get("config_sha256") == expected["config_sha256"]
        and asset_match
        and metadata.get("canonical_source_sha256") == human.source_sha256
        and metadata.get("canonical_source_file_sha256")
        == sha256_file(
            root / _pilot_fields(root)["canonical_path"]
        )
        and metadata.get("scale_variant") == variant
        and metadata.get("root_scale_multiplier") == root_multiplier
        and metadata.get("local_scale_multiplier") == local_multiplier
        and metadata.get("scale_protocol_sha256")
        == sha256_file(root / "configs/scale_policy_sensitivity.yaml")
        and metadata.get("experiment_role") == experiment_role
        and metadata.get("scale_execution_contract")
        == scale_execution_contract(root, method)
        and isinstance(metadata.get("scale_runtime_environment"), dict)
        and metadata["scale_runtime_environment"].get("environment_name")
        == scale_execution_contract(root, method)["environment_name"]
        and len(motion.qpos) == expected_frames
        and np.array_equal(
            motion.source_frame_idx, np.arange(expected_frames)
        )
        and bool(np.asarray(motion.valid, dtype=bool).all())
    )


def _archive_stale(path: Path) -> None:
    archive = path.with_name(
        f"canonical_g1.pre-provenance-v3-{sha256_file(path)[:12]}.npz"
    )
    if not archive.exists():
        shutil.copy2(path, archive)


def _proto_native_python() -> Path:
    override = os.environ.get("RTCMP_PROTOMOTIONS_V3_PYTHON")
    candidates = [Path(override).expanduser()] if override else []
    executable = Path(sys.executable).resolve()
    if len(executable.parents) >= 3:
        candidates.append(executable.parents[2] / "envs/egoallo/bin/python")
    candidates.append(Path.home() / "anaconda3/envs/egoallo/bin/python")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "ProtoMotions v3 native Python is unavailable; set "
        "RTCMP_PROTOMOTIONS_V3_PYTHON"
    )


def _complete_protomotions_v3_campaign(
    root: Path, fields: dict[str, str], human: CanonicalHuman
) -> None:
    """Complete the four scale solves using the existing official campaign.

    The native point is required to have been produced by the formal timing
    campaign.  The campaign reuses that exact artifact and materializes a
    hash-bound copy at the scale-response path; it never launches a fifth
    scale solve for the native point.
    """

    from .protomotions_v3_campaign import FORMAL_RUN_DIRECTORY, main as campaign_main

    sequence_id = fields["sequence_id"]
    formal = (
        root
        / "runs"
        / sequence_id
        / FORMAL_RUN_DIRECTORY
        / "formal/canonical_g1.npz"
    )
    if not formal.is_file():
        raise FileNotFoundError(
            "ProtoMotions v3 formal timing output is required before its scale campaign"
        )
    keypoints = (
        root / "source_adapters/protomotions_v3" / sequence_id / "keypoints.npy"
    )
    if not keypoints.is_file():
        raise FileNotFoundError(f"Missing audited ProtoMotions v3 input: {keypoints}")
    timing_summary_path = (
        root / "manifests/protomotions_v3_campaign.formal_timing_v3.json"
    )
    if not timing_summary_path.is_file():
        raise FileNotFoundError(
            "ProtoMotions v3 formal timing summary is required before scale variants"
        )
    timing_summary = json.loads(timing_summary_path.read_text(encoding="utf-8"))
    cold = timing_summary.get("independent_cold", {})
    cold_path = root / str(cold.get("path", ""))
    cold_evidence_value = timing_summary.get("independent_cold_evidence")
    if not isinstance(cold_evidence_value, str):
        raise ValueError("Formal ProtoMotions summary lacks independent cold evidence")
    cold_evidence = root / cold_evidence_value
    arguments = [
        "--repo-root",
        str(root),
        "--native-python",
        str(_proto_native_python()),
        "--source",
        str(root / fields["canonical_path"]),
        "--keypoints",
        str(keypoints),
        "--sequence-id",
        sequence_id,
        "--cold-output",
        str(cold_path),
        "--cold-evidence",
        str(cold_evidence),
        "--summary-json",
        "manifests/protomotions_v3_campaign.json",
    ]
    returncode = campaign_main(arguments)
    if returncode != 0:
        raise RuntimeError(
            f"ProtoMotions v3 scale campaign failed with exit code {returncode}"
        )
    native = native_output_path(root, sequence_id, "protomotions_v3", "native")
    if not native.is_file():
        raise RuntimeError("ProtoMotions campaign did not materialize its native scale point")
    motion = CanonicalG1.load(native)
    motion.validate(source_frame_count=len(human.timestamps))
    if not native_output_is_current(
        root, motion, human, "protomotions_v3", "native"
    ):
        raise RuntimeError("Materialized ProtoMotions native scale point failed provenance")


def run_native_response(
    method: str,
    repo_root: str | Path = ".",
    variants: list[str] | None = None,
) -> list[Path]:
    """Run/reuse one public pipeline's registered fixed-contact scale arm."""

    if method not in NATIVE_POLICY_METHODS:
        raise ValueError(f"Unsupported native-response method {method!r}")
    root = Path(repo_root).resolve()
    fields = _pilot_fields(root)
    human = CanonicalHuman.load(root / fields["canonical_path"])
    requested = variants or list(WITHIN_METHOD_VARIANTS)
    unknown = set(requested) - set(WITHIN_METHOD_VARIANTS)
    if unknown:
        raise ValueError(f"Unknown scale variants: {sorted(unknown)}")
    if len(requested) != len(set(requested)):
        raise ValueError("Scale variants must not be duplicated")

    # v3 is a whole-trajectory JAXLS campaign.  Its four non-native points are
    # always executed in the registered order; the formal native point is
    # reused and copied byte-for-byte by the campaign.
    if method == "protomotions_v3":
        needs_campaign = False
        for variant in requested:
            candidate = native_output_path(
                root, fields["sequence_id"], method, variant
            )
            if not candidate.is_file():
                needs_campaign = True
                break
            try:
                candidate_motion = CanonicalG1.load(candidate)
                candidate_motion.validate(
                    source_frame_count=len(human.timestamps)
                )
            except (OSError, ValueError, KeyError):
                needs_campaign = True
                break
            if not native_output_is_current(
                root, candidate_motion, human, method, variant
            ):
                needs_campaign = True
                break
        if needs_campaign:
            _complete_protomotions_v3_campaign(root, fields, human)
        outputs = []
        for variant in requested:
            output = native_output_path(root, fields["sequence_id"], method, variant)
            motion = CanonicalG1.load(output)
            motion.validate(source_frame_count=len(human.timestamps))
            if not native_output_is_current(root, motion, human, method, variant):
                raise RuntimeError(
                    f"ProtoMotions v3 {variant} artifact failed registered provenance"
                )
            outputs.append(output)
        return outputs

    outputs: list[Path] = []
    for variant in requested:
        root_multiplier, local_multiplier = WITHIN_METHOD_VARIANTS[variant]
        output = native_output_path(
            root, fields["sequence_id"], method, variant
        )
        if output.is_file():
            motion = CanonicalG1.load(output)
            motion.validate(source_frame_count=len(human.timestamps))
            if native_output_is_current(root, motion, human, method, variant):
                outputs.append(output)
                continue
            _archive_stale(output)
        started = time.perf_counter()
        if method == "gmr":
            motion = run_gmr(
                root,
                root / fields["cropped_source_file"],
                root / fields["canonical_path"],
                root_scale_multiplier=root_multiplier,
                local_scale_multiplier=local_multiplier,
            )
        elif method == "omniretarget":
            motion = run_holosoma(
                root,
                root / fields["canonical_path"],
                output.parent / "work",
                root_scale_multiplier=root_multiplier,
                local_scale_multiplier=local_multiplier,
                fixed_contact_labels=True,
            )
        else:
            import importlib

            module = importlib.import_module(
                ".protomotions_v2", package=__package__
            )
            function: Callable[..., CanonicalG1] = getattr(
                module, "run_scale_variant"
            )
            motion = function(
                repo_root=root,
                canonical_source=root / fields["canonical_path"],
                output_dir=output.parent / "work",
                root_scale_multiplier=root_multiplier,
                local_scale_multiplier=local_multiplier,
            )
        motion.metadata.update(
            {
                "experiment_role": "native_response_fixed_canonical_contact",
                "scale_variant": variant,
                "root_scale_multiplier": root_multiplier,
                "local_scale_multiplier": local_multiplier,
                "scale_protocol_sha256": sha256_file(
                    root / "configs/scale_policy_sensitivity.yaml"
                ),
                "fixed_contact_labels_for_pure_scale": True,
                "canonical_source_file_sha256": sha256_file(
                    root / fields["canonical_path"]
                ),
                "single_measurement_wall_time_s": time.perf_counter() - started,
                "scale_execution_contract": scale_execution_contract(root, method),
                "scale_runtime_environment": runtime_environment_observation(),
            }
        )
        _atomic_save_motion(
            motion, output, source_frame_count=len(human.timestamps)
        )
        if not native_output_is_current(root, motion, human, method, variant):
            raise RuntimeError(
                f"Fresh {method}/{variant} artifact failed registered provenance"
            )
        outputs.append(output)
    return outputs


def run_holosoma_native_contact_robustness(
    repo_root: str | Path = ".",
) -> list[Path]:
    """Run/reuse the separately labelled native contact-recomputation arm."""

    root = Path(repo_root).resolve()
    fields = _pilot_fields(root)
    human = CanonicalHuman.load(root / fields["canonical_path"])
    scale_hash = sha256_file(root / "configs/scale_policy_sensitivity.yaml")
    outputs: list[Path] = []
    for variant, (root_multiplier, local_multiplier) in WITHIN_METHOD_VARIANTS.items():
        directory = (
            root
            / "runs"
            / fields["sequence_id"]
            / "scale-sensitivity/native-response/omniretarget"
            / variant
        )
        output = directory / "canonical_g1.native-contact-v3.npz"
        if output.is_file():
            motion = CanonicalG1.load(output)
            motion.validate(source_frame_count=len(human.timestamps))
            if native_output_is_current(
                root,
                motion,
                human,
                "omniretarget",
                variant,
                experiment_role="native_response_native_recomputed_contact",
            ) and motion.metadata.get("fixed_contact_labels") is False:
                outputs.append(output)
                continue
            _archive_stale(output)
        started = time.perf_counter()
        motion = run_holosoma(
            root,
            root / fields["canonical_path"],
            directory / "work-native-contact-v3",
            root_scale_multiplier=root_multiplier,
            local_scale_multiplier=local_multiplier,
            fixed_contact_labels=False,
        )
        motion.metadata.update(
            {
                "experiment_role": "native_response_native_recomputed_contact",
                "scale_variant": variant,
                "root_scale_multiplier": root_multiplier,
                "local_scale_multiplier": local_multiplier,
                "scale_protocol_sha256": scale_hash,
                "fixed_contact_labels_for_pure_scale": False,
                "canonical_source_file_sha256": sha256_file(
                    root / fields["canonical_path"]
                ),
                "single_measurement_wall_time_s": time.perf_counter() - started,
                "scale_execution_contract": scale_execution_contract(
                    root, "omniretarget"
                ),
                "scale_runtime_environment": runtime_environment_observation(),
            }
        )
        _atomic_save_motion(
            motion, output, source_frame_count=len(human.timestamps)
        )
        if not native_output_is_current(
            root,
            motion,
            human,
            "omniretarget",
            variant,
            experiment_role="native_response_native_recomputed_contact",
        ):
            raise RuntimeError(
                f"Fresh Holosoma native-contact/{variant} artifact failed provenance"
            )
        outputs.append(output)
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--method", choices=("gmr", "omniretarget"), required=True)
    parser.add_argument("--variant", choices=tuple(WITHIN_METHOD_VARIANTS), required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--native-source")
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--source-count", type=int, required=True)
    args = parser.parse_args(argv)
    root = Path(args.repo_root).resolve()
    # Retain the historical single-worker arguments for subprocess
    # compatibility, but route execution through the same fail-closed path as
    # the public CLI.  The explicit source arguments are checked against the
    # frozen manifest instead of defining an alternate protocol.
    fields = _pilot_fields(root)
    if Path(args.source).resolve() != (root / fields["canonical_path"]).resolve():
        raise ValueError("Scale worker source differs from the frozen Pilot source")
    if args.sequence_id != fields["sequence_id"]:
        raise ValueError("Scale worker sequence differs from the frozen Pilot")
    if int(args.source_count) != len(CanonicalHuman.load(args.source).timestamps):
        raise ValueError("Scale worker source count differs from the canonical source")
    if args.method == "gmr" and Path(args.native_source).resolve() != (
        root / fields["cropped_source_file"]
    ).resolve():
        raise ValueError("GMR native source differs from the frozen crop")
    run_native_response(args.method, root, [args.variant])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
