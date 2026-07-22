"""Pre-solver scale-policy sensitivity experiments for the frozen Pilot.

The module deliberately keeps three claims separate:

* ``native`` is an upstream method with its published target construction;
* ``native_response`` perturbs root or root-relative targets before that solver;
* ``controlled_policy`` ports a published scale formula into one Dense-Mink
  solver and is therefore not an upstream-method result.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .calibration import load_evaluator_protocol
from .controlled_mink import ControlledMinkRetargeter
from .evaluator import evaluate_motion, save_evaluation
from .io_utils import atomic_write_json, load_yaml, sha256_file
from .method_adapters import run_gmr, run_holosoma
from .robot_model import CanonicalRobotModel, default_robot_scene
from .scale_metric_registry import (
    derive_registered_rank_stability,
    derive_registered_scale_slopes,
)
from .schemas import CanonicalG1, CanonicalHuman


WITHIN_METHOD_VARIANTS: dict[str, tuple[float, float]] = {
    "native": (1.0, 1.0),
    "root_minus_5": (0.95, 1.0),
    "root_plus_5": (1.05, 1.0),
    "local_minus_5": (1.0, 0.95),
    "local_plus_5": (1.0, 1.05),
}


def policy_specs(
    common_local_scale: float, common_root_scale: float | None = None
) -> dict[str, dict[str, Any]]:
    """Return frozen target-coordinate formulas in canonical Z-up axes."""

    root_scale = (
        float(common_local_scale)
        if common_root_scale is None
        else float(common_root_scale)
    )
    def uniform(value: float) -> list[float]:
        return [float(value)] * 3

    return {
        "common_shared_semantic_landmark_ls": {
            "root_axis": uniform(root_scale),
            "local_axes": {
                "root_torso_legs": uniform(common_local_scale),
                "arms": uniform(common_local_scale),
            },
            "evidence_role": "benchmark_common_shared_semantic_landmark_ls",
            "root_local_parameters_separate": True,
        },
        "holosoma_lafan_uniform": {
            "root_axis": uniform(1.27 / 1.70),
            "local_axes": {
                "root_torso_legs": uniform(1.27 / 1.70),
                "arms": uniform(1.27 / 1.70),
            },
            "evidence_role": "official_formula_transplant",
        },
        "gmr_region_wise": {
            "root_axis": uniform(0.90 * 1.75 / 1.80),
            "local_axes": {
                "root_torso_legs": uniform(0.90 * 1.75 / 1.80),
                "arms": uniform(0.75 * 1.75 / 1.80),
            },
            "evidence_role": "official_formula_transplant",
        },
        "protomotions_v2_3_world_axis": {
            "root_axis": [0.75, 1.00, 0.80],
            "local_axes": {
                "root_torso_legs": [0.75, 1.00, 0.80],
                "arms": [0.75, 1.00, 0.80],
            },
            "evidence_role": "official_amass_formula_transplant",
        },
        "protomotions_v3_lower_upper_axis": {
            "root_axis": [0.90, 0.90, 0.85],
            "local_axes": {
                "root_torso_legs": [0.90, 0.90, 0.85],
                "arms": [0.90, 0.90, 0.80],
            },
            "evidence_role": (
                "official_amass_formula_inspired_dense_extrapolation"
            ),
        },
    }


NATIVE_POLICY = {
    "gmr": "gmr_region_wise",
    "omniretarget": "holosoma_lafan_uniform",
    "protomotions_v2_3": "protomotions_v2_3_world_axis",
    "protomotions_v3": "protomotions_v3_lower_upper_axis",
}


def _policy_specs_from_protocol(protocol: dict[str, Any]) -> dict[str, dict[str, Any]]:
    scale = protocol["scale"]
    return policy_specs(
        float(scale["common_local_body_scale"]),
        float(scale["common_root_displacement_scale"]),
    )


def _scale_axes(policy: dict[str, Any], root_multiplier: float, local_multiplier: float):
    value = {
        "root_axis": (np.asarray(policy["root_axis"], dtype=float) * root_multiplier).tolist(),
        "local_axes": {
            name: (np.asarray(axis, dtype=float) * local_multiplier).tolist()
            for name, axis in policy["local_axes"].items()
        },
        "evidence_role": policy["evidence_role"],
    }
    return value


def transform_human(
    human: CanonicalHuman, root_multiplier: float, local_multiplier: float
) -> CanonicalHuman:
    """Perturb root path and local geometry about the frozen frame-zero root."""

    positions = np.asarray(human.world_positions, dtype=np.float64)
    roots = positions[:, 0]
    anchor = roots[0]
    scaled_roots = anchor + (roots - anchor) * root_multiplier
    scaled_positions = scaled_roots[:, None, :] + (
        positions - roots[:, None, :]
    ) * local_multiplier
    result = CanonicalHuman(
        joint_names=human.joint_names,
        parent_indices=human.parent_indices,
        local_rotations=human.local_rotations,
        world_rotations=human.world_rotations,
        world_positions=scaled_positions,
        root_translation=scaled_roots,
        fps=human.fps,
        timestamps=human.timestamps,
        # Fixed-label arm: no target perturbation is allowed to silently
        # change the evaluator's definition of stance.
        foot_contact_labels=human.foot_contact_labels,
        source_sha256=human.source_sha256,
    )
    result.validate()
    return result


def _target_array(
    root: Path, human: CanonicalHuman, policy: dict[str, Any]
) -> tuple[list[str], np.ndarray]:
    config = load_yaml(root / "configs" / "controlled_mink.yaml")
    evaluator = load_evaluator_protocol(root / "manifests" / "evaluator.yaml")
    specs = config["target_sets"]["dense"]
    indices = {name: index for index, name in enumerate(human.joint_names.astype(str))}
    source_root = human.world_positions[:, 0]
    source_root0 = source_root[0]
    common_root = float(config["common"]["root_displacement_scale"])
    alignment = np.asarray(
        evaluator["scale"]["common_root_alignment_translation_m"], dtype=float
    )
    anchor = source_root0 * common_root + alignment
    root_axis = np.asarray(policy["root_axis"], dtype=float)
    robot_root = anchor + (source_root - source_root0) * root_axis
    values = []
    labels = []
    for spec in specs:
        labels.append(str(spec["semantic"]))
        local_axis = np.asarray(policy["local_axes"][spec["scale_group"]], dtype=float)
        joint = human.world_positions[:, indices[spec["human_joint"]]]
        values.append(robot_root + (joint - source_root) * local_axis)
    return labels, np.stack(values, axis=1)


def _array_sha256(value: np.ndarray, labels: list[str]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(labels, separators=(",", ":")).encode())
    array = np.ascontiguousarray(value, dtype=np.float64)
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _target_geometry(
    labels: list[str],
    target: np.ndarray,
    root_trajectory: np.ndarray | None = None,
) -> dict[str, float]:
    if "root" in labels:
        root = target[:, labels.index("root")]
    elif root_trajectory is not None:
        root = np.asarray(root_trajectory, dtype=np.float64)
        if root.shape != (len(target), 3):
            raise ValueError("root_trajectory must have shape [T,3]")
    else:
        raise ValueError("Target geometry requires a root label or root trajectory")
    local = target - root[:, None, :]
    return {
        "root_path_m": float(np.linalg.norm(np.diff(root[:, :2], axis=0), axis=1).sum()),
        "mean_root_relative_radius_m": float(np.linalg.norm(local, axis=2).mean()),
        "max_root_relative_radius_m": float(np.linalg.norm(local, axis=2).max()),
        "mean_target_z_m": float(target[..., 2].mean()),
        "min_target_z_m": float(target[..., 2].min()),
        "max_target_z_m": float(target[..., 2].max()),
    }


def build_pre_solver_target_contract(repo_root: str | Path = ".") -> pd.DataFrame:
    root = Path(repo_root).resolve()
    sequence = load_yaml(root / "manifests" / "pilot_sequence.yaml")
    human = CanonicalHuman.load(root / sequence["canonical_path"])
    evaluator = load_evaluator_protocol(root / "manifests" / "evaluator.yaml")
    policies = _policy_specs_from_protocol(evaluator)
    target_dir = root / "runs" / sequence["sequence_id"] / "scale-sensitivity" / "targets"
    target_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    scopes = {"controlled_dense": list(policies)}
    scopes.update({method: [policy] for method, policy in NATIVE_POLICY.items()})
    for method, method_policies in scopes.items():
        variants = WITHIN_METHOD_VARIANTS if method != "controlled_dense" else {"native": (1.0, 1.0)}
        for policy_id in method_policies:
            for variant, (root_multiplier, local_multiplier) in variants.items():
                policy = _scale_axes(
                    policies[policy_id], root_multiplier, local_multiplier
                )
                labels, target = _target_array(root, human, policy)
                output = target_dir / f"{method}__{policy_id}__{variant}.npz"
                np.savez_compressed(
                    output,
                    target_positions=target,
                    landmark_names=np.asarray(labels),
                    timestamps=human.timestamps,
                    root_axis=np.asarray(policy["root_axis"]),
                    local_root_torso_legs=np.asarray(policy["local_axes"]["root_torso_legs"]),
                    local_arms=np.asarray(policy["local_axes"]["arms"]),
                    common_root_alignment_translation_m=np.asarray(
                        evaluator["scale"]["common_root_alignment_translation_m"],
                        dtype=np.float64,
                    ),
                )
                rows.append(
                    {
                        "method_scope": method,
                        "policy_id": policy_id,
                        "variant": variant,
                        "evidence_role": policy["evidence_role"],
                        "frames": len(target),
                        "landmarks": len(labels),
                        "root_multiplier": root_multiplier,
                        "local_multiplier": local_multiplier,
                        "root_axis": json.dumps(policy["root_axis"]),
                        "local_root_torso_legs": json.dumps(policy["local_axes"]["root_torso_legs"]),
                        "local_arms": json.dumps(policy["local_axes"]["arms"]),
                        "target_sha256": _array_sha256(target, labels),
                        "artifact_path": str(output.relative_to(root)),
                        "artifact_file_sha256": sha256_file(output),
                        "source_sha256": human.source_sha256,
                        "robot_asset_sha256": evaluator["robot_xml_sha256"],
                        "robot_joint_order_sha256": evaluator[
                            "robot_joint_order_sha256"
                        ],
                        "common_scale_definition": evaluator["scale"]["definition"],
                        "common_root_displacement_scale": evaluator["scale"][
                            "common_root_displacement_scale"
                        ],
                        "common_local_body_scale": evaluator["scale"][
                            "common_local_body_scale"
                        ],
                        "root_anchor_policy": evaluator["scale"][
                            "root_anchor_policy"
                        ],
                        "head_to_toe_role": evaluator["scale"]["diagnostics"][
                            "head_to_toe_role"
                        ],
                        **_target_geometry(labels, target),
                    }
                )
    frame = pd.DataFrame(rows).sort_values(["method_scope", "policy_id", "variant"])
    output = root / "manifests" / "pre_solver_targets.csv"
    frame.to_csv(output, index=False)
    frame.to_csv(root / "metrics" / "pre_solver_target_geometry.csv", index=False)
    return frame


def run_controlled_policy_transplants(repo_root: str | Path = ".") -> list[Path]:
    root = Path(repo_root).resolve()
    sequence = load_yaml(root / "manifests" / "pilot_sequence.yaml")
    human = CanonicalHuman.load(root / sequence["canonical_path"])
    evaluator = load_evaluator_protocol(root / "manifests" / "evaluator.yaml")
    config_hash = sha256_file(root / "configs" / "controlled_mink.yaml")
    evaluator_hash = sha256_file(root / "manifests" / "evaluator.yaml")
    outputs = []
    for policy_id, policy in _policy_specs_from_protocol(evaluator).items():
        run_dir = (
            root
            / "runs"
            / sequence["sequence_id"]
            / "scale-sensitivity"
            / "controlled-dense"
            / policy_id
        )
        output = run_dir / "canonical_g1.npz"
        if output.is_file():
            motion = CanonicalG1.load(output)
            motion.validate(source_frame_count=len(human.timestamps))
            current = bool(
                motion.metadata.get("policy_id") == policy_id
                and motion.metadata.get("config_sha256") == config_hash
                and motion.metadata.get("evaluator_sha256") == evaluator_hash
                and motion.metadata.get("canonical_source_sha256") == human.source_sha256
            )
            if current:
                outputs.append(output)
                continue
            archive = output.with_name(
                f"canonical_g1.pre-v6-{sha256_file(output)[:12]}.npz"
            )
            if not archive.exists():
                shutil.copy2(output, archive)
        start = time.perf_counter()
        motion = ControlledMinkRetargeter(
            root,
            "dense",
            scale_policy=policy,
            method_label=f"controlled-policy/{policy_id}",
        ).run(human)
        motion.metadata.update(
            {
                "experiment_role": "controlled_policy_ablation",
                "policy_id": policy_id,
                "scale_variant": "native",
                "root_scale_multiplier": 1.0,
                "local_scale_multiplier": 1.0,
                "official_method_label": False,
                "fixed_contact_labels": True,
                "contact_label_policy": "no contact objective in controlled Dense solver",
                "canonical_robot_xml_sha256": evaluator["robot_xml_sha256"],
                "canonical_joint_order_sha256": evaluator[
                    "robot_joint_order_sha256"
                ],
                "scale_protocol_sha256": sha256_file(
                    root / "configs/scale_policy_sensitivity.yaml"
                ),
                "common_scale_definition": evaluator["scale"]["definition"],
                "head_to_toe_role": evaluator["scale"]["diagnostics"][
                    "head_to_toe_role"
                ],
                "single_measurement_wall_time_s": time.perf_counter() - start,
            }
        )
        motion.save(output, source_frame_count=len(human.timestamps))
        outputs.append(output)
    return outputs


def _native_output_path(root: Path, sequence_id: str, method: str, variant: str) -> Path:
    from .scale_worker import native_output_path

    return native_output_path(root, sequence_id, method, variant)


def _native_expected_provenance(root: Path, method: str) -> dict[str, Any]:
    from .scale_worker import native_expected_provenance

    return native_expected_provenance(root, method)


def _native_output_is_current(
    root: Path,
    motion: CanonicalG1,
    human: CanonicalHuman,
    method: str,
    variant: str,
) -> bool:
    from .scale_worker import native_output_is_current

    return native_output_is_current(root, motion, human, method, variant)


def run_native_response(
    method: str,
    repo_root: str | Path = ".",
    variants: list[str] | None = None,
) -> list[Path]:
    """Run one public pipeline's pre-solver root/local response arm."""

    from .scale_worker import run_native_response as run_dependency_light

    return run_dependency_light(method, repo_root, variants)

    # Kept below temporarily as historical implementation context; execution
    # is centralized in the dependency-light, fail-closed worker above.
    root = Path(repo_root).resolve()
    sequence = load_yaml(root / "manifests" / "pilot_sequence.yaml")
    human = CanonicalHuman.load(root / sequence["canonical_path"])
    requested = variants or list(WITHIN_METHOD_VARIANTS)
    unknown = set(requested) - set(WITHIN_METHOD_VARIANTS)
    if unknown:
        raise ValueError(f"Unknown scale variants: {sorted(unknown)}")
    outputs = []
    for variant in requested:
        root_multiplier, local_multiplier = WITHIN_METHOD_VARIANTS[variant]
        output = _native_output_path(root, sequence["sequence_id"], method, variant)
        if output.is_file():
            motion = CanonicalG1.load(output)
            motion.validate(source_frame_count=len(human.timestamps))
            if _native_output_is_current(root, motion, human, method, variant):
                outputs.append(output)
                continue
            archive = output.with_name(
                f"canonical_g1.pre-provenance-v2-{sha256_file(output)[:12]}.npz"
            )
            if not archive.exists():
                shutil.copy2(output, archive)
        start = time.perf_counter()
        if method == "gmr":
            motion = run_gmr(
                root,
                root / sequence["cropped_source_file"],
                root / sequence["canonical_path"],
                root_scale_multiplier=root_multiplier,
                local_scale_multiplier=local_multiplier,
            )
        elif method in {"omniretarget", "holosoma"}:
            work = output.parent / "work"
            motion = run_holosoma(
                root,
                root / sequence["canonical_path"],
                work,
                root_scale_multiplier=root_multiplier,
                local_scale_multiplier=local_multiplier,
                fixed_contact_labels=True,
            )
        else:
            module_name = {
                "protomotions_v2_3": ".protomotions_v2",
                "protomotions_v3": ".protomotions_v3",
            }.get(method)
            if module_name is None:
                raise ValueError(f"Unsupported native-response method {method!r}")
            import importlib

            module = importlib.import_module(module_name, package=__package__)
            function: Callable[..., CanonicalG1] = getattr(module, "run_scale_variant")
            motion = function(
                repo_root=root,
                canonical_source=root / sequence["canonical_path"],
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
                "single_measurement_wall_time_s": time.perf_counter() - start,
            }
        )
        motion.save(output, source_frame_count=len(human.timestamps))
        outputs.append(output)
    return outputs


def run_holosoma_native_contact_robustness(
    repo_root: str | Path = ".",
) -> list[Path]:
    """Run the separately labelled native contact-recomputation robustness arm."""

    from .scale_worker import (
        run_holosoma_native_contact_robustness as run_dependency_light,
    )

    return run_dependency_light(repo_root)

    # Historical body retained for provenance archaeology; unreachable.
    root = Path(repo_root).resolve()
    sequence = load_yaml(root / "manifests/pilot_sequence.yaml")
    human = CanonicalHuman.load(root / sequence["canonical_path"])
    scale_hash = sha256_file(root / "configs/scale_policy_sensitivity.yaml")
    expected = _native_expected_provenance(root, "omniretarget")
    outputs: list[Path] = []
    for variant, (root_multiplier, local_multiplier) in WITHIN_METHOD_VARIANTS.items():
        directory = (
            root / "runs" / sequence["sequence_id"] / "scale-sensitivity"
            / "native-response" / "omniretarget" / variant
        )
        output = directory / "canonical_g1.native-contact-v3.npz"
        if output.is_file():
            motion = CanonicalG1.load(output)
            metadata = motion.metadata
            current = bool(
                all(metadata.get(key) == value for key, value in expected.items())
                and metadata.get("canonical_source_sha256") == human.source_sha256
                and metadata.get("scale_variant") == variant
                and metadata.get("scale_protocol_sha256") == scale_hash
                and metadata.get("fixed_contact_labels") is False
                and metadata.get("experiment_role")
                == "native_response_native_recomputed_contact"
                and len(motion.qpos) == len(human.timestamps)
            )
            if current:
                outputs.append(output)
                continue
        started = time.perf_counter()
        motion = run_holosoma(
            root,
            root / sequence["canonical_path"],
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
                "single_measurement_wall_time_s": time.perf_counter() - started,
            }
        )
        motion.save(output, source_frame_count=len(human.timestamps))
        outputs.append(output)
    return outputs


def build_contact_flip_diagnostics(repo_root: str | Path = ".") -> pd.DataFrame:
    root = Path(repo_root).resolve()
    sequence = load_yaml(root / "manifests" / "pilot_sequence.yaml")
    human = CanonicalHuman.load(root / sequence["canonical_path"])
    rows = []
    names = {name: index for index, name in enumerate(human.joint_names.astype(str))}
    toe_indices = [names["LeftToe"], names["RightToe"]]

    def holosoma_labels(value: CanonicalHuman) -> np.ndarray:
        # Exact semantics of upstream
        # extract_foot_sticking_sequence_velocity: frame displacement (not
        # metres/second), 1 cm threshold, and frame zero forced false.
        feet = value.world_positions[:, toe_indices, :2] * (1.27 / 1.70)
        speed = np.linalg.norm(np.diff(feet, axis=0), axis=2)
        speed = np.vstack((np.full((1, 2), 0.0100000001), speed))
        return speed <= 0.01

    native_holosoma = holosoma_labels(human)
    native_holosoma_edges = np.diff(native_holosoma.astype(np.int8), axis=0) != 0
    formal_canonical_contacts = np.asarray(human.foot_contact_labels, dtype=bool)
    for method in NATIVE_POLICY:
        for variant, (root_multiplier, local_multiplier) in WITHIN_METHOD_VARIANTS.items():
            applicable = method == "omniretarget"
            if applicable:
                transformed = transform_human(human, root_multiplier, local_multiplier)
                recomputed = holosoma_labels(transformed)
                edges = np.diff(recomputed.astype(np.int8), axis=0) != 0
                flip = float(np.mean(recomputed != native_holosoma))
                edge_flip = float(np.mean(edges != native_holosoma_edges))
                baseline_fraction = float(np.mean(native_holosoma))
                recomputed_fraction = float(np.mean(recomputed))
                native_vs_formal = float(
                    np.mean(recomputed != formal_canonical_contacts)
                )
                policy = "native toe-displacement labels recomputed after scale"
            elif method == "protomotions_v3":
                flip = edge_flip = 0.0
                baseline_fraction = recomputed_fraction = float(
                    np.mean(human.foot_contact_labels)
                )
                native_vs_formal = 0.0
                policy = "contacts constructed before target scale and therefore fixed"
            else:
                flip = edge_flip = baseline_fraction = recomputed_fraction = 0.0
                native_vs_formal = 0.0
                policy = "no contact objective in this retargeter"
            rows.append(
                {
                    "method": method,
                    "variant": variant,
                    "fixed_label_policy_for_formal_run": True,
                    "native_contact_recomputation_applicable": applicable,
                    "native_contact_policy": policy,
                    "recomputed_label_flip_rate": flip,
                    "recomputed_constraint_edge_flip_rate": edge_flip,
                    "baseline_contact_fraction": baseline_fraction,
                    "recomputed_contact_fraction": recomputed_fraction,
                    "formal_fixed_canonical_contact_fraction": float(
                        np.mean(formal_canonical_contacts)
                    ),
                    "native_recomputed_vs_formal_canonical_label_disagreement_rate": native_vs_formal,
                }
            )
    frame = pd.DataFrame(rows).sort_values(["method", "variant"])
    frame.to_csv(root / "metrics" / "contact_constraint_flip.csv", index=False)
    return frame


def build_actor_shape_probe(repo_root: str | Path = ".") -> pd.DataFrame:
    """Reconstruct deterministic AMASS actor-shape policy formulas.

    This is deliberately not an AMASS motion experiment.  A neutral SMPL-X
    model is evaluated at one frozen pose and one frozen five-frame root path.
    Formulas transcribed from hash-bound official source files construct a
    static diagnostic target tensor.  No native runtime constructor or solver
    is invoked, so this evidence may explain AMASS shape-policy differences but
    may not be used as exact pre-solver, accuracy, or ranking evidence.
    """

    import smplx
    import torch

    root = Path(repo_root).resolve()
    body_root = root.parent / "body_models"
    model_path = body_root / "smplx" / "SMPLX_NEUTRAL.npz"
    model = smplx.create(
        str(body_root),
        model_type="smplx",
        gender="neutral",
        ext="npz",
        use_pca=False,
        batch_size=1,
    )
    from smplx.joint_names import JOINT_NAMES

    profiles = {"short": -2.0, "zero": 0.0, "tall": 2.0}
    shape_values: dict[str, dict[str, Any]] = {}
    joint_index = {name: index for index, name in enumerate(JOINT_NAMES)}
    semantic_joint = {
        "root": "pelvis",
        "torso": "spine2",
        "spine1": "spine1",
        "head": "head",
        "left_shoulder": "left_shoulder",
        "right_shoulder": "right_shoulder",
        "left_elbow": "left_elbow",
        "right_elbow": "right_elbow",
        "left_wrist": "left_wrist",
        "right_wrist": "right_wrist",
        "left_hip": "left_hip",
        "right_hip": "right_hip",
        "left_knee": "left_knee",
        "right_knee": "right_knee",
        "left_ankle": "left_ankle",
        "right_ankle": "right_ankle",
        "left_toe": "left_foot",
        "right_toe": "right_foot",
    }
    frozen_root_path = np.asarray(
        [[0.0, 0.0, 0.0], [0.005, 0.0, 0.0], [0.020, 0.0, 0.0],
         [0.025, 0.0, 0.0], [0.040, 0.0, 0.0]],
        dtype=np.float64,
    )
    for profile, beta0 in profiles.items():
        betas = torch.zeros((1, int(model.num_betas)), dtype=torch.float32)
        betas[0, 0] = beta0
        with torch.no_grad():
            result = model(
                betas=betas,
                expression=torch.zeros((1, int(model.num_expression_coeffs))),
                return_verts=True,
            )
        vertices = result.vertices[0].detach().cpu().numpy()
        joints = result.joints[0].detach().cpu().numpy()
        mesh_height = float(vertices[:, 1].max() - vertices[:, 1].min())
        joint_span = float(joints[:, 1].max() - joints[:, 1].min())
        # SMPL-X is Y-up.  This fixed proper-axis view is used only for target
        # geometry: x stays x, source z becomes -y, source y becomes z.
        canonical = joints[:, [0, 2, 1]].astype(np.float64)
        canonical[:, 1] *= -1.0
        selected = {
            semantic: canonical[joint_index[name]].copy()
            for semantic, name in semantic_joint.items()
        }
        pelvis = selected["root"].copy()
        selected = {name: value - pelvis for name, value in selected.items()}
        shape_values[profile] = {
            "mesh_height_m": mesh_height,
            "joint_vertical_span_m": joint_span,
            "semantic_positions": selected,
        }

    target_sets = {
        "gmr": [
            "root", "left_hip", "right_hip", "left_knee", "right_knee",
            "left_ankle", "right_ankle", "torso", "left_shoulder",
            "right_shoulder", "left_elbow", "right_elbow", "left_wrist",
            "right_wrist",
        ],
        "omniretarget": [
            "spine1", "left_hip", "right_hip", "left_knee", "right_knee",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_ankle", "right_ankle", "left_toe", "right_toe",
            "left_wrist", "right_wrist",
        ],
        "protomotions_v2_3": [
            "root", "head", "left_hip", "right_hip", "left_knee",
            "right_knee", "left_ankle", "right_ankle", "left_elbow",
            "right_elbow", "left_wrist", "right_wrist", "left_shoulder",
            "right_shoulder",
        ],
        "protomotions_v3": [
            "root", "left_hip", "right_hip", "left_knee", "right_knee",
            "left_ankle", "right_ankle", "left_toe", "right_toe",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
        ],
    }
    official_formula_sources = {
        "gmr": (
            "external/GMR/general_motion_retargeting/utils/smpl.py",
            "external/GMR/general_motion_retargeting/motion_retarget.py",
            "external/GMR/general_motion_retargeting/ik_configs/smplx_to_g1.json",
        ),
        "omniretarget": (
            "external/holosoma/src/holosoma_retargeting/holosoma_retargeting/data_utils/prep_amass_smplx_for_rt.py",
            "external/holosoma/src/holosoma_retargeting/holosoma_retargeting/config_types/robot.py",
            "external/holosoma/src/holosoma_retargeting/holosoma_retargeting/src/utils.py",
        ),
        "protomotions_v2_3": (
            "external/ProtoMotions-v2.3/data/scripts/retargeting/config.py",
            "external/ProtoMotions-v2.3/data/scripts/retargeting/mink_retarget.py",
        ),
        "protomotions_v3": (
            "external/ProtoMotions/data/scripts/convert_amass_to_proto.py",
            "external/ProtoMotions/data/scripts/keypoint_utils.py",
            "external/ProtoMotions/pyroki/batch_retarget_to_g1_from_keypoints.py",
        ),
    }
    official_formula_hashes = {
        method: {path: sha256_file(root / path) for path in paths}
        for method, paths in official_formula_sources.items()
    }
    probe_root = (
        root / "runs" / "stage1-actor-shape-formula-probe" / "reconstructions"
    )
    rows: list[dict[str, Any]] = []
    contact_by_method_profile: dict[tuple[str, str], np.ndarray] = {}
    for profile, beta0 in profiles.items():
        observed = shape_values[profile]
        for method, labels in target_sets.items():
            consumes_shape = method in {"gmr", "omniretarget"}
            geometry_profile = profile if consumes_shape else "zero"
            geometry = shape_values[geometry_profile]["semantic_positions"]
            local = np.stack([geometry[name] for name in labels], axis=0)
            if method == "gmr":
                height = 1.66 + 0.1 * beta0
                root_axis = np.full(3, 0.9 * height / 1.8)
                torso_axis = root_axis.copy()
                arm_axis = np.full(3, 0.8 * height / 1.8)
                arm_names = {
                    "left_shoulder", "right_shoulder", "left_elbow",
                    "right_elbow", "left_wrist", "right_wrist",
                }
                local_axes = np.stack(
                    [arm_axis if name in arm_names else torso_axis for name in labels]
                )
                height_definition = "1.66 + 0.1*beta0"
                contact_policy = "no contact objective"
            elif method == "omniretarget":
                height = float(observed["mesh_height_m"])
                factor = 1.32 / height
                root_axis = np.full(3, factor)
                local_axes = np.broadcast_to(root_axis, (len(labels), 3)).copy()
                height_definition = "neutral SMPL-X zero-pose mesh vertical extent"
                contact_policy = "post-scale foot displacement <= 0.01 m/frame"
            elif method == "protomotions_v2_3":
                height = float(shape_values["zero"]["mesh_height_m"])
                root_axis = np.asarray([0.75, 1.0, 0.8])
                local_axes = np.broadcast_to(root_axis, (len(labels), 3)).copy()
                height_definition = "actor shape discarded; zero-beta neutral proxy"
                contact_policy = "no contact objective"
            else:
                height = float(shape_values["zero"]["mesh_height_m"])
                root_axis = np.asarray([0.9, 0.9, 0.85])
                upper_axis = np.asarray([0.9, 0.9, 0.8])
                upper = {
                    "left_shoulder", "right_shoulder", "left_elbow",
                    "right_elbow", "left_wrist", "right_wrist",
                }
                local_axes = np.stack(
                    [upper_axis if name in upper else root_axis for name in labels]
                )
                height_definition = "actor shape discarded; fixed static SMPL skeleton"
                contact_policy = "fixed-skeleton speed/height labels before scale"
            target = frozen_root_path[:, None, :] * root_axis + local[None] * local_axes[None]
            feet = np.stack(
                [
                    frozen_root_path + geometry["left_toe"],
                    frozen_root_path + geometry["right_toe"],
                ],
                axis=1,
            )
            if method == "omniretarget":
                feet = feet * root_axis
                displacement = np.linalg.norm(np.diff(feet[:, :, :2], axis=0), axis=2)
                contact = np.vstack((np.zeros((1, 2), dtype=bool), displacement <= 0.01))
            elif method == "protomotions_v3":
                displacement = np.linalg.norm(np.diff(feet[:, :, :2], axis=0), axis=2)
                near_ground = feet[:, :, 2] <= float(feet[:, :, 2].min()) + 0.05
                contact = np.vstack((np.zeros((1, 2), dtype=bool), displacement <= 0.01))
                contact &= near_ground
            else:
                contact = np.zeros((len(frozen_root_path), 2), dtype=bool)
            contact_by_method_profile[(method, profile)] = contact
            artifact = probe_root / method / f"{profile}.npz"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                artifact,
                target_positions=target,
                target_labels=np.asarray(labels),
                contact_labels=contact,
                frozen_root_path=frozen_root_path,
                beta0=np.asarray(beta0),
                geometry_profile=np.asarray(geometry_profile),
            )
            target_geometry = _target_geometry(
                labels,
                target,
                frozen_root_path * root_axis,
            )
            rows.append(
                {
                    "profile": profile,
                    "beta0": beta0,
                    "method": method,
                    "method_height_definition": height_definition,
                    "method_height_m": height,
                    "mesh_height_m": float(observed["mesh_height_m"]),
                    "joint_vertical_span_m": float(observed["joint_vertical_span_m"]),
                    "actor_shape_consumed": consumes_shape,
                    "geometry_profile_used": geometry_profile,
                    "motion_pose_and_root_path_frozen": True,
                    "root_gain_xyz": json.dumps(root_axis.tolist()),
                    "local_gain_xyz_by_landmark": json.dumps(
                        dict(zip(labels, local_axes.tolist())), sort_keys=True
                    ),
                    "target_count": len(labels),
                    "target_sha256": _array_sha256(target, labels),
                    "target_artifact": str(artifact.relative_to(root)),
                    "target_artifact_sha256": sha256_file(artifact),
                    "contact_policy": contact_policy,
                    "contact_fraction": float(contact.mean()),
                    "body_model_sha256": sha256_file(model_path),
                    "official_formula_source_paths_json": json.dumps(
                        list(official_formula_sources[method])
                    ),
                    "official_formula_source_sha256s_json": json.dumps(
                        official_formula_hashes[method], sort_keys=True
                    ),
                    "official_formula_source_bundle_sha256": hashlib.sha256(
                        json.dumps(
                            official_formula_hashes[method],
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest(),
                    "constructor_evidence_grade": (
                        "formula_reconstruction_bound_to_official_source_hash"
                    ),
                    "runtime_constructor_observed": False,
                    "not_used_as_accuracy_or_ranking_evidence": True,
                    "exact_native_pre_solver_target_claimed": False,
                    "constructor_scope": (
                        "formula reconstruction of AMASS actor-shape/height/scale "
                        "policy on a frozen static semantic set; no native runtime "
                        "constructor, optimizer, or AMASS motion"
                    ),
                    "interpretation_scope": (
                        "AMASS shape-policy difference only; exact LAFAN runtime "
                        "targets are evidenced separately by schema-2 native capture"
                    ),
                    **target_geometry,
                }
            )
    for row in rows:
        method = str(row["method"])
        profile = str(row["profile"])
        baseline = contact_by_method_profile[(method, "zero")]
        current = contact_by_method_profile[(method, profile)]
        row["contact_label_change_rate_vs_zero"] = float(np.mean(current != baseline))
        zero_hash = next(
            item["target_sha256"]
            for item in rows
            if item["method"] == method and item["profile"] == "zero"
        )
        row["target_identical_to_zero_profile"] = row["target_sha256"] == zero_hash
    frame = pd.DataFrame(rows).sort_values(["method", "beta0"])
    frame.to_csv(
        root / "metrics" / "actor_shape_policy_formula_probe.csv", index=False
    )
    return frame


def _collect_scale_outputs(root: Path) -> list[tuple[str, str, str, Path]]:
    sequence = load_yaml(root / "manifests" / "pilot_sequence.yaml")
    base = root / "runs" / sequence["sequence_id"] / "scale-sensitivity"
    rows = []
    accepted_policies = set(
        _policy_specs_from_protocol(
            load_evaluator_protocol(root / "manifests/evaluator.yaml")
        )
    )
    for path in sorted((base / "controlled-dense").glob("*/canonical_g1.npz")):
        if path.parent.name in accepted_policies:
            rows.append(("controlled_policy", "controlled_dense", path.parent.name, path))
    for path in sorted((base / "native-response").glob("*/*/canonical_g1.npz")):
        rows.append(
            (
                "native_response_fixed_canonical_contact",
                path.parents[1].name,
                path.parent.name,
                path,
            )
        )
    # Holosoma is contact-aware and the frozen pure-scale arm deliberately
    # uses canonical labels.  Preserve the already-run official recomputation
    # arm as a separately named robustness analysis; never mix the two graphs.
    for path in sorted(
        (base / "native-response" / "omniretarget").glob(
            "*/canonical_g1.native-contact-v3.npz"
        )
    ):
        rows.append(
            (
                "native_response_native_recomputed_contact",
                "omniretarget",
                path.parent.name,
                path,
            )
        )
    return rows


def build_scale_run_provenance(
    repo_root: str | Path,
    collected: list[tuple[str, str, str, Path]] | None = None,
) -> pd.DataFrame:
    """Validate every scale artifact against source/config/policy provenance."""

    root = Path(repo_root).resolve()
    sequence = load_yaml(root / "manifests/pilot_sequence.yaml")
    human = CanonicalHuman.load(root / sequence["canonical_path"])
    evaluator = load_evaluator_protocol(root / "manifests/evaluator.yaml")
    scale_protocol_hash = sha256_file(root / "configs/scale_policy_sensitivity.yaml")
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for role, method, variant, path in collected or _collect_scale_outputs(root):
        motion = CanonicalG1.load(path)
        motion.validate(source_frame_count=len(human.timestamps))
        metadata = motion.metadata
        if role == "controlled_policy":
            expected_config = sha256_file(root / "configs/controlled_mink.yaml")
            config_match = metadata.get("config_sha256") == expected_config
            upstream_match = True
            method_match = metadata.get("policy_id") == variant
            robot_hash = metadata.get("canonical_robot_xml_sha256")
            robot_match = robot_hash == evaluator["robot_xml_sha256"]
            role_match = metadata.get("experiment_role") == "controlled_policy_ablation"
            scale_protocol_match = (
                metadata.get("scale_protocol_sha256") == scale_protocol_hash
                and metadata.get("common_scale_definition")
                == evaluator["scale"]["definition"]
                and metadata.get("head_to_toe_role")
                == "diagnostic_only_not_common_policy"
                and metadata.get("canonical_joint_order_sha256")
                == evaluator["robot_joint_order_sha256"]
            )
            expected_root = expected_local = 1.0
        else:
            expected = _native_expected_provenance(root, method)
            expected_config = expected["config_sha256"]
            config_match = metadata.get("config_sha256") == expected_config
            upstream_match = metadata.get("upstream_commit") == expected["upstream_commit"]
            method_match = metadata.get("method") == expected["method"]
            robot_field = str(expected["solver_robot_asset_field"])
            robot_hash = metadata.get(robot_field)
            robot_match = robot_hash == expected["solver_robot_asset_sha256"]
            if method == "omniretarget":
                robot_match = bool(
                    robot_match
                    and metadata.get("solver_robot_urdf_sha256")
                    == expected["solver_robot_urdf_sha256"]
                    and metadata.get(
                        str(expected["canonical_evaluator_asset_field"])
                    )
                    == expected["canonical_evaluator_asset_sha256"]
                    and metadata.get("solver_and_evaluator_assets_identical")
                    is False
                )
            required_role = (
                "native_response_native_recomputed_contact"
                if role == "native_response_native_recomputed_contact"
                else "native_response_fixed_canonical_contact"
            )
            role_match = metadata.get("experiment_role") == required_role
            scale_protocol_match = metadata.get("scale_protocol_sha256") == scale_protocol_hash
            from .scale_worker import scale_execution_contract

            execution_contract_match = (
                metadata.get("scale_execution_contract")
                == scale_execution_contract(root, method)
                and isinstance(metadata.get("scale_runtime_environment"), dict)
                and metadata["scale_runtime_environment"].get(
                    "environment_name"
                )
                == scale_execution_contract(root, method)["environment_name"]
            )
            expected_root, expected_local = WITHIN_METHOD_VARIANTS[variant]
        exact_frames = bool(
            len(motion.qpos) == len(human.timestamps)
            and np.array_equal(motion.source_frame_idx, np.arange(len(human.timestamps)))
            and np.asarray(motion.valid, dtype=bool).all()
        )
        source_match = metadata.get("canonical_source_sha256") == human.source_sha256
        variant_match = bool(
            metadata.get("scale_variant") == variant
            and metadata.get("root_scale_multiplier") == expected_root
            and metadata.get("local_scale_multiplier") == expected_local
        )
        contact_match = True
        if method == "omniretarget":
            expected_fixed = role != "native_response_native_recomputed_contact"
            contact_match = bool(metadata.get("fixed_contact_labels") is expected_fixed)
        accepted = all(
            (
                config_match,
                upstream_match,
                method_match,
                robot_match,
                role_match,
                scale_protocol_match,
                execution_contract_match
                if role != "controlled_policy"
                else True,
                exact_frames,
                source_match,
                variant_match,
                contact_match,
            )
        )
        if not accepted:
            failures.append(f"{role}/{method}/{variant}")
        rows.append(
            {
                "experiment_role": role,
                "method": method,
                "variant": variant,
                "output_path": str(path.relative_to(root)),
                "output_sha256": sha256_file(path),
                "frames": len(motion.qpos),
                "source_sha256": human.source_sha256,
                "source_match": source_match,
                "config_sha256": expected_config,
                "config_match": config_match,
                "upstream_match": upstream_match,
                "method_match": method_match,
                "robot_asset_sha256": robot_hash,
                "robot_match": robot_match,
                "role_match": role_match,
                "scale_protocol_sha256": scale_protocol_hash,
                "scale_protocol_match": scale_protocol_match,
                "execution_contract_match": (
                    execution_contract_match
                    if role != "controlled_policy"
                    else True
                ),
                "variant_match": variant_match,
                "contact_policy_match": contact_match,
                "exact_full_frames": exact_frames,
                "accepted": accepted,
            }
        )
    frame = pd.DataFrame(rows).sort_values(["experiment_role", "method", "variant"])
    frame.to_csv(root / "manifests/scale_run_provenance.csv", index=False)
    if failures:
        raise ValueError(f"Scale provenance failed closed for: {', '.join(failures)}")
    return frame


def _reach_envelope_m(
    robot: CanonicalRobotModel,
    labels: list[str],
    *,
    samples: int = 512,
) -> dict[str, float]:
    """Return a deterministic sampled upper reach envelope per semantic point."""

    rng = np.random.default_rng(20260722)
    maxima = {label: 0.0 for label in labels if label != "root"}
    for sample in range(samples):
        qpos = robot.model.qpos0.copy()
        if sample:
            for joint_id in range(1, robot.model.njnt):
                if not robot.model.jnt_limited[joint_id]:
                    continue
                address = robot.model.jnt_qposadr[joint_id]
                lower, upper = robot.model.jnt_range[joint_id]
                qpos[address] = rng.uniform(lower, upper)
        points = robot.semantic_positions(qpos)
        root = points["root"]
        for label in maxima:
            maxima[label] = max(maxima[label], float(np.linalg.norm(points[label] - root)))
    return maxima


def _standardized_target_diagnostics(
    root: Path,
    human: CanonicalHuman,
    robot: CanonicalRobotModel,
    motion: CanonicalG1,
    policy: dict[str, Any],
) -> dict[str, float]:
    """Score one policy using a common Dense semantic target diagnostic.

    This is exact for the controlled Dense arm.  For native public pipelines it
    is deliberately named a standardized semantic diagnostic rather than the
    native solver objective (Holosoma is Laplacian and v3 mixes global/local
    residuals).  Root is compared as frame-zero displacement; all other points
    are compared root-relative, so native world anchors do not masquerade as
    reach error.
    """

    labels, target = _target_array(root, human, policy)
    frame_indices = np.asarray(motion.source_frame_idx, dtype=np.int64)
    target = target[frame_indices]
    robot_frames = [robot.semantic_positions(qpos) for qpos in motion.qpos]
    robot_target = np.stack(
        [[frame[label] for label in labels] for frame in robot_frames], axis=0
    )
    root_index = labels.index("root")
    target_root = target[:, root_index]
    robot_root = robot_target[:, root_index]
    target_local = target - target_root[:, None, :]
    robot_local = robot_target - robot_root[:, None, :]
    residual = np.linalg.norm(robot_local - target_local, axis=2)
    residual[:, root_index] = np.linalg.norm(
        (robot_root - robot_root[0]) - (target_root - target_root[0]), axis=1
    )
    envelope = _reach_envelope_m(robot, labels)
    exceedance = np.zeros((len(target), len(labels)), dtype=bool)
    for index, label in enumerate(labels):
        if label == "root":
            continue
        target_radius = np.linalg.norm(target_local[:, index], axis=1)
        exceedance[:, index] = target_radius > envelope[label] + 0.02
    return {
        "standardized_semantic_target_residual_mean_m": float(residual.mean()),
        "standardized_semantic_target_residual_p95_m": float(np.percentile(residual, 95)),
        "sampled_reach_envelope_exceedance_frame_rate": float(
            np.any(exceedance, axis=1).mean()
        ),
        "reach_envelope_samples": int(512),
        "reach_envelope_margin_m": 0.02,
    }


def summarize_scale_sensitivity(repo_root: str | Path = ".") -> pd.DataFrame:
    root = Path(repo_root).resolve()
    sequence = load_yaml(root / "manifests" / "pilot_sequence.yaml")
    human = CanonicalHuman.load(root / sequence["canonical_path"])
    robot = CanonicalRobotModel(default_robot_scene(root))
    protocol_path = root / "manifests" / "evaluator.yaml"
    protocol = load_evaluator_protocol(protocol_path)
    protocol["manifest_sha256"] = sha256_file(protocol_path)
    rows = []
    native_motion: dict[tuple[str, str], CanonicalG1] = {}
    collected = _collect_scale_outputs(root)
    build_scale_run_provenance(root, collected)
    for role, method, variant, path in collected:
        motion = CanonicalG1.load(path)
        if role.startswith("native_response_") and variant == "native":
            native_motion[(role, method)] = motion
    for role, method, variant, path in collected:
        motion = CanonicalG1.load(path)
        table, summary = evaluate_motion(human, motion, robot, protocol)
        label = f"{role}__{method}__{variant}".replace("/", "-")
        save_evaluation(table, summary, root / "metrics" / "scale_sensitivity", label)
        qpos_rms = float("nan")
        fk_rms = float("nan")
        if role.startswith("native_response_") and (role, method) in native_motion:
            reference = native_motion[(role, method)]
            count = min(len(reference.qpos), len(motion.qpos))
            difference = motion.qpos[:count] - reference.qpos[:count]
            difference[:, 7:] = np.arctan2(
                np.sin(difference[:, 7:]), np.cos(difference[:, 7:])
            )
            qpos_rms = float(np.sqrt(np.mean(difference**2)))
            current_fk = np.stack(
                [np.concatenate(list(robot.semantic_positions(q).values())) for q in motion.qpos[:count]]
            )
            native_fk = np.stack(
                [np.concatenate(list(robot.semantic_positions(q).values())) for q in reference.qpos[:count]]
            )
            fk_rms = float(np.sqrt(np.mean((current_fk - native_fk) ** 2)))
        if role == "controlled_policy":
            policy = _policy_specs_from_protocol(protocol)[variant]
        else:
            policy_id = NATIVE_POLICY[method]
            base_policy = _policy_specs_from_protocol(protocol)[policy_id]
            root_multiplier, local_multiplier = WITHIN_METHOD_VARIANTS[variant]
            policy = _scale_axes(base_policy, root_multiplier, local_multiplier)
        target_diagnostics = _standardized_target_diagnostics(
            root, human, robot, motion, policy
        )
        duration_s = len(motion.qpos) / float(human.fps)
        end_to_end_s = float(
            motion.metadata.get(
                "single_measurement_wall_time_s",
                motion.metadata.get(
                    "steady_end_to_end_total_s",
                    np.sum(motion.per_frame_solve_time_s),
                ),
            )
        )
        native_core_s = float(
            motion.metadata.get(
                "native_core_total_s", np.sum(motion.per_frame_solve_time_s)
            )
        )
        rows.append(
            {
                "experiment_role": role,
                "method": method,
                "variant": variant,
                "output_path": str(path.relative_to(root)),
                "output_sha256": sha256_file(path),
                "qpos_rms_from_native": qpos_rms,
                "fk_rms_from_native_m": fk_rms,
                "scale_end_to_end_single_wall_s": end_to_end_s,
                "scale_end_to_end_single_rtf": end_to_end_s / duration_s,
                "scale_native_core_total_s": native_core_s,
                "scale_native_core_rtf": native_core_s / duration_s,
                "timing_evidence_grade": "single_scale-response_run",
                **target_diagnostics,
                **summary,
            }
        )
    frame = pd.DataFrame(rows).sort_values(["experiment_role", "method", "variant"])
    frame.to_csv(root / "metrics" / "scale_policy_sensitivity_summary.csv", index=False)
    frame.to_parquet(root / "metrics" / "scale_policy_sensitivity_summary.parquet", index=False)
    _build_sensitivity_and_rank_tables(root, frame)
    return frame


def _build_sensitivity_and_rank_tables(root: Path, frame: pd.DataFrame) -> None:
    response = frame[frame.experiment_role.str.startswith("native_response_")]
    slopes = derive_registered_scale_slopes(response)
    slopes.to_csv(root / "metrics" / "scale_sensitivity_slopes.csv", index=False)

    formal_role = "native_response_fixed_canonical_contact"
    formal_methods = tuple(
        sorted(
            response.loc[
                response.experiment_role.astype(str).eq(formal_role), "method"
            ].astype(str).unique()
        )
    )
    ranks = derive_registered_rank_stability(
        response,
        methods=formal_methods,
        experiment_role=formal_role,
    )
    ranks.to_csv(root / "metrics" / "method_rank_stability.csv", index=False)


def run_scale_stage(
    repo_root: str | Path = ".", methods: list[str] | None = None
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    targets = build_pre_solver_target_contract(root)
    controlled = run_controlled_policy_transplants(root)
    build_contact_flip_diagnostics(root)
    build_actor_shape_probe(root)
    completed = {}
    for method in methods or ["gmr", "omniretarget", "protomotions_v2_3", "protomotions_v3"]:
        try:
            completed[method] = [str(path) for path in run_native_response(method, root)]
        except (ImportError, FileNotFoundError, RuntimeError, AttributeError) as error:
            completed[method] = {"status": "failed", "error": str(error)}
    if "omniretarget" in (methods or ["gmr", "omniretarget", "protomotions_v2_3", "protomotions_v3"]):
        completed["omniretarget_native_contact_robustness"] = [
            str(path) for path in run_holosoma_native_contact_robustness(root)
        ]
    summary = summarize_scale_sensitivity(root)
    result = {
        "pre_solver_contract_rows": len(targets),
        "controlled_outputs": [str(path) for path in controlled],
        "native_response": completed,
        "summary_rows": len(summary),
    }
    atomic_write_json(root / "manifests" / "scale_sensitivity_run.json", result)
    return result
