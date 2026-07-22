from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from retargeting_comparison.scale_sensitivity import (
    WITHIN_METHOD_VARIANTS,
    _scale_axes,
    policy_specs,
    transform_human,
)
from retargeting_comparison.schemas import CanonicalHuman
from retargeting_comparison.schemas import CanonicalG1
from retargeting_comparison.io_utils import sha256_file
from retargeting_comparison.scale_worker import (
    native_expected_provenance,
    native_output_is_current,
    scale_execution_contract,
)
from retargeting_comparison.scale_metric_registry import (
    SCALE_RANK_METRICS,
    SCALE_RESPONSE_METRIC_NAMES,
    derive_registered_rank_stability,
    derive_registered_scale_slopes,
)


def _human() -> CanonicalHuman:
    positions = np.asarray(
        [
            [[1.0, 2.0, 3.0], [2.0, 2.0, 3.0]],
            [[3.0, 4.0, 5.0], [5.0, 4.0, 5.0]],
        ]
    )
    rotations = np.zeros((2, 2, 4), dtype=float)
    rotations[..., 0] = 1.0
    return CanonicalHuman(
        joint_names=np.asarray(["Hips", "LeftFoot"]),
        parent_indices=np.asarray([-1, 0]),
        local_rotations=rotations,
        world_rotations=rotations,
        world_positions=positions,
        root_translation=positions[:, 0],
        fps=2.0,
        timestamps=np.asarray([0.0, 0.5]),
        foot_contact_labels=np.zeros((2, 2), dtype=bool),
        source_sha256="0" * 64,
    )


def test_registered_policy_set_and_native_perturbations_are_frozen():
    policies = policy_specs(0.742)
    assert set(policies) == {
        "common_shared_semantic_landmark_ls",
        "holosoma_lafan_uniform",
        "gmr_region_wise",
        "protomotions_v2_3_world_axis",
        "protomotions_v3_lower_upper_axis",
    }
    assert set(WITHIN_METHOD_VARIANTS) == {
        "native",
        "root_minus_5",
        "root_plus_5",
        "local_minus_5",
        "local_plus_5",
    }
    gmr = policies["gmr_region_wise"]
    assert gmr["root_axis"] == [0.875, 0.875, 0.875]
    assert np.allclose(gmr["local_axes"]["arms"], [0.7291666666666666] * 3)


def test_common_policy_keeps_registered_root_and_local_gains_separate():
    common = policy_specs(0.8, 0.9)["common_shared_semantic_landmark_ls"]
    assert common["root_axis"] == [0.9, 0.9, 0.9]
    assert common["local_axes"]["arms"] == [0.8, 0.8, 0.8]
    assert common["root_local_parameters_separate"] is True


def test_scale_axes_preserve_native_region_ratios():
    native = policy_specs(0.742)["gmr_region_wise"]
    perturbed = _scale_axes(native, 1.05, 0.95)
    assert np.allclose(np.asarray(perturbed["root_axis"]), np.asarray(native["root_axis"]) * 1.05)
    assert np.allclose(
        np.asarray(perturbed["local_axes"]["arms"]),
        np.asarray(native["local_axes"]["arms"]) * 0.95,
    )


def test_transform_human_uses_frame_zero_root_anchor_and_fixed_contacts():
    human = _human()
    transformed = transform_human(human, root_multiplier=0.5, local_multiplier=2.0)
    assert np.allclose(transformed.world_positions[0, 0], human.world_positions[0, 0])
    assert np.allclose(
        transformed.world_positions[1, 0],
        human.world_positions[0, 0]
        + 0.5 * (human.world_positions[1, 0] - human.world_positions[0, 0]),
    )
    expected_local = 2.0 * (
        human.world_positions[1, 1] - human.world_positions[1, 0]
    )
    assert np.allclose(
        transformed.world_positions[1, 1] - transformed.world_positions[1, 0],
        expected_local,
    )
    assert np.array_equal(transformed.foot_contact_labels, human.foot_contact_labels)


def test_holosoma_reuse_separates_solver_assets_from_canonical_evaluator() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "source/canonical_human/dance1_subject1_f000000_000600.npz"
    solver = (
        root
        / "external/holosoma/src/holosoma_retargeting/holosoma_retargeting"
        / "models/g1/g1_29dof.xml"
    )
    evaluator = (
        root
        / "external/holosoma/src/holosoma/holosoma/data/robots/g1/scenes"
        / "scene_g1_29dof_wbt_plane.xml"
    )
    if not source.is_file() or not solver.is_file() or not evaluator.is_file():
        pytest.skip("Generated Pilot source and pinned assets are external")
    human = CanonicalHuman.load(source)
    expected = native_expected_provenance(root, "omniretarget")
    assert expected["solver_robot_asset_sha256"] == sha256_file(solver)
    assert expected["canonical_evaluator_asset_sha256"] == sha256_file(evaluator)
    assert expected["solver_robot_asset_sha256"] != expected[
        "canonical_evaluator_asset_sha256"
    ]
    qpos = np.zeros((len(human.timestamps), 36), dtype=np.float64)
    qpos[:, 3] = 1.0
    metadata = {
        "method": expected["method"],
        "upstream_commit": expected["upstream_commit"],
        "config_sha256": expected["config_sha256"],
        "solver_robot_urdf_sha256": expected["solver_robot_urdf_sha256"],
        "solver_robot_xml_sha256": expected["solver_robot_asset_sha256"],
        "canonical_evaluator_robot_scene_sha256": expected[
            "canonical_evaluator_asset_sha256"
        ],
        "solver_and_evaluator_assets_identical": False,
        "canonical_source_sha256": human.source_sha256,
        "canonical_source_file_sha256": sha256_file(source),
        "scale_variant": "native",
        "root_scale_multiplier": 1.0,
        "local_scale_multiplier": 1.0,
        "scale_protocol_sha256": sha256_file(
            root / "configs/scale_policy_sensitivity.yaml"
        ),
        "experiment_role": "native_response_fixed_canonical_contact",
        "completion_status": "succeeded",
        "scale_execution_contract": scale_execution_contract(
            root, "omniretarget"
        ),
        "scale_runtime_environment": {"environment_name": "hsretargeting"},
    }
    motion = CanonicalG1(
        qpos=qpos,
        fps=human.fps,
        source_frame_idx=np.arange(len(qpos)),
        valid=np.ones(len(qpos), dtype=bool),
        per_frame_solve_time_s=np.zeros(len(qpos)),
        metadata=metadata,
    )
    assert native_output_is_current(
        root, motion, human, "omniretarget", "native"
    )
    motion.metadata["solver_robot_xml_sha256"] = expected[
        "canonical_evaluator_asset_sha256"
    ]
    assert not native_output_is_current(
        root, motion, human, "omniretarget", "native"
    )


def test_registered_scale_response_covers_every_metric_without_composite_score():
    methods = ("a", "b")
    rows = []
    for method_index, method in enumerate(methods, start=1):
        for variant_index, variant in enumerate(WITHIN_METHOD_VARIANTS):
            row = {
                "experiment_role": "formal",
                "method": method,
                "variant": variant,
            }
            for metric_index, metric in enumerate(SCALE_RESPONSE_METRIC_NAMES, start=1):
                row[metric] = method_index + metric_index / 100 + variant_index / 1000
            rows.append(row)
    response = pd.DataFrame(rows)
    slopes = derive_registered_scale_slopes(response)
    ranks = derive_registered_rank_stability(
        response,
        methods=methods,
        experiment_role="formal",
    )
    assert set(slopes.metric) == set(SCALE_RESPONSE_METRIC_NAMES)
    assert len(slopes) == len(methods) * len(SCALE_RESPONSE_METRIC_NAMES) * 2
    assert not slopes.composite_score_used.any()
    assert set(ranks.metric) == set(SCALE_RANK_METRICS)
    assert len(ranks) == len(methods) * len(SCALE_RANK_METRICS) * 5
    assert not ranks.composite_score_used.any()
