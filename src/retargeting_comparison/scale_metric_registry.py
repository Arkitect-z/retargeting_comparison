"""Pre-registered metric registry for Stage 1 scale-response evidence.

The registry deliberately separates response measurement from ranking.  Every
listed metric receives a finite-difference response table, but only metrics
with ``rank_eligible=True`` participate in the descriptive rank-stability
analysis.  No composite score is defined here or elsewhere in Stage 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScaleMetricSpec:
    name: str
    family: str
    preferred_direction: str
    rank_eligible: bool = False


SCALE_RESPONSE_METRICS = (
    # Root-frame pose fidelity.
    ScaleMetricSpec("rf_kpe_all_mean_m", "rf_kpe", "minimize", True),
    ScaleMetricSpec("rf_kpe_all_p95_m", "rf_kpe", "minimize", True),
    ScaleMetricSpec("rf_kpe_targeted_mean_m", "rf_kpe", "minimize", True),
    ScaleMetricSpec("rf_kpe_targeted_p95_m", "rf_kpe", "minimize", True),
    ScaleMetricSpec("rf_kpe_untracked_mean_m", "rf_kpe", "minimize", True),
    ScaleMetricSpec("rf_kpe_untracked_p95_m", "rf_kpe", "minimize", True),
    ScaleMetricSpec("bone_direction_mean_rad", "pose_structure", "minimize", True),
    ScaleMetricSpec("bone_direction_p95_rad", "pose_structure", "minimize", True),
    ScaleMetricSpec("bend_plane_mean_rad", "pose_structure", "minimize", True),
    ScaleMetricSpec("bend_plane_p95_rad", "pose_structure", "minimize", True),
    # Root-path fidelity under common, native, and fitted-scale views.
    ScaleMetricSpec(
        "root_translation_common_scale_mean_m", "root", "minimize", True
    ),
    ScaleMetricSpec(
        "root_translation_common_scale_p95_m", "root", "minimize", True
    ),
    ScaleMetricSpec(
        "root_translation_native_scale_mean_m", "root", "minimize", False
    ),
    ScaleMetricSpec(
        "root_translation_native_scale_p95_m", "root", "minimize", False
    ),
    ScaleMetricSpec(
        "root_translation_scale_invariant_mean_m", "root", "minimize", True
    ),
    ScaleMetricSpec(
        "root_translation_scale_invariant_p95_m", "root", "minimize", True
    ),
    ScaleMetricSpec("root_yaw_mean_rad", "root", "minimize", True),
    ScaleMetricSpec("root_yaw_p95_rad", "root", "minimize", True),
    # Temporal response.
    ScaleMetricSpec(
        "joint_velocity_rms_mean_rad_s", "temporal", "minimize", True
    ),
    ScaleMetricSpec(
        "joint_velocity_rms_p95_rad_s", "temporal", "minimize", True
    ),
    ScaleMetricSpec(
        "joint_acceleration_rms_mean_rad_s2", "temporal", "minimize", True
    ),
    ScaleMetricSpec(
        "joint_acceleration_rms_p95_rad_s2", "temporal", "minimize", True
    ),
    ScaleMetricSpec("joint_jerk_rms_mean_rad_s3", "temporal", "minimize", True),
    ScaleMetricSpec("joint_jerk_rms_p95_rad_s3", "temporal", "minimize", True),
    ScaleMetricSpec("pose_jump_mean_m", "temporal", "minimize", True),
    ScaleMetricSpec("pose_jump_p95_m", "temporal", "minimize", True),
    # Artifact components stay separate; ``artifact_rate`` is only their union.
    ScaleMetricSpec(
        "foot_skating_frame_rate", "artifact_component", "minimize", True
    ),
    ScaleMetricSpec(
        "ground_penetration_frame_rate", "artifact_component", "minimize", True
    ),
    ScaleMetricSpec(
        "ground_penetration_p95_m", "artifact_component", "minimize", True
    ),
    ScaleMetricSpec(
        "joint_limit_violation_frame_rate",
        "artifact_component",
        "minimize",
        True,
    ),
    ScaleMetricSpec(
        "joint_limit_violation_p95_rad",
        "artifact_component",
        "minimize",
        True,
    ),
    ScaleMetricSpec("invalid_frame_rate", "artifact_component", "minimize", True),
    ScaleMetricSpec("artifact_rate", "artifact_union", "minimize", True),
    ScaleMetricSpec("completion_ratio", "completion", "maximize", True),
    # Common diagnostic target and reach-envelope response.
    ScaleMetricSpec(
        "standardized_semantic_target_residual_mean_m",
        "declared_task_residual",
        "minimize",
        True,
    ),
    ScaleMetricSpec(
        "standardized_semantic_target_residual_p95_m",
        "declared_task_residual",
        "minimize",
        True,
    ),
    ScaleMetricSpec(
        "sampled_reach_envelope_exceedance_frame_rate",
        "reach_feasibility",
        "minimize",
        True,
    ),
    # Response distance from the method's native trajectory is not quality.
    ScaleMetricSpec(
        "qpos_rms_from_native",
        "native_trajectory_divergence",
        "minimize_for_robustness_not_quality",
        False,
    ),
    ScaleMetricSpec(
        "fk_rms_from_native_m",
        "native_trajectory_divergence",
        "minimize_for_robustness_not_quality",
        False,
    ),
    # Single scale-response runs describe cost sensitivity, not benchmark timing rank.
    ScaleMetricSpec(
        "scale_end_to_end_single_wall_s", "single_run_timing", "minimize", False
    ),
    ScaleMetricSpec(
        "scale_end_to_end_single_rtf", "single_run_timing", "minimize", False
    ),
    ScaleMetricSpec(
        "scale_native_core_total_s", "single_run_timing", "minimize", False
    ),
    ScaleMetricSpec(
        "scale_native_core_rtf", "single_run_timing", "minimize", False
    ),
)

SCALE_RESPONSE_METRIC_BY_NAME = {
    metric.name: metric for metric in SCALE_RESPONSE_METRICS
}
SCALE_RESPONSE_METRIC_NAMES = tuple(metric.name for metric in SCALE_RESPONSE_METRICS)
SCALE_RANK_METRICS = tuple(
    metric.name for metric in SCALE_RESPONSE_METRICS if metric.rank_eligible
)

SCALE_RESPONSE_VARIANTS = (
    "native",
    "root_minus_5",
    "root_plus_5",
    "local_minus_5",
    "local_plus_5",
)
SCALE_RESPONSE_FACTORS = (
    ("root", "root_minus_5", "root_plus_5"),
    ("local", "local_minus_5", "local_plus_5"),
)


def derive_registered_scale_slopes(
    response: pd.DataFrame,
    *,
    group_columns: tuple[str, ...] = ("experiment_role", "method"),
) -> pd.DataFrame:
    """Return complete, finite response rows for every registered metric."""

    missing_columns = (
        set(group_columns)
        | {"variant"}
        | set(SCALE_RESPONSE_METRIC_NAMES)
    ) - set(response)
    if missing_columns:
        raise ValueError(
            "Scale-response table lacks registered columns: "
            + str(sorted(missing_columns))
        )
    rows: list[dict[str, object]] = []
    group_key: str | list[str]
    group_key = group_columns[0] if len(group_columns) == 1 else list(group_columns)
    for group_values, group in response.groupby(group_key, sort=True):
        values = (group_values,) if len(group_columns) == 1 else tuple(group_values)
        identity = dict(zip(group_columns, values))
        lookup = group.set_index("variant")
        if lookup.index.duplicated().any():
            raise ValueError(f"Duplicate scale variants for {identity}")
        missing_variants = set(SCALE_RESPONSE_VARIANTS) - set(lookup.index.astype(str))
        if missing_variants:
            raise ValueError(
                f"Scale response for {identity} lacks {sorted(missing_variants)}"
            )
        for metric in SCALE_RESPONSE_METRICS:
            metric_values = lookup.loc[list(SCALE_RESPONSE_VARIANTS), metric.name].to_numpy(
                dtype=float
            )
            if not np.isfinite(metric_values).all():
                raise ValueError(
                    f"Scale response for {identity}/{metric.name} contains NaN/Inf"
                )
            native = float(lookup.loc["native", metric.name])
            for factor, minus_name, plus_name in SCALE_RESPONSE_FACTORS:
                minus = float(lookup.loc[minus_name, metric.name])
                plus = float(lookup.loc[plus_name, metric.name])
                slope = (plus - minus) / 0.10
                elasticity_defined = abs(native) > 1e-12
                rows.append(
                    {
                        **identity,
                        "factor": factor,
                        "minus_variant": minus_name,
                        "plus_variant": plus_name,
                        "metric": metric.name,
                        "metric_family": metric.family,
                        "preferred_direction": metric.preferred_direction,
                        "rank_eligible": metric.rank_eligible,
                        "native_value": native,
                        "minus_value": minus,
                        "plus_value": plus,
                        "minus_delta_from_native": minus - native,
                        "plus_delta_from_native": plus - native,
                        "response_range": max(native, minus, plus)
                        - min(native, minus, plus),
                        "central_difference_per_unit_multiplier": slope,
                        "elasticity_at_native": slope / native
                        if elasticity_defined
                        else 0.0,
                        "elasticity_defined": elasticity_defined,
                        "composite_score_used": False,
                    }
                )
    columns = [
        *group_columns,
        "factor",
        "minus_variant",
        "plus_variant",
        "metric",
        "metric_family",
        "preferred_direction",
        "rank_eligible",
        "native_value",
        "minus_value",
        "plus_value",
        "minus_delta_from_native",
        "plus_delta_from_native",
        "response_range",
        "central_difference_per_unit_multiplier",
        "elasticity_at_native",
        "elasticity_defined",
        "composite_score_used",
    ]
    return pd.DataFrame(rows, columns=columns)


def derive_registered_rank_stability(
    response: pd.DataFrame,
    *,
    methods: tuple[str, ...],
    experiment_role: str,
) -> pd.DataFrame:
    """Rank only the pre-registered quality subset, with explicit direction."""

    role_frame = response.loc[
        response.experiment_role.astype(str).eq(experiment_role)
    ].copy()
    if set(role_frame.method.astype(str)) != set(methods):
        raise ValueError("Formal rank table does not contain the exact method set")
    rows: list[dict[str, object]] = []
    for metric_name in SCALE_RANK_METRICS:
        spec = SCALE_RESPONSE_METRIC_BY_NAME[metric_name]
        pivot = role_frame.pivot(index="method", columns="variant", values=metric_name)
        pivot = pivot.reindex(index=methods, columns=SCALE_RESPONSE_VARIANTS)
        if pivot.shape != (len(methods), len(SCALE_RESPONSE_VARIANTS)):
            raise ValueError(f"Incomplete rank pivot for {metric_name}")
        values = pivot.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Rank metric {metric_name} contains NaN/Inf")
        ascending = spec.preferred_direction != "maximize"
        native_rank = pivot["native"].rank(method="average", ascending=ascending)
        for variant in SCALE_RESPONSE_VARIANTS:
            rank = pivot[variant].rank(method="average", ascending=ascending)
            raw_correlation = (
                native_rank.corr(rank, method="spearman")
                if native_rank.nunique() > 1 and rank.nunique() > 1
                else float("nan")
            )
            correlation_defined = bool(np.isfinite(raw_correlation))
            correlation = float(raw_correlation) if correlation_defined else 0.0
            for method in methods:
                rows.append(
                    {
                        "experiment_role": experiment_role,
                        "metric": metric_name,
                        "metric_family": spec.family,
                        "preferred_direction": spec.preferred_direction,
                        "higher_is_better": not ascending,
                        "rank_eligible": True,
                        "variant": variant,
                        "method": method,
                        "rank": float(rank[method]),
                        "native_rank": float(native_rank[method]),
                        "rank_changed": not np.isclose(
                            float(rank[method]), float(native_rank[method])
                        ),
                        "spearman_vs_native": correlation,
                        "spearman_defined": correlation_defined,
                        "composite_score_used": False,
                    }
                )
    return pd.DataFrame(rows)

if len(SCALE_RESPONSE_METRIC_BY_NAME) != len(SCALE_RESPONSE_METRICS):  # pragma: no cover
    raise RuntimeError("Scale-response registry contains duplicate metric names")
