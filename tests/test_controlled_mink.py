from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from retargeting_comparison.controlled_mink import (
    ControlledMinkRetargeter,
    common_baseline_config,
    seed_qpos,
)
from retargeting_comparison.io_utils import load_yaml


pytestmark = pytest.mark.skipif(
    not Path("external/GMR/assets/unitree_g1/g1_mocap_29dof.xml").is_file(),
    reason="frozen GMR checkout is not present",
)


def test_sparse_dense_only_vary_declared_target_set() -> None:
    config = load_yaml("configs/controlled_mink.yaml")
    assert common_baseline_config(config) == common_baseline_config(config)
    sparse = config["target_sets"]["sparse"]
    dense = config["target_sets"]["dense"]
    assert sparse == [target for target in dense if target["semantic"] in {
        "root", "left_wrist", "right_wrist", "left_ankle", "right_ankle"
    }]


def test_sparse_seeds_are_deterministic_and_distinct() -> None:
    retargeter = ControlledMinkRetargeter(".", "sparse")
    neutral = seed_qpos(retargeter.model, retargeter.config, "neutral")
    a_first = seed_qpos(retargeter.model, retargeter.config, "A")
    a_second = seed_qpos(retargeter.model, retargeter.config, "A")
    b = seed_qpos(retargeter.model, retargeter.config, "B")
    assert np.array_equal(a_first, a_second)
    assert not np.array_equal(neutral, a_first)
    assert not np.array_equal(a_first, b)
    assert np.allclose(a_first[:7], neutral[:7])
