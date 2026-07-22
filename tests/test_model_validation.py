from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODEL_ROOT = Path("../body_models")


@pytest.mark.skipif(
    not MODEL_ROOT.is_dir()
    or importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("smplx") is None,
    reason="licensed body models or their isolated dependencies are not present",
)
def test_smpl_and_smplx_zero_pose_forward_is_finite(tmp_path: Path) -> None:
    from retargeting_comparison.model_validation import validate_body_models

    result = validate_body_models(MODEL_ROOT, tmp_path / "body_models.yaml")
    assert result["smpl"]["finite_forward"] is True
    assert result["smplx"]["finite_forward"] is True
    assert result["smpl"]["format"] == "chumpy-free pkl"
    assert result["smplx"]["format"] == "npz"
    assert result["original_smpl_pickle_used"] is False
