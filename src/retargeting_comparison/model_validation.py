"""Read-only validation and hashing of external licensed body models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import smplx

from .io_utils import atomic_write_yaml, sha256_file


def validate_body_models(body_models_root: str | Path, output: str | Path) -> dict[str, Any]:
    configured_root = Path(body_models_root)
    root = configured_root.resolve()
    smpl_file = root / "smpl_chumpyfree" / "smpl" / "SMPL_NEUTRAL.pkl"
    smplx_file = root / "smplx" / "SMPLX_NEUTRAL.npz"
    for path in (smpl_file, smplx_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    smpl = smplx.create(
        str(root / "smpl_chumpyfree"), model_type="smpl", gender="neutral", ext="pkl"
    )
    smpl_result = smpl(
        global_orient=torch.zeros(1, 3),
        body_pose=torch.zeros(1, 69),
        betas=torch.zeros(1, int(smpl.num_betas)),
        transl=torch.zeros(1, 3),
    )
    if not torch.isfinite(smpl_result.vertices).all():
        raise ValueError("SMPL forward pass produced non-finite vertices")
    model_x = smplx.create(
        str(root), model_type="smplx", gender="neutral", ext="npz", use_pca=False
    )
    smplx_result = model_x(
        global_orient=torch.zeros(1, 3),
        body_pose=torch.zeros(1, 63),
        left_hand_pose=torch.zeros(1, 45),
        right_hand_pose=torch.zeros(1, 45),
        betas=torch.zeros(1, int(model_x.num_betas)),
        expression=torch.zeros(1, int(model_x.num_expression_coeffs)),
        transl=torch.zeros(1, 3),
    )
    if not torch.isfinite(smplx_result.vertices).all():
        raise ValueError("SMPL-X forward pass produced non-finite vertices")
    report = {
        "assets_committed_to_git": False,
        "body_models_root": str(configured_root),
        "smpl": {
            "path": str(configured_root / "smpl_chumpyfree" / "smpl" / "SMPL_NEUTRAL.pkl"),
            "format": "chumpy-free pkl",
            "size_bytes": smpl_file.stat().st_size,
            "sha256": sha256_file(smpl_file),
            "vertices": int(smpl_result.vertices.shape[1]),
            "joints": int(smpl_result.joints.shape[1]),
            "shape_coefficients": int(smpl.num_betas),
            "finite_forward": True,
        },
        "smplx": {
            "path": str(configured_root / "smplx" / "SMPLX_NEUTRAL.npz"),
            "format": "npz",
            "size_bytes": smplx_file.stat().st_size,
            "sha256": sha256_file(smplx_file),
            "vertices": int(smplx_result.vertices.shape[1]),
            "joints": int(smplx_result.joints.shape[1]),
            "shape_coefficients": int(model_x.num_betas),
            "expression_coefficients": int(model_x.num_expression_coeffs),
            "finite_forward": True,
        },
        "original_smpl_pickle_used": False,
        "license_files_redistributed": False,
    }
    atomic_write_yaml(output, report)
    return report
