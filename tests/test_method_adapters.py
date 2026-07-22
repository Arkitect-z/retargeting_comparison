from __future__ import annotations

import numpy as np

from retargeting_comparison.method_adapters import (
    HOLOSOMA_LAFAN_ORDER,
    prepare_holosoma_lafan_input,
)
from retargeting_comparison.schemas import CanonicalHuman


def test_holosoma_adapter_joint_sides_and_coordinate_roundtrip(tmp_path) -> None:
    source_names = [
        "LeftToe" if name == "LeftToeBase" else "RightToe" if name == "RightToeBase" else name
        for name in HOLOSOMA_LAFAN_ORDER
    ]
    frames = 2
    rotations = np.zeros((frames, len(source_names), 4))
    rotations[..., 0] = 1.0
    positions = np.arange(frames * len(source_names) * 3, dtype=float).reshape(
        frames, len(source_names), 3
    )
    human = CanonicalHuman(
        joint_names=np.asarray(source_names),
        parent_indices=np.asarray([-1] + [0] * (len(source_names) - 1)),
        local_rotations=rotations,
        world_rotations=rotations.copy(),
        world_positions=positions,
        root_translation=positions[:, 0],
        fps=30.0,
        timestamps=np.arange(frames) / 30.0,
        foot_contact_labels=np.zeros((frames, 2), dtype=bool),
        source_sha256="c" * 64,
    )
    path = prepare_holosoma_lafan_input(human, tmp_path, "pilot")
    native = np.load(path)
    names = {name: index for index, name in enumerate(human.joint_names.astype(str))}
    restored = native[..., [0, 2, 1]]
    left = HOLOSOMA_LAFAN_ORDER.index("LeftHand")
    right = HOLOSOMA_LAFAN_ORDER.index("RightHand")
    assert np.array_equal(restored[:, left], human.world_positions[:, names["LeftHand"]])
    assert np.array_equal(restored[:, right], human.world_positions[:, names["RightHand"]])
