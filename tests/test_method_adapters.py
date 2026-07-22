from __future__ import annotations

import numpy as np

from retargeting_comparison.method_adapters import (
    HOLOSOMA_LAFAN_ORDER,
    prepare_holosoma_lafan_input,
)
from retargeting_comparison.schemas import CanonicalHuman
from retargeting_comparison.source import foot_contacts
from retargeting_comparison.source_adapter_audit import adapter_error_metrics


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


def test_identical_source_adapter_has_zero_geometric_error() -> None:
    names = [
        "Hips",
        "LeftUpLeg",
        "RightUpLeg",
        "LeftFoot",
        "RightFoot",
        "LeftArm",
        "RightArm",
        "LeftHand",
        "RightHand",
    ]
    frames = 4
    positions = np.zeros((frames, len(names), 3), dtype=np.float64)
    index = {name: position for position, name in enumerate(names)}
    for frame in range(frames):
        positions[frame, index["Hips"]] = [0.02 * frame, 0.0, 1.0]
        positions[frame, index["LeftUpLeg"]] = [0.02 * frame, -0.1, 0.9]
        positions[frame, index["RightUpLeg"]] = [0.02 * frame, 0.1, 0.9]
        positions[frame, index["LeftFoot"]] = [0.02 * frame, -0.1, 0.0]
        positions[frame, index["RightFoot"]] = [0.02 * frame, 0.1, 0.0]
        positions[frame, index["LeftArm"]] = [0.02 * frame, -0.25, 1.3]
        positions[frame, index["RightArm"]] = [0.02 * frame, 0.25, 1.3]
        positions[frame, index["LeftHand"]] = [0.02 * frame, -0.6, 1.2]
        positions[frame, index["RightHand"]] = [0.02 * frame, 0.6, 1.2]
    rotations = np.zeros((frames, len(names), 4), dtype=np.float64)
    rotations[..., 0] = 1.0
    contacts = foot_contacts(positions, tuple(names), 30.0)
    human = CanonicalHuman(
        joint_names=np.asarray(names),
        parent_indices=np.asarray([-1] + [0] * (len(names) - 1)),
        local_rotations=rotations,
        world_rotations=rotations.copy(),
        world_positions=positions,
        root_translation=positions[:, 0],
        fps=30.0,
        timestamps=np.arange(frames) / 30.0,
        foot_contact_labels=contacts,
        source_sha256="d" * 64,
    )

    metrics = adapter_error_metrics(human, names, positions.copy())

    assert metrics["root_aligned_mpjpe_m"] == 0.0
    assert metrics["bone_length_error_mean_m"] == 0.0
    assert metrics["root_translation_error_mean_m"] == 0.0
    assert metrics["yaw_error_mean_rad"] == 0.0
    assert metrics["foot_contact_agreement"] == 1.0
    assert metrics["left_right_match"] is True
