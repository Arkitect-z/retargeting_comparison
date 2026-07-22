"""Dependency-light BVH parser and forward kinematics for LAFAN1."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .rotations import matrix_to_quaternion_wxyz

_TOKEN = re.compile(r"[{}]|[^\s{}]+")


@dataclass(frozen=True)
class BVHMotion:
    joint_names: tuple[str, ...]
    parent_indices: np.ndarray
    offsets: np.ndarray
    channels: tuple[tuple[str, ...], ...]
    channel_slices: tuple[slice, ...]
    frame_time: float
    channel_values: np.ndarray

    @property
    def fps(self) -> float:
        return 1.0 / self.frame_time

    @property
    def frame_count(self) -> int:
        return self.channel_values.shape[0]


class _Cursor:
    def __init__(self, tokens: list[str]):
        self.tokens = tokens
        self.index = 0

    def pop(self, expected: str | None = None) -> str:
        if self.index >= len(self.tokens):
            raise ValueError("Unexpected end of BVH")
        value = self.tokens[self.index]
        self.index += 1
        if expected is not None and value.lower() != expected.lower():
            raise ValueError(f"Expected {expected!r}, got {value!r}")
        return value

    def peek(self) -> str:
        return self.tokens[self.index]


def parse_bvh(path: str | Path) -> BVHMotion:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    cursor = _Cursor(_TOKEN.findall(text))
    cursor.pop("HIERARCHY")
    names: list[str] = []
    parents: list[int] = []
    offsets: list[list[float]] = []
    channels: list[tuple[str, ...]] = []

    def parse_joint(parent: int, keyword: str) -> None:
        cursor.pop(keyword)
        name = cursor.pop()
        index = len(names)
        names.append(name)
        parents.append(parent)
        offsets.append([0.0, 0.0, 0.0])
        channels.append(())
        cursor.pop("{")
        while cursor.peek() != "}":
            token = cursor.peek()
            if token.lower() == "offset":
                cursor.pop()
                offsets[index] = [float(cursor.pop()), float(cursor.pop()), float(cursor.pop())]
            elif token.lower() == "channels":
                cursor.pop()
                count = int(cursor.pop())
                channels[index] = tuple(cursor.pop() for _ in range(count))
            elif token.lower() == "joint":
                parse_joint(index, "JOINT")
            elif token.lower() == "end":
                cursor.pop("End")
                cursor.pop("Site")
                cursor.pop("{")
                cursor.pop("OFFSET")
                cursor.pop()
                cursor.pop()
                cursor.pop()
                cursor.pop("}")
            else:
                raise ValueError(f"Unexpected hierarchy token {token!r}")
        cursor.pop("}")

    parse_joint(-1, "ROOT")
    cursor.pop("MOTION")
    frames_token = cursor.pop()
    if frames_token.lower().rstrip(":") != "frames":
        raise ValueError(f"Expected Frames, got {frames_token!r}")
    frame_count = int(cursor.pop())
    cursor.pop("Frame")
    time_token = cursor.pop()
    if time_token.lower().rstrip(":") != "time":
        raise ValueError(f"Expected Time, got {time_token!r}")
    frame_time = float(cursor.pop())
    total_channels = sum(len(value) for value in channels)
    remaining = cursor.tokens[cursor.index :]
    if len(remaining) != frame_count * total_channels:
        raise ValueError(
            f"Expected {frame_count * total_channels} channel values, got {len(remaining)}"
        )
    values = np.asarray(remaining, dtype=np.float64).reshape(frame_count, total_channels)
    slices: list[slice] = []
    start = 0
    for value in channels:
        slices.append(slice(start, start + len(value)))
        start += len(value)
    return BVHMotion(
        joint_names=tuple(names),
        parent_indices=np.asarray(parents, dtype=np.int64),
        offsets=np.asarray(offsets, dtype=np.float64),
        channels=tuple(channels),
        channel_slices=tuple(slices),
        frame_time=frame_time,
        channel_values=values,
    )


def forward_kinematics(
    motion: BVHMotion,
    position_scale: float = 0.01,
    coordinate_transform: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frames = motion.frame_count
    joints = len(motion.joint_names)
    local_matrices = np.broadcast_to(np.eye(3), (frames, joints, 3, 3)).copy()
    local_translation = np.zeros((frames, joints, 3), dtype=np.float64)
    for joint, (names, channel_slice) in enumerate(zip(motion.channels, motion.channel_slices)):
        data = motion.channel_values[:, channel_slice]
        rotation_names = [name for name in names if name.lower().endswith("rotation")]
        if rotation_names:
            axes = "".join(name[0].upper() for name in rotation_names)
            angles = np.stack(
                [data[:, names.index(name)] for name in rotation_names], axis=-1
            )
            local_matrices[:, joint] = Rotation.from_euler(axes, angles, degrees=True).as_matrix()
        for axis, label in enumerate(("Xposition", "Yposition", "Zposition")):
            if label in names:
                local_translation[:, joint, axis] = data[:, names.index(label)] * position_scale

    offsets = motion.offsets * position_scale
    world_matrices = np.empty_like(local_matrices)
    world_positions = np.empty((frames, joints, 3), dtype=np.float64)
    for joint, parent in enumerate(motion.parent_indices):
        if parent == -1:
            world_matrices[:, joint] = local_matrices[:, joint]
            # LAFAN's root position channels already contain the global offset;
            # adding the ROOT OFFSET again doubles the initial translation.
            world_positions[:, joint] = local_translation[:, joint]
        else:
            world_matrices[:, joint] = world_matrices[:, parent] @ local_matrices[:, joint]
            local_offset = offsets[joint] + local_translation[:, joint]
            world_positions[:, joint] = world_positions[:, parent] + np.einsum(
                "tij,tj->ti", world_matrices[:, parent], local_offset
            )
    if coordinate_transform is not None:
        transform = np.asarray(coordinate_transform, dtype=np.float64)
        if transform.shape != (3, 3) or not np.allclose(transform.T @ transform, np.eye(3)):
            raise ValueError("coordinate_transform must be a 3x3 orthonormal matrix")
        world_matrices = transform @ world_matrices
        local_matrices[:, 0] = transform @ local_matrices[:, 0]
        world_positions = world_positions @ transform.T
        local_translation[:, 0] = local_translation[:, 0] @ transform.T
    return (
        matrix_to_quaternion_wxyz(local_matrices),
        matrix_to_quaternion_wxyz(world_matrices),
        world_positions,
        world_positions[:, 0].copy(),
    )
