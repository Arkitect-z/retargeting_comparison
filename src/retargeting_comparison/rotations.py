"""Quaternion operations using canonical wxyz storage."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def normalize_quaternion_wxyz(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    norms = np.linalg.norm(value, axis=-1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("Zero-norm quaternion")
    return value / norms


def wxyz_to_xyzw(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion)
    return np.concatenate((value[..., 1:], value[..., :1]), axis=-1)


def xyzw_to_wxyz(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion)
    return np.concatenate((value[..., -1:], value[..., :3]), axis=-1)


def matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape[-2:] != (3, 3):
        raise ValueError("Rotation matrices must end in shape [3,3]")
    leading = value.shape[:-2]
    xyzw = Rotation.from_matrix(value.reshape(-1, 3, 3)).as_quat().reshape(*leading, 4)
    result = xyzw_to_wxyz(xyzw)
    # q and -q encode the same rotation; canonicalize sign for deterministic files.
    result = np.where(result[..., :1] < 0.0, -result, result)
    return normalize_quaternion_wxyz(result)


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    value = normalize_quaternion_wxyz(quaternion)
    if value.shape[-1] != 4:
        raise ValueError("Quaternions must end in shape [4]")
    leading = value.shape[:-1]
    return Rotation.from_quat(wxyz_to_xyzw(value).reshape(-1, 4)).as_matrix().reshape(
        *leading, 3, 3
    )


def yaw_from_matrix(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix)
    return np.arctan2(value[..., 1, 0], value[..., 0, 0])
