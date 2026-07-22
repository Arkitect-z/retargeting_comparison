import numpy as np
from scipy.spatial.transform import Rotation

from retargeting_comparison.rotations import (
    matrix_to_quaternion_wxyz,
    quaternion_wxyz_to_matrix,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)


def test_quaternion_order_roundtrip() -> None:
    wxyz = np.asarray([[0.5, 0.5, 0.5, 0.5], [1.0, 0.0, 0.0, 0.0]])
    assert np.allclose(xyzw_to_wxyz(wxyz_to_xyzw(wxyz)), wxyz)


def test_matrix_quaternion_roundtrip() -> None:
    matrix = Rotation.from_euler("XYZ", [[20.0, -10.0, 35.0]], degrees=True).as_matrix()
    restored = quaternion_wxyz_to_matrix(matrix_to_quaternion_wxyz(matrix))
    assert np.allclose(restored, matrix, atol=1e-12)

