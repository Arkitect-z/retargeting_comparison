from __future__ import annotations

import numpy as np
import pytest

from retargeting_comparison.interaction import mesh_surface_diagnostics


def test_surface_metrics_use_geometry_distance_not_object_origin() -> None:
    mujoco = pytest.importorskip("mujoco")

    xml = """
    <mujoco><worldbody>
      <body name='robot' pos='1.2 0 0'><freejoint/><geom name='robot_surface' type='sphere' size='0.1'/></body>
      <body name='largebox_link'><geom name='largebox' type='box' size='1 1 1'/></body>
    </worldbody></mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    qpos = np.tile(model.qpos0, (2, 1))
    rows, summary = mesh_surface_diagnostics(
        model,
        qpos,
        "largebox",
        ["robot"],
        [{"left": False}, {"left": False}],
        0.001,
    )
    # Origin distance is 1.2 m, but the actual box/sphere surfaces are 0.1 m apart.
    assert rows[0]["minimum_robot_object_surface_distance_m"] == pytest.approx(0.1)
    assert summary["proximity_10cm_frame_rate"] == pytest.approx(1.0)
