from __future__ import annotations

from pathlib import Path

import numpy as np

from retargeting_comparison.bvh import forward_kinematics, parse_bvh
from retargeting_comparison.source import SourceFeatures, _remove_short_true_runs, pilot_selection_key

BVH = """HIERARCHY
ROOT Hips
{
  OFFSET 0 0 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT LeftFoot
  {
    OFFSET 0 0 -100
    CHANNELS 3 Zrotation Xrotation Yrotation
    End Site
    {
      OFFSET 0 0 -10
    }
  }
  JOINT RightFoot
  {
    OFFSET 0 0 -100
    CHANNELS 3 Zrotation Xrotation Yrotation
    End Site
    {
      OFFSET 0 0 -10
    }
  }
}
MOTION
Frames: 2
Frame Time: 0.033333333333
0 0 100 0 0 0 0 0 0 0 0 0
1 0 100 0 0 0 0 0 0 0 0 0
"""


def test_bvh_parse_and_fk(tmp_path: Path) -> None:
    path = tmp_path / "tiny.bvh"
    path.write_text(BVH)
    motion = parse_bvh(path)
    local, world, positions, root = forward_kinematics(motion, position_scale=0.01)
    assert motion.joint_names == ("Hips", "LeftFoot", "RightFoot")
    assert positions.shape == (2, 3, 3)
    assert np.allclose(root[:, 0], [0.0, 0.01])
    assert np.allclose(local[..., 0], 1.0)
    assert np.allclose(world[..., 0], 1.0)


def test_short_contact_runs_are_removed() -> None:
    mask = np.asarray([False, True, True, False, True, True, True, False])
    assert _remove_short_true_runs(mask).tolist() == [
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        False,
    ]


def test_pilot_selection_key_is_duration_name_then_frame() -> None:
    def feature(source_file: str, frame_start: int, duration: float) -> SourceFeatures:
        return SourceFeatures(
            sequence_id="test",
            source_file=source_file,
            frame_start=frame_start,
            frame_end=frame_start + 600,
            frames=600,
            fps=30.0,
            duration_s=duration,
            root_path_m=2.0,
            root_mean_speed_mps=0.1,
            root_yaw_range_rad=0.5,
            wrist_range_m=0.5,
            contact_transitions=4,
            eligible=True,
            rejection_reason="",
        )

    candidates = [
        feature("B.bvh", 0, 20.0),
        feature("a.bvh", 300, 20.0),
        feature("A.bvh", 0, 20.0),
        feature("closest.bvh", 0, 20.1),
    ]
    assert min(candidates, key=pilot_selection_key) == candidates[2]
