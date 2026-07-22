"""Canonical MuJoCo G1 model wrapper used by every evaluator path."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from .constants import CANONICAL_QPOS_WIDTH, G1_JOINT_NAMES
from .io_utils import sha256_file


SEMANTIC_BODIES = {
    "root": "pelvis",
    "torso": "torso_link",
    "left_shoulder": "left_shoulder_roll_link",
    "right_shoulder": "right_shoulder_roll_link",
    "left_elbow": "left_elbow_link",
    "right_elbow": "right_elbow_link",
    "left_wrist": "left_wrist_yaw_link",
    "right_wrist": "right_wrist_yaw_link",
    "left_hip": "left_hip_roll_link",
    "right_hip": "right_hip_roll_link",
    "left_knee": "left_knee_link",
    "right_knee": "right_knee_link",
    "left_ankle": "left_ankle_roll_link",
    "right_ankle": "right_ankle_roll_link",
    "left_toe": "left_foot_contact_point",
    "right_toe": "right_foot_contact_point",
}


class CanonicalRobotModel:
    def __init__(self, xml_path: str | Path):
        self.xml_path = Path(xml_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        if self.model.nq != CANONICAL_QPOS_WIDTH:
            raise ValueError(f"Canonical model nq={self.model.nq}, expected 36")
        joint_names = tuple(
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, index)
            for index in range(1, self.model.njnt)
        )
        if joint_names != G1_JOINT_NAMES:
            raise ValueError("Canonical G1 joint order differs from the frozen 29-DoF contract")
        self.body_ids = {
            semantic: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
            for semantic, body in SEMANTIC_BODIES.items()
        }
        if any(value < 0 for value in self.body_ids.values()):
            raise ValueError("Canonical model is missing a semantic evaluation body")
        self.floor_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        self.sha256 = sha256_file(self.xml_path)

    def forward(self, qpos: np.ndarray) -> None:
        value = np.asarray(qpos, dtype=np.float64)
        if value.shape != (CANONICAL_QPOS_WIDTH,):
            raise ValueError("qpos must have shape [36]")
        self.data.qpos[:] = value
        mujoco.mj_forward(self.model, self.data)

    def semantic_positions(self, qpos: np.ndarray) -> dict[str, np.ndarray]:
        self.forward(qpos)
        positions = {
            name: self.data.xpos[index].copy() for name, index in self.body_ids.items()
        }
        torso = self.body_ids["torso"]
        torso_rotation = self.data.xmat[torso].reshape(3, 3)
        positions["head"] = positions["torso"] + torso_rotation @ np.asarray(
            [0.0, 0.0, 0.35]
        )
        return positions

    def joint_limit_violation(self, qpos: np.ndarray, tolerance: float = 1e-4) -> float:
        value = np.asarray(qpos, dtype=np.float64)
        maximum = 0.0
        for joint_id in range(1, self.model.njnt):
            if not self.model.jnt_limited[joint_id]:
                continue
            address = self.model.jnt_qposadr[joint_id]
            lower, upper = self.model.jnt_range[joint_id]
            maximum = max(maximum, float(lower - value[address]), float(value[address] - upper))
        return max(0.0, maximum - tolerance)

    def contact_diagnostics(self, qpos: np.ndarray) -> tuple[float, int]:
        """Return maximum floor penetration and non-floor contact count."""
        self.forward(qpos)
        penetration = 0.0
        non_floor = 0
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if self.floor_geom_id in (contact.geom1, contact.geom2):
                penetration = max(penetration, -float(contact.dist))
            else:
                non_floor += 1
        return penetration, non_floor


def default_robot_scene(repo_root: str | Path = ".") -> Path:
    return (
        Path(repo_root)
        / "external"
        / "holosoma"
        / "src"
        / "holosoma"
        / "holosoma"
        / "data"
        / "robots"
        / "g1"
        / "scenes"
        / "scene_g1_29dof_wbt_plane.xml"
    )

