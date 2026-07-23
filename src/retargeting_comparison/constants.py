"""Project-wide immutable experiment constants."""

FULL_LAFAN_STOP_MESSAGE = (
    "FULL-LAFAN EXPERIMENTS NOT STARTED — WAITING FOR USER APPROVAL."
)
CANONICAL_QPOS_WIDTH = 36
G1_JOINT_COUNT = 29
MIN_COMPLETION_RATIO = 0.95

G1_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

RUN_STATUSES = ("pending", "running", "succeeded", "incomplete", "failed", "na")

# Protocol-v6 controlled outputs correct the common-scale estimator and use the
# same canonical Holosoma scene as evaluation.  Public-method revisions v3/v2
# bind the corrected in-memory timing boundary and runtime pre-solver witnesses;
# older qpos may be numerically identical but are not admissible evidence.
STAGE1_RUN_DIRECTORIES = {
    "sparse-neutral": "sparse-neutral-v6",
    "sparse-a": "sparse-a-v6",
    "sparse-b": "sparse-b-v6",
    "dense": "dense-v6",
    "gmr": "gmr-v3",
    "omniretarget": "omniretarget-v3",
    "protomotions-v2.3": "protomotions-v2.3-v3",
    "protomotions-v3": "protomotions-v3-v3",
    "unitree-reference": "unitree-attributed-reference",
}
