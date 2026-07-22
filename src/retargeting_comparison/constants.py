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

# Public outputs remain at their immutable v1 paths.  Controlled baselines were
# rerun after the non-result-driven evaluator-v2 scale/root-anchor/temporal
# correction.  Earlier v2 artifacts remain immutable under runs/.
STAGE1_RUN_DIRECTORIES = {
    "sparse-neutral": "sparse-neutral-v3",
    "sparse-a": "sparse-a-v3",
    "sparse-b": "sparse-b-v3",
    "dense": "dense-v3",
    "gmr": "gmr",
    "omniretarget": "omniretarget",
}
