# Sparse IK Analysis

Sparse tracks root translation/yaw, both wrists, and both ankles. It uses neutral and two deterministic perturbed initial postures, followed by sequential warm start. No torso, elbow, knee, contact, or learned prior is present.

| pair                     |   joint_angle_rms_mean_rad |   joint_angle_rms_max_rad |   robot_rf_point_rms_mean_m |   robot_rf_point_rms_max_m |
|:-------------------------|---------------------------:|--------------------------:|----------------------------:|---------------------------:|
| sparse-a__sparse-b       |                    0.13255 |                   0.24391 |                     0.02013 |                    0.07639 |
| sparse-neutral__sparse-a |                    0.05822 |                   0.12617 |                     0.00729 |                    0.06501 |
| sparse-neutral__sparse-b |                    0.10453 |                   0.21934 |                     0.01758 |                    0.04658 |

The pairwise table quantifies hidden-state sensitivity in joint space and in root-frame robot point space. Main comparisons use the neutral seed; A/B are diagnostics and are not silently averaged into the operating point.

FULL-LAFAN EXPERIMENTS NOT STARTED — WAITING FOR USER APPROVAL.
