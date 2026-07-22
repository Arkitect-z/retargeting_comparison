# Sparse IK Analysis

Sparse tracks root translation/yaw plus both wrists and ankles. Dense adds torso, head, shoulders, elbows, hips, knees, and toes. Both use the common scale `0.742037044`, the same geometry-derived rigid root anchor, joint limits, sequential warm start, weak fixed-posture regularization, and the same explicit weak `q[t-1]` temporal cost. Sparse contains no torso/elbow/knee task, contact objective, or learned prior.

| pair                     |   joint_angle_rms_mean_rad |   joint_angle_rms_max_rad |   robot_rf_point_rms_mean_m |   robot_rf_point_rms_max_m |
|:-------------------------|---------------------------:|--------------------------:|----------------------------:|---------------------------:|
| sparse-a__sparse-b       |                    0.11260 |                   0.16064 |                     0.02686 |                    0.08176 |
| sparse-neutral__sparse-a |                    0.05694 |                   0.13200 |                     0.00989 |                    0.03744 |
| sparse-neutral__sparse-b |                    0.09651 |                   0.14683 |                     0.02276 |                    0.07804 |

| label          |   root_position_residual_mean_m |   root_position_residual_p95_m |   four_ee_position_residual_mean_m |   four_ee_position_residual_p95_m |   dense_added_position_residual_mean_m |   all_declared_position_residual_mean_m |
|:---------------|--------------------------------:|-------------------------------:|-----------------------------------:|----------------------------------:|---------------------------------------:|----------------------------------------:|
| sparse-neutral |                         0.00016 |                        0.00077 |                            0.00486 |                           0.01489 |                                0.00000 |                                 0.00392 |
| sparse-a       |                         0.00016 |                        0.00077 |                            0.00490 |                           0.01487 |                                0.00000 |                                 0.00395 |
| sparse-b       |                         0.00015 |                        0.00073 |                            0.00484 |                           0.01489 |                                0.00000 |                                 0.00391 |
| dense          |                         0.00858 |                        0.01510 |                            0.01610 |                           0.03280 |                                0.10105 |                                 0.07562 |

The pairwise table quantifies hidden-state sensitivity in joint space and root-frame robot point space. Main comparisons use neutral; A/B are diagnostics and are never averaged into the operating point. Target residual and untracked-pose divergence are separate claims.

FULL-LAFAN EXPERIMENTS NOT STARTED — WAITING FOR USER APPROVAL.
