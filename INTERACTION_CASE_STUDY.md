# Interaction Case Study

| case   | variant   |   strict_contact_2cm_frame_rate |   near_contact_5cm_frame_rate |   proximity_10cm_frame_rate |   penetration_any_frame_rate |   penetration_frame_rate |   foot_sticking_violation_frame_rate |   end_to_end_rtf |
|:-------|:----------|--------------------------------:|------------------------------:|----------------------------:|-----------------------------:|-------------------------:|-------------------------------------:|-----------------:|
| box    | full      |                          0.9133 |                        0.9541 |                      1.0000 |                       0.8367 |                   0.0000 |                               0.0000 |          25.0097 |
| box    | no-hard   |                          0.9133 |                        0.9745 |                      1.0000 |                       0.8367 |                   0.8367 |                               0.8622 |           2.1380 |
| climb  | full      |                          0.8688 |                        0.8759 |                      0.9301 |                       0.7532 |                   0.2439 |                               0.0171 |          11.5533 |
| climb  | no-hard   |                          0.8887 |                        0.8916 |                      0.9358 |                       0.7603 |                   0.7518 |                               0.7233 |           2.1315 |

Full enables object non-penetration, foot sticking, and joint limits. No-Hard disables only the first two flags; initialization, input, object sampling seed, solver, iteration budget, and joint limits remain unchanged. A regression test checks that the patched non-penetration gate and the upstream foot-sticking gate both alter constraint construction.

The 2 cm, 5 cm, and 10 cm thresholds use signed geometry-surface distances from `mujoco.mj_geomDistance`, never distance to the object origin. `penetration_any_frame_rate` records every negative distance, while the primary `penetration_frame_rate` applies the frozen 1.1 mm tolerance-aware threshold. Results are two-case case-study evidence only.

FULL-LAFAN EXPERIMENTS NOT STARTED — WAITING FOR USER APPROVAL.
