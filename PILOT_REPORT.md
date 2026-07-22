# Human-to-G1 Retargeting Pilot Report

## Scope and frozen design

This Stage 1 experiment compares controlled Sparse and Dense Mink baselines, official GMR, and official OmniRetarget/Holosoma on the preselected 600-frame (`19.9998 s`) LAFAN1 window `dance1_subject1_f000000_000600`. The sequence, thresholds, method commits, evaluator model, and timing order were frozen before formal runs. Full-LAFAN was not started.

## Main results

| label          |   rf_kpe_all_mean_m |   rf_kpe_targeted_mean_m |   rf_kpe_untracked_mean_m |   artifact_rate |   end_to_end_rtf_median |   native_core_rtf_median |
|:---------------|--------------------:|-------------------------:|--------------------------:|----------------:|------------------------:|-------------------------:|
| dense          |              0.3221 |                   0.4368 |                    0.2839 |          0.3333 |                  0.0612 |                   0.0568 |
| gmr            |              0.3239 |                   0.4354 |                    0.2867 |          0.0667 |                  0.1192 |                   0.1192 |
| omniretarget   |              0.3167 |                   0.4245 |                    0.2808 |          0.5333 |                  9.5957 |                   9.3686 |
| sparse-neutral |              0.3255 |                   0.4385 |                    0.2878 |          0.9567 |                  0.0285 |                   0.0257 |

`omniretarget` has the lowest RF-KPE-all at this single operating point; `sparse-neutral` is fastest by median end-to-end RTF. These are Pilot observations, not dataset-level rankings. Adapter error is reported separately in `metrics/source_adapter_errors.csv`.

## Interpretation

The targeted and untracked columns are intentionally separate: a method can match hands/feet while degrading torso or limb structure. Temporal and artifact fields remain disaggregated in `metrics/runs/*_summary.json` and per-frame CSV/Parquet files. No composite score is used.

## Synchronized visual inspection

The rebuildable Rerun recording synchronizes the source human, all four operating points, and Sparse seeds A/B. Separate side-by-side world, overlaid world, root-frame, and Sparse-seed views expose root tracking, ground penetration, foot skating, and hidden-pose divergence. The viewer replays canonical outputs and is excluded from formal method timing; see `docs/RERUN_VISUALIZATION.md` and `manifests/rerun_visualization.json`.

## Interaction case study

| case   | variant   |   strict_contact_2cm_frame_rate |   near_contact_5cm_frame_rate |   proximity_10cm_frame_rate |   penetration_any_frame_rate |   penetration_frame_rate |   foot_sticking_violation_frame_rate |   end_to_end_rtf |
|:-------|:----------|--------------------------------:|------------------------------:|----------------------------:|-----------------------------:|-------------------------:|-------------------------------------:|-----------------:|
| box    | full      |                          0.9133 |                        0.9541 |                      1.0000 |                       0.8367 |                   0.0000 |                               0.0000 |          25.0097 |
| box    | no-hard   |                          0.9133 |                        0.9745 |                      1.0000 |                       0.8367 |                   0.8367 |                               0.8622 |           2.1380 |
| climb  | full      |                          0.8688 |                        0.8759 |                      0.9301 |                       0.7532 |                   0.2439 |                               0.0171 |          11.5533 |
| climb  | no-hard   |                          0.8887 |                        0.8916 |                      0.9358 |                       0.7603 |                   0.7518 |                               0.7233 |           2.1315 |

Distances are computed with MuJoCo geometry-surface queries over the actual collision meshes. `penetration_any_frame_rate` records every negative signed distance; the primary `penetration_frame_rate` records depth over 1.1 mm (the frozen 1 mm constraint tolerance plus 0.1 mm numerical margin). The evidence covers exactly one box sequence and one climbing sequence and must not be generalized to a dataset.

## Timing protocol

Each core run used a fresh cold process, one warm-up, and three measured warm repetitions with one CPU thread and no visualization. Raw end-to-end and native-core values are in `metrics/timing_raw.csv`; cold/import/initialization overhead remains separate.

## Stage 2 gate

| method         |   measured_pilot_end_to_end_rtf |   projected_steady_wall_hours |   projected_startup_and_adapter_hours |   raw_projected_wall_hours |   safety_factor |   safe_projected_wall_hours |
|:---------------|--------------------------------:|------------------------------:|--------------------------------------:|---------------------------:|----------------:|----------------------------:|
| sparse-neutral |                           0.029 |                         0.131 |                                 0.012 |                      0.143 |           1.500 |                       0.214 |
| dense          |                           0.061 |                         0.281 |                                 0.012 |                      0.294 |           1.500 |                       0.440 |
| gmr            |                           0.119 |                         0.548 |                                 0.039 |                      0.587 |           1.500 |                       0.880 |
| omniretarget   |                           9.596 |                        44.129 |                                 0.149 |                     44.278 |           1.500 |                      66.416 |

The runtime estimate combines steady RTF with measured cold-import and initialization/adapter overhead once for each of the 77 source sequences; the observed formal-run retry rate was zero. After the required 1.5× safety factor, the serial projection is 67.95 h and 3.47 GB. The runtime projection therefore does not fit the 48-hour gate. Stage 2 requires a new design discussion and explicit approval.

FULL-LAFAN EXPERIMENTS NOT STARTED — WAITING FOR USER APPROVAL.
