# Reproduce the Pilot

1. Follow `docs/UPSTREAM_SETUP.md`, verify frozen commits, and place licensed assets outside Git as recorded in `manifests/`.
2. Run `rtcmp audit`, `rtcmp prepare-source`, `rtcmp validate-models`, and `rtcmp freeze-evaluator` before any formal method.
3. Freeze the preprocessing-policy manifest and pre-solver target contract from `research/OFFICIAL_SCALE_AND_PREPROCESSING_AUDIT.md` and `configs/scale_policy_sensitivity.yaml`.
4. Run the legacy Sparse/Dense/GMR/OmniRetarget commands, then the required ProtoMotions v2.3 and v3 (`target_raw_frames=600`) adapters and full Pilot runs.
5. Capture every native pre-solver target; run the controlled policy transplant, registered root/local ±5% variants, contact-label diagnostics, and the hash-bound neutral-SMPL-X actor-shape policy formula probe (not runtime-constructor or ranking evidence).
6. Run both variants of `rtcmp run-interaction` for box and climb.
7. Build the publication evaluator tables first with `PYTHONPATH=src python -c 'from retargeting_comparison.stage1_publication import build_stage1_publication; build_stage1_publication(".")'`. These are the only metric tables accepted by visualization.
8. In conda env `vis`, run `rtcmp visualize-results`. This writes the exact nine-trajectory articulated-G1 RRD and must pass `rerun rrd verify`, decoded stats, entity/component, frame-marker, view-instance, and hash-binding checks.
9. Run the critical integration suite with `RTCMP_FINAL_TESTS=1`; absent ignored/generated Stage 1 evidence is a failure in this mode, never a skip. Then run `rtcmp build-report`: it deterministically rebuilds the same publication metrics, renders a PENDING browser artifact, validates once, finalizes Markdown from that verdict, renders the final HTML, and writes its post-render delivery audit sidecar. Validation must remain `NO-GO` while any revised requirement is absent.

Every run has an atomic status manifest and immutable output hash. Existing successful output is reused; a failed retry receives a new attempt directory. Raw datasets, body models, upstream history, trajectories, logs, and caches remain ignored.

FULL-LAFAN EXPERIMENTS NOT STARTED — WAITING FOR USER APPROVAL.
