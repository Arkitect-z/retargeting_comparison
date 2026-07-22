# Reproduce the Pilot

1. Follow `docs/UPSTREAM_SETUP.md`, verify frozen commits, and place licensed assets outside Git as recorded in `manifests/`.
2. Run `rtcmp audit`, `rtcmp prepare-source`, `rtcmp validate-models`, and `rtcmp freeze-evaluator` before any formal method.
3. Freeze the preprocessing-policy manifest and pre-solver target contract from `research/OFFICIAL_SCALE_AND_PREPROCESSING_AUDIT.md` and `configs/scale_policy_sensitivity.yaml`.
4. Run the legacy Sparse/Dense/GMR/OmniRetarget commands, then the required ProtoMotions v2.3 and v3 (`target_raw_frames=600`) adapters and full Pilot runs.
5. Capture every native pre-solver target; run the controlled policy transplant, registered root/local ±5% variants, contact-label diagnostics, and neutral-SMPL-X actor-shape target probe.
6. Run both variants of `rtcmp run-interaction` for box and climb.
7. In conda env `vis`, rebuild all articulated-G1 views; then run `rtcmp build-report` and `rtcmp validate-stage1`. Validation must remain `NO-GO` while any revised requirement is absent.

Every run has an atomic status manifest and immutable output hash. Existing successful output is reused; a failed retry receives a new attempt directory. Raw datasets, body models, upstream history, trajectories, logs, and caches remain ignored.

FULL-LAFAN EXPERIMENTS NOT STARTED — WAITING FOR USER APPROVAL.
