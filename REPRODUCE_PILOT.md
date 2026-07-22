# Reproduce the Pilot

1. Follow `docs/UPSTREAM_SETUP.md`, verify frozen commits, and place licensed assets outside Git as recorded in `manifests/`.
2. Run `rtcmp audit`, `rtcmp prepare-source`, `rtcmp validate-models`, and `rtcmp freeze-evaluator` before any formal method.
3. In the `robot` environment run `rtcmp audit-source-adapters`; then run `rtcmp gate-candidates`.
4. Run `rtcmp run-method --method sparse --seed neutral --revision v3 --sequence manifests/pilot_sequence.yaml`; repeat with seeds A/B and run Dense with `--revision v3`. Run GMR and OmniRetarget at their frozen revisions.
5. Run both variants of `rtcmp run-interaction` for box and climb.
6. In conda env `vis`, run `rtcmp visualize-results`; then run `rtcmp build-report` and `rtcmp validate-stage1`.

Every run has an atomic status manifest and immutable output hash. Existing successful output is reused; a failed retry receives a new attempt directory. Raw datasets, body models, upstream history, trajectories, logs, and caches remain ignored.

FULL-LAFAN EXPERIMENTS NOT STARTED — WAITING FOR USER APPROVAL.
