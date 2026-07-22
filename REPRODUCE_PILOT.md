# Reproduce the Pilot

1. Follow `docs/UPSTREAM_SETUP.md`, verify checkout commits, and place licensed assets outside Git as recorded in `manifests/`.
2. Run `rtcmp audit`, `rtcmp prepare-source`, and `rtcmp validate-models`.
3. Run core methods in `manifests/experiment_order.yaml`; Sparse uses neutral, A, and B.
4. Run `rtcmp run-interaction --case box --variant full`, repeat with `no-hard`, then repeat both variants for `climb`.
5. In conda env `vis`, run `rtcmp visualize-results` to build the complete synchronized Rerun recording and provenance manifest.
6. Run `rtcmp build-report` (which also rebuilds `INTERACTIVE_REPORT.html`) and `rtcmp validate-stage1`.

Every run has an atomic status manifest and immutable output hash. Existing successful output is reused; a failed retry receives a new attempt directory. Raw datasets, body models, upstream history, trajectories, logs, and caches remain ignored.

FULL-LAFAN EXPERIMENTS NOT STARTED — WAITING FOR USER APPROVAL.
