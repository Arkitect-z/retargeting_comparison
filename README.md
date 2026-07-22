# Human-to-G1 Retargeting Comparison

Reproducible Stage 0–1 research harness for comparing public human-motion to
Unitree G1 29-DoF retargeting systems on one frozen LAFAN1 Pilot sequence and
two official OmniRetarget interaction cases.

The repository contains original orchestration, adapters, evaluation code,
tests, small metrics, figures, and English Markdown reports. Licensed body
models, source datasets, upstream repositories, generated trajectories, and
videos remain external and are referenced by immutable hashes.

## Safety boundary

This repository must not start a full-LAFAN run without explicit user approval.
The Stage 1 validator enforces this boundary.

## Quick start

```bash
python -m pip install -e '.[dev]'
rtcmp --help
pytest
```

See [`REPRODUCE_PILOT.md`](REPRODUCE_PILOT.md) for frozen commands and
artifact locations.

Upstream checkout and environment instructions are in
[`docs/UPSTREAM_SETUP.md`](docs/UPSTREAM_SETUP.md). A normal Pilot method run
uses one fresh cold process followed by one warm-up and three measured runs:

```bash
rtcmp run-method --method sparse --sequence manifests/pilot_sequence.yaml --seed neutral
rtcmp run-method --method dense --sequence manifests/pilot_sequence.yaml
rtcmp run-method --method gmr --sequence manifests/pilot_sequence.yaml
rtcmp run-method --method omniretarget --sequence manifests/pilot_sequence.yaml
```

## Stage 1 deliverables

The decision is recorded in [`GO_NO_GO.md`](GO_NO_GO.md). Detailed evidence is
in [`PILOT_REPORT.md`](PILOT_REPORT.md), [`METHOD_SCOPE.md`](METHOD_SCOPE.md),
[`SPARSE_IK_ANALYSIS.md`](SPARSE_IK_ANALYSIS.md), and
[`INTERACTION_CASE_STUDY.md`](INTERACTION_CASE_STUDY.md). The concise briefing
is available as [`EXECUTIVE_SUMMARY.md`](EXECUTIVE_SUMMARY.md) and the
12-section [`PRESENTATION.md`](PRESENTATION.md). A dependency-free visual
narrative is available in [`INTERACTIVE_REPORT.html`](INTERACTIVE_REPORT.html);
rebuild it independently with `rtcmp build-interactive-report`. Machine-readable metrics,
figure source data, provenance, and validation results live under `metrics/`,
`figures/`, and `manifests/`.

For synchronized 3D inspection of the source, all four operating points, and
Sparse seeds A/B, use the Rerun workflow in
[`docs/RERUN_VISUALIZATION.md`](docs/RERUN_VISUALIZATION.md). The complete `.rrd`
is rebuildable from ignored canonical trajectories and is therefore not
committed. The plan-by-plan closure record is in
[`docs/STAGE1_COMPLETION_AUDIT.md`](docs/STAGE1_COMPLETION_AUDIT.md).

## License and assets

Original code is licensed under Apache-2.0. Upstream methods and restricted
data/model assets retain their own licenses and are not redistributed here.
