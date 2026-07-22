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

See `pilot/REPRODUCE_PILOT.md` after the Stage 1 run for frozen commands and
artifact locations.

## License and assets

Original code is licensed under Apache-2.0. Upstream methods and restricted
data/model assets retain their own licenses and are not redistributed here.

