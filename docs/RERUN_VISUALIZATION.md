# Rerun Stage 1 visualization

The Rerun viewer replays the canonical human source and every frozen Stage 1
G1 output on one synchronized timeline. It reads existing `.npz` trajectories
and per-frame evaluator CSVs; it never invokes a retargeter.

## Build the complete recording

Run from the repository root in the existing `vis` conda environment:

```bash
PYTHONPATH=src conda run -n vis --no-capture-output \
  python -m retargeting_comparison.cli visualize-results
```

This writes the ignored, rebuildable recording to
`artifacts/visualization/stage1_comparison.rrd` and commits only its portable
provenance in `manifests/rerun_visualization.json`. The command fails if any of
Sparse neutral/A/B, Dense, GMR, OmniRetarget, their per-frame metrics, or the
canonical source is missing or non-finite.

Open the saved recording with:

```bash
conda run -n vis rerun artifacts/visualization/stage1_comparison.rrd
```

To stream while building, add `--spawn`. For a fast diagnostic recording, use
`--max-frames 60`; use `--stride 2` only for visualization diagnostics, never
as a replacement for the full 600-frame acceptance recording.

## Views and encodings

- **Side-by-side · world motion** shows source, all four operating points, and
  both Sparse diagnostic seeds in separate lanes. Root displacement and ground
  height are preserved.
- **World overlay · root tracking** aligns the initial horizontal root position
  and overlays source plus the four operating points. It exposes translation,
  yaw, and trajectory-scale differences hidden by RF-KPE.
- **Root-frame pose overlay** removes each motion's root translation and yaw.
  This is the direct visual companion to root-frame KPE.
- **Sparse seed ambiguity** overlays neutral, A, and B after root alignment.
- Metric tabs show RF-KPE, root translation/yaw, ground penetration, binary
  artifact flags, and solve time from the frozen evaluator outputs.

Skeleton and joint colors identify methods. Green feet denote frozen source
stance without skating; red feet denote source stance with target foot speed
above the frozen `1 cm/s` threshold; gray feet are in swing. The displayed G1
semantic FK is calculated from the frozen Holosoma URDF and is regression-tested
against the canonical MuJoCo evaluator at neutral and random qpos.

The `.rrd` contains derived motion visualization and remains outside public Git,
just like canonical trajectories and videos. Its manifest records the source,
all six output hashes, G1 URDF hash, Rerun version, frame count, and recording
hash.

FULL-LAFAN EXPERIMENTS NOT STARTED — WAITING FOR USER APPROVAL.
