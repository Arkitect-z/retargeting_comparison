# Rerun Stage 1 visualization

The Rerun viewer replays the canonical human source and every registered Stage 1
G1 output on one source-frame-index timeline. Display time comes from the
canonical source; the nominal-30-Hz external reference is frame-index aligned,
not claimed to have identical timestamps. Every target trajectory—including
ProtoMotions v2.3, ProtoMotions v3, and the Unitree-attributed external
reference—is rendered as the complete articulated G1 robot, using all 35 visual
meshes from the frozen Holosoma G1 29-DoF URDF. No comparison method is rendered
as a stick figure. The viewer reads existing `.npz` trajectories and per-frame
evaluator CSVs; it never invokes a retargeter.

## Build the complete recording

Run from the repository root in the existing `vis` conda environment:

```bash
PYTHONPATH=src conda run -n vis --no-capture-output \
  python -m retargeting_comparison.cli visualize-results
```

This writes the ignored, rebuildable recording to
`artifacts/visualization/stage1_comparison.rrd` and commits only its portable
provenance in `manifests/rerun_visualization.json`. The command fails if any of
Sparse neutral/A/B, Dense, GMR, OmniRetarget, ProtoMotions v2.3,
ProtoMotions v3, the Unitree-attributed reference, their per-frame metrics, or
the canonical source is missing or non-finite. Inputs resolve only through the
current `STAGE1_RUN_DIRECTORIES` registry and
`metrics/stage1_publication/runs`; legacy run/metric fallbacks are forbidden.
Each trajectory binds its output, evaluator CSV/summary, canonical source,
evaluator protocol, evaluator robot XML, and visual URDF by SHA-256.

Open the saved recording with:

```bash
conda run -n vis rerun artifacts/visualization/stage1_comparison.rrd
```

To stream while building, add `--spawn`. For a fast diagnostic recording, use
`--max-frames 60`; use `--stride 2` only for visualization diagnostics, never
as a replacement for the full shared 450-frame acceptance recording. The
source and five other trajectories retain their 600-frame artifacts, but the
synchronized comparison stops at frame 449 because ProtoMotions v3's official
contract is 450 frames.

ProtoMotions v3 may be unavailable while its whole-trajectory solver is still
running. A diagnostic recording can explicitly skip incomplete methods through
the Python API:

```bash
PYTHONPATH=src conda run -n vis --no-capture-output python -c \
  'from retargeting_comparison.rerun_visualization import visualize_stage1; visualize_stage1(output="/tmp/stage1-diagnostic.rrd", manifest=None, skip_missing_methods=True)'
```

This is an opt-in diagnostic only. Its manifest enumerates `missing_methods`
and sets both `complete_registered_method_set` and
`stage1_visualization_acceptance_eligible` to `false`; it cannot serve as the
complete Stage 1 visualization artifact. A skipped run is forbidden from
writing the standard acceptance manifest. The regular CLI remains fail-closed.

## Views and encodings

- **Side-by-side · world motion** shows the source skeleton plus full G1 robots
  for all seven registered operating/reference trajectories and both Sparse
  diagnostic seeds in separate lanes. Root displacement and ground height are
  preserved.
- **World overlay · root tracking** aligns the initial horizontal root position
  and overlays source plus the operating points and external reference. It
  exposes translation, yaw, and trajectory-scale differences hidden by RF-KPE.
- **Root-frame pose overlay** removes each motion's root translation and yaw.
  This is the direct visual companion to root-frame KPE.
- **Sparse seed ambiguity** overlays neutral, A, and B after root alignment.
- **Nine synchronized G1 close-ups** provides one root-frame small multiple for
  each of the six operating points, the external reference, and Sparse A/B.
- Metric tabs show RF-KPE, common-scale root translation, root yaw, ground
  penetration, exact cause-triggered artifact flags, and solve time from the
  frozen evaluator-v3 outputs.

The gray/dark articulated surfaces are the official G1 visual assets and are
the default robot rendering. Colored diagnostic robot bones and joints remain
available in the entity tree but start hidden, so a stick figure cannot obscure
the articulated G1. Labels, root paths, and foot markers remain visible. Green feet
denote frozen source stance without skating; red feet denote source stance with
target foot speed above the frozen `1 cm/s` threshold; gray feet are in swing.
The per-frame annotation spells out `SKATING-L`, `SKATING-R`, `PENETRATION`,
`JOINT-LIMIT`, and `INVALID`; it never labels OmniRetarget or any whole method
as “Artifact.” The Unitree trajectory is always labeled
`Unitree-attributed external reference`: its quantitative evaluator traces stay
visible, but its entity label never acquires an artifact/cause suffix. This
prevents an external comparison corpus from being misrepresented as either a
method or verified ground truth. Source display scale and ground alignment are
frozen once and are not inferred from a method output.
Both the visual-link transforms and semantic FK are calculated from the same
frozen Holosoma URDF and are regression-tested against the canonical MuJoCo
evaluator at neutral and random qpos.

The `.rrd` contains derived motion visualization and remains outside public Git,
just like canonical trajectories and videos. Its schema-v5 manifest records the
complete evidence bindings, exact ordered nine-trajectory set, exact logical
instance count for every view, all 35 unique mesh hashes, 455 mesh entities,
recording size/hash, and a decoded structural audit. Generation runs
`rerun rrd verify`, `rerun rrd stats`, and targeted `rerun rrd print -vv`
checks; a large random file or a hash-only fake cannot satisfy the contract.

The required order is: build publication evaluator tables, generate/verify the
RRD in `vis`, render the PENDING browser artifact, run independent Stage 1
validation, then render and post-audit the final browser artifact. The final
HTML has its own embedded-data digest and `.manifest.json` delivery sidecar.

FULL-LAFAN EXPERIMENTS NOT STARTED — WAITING FOR USER APPROVAL.
