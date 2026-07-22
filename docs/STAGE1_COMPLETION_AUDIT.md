# Stage 1 completion audit

## Decision

The frozen Pilot execution is complete. The decision remains **GO WITH
CHANGES**, solely because the projected serial Full-LAFAN wall time is `67.95
h`, above the frozen `48 h` Stage 2 gate. Projected storage is `3.47 GB`, below
the `200 GB` gate. No Full-LAFAN job was started.

## Required experimental evidence

| Plan requirement | Evidence | Status |
|---|---|---|
| Deterministic source-only Pilot selection | `pilot_sequence.yaml`, selection CSV, selector regression | Complete |
| Canonical source and native adapters | canonical NPZ, GMR/Holosoma adapter agreement CSV | Complete |
| SMPL chumpy-free and neutral SMPL-X validation | finite zero-pose forwards and SHA-256 manifest | Complete |
| Sparse neutral/A/B | 600/600 canonical frames for all three seeds | Complete |
| Dense | 600/600 canonical frames | Complete |
| Official GMR | 600/600 canonical frames at frozen commit | Complete |
| Official OmniRetarget/Holosoma | 600/600 canonical frames at frozen commit | Complete |
| Four-method smoke tests | two-frame immutable smoke outputs and hashes | Complete |
| Unified evaluator | targeted, untracked, temporal, artifact, completion metrics | Complete |
| Timing protocol | cold process, one warm-up, three measured warm runs | Complete |
| Interaction case study | box/climb × Full/No-Hard with surface distances | Complete |
| Synchronized visual inspection | 600-frame Rerun recording for source and all six G1 outputs | Complete |
| Reports and presentation | eight required Markdown reports; 13 presentation sections | Complete |
| Stage 1 validator | all recorded acceptance checks true | Complete |

## Automated regression coverage

The `capture` suite currently passes 30 tests. It covers BVH parsing and the
frozen selector order, SMPL/SMPL-X finite forwards, quaternion conventions,
canonical schemas/completion, Sparse/Dense controls, deterministic seeds,
canonical MuJoCo versus visualization-URDF FK at random qpos, synthetic
RF-KPE/temporal/skating/penetration/joint-limit artifacts, source-side mapping,
hard-constraint gates, surface-distance metrics, run/timing manifests,
interactive-report determinism, and delivery contracts. Rerun 0.34.1 also
verifies the complete `.rrd` independently.

## Deliberate non-requirements and remaining scientific limits

- ProtoMotions v3 and PHC were conditional candidates. The core-first rule
  allowed them to remain unstarted once the four required operating points
  answered the Pilot question and OmniRetarget runtime became the Stage 2
  bottleneck.
- Sparse+Orientation and independent-frame Sparse were optional diagnostics
  and were not promoted into the primary comparison.
- Self-collision remains diagnostic-only because no common validated pair set
  was frozen.
- This remains one preselected 20-second LAFAN sequence plus two interaction
  cases. It is complete as a Stage 1 Pilot, not as a dataset-level ranking.
- Controller rollout, dynamic stability, torque, and hardware execution are
  outside this kinematic-retargeting Pilot.

## Stage 2 preparation

The full dataset remains gated. Before approval, either reduce OmniRetarget
wall time through measured sequence-level parallelism/optimization, or define
and explicitly rename a deterministic reduced-LAFAN design. Any new design must
retain the frozen source rules, evaluator, method commits, thresholds, and
1.5× projection safety factor.

FULL-LAFAN EXPERIMENTS NOT STARTED — WAITING FOR USER APPROVAL.
