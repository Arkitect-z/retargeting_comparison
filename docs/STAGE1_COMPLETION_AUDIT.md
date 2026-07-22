# Stage 1 completion audit

## Decision

The frozen Pilot execution is complete. The decision is **GO WITH CHANGES**:
the measured 1.5× serial Full-LAFAN projection is `68.04 h`, above the frozen
`48 h` Stage 2 gate. Projected retained storage is `6.43 GB`, below the `200 GB`
gate. No Full-LAFAN job was started.

## Required experimental evidence

| Plan requirement | Evidence | Status |
|---|---|---|
| Deterministic source-only Pilot selection | `pilot_sequence.yaml`, source-selection CSV, selector regression | Complete |
| One canonical source and audited native adapters | GMR/Holosoma adapter errors and hashed ignored native inputs | Complete |
| SMPL chumpy-free and neutral SMPL-X validation | finite zero-pose forwards and SHA-256 manifest | Complete |
| Method-independent scale and heading | evaluator-v2 manifest, neutral-G1 geometry scale, geometry heading | Complete |
| Controlled Sparse neutral/A/B v3 | 600/600 canonical frames for all three seeds | Complete |
| Controlled Dense v3 | 600/600 canonical frames | Complete |
| Official GMR | 600/600 canonical frames at frozen commit | Complete |
| Official OmniRetarget/Holosoma | 600/600 canonical frames at frozen commit | Complete |
| Four-method smoke tests | two-frame canonical outputs; controlled smoke uses v3 | Complete |
| Unified evaluator | targeted, untracked, root-scale, temporal, artifact and completion metrics | Complete |
| Timing protocol | one cold process, one warm-up, three measured warm runs | Complete |
| Interaction case study | box/climb × Full/No-Hard with geometry-surface distances | Complete |
| Conditional gates | ProtoMotions v3 and PHC have explicit bounded N/A evidence | Complete |
| Synchronized visual inspection | 600-frame Rerun recording with all 35 articulated G1 meshes | Complete |
| Reports and presentation | eight required English Markdown reports; 14 presentation sections | Complete |
| Stage 1 validator | all acceptance checks recorded in `stage1_validation.json` | Complete |

## Scale correction audit

The same source file and G1 model did not guarantee the same source-to-robot
scale. Frozen GMR declares `0.875`; Holosoma declares `1.27/1.7 = 0.7470588`;
the earlier evaluator also inferred height from each method's first output pose.
Evaluator v2 instead freezes one method-independent scale (`0.742037044`) from
neutral G1 and the selected source. Controlled v3 uses that scale plus one
geometry-derived rigid root anchor so G1 is not placed below the ground merely
because its pelvis-to-foot proportion differs from the human skeleton.

Primary root evidence now reports common-scale error, native-policy error and
scale-invariant path-shape error separately. RF-KPE uses the common scale and
removes root translation and heading by definition.

## Automated regression coverage

The `capture` suite passes 35 tests. It covers BVH parsing and deterministic
selection, body-model finite forwards, rotation conventions, schemas and
completion, controlled configuration equality, deterministic seeds and root
anchoring, canonical MuJoCo versus visualization-URDF FK, synthetic fidelity,
temporal and artifact metrics, source adapters, Holosoma constraint flags,
surface distances, timing/run manifests, interactive-report determinism and
delivery contracts. Rerun 0.34.1 independently verifies the complete `.rrd`.

## Deliberate non-requirements and limits

- ProtoMotions v3 has a same-source 600-frame keypoint package, but no frozen
  dedicated JAX/PyRoki environment or complete canonical output. PHC also lacks
  a lossless LAFAN-to-SMPL parameter adapter. Both remain N/A under the bounded
  integration gate; neither is assigned a quality score.
- Sparse+Orientation and independent-frame Sparse were optional diagnostics
  and were not promoted into the primary comparison.
- Self-collision remains diagnostic-only because no common validated pair set
  was frozen.
- This is one preselected 20-second LAFAN sequence plus two interaction cases.
  It is complete as a Stage 1 Pilot, not as a dataset-level ranking.
- Controller rollout, dynamic stability, torque and hardware execution are
  outside this kinematic-retargeting Pilot.

## Stage 2 preparation

Before approval, either measure sequence-level parallel execution/optimization
for OmniRetarget, or explicitly define and rename a deterministic reduced-LAFAN
design. Any Stage 2 design must retain the frozen source rules, evaluator,
method commits, thresholds and 1.5× projection safety factor.

FULL-LAFAN EXPERIMENTS NOT STARTED — WAITING FOR USER APPROVAL.
