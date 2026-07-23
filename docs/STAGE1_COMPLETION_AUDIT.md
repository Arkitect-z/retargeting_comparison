# Stage 1 completion audit

## Decision

**NO-GO — corrected formal campaign in progress.** The legacy four-core Pilot
and most expanded harness components exist, but the earlier ProtoMotions v3
campaign incorrectly overrode the documented fixed 450-frame trajectory
contract with 600 frames. That evidence is now quarantined. Acceptance requires
a new six-method formal campaign and regeneration of every dependent metric,
scale result, visualization, report, and validation manifest.

This is a completeness decision, not evidence that a method failed. No
Full-LAFAN or AMASS dataset run was started.

## Evidence status

| Revised requirement | Evidence | Status |
|---|---|---|
| Deterministic source-only Pilot selection | `pilot_sequence.yaml`, source-selection CSV, selector regression | Complete |
| Licensed body-model validation | chumpy-free SMPL and neutral SMPL-X finite forwards and hashes | Complete |
| Controlled Sparse neutral/A/B | 600/600 canonical frames | Complete |
| Controlled Dense | 600/600 canonical frames | Complete |
| Official GMR | 600/600 canonical frames at frozen commit | Complete |
| Official OmniRetarget/Holosoma | 600/600 canonical frames at frozen commit | Complete |
| Legacy evaluator/timing/artifact evidence | disaggregated metrics and raw timing | Complete |
| Interaction case study | box/climb × Full/No-Hard | Complete |
| Articulated G1 Rerun visualization | legacy recording exists; corrected acceptance recording must use shared frames 0–449 | Pending rebuild |
| GMR/Holosoma/Proto v2/v3/PHC AMASS policy audit | `research/OFFICIAL_SCALE_AND_PREPROCESSING_AUDIT.md` | Complete as code audit |
| ProtoMotions v2.3 dependency and input gates | frozen environment, 165-D contract, smoke path, and native-target capture implemented | Complete |
| ProtoMotions v2.3 canonical Pilot | prior output exists; new hash-bound formal campaign materialization pending | Pending formal rerun |
| ProtoMotions v3 canonical Pilot | 450-frame CUDA diagnostic succeeds; formal cold/warm-up/3×warm campaign pending | Pending formal rerun |
| PHC treatment | official fitting asset identified as 37-motor/noncanonical | Complete as exclusion evidence |
| Hashed pre-solver targets for every public method | exact captures and public aggregate exist | Complete; rebind after formal outputs |
| Controlled scale-policy transplant | registered; old dependent tables invalidated by v3 length correction | Pending corrected rebuild |
| Root/local ±5% within-method response | registered; v3 arm must be regenerated at 450 frames | Pending corrected rebuild |
| Neutral-SMPL-X actor-shape policy formula probe | short/zero/tall, official-source-hash-bound reconstruction; not runtime-constructor or ranking evidence | Generated; exact LAFAN targets require separate schema-2 native capture |
| Fixed-contact/constraint-flip evidence | protocol registered | Missing results |
| Revised interactive visualization/report | implementation exists; acceptance artifacts await corrected evidence | Pending rebuild |
| Revised Stage 1 validator | fail-closed implementation and regression suite pass | Complete; current decision remains NO-GO |
| Revised Stage 2 projection | design exists; timing-dependent projection must be regenerated | Pending corrected timing |

## What the legacy scale correction did and did not prove

Evaluator v2 correctly separated common-scale root error, native-policy root
error, and scale-invariant path-shape error. It also explained why RF-KPE can
cluster after removing root translation and heading. This repaired a metric
confound in the existing outputs.

It did not rerun any retargeter under a changed pre-solver scale policy.
Therefore it is not the newly required scale-sensitivity experiment. The
revised experiment changes root displacement and root-relative target geometry
before solving and retains native/controlled results as separate evidence.

## Method-set correction

- ProtoMotions v2.3 is an official G1-29 sequential Mink retargeter. It uses
  PHC-derived preprocessing/FK infrastructure but is not a PHC algorithm result.
- ProtoMotions v3 is the required whole-trajectory modified-PyRoki point. Its
  official contract trims/pads to 450 frames; the 600-frame source therefore
  yields 450/450 native completion and 450/600 full-source coverage. The shared
  comparison window is frames 0–449, and no missing frames may be fabricated.
- PHC remains lineage and AMASS-policy evidence. Its documented public G1
  fitting asset has 37 motors and cannot enter the canonical G1-29 main plot
  unchanged.
- The v2/v3 pair is not a pure backend ablation because preprocessing,
  representation, scale, contacts, grounding, costs, temporal scope, and limits
  all change.

## Stage 2 preparation

The previous 68.04-hour/6.43-GB projection covered only the legacy method set.
After revised Stage 1, recompute runtime and storage with the frozen 1.5× safety
factor. Full-LAFAN still requires a separate user decision and must satisfy the
48-hour/200-GB gate or use an explicitly discussed deterministic simplification.

FULL-LAFAN EXPERIMENTS NOT STARTED — WAITING FOR USER APPROVAL.
