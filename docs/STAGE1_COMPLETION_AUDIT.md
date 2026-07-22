# Stage 1 completion audit

## Decision

**NO-GO — revised Stage 1 work in progress.** The legacy four-core Pilot is
complete, but the accepted scope now also requires ProtoMotions v2.3/Mink,
ProtoMotions v3/modified-PyRoki, an official preprocessing-policy audit,
hashed pre-solver targets, controlled scale-policy transplantation, and
within-method root/local scale sensitivity.

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
| Articulated G1 Rerun visualization | synchronized 600-frame legacy recording | Complete |
| GMR/Holosoma/Proto v2/v3/PHC AMASS policy audit | `research/OFFICIAL_SCALE_AND_PREPROCESSING_AUDIT.md` | Complete as code audit |
| ProtoMotions v2.3 dependency and input gates | stable tag found; floating dependencies and 165-D contract identified | Pending implementation |
| ProtoMotions v2.3 canonical Pilot | no accepted 600-frame output yet | Missing |
| ProtoMotions v3 canonical Pilot | same-source keypoints exist; no accepted 600-frame output/environment yet | Missing |
| PHC treatment | official fitting asset identified as 37-motor/noncanonical | Complete as exclusion evidence |
| Hashed pre-solver targets for every public method | schema designed, packages not captured | Missing |
| Controlled scale-policy transplant | registered in `configs/scale_policy_sensitivity.yaml` | Missing results |
| Root/local ±5% within-method response | five variants registered | Missing results |
| Neutral-SMPL-X actor-shape target probe | short/zero/tall design registered | Missing results |
| Fixed-contact/constraint-flip evidence | protocol registered | Missing results |
| Revised interactive visualization/report | must include v2/v3 and active scale policy | Missing |
| Revised Stage 1 validator | must fail on every missing mandatory item | Pending implementation |
| Revised Stage 2 projection | legacy projection excludes new methods/variants | Invalid until rerun |

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
  600-frame CLI setting and tighter joint limits must be frozen explicitly.
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
