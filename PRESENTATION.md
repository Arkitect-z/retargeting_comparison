# Human-to-G1 Retargeting Pilot

## 1. Decision

**GO WITH CHANGES** — required Stage 1 Pilot execution and validation are complete; Stage 2 needs a runtime-budget change and explicit approval.

## 2. Question

How do sparse constraints, dense constraints, GMR, and OmniRetarget trade fidelity, artifacts, and speed on one frozen human-motion Pilot?

## 3. Frozen source

One source-only selected LAFAN1 window: 600 frames, 19.9998 seconds, selected before any retargeter ran.

## 4. Controlled baselines

Sparse and Dense v3 share robot, scale, rigid root anchor, solver, limits, initialization, posture and temporal costs. Only the task set changes; Sparse additionally exposes three deterministic diagnostic seeds.

## 5. Scale was a confound

![Root scale policies](figures/root_scale_policies.svg)

GMR uses `0.875`; Holosoma uses `0.7471`; evaluator v2 freezes one neutral-geometry scale for every method and reports native tracking separately.

## 6. Public methods

GMR and OmniRetarget run at frozen official commits in isolated subprocess environments with only I/O, provenance, and timing adapters.

## 7. RF-KPE and quality vs speed

![RTF versus RF-KPE](figures/rtf_vs_rf_kpe_all.svg)

RF-KPE is root-frame semantic pose error; it intentionally excludes root translation and heading.

## 8. Targeted vs untracked

![Targeted versus untracked](figures/targeted_vs_untracked.svg)

## 9. Artifacts are cause-specific

![Artifact components](figures/artifact_components.svg)

## 10. Sparse sensitivity

![Sparse seed divergence](figures/sparse_seed_divergence.svg)

## 11. Interaction ablation

![Full versus No-Hard](figures/interaction_full_vs_no_hard.svg)

## 12. Evidence limits

This is one LAFAN operating point and two interaction cases. No dataset-level ranking or controller claim is made.

## 13. Synchronized G1 evidence

The Rerun recording provides full articulated G1 meshes in side-by-side world, world overlay, root-frame pose, and Sparse seed views, together with foot-state and per-frame metrics for every canonical output.

## 14. Stage 2 budget

The 1.5× serial projection is 68.04 h and 6.43 GB. Discuss reduced-LAFAN or optimized/parallel execution before approval.

FULL-LAFAN EXPERIMENTS NOT STARTED — WAITING FOR USER APPROVAL.
