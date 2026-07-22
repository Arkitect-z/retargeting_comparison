# Stage 2 design: budgeted LAFAN evaluation

## Status

The Stage 2 execution and analysis harnesses are implemented and tested. The
user has explicitly authorized Stage 2; that
authorization is recorded in `configs/stage2.yaml`. Execution is nevertheless
conditional on a new Stage 1 validation manifest whose decision is `GO`, all
its checks being true, **and an independent byte-level audit of all six formal
600-frame Stage 1 outputs**. The latter verifies exact method identity, the
canonical Pilot source fps,
frames 0–599, finite qpos/timing, every `valid=True`, completion status, and
SHA-256. A missing integration cannot disappear through a readiness rule; only
the later pre-registered budget policy can exclude v3.

The limits are 48 hours wall time and 200 GB retained storage. Runtime guards
stop new work and terminate the entire subprocess group at 98% of either limit,
leaving a margin for manifest finalization and filesystem accounting.

## Scientific strata

The run matrix deliberately does not label every output as though it were the
same intervention.

| Stratum | Operating points | Meaning |
|---|---|---|
| `controlled_common_per_sequence_scale` | Sparse-neutral, Dense | Same solver family and one method-independent scalar least-squares scale over the registered frame-0, root-relative, heading-aligned shared semantic landmarks. Head/toe span is diagnostic only. The neutral G1 pelvis supplies the per-sequence root anchor. |
| `native_public_pipeline` | GMR, OmniRetarget / Holosoma | Each accepted public LAFAN pipeline retains its published preprocessing, scale, grounding, contacts, limits, and temporal formulation. |
| `benchmark_public_retargeter_port` | ProtoMotions v2.3, ProtoMotions v3 | Public G1 retargeter policies with benchmark-supplied LAFAN/keypoint adapters. Neither upstream release supplies the exact LAFAN path used here, so neither is labelled a native public LAFAN result. |
| `external_reference` | Unitree-attributed corpus | Precomputed trajectory evidence on the filename intersection only. It has no runtime provenance, is not verified ground truth, and is excluded from runtime rankings. |

Native and controlled results may be compared descriptively, but the plan
forbids a cross-stratum method ranking that silently attributes scale-policy
effects to a solver. No output qpos is rescaled after solving.

## Dataset freeze and reference intersection

`discover_lafan_sequences` sorts all BVHs by normalized relative path and
records the relative path, SHA-256, byte size, frame count, fps, and duration.
The expected Full-LAFAN contract is 77 files and 496,672 frames. A plan fails
closed if that inventory changes.

Reference CSVs are revision-pinned separately. Matching uses normalized BVH
and CSV basenames. This establishes basename and integer-frame-index pairing
only; it does not establish byte-identical human inputs or exact timestamp
identity. Planning, analysis, and launch all fail closed unless there are
exactly 40 files, 40 filename matches, all 36 columns are finite numeric values,
every CSV row count equals its BVH frame count, and actors subject1–5 are all
represented. Every CSV receives its own SHA-256 and byte count; interpolation
is forbidden.

At the pinned revision, the downloaded G1 corpus contains 40 CSV files and the
inventories intersect on 40 of 77 LAFAN filenames. The reference comparison is
therefore a substantial but incomplete subset; it is not dataset-level ground
truth or a Full-LAFAN upper bound.

## Per-sequence preparation

Before method work, each selected BVH is converted once to `CanonicalHuman`:

- right-handed Z-up positions in metres;
- local/world quaternions in `wxyz` order;
- original timestamps and source-frame indices;
- deterministic foot-contact labels;
- original and canonical file hashes.

The controlled common scale is recomputed **once for every source sequence**
with the registered root-relative,
heading-aligned shared-semantic-landmark scalar least-squares estimator and the
same hashed Holosoma G1 evaluator model. Local/body gain, root-displacement
gain, and neutral-pelvis frame-0 anchor are recorded separately; head/toe span
is diagnostic only. This fixes the Stage 1 limitation in which the Pilot
actor's calibration was embedded in the controlled configuration. Preparation attempts live under numbered
directories. An interruption never overwrites a partial attempt, and the
sequence manifest becomes authoritative only after canonicalization,
calibration, and hashing all succeed.

The actor label is never the calibration unit. It is retained only as a
dependence cluster for uncertainty analysis. Two motions from the same actor
receive two independently frozen sequence calibrations because their frame-0
poses and heading-aligned landmark geometry may differ.

## Production job contract

Every job is one method × one selected source sequence, except the external
reference, which is created only on the filename intersection. A fresh
subprocess receives exactly one production pass:

- no Stage 2 cold/warm timing repetitions;
- one thread for OpenMP, MKL, OpenBLAS, and NumExpr;
- a method-specific frozen conda environment;
- no visualization;
- attempt-specific output, logs, and native work directory.

Stage 1's cold + warmup + three measured-run protocol remains the source of
runtime estimates. Stage 2 is a production-quality pass, not another timing
experiment.

The job manifest records the immutable plan hash, exact scheduled-job and
method-policy contract hashes, repository/hardware contracts, configuration
hash, payload-hashed sequence manifest, command and command hash, Python/conda
and thread-environment hashes, timestamps, wall time, exit code, log hashes,
expected output path, output file/content hashes, and completion ratio.
A successful output is reusable only while both its plan identity and bytes
match. Failed and interrupted attempts are preserved; invoking execution again
adds a new numbered attempt and schedules only jobs not already verified.

Success requires exactly 100% of source frames, unchanged source fps,
contiguous `source_frame_idx=0..T-1`, every valid flag, finite qpos and solve
times, unit `wxyz` root quaternions, and exact method, sequence, and stratum
metadata. The former 95% execution threshold has been removed. Only
`external_reference` jobs receive a reference-relative-path argument.

## Deterministic parallel schedule

Methods are sequential phases, preventing cross-method CPU or accelerator
contention from changing the comparison. Within each phase, jobs are sorted by
decreasing estimated runtime and then sequence ID. Each job is assigned to the
lowest-load worker, breaking ties by worker index. This fixed longest-
processing-time schedule is serialized into the plan.

OmniRetarget / Holosoma uses six single-thread workers on the recorded
16-physical-core host.
The remaining ten cores cover the OS, I/O, process startup, and later
evaluation. Other CPU methods also use at most six workers. ProtoMotions v3 is
conservatively single-worker unless a later measured Stage 1 configuration
changes the checked-in design.

The budget projection includes source preparation, method phases, evaluator
work, and a 10% failed-runtime allowance. Retained storage adds analysis
artifacts and a 5% failed-attempt allowance before the global 1.5× safety
factor. These costs are not assumed to fit for free outside method execution.
Preparation and analysis are currently sequential and are projected as one
worker; no unimplemented six-way evaluator speedup is claimed. The evaluator
RTF allowance includes the direct reference FK pass.

The historical pre-six-method read-only preview was:

| Quantity | Current preview |
|---|---:|
| Sequences / frames | 77 / 496,672 |
| Raw projected wall time | 7.65 h |
| 1.5× safe wall time | 11.48 h |
| 1.5× safe retained storage | 13.03 GB |
| Holosoma phase (raw, six workers) | 7.47 h |
| Reference jobs | 40 |

This is not the final Stage 2 plan and is non-launchable. Once both
ProtoMotions Pilot outputs are accepted, their timing and bytes/frame are read
from those artifacts. No guessed performance value is permitted.

## Method simplification before dataset reduction

The planner first projects every Stage-1-accepted method on all 77 sequences.
If that all-method design exceeds either hard budget, it applies the checked-in
`budget_simplification.method_exclusion_order` before considering a smaller
dataset. The current order contains only ProtoMotions v3 because its official
formulation jointly optimizes a whole trajectory and the measured Pilot cost
is expected to dominate the Full-LAFAN projection. The method is removed only
when necessary, and planning stops removing methods as soon as all 77 sources
fit.

This is a cost decision, not a quality decision. The plan records the safe
wall/storage projections immediately before and after every exclusion,
asserts that no result metric participated, and retains the complete
ProtoMotions v3 Stage-1 evidence. This preserves dataset coverage for the
feasible methods instead of letting one high-cost candidate force every method
onto an unnecessarily small subset.

## Full versus reduced LAFAN

The planner always evaluates Full LAFAN first. If both safe projections fit,
the immutable design is named:

```text
full-lafan1-<inventory_sha256_prefix>
```

Only if Full LAFAN still exceeds a hard limit after the registered method
simplification does the planner consider a reduced design. It applies a frozen
SHA-256 permutation based solely on selection seed, source path, and source
hash; one reference-intersection sequence is placed first when available. It
chooses the largest prefix that fits and requires at least eight sequences.
The name is:

```text
reduced-lafan1-nNN-<selected_inventory_sha256_prefix>
```

The fallback manifest includes the failed full projection, seed, exact
selected IDs, and an assertion that no result metric participated in selection.
If even the minimum reduced design cannot fit, planning fails instead of
weakening the budget.

## Launch API and gates

Planning is safe and non-executing:

```python
from retargeting_comparison.stage2 import build_stage2_plan

plan = build_stage2_plan(".")
```

Execution defaults to preview. A real launch additionally requires the exact
plan digest to be repeated by the caller:

```python
from retargeting_comparison.stage2 import execute_stage2

preview = execute_stage2(
    ".",
    "runs/stage2/<design-id>/plan.json",
    expected_plan_sha256="<exact digest from the plan>",
    launch=False,
)

# Set launch=True only after reviewing preview and the Stage-1 GO evidence.
```

Immediately before launch, the harness re-hashes the plan, configuration,
Stage 1 validation and six formal trajectories, all 77 BVHs, all 40 reference
CSVs, the clean committed main worktree, four pinned upstream worktrees, the
exact Holosoma patch, conda environment histories, required G1 assets, and the
launch host's CPU/GPU identity. The plan records hostname, platform, logical
and physical core counts, configured core requirement, and GPU model/UUID/
memory rows; launch requires an exact match. Any mutation or host change closes
the gate and requires a new plan.

The generated `stage2_results/` publication tree is the sole main-worktree
identity exclusion, so a partial read-only analysis does not prevent resuming
jobs. Its own plan and artifact hashes are validated separately; source,
configuration, test, and other untracked paths remain launch blockers.

One process-wide guard owns the cumulative deadline, retained-byte accounting,
and output reservations for all worker lanes. A lane reserves its projected
artifact bytes before spawn, so concurrent reservations cannot overshoot the
limit. Wall/storage cancellation is shared. SIGINT, SIGTERM, Python exceptions,
and launcher finalization reap complete subprocess groups; failed attempts are
finalized instead of remaining permanently `running`.

The launcher also withholds the safety-adjusted evaluator wall time and tracked
analysis bytes from the production guards. Analysis is itself a resumable,
cumulatively budgeted attempt. Failed, interrupted, and repeated analysis wall
time is charged from immutable attempt ledgers; retained bytes include failed
attempts and a one-time raw-BVH/reference-CSV baseline even though those files
live outside the run tree. Live guards run between jobs as well as between
analysis phases. Each attempt builds in a private staging tree, publishes files
atomically, and replaces `analysis_manifest.json` last as the publication
commit point. Every validation invocation is also a payload-hashed attempt and
is charged before its atomic `STAGE2_VALIDATION.json` publication.
`validate-stage2` rejects an otherwise complete dataset if the independently
recomputed total exceeds either 48 h or 200 GB.

## Analysis and tracked delivery

Primary uncertainty uses the LAFAN actor as the bootstrap cluster: all
sequences from a sampled subject move together. Sequence-level means still
avoid frame weighting, and a leave-one-subject-out table exposes actor
influence. Within-stratum paired and external-reference intervals use the same
cluster rule.

External-reference analysis has two separate layers: differences between each
trajectory's human-source evaluator metrics, and direct method↔reference G1
disagreement on the normalized-basename/frame-index overlap. The latter is not
described as the same source or exact timeline. It reports absolute root
translation, frame-0 root-anchor offset, frame-0-subtracted root-displacement
mean/p95, root yaw/orientation, 29 joint angles/velocities, and MuJoCo FK
landmarks in world and root frames. Neither layer treats the external
trajectory as ground truth.

The reference URDF is not assumed to equal the evaluator robot merely because
both are named G1. Planning and validation independently bind the reference
URDF, canonical URDF, canonical MuJoCo scene, and evaluator manifest; compare
joint order/tree/origin/axis/limits; and reproduce deterministic neutral plus
random-pose URDF↔MuJoCo FK tolerances. This establishes a kinematic comparison
contract only, not visual, collision, inertial, actuator, or dynamics
equivalence.

## Registered calibration, sensitivity, and conclusion tables

The publication must include all of the following CSV/Parquet pairs. None may
be reconstructed from report prose alone.

| Table | Unit | Required contents | Claim boundary |
|---|---|---|---|
| `calibration_per_sequence` | one row per selected sequence | registered LS scale, local/root gains, calibration residual, head/toe diagnostic, root-anchor norm, sequence-manifest hash | proves calibration was sequence-specific and method-independent |
| `calibration_scale_variability` | all sequences and actor-labelled groups | N, mean, SD, median, min/max, coefficient of variation | describes calibration variability; actor groups are summaries, not calibration units |
| `scale_sensitivity_diagnostics` | method × metric | scale range, metric mean, OLS slope, Pearson/Spearman association, common-vs-scale-invariant root contrast | exploratory and observational; motion, pose, actor, and scale co-vary, so this is not a causal scale intervention |
| `conclusion_ledger` | one row per evidence stratum | methods, sequence coverage, allowed conclusion, forbidden conclusion | prevents cross-stratum order from becoming a solver-effect or universal-winner claim |

The Stage 1 fixed-scale intervention matrix remains the causal sensitivity
experiment. Stage 2 adds distributional evidence over independently frozen
sequence scales; it does not silently rerun or alter any public method's scale
policy.

Reports, CSV/Parquet, chart-source CSV, SVG/PDF/PNG, the immutable plan
snapshot, analysis-attempt ledger, analysis manifest, and
`STAGE2_VALIDATION.json` are written under the
non-ignored `stage2_results/<design-id>/` tree. `rtcmp validate-stage2 --plan
...` checks the tracked evidence hashes and returns GO only for a complete
selected design.

## Tests

`tests/test_stage2.py` covers:

- deterministic six-worker LPT assignment;
- reference jobs restricted to the filename intersection;
- Full-LAFAN preference whenever it fits;
- pre-registered high-cost method exclusion before any dataset reduction;
- deterministic, named reduced fallback only after a full-budget failure;
- one-pass Stage 2 timing semantics;
- output/manifest hash validation for resume;
- BVH inventory and CSV intersection discovery;
- explicit authorization and Stage 1 GO gates;
- stable plan reuse and exact digest acknowledgement;
- strict six-method Stage 1 formal-output evidence;
- finite/exact 40-file reference inventory and reference-path isolation;
- a fake production worker that passes the exact output contract;
- shared wall and concurrent-storage reservation cancellation.

`tests/test_stage2_analysis.py` covers:

- actor-cluster bootstrap, leave-one-subject-out influence, per-sequence
  calibration/scale diagnostics, and direct G1 reference disagreement on the
  basename/frame-index overlap;
- failed/rerun analysis accounting, raw-input storage charging, atomic
  publication, manifest/plan/job forgery, and reference-asset tamper tests.

## Remaining limitations before launch

1. Stage 1 must first accept complete ProtoMotions v2.3 and v3 outputs. Until
   then the strict formal-output gate refuses to create a launchable plan; it
   does not silently record them as N/A.
2. Six-worker Holosoma scaling is a deterministic resource policy, not yet a
   measured concurrency-efficiency claim. The 1.5× factor covers moderate
   contention; a Stage 1 concurrency probe should replace the estimate if it
   reveals worse scaling.
3. ProtoMotions v3 optimizes whole trajectories, so very long source files may
   have nonlinear memory/runtime behavior relative to the 600-frame Pilot. Its
   accepted estimate must therefore remain conservative. If its Full-LAFAN
   projection closes the hard gate, the registered policy excludes v3 from
   Stage 2 before any source reduction and records that exclusion explicitly.
4. The frame-0 shared-landmark least-squares scale is sequence-specific and still
   mildly pose-sensitive. It is frozen before any method result and is the
   registered controlled arm; a rest-skeleton or multi-frame robust estimator
   would be a distinct experiment, not a silent Stage 2 adjustment. The older
   Head-to-Toe ratio remains diagnostic only.
5. Metric aggregation, direct reference comparison, plots, reports, and final
   validation consume only verified succeeded manifests and remain a separate,
   restartable, non-launching phase.
