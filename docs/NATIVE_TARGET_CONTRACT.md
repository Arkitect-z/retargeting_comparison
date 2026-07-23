# Native Pre-Solver Target Contract

## Purpose

This contract records what each frozen Stage 1 method actually presents to its
optimizer before any solve. It prevents a controlled benchmark projection—for
example, the shared Dense target set—from being described as a public method's
native objective.

The evidence is frozen to `dance1_subject1_f000000_000600` (600 frames). The
capture workers execute loaders, preprocessing, target setters, and—in the
Holosoma case—interaction-mesh construction. They do **not** invoke an IK or
retargeting solver. Consequently, these artifacts characterize the objectives
and preprocessing, not the quality of a solved robot trajectory.

## Evidence classes

Two independent fields must be reported together:

- `capture_class=exact_solver_input` means the primary tensor is numerically the
  tensor consumed by the frozen solver boundary. It does not mean that every
  tensor was intercepted during a solve.
- `acquisition_mode` states how the value was obtained:
  `native_runtime_target_setter`, `official_loader_return_value`,
  `official_pre_solver_reconstruction`, or
  `formal_adapter_formula_reconstruction`.
- `normalized_projection` is a benchmark-defined common semantic projection.
  It is useful for controlled comparisons but is not native-method evidence.
  Every row in the present native manifest has
  `normalized_projection=false`.
- `geometry_class` separately classifies the position tensor used for the small
  geometry table. For Holosoma this is
  `exact_pre_solver_position_source`, because the actual solver tensor is a
  240-vertex Laplacian rather than 15 Cartesian joint positions.

An offline reconstruction may therefore be an `exact_solver_input` while still
being explicitly labeled as a reconstruction. “Exact” describes numerical
identity at the frozen boundary; `acquisition_mode` describes provenance.

## Frozen method contracts

| Method | Acquisition | Primary exact tensor | Position evidence | Native scale and grounding |
|---|---|---|---|---|
| GMR | Native `update_targets` execution; no `retarget` call | `position_targets`, `[600,28,3]`; 14 tasks in each of two task tables | The same 28 FrameTask translations | Root/legs/torso `0.875`; arms `0.7291666667`; ground offset disabled |
| OmniRetarget / Holosoma | Exact reconstruction with the frozen official loader, preprocessing, mesh, adjacency, and Laplacian functions | `solver_target_laplacian`, `[600,240,3]` | 22 preprocessed joints and the exact 15 mapped position sources | Spine correction, sequence-level toe grounding, then uniform `1.27/1.7 = 0.7470588235` |
| ProtoMotions v2.3 adapter | Exact formula execution through the formal adapter target helper | `position_targets`, `[600,14,3]` | The same 14 FrameTask translations | World-axis multiplier `[0.75,1.0,0.8]` |
| ProtoMotions v3 | Exact return value of official `load_motion_data` | `target_keypoints`, `[600,18,3]` | The same 18 keypoints | Root/lower `[0.9,0.9,0.85]`; upper root-relative `[0.9,0.9,0.8]` |

All tensors use metres and right-handed `xyz` coordinates with `+z` up after
native preprocessing. Each artifact also records source, configuration,
implementation, robot-asset, and upstream-commit provenance.

### GMR

The worker loads the Pilot with official `load_bvh_file`, constructs frozen
`GeneralMotionRetargeting`, and calls only `update_targets(frame,
offset_to_ground=False)`. It reads the translation and quaternion produced for
every active entry in both official task tables. No optimizer method is called.

For human body point `p`, root `r`, body scale `s_b`, and root scale `s_r`, the
captured translation follows

```text
p_target = s_r r + s_b (p - r)
```

before the configured local pose offset; the frozen LAFAN/G1 configuration has
zero ground offset in this capture. The official loader reports a 1.75 m actor,
the configuration assumes 1.8 m, and the resulting exact scale table is
preserved in `capture.json`. The two tables have equal position tensors but
different task costs, so they are not collapsed in the artifact.

### OmniRetarget / Holosoma

Holosoma does not hand a list of Cartesian joint targets directly to its
optimizer. Its exact target is the Laplacian coordinate tensor of an interaction
mesh. The frozen reconstruction executes this native chain:

```text
canonical adapter
  -> official LAFAN axis round-trip and explicit 22-joint order
  -> Spine1.z -= 0.06 m
  -> subtract the global minimum toe height
  -> multiply all coordinates by 1.27 / 1.7
  -> map 15 human joints to robot links
  -> append the frozen 15 x 15 ground grid
  -> Delaunay interaction mesh and adjacency
  -> uniform-weight Laplacian coordinates
```

The result is 240 vertices per frame: 15 mapped human points and 225 ground
points. The artifact stores the per-frame topology and adjacency hashes, the
exact 22-joint input order, the 15-joint mapping, the mapped robot links, and
foot-sticking flags. For this Pilot, the pre-scale grounding offset is
`-0.014545225654956556 m`.

The geometry CSV intentionally summarizes `mapped_position_targets`, not the
Laplacian. Those 15 points are exact inputs to mesh construction, but they are
classified as `exact_pre_solver_position_source`; the primary manifest still
identifies `solver_target_laplacian` as the solver input.

### ProtoMotions v2.3 adapter

The v2.3 evidence is exact for this repository's formal canonical-LAFAN adapter:
the worker instantiates that adapter, calls its target-position helper, and
captures the 14 translations that the adapter supplies to Mink FrameTasks. It
also stores the corresponding orientation targets.

```text
p_target = p_canonical * [0.75, 1.0, 0.8]
```

This is a port of the frozen official v2.3 G1 Mink task, scale, and solver
semantics. It is **not** evidence that upstream ProtoMotions v2.3 publishes an
official LAFAN entry point. Reports must retain that qualification.

### ProtoMotions v3

The worker imports the frozen official keypoint retargeting script and calls only
`load_motion_data`. The exact returned `target_keypoints`—the value subsequently
passed to `solve_retargeting` in the official script—is captured before the
solver is constructed or compiled. The tensor contains 15 semantic points and
three auxiliary points.

With root `r` and source point `p`, the frozen transformation is

```text
r'             = r * [0.9, 0.9, 0.85]
p'_lower       = r' + (p - r) * [0.9, 0.9, 0.85]   # indices 1:9
p'_upper       = r' + (p - r) * [0.9, 0.9, 0.8]    # indices 9:18
```

The official five-frame cross-faded ankle/toe contact outputs are preserved as
supplementary tensors.

## Why equal source and robot assets do not imply equal target scale

The target robot geometry is fixed, but the methods transform the human motion
before optimizing that robot. Those transformations change both the requested
root path and the requested body geometry:

- GMR uses body-group scales around a separately scaled root.
- Holosoma changes Spine1, grounds the whole sequence, applies a uniform scale,
  and optimizes interaction-mesh Laplacians.
- v2.3 scales absolute world axes, including root translation.
- v3 separately scales the root, lower-body offsets, and upper-body offsets.

The same 600 source frames therefore yield different pre-solver targets. This is
not a change in G1 body size. It is a native preprocessing/objective difference
that propagates into the solved root trajectory and pose.

The exact position evidence already shows the effect. The horizontal root-proxy
spans are shown below; Holosoma's proxy is `Spine1`, while the others use pelvis
or Hips, so this table diagnoses preprocessing and must not be interpreted as a
like-for-like accuracy ranking.

| Method | Root proxy | x span (m) | y span (m) | Path length (m) |
|---|---:|---:|---:|---:|
| GMR | Hips | 1.6201 | 3.3012 | 6.8464 |
| OmniRetarget / Holosoma | Spine1 | 1.4075 | 2.8093 | 5.8620 |
| ProtoMotions v2.3 adapter | Hips | 1.3887 | 3.7728 | 6.9598 |
| ProtoMotions v3 | pelvis | 1.6664 | 3.3956 | 7.0048 |

Accordingly, Stage 1 should report two views rather than erase the distinction:

1. the frozen native-policy result, which measures each public/formal pipeline
   as released; and
2. a controlled scale-policy sensitivity result, which holds the scale policy
   constant and isolates solver/task differences.

Neither view should substitute a controlled Dense projection for the native
targets. Targeted errors must state which native task set they evaluate, and
cross-method common-joint metrics must be labeled as evaluator projections.

## Artifacts and integrity

Large target arrays are intentionally ignored by Git and live under
`runs/dance1_subject1_f000000_000600/native-pre-solver-targets/<method>/` as
pickle-free deterministic `targets.npz` plus `capture.json`. The small public
evidence is:

- `manifests/native_pre_solver_targets.csv`: one provenance row per method;
- `metrics/native_pre_solver_target_geometry.csv`: 75 auditable trajectory
  summaries (28 GMR, 15 Holosoma position sources, 14 v2.3, and 18 v3).

Frozen numeric and container hashes are:

| Method | Primary tensor SHA-256 | Deterministic NPZ SHA-256 |
|---|---|---|
| GMR | `f5be367363cd1a89cb7c4099e9606d117551e01e937ad7d04d6771ffb780d4bc` | `638b34ad1301787376c0c381783cbd72f0b8b1388a395bd873e4d8fb80e970cb` |
| OmniRetarget / Holosoma | `0d56c519cd130c16772b189f57cad7b0126b2ffe58535e656789c520bba90f82` | `78796d68e12bbca9fe0497581951007e8089f3e6a662b40f996c69e1fc50466d` |
| ProtoMotions v2.3 adapter | `84949253060eb4d3bb79ab29565fd1a436b2af2d0aeadf078b06fecc71e20575` | `4719dfa20d128471bb61215a9c21a8315cee6369f5cc8305c77efa5a20c671f3` |
| ProtoMotions v3 | `4a5cb617bb04f4ff8772d986baafa8ae1c06e8036de315996f71d039565475af` | `13175bef9639dd1bc7ab9e30e664737e1ae86b5b75ff68423a6e484de73aac68` |

The aggregate public files currently hash to:

```text
native_pre_solver_targets.csv          45df36e03f6e38a358a808002f7bbaa0f557a78aed3fc056c202e0eaa5de180b
native_pre_solver_target_geometry.csv  196cecf76c4cdee73aac553ccc0f7bb5c00f384354c9c095c7c3923dba5f8ce2
```

The numeric tensor hash includes dtype, shape, and contiguous bytes. The NPZ
writer sorts keys, forbids object arrays, fixes ZIP timestamps and permissions,
and writes atomically. Aggregation rejects a changed artifact, tensor,
geometry tensor, capture class, projection flag, or solver-invocation flag.

## Reproduction and validation

Run each capture in its frozen method environment:

```bash
PYTHONPATH=src conda run -n robot --no-capture-output \
  python -m retargeting_comparison.native_target_capture capture --method gmr
PYTHONPATH=src conda run -n hsretargeting --no-capture-output \
  python -m retargeting_comparison.native_target_capture capture --method omniretarget
PYTHONPATH=src conda run -n capture --no-capture-output \
  python -m retargeting_comparison.native_target_capture capture --method protomotions-v2.3
PYTHONPATH=src:external/ProtoMotions:external/pyroki_upstream/src \
  conda run -n egoallo --no-capture-output \
  python -m retargeting_comparison.native_target_capture capture --method protomotions-v3
```

Then aggregate and test:

```bash
PYTHONPATH=src conda run -n capture --no-capture-output \
  python -m retargeting_comparison.native_target_capture aggregate
PYTHONPATH=src conda run -n capture --no-capture-output \
  python -m pytest -q tests/test_native_target_capture.py
```

The dedicated tests verify the four frozen numeric hashes, formula behavior,
source/config/asset/implementation hashes, each method's registered target
length (600 frames except ProtoMotions v3's official 450-frame prefix),
deterministic pickle-free output, byte-reproducible public aggregation, and the
invariant `solver_invoked=false`.

## Reporting rules

- Never call a controlled Sparse/Dense semantic projection a method's native
  solver target.
- Always pair `capture_class` with `acquisition_mode`.
- For Holosoma, name the Laplacian as the solver target and the 15 positions as
  exact mesh-construction sources.
- Qualify ProtoMotions v2.3 as the formal canonical-LAFAN adapter, not an
  upstream official LAFAN pipeline.
- Treat the root-proxy geometry table as preprocessing evidence, not outcome
  quality or an accuracy leaderboard.
- Use solved G1 trajectories for retargeting metrics; use these artifacts to
  explain which target policies produced those trajectories.
