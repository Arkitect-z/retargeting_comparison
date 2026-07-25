# LAFAN1 representation, PHC fitted targets, and corrected visualization

## Finding

The original LAFAN1 release is not an SMPL or SMPL-X dataset. Ubisoft's
official repository describes the animation files as BVH, and the local
licensed copy confirms a 22-joint BVH hierarchy sampled at 30 Hz. The frozen
Pilot source is a deterministic 600-frame crop of that BVH. Consequently, any
skinned body shown for this experiment is a fitted derivative and must not be
described as an original LAFAN1 SMPL asset.

Primary source:
[Ubisoft La Forge Animation Dataset](https://github.com/ubisoft/ubisoft-laforge-animation-dataset).

## Visualization-only BVH-to-SMPL adapter

`rtcmp prepare-smpl-skin` fits the audited neutral chumpy-free SMPL model to
the 22 canonical LAFAN joints. It optimizes one shared 10-dimensional shape,
one shared positive scalar, and per-frame SMPL pose. The objective contains
joint-position error, weak pose/shape priors, and a temporal pose prior. Root
translation is preserved from the canonical source.

For the frozen Pilot sequence:

| Property | Result |
|---|---:|
| Source frames | 600 |
| Source joints used | 22 |
| SMPL vertices | 6,890 |
| SMPL triangles | 13,776 |
| Fit iterations | 240 |
| Root-aligned MPJPE | 37.855 mm |
| Maximum fitted-joint error | 118.417 mm |
| Fitted actor body scalar | 1.047674 |

The fitted surface is retained only as an audited internal adapter needed to
provide PHC's public SMPL/AMASS input contract. Its non-zero fit error means it
is not a substitute for the canonical BVH. It is no longer rendered in either
Rerun recording, and no Stage 1 metric is recomputed from it.

The nine-method Rerun recording renders the original 22-joint BVH hierarchy
directly in its grid, world, and root-frame views. Every target remains the full
articulated canonical G1-29 mesh.

## What the public PHC fitting utility actually does

The implementation is frozen to PHC commit
`846988d433ce1f341e85ac6fbd2cd51911bb3341` and follows the official
[retargeting instructions](https://github.com/ZhengyiLuo/PHC/blob/master/docs/retargeting.md)
without changing its optimization code.

### Shape fit

`fit_smpl_shape.py robot=unitree_g1_fitting`:

1. constructs a neutral SMPL model;
2. applies PHC's robot-specific rest-pose modifiers;
3. matches 16 SMPL/robot landmarks, including the extended head and toes;
4. optimizes ten SMPL beta coefficients and one scalar for 3,000 Adam steps.

The upstream loss has no beta prior, scalar prior, or beta clamp. The resulting
coefficients are therefore numerically extreme and should not be interpreted
as a plausible human identity:

```text
[2.907651, 3.251746, -6.343317, 0.731995, 8.591267,
 4.393405, -16.638264, -6.056960, -29.382671, 16.771166]
```

The fitted PHC scalar is `0.654250`. This number is not directly comparable to
the visualization adapter's actor scalar because PHC simultaneously replaces
the actor shape with a different, unconstrained robot-fitted shape.

### Motion fit

`fit_smpl_motion.py`:

1. reads AMASS-style SMPL pose and translation;
2. keeps the first 66 pose components and zeroes the final six;
3. reads the input gender and betas but does not use either when constructing
   motion targets;
4. constructs targets from the neutral, robot-fitted SMPL shape and the
   robot-fitted scalar;
5. optimizes the full trajectory for 500 iterations, including robot DoFs,
   root heading, and one sequence-wide root-position offset;
6. clamps joints and applies a Gaussian temporal filter every iteration;
7. grounds the final robot from its first-frame mesh.

Thus, PHC does not preserve the LAFAN actor body shape during robot fitting.
It intentionally replaces it with a robot-conditioned proxy body.

The adapter supplies the official utility with an AMASS-shaped NPZ derived from
the fitted LAFAN SMPL pose. This is necessary because LAFAN1 is BVH, whereas
PHC's public utility expects SMPL/AMASS fields. The adapter boundary and both
hashes are recorded in `manifests/phc_visualization_preparation.json`.

### Robot embodiment mismatch

The public `unitree_g1_fitting` asset has:

- 37 actuated joints;
- 23 body joints;
- 14 hand/finger joints;
- 44 qpos values after adding the free root;
- 43 visual mesh assets.

This is not the canonical 29-DoF G1 used by the Stage 1 evaluator. The PHC
result is therefore an explanatory, separate visualization and is not added to
the main operating-point plot. Cropping or remapping it to G1-29 would create a
new method variant and would no longer be the untouched public PHC result.

Accordingly, the nine-trajectory Stage 1 recording contains no standalone PHC
entity. Its `ProtoMotions v2.3 · Mink` trajectory comes from a PHC-lineage
repository but uses the Mink retargeter path; lineage does not make it the PHC
algorithm or PHC's public 37-motor output.

## PHC three-way visual narrative

`rtcmp visualize-phc` shows:

1. **Original LAFAN1 BVH** — the exact 22 canonical joint positions and native
   BVH parent hierarchy, with no fitted skin.
2. **PHC robot-fitted targets** — the exact 24 joint positions saved in PHC's
   official motion output after its internal neutral-shape and scalar policy.
   These are keypoints, not a claim that LAFAN1 supplied SMPL.
3. **PHC public G1 result** — the full 43-mesh, 37-motor robot output produced
   by the official fitting utility.

The side-by-side view preserves trajectory evolution while separating the
three representations into lanes. The root-centered overlay draws the 16
official PHC target correspondences. Across all 600 frames, the saved target to
robot residual is 31.771 mm mean, 31.745 mm median, 63.808 mm p95, and
97.007 mm maximum. This confirms that the displayed G1 follows the saved fitted
targets within the public optimizer's residual.

### Corrected G1 mesh transform

The first implementation incorrectly logged each raw STL with MuJoCo's compiled
geom pose. MuJoCo first recenters/reorients raw mesh vertices and stores that
mesh-reference correction in the geom pose. Combining the compiled pose with
the uncompiled STL applied the correction twice, which made robot parts appear
detached even though the official `qpos` was valid.

The corrected cache embeds `model.mesh_vert` and `model.mesh_face`—MuJoCo's
compiled geom-local geometry—and then applies `data.geom_xpos` and
`data.geom_xmat` exactly once. Numeric assembly guards prove that frame zero
has a `1.452 × 0.286 × 1.271 m` bounding box and that every transformed mesh
corner remains within `0.845 m` of the free root over all 600 frames. A direct
MuJoCo render of the same `qpos` is also retained as local visual-validation
evidence.

## Reproduce

```bash
# GPU is used only for the local visualization-skin fit.
PYTHONPATH=src conda run -n capture --no-capture-output \
  python -m retargeting_comparison.cli prepare-smpl-skin --device cuda

# The official PHC scripts themselves run on CPU.
PYTHONPATH=src conda run -n capture --no-capture-output \
  python -m retargeting_comparison.cli prepare-phc-visualization

PYTHONPATH=src conda run -n vis --no-capture-output \
  python -m retargeting_comparison.cli visualize-phc

conda run -n vis rerun artifacts/visualization/phc_scale_three_way.rrd
```

The full PHC preparation took 24.64 s for the 3,000-step shape fit and 20.26 s
for the 600-frame, 500-step motion fit on the recorded host. The corrected
Rerun recording contains all 600 frames, native/keypoint human representations
only, and passed `rerun rrd verify`.

FULL-LAFAN EXPERIMENTS NOT STARTED — WAITING FOR USER APPROVAL.
