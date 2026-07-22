# Unitree-attributed LAFAN1 Reference Corpus Audit

Audit date: 2026-07-22
Status: admissible only as an external, precomputed reference corpus
Frozen revision: `ce1572906efe6157840e8474d5a0d7aa87481e74`

## Bottom line

The revision-pinned `g1/dance1_subject1.csv` can be paired with the frozen Stage 1 Pilot because it has the exact same `dance1_subject1` basename and contains all 600 Pilot frames. Its 36-value layout is compatible with the benchmark after converting the free-root quaternion from Pinocchio `xyzw` storage to canonical `wxyz` storage and normalizing it. Two explicit coordinate views are retained: the exact as-published world frame and a convention-normalized benchmark frame produced by one fixed `Rz(-pi/2)` rotation of both root XYZ and root orientation.

It must be labeled **Unitree-attributed reference corpus**, not “official Unitree ground truth,” “verified official baseline,” or a reproducible retargeting method. The Hugging Face uploader is `lvhaidong`, not a Unitree organization; no generation commit or configuration is supplied; and the dataset owner later stated that the dataset had been copied and that method material was unavailable. It is also precomputed, so no valid cold/core/steady-state runtime exists.

The first 600 frames reveal a highly structured but undocumented root policy: after first-frame anchoring, reference XY is almost a `0.74001746`-scaled, approximately 90-degree-rotated version of canonical LAFAN XY. The full 3,945-frame sequence confirms that the rotation is within `0.000152 rad` of exactly 90 degrees. Vertical root motion follows a different affine relation. This is evidence that the corpus applies a root-frame/scale/grounding policy, not evidence of which code produced it.

## Frozen provenance and non-claims

| Item | Frozen observation | Consequence |
|---|---|---|
| Repository | [`lvhaidong/LAFAN1_Retargeting_Dataset`](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) | Third-party Hugging Face corpus |
| Revision | [`ce1572906efe6157840e8474d5a0d7aa87481e74`](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/tree/ce1572906efe6157840e8474d5a0d7aa87481e74) | Immutable input for this audit |
| Pilot CSV | `g1/dance1_subject1.csv`, SHA-256 `e2a369e92e5ad076c5acffff7bce53157a0e6cd9e7e9f6a08dadfb1eb2928d8f` | Raw file remains under ignored `data/external/` |
| Claimed method | Interaction Mesh plus IK, end-effector pose/joint position/joint velocity constraints | A card-level claim only; no executable generation path is present |
| Claimed scope | Kinematics only; no dynamics or actuator limitations | Must not be described as dynamically executable ground truth |
| Owner discussion | [Discussion #1](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/discussions/1) says no retargeting material was available and that the dataset was copied | Blocks verified-official and reproducible-method status |

The corpus is useful as an additional frozen trajectory to inspect. It is not a substitute for GMR, OmniRetarget/Holosoma, controlled Mink, or any other runnable method.

## License boundary

The repository contains three statements that do not form an unambiguous license grant for the trajectory CSV:

- the card says the underlying LAFAN1 data are CC-BY-NC-ND-4.0;
- the root `LICENSE` is BSD-3-Clause with Unitree Robotics copyright;
- the card says “the code” is MIT, although there is no MIT license file in the frozen snapshot.

Therefore the benchmark must not commit or redistribute the raw CSV or derived full trajectory. Only code, hashes, aggregate measurements, and provenance may be published pending a separate license determination.

## File and adapter contract

The downloaded CSV has 3,945 rows, no header, 36 finite numeric columns, and near-unit root quaternions (maximum norm error in the Pilot slice below `5.7e-7`). The dataset card and its Pinocchio visualizer agree on this order:

```text
root_x, root_y, root_z, root_qx, root_qy, root_qz, root_qw,
29 G1 joint positions in the benchmark's frozen canonical order
```

The adapter in `src/retargeting_comparison/unitree_reference.py` first performs exactly:

1. revision/hash and width/finite-value validation;
2. slice rows `[0, 600)` for the Pilot;
3. `xyzw -> wxyz` root-quaternion conversion and unit normalization;
4. construction of canonical `qpos[T,36]` with source row indices.

It then exposes two named views:

- `as_published`: identity world transform, for provenance and raw visualization;
- `canonical_coordinate_view`: fixed active `Rz(-pi/2)` applied to absolute root XYZ and left-multiplied onto root orientation, for formal quality comparison.

The second operation is a coordinate convention conversion, not registration to the result. Its angle is exactly `-pi/2`, translation is exactly zero, scale is exactly one, and it is invertible to numerical precision. Tests prove root-distance preservation, unchanged joint values, and forward/inverse round-trip. Neither view applies anchoring, scale, ground offset, resampling, interpolation, joint reordering, or joint-angle post-processing. The solve-time array is an explicit zero placeholder marked unavailable and must be excluded from every runtime/RTF plot. Root translation in metres and joint positions in radians are strongly implied by direct Pinocchio-URDF visualization and numeric magnitude, but the card does not state these units explicitly; this assumption remains recorded.

The two Pilot qpos content hashes are frozen independently of NPZ container timestamps:

| View | qpos content SHA-256 | Use |
|---|---|---|
| `as_published` | `b0755c087c5187453f5fe5eff874af321a303404124e01deba627c0f1ff77f27` | Provenance/raw visualization |
| `canonical_coordinate_view` | `c2ad1eafaf229c52fa1b65baeb0e5318ec61420cb53f03929db2bdf502f8c958` | Formal quality comparison |

The source BVH parser reports `30.000300003 Hz`, while the card asserts `30 Hz`. At frame 599 the clock difference is only `0.000199667 s`, but timestamps are not claimed to be bit-identical and the adapter does not resample. The comparison is therefore described only as frame-index aligned, never timestamp-identical.

## G1 asset equivalence and one important mismatch

The corpus URDF and the canonical evaluator URDF have different byte hashes, but their 29 actuated joints have identical names, order, tree edges, origins, axes, and limits. A stronger cross-engine check evaluated the pelvis plus 29 actuated child-link frames at neutral and 10 deterministic random configurations (`seed=1947`):

| Comparison | Maximum position difference | Maximum rotation difference | Result |
|---|---:|---:|---|
| Corpus Pinocchio URDF vs canonical MuJoCo scene | `4.731e-7 m` | `1.456e-6 rad` | Kinematically equivalent at `1e-5` tolerance |

This establishes qpos/link-frame compatibility only. Visual meshes, collision shapes, inertias, transmissions, actuators, and dynamics are outside the equivalence claim.

There is a separate mismatch that matters for method comparison. Holosoma's retargeting package defaults to `holosoma_retargeting/models/g1/g1_29dof.urdf`, whereas the benchmark evaluator uses `holosoma/data/robots/g1/g1_29dof.urdf`/the corresponding MuJoCo scene. The retargeter URDF differs at the waist roll, waist pitch, and both shoulder pitch joint origins, with a maximum origin-component difference of `0.019 m`. For the same 11 qpos samples, link-frame positions differ by as much as `0.012011 m` (worst sampled link: torso), although rotations and joint order remain aligned.

This mismatch should not be silently “fixed” after solving. Stage 1 should record the solver asset and evaluator asset separately, report the resulting asset-transfer error floor, and add an optional same-asset diagnostic. Main public-method results should retain the official method asset; a harmonized-asset rerun, if performed, must be labeled an intervention rather than the public method.

## Observed Pilot root, scale, and ground policy

For descriptive diagnosis only, fit a proper 2-D row-vector similarity after independently subtracting the first root position:

```text
delta_xy_reference ~= scale * delta_xy_source @ R
```

The measured fit is:

```text
scale = 0.7400174555
R = [[ 0.0003254389,  0.9999999470],
     [-0.9999999470,  0.0003254389]]
XY RMSE = 0.00342379 m
XY maximum error = 0.01452484 m
```

Thus the dominant coordinate map is approximately `[x_ref, y_ref] = [-y_source, x_source]`, with first reference XY `[0.000098, -0.000161] m`. The benchmark source explicitly maps original Y-up LAFAN coordinates to `[x, -z, y]`; the published path is correspondingly consistent with the alternative right-handed Z-up axis order `[z, x, y]`. The upstream visualizer also declares a right-handed Z-up world and forwards the CSV qpos without another transform. These observations justify freezing the exact inverse axis permutation, `Rz(-pi/2)`, rather than fitting the measured `89.981 degrees` Pilot angle or applying a result-derived alignment.

This separates three effects that a raw root-translation error would otherwise conflate: first-frame anchoring, axis/heading convention, and motion scale.

The fitted scale is close to, but not identical to, Holosoma's LAFAN configuration ratio `1.27 / 1.70 = 0.74705882`: it is lower by `0.00704137` (about `0.943%`). That proximity is a useful hypothesis for follow-up, not proof that this corpus came from Holosoma or used that exact constant.

Vertical motion does not use the fitted XY transform:

```text
z_reference ~= 0.69763967 * z_source + 0.15715795 m
z RMSE = 0.00426053 m
```

This is consistent with a separate root-height/ground/IK treatment, but the absent generation configuration prevents causal attribution. The reference root ranges from `0.615946` to `0.801053 m` in the Pilot.

Under the canonical evaluator geometry, semantic toe frames stay above `z=0` (minimum `0.002224 m`), yet collision geometry reports floor penetration in 22 of 600 frames, with a maximum of `0.005746 m`. This is not a contradiction: a semantic link/contact frame is not the full collision surface, and collision equivalence with the upstream URDF was not established. These numbers must be labeled canonical-evaluator diagnostics, not violations of an unknown upstream constraint set.

The impact of coordinate-view choice is directly measurable and confirms that this is a convention correction rather than scale manipulation:

| Evaluator metric | As published | Canonical coordinate view |
|---|---:|---:|
| Mean common-scale root translation error | `2.292016 m` | `0.006835 m` |
| Mean root yaw error | `1.547857 rad` | `0.038415 rad` |
| Effective root XY scale | `0.000241` (axes crossed; meaningless) | `0.740017` |
| RF-KPE-all | `0.110279 m` | `0.110279 m` |

RF-KPE is unchanged because it removes a consistently applied global yaw. The native scale remains `0.740017`; the coordinate conversion did not change it. Reporting the raw view as a formal quality result would therefore manufacture a 2.29 m error from incompatible axes.

## Required Stage 1 treatment

1. Add the corpus only as a fifth, precomputed **reference** trajectory, never as a timed retargeter operating point.
2. Preserve both views and both hashes. Use `canonical_coordinate_view` for formal quality metrics and `as_published` only for provenance/raw visualization.
3. Treat the fixed `Rz(-pi/2)` as an input convention conversion. It may not estimate angle, translation, or scale from method results. Preserve the corpus's native first-frame anchor, XY scale, root height, and ground policy after that conversion.
4. If a later common registered-evaluation diagnostic is added, label it separately and never replace the coordinate-normalized native-policy view.
5. Keep root-policy metrics separate: anchor, axis map, effective XY scale, vertical affine diagnostic, and post-registration residual.
6. Evaluate qpos using the frozen canonical model, while reporting the solver/evaluator asset scope and the Holosoma retargeter-asset mismatch separately.
7. Exclude the corpus from runtime, cold-start, solver-convergence, and reproducibility rankings.
8. Do not infer algorithm identity from the scale's proximity to Holosoma, and do not use this corpus as ground truth for ranking other methods.

Structured evidence is in `research/unitree_reference_provenance.csv`, `research/unitree_reference_pilot_audit.csv`, and `configs/unitree_reference.yaml`. Raw data and the locally generated canonical NPZ remain ignored.

## Reproduction

Adapter and provenance tests:

```bash
PYTHONPATH=src /home/weizy/anaconda3/bin/python -m pytest -q tests/test_unitree_reference.py
```

Cross-engine FK audit (the `robot` environment supplies Pinocchio and MuJoCo):

```bash
PYTHONPATH=src /home/weizy/anaconda3/envs/robot/bin/python -m pytest -q tests/test_unitree_reference.py
```

FULL-LAFAN EXPERIMENTS NOT STARTED — WAITING FOR USER APPROVAL.
