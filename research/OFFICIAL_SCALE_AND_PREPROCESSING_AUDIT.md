# Official Scale and Preprocessing Policy Audit

Status: frozen design evidence for the revised Stage 1; no Full-LAFAN or AMASS
dataset run is authorized by this document.

## Executive conclusions

1. The methods do not consume the same mathematical human target even when
   they start from the same named motion and end on a G1 robot. Their official
   adapters differ in body model, actor-shape handling, scale definition,
   root anchoring, landmarks, orientation, time sampling, grounding, contact,
   and robot limits.
2. "Human height" is not a shared observable. GMR uses a linear proxy from
   `beta[0]`; Holosoma measures a neutral SMPL-X T-pose mesh; PHC fits a
   neutral SMPL proxy to its robot; ProtoMotions v2 and v3 do not retain the
   AMASS actor's shape. These numbers must not be treated as interchangeable.
3. The existing root-error decomposition is a useful diagnostic, but it is
   not a scale-sensitivity experiment. A sensitivity experiment must change
   pre-solver targets and rerun the solver.
4. ProtoMotions v2.3 should replace PHC's *experimental slot* in Stage 1. It is
   an official canonical G1-29 sequential Mink retargeter with PHC-derived
   preprocessing/FK infrastructure. It is not a PHC result and does not
   reproduce PHC's trajectory-fitting algorithm.
5. PHC remains important lineage and preprocessing evidence, but its public G1
   fitting model has 37 motors (23 body plus 14 hand/finger), not the canonical
   29-DoF embodiment. It therefore cannot enter the canonical main comparison
   unchanged.
6. The revised Stage 1 is incomplete until ProtoMotions v2.3 and v3 are run,
   native versus controlled preprocessing is separated, the scale-policy
   sensitivity experiment is complete, and the validator enforces those
   requirements.

## Scope and terminology

This audit covers the frozen official revisions used by the benchmark:

| System | Frozen revision | Stage 1 role after this audit |
|---|---|---|
| Controlled Sparse/Dense Mink | benchmark v3 | required controls; no official AMASS path |
| GMR | `bb1bbe40774794fceb2a7c579a3464a28e68c844` | required native public pipeline |
| OmniRetarget/Holosoma | `5f48635a3624656a5f46a07df26d43187e59f855` | required native public pipeline |
| ProtoMotions v2.3 | tag `v2.3`, commit `4a905b998101333a2fb91f2de8e2cab4bd0db68e` | required native public Mink pipeline |
| ProtoMotions v3 | `49fe5ad69de67ebbc07ea2b25d41b0f622c15c3c` | required native public PyRoki pipeline |
| PHC | `846988d433ce1f341e85ac6fbd2cd51911bb3341` | lineage and AMASS-policy evidence; not a canonical G1-29 point |

The following terms are deliberately kept separate:

- **Unit conversion** changes measurement units, for example centimetres to
  metres. It is not morphology retargeting.
- **Actor shape** is the AMASS subject's `betas` and gender-specific body
  model. LAFAN1 BVH does not provide an equivalent SMPL shape identity.
- **Local/body scale** changes root-relative target geometry.
- **Root-path scale** changes the displacement of the human root through the
  world. It need not equal local/body scale.
- **Root anchor** changes the rigid world placement of a trajectory and must
  not be confused with scale.
- **Robot morphology** is the fixed G1 kinematic/visual/collision asset. Using
  the same robot does not force different target constructors to request the
  same motion.
- **Native pipeline** means the public method with its official target and
  preprocessing policy intact.
- **Controlled port/ablation** means a benchmark intervention. It must never be
  labelled as an unchanged official method result.

## AMASS policy matrix

The matrix describes code paths, not paper-level intent. `N/A` means no
official path exists for the controlled benchmark itself.

| Policy | Controlled Sparse/Dense | GMR | Holosoma | ProtoMotions v2.3 | ProtoMotions v3 | PHC fitting utility |
|---|---|---|---|---|---|---|
| Official AMASS input | N/A | raw SMPL-X NPZ; explicitly not SMPL+H | processed SMPL-X N/neutral | AMASS-X/SMPL-X-like 165-D pose is implicitly required | AMASS converted to fixed SMPL/SMPL-X MotionLib/keypoints | raw AMASS SMPL |
| Actor `betas` | N/A | used in SMPL-X FK; `beta[0]` also drives height proxy | used in neutral SMPL-X FK and mesh-height measurement | read, then overwritten with zero | not read by raw converter; fixed static skeleton | read, then ignored for motion fitting |
| Actor gender | N/A | selects male/female/neutral SMPL-X | official tested domain is neutral; current preprocessing effectively neutral-only | read, then ignored; neutral proxy used | not read at runtime | read, then ignored; neutral proxy used |
| Human height | benchmark-defined semantic geometry | `1.66 + 0.1 * beta[0]` | T-pose neutral SMPL-X mesh vertical extent | no actor-height estimate | no actor-height normalization | robot-fitted neutral SMPL shape plus one fitted scalar |
| Scale family | common shared-semantic-landmark LS (head/toe diagnostic only) | actor-dependent body-region scale | actor-height-normalizing uniform scale | fixed world-axis anisotropic scale | fixed region/axis pre-scale plus optimized robot-pair scales | fitted uniform local skeleton scale |
| Root translation | benchmark rule | scaled by root-region factor about world origin; batch output later XY-reanchored | uniformly scaled with all targets after source grounding | all world coordinates, including root, multiplied by `[0.75, 1.0, 0.8]` | root/lower local target construction uses `[0.9, 0.9, 0.85]` | source metric root path retained; only one constant root offset is optimized |
| Orientations | benchmark task-dependent | global joint orientations are explicit IK targets | discarded after SMPL-X FK; position-only representation | read and assigned a tiny `1e-4` cost | used for conversion/surgery/root initialization, not a full-body orientation residual | source root reduced to heading; position loss dominates |
| Temporal policy | exact canonical timestamps | interpolation/SLERP using an integer ratio with edge cases | integer stride, then metadata hard-coded to 30 Hz | integer stride, then hard-coded to 30 Hz | fixed 15-second buffer: trim/pad to 450 frames at 30 Hz; a CLI override changes this official operating point | integer stride, hard-coded to 30 Hz |
| Grounding | shared benchmark rule | dataset path post-shifts by sequence-global robot minimum z and zeros initial root XY | source toe-min shift before scale; constraints act during solve | per-frame output-height adjustment against unscaled human lowest joint | one sequence-global body-minimum z shift before contacts | constant source and output z shifts derived from first-frame meshes |
| Contact | shared canonical labels if enabled | none in objective | velocity-derived feet; foot sticking and non-penetration constraints | none | velocity/height labels plus contact/tilt residuals | none in fitting loss |
| Solver scope | sequential Mink | two-stage sequential Mink | Sequential SOCP | sequential Mink, two steps/frame after repeated frame-0 warm-up | whole-trajectory modified PyRoki/JAXLS | whole-sequence Adam plus per-iteration Gaussian DOF smoothing |
| G1 model | canonical Holosoma G1-29 | official GMR G1 configuration | canonical Holosoma G1-29 | bundled 29-motor MJCF; equivalence still requires FK/limit audit | 29-DoF URDF with matching origins/axes but 15 tighter limits | public fitting MJCF is 37-motor / 23-body-DoF plus hands |

## Method-level findings

### GMR

For AMASS, GMR reconstructs an actor-specific SMPL-X body from the source
gender and `betas`, preserves `root_orient`, `pose_body`, and `trans`, and
creates world positions and global orientations. Hand, jaw, and eye poses are
zeroed. The source therefore already reflects actor shape before scaling.

The AMASS height proxy is

\[
h_G = 1.66 + 0.1\,\beta_0.
\]

With the official G1 SMPL-X configuration, the resulting target factors are

\[
s_{root,lower,torso}=0.9\frac{h_G}{1.8}=0.5h_G,
\qquad
s_{upper}=0.8\frac{h_G}{1.8}.
\]

For joint `j`, GMR constructs a target of the form

\[
p'_j=s_j(p_j-p_{root})+s_{root}p_{root}.
\]

Consequently, actor anatomy and actor-dependent scaling are both present. If
body spans grow with the estimated height, their combined effect can be
super-linear; that is an inference from the code, not an upstream claim.

This differs from GMR's LAFAN loader, whose fixed human-height value produces
the already observed root/leg factor
`0.9 * 1.75 / 1.8 = 0.875` and arm factor
`0.75 * 1.75 / 1.8 = 0.729166...`. The AMASS SMPL-X and LAFAN scale tables
therefore differ even within GMR.

Additional policy differences are scientifically relevant:

- the AMASS resampler uses an integer skip and separately interpolates
  translations and rotations; non-integer source-rate ratios do not reliably
  produce 30 Hz;
- the single-file and dataset scripts have different frame/ground/root output
  behavior;
- the dataset path grounds after solving and subtracts first-frame root XY;
- the solver has no source-contact objective; and
- saved quaternion order differs from the internal MuJoCo convention.

Primary code evidence:
`general_motion_retargeting/utils/smpl.py`,
`general_motion_retargeting/motion_retarget.py`,
`general_motion_retargeting/ik_configs/smplx_to_g1.json`, and
`scripts/smplx_to_robot_dataset.py` at the frozen GMR commit.

### OmniRetarget / Holosoma

Holosoma's documented AMASS path is SMPL-X N/neutral. It uses source `betas`
with a neutral body model, performs SMPL-X FK, and retains only 22 global body
joint positions. Source joint orientations are not passed to the retargeter.
Non-neutral AMASS metadata is outside the tested contract and should not be
silently treated as supported.

Holosoma measures the vertical extent of a zero-pose neutral SMPL-X mesh and
uses

\[
s_H=\frac{1.32}{h_{mesh}}.
\]

It first shifts the whole source using the sequence-global toe minimum, then
uniformly scales every world joint position. This approximately normalizes
actor height to a handwritten G1 height of 1.32 m. It does not guarantee a
least-squares match of shared human/G1 landmark reach.

This differs from Holosoma's LAFAN policy, where the observed default is
`1.27 / 1.7 = 0.7470588`. Thus even one method does not use a single universal
"official scale" across LAFAN and AMASS.

Contact labels are derived after scaling from foot displacement with a
per-frame threshold. Both the scale and the true sampling interval can change
the contact graph. The integer-stride AMASS preprocessing always labels output
as 30 Hz, even when the actual decimated rate differs.

Two frozen-code risks must be regression-tested before an AMASS run:

- robot-only initialization derives yaw using a dummy object at the world
  origin rather than the SMPL root orientation; a zero first-frame root XY can
  make this direction undefined; and
- dummy object-pose initialization can overwrite the last seven values of a
  robot-only locked configuration.

Primary code evidence:
`data_utils/prep_amass_smplx_for_rt.py`,
`examples/robot_retarget.py`, `src/utils.py`, and
`src/interaction_mesh_retargeter.py` at the frozen Holosoma commit.

### ProtoMotions v2.3 / Mink

ProtoMotions v2.3 is a real public G1-29 retargeter. Its retargeting config and
FK helper explicitly state that they were adapted from PHC's `h1_phc` branch,
and its environment depends on SMPLSim. The actual motion solver is sequential
Mink, not PHC's whole-sequence Adam fit.

The AMASS reader loads poses, translations, `betas`, and gender, then forces
all `betas` to zero and constructs a neutral SMPL-X/SMPLH proxy. Actor-specific
shape and gender are therefore discarded. The current reshape is compatible
with a 165-D SMPL-X/AMASS-X pose after its slice operation; a conventional
156-D SMPL-H pose fails that implicit contract and must be caught by a smoke
test.

Every world target position, including the root, is multiplied about the world
origin by

```text
[x, y, z] *= [0.75, 1.00, 0.80]
```

This is neither a height normalization nor a robot-derived similarity scale.
The target set contains 14 semantic points and hand/head auxiliary offsets.
The upstream AMASS orientation cost is `1e-4` relative to position costs of
order 10. For the canonical-LAFAN port it is explicitly set to zero because BVH
joint frames have not been validated as SMPL-X segment frames; this is a
benchmark adapter intervention, not an unchanged upstream LAFAN path. G1 has configuration limits
but no contact or G1 velocity objective. The output post-process aligns each
frame's robot lowest body point to the corresponding *unscaled* human lowest
joint, which can inject root-z changes and must be reported separately.

Tag `v2.3` does not fully freeze the environment: its SMPLSim dependency points
at a floating `master`, and Mink is not version-pinned there. The benchmark
must freeze the observed SMPLSim commit, Mink version, solver, and lockfile.

Primary evidence:
`v2.3:data/scripts/retargeting/config.py`,
`v2.3:data/scripts/retargeting/mink_retarget.py`,
`v2.3:data/scripts/retargeting/torch_humanoid_batch.py`, and
`v2.3:protomotions/data/assets/mjcf/g1.xml`.

### ProtoMotions v3 / modified PyRoki

The current AMASS converter reads motion, translation, and frame rate, but not
actor `betas` or gender. FK uses a fixed static SMPL/SMPL-X humanoid asset, so
actor-specific shape is again discarded. It rotates the pose convention,
grounds the complete sequence once, and derives contact from source body speed
and height.

Before optimization, root/lower and upper root-relative targets use different
axis factors:

```text
root and lower: [0.90, 0.90, 0.85]
upper:          [0.90, 0.90, 0.80]
```

The pipeline also edits landmark geometry: shoulder/elbow offsets, rebuilt
toe/ankle points, auxiliary hands, and an auxiliary pelvis. Its whole-trajectory
JAXLS solve optimizes robot-pair scales used by the local vector residual, but
the absolute global-position residual does not use those optimized scales.

The official trajectory contract is fixed at 450 frames (15 seconds at
30 Hz), with longer inputs trimmed and shorter inputs padded for efficient JAX
compilation and batching. Although the frozen script exposes
`--target-raw-frames`, overriding it to 600 changes the documented
fixed-shape operating point and is not an official-method result. Stage 1
therefore retains the 600-frame source, runs the official 450-frame prefix,
reports native-contract completion as 450/450 and full-source coverage as
450/600, and compares every method on the same source frames `[0, 450)`.

The bundled 29-DoF URDF has the same joint names, origins, and axes as the
Holosoma canonical URDF, but 15 joint limits are tighter. It is therefore the
same nominal topology/geometry but not the same feasible set.

Primary evidence:
`data/scripts/convert_amass_to_proto.py`, `keypoint_utils.py`, and
`pyroki/batch_retarget_to_g1_from_keypoints.py` at the frozen v3 commit.

### PHC fitting utility

PHC's documented utility first fits a neutral SMPL proxy to the robot in a
static pose. It optimizes ten shape coefficients and one scalar using matched
robot/SMPL joint positions. Motion fitting then uses that same robot-fitted
neutral shape and scalar for every AMASS actor. Although AMASS `betas` and
gender are read, they are not used to construct motion targets.

The fitted scalar is applied only to root-relative SMPL joints; the source root
path remains metric and a single constant root-position offset is optimized.
The motion variables are all-frame joint angles and root headings, optimized
with whole-sequence Adam. The loss is matched point position plus a small pose
penalty; joint angles are clamped and Gaussian-smoothed after every iteration.
Contact and collision are not fitting objectives.

The official fitting MJCF has 37 motors: 23 body motors and 14 hand/finger
motors. Removing the fingers does not yield canonical G1-29 because its waist
and wrist structure differ. A separate repository reference to an unshipped
29-DoF-with-hand USD is not part of the documented fitting path.

PHC's controller-training AMASS conversion has its own filtering and neutral
shape policy. That is a controller-data pipeline and must not be substituted
for the fitting utility in a retargeter comparison.

Primary evidence: `docs/retargeting.md`,
`scripts/data_process/fit_smpl_shape.py`,
`scripts/data_process/fit_smpl_motion.py`, and
`phc/data/cfg/robot/unitree_g1_fitting.yaml` at the frozen PHC commit.

## Frozen code-evidence index

Line references below apply to the revisions listed in the scope table. For
ProtoMotions v2.3, the `v2.3:` prefix means `git show v2.3:<path>`.

| Finding | Frozen first-party location |
|---|---|
| GMR consumes actor gender/betas and constructs SMPL-X positions/orientations | `external/GMR/general_motion_retargeting/utils/smpl.py:14-47` |
| GMR AMASS height proxy and resampling | `external/GMR/general_motion_retargeting/utils/smpl.py:42-46,170-259` |
| GMR height ratio and world/root-relative target scaling | `external/GMR/general_motion_retargeting/motion_retarget.py:62-70,243-266` |
| GMR SMPL-X region factors | `external/GMR/general_motion_retargeting/ik_configs/smplx_to_g1.json:5-24` |
| GMR LAFAN region factors | `external/GMR/general_motion_retargeting/ik_configs/bvh_lafan1_to_g1.json:5-24` |
| GMR dataset post-ground and first-frame XY reanchor | `external/GMR/scripts/smplx_to_robot_dataset.py:118-132` |
| Holosoma AMASS input, stride and neutral-body behavior | `external/holosoma/src/holosoma_retargeting/holosoma_retargeting/data_utils/prep_amass_smplx_for_rt.py:13-40,80-118` |
| Holosoma mesh-height computation and position-only export | same file, `:176-202,257-289` |
| Holosoma G1 height ratio | `external/holosoma/src/holosoma_retargeting/holosoma_retargeting/examples/robot_retarget.py:235-250` |
| Holosoma grounding/scaling, initialization frame and contact labels | `external/holosoma/src/holosoma_retargeting/holosoma_retargeting/src/utils.py:208-253,343-370,686-716` |
| Holosoma position relations, output FPS and constraint blocks | `external/holosoma/src/holosoma_retargeting/holosoma_retargeting/src/interaction_mesh_retargeter.py:419-475,509-516,628-701` |
| Proto v2.3 PHC-derived config and G1 mapping | `v2.3:data/scripts/retargeting/config.py:1,78-135` |
| Proto v2.3 axis scale, targets and sequential Mink solve | `v2.3:data/scripts/retargeting/mink_retarget.py:67-94,425-554` |
| Proto v2.3 AMASS pose contract, neutral shape and output construction | same file, `:600-715` |
| Proto v2.3 floating SMPLSim dependency | `v2.3:requirements_isaacgym.txt:10` |
| Proto v3 AMASS fields, fixed body, frame rate, ground and contacts | `external/ProtoMotions/data/scripts/convert_amass_to_proto.py:137-242,316-340,428-540` |
| Proto v3 landmark surgery | `external/ProtoMotions/data/scripts/keypoint_utils.py:159-300` |
| Proto v3 scale, fixed buffer/CLI, contacts and local/global scale use | `external/ProtoMotions/pyroki/batch_retarget_to_g1_from_keypoints.py:132-229,406-410,768-899,944-1133` |
| PHC documented shape-then-motion workflow | `external/PHC/docs/retargeting.md:4-20` |
| PHC robot-fitted neutral shape and scalar | `external/PHC/scripts/data_process/fit_smpl_shape.py:49-98` |
| PHC ignores actor shape, keeps root path and performs whole-sequence Adam/smoothing | `external/PHC/scripts/data_process/fit_smpl_motion.py:34-52,68-141,164-187` |
| PHC fitting asset/mapping | `external/PHC/phc/data/cfg/robot/unitree_g1_fitting.yaml` |

## Inconsistencies beyond scale

Scale is only one of the active treatment differences. The revised benchmark
must record all of the following before interpreting output differences:

1. **Source body representation:** BVH skeleton, actor-specific SMPL-X,
   neutral SMPL-X, fixed SMPL skeleton, or robot-fitted neutral SMPL proxy.
2. **Pose information:** positions plus orientations versus positions only;
   root heading only versus full orientation targets.
3. **Landmark semantics:** different pelvis, ankle, toe, wrist, head, and
   shoulder link choices; fixed auxiliary offsets and keypoint surgery.
4. **Coordinate conventions:** up axis, handedness, root rotation offsets, and
   internal/saved quaternion order.
5. **Time:** interpolation versus stride decimation, true versus declared FPS,
   crop/pad rules, fixed buffers, and frame-0 inclusion.
6. **Ground/root convention:** pre-solve versus post-solve grounding,
   per-sequence versus per-frame shifts, first-frame XY anchoring, and root
   displacement scaling.
7. **Contact:** absent, inferred before/after scaling, fixed canonical labels,
   or native labels that change the constraint graph.
8. **Optimization:** sequential versus whole-trajectory, warm-up, temporal
   regularization, joint/velocity limits, collision, and hard constraints.
9. **Robot embodiment:** asset hash, joint order, link origins/axes, limits,
   collision geometry, auxiliary links, and 29 versus 37 motors.
10. **Dataset selection:** silent hard-motion exclusions, short-sequence
    filtering, failure handling, or default single-sequence selection.
11. **Timing boundary:** source FK, resampling, target construction, JIT, solve,
    output grounding, and conversion must be separately identifiable.

These differences explain why "same input file + same target robot" is not
sufficient to guarantee the same scale or even the same optimization target.
The robot fixes the feasible kinematics; the adapter decides which human
geometry and world trajectory the solver is asked to reproduce.

## Consequences for current metrics

- **Root translation error** mixes root-path scale, first-frame anchor, and
  post-solve shifts unless those components are reported separately.
- **RF-KPE** intentionally removes root translation and heading. It can hide
  large path-scale differences and can cluster because all outputs share one
  robot morphology. It remains useful only beside root and task residuals.
- **Foot skating/contact** changes when contact labels depend on scale or an
  incorrectly declared FPS. Contact labels must be frozen for a pure scale
  intervention and recomputed only in a separately named native-response arm.
- **Ground penetration** depends on whether grounding happens before or after
  the solve and whether it is constant or frame-dependent.
- **Joint-limit violations/completion** depend on the native robot limit set,
  not just nominal G1 joint names.
- **RTF** is incomparable if one method's source FK, JIT, or postprocessing is
  omitted. End-to-end and native-core timing must remain separate.
- **A post-hoc Procrustes fit is not a fix.** It can diagnose path shape, but it
  cannot replace rerunning a solver with controlled pre-solver targets.

## Revised Stage 1 design

### Required method set

The required operating points become:

1. Controlled Sparse Mink, neutral plus deterministic seeds A/B;
2. Controlled Dense Mink;
3. GMR / two-stage Mink;
4. OmniRetarget/Holosoma / Sequential SOCP;
5. ProtoMotions v2.3 / sequential Mink, labelled
   `PHC-derived preprocessing and FK infrastructure`;
6. ProtoMotions v3 / modified PyRoki.

PHC stays in taxonomy, lineage, and the AMASS audit with the explicit outcome:

```text
N/A — official public G1 fitting asset is not the canonical 29-DoF embodiment
```

A future `PHC canonical-G1 port` would be a benchmark intervention and an
exploratory appendix result, not an official PHC operating point.

### Protocol A: native-pipeline evidence

Run each public method unchanged except for auditable I/O, timing, and canonical
output adapters. Save the pre-solver target package in addition to qpos. This
answers: "What does a user obtain from each frozen public pipeline?"

Each native manifest must include:

```text
body_model, actor_shape_policy, gender_policy, human_height_definition
local_scale_policy, root_path_scale_policy, root_anchor_policy
unit, axes, quaternion_order, landmark_map, orientation_policy
fps_input, fps_actual, fps_declared, resampling_rule, crop_pad_rule
ground_policy, contact_label_rule, constraint_graph_hash
robot_asset_hash, joint_order_hash, joint_limit_hash
pre_solver_target_hash, postprocess_policy, dependency_lock_hash
```

### Protocol B: controlled target-policy ablation

Use one Dense Mink solver, the canonical G1 asset, fixed targets/weights/limits,
fixed root anchor, fixed timestamps, and fixed canonical contacts. Change only
the target scale constructor. Required policy transplants are:

- shared semantic landmark least-squares policy;
- Holosoma LAFAN uniform policy;
- GMR region-wise policy;
- ProtoMotions v2.3 world-axis policy; and
- ProtoMotions v3 lower/upper axis-wise policy.

These rows are labelled `controlled policy ablation`, never GMR, Holosoma, or
ProtoMotions results. PHC's fitted shape cannot be reduced faithfully to one
static factor and does not enter this transplant unless its proxy is explicitly
constructed and validated.

### Protocol C: within-method scale response

For every required public method, rerun the frozen Pilot from pre-solver
targets using two independent, origin-safe multipliers:

\[
\Delta r'_t=g_{root}(r_t-r_0),
\qquad
o'_{t,j}=g_{local}(p_{t,j}-r_t).
\]

Required variants are:

```text
(g_root, g_local) =
  (1.00, 1.00) native
  (0.95, 1.00) root-minus-5
  (1.05, 1.00) root-plus-5
  (1.00, 0.95) local-minus-5
  (1.00, 1.05) local-plus-5
```

The first-frame root anchor is fixed. For a native anisotropic or region-wise
policy, the perturbation multiplies the policy output; it does not erase that
policy's structure. No finished qpos may be rescaled.

For Holosoma and ProtoMotions v3, solve a fixed-contact-label arm to isolate
geometry and record how many labels/constraints would change under native
recomputation. A separately named recomputed-contact arm is required if the
label-flip rate is nonzero enough to affect conclusions.

### Common scale definition — conformance correction, 2026-07-22

The controlled policy does not inherit any method's handwritten "height". The
Stage 1 formal common scale now follows the registered estimator in task-prompt
Section 6.7:

\[
s^*=\frac{\sum_k w_k(h_k-h_{root})^T(r_k-r_{root})}
{\sum_k w_k\|h_k-h_{root}\|^2}
=0.8280075950863185.
\]

The source vectors use frame-0 Hips as origin and remove the geometry-derived
body heading; robot vectors use the neutral canonical Holosoma G1 pelvis and
+X-forward frame. The equal-weight registered set is head, bilateral
shoulder/hip/knee/ankle/toe. LAFAN Spine2 versus G1 torso-link origins are not
homologous, and distal arms are excluded because source frame-0 and neutral-G1
arm poses differ. The exact ordered points, weights, sufficient statistics,
residuals, scene hash, and joint-order hash are frozen in
`manifests/evaluator.yaml`.

The former frame-0 `head→mean(toes)` result is retained only as a diagnostic:
`0.7833775086249071` with the canonical Holosoma `mid360` head site. The earlier
`0.7420370439847394` value used a synthetic torso-offset head point while the
controlled solver itself used the non-canonical GMR mocap scene. It is a legacy
result, not a formal common-policy arm. Neither diagnostic is permitted in
`TRANSPLANT_POLICIES`.

Local/body scale, root-displacement scale, and the rigid frame-0 root anchor
are separate registered parameters. The first two deliberately start with the
same numerical LS value; the ±5% intervention perturbs them independently.
Stage 2 recomputes the same estimator per source actor and continues to report
head/toe span as a diagnostic rather than silently changing the estimator.

### AMASS actor-shape policy formula probe without an AMASS dataset run

Using the licensed neutral SMPL-X model, reconstruct the published formulas on
the same static pose and root path for three predeclared neutral beta vectors
(short, zero, tall), with every formula bound to hashes of its official source,
and record:

- mesh/landmark height under that method's definition;
- root and local target gains;
- pre-solver landmark spans;
- contact-label changes; and
- target hashes.

This is formula-reconstruction evidence, not an observed native constructor,
pre-solver response, or AMASS motion benchmark. It may explain differences in
AMASS actor-shape policy, but it is forbidden as accuracy or ranking evidence.
Exact LAFAN runtime targets are established separately by the schema-2 native
pre-solver capture contract.

### Metrics and plots

For every scale variant report raw values and finite-difference response:

- targeted and untracked RF-KPE;
- root common-policy, native-policy, and scale-invariant path error;
- declared task residual and reach-infeasible-frame rate;
- foot skating/contact-label flip rate, penetration, joint-limit violations,
  invalid frames, and completion;
- velocity, acceleration, jerk, and pose jumps;
- qpos and canonical FK divergence from the native run;
- end-to-end and native-core timing; and
- method ranking stability, without a composite score.

For metric `m` and scale factor `g`, report the centered response where possible:

\[
S_m=\frac{m(1.05)-m(0.95)}{0.10}.
\]

If rankings reverse under a reasonable policy perturbation, the conclusion is
"pipeline comparison is preprocessing-sensitive", not that one solver is
universally superior.

Required visuals are a native-versus-controlled faceted chart, scale-response
curves with the native point marked, a root/local sensitivity heatmap or slope
chart, target-geometry diagnostics, contact-label/constraint changes, and rank
stability. Interactive views must expose the active policy and pre-solver
targets alongside the articulated G1 result.

## What must be unified, and what must remain native

Unify for the controlled scientific comparison:

- canonical timestamps and actual FPS;
- units, axes, handedness, quaternion convention;
- first-frame root anchor and definition of root displacement;
- canonical G1 asset, joint order, and joint limits;
- shared semantic landmark names;
- fixed contact labels for pure scale interventions;
- completion and failure rules; and
- end-to-end timing boundaries.

Preserve and report in the native-pipeline arm:

- official actor-shape and height policy;
- official landmark construction and auxiliary offsets;
- official contact/grounding behavior;
- official optimizer, weights, limits, and temporal scope; and
- official output postprocessing.

Do not silently make a public method "fairer" and keep its original label.
Instead, publish paired columns: `native official pipeline` and `controlled
benchmark port/ablation`.

## ProtoMotions v2 versus PHC decision

Using ProtoMotions v2.3 in place of PHC is correct for the canonical Stage 1
method set, with three qualifications:

1. It replaces the experimental slot, not PHC's algorithmic evidence.
2. Its scientific label is
   `ProtoMotions v2.3 / Mink (PHC-derived preprocessing and FK infrastructure)`.
3. It adds an official task/preprocessing design point, not a new solver family;
   Controlled Sparse/Dense and GMR also use Mink.

The v2-to-v3 pair is valuable lineage evidence, but it is not a pure
Mink-versus-PyRoki ablation. At the same time it changes actor representation,
scale, landmarks, contacts, grounding, temporal scope, costs, and limits. Only
Protocol B can isolate a target-policy change, and a separate controlled solver
experiment would be needed to isolate the backend.

Before v2.3 is accepted, Stage 1 must freeze its floating dependencies, verify
the 165-D input contract, and compare neutral/random-qpos FK, joint order,
limits, and link geometry against the canonical G1 model. Native-asset results
may be retained, but any canonical-asset port must be labelled explicitly.

## Revised acceptance and current status

The legacy four-point LAFAN execution is complete. The revised Stage 1 is not.
It may return `GO` or `GO WITH CHANGES` only when all of the following are true:

- the AMASS/LAFAN preprocessing policy audit is frozen with file-level evidence;
- ProtoMotions v2.3 produces a valid 600/600-frame output, while ProtoMotions
  v3 produces its official 450/450-frame native output and explicitly reports
  450/600 full-source coverage;
- the v2.3 dependency and robot-asset compatibility gates pass;
- every experimental point exposes a hashed pre-solver target package;
- native official and controlled ablation results are visually and
  terminologically separated;
- all five within-method scale variants complete for required public methods;
- the controlled policy transplant and actor-shape formula probe complete;
- scale variants differ only in registered scale fields and contact-mode fields;
- sensitivity metrics, raw results, plots, and interactive views are complete;
- the validator returns `NO-GO` when any mandatory item above is missing; and
- the revised Stage 2 runtime/storage projection remains within the frozen
  48-hour/200-GB gate or proposes a deterministic simplification for discussion.

Until then the correct decision is `NO-GO — revised Stage 1 work in progress`.
This is a completeness status, not a negative experimental result.

## Questions reserved for the user's scale exploration

No answer is required before implementation of the native audit and ±5% Pilot
response. Before freezing the controlled common policy, the user's follow-up
should resolve whether:

1. whether a future, separately preregistered actor-rest-skeleton calibration
   should supplement (not retroactively replace) the Stage 1 frame-0 semantic
   landmark estimator; and
2. the optional ±10% extension is scientifically useful after the ±5% response
   is inspected.

Any change after the first formal sensitivity run requires a dated protocol
amendment; results may not drive a silent choice of scale or range.

## First-party references

- [GMR repository](https://github.com/YanjieZe/GMR)
- [Holosoma retargeting documentation](https://github.com/amazon-far/holosoma/tree/main/src/holosoma_retargeting)
- [ProtoMotions repository](https://github.com/NVlabs/ProtoMotions)
- [ProtoMotions PyRoki workflow](https://nvlabs.github.io/ProtoMotions/tutorials/workflows/retargeting_pyroki.html)
- [PHC retargeting documentation](https://github.com/ZhengyiLuo/PHC/blob/master/docs/retargeting.md)

FULL-LAFAN EXPERIMENTS NOT STARTED — WAITING FOR USER APPROVAL.
