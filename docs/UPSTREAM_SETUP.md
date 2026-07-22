# Frozen upstream setup

The harness does not vendor third-party history. Create the ignored checkouts and verify their clean commits:

```bash
git clone https://github.com/YanjieZe/GMR.git external/GMR
git -C external/GMR checkout bb1bbe40774794fceb2a7c579a3464a28e68c844
git clone https://github.com/amazon-far/holosoma.git external/holosoma
git -C external/holosoma checkout 5f48635a3624656a5f46a07df26d43187e59f855
git -C external/holosoma apply ../../patches/holosoma/interaction-hard-constraint-flags.patch
git clone https://github.com/NVlabs/ProtoMotions.git external/ProtoMotions
git -C external/ProtoMotions checkout 49fe5ad69de67ebbc07ea2b25d41b0f622c15c3c
git -C external/ProtoMotions worktree add ../ProtoMotions-v2 4a905b998101333a2fb91f2de8e2cab4bd0db68e
git clone https://github.com/ZhengyiLuo/PHC.git external/PHC
git -C external/PHC checkout 846988d433ce1f341e85ac6fbd2cd51911bb3341
```

GMR runs in `robot`; Holosoma runs in `hsretargeting`; the controlled baselines and evaluator run in `capture`. Exact package versions are in `environments/environment-locks.yaml`. The launcher fixes relevant CPU thread variables to one and disables visualization.

ProtoMotions v2.3 and v3 are required by the revised Stage 1. V2.3 uses the
separate worktree because the two official revisions must coexist. Before any
formal v2.3 run, freeze its floating SMPLSim `master`, Mink/solver versions, and
complete dependency lock; verify the implicit 165-D SMPL-X input contract and
canonical G1 FK/limits/geometry. V3 must freeze `--target-raw-frames 600` and
record its tighter joint-limit hash. PHC is lineage/AMASS-policy evidence only:
its documented fitting asset is 37-motor rather than canonical G1-29.

The synchronized result viewer runs separately in `vis` with Rerun. It is not
part of formal method timing and does not import either upstream method's
Python environment. It reads the 35 official G1 visual meshes from the frozen
Holosoma URDF and instances them using canonical qpos FK. See
`docs/RERUN_VISUALIZATION.md`.

The Holosoma patch only makes the already-exposed `activate_obj_non_penetration` setting govern construction of its constraint block. The upstream foot-sticking block already checks `activate_foot_sticking`. `tests/test_holosoma_patch.py` protects both gates.

Licensed LAFAN and SMPL/SMPL-X files remain outside Git. See `manifests/dataset.yaml` and `manifests/body_models.yaml` for hashes and formats.
