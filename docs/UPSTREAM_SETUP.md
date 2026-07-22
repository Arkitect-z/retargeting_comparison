# Frozen upstream setup

The harness does not vendor third-party history. Create the ignored checkouts and verify their clean commits:

```bash
git clone https://github.com/YanjieZe/GMR.git external/GMR
git -C external/GMR checkout bb1bbe40774794fceb2a7c579a3464a28e68c844
git clone https://github.com/amazon-far/holosoma.git external/holosoma
git -C external/holosoma checkout 5f48635a3624656a5f46a07df26d43187e59f855
git -C external/holosoma apply ../../patches/holosoma/interaction-hard-constraint-flags.patch
```

GMR runs in `robot`; Holosoma runs in `hsretargeting`; the controlled baselines and evaluator run in `capture`. Exact package versions are in `environments/environment-locks.yaml`. The launcher fixes relevant CPU thread variables to one and disables visualization.

The Holosoma patch only makes the already-exposed `activate_obj_non_penetration` setting govern construction of its constraint block. The upstream foot-sticking block already checks `activate_foot_sticking`. `tests/test_holosoma_patch.py` protects both gates.

Licensed LAFAN and SMPL/SMPL-X files remain outside Git. See `manifests/dataset.yaml` and `manifests/body_models.yaml` for hashes and formats.
