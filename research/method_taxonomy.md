# Method scope and taxonomy

The experimental boundary is a complete public path from a human motion to a canonical Unitree G1 29-DoF reference trajectory.
Solvers, controllers, trackers, datasets, and benchmarks are retained as lineage or pipeline context but never promoted to retargeter results.

| System | Classification | Stage 1 role | Rationale |
|---|---|---|---|
| Controlled Sparse-IK | controlled_baseline | required | required; completed |
| Controlled Dense-KeyBody IK | controlled_baseline | required | required; completed |
| GMR | retargeter | required | required; completed |
| OmniRetarget / Holosoma | retargeter | required | required; completed |
| ProtoMotions v3 | retargeter_and_framework | required | 600-frame canonical output and frozen environment still required |
| PHC retargeter | retargeter_component | lineage_noncanonical_asset | official public fitting asset is not the canonical 29-DoF embodiment |
| ProtoMotions v2.3 | historical_retargeter | required | floating SMPLSim/Mink dependencies and canonical asset compatibility must be frozen |
| SOMA Retargeter | retargeter | lineage_input_incompatible | no verified lossless LAFAN-to-SOMA adapter |
| cuRoboV2 MotionRetargeter | retargeter | lineage_input_incompatible | no verified lossless LAFAN-to-SOMA adapter |
| PhySINK / PHUMA | physics_aware_retargeting | literature_only | no frozen public end-to-end Pilot pipeline verified |
| MaskedMimic | controller_tracker | pipeline_context_only | consumes conditions as a trained controller; not offline human-to-G1 retargeting |
| BeyondMimic | controller_tracker | pipeline_context_only | requires an existing generalized-coordinate robot reference |
| LocoMuJoCo | dataset_benchmark | pipeline_context_only | benchmark/data rather than an independent raw-human-to-G1 algorithm |
| Mink | solver_backend | backend_only | not a complete input-to-output retargeter |
| PyRoki | solver_backend | backend_only | not a complete input-to-output retargeter |
| MIRROR | retargeter_non_g1 | excluded | target robot is not Unitree G1 |
| ReActor | physics_aware_controller | literature_only | RL training and dynamics rollout are outside Stage 1 |
