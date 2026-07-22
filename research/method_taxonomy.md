# Method scope and taxonomy

The experimental boundary is a complete public path from a human motion to a canonical Unitree G1 29-DoF reference trajectory.
Solvers, controllers, trackers, datasets, and benchmarks are retained as lineage or pipeline context but never promoted to retargeter results.

| System | Classification | Stage 1 role | Rationale |
|---|---|---|---|
| Controlled Sparse-IK | controlled_baseline | required | Tests the root+4EE sufficiency hypothesis under a controlled backend. |
| Controlled Dense-KeyBody IK | controlled_baseline | required | Task-density control sharing all Sparse solver settings. |
| GMR | retargeter | required | Public direct human-to-G1 pipeline and controlled-baseline anchor. |
| OmniRetarget / Holosoma | retargeter | required | Primary interaction-aware method and official case-study source. |
| ProtoMotions v3 | retargeter_and_framework | conditional_1 | Current public generation; two-hour native-adapter gate. |
| PHC retargeter | retargeter_component | conditional_2 | Historically important learned-humanoid lineage; two-hour gate. |
| ProtoMotions v2 | historical_retargeter | lineage_only | Historical v2/v3 backend evidence; not core-first execution. |
| SOMA Retargeter | retargeter | lineage_unless_native_adapter | No unverified LAFAN-to-SOMA hidden retargeter is permitted. |
| cuRoboV2 MotionRetargeter | retargeter | lineage_unless_native_adapter | Input compatibility must be demonstrated without a hidden retargeter. |
| MaskedMimic | controller_tracker | pipeline_context_only | Downstream tracking/control must not become an experimental retargeter point. |
| BeyondMimic | controller_tracker | pipeline_context_only | Consumes references rather than generating them from human motion. |
| LocoMuJoCo | dataset_benchmark | pipeline_context_only | Benchmark/data must not be represented as a retargeting algorithm. |
| Mink | solver_backend | backend_only | A naked solver is not an independent full retargeter. |
| PyRoki | solver_backend | backend_only | A backend is lineage evidence, not an experimental system point. |
| MIRROR | retargeter_non_g1 | excluded | Target robot does not satisfy the frozen Unitree G1 scope. |
| ReActor | physics_aware_controller | literature_only | RL training/dynamics rollout is explicitly excluded from Stage 1. |
