# Method Scope

The revised required set is controlled Sparse Mink, controlled Dense Mink, official GMR at `bb1bbe40774794fceb2a7c579a3464a28e68c844`, official OmniRetarget/Holosoma at `5f48635a3624656a5f46a07df26d43187e59f855`, ProtoMotions v2.3/Mink at `4a905b998101333a2fb91f2de8e2cab4bd0db68e`, and ProtoMotions v3/modified-PyRoki at `49fe5ad69de67ebbc07ea2b25d41b0f622c15c3c`. Sparse and Dense share robot, scale, root anchor, solver, limits, initialization, and regularization; only the declared task set differs.

ProtoMotions v2.3 is labelled `PHC-derived preprocessing/FK infrastructure + sequential Mink`; it is not a PHC result. ProtoMotions v2/v3 form a native pipeline lineage pair, not a pure backend ablation. PHC is lineage and AMASS-policy evidence because its official fitting asset has 37 motors rather than canonical G1-29. Native official results and controlled preprocessing/scale ablations must be separate. SOMA/cuRobo remain conditional input-compatibility candidates; bare solvers, controllers, and benchmarks remain outside the retargeter scatter.

The mandatory policy design is frozen in `research/OFFICIAL_SCALE_AND_PREPROCESSING_AUDIT.md` and `configs/scale_policy_sensitivity.yaml`. The legacy four-core results remain valid evidence, but revised Stage 1 is incomplete.

FULL-LAFAN EXPERIMENTS NOT STARTED — WAITING FOR USER APPROVAL.
