# Method Scope

The experimental core is controlled Sparse Mink, controlled Dense Mink, official GMR at `bb1bbe40774794fceb2a7c579a3464a28e68c844`, and official OmniRetarget/Holosoma at `5f48635a3624656a5f46a07df26d43187e59f855`. Sparse and Dense v3 share robot, uniform scale, rigid root anchor, solver, limits, warm start, iteration budget, damping, posture regularization, explicit weak temporal cost, and neutral initialization; only their declared target sets differ in the main operating-point comparison.

ProtoMotions v3 and PHC are conditional candidates subject to the two-hour gate recorded in `metrics/conditional_candidates.csv`. SOMA Retargeter and cuRoboV2 remain lineage/input-compatibility evidence. Mink/PyRoki are backends; MaskedMimic/BeyondMimic are controllers or trackers; LocoMuJoCo is a benchmark; MIRROR is non-G1. Historical, scope, and experimental claims are separated in `research/claims.csv`.

FULL-LAFAN EXPERIMENTS NOT STARTED — WAITING FOR USER APPROVAL.
