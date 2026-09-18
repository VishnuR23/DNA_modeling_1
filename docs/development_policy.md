# Development policy — 2026-09-18

The user explicitly authorized changing blocking specifications to keep development moving. This policy supersedes earlier statements that forbid all later-phase implementation until Phase 1 passes, including historical notes and deviation D69.

Scientific acceptance and permission to develop are separate. Keep measured failures, thresholds, missing evidence, and provenance unchanged. Failed molecular gates permit exploratory code, synthetic validation, solver diagnostics, and bounded local pilots. They do not support a reproduction, physical validity, transferability, detector AUROC, or speedup claim.

The legacy report field `phase2_allowed` retains its scientific-gate meaning for compatibility. Use `phase2_exploration_allowed` for development permission; `phase2_validated_claims_allowed` explicitly names the former interpretation. Passing Phase 1 is necessary, not sufficient, for validated Phase 2 claims: Phase 2 still requires its own molecular evidence.

All execution remains on local CPU with existing dependencies, no paid service or remote compute. Downloads remain opt-in. Each new pilot has explicit steps and wall-time bounds and preserves results, including failures. Commit each small implementation step and its relevant checks.

The next milestone is a variable-size, element-conditioned conditional flow prototype with a straight Gaussian-to-target path and a numerical ODE sampler. First validate centering, rotation/permutation equivariance, the path derivative, and integration on an analytic field, then train on generated synthetic data with held-out sizes. Use existing local data for subsequent molecular experiments. This initial prototype does not implement TITO's optimal-transport alignment, bond features, peptide dataset, or full architecture.

Missing large datasets trigger local synthetic or cached-data experiments, clearly labeled. Poor model metrics trigger diagnosis and reported uncertainty, rather than relabeling a failed run as accepted. No expensive campaign or new dependency is implicit in this policy.
