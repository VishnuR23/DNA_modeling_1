# Local CPU Phase 1 status — 2026-09-10

**Software pipeline works; scientific reproduction has not passed.** This is the historical Phase 1 report. The [2026-09-18 development policy](development_policy.md) now permits later-phase exploration; see [Phase 2 status](phase2_status.md). The interpretation of the epsilon skip below is corrected in the [new denoising report](denoising_status.md). All execution described here occurred on this MacBookPro17,1 (Apple M1,8 GiB RAM), without CUDA/MPS/remote compute. The original research phase gates remain binding.

## Implemented and checked

- Python3.11 local environment and recorded dependency versions.
- CPU-only Hydra experiments, seeds, logs, configs, strict metrics JSON and numerical figure data.
- Full original MDShare alanine download with recorded/pinned SHA256 and validated timestamp reset handling:3×250,000 frames,22 atoms,1ps spacing,750ns aggregate.
- Centered data, boundary-safe lag pairs, ChiroPaiNN DDPM, lag conditioning, corrected noise indexing, EMA sampling, bounded training and exact checkpoint resume.
- Physical rollouts, nested/direct sampling, phi/psi free energy, JSD, fixed-grid MSM and VAMP-2 diagnostics, chain bootstrap and matched-count reference controls.
- OpenMM ff14SB/OBC2 implicit control on its Reference CPU platform.
- Seven tests passed, including proper-rotation equivariance and a tiny synthetic train/sample/resume test under120seconds. Later architecture changes are also tested before completion.

## First molecular pilot:2,000 updates

Run `runs/train/20260910_203110_183370`; evaluation `runs/evaluate/20260910_203226_407589`; audit `runs/audit/20260910_203505_274504`. Seed20260910, width16, conditioning1+score2 blocks, batch8,19,249 parameters. Training44.101s (0.02205s/update); evaluation87.641s. Final stochastic minibatch loss0.316892 is not a validation score.

**This first implementation omitted the upstream conditioning network's gated readout.** It is therefore a simplified architecture pilot, not an exact reduced-width ITO reproduction. The omission was identified in a subsequent source audit and corrected in the default model. Old configurations/checkpoints remain interpretable with `condition_readout=false`; `alanine_pilot` and `alanine_cpu_fit` explicitly preserve that legacy variant. This was an implementation error discovered during review, not an intended paper choice.

| Lag(ps) | Model JSD(nats) | Matching-count reference JSD mean | Reference slowest time(ps) | Model slowest time(ps) | Model/reference ratio |
|---|---:|---:|---:|---:|---:|
|10|0.586190|0.293187|1421.940|125.372|0.08817|
|100|0.639003|0.294820|1411.104|1179.227|0.83568|
|1000|0.615277|0.293137|1325.775|8641.078|6.51776|

Each model estimate uses only4chains×64transitions. MSM uses12×12 periodic states and symmetrized counts, not the paper's Bayesian estimator. At1,000ps there is also a one-frame upper-end extrapolation from DisExp(1000), whose integer training support is1…999ps. Kinetic numbers are noisy diagnostics, with no production confidence bounds. Only the100ps point meets the requested factor-of-two screen; all three are required.

CK compares one500ps step with five100ps steps from one initial configuration: JSD **0.682317nats**,64independent branches per route. This does not satisfy the proposed0.05target and lacks basin-specific initial conditions and enough branches for a meaningful small-divergence decision.

The free-energy figure looks wrong: generated torsions are broadly scattered rather than concentrated in the reference basins. Reference-count controls already have JSD≈0.294on the64×64grid, but the model errors are substantially larger. Broad regional counts alone can pass even for scattered samples and therefore do not establish basin recovery. Whole-chain bootstrap intervals hold reference fixed and have only four chains: they are not full uncertainty estimates.

## Separate implicit-solvent OpenMM pilot

Run `runs/reference/20260910_202823_709647`:100ps production+10ps equilibration,2fs integration,1ps saving,300K,seed20260910. Walltime42.280s; throughput **9.366ns/CPU wall-hour**, including equilibration/minimization overhead in the timed region. Energies remained finite. This is a pipeline/stability result only, not reference convergence or a surrogate/OpenMM speedup comparison. It uses a different solvent/force field from MDShare.

## What still blocks Phase 1

The thermodynamic basin structure is wrong in the first pilot, two kinetic ratios miss the factor-of-two screen, and CK is both poor and undersampled. Production-quality confidence intervals, enough reference/model transitions, per-basin CK with matched controls, independent reference convergence and a source-faithful solver comparison remain outstanding. The proposed JSD≤0.10nats and CK≤0.05nats thresholds have not been relaxed.

No directly comparable scalar alanine JSD is reported by the predecessor; the project's thresholds must not be presented as paper numbers. TITO's peptide metrics are not interchangeable with these alanine diagnostics. No Phase2–4 implementation or extension result exists.

## Diffusion-output consistency diagnostic

During continued review, the denoiser was found to return `noisy + predicted_noise` even though the loss compares its result directly with Gaussian epsilon. The implementation now returns the centered epsilon prediction, and a regression test covers this contract. A fresh 500-update corrected-readout run (`runs/pilot/20260910_204755_*/`, exact timestamp in the run directory) took13.07s for training and81.69s for evaluation; its JSDs were **0.631,0.648,0.644nats**, timescale ratios **0.063,0.449,4.805**, and CK JSD **0.693nats**. This sanity run also fails. The previous pilot numbers remain valid for their committed code/configuration; the corrected implementation requires a fresh longer fit before any quantitative comparison.

The next scientifically justified experiment uses the corrected conditioning readout, then compares sampling solvers and increases generated samples before any acceptance claim. A failed pilot is retained as evidence rather than discarded.

## Completed follow-ups

| Variant | Updates | Training seconds | Evaluation seconds | JSD at10/100/1000ps(nats) | Timescale ratios | CK JSD(nats) |
|---|---:|---:|---:|---|---|---:|
| Legacy conditioning, longer training |20,000|479.389|84.149|0.612086 /0.624178 /0.633473|0.101083 /0.631293 /5.751795|0.638995|
| Corrected conditioning readout |2,000|60.320|93.676|0.623527 /0.622394 /0.617308|0.048998 /0.552225 /4.373043|0.693147|

Legacy follow-up: `runs/pilot/20260910_203505_844138`. Corrected pilot: `runs/pilot/20260910_204051_246711`,20,593 parameters. Full metrics/config snapshots are committed under `docs/results/`. These two jobs partly overlapped on CPU; their wall times are observed campaign costs, not controlled architecture-speed benchmarks. Both finished their configured step counts before the600s cap, then completed evaluation. No jobs were left running.

The machine-checkable gate command is `experiment=phase1_gates`. Its current report is `failed_or_incomplete` with `phase2_allowed=false`; this is the authoritative project status.

More training on the legacy variant did not materially improve thermodynamics. The corrected2,000-update model still fails to reproduce molecular structure; restoring that layer alone is not sufficient. None of these configurations passes Phase1, so Phase2 has not started. Further work should first validate the corrected architecture and sampling solver at molecular scale, then fund a longer corrected-model fit and sufficiently sampled evaluation. Increasing steps blindly on the legacy model is not supported by these results.

Evaluation RNG policy is now explicit: seed-based standalone evaluation, checkpoint-based named molecular pipelines. The latter restores the checkpoint's RNG before constructing the evaluation model, preserving the pipeline's original random sequence. Re-evaluation no longer depends on unrelated prior torch draws; the CPU test checks this contract. Existing standalone2,000-update results use seed mode, and the two follow-ups use checkpoint mode. These different random sequences mean the three pilots are not a paired ablation.
