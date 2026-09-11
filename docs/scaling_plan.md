# Local CPU execution plan

The user's 2026-09-10 instruction replaces the GPU requirement: **all computation runs on this computer, with no CUDA, MPS, cloud jobs, or GPU prerequisites**. The previous plan is preserved in [scaling_plan_gpu_archived.md](scaling_plan_gpu_archived.md) as historical context only. Scientific acceptance gates still apply; Phase 2 cannot start before Phase 1 passes.

## Hardware and resource policy

Measured machine: Apple MacBookPro17,1, arm64 macOS, 8 CPU cores and 8 GiB unified RAM. Python 3.11 lives in `.venv`. PyTorch explicitly uses CPU; requesting CUDA or MPS fails. OpenMM accepts CPU or Reference platforms only; Reference is a CPU implementation. Start with two PyTorch threads. Public source/data downloads are the only network prerequisite; experiments operate offline after installation and ingestion.

This is a bounded CPU research effort, not a promise to reproduce GPU-scale results in the same time. Default training stops after 600 seconds and saves resumable checkpoints. No unattended multi-day process starts implicitly. Keep the initial campaign below one CPU wall-hour, then reassess measured results. There is no fixed conversion from the old 60 GPU-hours.

## Phase 1: current implementation and initial budget

| Work | Initial cap/estimate | Scope |
|---|---|---|
| Setup and public download | 10 minutes, network dependent | Python 3.11; original three trajectories, about 129 MB raw / 198 MB processed |
| Software validation | tiny train/sample under 2 minutes | proper rotations, centering, boundary-safe sampling, schedule, exact resume, GPU rejection |
| Implicit OpenMM calibration | 2 minutes | 100 ps production plus 10 ps equilibration, 1 ps saving |
| Molecular training pilot | 10 minutes maximum | width 16, 1+2 blocks, batch 8, 2,000 updates |
| Three-lag evaluation and CK pilot | 10 minutes estimate | four chains ×64 transitions/lag; 50 DDIM evaluations; 64 CK branches |
| Fixes and reporting | remaining initial hour | repeat affected checks only |

Actual timings go in the local status report. Training estimate: updates × measured seconds/update. Sampling estimate: transitions × solver evaluations × measured seconds/model call. Reference estimate: simulated ns / measured ns per CPU-hour. Synthetic timing does not predict 22-atom performance. Dense edges scale quadratically with atom count; 8 GiB memory excludes casually scaling peptide batches.

Width **16 versus 64**, **1+2 versus 2+5 blocks**, batch **8 versus 128**. The native torch vendor adaptation avoids torch-scatter and Lightning dependencies. Preserve 1,000 diffusion levels and DisExp(1000): reducing their range does not substantially reduce a training update's cost. Coordinates use source scale 0.1661689 nm. Initial sampling is **DDIM, eta=0, with fresh Gaussian noise per physical transition**, 50 evaluations, rather than DPM-Solver. This is a numerical-method deviation. EMA is actually used and schedule indexing is corrected.

The full original MDShare explicit-solvent ensemble is the paper-comparison data. Its XTC timestamps repeat every 1,000 frames; validate the exact pattern and reconstruct continuous time using the provider's documented 1 ps spacing. Keep replica boundaries. Center already superimposed coordinates; do not mix ensembles.

The requested ff14SB/OBC2, 300 K, 2 fs system is a separate **implicit-solvent control**, with HBond constraints, NoCutoff, friction 0.3 ps^-1 and Reference CPU platform. Its initial 100 ps is a pipeline/stability measurement, **7,500 times shorter** than the predecessor's aggregate 750 ns and 1,500 times shorter than the previous 150 ns control proposal. Ten ps equilibration is also a reduction from the proposed 1 ns. It cannot establish reference convergence. At an assumed 10 ns/CPU-hour, 150 ns takes 15 hours; use measured timing before extension.

## Scientific gates and diagnostic limits

Project thresholds proposed before observing results, not paper-reported cutoffs: phi/psi JSD ≤0.10 **nats**, recovery of three main basins, slowest timescale ratio in [0.5,2] at 10/100/1000 ps, nested/direct CK JSD ≤0.05 nats. Do not loosen them after seeing results. The paper does not supply a directly comparable single alanine JSD number; see paper notes.

The current fixed-grid MSM uses symmetrized counts, not the paper's Bayesian MSM. Disconnected/undersampled graphs are inconclusive. Tiny histograms and one CK starting configuration cannot determine acceptance. Production acceptance still requires confidence intervals, per-basin CK with matched finite-sample controls, adequate transitions and independent reference convergence. Failure/incomplete evaluation blocks Phase 2 even when software tests pass.

## Later phases: conditional CPU feasibility

No Phase 2–4 implementation starts now. Before each phase, recalibrate locally and update its plan. Preliminary Phase 2 allowance: 30 CPU minutes for variable-size calibration, 2 CPU hours for training pilot, 2 CPU hours for evaluation calibration. These are **future estimates, not launched jobs or sufficient-training claims**. Candidate width 16, 1+2 blocks, physical batch 1–2, 16 seeded training peptides out of 1,457 (1.10%), preserving validation/test inventories. This replaces the old 64-molecule proposal. Full validation/test transfer is roughly 762 GB raw and may be impractical locally; do not quietly subsample those splits. If infeasible, Phase 2 remains incomplete. TITO requires flow matching, not an unchanged DDPM.

MDQM9-nc is public but outside the current peptide-first numbered phases. No 44.63 GB HDF5 download or new chemistry dependencies are implicit. No DESRES data is used.

Phase 3 estimates must use measured CPU rates: 4,096 labels ×5 ps requires 20.48 ns plus minimization; the previous CK detector proposal requires 393,216 generated transitions. Plan a 30-minute calibration before choosing a bounded campaign. No honest full runtime estimate exists before Phase 2. AUROC >0.8 and filtering-bias evaluation remain requirements. Replace same-GPU throughput with **same-computer CPU OpenMM versus CPU surrogate**, in ns/CPU wall-hour; never call this GPU throughput. Phase 4 can consider data efficiency after all earlier gates pass. Triton is incompatible with the no-GPU instruction and is not installed.

Every launch writes resolved config, seed/device/environment, log, metrics JSON and figures under `runs/`. Training adds resumable optimizer/scheduler/EMA/RNG state; sampling saves coordinates and histograms. Runs are gitignored; committed status reports preserve results. Resume must preserve data, training, seed and thread semantics. No notebooks are imported.

## Calibration update before the follow-up

The first molecular pilot measured 0.02205 s/update (2,000 in44.10s), sampling/evaluation87.64s, and implicit OpenMM9.366ns/CPU-hour. A single20,000-update follow-up is therefore estimated at7.35min training plus1.5min evaluation, inside the initial campaign hour. `alanine_cpu_fit` starts a fresh cosine schedule with the same600s training cap. The failed2,000-update results remain preserved; thresholds and sample counts are unchanged. At the measured reference rate,150ns of implicit production alone would take about16hours, excluding extended equilibration.
