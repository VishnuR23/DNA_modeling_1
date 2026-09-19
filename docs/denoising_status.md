# Phase 1 denoising diagnostic — 2026-09-18

The new diagnostic measures epsilon prediction on forward-noised cached reference configurations. These are training-source pairs, not held-out molecular validation. It does not measure generated trajectories, equilibrium, or kinetic agreement.

First, the existing 500-update no-skip checkpoint showed epsilon MSE 0.195 at noise level 1,000, versus 0.0343 for the identity prediction `epsilon = noisy`. The schedule ends at alpha-bar 0.01134. Algebraically, reconstruction multiplies squared epsilon error by `(1-alpha_bar)/alpha_bar`, about 87.15 at that level. Its reconstruction error was 16.99 in squared dimensionless coordinate units (Cartesian-sum/atom-mean). This motivated a paired parameterization experiment.

Correction to the historical diagnosis: `epsilon = noisy + learned_residual` is a valid epsilon parameterization if the loss supervises the full output against noise, as this implementation does. Removing the skip is not inherently a consistency fix. `model.epsilon_skip=false` preserves the current default; `true` explicitly enables the residual variant. The flow prototype uses no skip.

Both paired runs used seed 20260910, 1,000 updates, the corrected conditioning readout, identical size/batch/data/schedule, and fresh initialization. Both finished within their 60-second training caps: 24.27 seconds plain, 27.89 seconds residual. Diagnostic inputs use identical 64 source pairs and the same noise. The timing difference is not a controlled throughput claim.

| Noise level | Plain epsilon MSE | Residual epsilon MSE | Identity baseline MSE |
|---|---:|---:|---:|
| 1 | 2.9644 | 2.9311 | 5.7703 |
| 100 | 1.5742 | 1.5203 | 4.7308 |
| 500 | 0.4862 | 0.4687 | 2.1401 |
| 1,000 | 0.05494 | 0.01806 | 0.03433 |

The residual variant reduces terminal epsilon MSE by about 67%, and terminal reconstruction error from 4.788 to 1.574. Both variants are slightly worse than the zero-epsilon baseline (2.7763) at the lowest noise level. These single-seed diagnostics support further residual-model sampling experiments, not molecular acceptance. Neither original thresholds nor saved failed gates changed.

Run the paired experiment with cached alanine data:

```sh
.venv/bin/python scripts/compare_epsilon.py
```

Diagnose a specific checkpoint:

```sh
.venv/bin/python -m tito_repro.cli experiment=alanine_denoising diagnostic.checkpoint=<local-checkpoint.pt>
```

The paired metrics, configs, source hashes and checkpoint hashes are committed under `docs/results/epsilon_*1000*`. Original artifacts are under `runs/epsilon_ablation/20260918_164803_442185`. Checkpoints from before the historical skip removal must be interpreted with their original code revision; missing `epsilon_skip` currently defaults to false and cannot identify the architecture of every historical checkpoint.

## Paired generated-sample follow-up

The two 1,000-update checkpoints were sampled with identical initial configurations and reset Gaussian RNG streams at each lag. Each variant generated 4 chains × 32 transitions at 10, 100 and 1,000 ps with 50 DDIM evaluations per transition. The sampling/comparison stage took 57.44 seconds locally, excluding checkpoint/data setup. No new training or downloads occurred.

| Lag (ps) | Bins per angle | Plain JSD | Residual JSD | Matched-count reference JSD mean |
|---|---:|---:|---:|---:|
| 10 | 32 | 0.627 | 0.641 | 0.200 |
| 100 | 32 | 0.596 | 0.608 | 0.202 |
| 1,000 | 32 | 0.615 | 0.617 | 0.200 |
| 10 | 64 | 0.666 | 0.672 | 0.395 |
| 100 | 64 | 0.650 | 0.655 | 0.394 |
| 1,000 | 64 | 0.662 | 0.661 | 0.395 |

All JSD values are in nats. Every 95% paired whole-chain bootstrap interval for residual-minus-plain JSD includes zero (200 replicates; reference histogram fixed). Four chains are insufficient for strong uncertainty claims. IID reference controls measure finite-histogram effects, not temporal uncertainty. The 1,000-frame lag remains a boundary extrapolation from DisExp training support 1…999.

The lower high-noise prediction error did **not** produce clear torsion-distribution improvement. Both variants remain much worse than matched-count reference controls. The default stays unchanged. This test does not measure bond validity, energy, kinetic or CK agreement, so no acceptance gate is passed.

```sh
.venv/bin/python -m tito_repro.cli experiment=epsilon_sampling \
  comparison.checkpoints.plain=<plain-checkpoint.pt> \
  comparison.checkpoints.residual=<residual-checkpoint.pt>
```

Metrics/config/environment snapshots are `docs/results/epsilon_sampling_1000*`; raw trajectories are in `runs/epsilon_sampling/20260918_184641_659602`. Next molecular work should diagnose generated bond geometry and short-lag conditional behavior before increasing training blindly. Exploratory Phase 2 continues independently.
