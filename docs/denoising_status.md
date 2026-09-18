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

Next: compare generated distributions using matched seeds and solver budgets, retain finite-sample controls, and expand only if improvements survive those checks. Exploratory Phase 2 development proceeds independently under the revised policy.
