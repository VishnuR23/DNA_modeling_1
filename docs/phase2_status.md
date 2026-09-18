# Exploratory Phase 2 status — 2026-09-18

Phase 2 development has started under the revised [development policy](development_policy.md). Phase 1 scientific acceptance remains failed/incomplete. Everything below ran locally on CPU with existing dependencies and no downloaded data.

The prototype learns a velocity field along `x(t) = (1-t) z + t y`, where `z` is a centered Gaussian and `y` is the target. The supervised velocity is `y-z`; sampling integrates from 0 to 1 using Heun's method. This is a basic conditional flow construction in the [flow matching framework](https://arxiv.org/abs/2210.02747), with independent endpoint pairing. It does not implement TITO's optimal-transport alignment, chemical bond features, or full architecture. Flow time is dimensionless and is separate from the lag between synthetic frames.

One set of model parameters handles different atom counts. Element-number embeddings replace fixed atom-index identities for this model only. Each batch has a single size/topology; batches alternate sizes and lags without padding. Element labels on Gaussian point clouds are synthetic test inputs, not chemical structures. Phase 1 keeps its original atom-index interface.

The first pilot trained 2,000 updates on 3- and 4-atom centered Gaussian AR(1) pairs, correlation 0.8 per frame, lags 1 and 4, batch 16, seed 20260910. Training took 7.95 seconds with two CPU threads. Evaluation used 128 independent pairs per size/lag, fixed before/after inputs, and 20 Heun steps (40 field evaluations). Five-atom inputs were excluded from training.

| Atoms | Lag (frames) | Velocity MSE before → after | Sample residual second moment | Analytic conditional variance |
|---|---:|---:|---:|---:|
| 3 | 1 | 4.104 → 1.971 | 0.246 | 0.240 |
| 3 | 4 | 4.156 → 2.924 | 0.566 | 0.555 |
| 4 | 1 | 4.555 → 2.323 | 0.250 | 0.270 |
| 4 | 4 | 4.443 → 3.092 | 0.578 | 0.624 |
| 5 (held-out size) | 1 | 4.770 → 2.178 | 0.223 | 0.288 |
| 5 (held-out size) | 4 | 4.663 → 3.278 | 0.567 | 0.666 |

Velocity MSE sums Cartesian error and averages over atoms and examples. Residual second moment averages squared coordinate residuals around the known conditional mean; it includes bias and is not a complete distribution test. The analytic per-coordinate variance includes the centering factor `(1-1/atoms)`. The held-out size underestimates the target second moment by approximately 22% and 15%, respectively. No uncertainty intervals or acceptance threshold were applied. Finite, centered outputs and lower loss show a working prototype, not molecular transferability.

Reproduce from the repository root:

```sh
.venv/bin/python -m tito_repro.cli experiment=phase2_synthetic
```

Training is limited to 2,000 updates or 60 seconds, checked between updates. Setup and bounded evaluation are outside that training timer. Periodic checkpoints and loss arrays are saved, but this prototype has no resume command or EMA yet. Results, config and checkpoint provenance are committed under `docs/results/phase2_synthetic_2000*`; the original artifacts are under `runs/phase2_synthetic/20260918_164507_673450`.

Next useful work is to measure calibration across more seeds and sizes, then add validated topology/bond features and a cached-data molecular pilot. A missing peptide dataset does not block synthetic development. Molecular reproduction, optimal-transport matching and peptide transferability remain unestablished.
