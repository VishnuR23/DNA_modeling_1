# Deviations, ambiguities, and review decisions

Prepared 2026-09-10. **All implementation choices below are proposals; no phase implementation or scientific result exists.** Source abbreviations and full details are in [paper_notes.md](paper_notes.md). Scale choices and numeric gates are in [scaling_plan.md](scaling_plan.md). This log must be amended whenever an actual run differs.

## Decisions to review before Phase 1

1. **Benchmark identity:** recommended paper-comparison track uses original explicit-solvent MDShare alanine data; retain the requested ff14SB/OBC2 generation as a separate 150 ns control. Using only the requested implicit system cannot establish faithful numerical reproduction of I's explicit-water result.
2. **Primary method:** Phase 2 should implement TITO flow matching, not the diffusion objective described in the prompt. This is required by the primary source, and changes what Phase 3's “denoiser uncertainty” means.
3. **Compute scope:** approve the proposed 16/28/12-hour phase caps plus 4-hour contingency and reduced molecular/sample counts. These can fail the gates. MDQM9-nc is public; deferring its training is a budget proposal, not an access-based fallback.
4. **Dependency permission:** `h5py` would be needed for original MDQM9 HDF5 ingestion and is outside the allowed list. It has **not** been added. No extra dependency is required for the proposed peptide-only NumPy pipeline. RDKit/GAFF tooling and Triton would require separate approval if those branches are enabled; none is currently added.
5. **Provisional quantitative gates:** alanine JSD≤.10 nats, CK≤.05 nats and TITO's 3× published mean/median tolerances are new choices; the source does not provide these acceptance bounds. Freeze before training rather than interpreting “same regime” after seeing results.
6. **TITO released-code defects:** corrected permutation coupling and deterministic zero vector initialization are proposed. The latter is a material deviation from TC's per-forward random vectors; it follows deterministic equivariant-flow intent, but is not proven interchangeable with the trained published model.

## Scientific and implementation register

| ID | Difference / ambiguity | Proposed treatment and reason |
|---|---|---|
| D01 | Prompt says transferable diffusion; T uses OT flow matching | DDPM for Phase 1, CFM/Euler for Phase 2. Preserve the published distinction. |
| D02 | ff14SB/OBC2 vs I's ff99SB-ILDN/TIP3P/PME | Separate reference ensembles/models; no cross-ensemble numeric reproduction claim. |
| D03 | New implicit reference 150 ns vs original 750 ns | Fivefold reduction to fit MD allowance; retain original data length in downloaded paper track. |
| D04 | Phase 1 unspecified friction/equilibration/nonbonded details | Propose .3 ps⁻¹ from TW, 1 ns/replica burn-in, NoCutoff and H-bond constraints; declare these choices rather than inventing I values. |
| D05 | Discrete lag “set” in prompt vs I DisExp | Preserve floor(exp Uniform) and its exclusive endpoint; evaluate 1000 explicitly. No powers-of-ten training replacement. |
| D06 | Expected lag-count cut offers little step-cost saving | Preserve lag support/range; cut updates/model/sample counts instead. |
| D07 | I/ T equations print unsquared norms | Follow squared-error training in released implementations; distinguish molecule-normalized ITO and atom-normalized TITO losses. |
| D08 | IC exclusive ᾱ indexing vs solver inclusive indexing | Use consistent ᾱ_0 and 1-based noise levels; not bitwise IC behavior. |
| D09 | Mean-free paper prior vs uncentered IC noise | Project noise and vectors per molecule, following paper's translation treatment. |
| D10 | IC Fourier rank starts at 0 vs I at 1 | Paper baseline uses rank 1; scales separate for lag, diffusion and radius. |
| D11 | IC radial length 10 vs I 3 | ITO paper config uses 3; TITO code-informed config uses 10. |
| D12 | I Fig.11 Uv·Vv vs IC ||Vv||² update | Preserve/document released-code update initially. Do not silently describe it as exact diagram implementation. |
| D13 | IC cap=100 neighbors vs full graphs | Full graph per molecule at this scale; batch reduction instead of interaction truncation. |
| D14 | IC single-batch overfit and ignored CLI depth | Fresh config-driven trainer; distinct condition/score depths; remove overfit mode except explicit test. |
| D15 | IC broken EMA helper/state assumptions; sampling ignores EMA | Fresh EMA buffers, explicit restore/sampling, checkpoint round-trip. |
| D16 | IC 20-epoch cosine vs finite update budget | Step-based cosine over configured budget for ITO; record LR trace. |
| D17 | IC sampling saves scaled coordinates/no clear device policy | Explicit device, evaluation mode and nm conversion; CPU fallback tests. |
| D18 | IC analysis silently excludes NaN trajectories | Count and report all failures; never improve metrics by silent deletion. |
| D19 | TC σ=.001 absent from T interpolant | Configurable smoothing, .001 code-compatible baseline, explicitly documented. |
| D20 | TC permutation uses row instead of column assignment | Correct coupling with assignment and proper-rotation tests. |
| D21 | TC random vector features in every flow evaluation | Proposed zeros for deterministic equivariance; compatibility stochastic-input variant only if explicitly specified. |
| D22 | Prompt expects residue/molecule embeddings | Actual baseline uses elements and edges; no molecule-ID or extra residue embeddings. |
| D23 | TC has LayerNorm and edge updates absent from abbreviated T text | Follow TC details and attribute source; no TC code reuse without fresh implementation. |
| D24 | T .01 LR/batch 750 vs reduced model and TC .001 | Proposed LR .001, effective batch 128; no claim these match paper hyperparameters. |
| D25 | TITO EMA requested but absent in T/TC | Add .99 EMA and compare validation to raw weights; declare extension to training procedure. |
| D26 | Primary model cut | TITO width 48, 1+3 blocks vs 64, 2+5; ITO initially keeps paper architecture. |
| D27 | Training set cut | 64/1457 training peptides, 4.39%; no validation/test molecule subsampling. |
| D28 | Unavailable original update counts | Capped 50k ITO/30k TITO targets with honest convergence checks; report unfinished training. |
| D29 | Current Timewarp inventory vs card and T's 92 test peptides | Preserve available official split names, use 96 huge/test pairs before QC; enumerate exclusions. |
| D30 | Raw Timewarp array indices are not uniform physical times | Extract/validate 5 ps anchors; never relabel raw indices as equal-time steps. |
| D31 | TW peptides OBC1/310 K/.5 fs differ from Phase 1 OBC2/300 K/2 fs | Match TW for Phase 2/3 references and energy scoring. |
| D32 | Public MDQM9 data is obtainable | Do not invoke inaccessible-data fallback; propose deferring small-molecule training for budget/dependency scope. |
| D33 | Public 100 ns RE subset vs T's new 1 μs/ultralong references | Do not claim equivalent reference coverage or reproduce full Fig.3 campaigns. |
| D34 | New ITO FES JSD/MSM/CK thresholds absent from I | New explicit project metrics/gates, separately labeled. |
| D35 | TICA vs requested backbone torsion JSD | Compute both; compare paper numbers only to matching projection as far as reconstructable. |
| D36 | T JSD bound vs TC natural logs | Report nats and bits; retain natural logs for code-informed comparisons. |
| D37 | I and T VAMP gap signs reversed | Explicit direction in metric names; no shared ambiguous `gap`. |
| D38 | TC reference cross-score vs T singular-value formula | Report both; published figure-generation convention unresolved. |
| D39 | VAMP singular-value times vs MSM eigenvalue times | Separate estimators/metrics, same physical lags, report stationary/invalid modes. |
| D40 | Paper bin grids/features/rank settings incomplete | Predeclare 64×64 histograms and reference-only transforms; flag approximate comparability and assess bin sensitivity. |
| D41 | Nonequilibrium relaxation is not T's equilibrium invariance test | Implement both as distinct experiments. |
| D42 | T throughput arXiv `.67 s` vs journal `.67 μs` | Use corrected journal unit, measured same-device baseline for our claim. |
| D43 | Smaller GPU than T's A100 80 GB, reduced branch/chain counts | Report memory, rate, sampling uncertainty and exact solver evaluations; no borrowed speedup. |
| D44 | Penta/hexa do not double sequence length | Include one octapeptide and one heptapeptide if longer-system test proceeds. |
| D45 | Six sequences/length and 100 ns–1 μs refs exceed budget | Propose one/length, 20 ns each, uniform fixed-seed sequences rather than vertebrate frequencies; provisional structure-only conclusions if kinetics unresolved. |
| D46 | Base-size prior necessary for extrapolation | Apply (n_res/4)^.688 to std; record compaction and unstable rollouts. |
| D47 | JSD outliers can be valid novel basins | Retain requested operational label; separate energy, support and failure reasons/uncertainty. |
| D48 | Short label relaxation is not physical ground truth | 5 ps default, configurable/sensitivity checks, minimized-reference threshold matched to protocol. |
| D49 | Energy detector shares a feature with ground-truth rule | Report basin-only and energy-only label ablations; avoid overstating classifier independence. |
| D50 | TITO has a flow field, not a DDPM denoiser | Define field/reconstruction disagreement; correlated EMA variance is only a proxy. |
| D51 | CK detector computationally expensive | Budget all 1+k branches and evaluate estimator noise; detector-only kernel statistic possible with explicit validation. |
| D52 | Filtering changes dynamics and marginal measure | Keep original timestamps and censored transition accounting, report bias/retention; never join separated frames. |
| D53 | >.8 AUROC may not occur, or too few positives | Failed/inconclusive gate prevents Phase 4; still report actual values and uncertainty. |
| D54 | All-upstream package installation violates dependency constraints | Native torch graph operations; fresh training/download/EMA; no Lightning/PyG/RDKit/h5py installed. |
| D55 | Reuse licensing | Adapted IC code only under vendor with MIT+headers; additionally check DPM-Solver provenance/license; fresh EMA avoids its TensorFlow provenance ambiguity. No code vendored yet. |
| D56 | Unmeasured 60-hour feasibility | Pilot timings and per-phase caps; revise openly rather than declaring acceptance from software tests. |
| D57 | Phase 4 retraining/kernel adds cost/dependencies | No budget committed; only after all gates pass and remaining resources are known. |

## Preparatory-stage differences from the requested workflow

- Added `docs/data_availability.md` and checked-in source/dataset manifests to make access claims auditable; these support the requested notes and plan.
- Reviewed selected portions of the now-public TITO and Timewarp code in addition to all predecessor text files, because the paper alone leaves critical details ambiguous. No extra research packages were installed; temporary standard-library/native macOS utilities were used to read papers and manifests.
- Created an isolated Git repository at `tito-repro/` inside the empty workspace, matching the requested layout and avoiding the unrelated ancestor Git repository.
- Created only documentation, provenance and `.gitignore`. No model/training/config implementation, tests, production data download or simulation was started. Runtime checks will be meaningful only after implementation and the review gate.
- Missing information is listed rather than guessed: exact TITO original seeds/updates/figure settings/92-system exclusions and source-run configuration provenance remain unresolved. Dataset byte probes are not substitutes for full integrity checks.
