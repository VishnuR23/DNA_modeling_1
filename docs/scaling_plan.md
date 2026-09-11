# Reduced-scale execution plan: 60 GPU-hours maximum

**Status: proposed for review, no phase started.** Estimates below are planning assumptions, not measured runtime or promised accuracy. This workspace currently uses macOS with a default Python 3.13 interpreter; PyTorch/OpenMM are absent from that interpreter. A Python 3.11 environment and an actual 16–24 GB CUDA device remain execution prerequisites. We have not reserved or purchased remote compute. CPU fallback applies to functional tests, not a promise to finish scientific training on CPU.

The source review took **0 GPU-hours**. Preserve the user's ordering: Phase 1 must pass before any Phase 2 implementation/training, Phase 2 before Phase 3, and all three before Phase 4. At every boundary stop with results, failures, and an updated deviation list. Spending the allocated time does not constitute passing. An unmet gate is reported as failed or inconclusive and later phases do not start.

## Budget and calibration

| Work | GPU-hour cap | Proposed allocation |
|---|---:|---|
| Phase 1 | **16** | reference OpenMM 3; main ITO training 7; implicit-system control training 2; sampling/evaluation 3; calibration 1 |
| Phase 2 | **28** | TITO training 12; longer-peptide reference MD 4; all-set sampling/evaluation 10; solver/OpenMM benchmarking 2 |
| Phase 3 | **12** | labels 3; detectors/CK branches 5; filtering and repeated metrics 3; timing 1 |
| Contingency | **4** | failed jobs, reference extension, or uncertainty checks; not an automatic hyperparameter sweep |
| Total | **60** | all GPU activity, including failed runs and pilots, counts |
| Phase 4 | **0 committed** | only unused time after all preceding gates pass |

Download, parsing, plotting, and CPU tests use wall/CPU time in addition to this budget. Stop if a phase is forecast to exceed its cap; publish the revised estimate before consuming the contingency. Do not lower an acceptance threshold after observing outcomes.

Before scientific training in each phase, spend its calibration allowance measuring at least 200 warm-up and 500 timed training steps, representative inference batches and peak allocated/reserved CUDA memory, and unbiased OpenMM integration. Synchronize timing. Record exact GPU, driver, CUDA/PyTorch versions, precision, atom count distribution, batch and edge count. The longest molecule—not the average—determines the memory test. Use CPU numerical/equivariance checks first.

Runtime equations:

    training_hours = updates × measured_seconds_per_update / 3600
    sampling_hours = transitions / measured_transitions_per_second / 3600
    detector_CK_transitions = scored_states × branches × (1 + nested_steps)
    reference_hours = total_simulated_ns / measured_OpenMM_ns_per_hour

At .1–.5 s/update, 50,000 ITO updates cost 1.39–6.94 hours; at .3–1.4 s/update, 30,000 peptide updates cost 2.5–11.67 hours. These are deliberately broad assumed ranges, not measurements. The source paper's 2–4 days to converge makes a 7-hour ITO allocation a serious risk. Early training loss convergence alone cannot approve Phase 2.

## Memory and data reductions

The published ITO already fits approximately 12 GB GPUs at width 64, 2+5 blocks and batch 128. Preserve that architecture initially to reduce scientific confounding. A config-selectable width 48, 1+3-block fallback is allowed only if the pilot forecasts failure of the budget; its use must be recorded before the final run. TITO starts at **48 channels, 1 condition + 3 velocity blocks**, versus 64 and 2+5. Parameter-dominated operations scale roughly with blocks×width²; the ratio (4/7)(48/64)²≈.32 is only a rough operation-count estimate, not a runtime speedup claim.

For B molecules of A atoms, E≈BA(A−1). Float32 storage of a single [E,F,3] vector activation costs 12EF bytes. At B=32, A=100, F=48, that is ~182 MB for one such activation. Gates, gradients and multiple blocks multiply this substantially; start physical batch **16** for TITO, benchmark **32**, then gradient-accumulate to effective batch **128**. Reduce the physical batch to 8 for longer molecules if needed. Do not use an arbitrary neighbor cap to hide quadratic memory. Native torch masked complete edges retain the paper's interactions. CPU tests use 3–5 atoms, batch 2, width 8, one block, not padded peptide batches.

**Retain the lag distributions and ranges.** ITO: DisExp(1000); TITO: uniform over 5,…,5000 ps. Drawing one random lag per pair has essentially the same step cost as drawing from a smaller list. Cutting lag coverage would weaken the central reproduction more than it saves compute. Reduce updates, width/depth, training molecules, and generated sample counts first. Keep ITO's 1000 noise levels; training samples one level per example, and inference uses 50 solver evaluations. TITO uses 40 Euler evaluations by default, with 20/40/80 calibration on validation only; use 100 for the paper's larger-peptide single-step comparison.

TITO training subset: **64 of 1457** complete large/train peptides (4.39%, a 95.61% reduction), chosen by a fixed seed from the sorted inventory before observing dynamics/model errors. Use the full available trajectory per selected training molecule on the regular 5 ps grid, targeting the stated 50 ns rather than shortening individual trajectories. Proposed fallback is 32 training molecules only if ingestion/training is infeasible; record it as another scale reduction, not an equivalent reproduction. Validation and test molecule sets are never subsampled.

At the live inventory's average training file size, 64 raw training NPZs require ~6.9 GB. All available large validation, large test, and huge test files total ~762 GB; together with training this is ~769 GB of network transfer if every file is fetched. At 10–100 MB/s sustained, transfer alone is roughly 21.4–2.1 hours. These rates are assumptions. Stream source files one at a time, preserve hashes, and keep compact position/time arrays plus topology instead of every raw velocity/force array. Avoid retaining a terabyte of unneeded raw files. Estimate exact compact size from shape metadata during ingestion; float32 coordinate storage costs frames×atoms×12 bytes. Source deletion after verified conversion must be an explicit cache policy, never deletion of the only unverified copy.

MDQM9-nc is public. **Proposed scope reduction for review:** retain its provenance and a future loader configuration, but spend no core training GPU budget on small-molecule models. The user's numbered Phases 2–3 target peptides; adding a second transferable training campaign, GAFF setup and reference simulations threatens the 60-hour budget. This is not an access-based drop. Enabling this branch requires revising the budget and approving HDF5/chemistry dependencies; it cannot consume time reserved for unpassed peptide phases.

## Phase 1 plan and acceptance contract

Before coding, put a short phase plan comment at the top of the future `src/tito_repro/train/phase1.py` entry module. That module and its configs do not exist yet.

**Proposed benchmark resolution requiring review:** use the three original 250 ns MDShare trajectories for the paper-comparison ITO model. Separately generate the requested ff14SB/OBC2/300 K/2 fs OpenMM alanine reference as a changed-force-field control: start with **three 50 ns replicas**, saved every 1 ps (150 ns, fivefold shorter than the paper). Train a separate small control ITO only within its 2-hour allowance; never mix explicit and implicit trajectories in one single-molecule model or compare implicit dynamics numerically to the explicit reference as if identical. If the user prefers only the requested implicit system, rename the outcome “ITO reimplementation on a different alanine ensemble,” not faithful reproduction of the 2023 alanine result.

For the requested 2 fs implicit simulation, propose hydrogen-bond constraints, Langevin integration with friction .3 ps⁻¹ (sourced from TW but not recovered from I), 1 ns equilibration per replica, NoCutoff for this small nonperiodic molecule, `amber14-all.xml` and `implicit/obc2.xml`. These are project choices where the prompt/I do not specify details; validate topology/energy stability and report them. No folding/basin convergence is assumed from 150 ns. The 153 ns including burn-in takes 3 hours only if the measured rate is ≥51 ns/hour; otherwise shorten nothing silently—revise the proposal at the phase checkpoint.

For the original MDShare track, match I's all-data training for the main comparison but identify that its reference is in-sample. Add trajectory-block holdout diagnostics with at least a max-lag guard between train and validation if used for selection, explicitly labeling any model trained on fewer trajectories. The independent implicit trajectories are not holdout data for the explicit ensemble. All seeds, scales and boundaries are recorded.

Training target: up to 50,000 updates for the main model, 7 hours maximum, Adam .001, 128 batch, consistent sigmoid DDPM schedule, EMA .99 with warm-up. Proposed LR policy is step-based cosine decay over the configured update budget, rather than IC's 20-epoch cycle; log this deviation. Use raw and EMA checkpoint state; evaluate EMA by default. A brief fixed-lag=100 ps control can use the contingency only if multi-lag acceptance is already credible; lack of multi-lag superiority does not fail reproduction.

Sampling target: 32 independent chains×1000 transitions at each of **10,100,1000 ps**, plus ≥4096 terminal branches per initial basin for CK. Use nested k=5 at L=10,100,200 ps against direct 50,500,1000 ps. All within the conditioning range except the literal DisExp endpoint nuance, which is disclosed. I's 15,000 branches and 4/64/512-step figure are reduced; these comparisons test the same principle without pretending to reproduce the exact Monte Carlo campaign. Initial conditions and chain burn-in must be stated.

Proposed gates, to freeze before training:

- **Thermodynamics:** periodic 64×64 phi/psi histogram, all three reference main basins recovered, with `jsd_nats≤.10` and matched-count MD-versus-MD split JSD also reported. Basin regions are fixed from reference data before scoring the model; report each basin mass and minimum location. This .10 threshold is a new engineering criterion, not a number from I's alanine figure. The original request's “comparable to figure” is assessed visually alongside this number and cannot be claimed as a numerical equality to an unreported paper JSD.
- **Kinetics:** use a fixed reference-derived phi/psi discretization/MSM; the slowest identifiable nonstationary timescale has model/reference ratio in [.5,2] at all three physical evaluation lags 10,100,1000 ps. Report 95% trajectory-block bootstrap intervals, transition counts and effective rank; insufficient rare transitions makes the gate **inconclusive**, not passed. Also report I's four-feature VAMP-2 gap.
- **CK:** direct versus nested terminal JSD ≤.05 nats for each prespecified initial basin and each total lag, with terminal-sample bootstrap intervals and a direct-versus-direct finite-sample control. If the control alone exceeds the cutoff, increase samples within the cap or mark inconclusive; do not relax the cutoff.
- **Software:** denoiser proper-rotation equivariance, translation handling, trajectory boundary-safe lag-pair shapes, schedule consistency, checkpoint/EMA round-trip, and tiny CPU synthetic train+sample under two minutes. Functional tests cannot substitute for the three scientific gates.

There is no empirical guarantee that 50,000 updates or the reduced reference reaches these gates. If it does not, stop after Phase 1 with the honest gap and a proposed revision; do not implement transfer early to appear productive.

## Phase 2 plan and acceptance contract

Main module comment will describe the phase before code is written. Begin only after Phase 1 passes and review authorizes continuing.

Use element/bond-conditioned **flow matching**, correcting the documented OT indexing and random-vector issues explicitly. Initial reduced settings: width 48, 1+3 blocks, physical batch 16–32, effective batch 128, 30,000 updates or 12 hours. Use Adam .001, matching TC rather than T's .01 with batch 750, because batch and model size change substantially; this is a deliberate hyperparameter deviation. No automatic learning-rate sweep. Adopt σ=.001 smoothing. Save raw weights and EMA=.99 as a project addition; compare on validation, fix the sampling choice before testing.

Preserve the 368 available validation and 164+96 test molecular inventories. Compute validation loss on a fixed seeded equal number of pairs per peptide; reference trajectories themselves are not subsampled for storage/preprocessing apart from the specified 5 ps grid. Start final sampling at 16 chains×500 transitions per peptide at 250 ps, with 40 Euler steps. Include the 64 training peptides, 164 large test peptides and 96 huge test peptides in result tables; do not report overlapping large/huge test sequences as independent systems. Compute all molecular thermodynamic summaries; evaluate additional kinetic lags through frame separation with the original timestamps.

For conditional invariance/relaxation and CK, use a prespecified initial-state protocol for each peptide, with 256–1024 terminal branches depending on timed cost. Do not use a tiny branch count as conclusive evidence for a low JSD; attach sampling error. The 10-hour sampling cap must be checked against actual counts. For example, 324 peptides×16×500 = 2,592,000 transitions before additional CK/invariance work; that portion alone takes 3.6–14.4 hours at assumed 200–50 transitions/s. A measured slow rate makes the proposed full evaluation infeasible and must trigger a review, not removal of difficult peptides.

Longer-peptide initial proposal: **one prespecified sequence each at lengths 5,6,7,8**, using fixed-seed canonical residue draws (uniform sampling is a departure from vertebrate frequencies) and the exact TW force field. Generate **20 ns per sequence**, aggregate 80 ns; at J's A100 peptide baseline 27.9 ns/hour this is ~2.9 hours, but longer molecules may take more. Allocate 4 hours and benchmark first. This is 5–50× shorter per system than T's conflicting 100 ns/1 μs statements, and sixfold fewer sequences. It can support only provisional structural comparisons; require adequate observed transitions before quoting kinetic agreement. Include an octapeptide to test doubled length, and show any instability openly. Rescale Gaussian std by (n_res/4)^.688. Retain the stronger reference-extension option using remaining contingency if it changes a conclusion.

Proposed “same regime” gate: main huge-test mean TICA JSD ≤.126 and median ≤.108 nats (3× T's .042/.036); mean top-10 discrepancy ≤3.612 and median ≤1.302 (3× T's 1.204/.434). The multiplier is a **project tolerance for review**, not an established definition of reproduction. Publish raw and relative gaps, per-peptide ranks/failures, quantiles and molecule-bootstrap intervals. Every requested paper metric and per-peptide table must exist; scores on a different projection/binning must be labeled approximately comparable. Non-estimable cases stay in the table and prevent an unconditional pass if they conceal missing kinetics. The same-GPU OpenMM throughput benchmark and figure-structure reproductions are mandatory. No maximum speedup target is invented.

## Phase 3 plan and acceptance contract

No detector code before Phase 2 passes. Write the plan comment in `eval/hallucination.py` before implementation.

Proposed initial label budget: 4096 presampled configurations balanced across training peptides, large/huge held-out peptide groups and the four longer systems, plus separate reference configurations and a validation calibration set. Stratification is fixed before detector scores; random within strata, no selecting only high-energy examples to inflate AUROC. Use 5 ps relaxation, a configured light minimization and a configured stronger label minimization, with independent seeds. Reference-energy threshold proposal: 99th percentile of identically minimized reference energy plus a configurable 5RT margin. Both choices are project definitions, with sensitivity at 1RT/10RT and 2/5/10 ps if budget permits. Do not optimize labels to reach AUROC>.8. Cache minimization outcomes, relaxation traces, terminal basin labels and failures.

4096×5 ps = 20.48 ns integrated dynamics, excluding reference/calibration replicas and minimization overhead. At 27.9 ns/hour the propagation lower-order estimate is .73 hours; three hours allows context creation, minimization and reference-label work. This is a planning lower bound, particularly for larger peptides.

Detectors:

- Raw and lightly minimized force-field energies, per-molecule z-scored against an independent pool of model samples; include energy drop and state precisely how combined scores are defined.
- CK branch divergence at total L=250 ps versus five 50 ps transitions, initially 16 independent branches each. 4096×16×6 = **393,216 physical transitions**, with each requiring 40 field evaluations. Estimate this cost from Phase 2; at 50–200 transitions/s this is .55–2.18 hours. Sixteen branches may be noisy: increase to 64 on a validation subset to quantify stability, or use a preregistered kernel two-sample statistic on sin/cos torsion features rather than a mostly empty 2D histogram. This metric choice is a detector-specific deviation, not a change to the reproduction CK test.
- Flow-field uncertainty: disagreement of predicted endpoint reconstructions across 4 flow levels/4 noise draws or 3 nearby EMA checkpoints. Do not train a new independent ensemble within this budget. Label correlated-checkpoint variance as a proxy.

Calibrate score orientation/thresholds on validation molecules, freeze all choices, then evaluate held-out molecules. Report per-group prevalence, class counts, AUROC, precision at target validation FPR .01 and achieved held-out FPR, plus molecule-bootstrap uncertainty. A low hallucination prevalence could leave too few positives; expand labels only within budget, otherwise report inconclusive. Report total seconds/sample including expensive rollout branches and preprocessing, not just arithmetic on already computed features.

Choose the detector using validation only; rerun Phase 2 histograms and timestamp-preserving kinetic estimates after filtering, with retention and selection-bias diagnostics. If low-reference-support labels dominate, include the basin-only and energy-only label ablations; do not claim proof of physical invalidity from a finite reference.

Gate: held-out AUROC **>.8** for at least one detector, honestly measured; single-command quality/cost and filtering figures; final README with our measured numbers next to the paper, headline figure, complete reproduction commands and specific limitations. Reporting an AUROC below .8 fulfills honest reporting but **does not pass** this gate. Phase 4 cannot start after a failure.

## Phase 4 and experiment contracts

Prefer the data-efficiency curve if there is time left after all earlier gates pass: 10/30/100% of the **reduced training trajectory time**, with whole chronological blocks and boundary guards, comparing held-out kinetic error to an MSM fitted on the same data budget. Clarify that a direct MSM cannot transfer across different peptide topologies in the same way as TITO; per-held-out reference-data MSM curves require a separately stated data-access protocol. Three retrainings likely cost 6–24 extra GPU-hours and are not funded in the current cap. Triton would be a new dependency requiring permission and is not silently installed as a stretch task.

Every future experiment uses Hydra composition from `configs/experiment/` and writes to a unique `runs/<experiment>/<seed>/<run_id>/`: resolved config; seed/RNG states; source revision and dirty-state marker; environment/device metadata; input/split/scaling hashes; log; checkpoints with optimizer/scheduler/EMA state; metrics JSON; figures and their plotted numerical arrays. Resume must restore RNG/data position. Errors and undefined estimates have explicit status/reason fields, not fabricated numeric zeros. Public docstrings state physical units and tensor shapes. No notebooks are imported.

Figure command names will be written only when implemented and verified. This preparation does not provide placeholder commands that falsely imply a working pipeline. Later commands should reproduce figures from saved runs separately from optionally rerunning MD/training, so their actual compute cost is clear.
