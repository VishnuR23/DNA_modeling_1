# Source review and implementation specification

Prepared 2026-09-10. This is a pre-implementation specification, not an experimental report. “Paper” values, “released code” values, and “proposed” choices are deliberately distinguished. No simulation, model training, or physical evaluation has run.

## Sources and reading scope

- **I:** Schreiner, Winther, Olsson, [ITO, arXiv:2305.18046v2](https://arxiv.org/html/2305.18046v2), 28 October 2023; [23-page PDF](https://arxiv.org/pdf/2305.18046v2). Read the complete paper and appendices A–G, including the dataset, architecture, training, compute, and VAMP sections.
- **IC:** [olsson-group/ito](https://github.com/olsson-group/ito/tree/8310311250e0e3893bc10bbd80ab67d2eab4ae6a), commit `8310311250e0e3893bc10bbd80ab67d2eab4ae6a`. Read all 18 tracked text files (including empty initializers, README, license, build files, and the complete 1,318-line DPM solver); the remaining tracked GIF is a presentation asset, not a training artifact. No historical commits or unavailable training artifacts are asserted to have been reviewed.
- **T:** Diez, Schreiner, Olsson, [TITO, arXiv:2510.07589v1](https://arxiv.org/html/2510.07589v1), 8 October 2025; [42-page PDF including supplement](https://arxiv.org/pdf/2510.07589v1). Read the complete paper and supplement. Figure 2 numbers below were checked in the rendered PDF, not estimated from histogram heights.
- **J:** [Science Advances 12(15), eaed2333](https://pmc.ncbi.nlm.nih.gov/articles/PMC13060594/), published 8 April 2026. Cross-checked publication identity, availability, and the corrected throughput unit. This is not a claim to have audited every change in the journal supplement; comparisons below are pinned to the requested arXiv version unless explicitly marked J.
- **TC:** The primary method's code is now public: [olsson-group/tito](https://github.com/olsson-group/tito/tree/97949de311caa6621b17b5b5f0320da78b39e26b), commit `97949de311caa6621b17b5b5f0320da78b39e26b`. Read relevant training, flow, velocity, ChiroPaiNN, embedding, lag/OT, preprocessing, and evaluation code to resolve omissions in T. No TC code has been copied into this project.
- **TW:** [microsoft/timewarp](https://github.com/microsoft/timewarp/tree/df25511dda579b8e73bcc5f8f8754a9b233f6c8e), commit `df25511dda579b8e73bcc5f8f8754a9b233f6c8e`; dataset card, simulation presets, simulation job configurations, and dataset inventory definitions checked.
- **P:** [Protein Language Model Embeddings Improve Generalization of Implicit Transfer Operators, arXiv:2602.11216](https://arxiv.org/html/2602.11216); fetched PDF, skimmed method, main results, and architecture/training appendices. This is a different, coarse-grained extension, not the all-atom TITO baseline.

PDF SHA-256 hashes and source revision information are in [sources/review_manifest.json](sources/review_manifest.json). Research downloads were kept outside the project in `/tmp/tito-source-review`; only provenance and dataset manifests are committed. No DESRES trajectory data was requested or downloaded.

## What must change in the prompt's scientific description

1. ITO uses a conditional DDPM; TITO uses **equivariant optimal-transport conditional flow matching**, not the same diffusion loss with molecule embeddings added.
2. Neither objective explicitly penalizes CK violations or imposes exact Boltzmann invariance. Both properties are empirical assessments of a learned surrogate. Positions alone marginalize velocities and sometimes solvent, so exact Markovianity is not assured at every lag.
3. ITO's alanine reference is explicitly solvated ff99SB-ILDN, not ff14SB/OBC2. The requested OpenMM experiment is scientifically useful but targets a different transition kernel and equilibrium distribution.
4. ITO does not report an alanine histogram JSD target or an alanine factor-of-two MSM timescale gate. These will be new quantitative criteria, not quoted paper results.
5. High JSD does not establish hallucination. T investigates outliers with replica exchange and explicit trajectories and finds previously unsampled, physically metastable states.
6. Pentapeptides and hexapeptides test 1.25× and 1.5× sequence length. Testing “twice as large” requires octapeptides and atom counts must also be reported.

## ITO objective, sampling, and conditioning

Let x∈R^(A×3) contain the solute positions, τ_save=1 ps for alanine, and Δ=Nτ_save. Center each configuration and apply a single scalar coordinate normalization. Choose a trajectory start uniformly from the union of trajectories with the last N_max frames excluded. Never form pairs across trajectory boundaries.

I Algorithm 3 draws u∼Uniform(0, ln N_max), then N=floor(exp u). Thus for N_max=1000 the stochastic support is 1,…,999 (the endpoint 1000 has probability zero), with P(N=n)=ln((n+1)/n)/ln(1000). This is not a discrete set of powers of ten. IC implements this literally. Fixed-lag controls explicitly use N=N_max. The max-lag=1 case needs an explicit branch. I's alanine VAMP plots evaluate N=1,10,100,1000; its Müller–Brown fixed-lag experiments use 10,100,1000.

For a dimensionless target y, standard DDPM notation is

    α_d = 1 − β_d
    ᾱ_d = product_{j=1}^d α_j,  ᾱ_0 = 1
    ε ∼ N(0,I),  y_d = sqrt(ᾱ_d)y + sqrt(1−ᾱ_d)ε
    L = E[ (1/A) sum_a ||ε_a − εθ(y_d, x_condition, N, d)_a||² ]

IC uses this atom-mean / Cartesian-sum MSE (then averages molecules). I Eq. 10 and Algorithm 1 print an unsquared L2 norm, although described as the DDPM simplified objective. **Choice:** use squared error as in IC, and record the typographical ambiguity. I describes a mean-free Gaussian; IC draws uncentered noise. **Choice:** project noise, output vectors, and priors into each molecule's zero-centroid subspace, following the paper's symmetry construction. This makes the covariance a centering projection, not a full-rank identity on R^(3A).

Noise schedule (I Appendix E.3, IC `beta_schedule.py`): 1,000 points β=sigmoid(linspace(−8,−4,1000)), approximately 0.00033535 to 0.01798621. `beta_min=1e-4` and `beta_max=.02` passed in IC do not affect this scheduler. Do not replace it with a cosine β schedule. IC's cosine schedule is for the **learning rate**.

IC stores ᾱ **before** multiplying by the current α and trains at d=1,…,999; its DPM solver instead cumulatively multiplies including the current β. This is a one-index discrepancy between training and solver schedules. **Choice:** use explicit ᾱ_0 plus indices 1,…,1000 consistently throughout; validate the analytic noising and reverse coefficients. This repairs a released-code inconsistency and is not bitwise upstream reproduction.

An ancestral DDPM update has mean

    μ_d = [y_d − β_d εθ(y_d,…)/sqrt(1−ᾱ_d)] / sqrt(α_d).

IC adds sqrt(β_d)ε rather than posterior-variance noise, and stops at d=1 without a final noise-free step. I leaves σ_d unspecified. **Choice:** expose variance convention; baseline uses β_d for stochastic intermediate steps and a deterministic final reconstruction, with the consistent index convention above. For reported sampling, follow I Appendix C.1: 50 probability-flow ODE evaluations using DPM-Solver. IC currently defaults to DPM-Solver++, order 2, multistep, uniform solver-time grid, from 1 to 1/1000, no final `denoise_to_zero`; the solver's time-to-label mapping contains a literal 1000. Preserve 1,000 training noise levels and explicitly map solver time to the corrected training labels.

An ancestral physical rollout repeatedly conditions on its last generated configuration, drawing fresh Gaussian noise each transition. Direct versus nested comparison holds the same initial configuration and total physical time fixed; direct uses kL, nested uses k successive transitions at L. Intermediate states are integrated out; a product of conditional kernels alone is not the terminal CK density without those integrals.

### ITO denoiser architecture

I Fig. 2 and Appendix D/Fig. 11 describe two ChiroPaiNN networks. IC `cpainn.py` and `embedding.py` implement them with native torch layers, `torch_geometric` graph construction/batching, and `torch_scatter` aggregation. There is no active e3nn or external PaiNN package; e3nn's optional soft-one-hot path is commented out and unusable as shipped. We can use native PyTorch dense/masked edges and `index_add` at this scale, avoiding graph-library installations. Any adapted IC implementation belongs under `src/tito_repro/vendor/ito/`, with its MIT license and attribution headers.

For A atoms, F scalar channels s have shape [A,F], vector channels v shape [A,F,3], edges [2,E], and coordinates [A,3]. Use complete directed graphs without self edges and isolate molecules within a batch. IC implements infinite-radius graphs with a 100-neighbor cap, harmless for 22 atoms but not a complete graph for all larger systems.

- Condition path: zero initial vector channels; learned nominal atom embedding; concatenate lag embedding; combine via MLP; 2 message/update blocks; scalar and vector readout.
- Score path: condition scalar/vector features, noisy target geometry, and diffusion-time embedding; combine scalar channels via MLP; 5 message/update blocks; scalar-gated, bias-free vector readout to one vector per atom. IC returns `noisy_coordinates + vector_readout` as its noise estimate; preserve/document that skip parameterization if adapting it.
- Scalar MLPs have 3 Linear layers with 2 SiLU activations. Vector Linear layers mix feature channels without Cartesian bias. Radial encodings depend only on interatomic distance. In IC the direction-like feature is r/(1+||r||), not a unit direction.
- The message gates multiply a source-node scalar MLP by a radial MLP. Four groups scale source vectors, displacement directions, a cross product between direction and destination vectors, and scalar messages. Sum over incoming neighbors and update residually.
- Scalar updates use vector norms; vector updates are scalar-gated linear transforms. Fig. 11 indicates an inner product between transformed vector channels Uv and Vv, whereas IC uses ||Vv||². **Proposed:** preserve the readable released-code update and flag the difference from the diagram, rather than silently mixing architectures.
- Cross products mix axial and polar vectors. The model respects proper rotations SO(3) while allowing chirality discrimination; generic reflection equivariance must not be an acceptance requirement. Translation is removed by centering. Rotation tests must rotate both the conditioning and noisy target coordinates.

I Eq. 21 sinusoidal encoding: for k=1,…,F/2, concatenate cos(kπz/l₀), sin(kπz/l₀). IC instead uses k=0,…,F/2−1, including a constant pair. IC normalizes physical lag by N_max and diffusion index by diffusion_steps; its radial length-scale defaults to 10, while I gives 3. **Proposed paper baseline:** use nonzero frequencies as Eq. 21, radial l₀=3 in standardized coordinates, lag denominator N_max, diffusion denominator 1000. These distinct length scales must be separate config fields.

I Appendix B.2 makes all 22 alanine atoms distinguishable with separate learned nominal embeddings, even when elements coincide. IC training defaults to IDs 0,…,21 and has 167 embedding entries. The element list is [1,6,1,1,6,8,7,1,6,1,6,1,1,1,6,8,7,1,6,1,1,1]. The single molecule needs no molecule-ID embedding. Such atom-index identity must not be carried into transferable TITO.

### ITO data and hyperparameters

I Table 3: three 250 ns MDShare trajectories (750 ns total), ACEMD; AMBER ff99SB-ILDN; Langevin; integration 2 fs; output every 1 ps; 300 K; periodic box (2.3222 nm)³; 651 TIP3P waters; PME real-space cutoff .9 nm; mesh spacing .1 nm; PME updates every two integration steps; hydrogen–heavy-atom bond constraints. Model only the 22 solute atoms, no velocities or waters. Friction, initial equilibration and seed are not specified there. Do not invent them for a purported exact regeneration.

IC downloads `alanine-dipeptide-{0,1,2}-250ns-nowater.xtc` plus `alanine-dipeptide-nowater.pdb` via MDShare. It calls MDTraj `center_coordinates` and divides by the hardcoded scalar **0.1661689 nm** when scaling is enabled. No Kabsch pair alignment is present. **Choice:** do not align MD endpoints to each other; center each frame, use one train-derived isotropic scale for newly generated data, and retain the upstream scalar only in the MDShare compatibility configuration.

| Parameter | I paper | IC released default / detail |
|---|---|---|
| Features, condition blocks, score blocks | 64, 2, 5 | same in training; bare score constructor defaults to 32 features |
| Maximum lag | 1000 stored frames | 1000 |
| Radial length scale | 3 | 10 |
| Diffusion steps | 1000 | 1000 |
| Batch size | 128 | 128 |
| Optimizer, learning rate | Adam, .001 | Adam, .001 |
| LR schedule | cosine mentioned for Müller–Brown; molecular details incomplete | CosineAnnealingLR T_max=20, default eta_min=0 |
| Gradient clipping | unspecified | norm clip 1.0 via Lightning |
| Epochs / updates | exact total absent; convergence 2–4 days | 50 epochs and `overfit_batches=1` |
| EMA | evaluation curves described as EMA; weight-decay setting not in paper | weight decay .99; warm-up min(.99,(1+u)/(10+u)) |
| Sampling | 50 ODE steps for all reported samples | 50 default; CLI 10 chains ×100 transitions, physical lag 100 |
| Train/val/test | all available trajectories trained; no held-out split | all three trajectories loaded |
| Alanine Fig. 3 | total lags 4,64,512 ps; 15,000 terminal samples; nested counts 4,64,512 at 1 ps | no equivalent full figure driver |
| Alanine throughput | 48 generated transitions/s, TITAN V, memory filled | not a local benchmark |

I Figure 3 contains conditional distributions, not merely a pooled equilibrium free-energy plot. Need a supplementary project equilibrium histogram plus the direct/nested conditional comparison to preserve that distinction. Fixed versus stochastic lag was statistically indistinguishable for alanine in I Fig. 6; improvement was observed for Müller–Brown. Do not claim alanine multi-lag superiority as an expected mandatory result.

For completeness, I's other experiments use: Müller–Brown 32 trajectories, 100,000 Brownian integration steps each, 1,000 burn-in, save every 10; random x∈[−1.5,1.2], y∈[−.2,2]; 5-layer MLP with 32 hidden features and 32-dimensional time encodings; N_max=1000; Fig. 5 uses 250,000 samples. Brownian step/friction values are not specified in Appendix B.1. Its potential coefficients are A=(−200,−100,−170,15), a=(−1,−1,−6.5,.7), b=(0,0,11,.6), c=(−10,−10,−6.5,.7), x̄=(1,0,−.5,−1), ȳ=(0,.5,1.5,1). These toy calculations are not scheduled as reproduction experiments.

Excluded protein experiments: Cα-only representations with 10/20/28/35 particles, original integration 2.5 fs, storage 200 ps, coordinate centering and standardization. Reference MSMs use 100 k-means centers in 5 TICs; TICA lag 1 ns; MSM lags 100/100/800/200 ns for Chignolin/Trp-cage/BBA/Villin; Table 4 lists ITO lag 200 ns. Some captions instead state 200 ps and the main text has a trajectory-length/step-count inconsistency. These settings are not imported into alanine. I reports total research expenditure ~3,000 training GPU-hours and 589 sampling GPU-hours, predominantly ~12 GB TITAN GPUs. No DESRES experiment is planned.

### Complete ITO repository audit

| Files | Finding and planned handling |
|---|---|
| `ito/data.py`, `ito/utils.py` | Correct per-trajectory start indexing; centering and fixed scale; no split or seed policy; fresh loaders/configuration required. |
| `ito/model/cpainn.py`, `embedding.py` | Reusable ChiroPaiNN structure, with architecture discrepancies listed above. `--n_layers` never enters model kwargs in training; radial 10 vs paper 3; zero-frequency pair vs paper. Optional soft-one-hot references a missing implementation. |
| `ito/model/beta_schedule.py` | Sigmoid endpoints match paper; exclusive cumulative-product indexing needs correction. Unused SNR helper squares ᾱ and should not be adopted. |
| `ito/model/ddpm.py` | Correct MSE intent; train/noise schedule discrepancy; no mean-free projection; standard DDPM and DPM paths; scheduler/clip config must become explicit. |
| `ito/model/ema.py` | TensorFlow moving-average provenance comment; `_params_refs` stores parameters but methods call them as weakrefs; custom state-loading does not match ordinary Module state layout; shadow parameters registered as ParameterList. Rewrite fresh EMA with buffers and explicit checkpoint state. |
| `ito/model/dpm_solve.py` | Adapted from LuChengTHU/dpm-solver; supports discrete/linear/cosine VP schedules, orders 1–3, single/multistep/adaptive, DPM-Solver/++. Only the required sampling path should be vendored, retaining upstream attribution and verifying its upstream license before copying. |
| `scripts/train_tlddpm.py` | `overfit_batches=1` repeats one batch and is unsuitable for scientific training; no validation, no explicit seeding; all checkpoints saved without a monitored best metric. Rewrite. |
| `scripts/sample_tlddpm.py` | Never swaps EMA weights in; no explicit CUDA placement/eval policy; normalization not reversed before saving. Rewrite with metadata and correct physical units. |
| `scripts/analyse_trajs.py` | VAMP on sin/cos phi,psi; phi indices [4,6,8,14], psi [6,8,14,16] are topology-specific. Silently drops trajectories with NaNs; plots on log marginal axes; no JSD/MSM/CK pipeline. Rewrite and report invalid samples explicitly. |
| `README.md`, `setup.py`, `makefile`, `.gitignore`, `LICENSE`, two initializers | MIT, 2023 olsson-group. Dependencies include Lightning, PyG, mdshare, tqdm; makefile also installs scatter/sparse/cluster. No pinned environment or experiment configs. Do not install the upstream package wholesale. |

Static inspection findings have not been validated by executing upstream training. Bugs described above are visible in the code, not results of an upstream runtime test.

## TITO objective and transfer architecture

Use y₁=x_(t+Δ) in centered, standardized coordinates and a centered isotropic Gaussian y₀. After an equivariant coupling, interpolate y_s=(1−s)y₀+s y₁ for s∼Uniform(0,1); target u=y₁−y₀. Fit

    L_CFM = E[ ||vθ(y_s, x_t, Δ, s) − (y₁−y₀)||² ].

T prints an unsquared norm, but TC `models/model.py` squares Cartesian norms and averages **atoms across the batch**; larger molecules consequently contribute more. This differs from IC's molecule-normalized loss. Proposed baseline uses the TC reduction and logs it. TC adds σ ε with **σ=.001 in model coordinates**, independent of s, after centering ε. T's written interpolant has no such perturbation. Proposed baseline exposes and uses σ=.001 as the released implementation, with σ=0 available for a paper-formula ablation. This is a smoothing perturbation, not a DDPM β schedule.

T's OT procedure first solves a linear assignment then a proper rotational Procrustes problem. Couple Gaussian points to the target within each molecule; the target atom order/topology and physical condition stay fixed. Permuting iid base noise does not require changing chemical labels. Do not permute the physical target among chemically different atoms. TC `permute_ot` uses the returned **row** indices and discards the assignment columns, so square assignment normally yields identity rather than the intended permutation. Its proper-rotation correction also merits a targeted determinant test. Proposed implementation uses SciPy assignment columns with explicitly tested orientation, then a proper Kabsch rotation (det R=+1), rotating only the Gaussian base. No rotation of the future MD endpoint relative to its condition is introduced.

T samples physical lags uniformly; TC makes this concrete: for Timewarp, stored spacing 5 ps, K=int(Δ_max/5), draw n uniformly in 1,…,K and return Δ=5n ps. Paper max Δ=5000 ps, so 1,000 possible lags. Molecules/pairs are sampled through the concatenated dataset, so selection is proportional to available start counts, not explicitly balanced by chemical family. Fixed-length training trajectories make those approximately equal. Proposed default preserves this law, rejects nonintegral requested frame offsets, and retains exact timestamps.

TC resolves architecture details missing from T:

- Element atomic numbers in a 167-entry learned node embedding; 13-entry edge-type embedding. Complete graph, no molecule-ID lookup, no residue embedding in this all-atom implementation. Residue names/indices remain metadata for topology and torsions. Do not add a new learned molecule identity or claim residue embeddings are a documented TITO addition.
- Two condition ChiroPaiNN blocks and five velocity blocks, width 64. Three-linear-layer scalar MLPs now add LayerNorm before each of the two SiLU activations. Vector maps remain bias-free.
- Five edge-message outputs: scalar gate for source vectors, displacement gate, scalar node increment, edge-feature increment, cross-product gate. Source scalar features concatenate with invariant edge features; a radial MLP gates them; messages sum over neighbors. Edge features update residually as well as node features.
- Physical lag and flow time join the condition **scalar** features in the velocity path. Their sine/cosine embeddings use frequencies 1,…,F/2, denominators max_lag in ps and 1, respectively. This placement differs from ITO's lag-conditioned condition encoder.
- Bond features distinguish single, double, triple, and through-space interactions in T. TC also provisions aromatic/virtual categories; exact supported chemistry and priority must be recorded in topology metadata. Do not infer covalent bonds anew from distorted generated coordinates.
- TC initializes velocity-path vector features with fresh iid random numbers at every forward call instead of copying condition vectors, adds the current interpolated coordinates to the readout, and centers the result. Random vector features can give rotation symmetry **in distribution**, but are not a deterministic equivariant ODE field and fail an ordinary fixed-input equivariance check. Proposed resolution: zero initial vector features in that path, retaining scalar condition information and geometry-derived vectors. This is an explicit, material implementation deviation requiring review, not a proven equivalent replacement. A stochastic-input compatibility variant would need random vectors treated as explicit rotating inputs and held fixed per flow solve.
- TC condition radial scale is configurable (default 10); the velocity network is constructed with its default 10 regardless of that argument. No cutoff or virtual node is enabled by the baseline training script. Proposed implementation gives both radial scales explicit configuration fields, initially 10.

Sampling solves dy/ds=vθ with **forward Euler** and fresh base noise for each physical transition. TC uses s_j=j/M, h=1/M, j=0,…,M−1. T's main peptide figures use M=40; the extrapolation figure uses 100. Do not substitute DDPM/DPM-Solver for this ODE and call it TITO.

### TITO hyperparameters and preprocessing

| Setting | T supplement / paper | TC default or resolution |
|---|---|---|
| Velocity layers, condition layers, width | 5, 2, 64 | same |
| Learning rate | .01 | .001; Adam, no LR scheduler |
| Batch size | 750 | 64 |
| Max lag, small molecules / peptides | 1000 / 5000 ps | CLI 100 ps; comments mention other values, not authoritative runs |
| Lag law | uniform | inclusive integer frame-lag uniform |
| Training epochs/updates | not reported | CLI 100000 epochs is a limit, not a measured paper training duration |
| Seed | not reported | 808313 |
| EMA | not reported | absent from released CFM training; project EMA will be an addition |
| Noise smoothing σ | not stated | .001 |
| Base std | Gaussian; explicit normalization not stated | 1.0 model units |
| Coordinate scale, peptides | not stated | .277 nm; commented older .48458207 not active |
| Coordinate scale, MDQM9 | not stated | .20754094 nm; commented .1504218429 not active |
| Precision / clip | not reported | medium float32 matmul precision; bf16 commented out; no configured gradient clip |
| Checkpoint interval | not reported | every 30 min, monitors val/loss unless no-evaluate |

T Table S3 (slash means small molecules / tetrapeptides):

| Figure | Lag ps | Nested transitions | Euler steps | Batch |
|---|---:|---:|---:|---:|
| 2 | 57 / 250 | 640 / 500 | 20 / 40 | 32 |
| 3 A,B | 1000 | 1000 | 20 | 128 |
| 3 C | 1000 | 50000 | 20 | 32 |
| 4 | figure-dependent | 1 or 5 | 40 | 50000 |
| 5 | 5000 | 1 | 100 | 51200 |

For Fig. 3 C, S3 also lists 10 ps MD fine-tuning ×32,000 replicas and 500 ns ultra-long MD ×32,000 replicas (16 ms total). Those are not feasible here. Fig. 3's caption describes 32 μs for TITO comparison and 160 μs for propiolamide; table/caption totals are not fully reconcilable without the original run manifests. Do not claim these extreme campaigns will be reproduced.

**MDQM9-nc:** 12,530 noncyclic QM9 molecules, vacuum GAFF at room temperature, median trajectory 36.5 ns, size-dependent sampling. T adds eight-temperature RE at 300,400,…,1000 K, 1 μs; average exchange 58%. The public dataset's older RE subset is 100 ns for 100 test molecules and is not the TITO RE campaign. TC reads each molecule's `data/time_lag`; do not assume uniform physical spacing across molecules. Public HDF5/SDF/split data details are in the availability audit.

**Timewarp:** T says 1457 `large` training molecules and 92 `huge` test molecules, 50 ns and 1 μs respectively. The PDF's malformed “50 mns” is resolved as 50 ns by J, the dataset card and simulation step count. “Huge” refers to trajectories, not a new peptide-length class; both are tetrapeptides.

TC preprocessing takes every fifth entry of raw `positions`, `time`, and `energies` and asserts the retained timestamps differ by 5 ps. Timewarp's raw recording contains closely spaced frames around coarse sampling points; **do not interpret the raw frame index as a regular 1 ps grid**. TW job configs specify 10,000 integration-step spacing with a 0.5 fs integrator, i.e. 5 ps coarse anchors. In Phase 2, inspect all NPZ timestamp sequences, select exact regular anchors, and fail or document any mismatch rather than blindly decimating. Keep all atoms including H and preserve atom order. Center by arithmetic centroid and standardize by .277 nm for released-code compatibility. TC uses RDKit on a PDB with coordinates replaced by the first trajectory frame because original PDB coordinates may be unsuitable for bond inference.

TW `simulation/md.py`, `amber14-implicit` preset (used for all 4AA datasets): `amber14-all.xml` + `implicit/obc1.xml`; **OBC1, 310 K, LangevinMiddleIntegrator, 0.3 ps⁻¹, 0.5 fs, CutoffNonPeriodic at 2 nm, constraints=None**. These replace the vague “room temperature” description for matching peptide references. They are distinct from the user-requested Phase 1 OBC2/300 K/2 fs setup. Match TW termini, protonation, H atoms and force-field parameters when assessing energies; topology validation remains a Phase 2 task.

**Larger peptides:** T uses six sequences each of lengths 5,6,7,8, sampled using vertebrate residue frequencies, with the TW simulation setup. Main methods state 1 μs/reference; S10 says 100 ns per system, an unresolved disagreement. Base Gaussian standard deviation is multiplied by (n_res/4)^0.688, interpreting the stated Flory law relative to the training tetrapeptide length. It is the standard deviation, not variance, that scales. Without this correction T reports failure beyond pentapeptides; long nested runs remain unstable for the largest molecules even with scaling. Do not promise stable kinetics for octapeptides.

## Evaluation definitions, paper numbers, and project choices

**Probability/free energy:** estimate normalized histogram masses p_b=c_b/Σc. F_b=−RT ln p_b + C in kJ/mol, with min finite F set to zero for plotting. Zero-count bins are masked/infinite, not silently filled as evidence of support. Also plot F/(RT) when comparing dimensionless paper panels. I's protein folding ΔG/(kT)=−ln[p_fold/(1−p_fold)] is a different scalar observable. Basin probabilities sum histogram masses in predeclared regions.

**JSD:** m=(p+q)/2; JSD=.5 Σ p ln(p/m)+.5 Σ q ln(q/m), with 0 ln 0=0. TC uses natural logs although T says bounded by 1; the natural-log bound is ln2. Primary comparisons will report nats and secondary bits; SciPy's `jensenshannon` returns the square root and must be squared if used. Use exactly shared bin edges, and never drop out-of-range generated samples; explicit overflow/invalid mass must be reported. T evaluates first-two-TIC histograms, not phi/psi histograms. The latter are required additional project metrics and not numerically interchangeable with the published TICA JSD.

**TICA:** learn the slow linear projection from reference MD only, then transform model samples with that fitted projection. TC uses sin/cos molecular dihedrals and `TICA(lagtime=...,dim=2)`; analysis CLI defaults to one MD frame. Exact feature selection and histogram-bin count for the published figure are not fully specified. Proposed peptide baseline: sin/cos backbone plus side-chain rotatable dihedrals when unambiguously defined, 5 ps TICA lag, 64×64 shared histogram; additional phi/psi histograms per residue. Record each dihedral index and any feature exclusions. Fix reference-derived transforms before looking at generated errors; bin-resolution sensitivity at 32 and 64 distinguishes estimator noise from model error.

**VAMP:** for feature covariance matrices C_00,C_01,C_11, whiten K=C_00^(-1/2) C_01 C_11^(-1/2) on retained numerical subspaces. Singular values σ_i characterize VAMP modes; VAMP2=Σ_i σ_i². Include/omit the stationary mode consistently and log feature rank and numerical cutoff. I's released alanine features are [cos φ,sin φ,cos ψ,sin ψ] and `deeptime.decomposition.VAMP(lag).score(2)`; its score includes the estimator's constant-function convention. I's gap is **model minus reference**, whereas T's formula is **reference minus model**. Keep explicit key names so signs cannot be confused.

TC's evaluation routine computes the reference score on the reference model and cross-scores the predicted model with the reference VAMP model (`ref.score(2,pred)`), rather than plainly subtracting independently calculated Σσ². Record both `vamp2_gap_formula` and `vamp2_gap_reference_cross_score`; original figure provenance is insufficient to prove which code revision generated it. Do not equate a cross-score with the singular-value sum without verification.

**Timescales:** T uses t_i=−τ/ln σ_i despite referring to “eigenvalues” in prose. It defines top-10 relative discrepancy r=(1/10)Σ_i |t_i^MD−t_i^TITO|/t_i^MD, sorting nontrivial timescales decreasingly. A reversible MSM instead uses its transition-matrix eigenvalues λ_i, with t_i=−τ/ln|λ_i| when meaningful. These must be separate outputs: the user's MSM requirement does not replace T's VAMP timescales. Exclude the stationary mode; flag nondecaying/negative/complex/undefined modes and insufficient rank rather than clipping all failures into plausible finite times. Negative eigenvalues merit an explicit diagnostic. Compare equal **physical estimator lags**, not equal frame counts. No top-10 metric can be estimated from only four alanine features.

**MSM and first-passage observables:** I's protein pipeline fits reference TICA and k-means, then maps generated data into the same discretization; Bayesian reversible MSM posteriors give uncertainty; PCCA identifies folded/unfolded sets. A computational definition of MFPT solves h_i=1+Σ_j P_ij h_j outside the target, with h=0 inside, and converts steps to physical time, averaging initial states using a stated distribution. The paper's displayed first-hit expression does not specify this estimator adequately; use the MSM definition. No folding MFPT is required for alanine or tetrapeptides.

**CK:** compare p_(kL)(·|x) against ∫∏_{j=0}^{k−1}p_L(x_(j+1)|x_j) dx_1…dx_(k−1), terminal x_k retained. Estimate their projected terminal histograms from independent replicate branches starting at identical x. I compares phi/psi; T Fig. 4 compares TICs with five nested steps. Neither gives a numerical JSD cutoff. Proposed project gates and bootstrap procedure are in the scaling plan. Passing in a 2D projection is evidence only in that projection, not proof of full-state Markovianity.

**Boltzmann test:** T starts from MD equilibrium and tests invariance under propagation. The user's nonequilibrium-to-reference test assesses relaxation/convergence. Implement both; convergence to one marginal does not establish correct kinetics, and stationarity from equilibrium does not establish attraction from every initial condition.

**Coverage/precision:** let S={b:c_model,b>0 and c_ref,b>0}. Coverage=Σ_(b∈S) p_ref,b; precision=Σ_(b∈S) p_model,b. T describes a general density threshold δ but implements nonzero empirical counts. Binning and finite sample size strongly affect both. RE is the reference for T Fig. 3; MD-only results here must be labeled differently.

**Fast modes/energy:** compare bond-distance and bond-angle marginals via −RT ln histogram, and potential-energy distributions. T Fig. 4C labels non-harmonic energy; total raw force-field potential energy in the extension is different. Implement both total U and a force-group decomposition when reproducing that panel; state exactly which bonded harmonic terms were removed. T underestimates fast-mode variance slightly and consequently shifts some energies lower. I's alanine bond-length variances are slightly too broad. Energy closeness is a distribution comparison, not a guarantee of time-stepwise conservation in thermostatted stochastic MD.

**Size extrapolation:** compare radius of gyration R_g=sqrt[(1/A)Σ_a||x_a−centroid||²] unless a explicitly labeled mass-weighted convention is used; record all/heavy atom selection. Fit slope ν in log mean R_g vs log size; T S10 plots heavy-atom count although the sampling correction uses residue count. T S11 displays only samples within MD's energy range; our plots must also report out-of-range fraction so failures cannot disappear.

**Throughput:** aggregate generated physical ns divided by actual GPU wall-hours, with synchronization, solver evaluations, molecule size, batch, precision, warm-up, and I/O policy logged. Report aggregate ensemble throughput and sequential-chain latency separately. Count each transition's lag once, never each solver step. Compare unbiased OpenMM using the same GPU, force field, temperature and atom topology; report retained-sample throughput separately if filtering is used.

| Figure 2 statistic (arXiv T, PDF p.8) | Small molecules | Tetrapeptides |
|---|---:|---:|
| TICA JSD mean / median, printed values | .097 / .087 | **.042 / .036** |
| VAMP gap mean / median | −.388 / −.072 | **−1.300 / −.961** |
| Top-10 relative timescale discrepancy mean / median | .554 / .192 | **1.204 / .434** |
| Illustrative molecule relative error / gap | .112 / −.043 | .080 / −.148 |

These are reported values, not our results. Natural-log interpretation is supported by TC; figure-generation settings are not completely archived. The broad peptide mean timescale error is materially worse than the single illustrative peptide and must not be concealed by quoting only the example. T S1's example JSDs are .09,.21,.35. S1 Hamming-distance table reports mean JSD .043/.040 at listed dissimilarities .25/.75, with 64%/26% of test peptides; those percentages do not account for all test peptides and are not a split definition.

Throughput (J Table 1, A100 80 GB): small-molecule MD 3.5 μs/day, TITO 11.3 ms/day; peptide MD **.67 μs/day**, TITO **10.3 ms/day**. ArXiv T prints `.67 s` for peptide MD, inconsistent with the speedup statement; the journal resolves this. Converted: peptide MD ≈27.92 ns/GPU-hour and TITO ≈429,167 ns/GPU-hour, ≈15,373× ratio. These are maximum reported rates on an 80 GB GPU, not estimates for our machine or for every solver/lag configuration.

I's protein observables (read for completeness, excluded from our reproduction): MD/ITO ΔG/kT = −1.28(1)/−.64(33), 1.47(6)/2.84(6), .97(3)/1.52(3), 1.21(2)/2.22(3); folding MFPT μs=.565(4)/1.02(24),13.6(4)/37(2),11.7(2)/8.6(2),2.41(3)/3.27(7); unfolding MFPT μs=2.01(2)/2.12(34),3.4(2)/2.85(9),5.1(1)/1.75(4),.68(1)/.354(5), in Chignolin/Trp-cage/BBA/Villin order. These numbers are not alanine benchmarks.

## Extension implications and follow-up skim

The extension is new work, with no paper AUROC target. The proposed relaxation/energy label is an **operational reference-disagreement label**: a true new metastable basin can remain outside a finite MD reference after several ps. Keep separate reasons (`reference_basin_absent`, `high_minimized_energy`, `simulation_failure`) and an uncertain category, in addition to the requested composite binary label. Compute reference minimized-energy thresholds using reference configurations minimized with the same protocol. Compare minimization sensitivity; a labeler sharing energy with the energy detector creates an inherent advantage and should be reported with a basin-only label ablation. Save every threshold, force-field checksum, seed, relaxation length, and label-cache fingerprint.

For reference-free use, calibrate raw/minimized energy z-scores within molecule against an independent model-generated calibration pool; do not use held-out MD to normalize detector features. “Reference-free” applies at inference, not to evaluation labels. CK scores require replicate future distributions, so a single direct/nested endpoint pair is not a divergence estimate. For flow models the requested denoiser uncertainty becomes **velocity-field/reconstruction disagreement** at flow interpolation levels or across EMA checkpoints; it is a proxy, not calibrated epistemic uncertainty, and EMA checkpoints are correlated.

ROC uses TPR=TP/(TP+FN) vs FPR=FP/(FP+TN); AUROC integrates the empirical curve with tie handling. Report precision=TP/(TP+FP) at a threshold set on validation data for FPR≤.01, and achieved test FPR. If either class is absent, report undefined, not AUROC=1. Cluster bootstrap by molecule; also provide per-molecule class counts. Several thousand samples may still contain too few genuine problems to support an AUROC>.8 claim or stable 1% FPR estimates.

Filtering alters both equilibrium selection probabilities and dynamics. Never concatenate retained frames into a fictitious regular-time trajectory. Keep original time indices, count only genuinely observed transitions at the specified physical lag, and label remaining MSM estimates as selected/censored dynamics. Thermodynamic agreement can improve while kinetic bias worsens; report retention and basin-dependent retention rates. Rejection-resampling would define a different Markov kernel and is not silently substituted for post-hoc filtering.

P uses a coarse-grained backbone representation and transformer-based conditional flow matching, with ESMC-300M embeddings for its smaller model and ESMC-6B for PLaTITO-19M. Optional Proteina structure embeddings and LLM annotations are separate ablations. Its training scale is 8 GPUs, ~1100 GPU-hours per model and 56 ms of MD, with AdamW; 3M models use width 128, 3 transformer layers, 6 attention heads, 10 registers, batch 140, LR .001, clip .1, 400k steps (350k with structure); 19M uses 6 layers, residue width 256, conditioning/pair width 196, batch 82, LR .0001, 200k steps. Max physical step is 200 ns. These are not TITO all-atom hyperparameters. The useful extension lesson is to distinguish chemical/size distribution shift from detector confidence and report reference coverage limitations. We will not add pLM inference, new dependencies, or its DESRES benchmark to this project.

## Remaining uncertainties before implementation

The explicit-versus-implicit alanine benchmark choice, deterministic replacement of TC random vector features, HDF5 dependency permission, and concrete phase thresholds need review; proposed resolutions are in [deviations.md](deviations.md). Exact TITO training step count, original run configs, the paper's 92 test peptide identities/exclusions, original histogram grid, exact dihedral feature list, and all original seeds are unavailable from the inspected sources. Do not present chosen replacements as recovered facts. Data integrity, topology, and actual GPU cost remain unmeasured and must be checked in their appropriate phase before scientific claims.
