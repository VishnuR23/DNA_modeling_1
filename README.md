# TITO reproduction and reference-free sample assessment

This project will reproduce implicit transfer operators for molecular dynamics at reduced scale. It will then test whether sample-quality detectors can identify problematic configurations without reference trajectories at inference time.

**Status: source review only; Phase 1 has not started. No reproduction or detector results exist.**

Read [paper notes](docs/paper_notes.md), the [60 GPU-hour scaling plan](docs/scaling_plan.md), [dataset availability audit](docs/data_availability.md), and [deviations and review decisions](docs/deviations.md). Physical units are specified in [units.md](units.md).

The project targets Python 3.11 and one 16–24 GB CUDA GPU, with CPU tests. No project dependencies have been installed. The experiment configurations, implementation, tests, and figure commands will be added only after review of this preparatory stage.

The eventual reproduction report will compare measured results to the papers, including unsuccessful outcomes. No headline figure or one-command figure reproduction is claimed yet. The phase gates and planned measurements are specified in the scaling plan.

Current limitations: the requested implicit-solvent alanine simulation differs from the predecessor's explicit-solvent reference; published TITO details and released code disagree in several places; the available Timewarp test inventory differs from the paper; 60 GPU-hours is a capped feasibility experiment rather than a guarantee of paper-level convergence. Short reference trajectories cannot prove that a generated basin is unphysical.
