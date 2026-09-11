# Units and coordinate conventions

All persisted physical quantities and public interfaces use nanometers (positions, distances), picoseconds (MD time and lag), kJ/mol (potential and molar free energy), and Kelvin (temperature). Angles use radians, normally wrapped to [-π, π). Forces use kJ mol⁻¹ nm⁻¹; friction uses ps⁻¹. Mass uses daltons when needed by OpenMM.

Conversions: 1 Å = 0.1 nm; 1 fs = 0.001 ps; 1 ns = 1,000 ps; 1 μs = 1,000,000 ps. PDB files store Å while MDTraj's `xyz` uses nm; conversion must occur exactly once at the reader boundary. OpenMM quantities must be converted explicitly rather than stripped of units by coercion.

Diffusion index d, continuous flow time s∈[0,1], and physical lag Δ are distinct variables. Neither diffusion steps nor ODE steps advance physical MD time. A generated transition with lag Δ advances a trajectory by Δ regardless of solver evaluations.

Model coordinates are dimensionless: y=(x−arithmetic_centroid(x))/s_x, where s_x is a positive scale in nm recorded in configuration and checkpoints. Centers and scaling are per molecule; padding never enters centroids or losses. Isotropic scaling preserves rotation equivariance; independent coordinate-axis standardization does not. Generated coordinates must be multiplied by s_x before energy or trajectory evaluation. Noise and predicted noise are dimensionless; a flow velocity has dimensionless coordinates per dimensionless flow time and is not an MD velocity.

Use the molar gas constant from `openmm.unit.MOLAR_GAS_CONSTANT_R` for F=−RT ln p, in kJ/mol. The exact SI constants are k_B=1.380649×10⁻²³ J/K and N_A=6.02214076×10²³ mol⁻¹, giving R=0.00831446261815324 kJ mol⁻¹ K⁻¹. Source: [BIPM SI Brochure, defining constants](https://www.bipm.org/en/publications/si-brochure). Retrieve constants through OpenMM in implementation and cite this source in the relevant comment.

JSD is dimensionless. Report `jsd_nats` with natural logs, bounded by ln 2, and `jsd_bits=jsd_nats/ln(2)`, bounded by 1. Do not confuse JSD with its square-root distance. Free-energy logarithms are natural regardless of the JSD convention.
