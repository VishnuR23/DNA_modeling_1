# Phase 1 reference plan: load a capped alanine topology, configure the requested
# ff14SB/OBC2 model, minimize/equilibrate, record regular 1 ps CPU-only trajectories;
# report actual simulated duration and energies, without claiming converged basins.
"""Local OpenMM implicit-solvent alanine simulation in nm, ps, kJ/mol and K."""
import logging
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import openmm as mm
from openmm import app, unit
from omegaconf import DictConfig

from tito_repro.utils.runtime import digest, write_json


def simulate(cfg: DictConfig, output: Path) -> dict[str, Any]:
    """Generate configured replicas [R,T,A,3] nm on CPU/Reference OpenMM platforms only."""
    ref = cfg.reference
    if ref.platform not in ("CPU", "Reference"):
        raise ValueError("OpenMM CPU or Reference platform required; GPUs forbidden")
    stride = int(round(ref.save_ps / ref.step_ps))
    if stride < 1 or not np.isclose(stride * ref.step_ps, ref.save_ps):
        raise ValueError("save_ps must be a positive integer multiple of step_ps")
    topology = app.PDBFile(cfg.data.topology)
    forcefield = app.ForceField(*ref.forcefield_files)
    system = forcefield.createSystem(topology.topology, nonbondedMethod=app.NoCutoff,
                                     constraints=app.HBonds if ref.hydrogen_constraints else None)
    system_xml = mm.XmlSerializer.serialize(system)
    (output / "system.xml").write_text(system_xml)
    platform = mm.Platform.getPlatformByName(ref.platform)
    properties = {"Threads": str(cfg.runtime.threads)} if ref.platform == "CPU" else {}
    positions, energies = [], []
    started = time.perf_counter()
    for replica in range(ref.replicas):
        integrator = mm.LangevinIntegrator(ref.temperature_k * unit.kelvin,
                                           ref.friction_per_ps / unit.picosecond,
                                           ref.step_ps * unit.picosecond)
        integrator.setRandomNumberSeed(cfg.seed + replica)
        simulation = app.Simulation(topology.topology, system, integrator, platform, properties)
        simulation.context.setPositions(topology.positions)
        simulation.minimizeEnergy(tolerance=ref.minimize_tolerance_kj_mol_nm * unit.kilojoule_per_mole / unit.nanometer,
                                  maxIterations=ref.minimize_iterations)
        simulation.context.setVelocitiesToTemperature(ref.temperature_k * unit.kelvin, cfg.seed + replica)
        simulation.step(int(round(ref.equilibration_ps / ref.step_ps)))
        xyz, energy = [], []
        for frame in range(ref.frames):
            simulation.step(stride)
            state = simulation.context.getState(getPositions=True, getEnergy=True)
            xyz.append(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
            energy.append(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
            if not np.isfinite(energy[-1]):
                raise FloatingPointError("Nonfinite reference energy")
            if (frame + 1) % ref.log_every_frames == 0:
                logging.info("reference replica=%d frame=%d/%d elapsed_s=%.1f", replica, frame + 1,
                             ref.frames, time.perf_counter() - started)
        positions.append(np.asarray(xyz, dtype=np.float32))
        energies.append(energy)
        del simulation, integrator
    elapsed = time.perf_counter() - started
    np.savez(output / "reference.npz", positions_nm=np.stack(positions), energies_kj_mol=energies,
             spacing_ps=ref.save_ps)
    simulated_ps = ref.replicas * (ref.frames * ref.save_ps + ref.equilibration_ps)
    result = {"status": "completed", "platform": ref.platform, "device": "cpu", "seed": cfg.seed,
              "production_ps": ref.replicas * ref.frames * ref.save_ps, "integrated_ps": simulated_ps,
              "wall_seconds": elapsed, "ns_per_cpu_hour": simulated_ps / 1000 / elapsed * 3600,
              "topology_sha256": digest(cfg.data.topology), "system_sha256": digest(output / "system.xml"),
              "scientific_acceptance": "not_evaluated_short_reference", "openmm_version": mm.__version__}
    fig, ax = plt.subplots()
    for replica, energy in enumerate(energies):
        ax.plot(np.arange(1, len(energy) + 1) * ref.save_ps, energy, label=f"replica {replica}")
    ax.set(xlabel="Production time (ps)", ylabel="Potential energy (kJ/mol)")
    ax.legend()
    fig.savefig(output / "energy.png", dpi=140)
    plt.close(fig)
    write_json(output / "metrics.json", result)
    return result
