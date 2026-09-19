"""Paired cached-checkpoint rollouts with shared initial states and sampling RNG."""
from pathlib import Path
import time

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from tito_repro.data.pairs import center
from tito_repro.eval.metrics import histogram, jsd, torsions
from tito_repro.sample.rollout import rollout
from tito_repro.train.phase1 import get_data, make_model
from tito_repro.utils.runtime import digest, write_json


def compare_histograms(reference: np.ndarray, plain: np.ndarray, residual: np.ndarray,
                       bins: int, repeats: int, seed: int) -> dict:
    """Angles: reference [N,2], paired generated [chains,frames,2], all radians."""
    if plain.shape != residual.shape or plain.ndim != 3 or plain.shape[-1] != 2:
        raise ValueError("Matching [chains,frames,2] arrays required")
    if min(plain.shape[:2]) < 1 or repeats < 2 or bins < 2:
        raise ValueError("Positive sample counts and at least two bins/replicates required")
    ref_hist = histogram(reference, bins)
    chain_histograms = [np.stack([histogram(chain, bins) for chain in samples])
                        for samples in (plain, residual)]
    values = [jsd(ref_hist, counts.sum(0)) for counts in chain_histograms]
    rng = np.random.default_rng(seed)
    differences, controls = [], []
    for _ in range(repeats):
        chosen = rng.integers(len(plain), size=len(plain))
        boot = [jsd(ref_hist, counts[chosen].sum(0)) for counts in chain_histograms]
        differences.append(boot[1] - boot[0])
        draw = reference[rng.integers(len(reference), size=plain.shape[0] * plain.shape[1])]
        controls.append(jsd(ref_hist, histogram(draw, bins)))
    return {"bins": bins, "plain_jsd_nats": values[0], "residual_jsd_nats": values[1],
            "residual_minus_plain_jsd_nats": values[1] - values[0],
            "paired_chain_bootstrap_interval_nats": np.quantile(differences, [.025, .975]).tolist(),
            "matched_count_reference_jsd_mean_nats": float(np.mean(controls)),
            "matched_count_reference_interval_nats": np.quantile(controls, [.025, .975]).tolist()}


@torch.no_grad()
def compare_sampling(cfg: DictConfig, output: Path) -> dict:
    settings = cfg.comparison
    if (set(settings.checkpoints) != {"plain", "residual"}
            or any(not path for path in settings.checkpoints.values())):
        raise ValueError("Set comparison.checkpoints.plain and .residual to local checkpoints")
    if (min(settings.chains, settings.transitions, settings.solver_steps) < 1
            or not settings.lags or min(settings.lags) < 1
            or not settings.histogram_bins or min(settings.histogram_bins) < 2
            or settings.bootstrap_replicates < 2):
        raise ValueError("Invalid paired sampling budget")
    states, models = {}, {}
    for name, path in settings.checkpoints.items():
        state = torch.load(path, map_location="cpu", weights_only=False)
        model_cfg = OmegaConf.create(state["config"])
        model = make_model(model_cfg)
        model.load_state_dict(state["ema"])
        models[name], states[name] = model.eval(), state
    base = OmegaConf.create(states["plain"]["config"])
    other = OmegaConf.create(states["residual"]["config"])
    for key in ("data", "train", "seed", "runtime"):
        if base[key] != other[key]:
            raise ValueError(f"Paired checkpoints differ in {key}")
    a, b = dict(base.model), dict(other.model)
    if a.pop("epsilon_skip", False) or not b.pop("epsilon_skip", False) or a != b:
        raise ValueError("Expected identical models differing only in epsilon_skip=false/true")
    if states["plain"]["step"] != states["residual"]["step"]:
        raise ValueError("Paired checkpoints must have matching completed steps")
    if max(settings.lags) > base.model.max_lag or settings.solver_steps > base.model.diffusion_steps:
        raise ValueError("Sampling lag or solver budget exceeds checkpoint configuration")
    trajectories, spacing, provenance = get_data(base)
    if any(state["provenance"] != provenance for state in states.values()):
        raise ValueError("Checkpoint/source fingerprint mismatch")
    positions = np.concatenate(trajectories)
    reference = torsions(positions, base.data.topology, base.data.torsions)
    rng = np.random.default_rng(cfg.seed)
    initial = center(torch.tensor(positions[rng.integers(len(positions), size=settings.chains)])) / states["plain"]["scale_nm"]
    result = {"status": "running", "device": "cpu", "seed": cfg.seed,
              "scientific_acceptance": "not_evaluated", "data": provenance,
              "checkpoint_sha256": {name: digest(path) for name, path in settings.checkpoints.items()},
              "checkpoint_steps": states["plain"]["step"], "lags": [],
              "limitations": ["four-chain default bootstrap is exploratory", "reference held fixed in bootstrap",
                              "IID reference controls do not measure temporal uncertainty",
                              "torsion histograms do not establish chemical validity or kinetics",
                              "1000-frame lag is the upper boundary beyond DisExp training support"]}
    started = time.perf_counter()
    for lag in settings.lags:
        samples = {}
        for name, model in models.items():
            # Reset after model construction so the two variants receive identical
            # Gaussian draws at each physical transition, even as trajectories diverge.
            torch.manual_seed(cfg.seed + int(lag))
            coordinates = rollout(model, initial, lag, settings.transitions, settings.solver_steps).numpy() * states[name]["scale_nm"]
            np.save(output / f"{name}_lag_{lag}_nm.npy", coordinates)
            samples[name] = np.stack([torsions(chain[1:], base.data.topology, base.data.torsions)
                                      for chain in coordinates])
        row = {"lag_ps": lag * spacing, "generated_frames_per_variant": settings.chains * settings.transitions,
               "histograms": [compare_histograms(reference, samples["plain"], samples["residual"],
                    bins, settings.bootstrap_replicates, cfg.seed + lag) for bins in settings.histogram_bins]}
        result["lags"].append(row)
        result["wall_seconds"] = time.perf_counter() - started
        write_json(output / "metrics.json", result)
    result["status"] = "completed"
    write_json(output / "metrics.json", result)
    return result
