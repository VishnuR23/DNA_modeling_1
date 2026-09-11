# Phase 1 plan: validate deterministic CPU data/noise/equivariance plumbing;
# train a reduced ChiroPaiNN DDPM with EMA on explicit, separately identified data;
# save resumable checkpoints and physical rollouts; evaluate torsions/MSM/CK;
# report scientific gates as unpassed until every acceptance requirement is measured.
"""Bounded local training and reproducible sampling; no accelerator code paths."""
import copy
import logging
import random
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from tito_repro.data.pairs import LagPairs, center, load_trajectories, synthetic_trajectories
from tito_repro.eval.metrics import histogram, jsd, msm_timescale, torsions, vamp_score
from tito_repro.models.denoiser import Denoiser
from tito_repro.models.diffusion import Diffusion
from tito_repro.sample.rollout import rollout
from tito_repro.utils.runtime import digest, write_json

LOG = logging.getLogger(__name__)


def make_model(cfg: DictConfig) -> Diffusion:
    """Construct a CPU DDPM; configuration scales are dimensionless after normalization."""
    kwargs = OmegaConf.to_container(cfg.model, resolve=True)
    kwargs.pop("beta_start")
    kwargs.pop("beta_end")
    return Diffusion(Denoiser(**kwargs), cfg.model.diffusion_steps, cfg.model.beta_start, cfg.model.beta_end)


def get_data(cfg: DictConfig) -> tuple[list[np.ndarray], float, dict[str, Any]]:
    """Load configured trajectories [T,A,3] nm, spacing ps and data fingerprint."""
    if cfg.data.kind == "synthetic":
        t = synthetic_trajectories(cfg.data.replicas, cfg.data.frames, cfg.model.atoms,
                                   cfg.seed, cfg.data.correlation, cfg.data.scale_nm)
        return t, cfg.data.spacing_ps, {"kind": "synthetic", "seed": cfg.seed}
    trajectories, spacing = load_trajectories(cfg.data.path)
    if trajectories[0].shape[1] != cfg.model.atoms:
        raise ValueError("Configured atom count does not match input")
    return trajectories, spacing, {"kind": cfg.data.kind, "sha256": digest(cfg.data.path),
                                   "path": str(Path(cfg.data.path).resolve())}


def train(cfg: DictConfig, output: Path) -> dict[str, Any]:
    """Train with bounded CPU wall time; saves dimensionless weights and nm scale metadata."""
    trajectories, spacing, provenance = get_data(cfg)
    sampler = LagPairs(trajectories, cfg.model.max_lag, cfg.seed, cfg.train.lag_law, cfg.train.fixed_lag)
    scale = float(cfg.data.scale_nm)
    if scale <= 0:
        raise ValueError("scale_nm must be positive")
    model = make_model(cfg)
    average = copy.deepcopy(model)
    for parameter in average.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.steps)
    losses, start_step = [], 0
    if cfg.train.resume:
        state = torch.load(cfg.train.resume, map_location="cpu", weights_only=False)
        if state["config"]["seed"] != cfg.seed or state["config"]["runtime"] != OmegaConf.to_container(cfg.runtime):
            raise ValueError("Resume must preserve seed and CPU runtime configuration")
        for key in ("model", "data", "train"):
            previous = state["config"][key].copy()
            current = OmegaConf.to_container(cfg[key], resolve=True)
            if key == "train":
                for allowed in ("resume", "max_seconds", "checkpoint_every", "log_every"):
                    previous.pop(allowed, None)
                    current.pop(allowed, None)
            if previous != current:
                raise ValueError(f"Resume changes {key} semantics; start a new run instead")
        if provenance != state["provenance"]:
            raise ValueError("Resume data fingerprint differs")
        model.load_state_dict(state["model"])
        average.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        sampler.generator.set_state(state["pair_rng"])
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        start_step, losses = state["step"], state["losses"]

    def save(step: int) -> None:
        payload = {"model": model.state_dict(), "ema": average.state_dict(), "step": step,
                   "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                   "pair_rng": sampler.generator.get_state(), "torch_rng": torch.get_rng_state(),
                   "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
                   "config": OmegaConf.to_container(cfg, resolve=True), "scale_nm": scale,
                   "spacing_ps": spacing, "provenance": provenance, "losses": losses}
        temporary = output / "checkpoint.tmp"
        torch.save(payload, temporary)
        temporary.replace(output / "checkpoint.pt")

    started, completed = time.perf_counter(), start_step
    for step in range(start_step, cfg.train.steps):
        if time.perf_counter() - started >= cfg.train.max_seconds:
            break
        x0, x1, lag = sampler.sample(cfg.train.batch_size)
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(center(x0) / scale, center(x1) / scale, lag)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.gradient_clip, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        decay = min(cfg.train.ema_decay, (step + 2) / (step + 11))
        with torch.no_grad():
            for shadow, raw in zip(average.parameters(), model.parameters()):
                shadow.lerp_(raw, 1 - decay)
        losses.append(float(loss.detach()))
        completed = step + 1
        if completed % cfg.train.log_every == 0:
            LOG.info("step=%d loss=%.6f elapsed_s=%.2f", completed, losses[-1], time.perf_counter() - started)
        if completed % cfg.train.checkpoint_every == 0:
            save(completed)
    save(completed)
    elapsed = time.perf_counter() - started
    np.save(output / "loss.npy", losses)
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(losses) + 1), losses)
    ax.set(xlabel="Optimizer update", ylabel="Noise MSE", title="CPU ITO training")
    fig.savefig(output / "loss.png", dpi=140)
    plt.close(fig)
    result = {"status": "completed" if completed == cfg.train.steps else "time_limit",
              "steps": completed, "steps_this_run": completed - start_step, "wall_seconds": elapsed,
              "seconds_per_step": elapsed / max(1, completed - start_step), "seed": cfg.seed,
              "device": "cpu", "parameters": sum(p.numel() for p in model.parameters()),
              "final_loss": losses[-1] if losses else None, "scientific_acceptance": "not_evaluated",
              "data": provenance}
    write_json(output / "metrics.json", result)
    return result


def evaluate(cfg: DictConfig, output: Path) -> dict[str, Any]:
    """Evaluate checkpoint rollouts; positions nm, torsions rad, lags/timescales ps."""
    state = torch.load(cfg.evaluation.checkpoint, map_location="cpu", weights_only=False)
    model_cfg = OmegaConf.create(state["config"])
    model = make_model(model_cfg)
    model.load_state_dict(state["ema"])
    model.eval()
    trajectories, spacing, provenance = get_data(model_cfg)
    if provenance != state["provenance"]:
        raise ValueError("Checkpoint and evaluation source fingerprints differ")
    scale = state["scale_nm"]
    reference = np.concatenate(trajectories)
    rng = np.random.default_rng(cfg.seed)
    initial = center(torch.tensor(reference[rng.integers(len(reference), size=cfg.evaluation.chains)])) / scale
    started = time.perf_counter()
    results: dict[str, Any] = {"status": "diagnostic", "device": "cpu", "seed": cfg.seed,
                              "scientific_acceptance": "incomplete", "lags": [],
                              "checkpoint_sha256": digest(cfg.evaluation.checkpoint)}
    if model_cfg.data.kind == "synthetic":
        samples = rollout(model, initial, cfg.evaluation.lags[0], cfg.evaluation.transitions,
                          cfg.evaluation.solver_steps).numpy() * scale
        np.save(output / "samples_nm.npy", samples)
        fig, ax = plt.subplots()
        ax.hist(reference.ravel(), bins=cfg.evaluation.bins, density=True, alpha=.5, label="Synthetic reference")
        ax.hist(samples[:, 1:].ravel(), bins=cfg.evaluation.bins, density=True, alpha=.5, label="ITO")
        ax.set(xlabel="Coordinate (nm)", ylabel="Density", title="Software smoke test, not molecular reproduction")
        ax.legend()
        fig.savefig(output / "distribution.png", dpi=140)
        plt.close(fig)
        results.update({"generated_frames": int(samples.shape[0] * (samples.shape[1] - 1)),
                        "finite_samples": bool(np.isfinite(samples).all()),
                        "max_centroid_nm": float(np.abs(samples.mean(axis=-2)).max()),
                        "scientific_acceptance": "not_applicable_synthetic"})
    else:
        angle_trajs = [torsions(t, model_cfg.data.topology, model_cfg.data.torsions) for t in trajectories]
        ref_hist = histogram(np.concatenate(angle_trajs), cfg.evaluation.bins)
        # Molar gas constant via OpenMM; BIPM SI Brochure (units.md).
        from openmm import unit
        rt = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilojoule_per_mole / unit.kelvin) * model_cfg.data.temperature_k
        fig, axes = plt.subplots(len(cfg.evaluation.lags), 2, squeeze=False, figsize=(8, 3 * len(cfg.evaluation.lags)))
        for row, lag in enumerate(cfg.evaluation.lags):
            if lag > model_cfg.model.max_lag:
                raise ValueError("Evaluation physical lag exceeds model conditioning range")
            samples = rollout(model, initial, lag, cfg.evaluation.transitions, cfg.evaluation.solver_steps).numpy() * scale
            np.save(output / f"rollout_lag_{lag}_nm.npy", samples)
            angles = [torsions(chain, model_cfg.data.topology, model_cfg.data.torsions) for chain in samples]
            generated = histogram(np.concatenate([a[1:] for a in angles]), cfg.evaluation.bins)
            reference_kinetics = msm_timescale(angle_trajs, lag, spacing, cfg.evaluation.msm_bins, cfg.evaluation.min_transitions)
            model_kinetics = msm_timescale(angles, 1, lag * spacing, cfg.evaluation.msm_bins, cfg.evaluation.min_transitions)
            ratio = None
            if reference_kinetics["timescale_ps"] and model_kinetics["timescale_ps"]:
                ratio = model_kinetics["timescale_ps"] / reference_kinetics["timescale_ps"]
            divergence = jsd(ref_hist, generated)
            results["lags"].append({"lag_ps": lag * spacing, "jsd_nats": divergence,
                "jsd_bits": divergence / np.log(2), "reference_msm": reference_kinetics,
                "model_msm": model_kinetics, "timescale_ratio": ratio,
                "reference_vamp2": vamp_score(angle_trajs, lag), "model_vamp2": vamp_score(angles, 1)})
            for ax, counts, title in zip(axes[row], (ref_hist, generated), ("Reference MD", f"ITO: lag {lag * spacing:g} ps")):
                probability = np.ma.masked_equal(counts / counts.sum(), 0)
                free_energy = -rt * np.ma.log(probability)
                free_energy -= free_energy.min()
                plot = ax.imshow(free_energy.T, origin="lower", extent=(-np.pi, np.pi, -np.pi, np.pi),
                                 vmin=0, vmax=cfg.evaluation.fes_max_kj_mol)
                ax.set(xlabel="phi (rad)", ylabel="psi (rad)", title=title)
                fig.colorbar(plot, ax=ax, label="kJ/mol")
            np.savez(output / f"histograms_lag_{lag}.npz", reference=ref_hist, generated=generated)
        fig.tight_layout()
        fig.savefig(output / "free_energy.png", dpi=140)
        plt.close(fig)
        branch = initial[:1].expand(cfg.evaluation.ck_branches, -1, -1).clone()
        small, k = cfg.evaluation.ck_lag, cfg.evaluation.ck_steps
        if small * k > model_cfg.model.max_lag:
            raise ValueError("CK total lag exceeds training range")
        direct = rollout(model, branch, small * k, 1, cfg.evaluation.solver_steps)[:, -1].numpy() * scale
        nested = rollout(model, branch, small, k, cfg.evaluation.solver_steps)[:, -1].numpy() * scale
        dh, nh = [histogram(torsions(x, model_cfg.data.topology, model_cfg.data.torsions), cfg.evaluation.bins)
                  for x in (direct, nested)]
        np.savez(output / "ck_samples_nm.npz", direct=direct, nested=nested, initial=branch[0].numpy() * scale)
        results["ck"] = {"jsd_nats": jsd(dh, nh), "branches": cfg.evaluation.ck_branches,
                         "total_lag_ps": small * k * spacing, "initial_conditions": 1}
        results["acceptance_missing"] = ["three reference basin recovery checks", "bootstrap confidence intervals",
                                          "CK per initial basin and finite-sample control", "independent reference convergence"]
    results["wall_seconds"] = time.perf_counter() - started
    write_json(output / "metrics.json", results)
    return results
