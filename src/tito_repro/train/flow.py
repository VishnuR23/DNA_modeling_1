"""Bounded synthetic Phase 2 development pilot; no molecular claims or downloads."""
import logging
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from tito_repro.data.pairs import center
from tito_repro.models.flow import ConditionalFlow, linear_path
from tito_repro.utils.runtime import write_json


def pairs(atoms: int, batch: int, lag: int, correlation: float,
          generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact centered Gaussian AR(1) pairs, dimensionless; lag counts synthetic frames."""
    condition = center(torch.randn(batch, atoms, 3, generator=generator))
    rho = correlation ** lag
    target = rho * condition + (1 - rho ** 2) ** .5 * center(torch.randn(batch, atoms, 3, generator=generator))
    return condition, target, torch.full((batch,), lag, dtype=torch.long)


def element_ids(atoms: int, vocabulary: list[int]) -> torch.Tensor:
    """Synthetic labels only: these random clouds do not represent chemical structures."""
    return torch.tensor([vocabulary[i % len(vocabulary)] for i in range(atoms)], dtype=torch.long)


@torch.no_grad()
def diagnose(model: ConditionalFlow, cfg: DictConfig) -> list[dict]:
    """Fixed independent pairs/noise for before/after loss and conditional moments."""
    model.eval()
    rows = []
    rng = torch.Generator().manual_seed(cfg.seed + 1000)
    for atoms in cfg.flow.evaluation_atoms:
        for lag in cfg.flow.lags:
            condition, target, lags = pairs(atoms, cfg.flow.evaluation_samples, lag, cfg.flow.correlation, rng)
            elements = element_ids(atoms, cfg.flow.elements)
            prior = center(torch.randn(condition.shape, generator=rng))
            times = torch.rand(len(condition), generator=rng)
            x, velocity = linear_path(prior, target, times)
            prediction = model(x, condition, lags, times, elements)
            samples = model.sample(condition, lags, elements, cfg.flow.solver_steps, prior=prior)
            expected_mean = cfg.flow.correlation ** lag * condition
            expected_variance = (1 - cfg.flow.correlation ** (2 * lag)) * (1 - 1 / atoms)
            rows.append({
                "atoms": atoms, "lag_frames": lag, "held_out_size": atoms not in cfg.flow.train_atoms,
                "velocity_mse": float((prediction - velocity).square().sum(-1).mean()),
                "zero_velocity_mse": float(velocity.square().sum(-1).mean()),
                "conditional_residual_second_moment": float((samples - expected_mean).square().mean()),
                "expected_conditional_variance": expected_variance,
                "reference_residual_second_moment": float((target - expected_mean).square().mean()),
                "finite_samples": bool(torch.isfinite(samples).all()),
                "max_centroid": float(samples.mean(-2).abs().max()),
            })
    return rows


def train_flow(cfg: DictConfig, output: Path) -> dict:
    """Train alternating unpadded sizes/lags with a bounded local CPU budget."""
    f = cfg.flow
    if (not f.train_atoms or not f.evaluation_atoms or not f.lags or not f.elements
            or min(f.train_atoms) < 2 or min(f.evaluation_atoms) < 2 or min(f.lags) < 1
            or not 0 <= f.correlation < 1 or f.steps < 1 or f.max_seconds <= 0
            or f.batch_size < 1 or f.checkpoint_every < 1 or f.evaluation_samples < 1):
        raise ValueError("Invalid synthetic flow configuration")
    model = ConditionalFlow(f.width, max(f.lags))
    before = diagnose(model, cfg)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=f.learning_rate)
    rng = torch.Generator().manual_seed(cfg.seed + 1)
    combinations = [(a, lag) for a in f.train_atoms for lag in f.lags]
    losses = []

    def save() -> None:
        temporary = output / "checkpoint.tmp"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "steps": len(losses), "config": OmegaConf.to_container(cfg, resolve=True),
                    "torch_rng": torch.get_rng_state(), "pair_rng": rng.get_state(),
                    "losses": losses, "purpose": "synthetic_flow_prototype"}, temporary)
        temporary.replace(output / "checkpoint.pt")

    started = time.perf_counter()
    for step in range(f.steps):
        if time.perf_counter() - started >= f.max_seconds:
            break
        atoms, lag = combinations[step % len(combinations)]
        condition, target, lags = pairs(atoms, f.batch_size, lag, f.correlation, rng)
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(condition, target, lags, element_ids(atoms, f.elements))
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite flow training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
        if len(losses) % f.checkpoint_every == 0:
            save()
            logging.info("flow step=%d loss=%.4f elapsed_s=%.2f", len(losses), losses[-1], time.perf_counter() - started)
    save()
    elapsed = time.perf_counter() - started
    after = diagnose(model, cfg)
    np.save(output / "loss.npy", losses)
    result = {"status": "completed" if len(losses) == f.steps else "time_limit",
              "device": "cpu", "phase": "phase2_exploration", "seed": cfg.seed,
              "scientific_acceptance": "not_applicable_synthetic", "data": "generated_gaussian_ar1",
              "coupling": "independent_gaussian_linear_path", "solver": "heun",
              "field_evaluations_per_sample": 2 * f.solver_steps,
              "train_atoms": list(f.train_atoms), "steps": len(losses),
              "training_wall_seconds": elapsed, "before": before, "after": after,
              "limitations": ["no molecular data or chemical validity", "no optimal-transport alignment",
                              "no peptide transferability claim", "point estimates without uncertainty intervals",
                              "checkpoint resume not implemented for this prototype"]}
    write_json(output / "metrics.json", result)
    return result
