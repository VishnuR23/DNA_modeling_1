"""Noise-level diagnostics on cached training data, not a held-out validation set."""
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from tito_repro.data.pairs import LagPairs, center
from tito_repro.train.phase1 import get_data, make_model
from tito_repro.utils.runtime import digest, write_json


def noise_metrics(noisy: torch.Tensor, target: torch.Tensor, noise: torch.Tensor,
                  prediction: torch.Tensor, alpha: torch.Tensor) -> dict[str, float]:
    """Cartesian-sum/atom-mean errors; input and reconstruction are dimensionless."""
    clean = (noisy - (1 - alpha).sqrt() * prediction) / alpha.sqrt()
    mse = lambda x: float(x.square().sum(-1).mean())
    return {"epsilon_mse": mse(prediction - noise), "zero_epsilon_mse": mse(noise),
            "identity_epsilon_mse": mse(noisy - noise),
            "clean_reconstruction_mse": mse(clean - target),
            "epsilon_error_amplification": float((1 - alpha) / alpha)}


@torch.no_grad()
def diagnose_denoising(cfg: DictConfig, output: Path) -> dict:
    if not cfg.diagnostic.checkpoint:
        raise ValueError("Set diagnostic.checkpoint to a local Phase 1 checkpoint")
    state = torch.load(cfg.diagnostic.checkpoint, map_location="cpu", weights_only=False)
    model_cfg = OmegaConf.create(state["config"])
    model = make_model(model_cfg)
    model.load_state_dict(state["ema"])
    model.eval()
    trajectories, _, provenance = get_data(model_cfg)
    if provenance != state["provenance"]:
        raise ValueError("Checkpoint and diagnostic data fingerprints differ")
    if cfg.diagnostic.samples < 1 or not cfg.diagnostic.levels or any(
        not 1 <= d <= model.steps for d in cfg.diagnostic.levels
    ):
        raise ValueError("Positive sample count and valid noise levels required")
    pairs = LagPairs(trajectories, model_cfg.model.max_lag, cfg.seed,
                     model_cfg.train.lag_law, model_cfg.train.fixed_lag)
    x0, x1, lag = pairs.sample(cfg.diagnostic.samples)
    condition, target = center(x0) / state["scale_nm"], center(x1) / state["scale_nm"]
    rng = torch.Generator().manual_seed(cfg.seed)
    noise = center(torch.randn(target.shape, generator=rng))
    rows = []
    for level in cfg.diagnostic.levels:
        alpha = model.alpha_bar[level]
        noisy = alpha.sqrt() * target + (1 - alpha).sqrt() * noise
        prediction = model.denoiser(noisy, condition, lag, torch.full_like(lag, level))
        rows.append({"noise_level": level, "alpha_bar": float(alpha),
                     **noise_metrics(noisy, target, noise, prediction, alpha)})
    result = {"status": "diagnostic", "device": "cpu", "seed": cfg.seed,
              "samples": cfg.diagnostic.samples, "checkpoint_sha256": digest(cfg.diagnostic.checkpoint),
              "checkpoint_steps": state["step"], "data": provenance, "levels": rows,
              "scientific_acceptance": "not_evaluated",
              "limitations": ["training-source pairs, not held-out validation",
                              "forward-noised reference states, not autoregressive samples",
                              "single seed and finite sample diagnostics"]}
    write_json(output / "metrics.json", result)
    return result
