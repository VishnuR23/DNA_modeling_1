"""Boundary-safe sampling with explicit physical timestamps."""
from pathlib import Path

import numpy as np
import torch


def center(x: torch.Tensor) -> torch.Tensor:
    """Remove arithmetic atom centroid from [..., atoms, 3]; preserve input length units."""
    return x - x.mean(dim=-2, keepdim=True)


def load_trajectories(path: str | Path) -> tuple[list[np.ndarray], float]:
    """Read NPZ positions [replicas, frames, atoms, 3] in nm and spacing in ps."""
    with np.load(path, allow_pickle=False) as f:
        positions = f["positions_nm"].astype(np.float32)
        spacing = float(f["spacing_ps"])
    if positions.ndim != 4 or positions.shape[-1] != 3 or not np.isfinite(positions).all():
        raise ValueError("Expected finite [replicas, frames, atoms, 3] coordinates")
    if spacing <= 0:
        raise ValueError("Positive frame spacing required")
    return list(positions), spacing


class LagPairs:
    """Draw [batch, atoms, 3] position pairs (nm), lag multiples, without crossing replicas."""

    def __init__(self, trajectories: list[np.ndarray], max_lag: int, seed: int,
                 law: str = "disexp", fixed_lag: int | None = None) -> None:
        """Index [T,A,3] nm replicas; max_lag/fixed_lag are integer stored-frame counts."""
        if max_lag < 1 or law not in {"disexp", "uniform", "fixed"}:
            raise ValueError("Invalid max_lag or lag law")
        if not trajectories or any(len(t) <= max_lag for t in trajectories):
            raise ValueError("Every trajectory must have more frames than max_lag")
        if law == "fixed" and (fixed_lag is None or not 1 <= fixed_lag <= max_lag):
            raise ValueError("Fixed lag must lie in [1,max_lag]")
        self.trajs = [torch.as_tensor(t, dtype=torch.float32) for t in trajectories]
        self.counts = torch.tensor([len(t) - max_lag for t in trajectories])
        self.ends = self.counts.cumsum(0)
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.max_lag, self.law, self.fixed_lag = max_lag, law, fixed_lag

    def sample(self, batch: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return x0,x1 [B,A,3] in nm and integer lag [B] in stored-frame multiples."""
        ids = torch.randint(int(self.ends[-1]), (batch,), generator=self.generator)
        replicas = torch.searchsorted(self.ends, ids, right=True)
        starts = ids - torch.cat([torch.zeros(1, dtype=torch.long), self.ends[:-1]])[replicas]
        if self.law == "fixed":
            lag = torch.full((batch,), self.fixed_lag, dtype=torch.long)
        elif self.law == "uniform":
            lag = torch.randint(1, self.max_lag + 1, (batch,), generator=self.generator)
        else:
            lag = torch.exp(torch.rand(batch, generator=self.generator) * np.log(self.max_lag)).long()
        x0 = torch.stack([self.trajs[r][s] for r, s in zip(replicas, starts)])
        x1 = torch.stack([self.trajs[r][s + n] for r, s, n in zip(replicas, starts, lag)])
        return x0, x1, lag


def synthetic_trajectories(replicas: int, frames: int, atoms: int, seed: int,
                           correlation: float, scale_nm: float) -> list[np.ndarray]:
    """Generate stationary Gaussian AR(1) smoke data [replicas,frames,atoms,3], in nm."""
    if not 0 <= correlation < 1 or scale_nm <= 0:
        raise ValueError("Invalid synthetic correlation or length scale")
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(replicas, frames, atoms, 3)).astype(np.float32) * scale_nm
    for i in range(1, frames):
        x[:, i] = correlation * x[:, i - 1] + np.sqrt(1 - correlation**2) * x[:, i]
    return list(x - x.mean(axis=2, keepdims=True))
