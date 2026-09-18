"""Exploratory conditional flow; independent Gaussian coupling, no OT alignment."""
import torch
from torch import nn

from tito_repro.data.pairs import center
from tito_repro.models.denoiser import Denoiser


def linear_path(prior: torch.Tensor, target: torch.Tensor,
                time: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return centered interpolation and its exact derivative; time is [batch]."""
    prior, target = center(prior), center(target)
    t = time[:, None, None]
    return (1 - t) * prior + t * target, target - prior


class ConditionalFlow(nn.Module):
    """Variable atom count, one topology per batch; coordinates are dimensionless."""

    def __init__(self, width: int = 8, max_lag: int = 16) -> None:
        super().__init__()
        self.field = Denoiser(atoms=119, width=width, condition_layers=1,
                              score_layers=1, max_lag=max_lag,
                              diffusion_steps=1, radial_scale=3.0)

    def forward(self, x: torch.Tensor, condition: torch.Tensor, lag: torch.Tensor,
                time: torch.Tensor, elements: torch.Tensor) -> torch.Tensor:
        """Predict velocity at continuous time [0,1], using atomic numbers [A]."""
        if x.shape != condition.shape or x.ndim != 3 or x.shape[-1] != 3:
            raise ValueError("Matching [batch,atoms,3] coordinates required")
        if elements.shape != (x.shape[1],) or elements.dtype != torch.long:
            raise ValueError("Atomic numbers must be long [atoms]")
        if torch.any((elements < 1) | (elements > 118)):
            raise ValueError("Atomic numbers must be in [1,118]")
        if time.shape != (len(x),) or not torch.isfinite(time).all() or torch.any((time < 0) | (time > 1)):
            raise ValueError("Flow time must be finite [batch] in [0,1]")
        return self.field(x, condition, lag, time, elements)

    def loss(self, condition: torch.Tensor, target: torch.Tensor,
             lag: torch.Tensor, elements: torch.Tensor) -> torch.Tensor:
        time = torch.rand(len(target), dtype=target.dtype)
        x, velocity = linear_path(torch.randn_like(target), target, time)
        return (self(x, condition, lag, time, elements) - velocity).square().sum(-1).mean()

    @torch.no_grad()
    def sample(self, condition: torch.Tensor, lag: torch.Tensor, elements: torch.Tensor,
               steps: int = 20, prior: torch.Tensor | None = None) -> torch.Tensor:
        """Heun integration 0→1, two field evaluations per step; fresh Gaussian prior."""
        if not isinstance(steps, int) or steps < 1:
            raise ValueError("Positive integer solver steps required")
        if prior is not None and prior.shape != condition.shape:
            raise ValueError("Prior and condition shapes must match")
        x = center(torch.randn_like(condition) if prior is None else prior.clone())
        for index in range(steps):
            time = torch.full((len(x),), index / steps, dtype=x.dtype)
            next_time = torch.full_like(time, (index + 1) / steps)
            first = self(x, condition, lag, time, elements)
            second = self(x + first / steps, condition, lag, next_time, elements)
            x = center(x + (first + second) / (2 * steps))
            if not torch.isfinite(x).all():
                raise FloatingPointError("Nonfinite flow sample")
        return x
