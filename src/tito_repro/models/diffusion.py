"""Consistently indexed DDPM training and deterministic DDIM sampling."""
import torch
from torch import nn

from tito_repro.data.pairs import center
from tito_repro.models.denoiser import Denoiser


class Diffusion(nn.Module):
    """Conditional DDPM on centered [B,A,3] dimensionless coordinates."""

    def __init__(self, denoiser: Denoiser, steps: int, beta_start: float, beta_end: float) -> None:
        super().__init__()
        self.denoiser, self.steps = denoiser, steps
        # ITO Appendix E.3: sigmoid schedule on [-8,-4], 1000 levels in scientific configs.
        beta = torch.sigmoid(torch.linspace(beta_start, beta_end, steps))
        self.register_buffer("alpha_bar", torch.cat((torch.ones(1), (1 - beta).cumprod(0))))

    def loss(self, x0: torch.Tensor, x1: torch.Tensor, lag: torch.Tensor) -> torch.Tensor:
        """Return Cartesian-sum/atom-mean MSE; x0,x1 [B,A,3] are dimensionless."""
        d = torch.randint(1, self.steps + 1, (len(x0),))
        eps = center(torch.randn_like(x1))
        alpha = self.alpha_bar[d, None, None]
        noisy = alpha.sqrt() * center(x1) + (1 - alpha).sqrt() * eps
        prediction = self.denoiser(noisy, x0, lag, d)
        return (prediction - eps).square().sum(-1).mean()

    @torch.no_grad()
    def sample(self, condition: torch.Tensor, lag: torch.Tensor, evaluations: int) -> torch.Tensor:
        """DDIM eta=0 sample [B,A,3]; dimensionless coordinates, lag [B] in frame multiples."""
        if not 1 <= evaluations <= self.steps:
            raise ValueError("Sampling evaluations must be in [1,diffusion_steps]")
        x = center(torch.randn_like(condition))
        grid = torch.linspace(self.steps, 0, evaluations + 1).round().long()
        for d, previous in zip(grid[:-1], grid[1:]):
            eps = self.denoiser(x, condition, lag, d.expand(len(x)))
            a, ap = self.alpha_bar[d], self.alpha_bar[previous]
            clean = (x - (1 - a).sqrt() * eps) / a.sqrt()
            x = center(ap.sqrt() * clean + (1 - ap).sqrt() * eps)
        return x
