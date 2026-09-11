"""Single-molecule ITO noise estimator, using centered dimensionless coordinates."""
import torch
from torch import nn

from tito_repro.data.pairs import center
from tito_repro.vendor.ito.backbone import ChiroBlock, fourier, mlp


class Denoiser(nn.Module):
    """Lag-conditioned SE(3) denoiser for tensors [batch,atoms,3]; CPU only."""

    def __init__(self, atoms: int, width: int, condition_layers: int, score_layers: int,
                 max_lag: int, diffusion_steps: int, radial_scale: float,
                 condition_readout: bool = True) -> None:
        """Configure [B,atoms,3] noise prediction; lag uses frames and radial_scale is dimensionless."""
        super().__init__()
        if width < 2 or width % 2 or min(atoms, condition_layers, score_layers) < 1:
            raise ValueError("Positive sizes and even feature width required")
        self.width, self.max_lag, self.diffusion_steps = width, max_lag, diffusion_steps
        self.atom_embedding = nn.Embedding(atoms, width)
        self.condition_mix, self.time_mix = mlp(2 * width, width, width), mlp(2 * width, width, width)
        self.condition = nn.ModuleList([ChiroBlock(width, radial_scale) for _ in range(condition_layers)])
        self.condition_readout = condition_readout
        if condition_readout:
            self.condition_scalar = mlp(width, width, 2 * width)
            self.condition_vector = nn.Linear(width, width, bias=False)
        self.score = nn.ModuleList([ChiroBlock(width, radial_scale) for _ in range(score_layers)])
        self.readout = nn.Linear(width, 1, bias=False)
        self.gate = mlp(width, width, 1)

    def forward(self, noisy: torch.Tensor, condition: torch.Tensor,
                lag: torch.Tensor, noise_step: torch.Tensor) -> torch.Tensor:
        """Predict mean-free noise [B,A,3]; lag [B] is frame multiples, step [B] dimensionless."""
        if noisy.device.type != "cpu":
            raise ValueError("CPU tensors required")
        noisy, condition = center(noisy), center(condition)
        b, a, _ = noisy.shape
        embedding = self.atom_embedding(torch.arange(a)).unsqueeze(0).expand(b, -1, -1)
        times = fourier(lag.to(noisy.dtype), self.width, self.max_lag)[:, None].expand(-1, a, -1)
        s = self.condition_mix(torch.cat((embedding, times), dim=-1))
        v = torch.zeros(b, a, self.width, 3, dtype=noisy.dtype)
        for block in self.condition:
            s, v = block(condition, s, v)
        if self.condition_readout:
            s, gate = self.condition_scalar(s).chunk(2, dim=-1)
            v = self.condition_vector(v.transpose(-1, -2)).transpose(-1, -2) * gate.unsqueeze(-1)
        times = fourier(noise_step.to(noisy.dtype), self.width, self.diffusion_steps)[:, None].expand(-1, a, -1)
        s = self.time_mix(torch.cat((s, times), dim=-1))
        for block in self.score:
            s, v = block(noisy, s, v)
        output = self.readout(v.transpose(-1, -2)).squeeze(-1) * self.gate(s)
        return center(noisy + output)
