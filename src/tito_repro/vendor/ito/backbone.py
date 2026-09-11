# Adapted from olsson-group/ito (MIT), commit 8310311250e0e3893bc10bbd80ab67d2eab4ae6a.
# Authors: olsson-group, 2023. See adjacent LICENSE. Dense CPU tensor implementation,
# explicit shapes, corrected Fourier indexing, and centering added by tito-repro.
"""ChiroPaiNN blocks with proper-rotation equivariance and complete directed graphs."""
import math

import torch
from torch import nn


def fourier(x: torch.Tensor, width: int, length: float) -> torch.Tensor:
    """Encode scalar [...]-shaped dimensionless x as [...,width] sine/cosine features."""
    ranks = torch.arange(1, width // 2 + 1, dtype=x.dtype, device=x.device)
    phase = x.unsqueeze(-1) * ranks * (math.pi / length)
    return torch.stack((phase.cos(), phase.sin()), dim=-1).flatten(-2)


def mlp(inputs: int, width: int, outputs: int) -> nn.Sequential:
    """Build a scalar feature MLP [...,inputs]→[...,outputs]; dimensionless features."""
    return nn.Sequential(nn.Linear(inputs, width), nn.SiLU(), nn.Linear(width, width),
                         nn.SiLU(), nn.Linear(width, outputs))


class ChiroBlock(nn.Module):
    """Complete-graph message/update block: scalars [B,A,F], vectors [B,A,F,3]."""

    def __init__(self, width: int, radial_scale: float) -> None:
        """Set feature width F and dimensionless radial Fourier scale for [B,A,F,3] vectors."""
        super().__init__()
        self.width, self.radial_scale = width, radial_scale
        self.phi, self.radial = mlp(width, width, 4 * width), mlp(width, width, 4 * width)
        self.u, self.v = nn.Linear(width, width, bias=False), nn.Linear(width, width, bias=False)
        self.update = mlp(2 * width, width, 3 * width)

    def forward(self, x: torch.Tensor, s: torch.Tensor,
                v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Update scalar/vector channels from dimensionless positions x [B,A,3]."""
        atoms = x.shape[1]
        dst, src = torch.where(~torch.eye(atoms, dtype=torch.bool, device=x.device))
        r = x[:, src] - x[:, dst]
        distance = r.norm(dim=-1)
        direction = r / (1 + distance.unsqueeze(-1))
        gates = self.phi(s[:, src]) * self.radial(fourier(distance, self.width, self.radial_scale))
        g, cross_g, dir_g, scalar_g = gates.chunk(4, dim=-1)
        dv = (g.unsqueeze(-1) * v[:, src] + dir_g.unsqueeze(-1) * direction.unsqueeze(-2)
              + cross_g.unsqueeze(-1) * torch.cross(direction.unsqueeze(-2).expand_as(v[:, dst]),
                                                    v[:, dst], dim=-1))
        v = v + torch.zeros_like(v).index_add(1, dst, dv)
        s = s + torch.zeros_like(s).index_add(1, dst, scalar_g * s[:, src])
        uv = self.u(v.transpose(-1, -2)).transpose(-1, -2)
        vv = self.v(v.transpose(-1, -2)).transpose(-1, -2)
        norm = torch.linalg.vector_norm(vv, dim=-1)
        gate, scale, additive = self.update(torch.cat((s, norm), dim=-1)).chunk(3, dim=-1)
        return s + scale * norm.square() + additive, v + gate.unsqueeze(-1) * uv
