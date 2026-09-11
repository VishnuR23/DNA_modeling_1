"""Physical-time rollouts; solver time is never interpreted as MD time."""
import torch

from tito_repro.models.diffusion import Diffusion


@torch.no_grad()
def rollout(model: Diffusion, initial: torch.Tensor, lag: int,
            transitions: int, evaluations: int) -> torch.Tensor:
    """Return [chains,transitions+1,atoms,3] dimensionless coordinates; lag is frames."""
    if transitions < 1 or lag < 1:
        raise ValueError("Positive transition count and lag required")
    model.eval()
    x, frames = initial.clone(), [initial.clone()]
    lags = torch.full((len(initial),), lag, dtype=torch.long)
    for _ in range(transitions):
        x = model.sample(x, lags, evaluations)
        if not torch.isfinite(x).all():
            raise FloatingPointError("Nonfinite generated coordinates; rollout aborted")
        frames.append(x.clone())
    return torch.stack(frames, dim=1)
