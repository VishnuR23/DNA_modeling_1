"""Residual epsilon parameterization must retain a consistent loss/sampler API."""
import torch

from tito_repro.data.pairs import center
from tito_repro.models.denoiser import Denoiser
from tito_repro.utils.runtime import seed_cpu


def test_skip_changes_full_prediction_by_centered_noisy_input():
    seed_cpu(7, 1)
    plain = Denoiser(4, 8, 1, 1, 16, 1000, 3.)
    residual = Denoiser(4, 8, 1, 1, 16, 1000, 3., epsilon_skip=True)
    residual.load_state_dict(plain.state_dict())
    x, c = torch.randn(2, 4, 3), torch.randn(2, 4, 3)
    lag, level = torch.tensor([1, 4]), torch.tensor([100, 1000])
    torch.testing.assert_close(residual(x, c, lag, level) - plain(x, c, lag, level), center(x))
    # A zero residual is the identity epsilon baseline, not a clean-state estimate.
    for parameter in residual.parameters():
        parameter.data.zero_()
    torch.testing.assert_close(residual(x, c, lag, level), center(x))
