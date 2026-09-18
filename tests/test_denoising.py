"""Known epsilon errors produce the expected reconstruction amplification."""
import pytest
import torch

from tito_repro.eval.denoising import noise_metrics


def test_noise_error_amplification():
    target, noise = torch.randn(2, 4, 3), torch.randn(2, 4, 3)
    alpha = torch.tensor(.01)
    noisy = alpha.sqrt() * target + (1 - alpha).sqrt() * noise
    perfect = noise_metrics(noisy, target, noise, noise, alpha)
    assert perfect["clean_reconstruction_mse"] < 1e-10
    imperfect = noise_metrics(noisy, target, noise, noise + .1, alpha)
    assert imperfect["epsilon_error_amplification"] == pytest.approx(99)
    assert imperfect["clean_reconstruction_mse"] == pytest.approx(99 * imperfect["epsilon_mse"], rel=1e-5)
