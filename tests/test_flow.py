"""Analytic flow and symmetry checks, independent of molecular acceptance."""
import pytest
import torch

from tito_repro.data.pairs import center
from tito_repro.models.flow import ConditionalFlow, linear_path
from tito_repro.utils.runtime import seed_cpu


def test_path_endpoints_and_derivative():
    seed_cpu(1, 1)
    a, b = torch.randn(2, 4, 3, dtype=torch.float64), torch.randn(2, 4, 3, dtype=torch.float64)
    t = torch.tensor([.2, .7], dtype=torch.float64)
    assert torch.allclose(linear_path(a, b, t * 0)[0], center(a))
    assert torch.allclose(linear_path(a, b, t * 0 + 1)[0], center(b))
    finite_difference = (linear_path(a, b, t + 1e-5)[0] - linear_path(a, b, t - 1e-5)[0]) / 2e-5
    torch.testing.assert_close(finite_difference, linear_path(a, b, t)[1])


def test_flow_variable_size_rotation_and_permutation():
    seed_cpu(2, 1)
    model = ConditionalFlow().double()
    for atoms in (3, 5):
        x, c = torch.randn(2, atoms, 3, dtype=torch.float64), torch.randn(2, atoms, 3, dtype=torch.float64)
        elements = torch.tensor([6, 1, 8, 1, 7][:atoms])
        lag, time = torch.tensor([1, 4]), torch.tensor([.2, .8], dtype=torch.float64)
        q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
        q[:, -1] *= torch.linalg.det(q)
        perm = torch.randperm(atoms)
        expected = model(x, c, lag, time, elements)
        actual = model(x[:, perm] @ q + 3, c[:, perm] @ q - 2, lag, time, elements[perm])
        torch.testing.assert_close(actual, expected[:, perm] @ q, atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(expected.mean(1), torch.zeros(2, 3, dtype=torch.float64), atol=1e-12, rtol=0)


def test_heun_integrates_known_time_dependent_field():
    class AnalyticFlow(ConditionalFlow):
        def forward(self, x, condition, lag, time, elements):
            return 2 * time[:, None, None] * center(condition)

    model = AnalyticFlow()
    condition, prior = torch.randn(2, 3, 3), torch.randn(2, 3, 3)
    result = model.sample(condition, torch.ones(2), torch.tensor([6, 1, 1]), steps=4, prior=prior)
    torch.testing.assert_close(result, center(prior) + center(condition))
    with pytest.raises(ValueError, match="solver steps"):
        model.sample(condition, torch.ones(2), torch.tensor([6, 1, 1]), steps=0)


def test_flow_loss_backward_and_sample():
    seed_cpu(3, 1)
    model = ConditionalFlow()
    c, target = torch.randn(2, 4, 3), torch.randn(2, 4, 3)
    lag, elements = torch.tensor([1, 4]), torch.tensor([6, 1, 1, 8])
    loss = model.loss(c, target, lag, elements)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    result = model.sample(c, lag, elements, steps=2)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result.mean(1), torch.zeros(2, 3), atol=1e-6, rtol=0)
