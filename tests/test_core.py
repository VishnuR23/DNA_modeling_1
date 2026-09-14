"""CPU numerical contracts; physical quantities are explicitly labeled."""
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from tito_repro.data.pairs import LagPairs, center, synthetic_trajectories
from tito_repro.eval.metrics import jsd, msm_timescale
from tito_repro.eval.gates import phase1_gate_report
from tito_repro.train.phase1 import evaluate, make_model, train
from tito_repro.utils.runtime import seed_cpu


@pytest.fixture
def config() -> DictConfig:
    """Compose synthetic configuration with dimensionless model and nm data."""
    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).parents[1] / "configs")):
        return compose(config_name="config", overrides=["experiment=smoke"])


def test_equivariance(config: DictConfig) -> None:
    """Both condition and noisy [B,A,3] rotate together; translations are removed."""
    seed_cpu(4, 1)
    net = make_model(config).double().denoiser
    x, c = torch.randn(2, 4, 3, dtype=torch.float64), torch.randn(2, 4, 3, dtype=torch.float64)
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    q[:, -1] *= torch.linalg.det(q)
    lag, d = torch.tensor([1, 8]), torch.tensor([10, 999])
    expected = net(x, c, lag, d) @ q
    actual = net(x @ q + 3, c @ q - 2, lag, d)
    torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(actual.mean(1), torch.zeros(2, 3, dtype=torch.float64), atol=1e-12, rtol=0)


def test_pair_boundaries_and_rng() -> None:
    """Replica identity remains intact even at its final admissible starts."""
    trajectories = [np.full((20, 3, 3), value, dtype=np.float32) for value in (10, 50)]
    sampler = LagPairs(trajectories, 8, 4)
    x0, x1, lag = sampler.sample(1000)
    assert x0.shape == x1.shape == (1000, 3, 3)
    assert torch.equal(x0, x1)
    assert lag.min() >= 1 and lag.max() < 8
    state = sampler.generator.get_state()
    first = sampler.sample(4)
    sampler.generator.set_state(state)
    assert all(torch.equal(a, b) for a, b in zip(first, sampler.sample(4)))
    with pytest.raises(ValueError):
        LagPairs(trajectories, 20, 1)


def test_schedule_and_sample(config: DictConfig) -> None:
    """Cumulative alpha includes every beta; centered deterministic DDIM stays finite."""
    seed_cpu(2, 1)
    model = make_model(config)
    assert model.alpha_bar[0] == 1
    beta = torch.sigmoid(torch.linspace(-8, -4, 1000))
    torch.testing.assert_close(model.alpha_bar[1:], (1 - beta).cumprod(0))
    c = center(torch.randn(2, 4, 3))
    output = model.sample(c, torch.tensor([1, 2]), 10)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output.mean(1), torch.zeros(2, 3), atol=1e-5, rtol=0)
    with pytest.raises(ValueError):
        model.sample(c, torch.tensor([1, 2]), 1001)


def test_denoiser_returns_noise_not_noisy_state(config: DictConfig) -> None:
    """Denoiser output is a centered epsilon prediction, not x plus epsilon."""
    seed_cpu(22, 1)
    model = make_model(config)
    x = center(torch.randn(2, 4, 3))
    out = model.denoiser(x, x, torch.tensor([1, 2]), torch.tensor([10, 20]))
    assert out.shape == x.shape
    torch.testing.assert_close(out.mean(1), torch.zeros(2, 3), atol=1e-5, rtol=0)
    assert not torch.allclose(out, x)


def test_histogram_and_disconnected_msm() -> None:
    """JSD conventions and disconnected kinetics cannot silently report success."""
    assert jsd(np.array([1, 0]), np.array([0, 1])) == pytest.approx(np.log(2))
    assert jsd(np.ones(4), np.ones(4)) == pytest.approx(0)
    with pytest.raises(ValueError):
        jsd(np.zeros(3), np.zeros(3))
    result = msm_timescale([np.full((50, 2), -2.), np.full((50, 2), 2.)], 1, 1., 4, 10)
    assert result["status"] == "inconclusive" and result["timescale_ps"] is None


def test_cpu_smoke_and_exact_resume(config: DictConfig, tmp_path: Path) -> None:
    """End-to-end train/sample plus resume preserving optimizer, EMA and RNG in <120 s."""
    started = time.perf_counter()
    seed_cpu(config.seed, 1)
    whole = tmp_path / "whole"
    whole.mkdir()
    result = train(config, whole)
    assert result["steps"] == config.train.steps
    # A checkpoint at update 10 from the same 30-step schedule is simulated by
    # intercepting the sampler after 10 batches, preserving the ordinary saved checkpoint.
    from unittest.mock import patch
    partial = tmp_path / "partial"
    partial.mkdir()
    config.train.checkpoint_every = 10
    seed_cpu(config.seed, 1)
    original = LagPairs.sample
    calls = 0

    def interrupted(self: LagPairs, batch: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nonlocal calls
        calls += 1
        if calls > 10:
            raise InterruptedError("simulated interrupted process")
        return original(self, batch)

    with patch.object(LagPairs, "sample", interrupted), pytest.raises(InterruptedError):
        train(config, partial)
    config.train.resume = str(partial / "checkpoint.pt")
    resumed = tmp_path / "resumed"
    resumed.mkdir()
    train(config, resumed)
    a = torch.load(whole / "checkpoint.pt", weights_only=False)
    b = torch.load(resumed / "checkpoint.pt", weights_only=False)
    assert a["losses"] == b["losses"]
    for name in a["ema"]:
        torch.testing.assert_close(a["ema"][name], b["ema"][name], atol=0, rtol=0)
    config.evaluation.checkpoint = str(resumed / "checkpoint.pt")
    evaluation = tmp_path / "eval"
    evaluation.mkdir()
    result = evaluate(config, evaluation)
    assert result["finite_samples"]
    assert result["scientific_acceptance"] == "not_applicable_synthetic"
    repeated = tmp_path / "eval_repeated"
    repeated.mkdir()
    torch.randn(17)  # Unrelated work must not change configured evaluation samples.
    evaluate(config, repeated)
    np.testing.assert_array_equal(np.load(evaluation / "samples_nm.npy"), np.load(repeated / "samples_nm.npy"))
    assert time.perf_counter() - started < 120


def test_gpu_is_rejected() -> None:
    """No physical experiment may accidentally use MPS/CUDA."""
    with pytest.raises(ValueError, match="CPU"):
        seed_cpu(1, 2, "mps")


def test_basin_screen_boundaries() -> None:
    """A zero-angle boundary belongs to only one configured half-open radian region."""
    from tito_repro.eval.phase1_audit import basin_counts
    counts = basin_counts(np.array([[-1., -1.], [-1., 0.], [0., 0.]]),
                          {"a": [-np.pi, 0, -np.pi, 0], "b": [-np.pi, 0, 0, np.pi],
                           "c": [0, np.pi, -np.pi, np.pi]})
    assert counts == {"a": 1, "b": 1, "c": 1}


def test_phase1_gate_requires_all_metrics() -> None:
    """Missing or inconclusive scientific metrics cannot unlock Phase 2."""
    failed = phase1_gate_report({"lags": [], "ck": {}, "acceptance_missing": []})
    assert failed["status"] == "failed_or_incomplete"
    assert failed["phase2_allowed"] is False
    passing = phase1_gate_report({"lags": [{"jsd_nats": .01, "timescale_ratio": 1.,
        "reference_msm": {"status": "estimated"}, "model_msm": {"status": "estimated"}}],
        "ck": {"jsd_nats": .01}, "acceptance_missing": []})
    assert passing["passed"] is True
