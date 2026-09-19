"""Paired uncertainty must preserve shared chains and difference direction."""
import numpy as np
import pytest

from tito_repro.eval.paired_sampling import compare_histograms


def test_identical_chains_have_zero_paired_difference():
    rng = np.random.default_rng(1)
    reference = rng.uniform(-np.pi, np.pi, (200, 2))
    samples = rng.uniform(-np.pi, np.pi, (4, 8, 2))
    result = compare_histograms(reference, samples, samples, 8, 20, 2)
    assert result["residual_minus_plain_jsd_nats"] == 0
    assert result["paired_chain_bootstrap_interval_nats"] == [0., 0.]
    assert result == compare_histograms(reference, samples, samples, 8, 20, 2)


def test_better_residual_has_negative_difference():
    reference = np.zeros((100, 2))
    plain = np.ones((4, 8, 2)) * 2
    residual = np.zeros_like(plain)
    result = compare_histograms(reference, plain, residual, 8, 20, 2)
    assert result["residual_minus_plain_jsd_nats"] == pytest.approx(-np.log(2))
    assert result["matched_count_reference_jsd_mean_nats"] == 0
    with pytest.raises(ValueError, match="Matching"):
        compare_histograms(reference, plain, residual[:2], 8, 20, 2)
