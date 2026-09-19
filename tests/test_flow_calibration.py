"""Seed summaries must expose variation and reject misleading pooling."""
from copy import deepcopy

import pytest
import numpy as np

from tito_repro.eval.flow_calibration import summarize_seeds, verify_shared_inputs


def result(seed, moment):
    return {"seed": seed, "status": "completed", "scientific_acceptance": "not_applicable_synthetic",
            "steps": 10, "train_atoms": [3, 4], "data": "generated_gaussian_ar1",
            "coupling": "independent_gaussian_linear_path", "solver": "heun",
            "field_evaluations_per_sample": 40,
            "after": [{"atoms": 5, "lag_frames": 1, "held_out_size": True,
                       "expected_conditional_variance": .5,
                       "conditional_residual_second_moment": moment,
                       "velocity_mse": 2., "zero_velocity_mse": 4., "finite_samples": True}]}


def test_summary_exposes_seed_spread():
    summary = summarize_seeds([result(1, .25), result(2, .75)])
    row = summary["rows"][0]
    assert row["second_moment_ratio_mean"] == 1
    assert row["second_moment_ratio_range"] == [.5, 1.5]
    assert row["second_moment_ratio_seed_std"] == pytest.approx(2 ** -.5)
    assert row["velocity_mse_to_zero_baseline_mean"] == .5


@pytest.mark.parametrize("change", ["seed", "steps", "grid", "status"])
def test_incompatible_runs_are_not_pooled(change):
    first = result(1, .5)
    second = deepcopy(result(2, .5))
    if change == "seed":
        second["seed"] = 1
    elif change == "steps":
        second["steps"] = 5
    elif change == "grid":
        second["after"][0]["atoms"] = 8
    else:
        second["status"] = "time_limit"
    with pytest.raises(ValueError):
        summarize_seeds([first, second])


def test_shared_inputs_allow_different_predictions_but_reject_different_noise(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    inputs = {key: np.ones((2, 3, 3)) for key in ("condition", "reference", "prior")}
    np.savez(first / "sample.npz", **inputs, generated=np.zeros((2, 3, 3)))
    np.savez(second / "sample.npz", **inputs, generated=np.ones((2, 3, 3)))
    verify_shared_inputs(first, second)
    inputs["prior"] = np.zeros((2, 3, 3))
    np.savez(second / "sample.npz", **inputs)
    with pytest.raises(ValueError, match="prior"):
        verify_shared_inputs(first, second)
