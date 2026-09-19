"""Aggregate independent synthetic flow seeds without claiming confidence bounds."""
import numpy as np
from pathlib import Path


def verify_shared_inputs(first: Path, second: Path) -> None:
    """Require identical conditions, targets and priors; predictions may differ."""
    names = {path.name for path in first.glob("*.npz")}
    if not names or names != {path.name for path in second.glob("*.npz")}:
        raise ValueError("Saved evaluation grids differ or are empty")
    for name in sorted(names):
        with np.load(first / name, allow_pickle=False) as a, np.load(second / name, allow_pickle=False) as b:
            for key in ("condition", "reference", "prior"):
                if not np.array_equal(a[key], b[key]):
                    raise ValueError(f"Evaluation inputs differ: {name}, {key}")


def summarize_seeds(runs: list[dict]) -> dict:
    """Report seed spread at each size/lag; reject incomplete or incompatible runs."""
    if len(runs) < 2 or len({run["seed"] for run in runs}) != len(runs):
        raise ValueError("At least two distinct seeds required")
    first = runs[0]
    indexed = []
    for run in runs:
        if run["status"] != "completed" or run["scientific_acceptance"] != "not_applicable_synthetic":
            raise ValueError("Completed synthetic runs required")
        for key in ("steps", "train_atoms", "data", "coupling", "solver", "field_evaluations_per_sample"):
            if run[key] != first[key]:
                raise ValueError(f"Calibration runs differ in {key}")
        rows = {(row["atoms"], row["lag_frames"]): row for row in run["after"]}
        if len(rows) != len(run["after"]) or not rows:
            raise ValueError("Unique nonempty size/lag rows required")
        if indexed and rows.keys() != indexed[0].keys():
            raise ValueError("Calibration grids differ")
        indexed.append(rows)
    output = []
    for key in indexed[0]:
        rows = [index[key] for index in indexed]
        expected = rows[0]["expected_conditional_variance"]
        held_out = rows[0]["held_out_size"]
        if expected <= 0 or any(row["expected_conditional_variance"] != expected
                                or row["held_out_size"] != held_out for row in rows):
            raise ValueError("Calibration targets differ or have nonpositive variance")
        ratios = np.array([row["conditional_residual_second_moment"] / expected for row in rows])
        velocity = np.array([row["velocity_mse"] / row["zero_velocity_mse"] for row in rows])
        if not np.isfinite(ratios).all() or not np.isfinite(velocity).all():
            raise ValueError("Finite calibration ratios required")
        output.append({"atoms": key[0], "lag_frames": key[1], "held_out_size": held_out,
                       "second_moment_ratio_mean": float(ratios.mean()),
                       "second_moment_ratio_seed_std": float(ratios.std(ddof=1)),
                       "second_moment_ratio_range": [float(ratios.min()), float(ratios.max())],
                       "velocity_mse_to_zero_baseline_mean": float(velocity.mean()),
                       "all_samples_finite": all(row["finite_samples"] for row in rows)})
    return {"status": "completed", "seeds": [run["seed"] for run in runs], "rows": output,
            "scientific_acceptance": "not_applicable_synthetic",
            "uncertainty": "sample standard deviation and range across seeds; not confidence intervals",
            "limitations": ["small seed count", "synthetic point clouds, not molecules",
                            "second moments do not test the complete conditional distribution"]}
