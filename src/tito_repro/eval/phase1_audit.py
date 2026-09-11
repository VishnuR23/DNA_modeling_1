# Phase 1 audit plan: reuse saved molecular rollouts without additional sampling;
# quantify finite-histogram effects and whole-chain bootstrap uncertainty; expose
# basin occupancy and explicit failed/incomplete gates before any transferable work.
"""Reproducible CPU audit of saved Phase 1 molecular diagnostic artifacts."""
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf

from tito_repro.data.pairs import load_trajectories
from tito_repro.eval.metrics import histogram, jsd, torsions
from tito_repro.utils.runtime import digest, write_json


def basin_counts(angles: np.ndarray, regions: dict[str, list[float]]) -> dict[str, int]:
    """Count [N,2] phi/psi radians in configured half-open rectangular regions."""
    return {name: int(((angles[:, 0] >= lo) & (angles[:, 0] < hi)
                      & (angles[:, 1] >= bottom) & (angles[:, 1] < top)).sum())
            for name, (lo, hi, bottom, top) in regions.items()}


def audit(cfg: DictConfig, output: Path) -> dict[str, Any]:
    """Audit saved trajectories in nm; output JSD nats, occupancies, timescale ratios."""
    source = Path(cfg.audit.evaluation_run)
    previous = json.loads((source / "metrics.json").read_text())
    run_cfg = OmegaConf.load(source / "config.yaml")
    data, _ = load_trajectories(run_cfg.data.path)
    ref_angles = np.concatenate([torsions(x, run_cfg.data.topology, run_cfg.data.torsions) for x in data])
    bins = run_cfg.evaluation.bins
    ref_hist = histogram(ref_angles, bins)
    regions = OmegaConf.to_container(cfg.audit.basins)
    ref_counts = basin_counts(ref_angles, regions)
    rng = np.random.default_rng(cfg.seed)
    tail = (1 - cfg.audit.confidence) / 2
    result: dict[str, Any] = {"device": "cpu", "seed": cfg.seed, "status": "diagnostic",
        "source_run": str(source), "source_metrics_sha256": digest(source / "metrics.json"),
        "scientific_acceptance": "not_passed", "reference_basin_counts": ref_counts,
        "bootstrap_method": "whole generated chains; reference histogram held fixed",
        "control_method": "IID reference draws of matching frame count; not temporal uncertainty",
        "basin_definition": "broad project regions, not fitted paper boundaries", "lags": []}
    for lag, metric in zip(run_cfg.evaluation.lags, previous["lags"]):
        samples = np.load(source / f"rollout_lag_{lag}_nm.npy")
        chains = [torsions(x[1:], run_cfg.data.topology, run_cfg.data.torsions) for x in samples]
        angles = np.concatenate(chains)
        boot, controls = [], []
        for _ in range(cfg.audit.bootstrap_replicates):
            chosen = rng.integers(len(chains), size=len(chains))
            boot.append(jsd(ref_hist, histogram(np.concatenate([chains[i] for i in chosen]), bins)))
            control = ref_angles[rng.integers(len(ref_angles), size=len(angles))]
            controls.append(jsd(ref_hist, histogram(control, bins)))
        counts = basin_counts(angles, regions)
        ratio = metric["timescale_ratio"]
        lo, hi = cfg.audit.timescale_ratio_bounds
        result["lags"].append({"lag_ps": metric["lag_ps"], "jsd_nats": metric["jsd_nats"],
            "chain_bootstrap_interval_nats": np.quantile(boot, [tail, 1-tail]).tolist(),
            "matched_count_reference_jsd_mean_nats": float(np.mean(controls)),
            "matched_count_reference_jsd_interval_nats": np.quantile(controls, [tail, 1-tail]).tolist(),
            "basin_counts": counts, "generated_frames": len(angles),
            "basin_count_screen_pass": all(n >= cfg.audit.minimum_basin_samples for n in counts.values()),
            "thermo_point_threshold_pass": metric["jsd_nats"] <= cfg.audit.jsd_threshold_nats,
            "timescale_ratio": ratio, "kinetic_point_threshold_pass": None if ratio is None else lo <= ratio <= hi})
    ck = previous["ck"]
    result["ck"] = {**ck, "point_threshold_pass": ck["jsd_nats"] <= cfg.audit.ck_threshold_nats,
                    "acceptance": "incomplete: one initial condition and sparse branches"}
    result["acceptance_missing"] = previous["acceptance_missing"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    lags = np.array([m["lag_ps"] for m in result["lags"]])
    axes[0].plot(lags, [m["jsd_nats"] for m in result["lags"]], "o-", label="Model")
    axes[0].plot(lags, [m["matched_count_reference_jsd_mean_nats"] for m in result["lags"]], "o-", label="IID reference, same count")
    axes[0].axhline(cfg.audit.jsd_threshold_nats, color="gray", linestyle="--", label="Project target")
    axes[0].set(xscale="log", xlabel="Physical lag (ps)", ylabel="JSD (nats)", title="Pilot, not reproduction")
    axes[0].legend(fontsize=8)
    names = list(regions)
    width = 0.8 / (len(result["lags"]) + 1)
    ticks = np.arange(len(names))
    axes[1].bar(ticks, [ref_counts[n] / len(ref_angles) for n in names], width, label="Reference")
    for i, metric in enumerate(result["lags"], 1):
        axes[1].bar(ticks + i*width, [metric["basin_counts"][n] / metric["generated_frames"] for n in names], width,
                    label=f'{metric["lag_ps"]:g} ps')
    axes[1].set(xticks=ticks + width*len(result["lags"])/2, xticklabels=names, ylabel="Fraction", title="Broad torsion regions")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "audit.png", dpi=140)
    plt.close(fig)
    write_json(output / "metrics.json", result)
    return result
