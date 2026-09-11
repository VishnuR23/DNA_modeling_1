"""Histogram and kinetic diagnostics with explicit failure states."""
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Return Jensen–Shannon divergence in nats of equal-shaped nonnegative counts."""
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    if p.shape != q.shape or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("Finite equal-shaped histogram arrays required")
    if min(p.sum(), q.sum()) <= 0 or (p < 0).any() or (q < 0).any():
        raise ValueError("Nonnegative histograms with positive mass required")
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2
    return float(.5 * sum((x[x > 0] * np.log(x[x > 0] / m[x > 0])).sum() for x in (p, q)))


def torsions(positions: np.ndarray, topology: str,
             indices: list[list[int]]) -> np.ndarray:
    """Compute configured dihedrals [frames,torsions] in radians from [frames,atoms,3] nm."""
    import mdtraj as md
    top = md.load_topology(topology)
    if not np.isfinite(positions).all():
        raise ValueError("Invalid coordinates must be reported, not silently discarded")
    return md.compute_dihedrals(md.Trajectory(positions, top), np.asarray(indices))


def histogram(angles: np.ndarray, bins: int) -> np.ndarray:
    """Return periodic phi/psi counts [bins,bins] for finite [frames,2] radians."""
    if angles.ndim != 2 or angles.shape[1] != 2 or not np.isfinite(angles).all():
        raise ValueError("Expected finite [frames,2] torsions")
    wrapped = (angles + np.pi) % (2 * np.pi) - np.pi
    return np.histogram2d(*wrapped.T, bins=bins, range=[[-np.pi, np.pi]] * 2)[0]


def msm_timescale(trajectories: list[np.ndarray], lag_frames: int, spacing_ps: float,
                  bins: int, min_transitions: int) -> dict[str, Any]:
    """Estimate reversible count MSM slowest time in ps; input angle trajectories [T,2] rad.

    A fixed periodic grid is shared by reference and generated data. Disconnected
    support is reported as inconclusive rather than silently selecting one basin.
    """
    count = np.zeros((bins * bins, bins * bins), dtype=np.float64)
    for angles in trajectories:
        if len(angles) <= lag_frames:
            continue
        cells = np.floor(((angles + np.pi) % (2 * np.pi)) / (2 * np.pi) * bins).astype(int)
        states = cells[:, 0] * bins + cells[:, 1]
        np.add.at(count, (states[:-lag_frames], states[lag_frames:]), 1)
    transitions = int(count.sum())
    symmetric = count + count.T
    active = symmetric.sum(1) > 0
    matrix = symmetric[np.ix_(active, active)]
    result: dict[str, Any] = {"lag_ps": lag_frames * spacing_ps, "transitions": transitions,
                              "active_states": int(active.sum()), "timescale_ps": None}
    if transitions < min_transitions or len(matrix) < 2:
        return {**result, "status": "inconclusive", "reason": "insufficient transitions or states"}
    components = connected_components(csr_matrix(matrix), directed=False)[0]
    if components != 1:
        return {**result, "status": "inconclusive", "reason": "disconnected state graph",
                "components": int(components)}
    mass = matrix.sum(1)
    reversible = matrix / np.sqrt(mass[:, None] * mass[None, :])
    eigenvalues = np.linalg.eigvalsh(reversible)[::-1]
    second = float(eigenvalues[1])
    if not 0 < second < 1:
        return {**result, "status": "inconclusive", "reason": "nonpositive or nondecaying mode"}
    return {**result, "status": "estimated", "timescale_ps": float(-lag_frames * spacing_ps / np.log(second)),
            "estimator": "symmetrized_counts", "second_eigenvalue": second}


def vamp_score(trajectories: list[np.ndarray], lag_frames: int) -> float | None:
    """Return VAMP-2 score from sin/cos [frames,2] angle features; dimensionless."""
    from deeptime.decomposition import VAMP
    features = [np.concatenate((np.cos(x), np.sin(x)), axis=1) for x in trajectories if len(x) > lag_frames]
    if not features:
        return None
    estimator = VAMP(lagtime=lag_frames)
    for features_i in features:
        estimator.partial_fit((features_i[:-lag_frames], features_i[lag_frames:]))
    score = float(estimator.fetch_model().score(2))
    return score if np.isfinite(score) else None
