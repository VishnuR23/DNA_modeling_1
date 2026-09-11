"""Public alanine data ingestion without a third-party downloader."""
import subprocess
from pathlib import Path
from typing import Any

import mdtraj as md
import numpy as np
from omegaconf import DictConfig

from tito_repro.utils.runtime import digest, write_json


def download_alanine(cfg: DictConfig, output: Path) -> dict[str, Any]:
    """Fetch configured MDShare files and save float32 [R,T,A,3] nm with spacing ps."""
    root = Path(cfg.download.directory)
    root.mkdir(parents=True, exist_ok=True)
    sources = []
    for filename in cfg.download.files:
        destination = root / filename
        url = cfg.download.base_url.rstrip("/") + "/" + filename
        if not destination.exists():
            temporary = destination.with_suffix(destination.suffix + ".part")
            subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                            "--max-time", str(cfg.download.timeout_seconds), url, "-o", str(temporary)], check=True)
            temporary.replace(destination)
        checksum = digest(destination)
        if checksum != cfg.download.sha256[filename]:
            raise ValueError(f"Downloaded/cached checksum differs for {filename}")
        sources.append({"url": url, "sha256": checksum, "bytes": destination.stat().st_size})
    topology = root / cfg.download.topology
    trajectories = [md.load(str(root / name), top=str(topology)) for name in cfg.download.trajectories]
    # MDShare ALA2 documents 250 ns continuous replicas at 1 ps spacing:
    # https://markovmodel.github.io/mdshare/ALA2/ . XTC times reset each 1000 frames.
    # Accept only the configured, exactly verified periodic timestamp pattern.
    for trajectory in trajectories:
        expected = (np.arange(len(trajectory)) % cfg.download.timestamp_period_frames) * cfg.data.spacing_ps
        if not np.allclose(trajectory.time, expected):
            raise ValueError("Trajectory timestamps do not match configured periodic pattern")
    positions = np.stack([t.xyz.astype(np.float32) for t in trajectories])
    np.savez(cfg.data.path, positions_nm=positions, spacing_ps=cfg.data.spacing_ps)
    result = {"status": "completed", "sources": sources, "positions_shape": list(positions.shape),
              "processed_sha256": digest(cfg.data.path), "spacing_ps": cfg.data.spacing_ps,
              "timestamp_policy": "continuous frame index using documented MDShare spacing",
              "verified_timestamp_period_frames": cfg.download.timestamp_period_frames,
              "scientific_acceptance": "not_evaluated", "device": "cpu"}
    write_json(output / "metrics.json", result)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.bar(range(len(trajectories)), [len(t) * cfg.data.spacing_ps for t in trajectories])
    ax.set(xlabel="Replica", ylabel="Stored trajectory duration (ps)")
    fig.savefig(output / "data_lengths.png")
    plt.close(fig)
    return result
