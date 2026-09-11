"""Local experiment records and deterministic CPU execution."""
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def seed_cpu(seed: int, threads: int, device: str = "cpu") -> None:
    """Seed scalar RNGs and configure CPU threads; no physical quantities."""
    if device != "cpu":
        raise ValueError("This project permits CPU execution only (no CUDA or MPS).")
    if threads < 1:
        raise ValueError("threads must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    os.environ["OPENMM_CPU_THREADS"] = str(threads)


def digest(path: str | Path) -> str:
    """Return streaming SHA-256 of a local file; no tensor or units."""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: str | Path, values: dict[str, Any]) -> None:
    """Atomically write strict JSON (NaN/Infinity forbidden); quantities carry named units."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(values, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def record_environment(cfg: DictConfig, output: Path) -> None:
    """Save resolved config, source hashes and package versions; no physical tensors."""
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output / "config.yaml", resolve=True)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    root = Path(__file__).resolve().parents[3]
    sources = {str(p.relative_to(root)): digest(p)
               for directory in ("src", "configs") for p in sorted((root / directory).rglob("*"))
               if p.is_file() and p.suffix in (".py", ".yaml")}
    write_json(output / "environment.json", {
        "device": "cpu", "python": platform.python_version(), "machine": platform.machine(),
        "system": platform.platform(), "torch": torch.__version__, "numpy": np.__version__,
        "threads": cfg.runtime.threads, "seed": cfg.seed,
        "revision": revision.stdout.strip(), "dirty": bool(dirty.stdout.strip()),
        "source_sha256": sources,
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "openmm", "mdtraj", "deeptime", "numpy", "scipy", "matplotlib", "hydra-core", "pytest")},
    })
