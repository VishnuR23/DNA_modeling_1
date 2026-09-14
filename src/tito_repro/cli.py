"""Configuration-only entry point for local Phase 1 experiments."""
import logging
import os
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from tito_repro.utils.runtime import record_environment, seed_cpu, write_json


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Execute a CPU experiment; configuration declares all tensor sizes and physical units."""
    seed_cpu(cfg.seed, cfg.runtime.threads, cfg.runtime.device)
    output = Path(HydraConfig.get().runtime.output_dir)
    record_environment(cfg, output)
    cache = Path(__file__).resolve().parents[2] / ".cache"
    cache.mkdir(exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache / "matplotlib")
    os.environ["XDG_CACHE_HOME"] = str(cache)
    try:
        from tito_repro.train.phase1 import evaluate, train
        if cfg.action == "train":
            result = train(cfg, output)
        elif cfg.action == "evaluate":
            result = evaluate(cfg, output)
        elif cfg.action in ("smoke", "pilot"):
            result = train(cfg, output)
            cfg.evaluation.checkpoint = str(output / "checkpoint.pt")
            child = output / "evaluation"
            child.mkdir()
            record_environment(cfg, child)
            result["evaluation"] = evaluate(cfg, child)
            write_json(output / "metrics.json", result)
        elif cfg.action == "reference":
            from tito_repro.reference.alanine import simulate
            result = simulate(cfg, output)
        elif cfg.action == "download":
            from tito_repro.data.download import download_alanine
            result = download_alanine(cfg, output)
        elif cfg.action == "audit":
            from tito_repro.eval.phase1_audit import audit
            result = audit(cfg, output)
        elif cfg.action == "gates":
            from tito_repro.eval.gates import write_phase1_gate_report
            result = write_phase1_gate_report(cfg.gates.metrics, output / "phase1_gates.json")
        else:
            raise ValueError(f"Unknown action: {cfg.action}")
        logging.info("Result: %s", result)
    except Exception as error:
        write_json(output / "metrics.json", {"status": "failed", "error": str(error),
                                            "scientific_acceptance": "not_passed", "device": "cpu"})
        logging.exception("Experiment failed")
        raise


if __name__ == "__main__":
    main()
