"""Compare saved flow models on identical evaluation inputs without retraining."""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

from tito_repro.eval.flow_calibration import summarize_seeds, verify_shared_inputs
from tito_repro.utils.runtime import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--evaluation-seed", type=int, default=20260910)
    parser.add_argument("--samples", type=int, default=512)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    campaign = args.campaign if args.campaign.is_absolute() else root / args.campaign
    source = json.loads((campaign / "campaign.json").read_text())
    if source["status"] != "completed" or len(source["seeds"]) < 2 or args.samples < 1:
        parser.error("a completed multi-seed campaign and positive sample count are required")
    seeds = source["seeds"]
    configs = [OmegaConf.to_container(OmegaConf.load(campaign / str(seed) / "config.yaml").flow)
               for seed in seeds]
    if any(config != configs[0] for config in configs[1:]):
        parser.error("source flow configurations differ")
    output = root / "runs" / "flow_shared_inputs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    manifest = {"status": "running", "source_campaign": str(campaign),
                "training_seeds": seeds, "evaluation_seed": args.evaluation_seed,
                "samples_per_size_lag": args.samples, "completed_runs": []}
    write_json(output / "campaign.json", manifest)
    runs = []
    try:
        for seed in seeds:
            run = output / str(seed)
            subprocess.run([sys.executable, "-m", "tito_repro.cli", "experiment=phase2_evaluate",
                            f"flow_evaluation.checkpoint={campaign / str(seed) / 'checkpoint.pt'}",
                            f"seed={args.evaluation_seed}", f"flow_evaluation.samples={args.samples}",
                            f"hydra.run.dir={run}"], cwd=root, check=True, timeout=180)
            evaluation = json.loads((run / "metrics.json").read_text())
            if evaluation["training_seed"] != seed:
                raise ValueError("Checkpoint training seed differs from campaign")
            if runs:
                verify_shared_inputs(output / str(seeds[0]) / "samples", run / "samples")
            original = json.loads((campaign / str(seed) / "metrics.json").read_text())
            original["after"] = evaluation["rows"]
            original["field_evaluations_per_sample"] = evaluation["field_evaluations_per_sample"]
            runs.append(original)
            manifest["completed_runs"].append(str(run.relative_to(root)))
            write_json(output / "campaign.json", manifest)
        summary = summarize_seeds(runs)
        summary.update(evaluation_seed=args.evaluation_seed, samples_per_size_lag=args.samples,
                       shared_inputs_verified=True, source_campaign=str(campaign),
                       interpretation="model variation conditional on one fixed finite evaluation sample")
        summary["limitations"].append("one shared evaluation seed; finite-sample uncertainty not estimated")
        write_json(output / "summary.json", summary)
        manifest["status"] = "completed"
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        write_json(output / "campaign.json", manifest)
    print(f"Shared-input calibration saved to {output / 'summary.json'}")


if __name__ == "__main__":
    main()
