"""Bounded local multi-seed synthetic calibration, without downloads."""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from tito_repro.eval.flow_calibration import summarize_seeds
from tito_repro.utils.runtime import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260910, 20260911, 20260912])
    args = parser.parse_args()
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        parser.error("provide at least two distinct seeds")
    root = Path(__file__).resolve().parents[1]
    output = root / "runs" / "flow_calibration" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    manifest = {"status": "running", "seeds": args.seeds, "completed_runs": []}
    write_json(output / "campaign.json", manifest)
    runs = []
    try:
        for seed in args.seeds:
            run = output / str(seed)
            subprocess.run([sys.executable, "-m", "tito_repro.cli", "experiment=phase2_synthetic",
                            f"seed={seed}", "flow.evaluation_atoms=[3,4,5,8]", f"hydra.run.dir={run}"],
                           cwd=root, check=True, timeout=180)
            runs.append(json.loads((run / "metrics.json").read_text()))
            manifest["completed_runs"].append(str(run.relative_to(root)))
            write_json(output / "campaign.json", manifest)
        write_json(output / "summary.json", summarize_seeds(runs))
        manifest["status"] = "completed"
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        write_json(output / "campaign.json", manifest)
    print(f"Calibration saved to {output / 'summary.json'}")


if __name__ == "__main__":
    main()
