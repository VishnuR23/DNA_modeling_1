"""Paired local residual/no-residual pilots; uses cached alanine data only."""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "runs" / "epsilon_ablation" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    results = {}
    for variant, enabled in (("plain", "false"), ("residual", "true")):
        training = output / variant / "train"
        diagnosis = output / variant / "diagnostic"
        for arguments in (
            ["experiment=alanine_epsilon_ablation", f"model.epsilon_skip={enabled}",
             f"hydra.run.dir={training}"],
            ["experiment=alanine_denoising", f"diagnostic.checkpoint={training / 'checkpoint.pt'}",
             f"hydra.run.dir={diagnosis}"],
        ):
            subprocess.run([sys.executable, "-m", "tito_repro.cli", *arguments], cwd=root, check=True)
        results[variant] = {"training": json.loads((training / "metrics.json").read_text()),
                            "diagnostic": json.loads((diagnosis / "metrics.json").read_text())}
    results["matched_completed_steps"] = (
        results["plain"]["training"]["status"] == results["residual"]["training"]["status"] == "completed"
        and results["plain"]["training"]["steps"] == results["residual"]["training"]["steps"])
    results["scientific_acceptance"] = "not_evaluated"
    (output / "comparison.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(f"Comparison saved to {output / 'comparison.json'}")


if __name__ == "__main__":
    main()
