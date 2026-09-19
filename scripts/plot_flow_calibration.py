"""Plot committed seed ranges; bars are not confidence intervals."""
import json
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=ROOT / "docs/results/flow_calibration_3seeds.json")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/figures/phase2_calibration.png")
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    shared = summary.get("shared_inputs_verified", False)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, lag in zip(axes, (1, 4)):
        rows = [row for row in summary["rows"] if row["lag_frames"] == lag]
        means = np.array([row["second_moment_ratio_mean"] for row in rows])
        intervals = np.array([row["second_moment_ratio_range"] for row in rows])
        ax.axhline(1, color="gray", linestyle="--", linewidth=1, label="Analytic target")
        ax.axvspan(2.7, 4.3, color="lightgray", alpha=.3, label="Training sizes")
        ax.errorbar([row["atoms"] for row in rows], means,
                    yerr=np.stack((means - intervals[:, 0], intervals[:, 1] - means)),
                    fmt="o-", capsize=5, color="#2463A5",
                    label=f"Mean and range of {len(summary['seeds'])} models" if shared
                    else f"Mean and range of {len(summary['seeds'])} seeds")
        ax.set(xlabel="Atom count", title=f"Synthetic lag = {lag} frames", xticks=[3, 4, 5, 8],
               xlim=(2.7, 8.3), ylim=(.55, 1.45))
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Residual second moment / analytic variance")
    axes[1].legend(fontsize=8, loc="lower left")
    fig.suptitle("Model variation persists with identical evaluation inputs" if shared
                 else "Held-out sizes show substantial seed variation", fontsize=12)
    fig.tight_layout()
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
