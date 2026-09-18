"""Verify the local installation without downloading data or using remote services.

Run with .venv/bin/python scripts/verify_local.py from the repository root.
The synthetic experiment checks software only; molecular gates may remain failed.
"""
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    commands = [
        ("Software tests", ["-m", "pytest", "-q"]),
        ("Tiny synthetic CPU experiment", ["-m", "tito_repro.cli", "experiment=smoke"]),
        ("Saved molecular acceptance report", ["-m", "tito_repro.cli", "experiment=phase1_gates"]),
    ]
    for label, arguments in commands:
        print(f"\n{label}", flush=True)
        subprocess.run([sys.executable, *arguments], cwd=root, check=True)
    print("\nLocal software verification completed. See runs/ for artifacts. "
          "Failed molecular gates block scientific acceptance, not exploratory development.")


if __name__ == "__main__":
    main()
