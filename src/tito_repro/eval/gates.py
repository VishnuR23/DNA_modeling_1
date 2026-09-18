"""Machine-checkable Phase 1 acceptance gates."""
from pathlib import Path
from typing import Any
import json

from tito_repro.utils.runtime import write_json


def phase1_gate_report(metrics: dict[str, Any], jsd_limit: float = 0.10,
                       ck_limit: float = 0.05, ratio_low: float = 0.5,
                       ratio_high: float = 2.0) -> dict[str, Any]:
    """Evaluate saved Phase 1 metrics; inconclusive estimates never pass."""
    lags = metrics.get("lags", [])
    thermo = bool(lags) and all(isinstance(row.get("jsd_nats"), (int, float))
                               and row["jsd_nats"] <= jsd_limit for row in lags)
    kinetic = bool(lags) and all(
        isinstance(row.get("timescale_ratio"), (int, float))
        and ratio_low <= row["timescale_ratio"] <= ratio_high
        and row.get("reference_msm", {}).get("status") == "estimated"
        and row.get("model_msm", {}).get("status") == "estimated" for row in lags)
    ck = metrics.get("ck", {})
    ck_pass = isinstance(ck.get("jsd_nats"), (int, float)) and ck["jsd_nats"] <= ck_limit
    missing = list(metrics.get("acceptance_missing", []))
    passed = thermo and kinetic and ck_pass and not missing
    return {"status": "passed" if passed else "failed_or_incomplete", "passed": passed,
            "thermodynamics": {"passed": thermo, "limit_jsd_nats": jsd_limit},
            "kinetics": {"passed": kinetic, "ratio_range": [ratio_low, ratio_high]},
            "chapman_kolmogorov": {"passed": ck_pass, "limit_jsd_nats": ck_limit},
            "missing_requirements": missing, "phase2_allowed": passed,
            "phase2_exploration_allowed": True,
            "phase2_validated_claims_allowed": passed,
            "development_policy": "exploration_without_scientific_acceptance"}


def write_phase1_gate_report(metrics_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Read standalone or pipeline metrics and write the Phase 1 gate report."""
    metrics = json.loads(Path(metrics_path).read_text())
    if "evaluation" in metrics:
        metrics = metrics["evaluation"]
    report = phase1_gate_report(metrics)
    write_json(output_path, report)
    return report
