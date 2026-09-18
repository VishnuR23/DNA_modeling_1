"""A fresh checkout can inspect scientific gates without expensive reruns."""
import json
from pathlib import Path

from hydra import compose, initialize_config_dir

from tito_repro.eval.gates import phase1_gate_report, write_phase1_gate_report


def test_default_gates_use_committed_results(tmp_path):
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(version_base=None, config_dir=str(root / "configs")):
        cfg = compose(config_name="config", overrides=["experiment=phase1_gates"])
    source = root / cfg.gates.metrics
    saved = json.loads(source.read_text())
    destination = tmp_path / "gates.json"
    report = write_phase1_gate_report(source, destination)
    assert report == phase1_gate_report(saved["evaluation"])
    assert json.loads(destination.read_text()) == report
    assert report["phase2_allowed"] is False
    assert report["phase2_exploration_allowed"] is True
    assert report["phase2_validated_claims_allowed"] is False
