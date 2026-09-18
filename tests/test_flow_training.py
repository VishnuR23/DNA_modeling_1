"""Tiny real training and checkpoint replay for the exploratory flow pipeline."""
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

from tito_repro.models.flow import ConditionalFlow
from tito_repro.train.flow import diagnose, train_flow
from tito_repro.utils.runtime import seed_cpu


def test_flow_checkpoint_replays_diagnostics(tmp_path):
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(version_base=None, config_dir=str(root / "configs")):
        cfg = compose(config_name="config", overrides=["experiment=phase2_synthetic",
            "flow.steps=4", "flow.batch_size=2", "flow.evaluation_samples=4", "flow.solver_steps=2"])
    seed_cpu(cfg.seed, 1)
    result = train_flow(cfg, tmp_path)
    state = torch.load(tmp_path / "checkpoint.pt", map_location="cpu", weights_only=False)
    model = ConditionalFlow(cfg.flow.width, max(cfg.flow.lags))
    model.load_state_dict(state["model"])
    assert result["steps"] == state["steps"] == 4
    assert result["scientific_acceptance"] == "not_applicable_synthetic"
    assert diagnose(model, cfg) == result["after"]
    assert any(row["held_out_size"] for row in result["after"])
    assert all(row["finite_samples"] for row in result["after"])
