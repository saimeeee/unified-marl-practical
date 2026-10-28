"""Regression contracts only. No environment, rollout or optimizer is created."""
from dataclasses import asdict
import copy
import json
from types import SimpleNamespace

import pytest

from unified_marl import api
from unified_marl.applications import warehouse, export_template, validate
from unified_marl.applications.cli import main as cli
from unified_marl.config import load_config


class ValidationReachedDeviceCheck(Exception):
    """Stop after resume validation, before any environment can be created."""


def stored_config():
    cfg = load_config("pymarl/qmix", "toy", overrides={"max_updates": 1})
    cfg["env_args"]["seed"] = cfg["seed"]
    return cfg


@pytest.mark.parametrize("provided", [{}, {"episode_limit": 4}, {"seed": 1},
                                        {"n_agents": 2, "episode_limit": 4, "truncate": False}])
def test_resume_accepts_omitted_stored_defaults_and_seed(monkeypatch, provided):
    cfg = stored_config()
    original = copy.deepcopy(cfg)
    monkeypatch.setattr(api, "read_checkpoint", lambda *a, **k: {"kind": "training", "config": cfg})

    def stop(_):
        raise ValidationReachedDeviceCheck

    monkeypatch.setattr(api, "check_device", stop)
    with pytest.raises(ValidationReachedDeviceCheck):
        api.train(resume="training.pt", env_args=provided, overrides={"max_updates": 2})
    assert cfg == original


@pytest.mark.parametrize("provided", [{"seed": 2}, {"n_agents": 3}, {"episode_limit": 7},
                                        {"new_unstored_option": 1}])
def test_resume_still_rejects_explicit_environment_changes(monkeypatch, provided):
    cfg = stored_config()
    monkeypatch.setattr(api, "read_checkpoint", lambda *a, **k: {"kind": "training", "config": cfg})
    with pytest.raises(ValueError, match="Cannot change environment arguments"):
        api.train(resume="training.pt", env_args=provided)


def test_resume_still_rejects_global_seed_change_and_policy_checkpoint(monkeypatch):
    cfg = stored_config()
    monkeypatch.setattr(api, "read_checkpoint", lambda *a, **k: {"kind": "training", "config": cfg})
    with pytest.raises(ValueError, match="seed cannot change"):
        api.train(resume="training.pt", seed=2)
    monkeypatch.setattr(api, "read_checkpoint", lambda *a, **k: {"kind": "policy", "config": cfg})
    with pytest.raises(ValueError, match="policy-only"):
        api.train(resume="policy.pt")


def test_warehouse_api_resume_reaches_core_without_seed_mismatch(monkeypatch, tmp_path):
    export_template("warehouse", tmp_path / "scene")
    path = tmp_path / "scene/warehouse.yaml"
    report = validate(path)
    first = tmp_path / "first"
    first.mkdir()
    (first / "application.json").write_text(json.dumps({"config": report["config"]}), encoding="utf-8")
    cfg = load_config(report["algorithm"], "inventory", config=report["config"]["training"])
    cfg["env_args"] = dict(report["config"]["env_args"], seed=cfg["seed"])
    monkeypatch.setattr(api, "read_checkpoint", lambda *a, **k: {"kind": "training", "config": cfg})

    def stop(_):
        raise ValidationReachedDeviceCheck

    monkeypatch.setattr(api, "check_device", stop)
    with pytest.raises(ValidationReachedDeviceCheck):
        warehouse.train(path, output_dir=tmp_path / "resumed", resume=first / "training.pt",
                        overrides={"training.max_updates": 2})


def test_cli_resume_forwards_budget_and_writes_application_results(monkeypatch, tmp_path, capsys):
    import unified_marl
    export_template("warehouse", tmp_path / "scene")
    config = tmp_path / "scene/warehouse.yaml"
    report = validate(config)
    first = tmp_path / "first"
    first.mkdir()
    (first / "application.json").write_text(json.dumps({"config": report["config"]}), encoding="utf-8")
    seen = {}

    def fake_train(*args, **kwargs):
        seen.update(kwargs)
        out = kwargs["output_dir"]
        return api.TrainResult(report["algorithm"], str(out), str(out / "policy.pt"), str(out / "training.pt"),
                               str(out / "metrics.jsonl"), str(out / "config.json"), 8, 2, 2, 0.01, {"loss": 0.1})

    monkeypatch.setattr(unified_marl, "train", fake_train)
    out = tmp_path / "resumed"
    assert cli(["train", "--config", str(config), "--output-dir", str(out),
                "--resume", str(first / "training.pt"), "--set", "training.max_updates=2"]) == 0
    assert seen["config"] is None and seen["overrides"]["max_updates"] == 2
    assert "seed" not in seen["env_args"]
    assert json.loads((out / "application-result.json").read_text())["updates"] == 2
    assert (out / "application.json").is_file()
    assert json.loads(capsys.readouterr().out)["env_steps"] == 8


def test_evaluation_counts_only_recorded_environment_transitions(monkeypatch):
    cfg = stored_config()
    payload = {"config": cfg, "env_info": {"test": "metadata"}}
    monkeypatch.setattr(api, "read_checkpoint", lambda *a, **k: payload)
    monkeypatch.setattr(api, "Policy", lambda *a: SimpleNamespace(scheme={}, groups={}, preprocess={}, mac=None))
    closed = []

    class RecordedRunner:
        def __init__(self, *args):
            self.index = 0

        def get_env_info(self):
            return payload["env_info"]

        def setup(self, *args):
            pass

        def run(self, test_mode):
            assert test_mode
            self.last_returns = [[1.0], [3.0]][self.index]
            self.last_lengths = [[2], [4]][self.index]
            self.index += 1

        def close_env(self):
            closed.append(True)

    monkeypatch.setitem(api.RUNNERS, "episode", RecordedRunner)
    result = api.evaluate("policy.pt", episodes=2)
    assert result.episodes == 2 and result.returns == [1.0, 3.0]
    assert result.episode_lengths == [2, 4] and result.env_steps == 6
    assert asdict(result)["env_steps"] == 6
    assert closed == [True]
