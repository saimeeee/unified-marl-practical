import copy
from dataclasses import asdict
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from unified_marl import evaluate, load_policy, train
from unified_marl.api import _make_model, make_scheme
from unified_marl.checkpoints import read_checkpoint
from unified_marl.components.episode_buffer import EpisodeBatch, ReplayBuffer
from unified_marl.config import load_config
from unified_marl.envs import make_env
from unified_marl.learners.base import UpdateContext
from unified_marl.runners import REGISTRY as RUNNERS
from unified_marl.utils.logging import Logger

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/stage2"
EVIDENCE.mkdir(parents=True, exist_ok=True)


@pytest.fixture(autouse=True)
def fast_cpu(monkeypatch):
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def options(algorithm, parallel=False):
    opts = {"batch_size": 2, "buffer_size": 8, "t_max": 24, "test_nepisode": 2, "rnn_hidden_dim": 16,
            "learner_log_interval": 1}
    if algorithm == "coma" or parallel:
        opts.update(runner="parallel", batch_size_run=2)
    return opts


@pytest.mark.parametrize("algorithm", ["qmix", "vdn", "iql", "coma", "qtran"])
def test_training_evaluation_inference(algorithm, tmp_path):
    result = train(algorithm, "toy", output_dir=tmp_path / algorithm, overrides=options(algorithm))
    assert result.updates > 0 and result.parameter_delta_l2 > 0
    assert all(math.isfinite(v) for v in result.metrics.values())
    assert Path(result.checkpoint).is_file() and Path(result.training_checkpoint).is_file()
    evaluation = evaluate(result.checkpoint, episodes=2)
    assert evaluation.episodes == 2 and len(evaluation.returns) == 2
    policy = load_policy(result.checkpoint)
    env = make_env("toy", seed=1)
    try:
        env.reset()
        observation, available = env.get_obs(), np.asarray(env.get_avail_actions())
        first = policy.act(observation, available)
        assert available[np.arange(env.n_agents), first].all()
        policy.act(observation, available)
        policy.reset()
        assert torch.count_nonzero(policy.mac.hidden_states) == 0
        np.testing.assert_array_equal(policy.act(observation, available), first)
        with pytest.raises(ValueError, match="mask"):
            policy.act(observation, np.zeros_like(available))
    finally:
        env.close()
    payload = read_checkpoint(result.training_checkpoint)
    assert payload["kind"] == "training" and payload["learner"]["counters"]
    assert "agent_optimiser" in payload["learner"] if algorithm == "coma" else "optimiser" in payload["learner"]
    (EVIDENCE / f"smoke-{algorithm}.json").write_text(json.dumps({"train": asdict(result), "evaluation": asdict(evaluation)}, indent=2), encoding="utf-8")


def _fixed_batch(scheme, groups, preprocess):
    batch = EpisodeBatch(scheme, groups, 2, 4, preprocess=preprocess)
    rng = np.random.default_rng(701)
    batch.update({"state": rng.normal(size=(2, 4, 3)), "obs": rng.normal(size=(2, 4, 2, 4)),
                  "avail_actions": np.ones((2, 4, 2, 3), dtype=int),
                  "actions": rng.integers(0, 3, (2, 4, 2, 1)), "reward": rng.normal(size=(2, 4, 1)),
                  "terminated": np.zeros((2, 4, 1), dtype=int)})
    batch["terminated"][0, 2] = 1
    batch["terminated"][1, 1] = 1
    batch["filled"][1, 3] = 0
    return batch


@pytest.mark.parametrize("algorithm", ["qmix", "vdn", "iql", "coma", "qtran"])
def test_original_network_loss_and_update_parity(algorithm, monkeypatch):
    """Original modules load only for validation, never used by installed package."""
    source = ROOT.parent / "pymarl-master/pymarl-master/src"
    cfg = load_config(algorithm, "toy", overrides=options(algorithm))
    info = make_env("toy").get_env_info()
    torch.manual_seed(2026)
    args, scheme, groups, preprocess, mac, learner = _make_model(cfg, info, Logger())
    batch = _fixed_batch(scheme, groups, preprocess)
    prefixes = ("components", "controllers", "learners", "modules", "utils")
    with monkeypatch.context() as context:
        context.syspath_prepend(str(source))
        for name in list(sys.modules):
            if name.split(".")[0] in prefixes:
                context.delitem(sys.modules, name)
        old_controllers = importlib.import_module("controllers")
        old_learners = importlib.import_module("learners")
        torch.manual_seed(2026)
        old_mac = old_controllers.REGISTRY["basic_mac"](batch.scheme, groups, args)
        old_logger = Logger()
        old_learner = old_learners.REGISTRY[cfg["learner"]](old_mac, batch.scheme, old_logger, args)
        mac.init_hidden(2)
        old_mac.init_hidden(2)
        output_error = 0.0
        for t in range(4):
            actual, expected = mac.forward(batch, t), old_mac.forward(batch, t)
            torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
            output_error = max(output_error, float((actual - expected).abs().max().detach()))
        old_learner.train(copy.deepcopy(batch), 8, 2)
        metrics = learner.update(copy.deepcopy(batch), UpdateContext(8, 2, 0))
        key = "coma_loss" if algorithm == "coma" else "loss"
        loss_error = abs(metrics["loss"] - old_logger.latest[key])
        assert loss_error <= 1e-6
        parameter_error = 0.0
        for name in ["mac", "mixer", "critic"]:
            a, b = getattr(learner, name, None), getattr(old_learner, name, None)
            if a is None:
                continue
            for x, y in zip(a.parameters(), b.parameters()):
                torch.testing.assert_close(x, y, atol=1e-7, rtol=1e-6)
                parameter_error = max(parameter_error, float((x - y).abs().max().detach()))
        for name in list(sys.modules):
            if name.split(".")[0] in prefixes:
                context.delitem(sys.modules, name)
    (EVIDENCE / f"parity-{algorithm}.json").write_text(json.dumps({"algorithm": algorithm, "seed": 2026,
        "network_max_abs_error": output_error, "loss_abs_error": loss_error, "updated_parameter_max_abs_error": parameter_error,
        "atol": 1e-7, "rtol": 1e-6, "source": str(source), "result": "PASS"}, indent=2), encoding="utf-8")


def test_full_state_resume_matches_uninterrupted_toy(tmp_path):
    opts = dict(options("qmix"), t_max=16)
    first = train("qmix", output_dir=tmp_path / "first", overrides=opts)
    resumed = train(resume=first.training_checkpoint, output_dir=tmp_path / "resumed", overrides={"t_max": 24})
    whole = train("qmix", output_dir=tmp_path / "whole", overrides=dict(opts, t_max=24))
    a, b = read_checkpoint(resumed.training_checkpoint), read_checkpoint(whole.training_checkpoint)
    for key in ["agent", "target_agent", "mixer", "target_mixer"]:
        for name in a["learner"][key]:
            torch.testing.assert_close(a["learner"][key][name], b["learner"][key][name], atol=0, rtol=0)
    assert a["progress"] == b["progress"]
    with pytest.raises(ValueError, match="training.pt"):
        train(resume=first.checkpoint, output_dir=tmp_path / "bad")
    (EVIDENCE / "resume.json").write_text(json.dumps({"result": "PASS", "scope": "QMIX deterministic toy, episode boundary, exact network states", "progress": a["progress"]}, indent=2), encoding="utf-8")


def test_configuration_precedence_and_validation(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("lr: 0.001\nenv_args:\n  episode_limit: 5\n", encoding="utf-8")
    cfg = load_config("qmix", "toy", path, {"lr": 0.002, "env_args.episode_limit": 6})
    assert cfg["lr"] == 0.002 and cfg["env_args"] == {"n_agents": 2, "episode_limit": 6, "truncate": False}
    assert cfg["mixer"] == "qmix" and cfg["env"] == "toy"
    with pytest.raises(ValueError, match="requires"):
        load_config("qmix", overrides={"mixer": "vdn"})


@pytest.mark.parametrize("truncate", [True, False])
def test_terminal_and_truncation_masks(truncate):
    cfg = load_config("qmix", "toy", overrides={"env_args.truncate": truncate})
    runner = RUNNERS["episode"](SimpleNamespace(**cfg), Logger())
    info = runner.get_env_info()
    args, scheme, groups, preprocess, mac, learner = _make_model(cfg, info, Logger())
    runner.args = args
    runner.setup(scheme, groups, preprocess, mac)
    try:
        batch = runner.run()
        assert batch.max_seq_length == 5
        assert batch["filled"].sum() == 5  # four transitions + one bootstrap state
        assert batch["terminated"][0, 3, 0].item() == int(not truncate)
        assert batch["truncated"][0, 3, 0].item() == int(truncate)
        assert batch["terminated"][0, :3].sum() == 0
        selected = batch["avail_actions"].gather(-1, batch["actions"])
        assert selected.all()
    finally:
        runner.close_env()


def test_replay_wraparound_clears_stale_padding():
    scheme, groups, preprocess = make_scheme(make_env("toy").get_env_info())
    replay = ReplayBuffer(scheme, groups, 2, 5, preprocess=preprocess)
    long = EpisodeBatch(scheme, groups, 2, 5, preprocess=preprocess)
    long.update({"reward": torch.ones(2, 5, 1)})
    replay.insert_episode_batch(long)
    short = EpisodeBatch(scheme, groups, 1, 2, preprocess=preprocess)
    short.update({"reward": torch.zeros(1, 2, 1)})
    replay.insert_episode_batch(short)
    assert replay["filled"][0, 2:].sum() == 0
    assert replay["reward"][0].sum() == 0
    replay.extend({"roles": {"vshape": (1,), "group": "agents", "dtype": torch.long}})
    assert replay["roles"].shape == (2, 5, 2, 1)
    assert replay["filled"][1].sum() == 5


def test_optional_smac_missing_is_explicit():
    if importlib.util.find_spec("smac"):
        pytest.skip("SMAC installed; actual environment needs separate smoke")
    with pytest.raises(ImportError, match="optional"):
        make_env("sc2", map_name="3m")


def test_import_does_not_import_training_or_optional_environments():
    code = "import sys,unified_marl; assert 'torch' not in sys.modules; assert 'smac' not in sys.modules; assert 'sacred' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT / ".work")


def test_parallel_cleanup_on_worker_error():
    cfg = load_config("qmix", overrides={"runner": "parallel", "batch_size_run": 2, "env_args.state_last_action": True, "worker_timeout": 5})
    import multiprocessing as mp
    before = {p.pid for p in mp.active_children()}
    with pytest.raises(RuntimeError, match="failed"):
        RUNNERS["parallel"](SimpleNamespace(**cfg), Logger())
    assert {p.pid for p in mp.active_children()} == before


@pytest.mark.parametrize("algorithm", ["qmix_beta", "iql_beta", "vdn_beta"])
def test_beta_resources_load(algorithm):
    cfg = load_config(algorithm, "toy")
    assert cfg["algorithm"] == "pymarl/" + algorithm
