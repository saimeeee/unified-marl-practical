"""Orchestration unit tests use mocked backends, not real training evidence."""
import csv
import json
from pathlib import Path
import sys

import pytest
import yaml

from unified_marl import experiments as ex


def write_yaml(tmp_path, value, name="experiment.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(value, allow_unicode=True), encoding="utf-8")
    return path


def single(**changes):
    return {"algorithm": "pymarl/qmix", "env": "toy", "seed": 7,
            "training": {"t_max": 12}, **changes}


@pytest.fixture
def backend(monkeypatch):
    calls = {"train": [], "evaluate": [], "config": []}

    def config(algorithm, env, config=None, overrides=None):
        calls["config"].append((algorithm, env, config))
        defaults = {"algorithm": algorithm, "env": env, "env_args": {},
                    "t_max": 100, "max_updates": None, "batch_size": 2,
                    "gamma": .99, "options": {"scale": 1}}
        return ex.merge(ex.merge(defaults, config), ex.dotted_overrides(overrides))

    def train(algorithm, env, env_args, **kwargs):
        calls["train"].append((algorithm, env, env_args, kwargs))
        if algorithm == "fails":
            raise RuntimeError("the selected backend is unavailable")
        directory = Path(kwargs["output_dir"])
        return {"checkpoint": str(directory / "policy.pt"), "updates": 1}

    def evaluate(checkpoint, **kwargs):
        calls["evaluate"].append((checkpoint, kwargs))
        return {"episodes": kwargs["episodes"], "returns": [float(kwargs["seed"] % 100)],
                "return_mean": float(kwargs["seed"] % 100), "return_std": 0.0}

    monkeypatch.setattr(ex, "load_config", config)
    monkeypatch.setattr(ex, "train", train)
    monkeypatch.setattr(ex, "evaluate", evaluate)
    return calls


def test_paths_are_explicit_and_relative_to_yaml(tmp_path, monkeypatch):
    path = write_yaml(tmp_path, single(env_args={"agents_file": "data/agents.csv", "task_type": "Standard",
                                                "nested": {"weights": "models/a.pt"}},
                                      path_keys=["nested.weights"]))
    monkeypatch.chdir(tmp_path.parent)
    result = ex.load_experiment(path)
    assert result["env_args"]["agents_file"] == str(tmp_path / "data" / "agents.csv")
    assert result["env_args"]["nested"]["weights"] == str(tmp_path / "models" / "a.pt")
    assert result["env_args"]["task_type"] == "Standard"
    assert result["evaluation"]["seeds"] != [result["seed"]]


@pytest.mark.parametrize("changes,match", [
    ({"trianing": {}}, "Unknown experiment"),
    ({"seed": True}, "seed must"),
    ({"env_args": {"seed": 1}}, "top level"),
    ({"plugins": "my_plugin"}, "plugins must"),
    ({"path_keys": ["missing"]}, "does not exist"),
    ({"evaluation": {"seeds": [7]}}, "disjoint"),
    ({"evaluation": {"seeds": []}}, "nonempty"),
    ({"evaluation": {"episodes": 0}}, "positive"),
    ({"schema_version": True}, "schema_version"),
])
def test_invalid_experiment_rejected(tmp_path, changes, match):
    with pytest.raises(ValueError, match=match):
        ex.load_experiment(write_yaml(tmp_path, single(**changes)))


def test_duplicate_yaml_keys_rejected(tmp_path):
    path = tmp_path / "duplicate.yaml"
    path.write_text("algorithm: qmix\nenv: toy\nenv: inventory\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate YAML key"):
        ex.load_experiment(path)


def test_run_forwards_every_setting_and_keeps_environment(tmp_path, backend):
    path = write_yaml(tmp_path, single(env="inventory", env_args={"agents_file": "agents.csv"},
                                     evaluation={"episodes": 3, "seeds": [101, 102]}))
    result = ex.run_experiment(path, tmp_path / "run", overrides={"gamma": .8, "env_args.agents_file": "other.csv"})
    algorithm, env, env_args, kwargs = backend["train"][0]
    assert (algorithm, env) == ("pymarl/qmix", "inventory")
    assert env_args["agents_file"] == str(tmp_path / "other.csv")
    assert kwargs["config"] == {"t_max": 12, "plugins": []}
    assert kwargs["overrides"] == {"gamma": .8}
    assert kwargs["seed"] == 7
    assert [call[1]["seed"] for call in backend["evaluate"]] == [101, 102]
    assert all(call[1]["env"] == "inventory" for call in backend["evaluate"])
    assert result["status"] == "success"
    saved = json.loads((tmp_path / "run" / "experiment_result.json").read_text(encoding="utf-8"))
    assert saved["evaluation_return_mean"] == 1.5


@pytest.mark.parametrize("training,overrides", [
    ({"t_mx": 12}, None), ({"seed": 1}, None), ({"save_model": True}, None),
    ({"options": {"typo": 2}}, None), ({}, {"gama": .9}), ({}, {"device": "cpu"}),
    ({}, {"env_args.seed": 4}), ({}, {"gamma": float("nan")}),
])
def test_unrecognised_or_ignored_parameters_are_recorded_failures(tmp_path, backend, training, overrides):
    path = write_yaml(tmp_path, single(training=training))
    with pytest.raises(ex.ExperimentRunError) as caught:
        ex.run_experiment(path, tmp_path / "run", overrides=overrides)
    assert caught.value.result["status"] == "failed"
    assert not backend["train"]
    assert json.loads((tmp_path / "run" / "experiment_result.json").read_text())["status"] == "failed"


def test_plugins_import_before_configuration_and_persist(tmp_path, backend, monkeypatch):
    (tmp_path / "experiment_plugin.py").write_text("registered = True\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    previous = ex.load_config

    def config(*args, **kwargs):
        assert sys.modules["experiment_plugin"].registered
        return previous(*args, **kwargs)

    monkeypatch.setattr(ex, "load_config", config)
    path = write_yaml(tmp_path, single(plugins=["experiment_plugin"]))
    assert "experiment_plugin" not in sys.modules
    ex.load_experiment(path)
    assert "experiment_plugin" not in sys.modules
    try:
        ex.run_experiment(path, tmp_path / "run")
        assert backend["config"][0][2] == {"plugins": ["experiment_plugin"]}
        assert backend["train"][0][3]["config"]["plugins"] == ["experiment_plugin"]
    finally:
        sys.modules.pop("experiment_plugin", None)


def test_existing_output_never_overwritten(tmp_path, backend):
    path = write_yaml(tmp_path, single())
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("existing work")
    with pytest.raises(FileExistsError):
        ex.run_experiment(path, output)
    assert marker.read_text() == "existing work"
    assert not backend["train"]


def test_suite_cross_product_and_failures_have_final_status(tmp_path, backend):
    path = write_yaml(tmp_path, {"algorithms": ["qmix", "fails"],
                                 "envs": ["toy", {"name": "inventory", "env_args": {"agents_file": "a.csv"}}],
                                 "seeds": [1, 2], "training": {"t_max": 12},
                                 "evaluation": {"seed_count": 2}}, "suite.yaml")
    result = ex.run_suite(path, tmp_path / "suite")
    assert result["expected_runs"] == 8
    assert result["successful_runs"] == result["failed_runs"] == 4
    assert result["status"] == "partial_failure"
    assert len({row["output_dir"] for row in result["runs"]}) == 8
    assert len(result["summary"]) == 4
    for group in result["summary"]:
        assert "rank" not in group
        assert group["successful_runs"] + group["failed_runs"] == 2
    assert all(call[1]["seed"] not in {1, 2} for call in backend["evaluate"])
    saved = json.loads((tmp_path / "suite" / "suite_result.json").read_text(encoding="utf-8"))
    assert saved["status"] == "partial_failure"
    with (tmp_path / "suite" / "suite_result.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8 and sum(row["status"] == "failed" for row in rows) == 4


@pytest.mark.parametrize("axis", ["algorithms", "envs", "seeds"])
def test_empty_matrix_rejected(tmp_path, backend, axis):
    value = {"algorithms": ["qmix"], "envs": ["toy"], "seeds": [1], axis: []}
    with pytest.raises(ValueError, match="nonempty"):
        ex.run_suite(write_yaml(tmp_path, value), tmp_path / "suite")
    assert not (tmp_path / "suite").exists()


def test_suite_eval_seeds_disjoint_from_whole_matrix(tmp_path, backend):
    value = {"algorithms": ["qmix"], "envs": ["toy"], "seeds": [1, 1000004]}
    result = ex.run_suite(write_yaml(tmp_path, value), tmp_path / "suite")
    assert result["status"] == "success"
    assert {call[1]["seed"] for call in backend["evaluate"]} == {1000005}


def test_suite_all_failed_status(tmp_path, backend):
    value = {"algorithms": ["fails"], "envs": ["toy"], "seeds": [1]}
    result = ex.run_suite(write_yaml(tmp_path, value), tmp_path / "suite")
    assert result["status"] == "failed"
    assert result["summary"][0]["return_mean"] is None


def test_invalid_backend_metrics_record_failure(tmp_path, backend, monkeypatch):
    monkeypatch.setattr(ex, "evaluate", lambda *a, **k: {"return_mean": float("nan")})
    with pytest.raises(ex.ExperimentRunError) as caught:
        ex.run_experiment(write_yaml(tmp_path, single()), tmp_path / "run")
    assert caught.value.result["error"]["type"] == "ValueError"
    assert json.loads((tmp_path / "run" / "experiment_result.json").read_text())["status"] == "failed"


def test_suite_interrupt_is_persisted_and_propagated(tmp_path, backend, monkeypatch):
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(ex, "train", interrupted)
    value = {"algorithms": ["qmix"], "envs": ["toy"], "seeds": [1, 2]}
    with pytest.raises(KeyboardInterrupt):
        ex.run_suite(write_yaml(tmp_path, value), tmp_path / "suite")
    result = json.loads((tmp_path / "suite" / "suite_result.json").read_text(encoding="utf-8"))
    assert result["status"] == result["runs"][0]["status"] == "interrupted"
    assert len(result["runs"]) == 1


def test_check_defaults_to_configuration_without_environment(tmp_path, backend, monkeypatch):
    import unified_marl.envs as envs

    def should_not_construct(*args, **kwargs):
        pytest.fail("configuration-only checks must not construct the environment")

    monkeypatch.setattr(envs, "make_env", should_not_construct)
    result = ex.check_experiment(write_yaml(tmp_path, single()), overrides={"gamma": .8})
    assert result["validation_level"] == "configuration"
    assert result["config"]["gamma"] == .8
    assert not result["instantiated"] and not result["training_executed"]
    assert not backend["train"] and not backend["evaluate"]


@pytest.mark.parametrize("fail", [False, True])
def test_check_instantiation_always_closes_and_never_steps(tmp_path, backend, monkeypatch, fail):
    import unified_marl.envs as envs
    closed = []

    class MetadataOnlyEnv:
        def get_env_info(self):
            if fail:
                raise RuntimeError("metadata unavailable")
            return {"n_agents": 2, "n_actions": 3, "state_shape": 4, "obs_shape": 2, "episode_limit": 4}

        def close(self):
            closed.append(True)

        def step(self, actions):
            pytest.fail("environment check must not step")

    monkeypatch.setattr(envs, "make_env", lambda *a, **k: MetadataOnlyEnv())
    path = write_yaml(tmp_path, single())
    if fail:
        with pytest.raises(RuntimeError, match="metadata unavailable"):
            ex.check_experiment(path, instantiate=True)
    else:
        result = ex.check_experiment(path, instantiate=True)
        assert result["validation_level"] == "environment_metadata"
        assert result["env_info"]["n_agents"] == 2
    assert closed == [True]
    assert not backend["train"] and not backend["evaluate"]


def test_configuration_cannot_silently_replace_environment(tmp_path, backend, monkeypatch):
    previous = ex.load_config

    def redirect(*args, **kwargs):
        return dict(previous(*args, **kwargs), env="toy")

    monkeypatch.setattr(ex, "load_config", redirect)
    path = write_yaml(tmp_path, single(env="inventory"))
    with pytest.raises(ValueError, match="substitution is forbidden"):
        ex.check_experiment(path)
    assert not backend["train"]
