"""Exercise importable algorithms across real simulators and process boundaries."""
from dataclasses import replace
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace
import uuid

import numpy as np
import pytest
import torch

from unified_marl import evaluate, load_policy, train
from unified_marl.api import make_scheme
from unified_marl.checkpoints import read_checkpoint
from unified_marl.config import load_config
from unified_marl.envs import EnvironmentSpec, make_env
from unified_marl.registry import get_algorithm_spec, register_algorithm


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_MODULE = "unified_marl.contrib.huber_iql"
EXAMPLE_ALGORITHM = "example/huber_iql"


@pytest.fixture(autouse=True)
def fast_cpu():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def options(**updates):
    values = dict(batch_size=2, buffer_size=8, t_max=16, test_nepisode=2,
                  rnn_hidden_dim=8, num_threads=1, learner_log_interval=1,
                  target_update_interval=2, worker_timeout=20)
    values.update(updates)
    return values


def inventory_problem(path):
    path.mkdir()
    agents = path / "agents.csv"
    agents.write_text(
        "agent_id,warehouse_id,sku_id,initial_stock,capacity,mean_demand,lead_time,unit_price,unit_cost,holding_cost,shortage_cost\n"
        "a,w1,s1,12,50,4,1,10,2,0.1,1\n"
        "b,w1,s2,8,40,3,1,8,1,0.1,2\n", encoding="utf-8")
    demands = path / "demand.csv"
    demands.write_text("period,agent_id,demand\n" + "".join(
        f"{t},{agent},{demand}\n" for t in range(4) for agent, demand in [("a", 4), ("b", 3)]), encoding="utf-8")
    return dict(agents_file=str(agents), demand_file=str(demands), horizon=4,
                action_multipliers=[0, 1, 2])


def run_python(script, path, *, module_path=None, arguments=()):
    file = path / ("child_" + uuid.uuid4().hex + ".py")
    # Resume may spawn environment workers on Windows; keep the child entry guarded.
    file.write_text('if __name__ == "__main__":\n' + textwrap.indent(textwrap.dedent(script), "    "), encoding="utf-8")
    environment = dict(os.environ)
    paths = [str(ROOT / "src")]
    if module_path:
        paths.insert(0, str(module_path))
    if environment.get("PYTHONPATH"):
        paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    result = subprocess.run([sys.executable, "-X", "utf8", "-B", str(file), *map(str, arguments)],
                            env=environment, cwd=path, text=True, encoding="utf-8",
                            capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("environment", ["mpe", "inventory"])
def test_example_plugin_trains_on_two_real_environment_families(environment, tmp_path):
    args = ({"episode_limit": 4} if environment == "mpe" else inventory_problem(tmp_path / "problem"))
    result = train(EXAMPLE_ALGORITHM, environment, args, plugins=[EXAMPLE_MODULE],
                   output_dir=tmp_path / "run", overrides=options())
    assert result.updates > 0 and result.parameter_delta_l2 > 0
    assert all(np.isfinite(value) for value in result.metrics.values())
    saved = read_checkpoint(result.training_checkpoint)
    assert saved["config"]["plugins"] == [EXAMPLE_MODULE]
    assert saved["config"]["algorithm"] == EXAMPLE_ALGORITHM
    assert saved["config"]["env"] == environment
    assessment = evaluate(result.checkpoint, episodes=2)
    assert assessment.episodes == 2 and assessment.env_steps == 8
    policy = load_policy(result.checkpoint)
    env = make_env(environment, **saved["config"]["env_args"])
    env.reset()
    expected_actions = policy.act(env.get_obs(), env.get_avail_actions()).tolist()
    env.close()
    restored = run_python("""
        import json, sys
        from unified_marl import load_policy, evaluate
        from unified_marl.checkpoints import read_checkpoint
        from unified_marl.envs import make_env
        checkpoint = sys.argv[1]
        policy = load_policy(checkpoint)  # Imports saved plugins in a clean process.
        cfg = read_checkpoint(checkpoint)['config']
        env = make_env(cfg['env'], **cfg['env_args'])
        env.reset()
        actions = policy.act(env.get_obs(), env.get_avail_actions()).tolist()
        env.close()
        result = evaluate(checkpoint, episodes=1)
        print(json.dumps({'actions': actions, 'steps': result.env_steps}))
    """, tmp_path, arguments=[result.checkpoint])
    assert restored == {"actions": expected_actions, "steps": 4}


def test_example_plugin_parallel_sampling_and_resume(tmp_path):
    first = train(EXAMPLE_ALGORITHM, "mpe", {"episode_limit": 4},
                  plugins=[EXAMPLE_MODULE], output_dir=tmp_path / "first",
                  overrides=options(runner="parallel", batch_size_run=2, t_max=8))
    before = read_checkpoint(first.training_checkpoint)
    resumed = train(resume=first.training_checkpoint, output_dir=tmp_path / "resumed",
                    overrides={"t_max": 16})
    after = read_checkpoint(resumed.training_checkpoint)
    assert resumed.updates > first.updates
    assert resumed.env_steps == 16 and after["progress"]["episodes"] == 4
    assert resumed.parameter_delta_l2 > 0
    assert before["config"]["plugins"] == after["config"]["plugins"]
    with pytest.raises(ValueError, match="plugins"):
        train(resume=first.training_checkpoint, plugins=[], output_dir=tmp_path / "bad")


PLUGIN_SOURCE = '''
import torch
from unified_marl.controllers.basic_controller import BasicMAC
from unified_marl.contrib.huber_iql import HuberIQLLearner
from unified_marl.envs import EnvironmentSpec, register_environment
from unified_marl.envs.mpe import MPEEnv
from unified_marl.registry import AlgorithmSpec, register_algorithm
from unified_marl.config import resource
from unified_marl.checkpoints import controller_state, restore_controller

class TraceMAC(BasicMAC):
    def __init__(self, *args):
        super().__init__(*args)
        self.persistent_tag = torch.tensor(23)

    def select_actions(self, batch, t_ep, t_env, bs=slice(None), test_mode=False):
        actions = super().select_actions(batch, t_ep, t_env, bs=bs, test_mode=test_mode)
        self.rollout_data = {'trace_token': torch.full((*actions.shape, 1),
                                                     float(t_ep + 1), device=actions.device)}
        return actions

    def state_dict(self):
        return {'agent': self.agent.state_dict(), 'persistent_tag': self.persistent_tag}

    def load_state_dict(self, state):
        self.agent.load_state_dict(state['agent'])
        self.persistent_tag = state['persistent_tag'].clone()

def trace_scheme(info, config):
    return {'trace_token': {'vshape': (1,), 'group': 'agents'}}

def illegal_scheme(info, config):
    return {'obs': {'vshape': (999,), 'group': 'agents'}}

class StatefulLearner(HuberIQLLearner):
    def __init__(self, *args):
        super().__init__(*args)
        self.calls = 0

    def update(self, batch, context):
        mask = batch['filled'].unsqueeze(2).expand_as(batch['trace_token']).bool()
        assert (batch['trace_token'][mask] > 0).all(), 'schema data was lost during rollout/replay'
        result = super().update(batch, context)
        self.calls += 1
        self.mac.persistent_tag += 1
        return dict(result, calls=float(self.calls))

    def extra_state(self):
        return {'calls': self.calls}

    def load_extra_state(self, state):
        self.calls = state['calls']

class FullStateLearner(StatefulLearner):
    def checkpoint_state(self):
        return {'controller': controller_state(self.mac),
                'target_controller': controller_state(self.target_mac),
                'optimiser': self.optimiser.state_dict(), 'calls': self.calls,
                'last_target_update_episode': self.last_target_update_episode,
                'log_stats_t': self.log_stats_t,
                'epsilon': float(self.mac.action_selector.epsilon)}

    def load_checkpoint_state(self, state):
        restore_controller(self.mac, state['controller'])
        restore_controller(self.target_mac, state['target_controller'])
        self.optimiser.load_state_dict(state['optimiser'])
        self.calls = state['calls']
        self.last_target_update_episode = state['last_target_update_episode']
        self.log_stats_t = state['log_stats_t']
        self.mac.action_selector.epsilon = state['epsilon']

def make_particles(**kwargs):
    return MPEEnv(**kwargs)

defaults = dict(resource('algs/iql.yaml'), huber_delta=1., learner='trace', buffer_size=8)
locked = {'mixer': None, 'mac': 'basic_mac', 'agent': 'rnn',
          'agent_output_type': 'q', 'action_selector': 'epsilon_greedy'}
for name, learner, scheme, mode in (
    ('trace', 'StatefulLearner', 'trace_scheme', 'off_policy'),
    ('fullstate', 'FullStateLearner', 'trace_scheme', 'off_policy'),
    ('fresh', 'StatefulLearner', 'trace_scheme', 'on_policy'),
    ('invalid_scheme', 'StatefulLearner', 'illegal_scheme', 'off_policy'),
):
    register_algorithm(AlgorithmSpec(name='review/' + name, controller=__name__ + ':TraceMAC',
        learner=__name__ + ':' + learner, defaults=defaults, locked_fields=locked,
        scheme_factory=__name__ + ':' + scheme, training_mode=mode))
register_environment(EnvironmentSpec(name='review_particles', factory=make_particles,
    defaults={'env_args': {'episode_limit': 4, 'n_agents': 2}}))
'''


@pytest.fixture
def external_plugin(tmp_path, monkeypatch):
    directory = tmp_path / "plugins"
    directory.mkdir()
    name = "review_plugin_" + uuid.uuid4().hex
    (directory / (name + ".py")).write_text(textwrap.dedent(PLUGIN_SOURCE), encoding="utf-8")
    monkeypatch.syspath_prepend(str(directory))
    yield name, directory
    from unified_marl.envs import REGISTRY, SPECS
    from unified_marl.registry import _PLUGINS
    for key in ("review/trace", "review/fullstate", "review/fresh", "review/invalid_scheme"):
        _PLUGINS.pop(key, None)
    REGISTRY.pop("review_particles", None)
    SPECS.pop("review_particles", None)
    sys.modules.pop(name, None)


@pytest.mark.parametrize("algorithm", ["review/trace", "review/fullstate"])
def test_plugin_schema_and_complete_state_survive_spawn_and_independent_resume(algorithm, external_plugin, tmp_path):
    module_name, module_path = external_plugin
    first = train(algorithm, "review_particles", plugins=[module_name], output_dir=tmp_path / "first",
                  overrides=options(runner="parallel", batch_size_run=2, t_max=8))
    payload = read_checkpoint(first.training_checkpoint)
    key = "plugin_training_state" if algorithm.endswith("fullstate") else "extra"
    assert payload["learner"][key]["calls"] == first.updates
    assert payload["controller"]["persistent_tag"].item() == 23 + first.updates
    assert payload["replay"]["transition_data"]["trace_token"].max().item() == 5
    child = run_python("""
        import json, sys
        from unified_marl import train, load_policy, evaluate
        from unified_marl.checkpoints import read_checkpoint
        before = read_checkpoint(sys.argv[1])
        policy = load_policy(sys.argv[1])
        first_tag = policy.mac.persistent_tag.item()
        continued = train(resume=sys.argv[1], output_dir=sys.argv[2], overrides={'t_max': 16})
        after = read_checkpoint(continued.training_checkpoint)
        key = 'plugin_training_state' if 'plugin_training_state' in after['learner'] else 'extra'
        evaluation = evaluate(continued.checkpoint, episodes=1)
        print(json.dumps({'before_tag': first_tag, 'after_tag': after['controller']['persistent_tag'].item(),
                          'calls': after['learner'][key]['calls'], 'updates': continued.updates,
                          'steps': evaluation.env_steps}))
    """, tmp_path, module_path=module_path,
        arguments=[first.training_checkpoint, tmp_path / "child_resumed"])
    assert child["before_tag"] == 23 + first.updates
    assert child["calls"] == child["updates"] == first.updates + 1
    assert child["after_tag"] == 23 + child["updates"] and child["steps"] == 4


def test_on_policy_plugin_consumes_fresh_rollout_without_replay(external_plugin, tmp_path):
    module, _ = external_plugin
    with pytest.raises(ValueError, match="fresh rollouts"):
        load_config("review/fresh", "mpe", config={"plugins": [module]}, overrides=options())
    result = train("review/fresh", "mpe", {"episode_limit": 4}, plugins=[module],
                   output_dir=tmp_path / "fresh", overrides=options(batch_size=1, t_max=8))
    saved = read_checkpoint(result.training_checkpoint)
    assert saved["replay"] is None
    assert result.updates == result.episodes == 2


def test_scheme_factory_cannot_replace_core_transition_fields(external_plugin):
    module, _ = external_plugin
    config = load_config("review/invalid_scheme", "mpe", config={"plugins": [module]}, overrides=options())
    with pytest.raises(ValueError, match="standard transition fields"):
        make_scheme(make_env("mpe").get_env_info(), config)


@pytest.mark.parametrize("reserved", ["filled", "actions_onehot"])
def test_scheme_factory_rejects_automatically_generated_fields(external_plugin, monkeypatch, reserved):
    module_name, _ = external_plugin
    config = load_config("review/invalid_scheme", "mpe", config={"plugins": [module_name]}, overrides=options())
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "illegal_scheme", lambda info, cfg: {reserved: {"vshape": (1,)}})
    with pytest.raises(ValueError, match="(reserved|standard transition|replacing)"):
        make_scheme(make_env("mpe").get_env_info(), config)


def test_rollout_field_without_schema_is_not_silently_dropped(external_plugin, monkeypatch, tmp_path):
    module_name, _ = external_plugin
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "trace_scheme", lambda info, cfg: {})
    with pytest.raises(ValueError, match="rollout_data.*scheme_factory"):
        train("review/trace", "mpe", {"episode_limit": 4}, plugins=[module_name],
              output_dir=tmp_path / "undeclared", overrides=options(batch_size=1, t_max=4))


def test_checkpoint_writer_requires_matching_restore_hook():
    from unified_marl.checkpoints import learner_state
    with pytest.raises(ValueError, match="load_checkpoint_state"):
        learner_state(SimpleNamespace(checkpoint_state=lambda: {}))


def test_environment_capability_rejection_precedes_factory_execution(monkeypatch):
    from unified_marl.envs import REGISTRY, SPECS
    importlib.import_module(EXAMPLE_MODULE)
    called = []
    def forbidden_factory(**kwargs):
        called.append(kwargs)
        raise AssertionError("incompatible simulator must not be constructed")
    name = "review_continuous"
    spec = EnvironmentSpec(name, forbidden_factory, capabilities=("continuous", "fixed_agents", "team_reward"))
    monkeypatch.setitem(SPECS, name, spec)
    monkeypatch.setitem(REGISTRY, name, forbidden_factory)
    with pytest.raises(ValueError, match="capabilities"):
        load_config(EXAMPLE_ALGORITHM, name)
    assert not called


@pytest.mark.parametrize("key,value", [("n_agents", 0), ("n_actions", 2.5),
                                      ("obs_shape", True), ("state_shape", -1), ("episode_limit", 0)])
def test_invalid_environment_metadata_is_rejected_before_model_build(key, value):
    from unified_marl.api import _make_model
    from unified_marl.utils.logging import Logger
    config = load_config(EXAMPLE_ALGORITHM, "mpe", config={"plugins": [EXAMPLE_MODULE]}, overrides=options())
    info = make_env("mpe").get_env_info()
    info[key] = value
    with pytest.raises(ValueError, match="positive integer"):
        _make_model(config, info, Logger())


def test_algorithm_defaults_and_locks_do_not_bind_environment():
    importlib.import_module(EXAMPLE_MODULE)
    spec = get_algorithm_spec(EXAMPLE_ALGORITHM)
    for defaults in ({"env": "mpe"}, {"env_args": {"episode_limit": 4}}):
        with pytest.raises(ValueError, match="bind an environment"):
            register_algorithm(replace(spec, name="review/bound", defaults=defaults))
    with pytest.raises(ValueError, match="requires mixer"):
        load_config(EXAMPLE_ALGORITHM, "mpe", overrides={"mixer": "qmix"})
    with pytest.raises(ValueError, match="built-in"):
        register_algorithm(replace(spec, name="pymarl/new_algorithm"))


def test_huber_variant_uses_linear_tail_and_ignores_padding():
    from unified_marl.contrib.huber_iql import HuberIQLLearner
    # A direct numerical contract for the advertised loss, not an implementation mirror.
    owner = SimpleNamespace(huber_delta=1.)
    errors = torch.tensor([0.5, -2., 1000.], requires_grad=True)
    loss = HuberIQLLearner.td_loss(owner, errors, torch.tensor([1., 1., 0.]))
    assert loss.item() == pytest.approx((0.125 + 1.5) / 2)
    loss.backward()
    torch.testing.assert_close(errors.grad, torch.tensor([0.25, -0.5, 0.]))


def test_legacy_offline_configuration_does_not_require_online_environment():
    cfg = load_config("kou/bc", env="unused_offline_environment", overrides={"training_updates": 1})
    assert cfg["kou_family"] == "icq" and cfg["training_updates"] == 1
    assert cfg["env_config"] == "unused_offline_environment"
