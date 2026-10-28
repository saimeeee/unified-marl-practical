"""Source parity is a test-only import; installed runtime never sees old roots."""
from contextlib import contextmanager, redirect_stdout
import copy
import importlib
import inspect
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace, ModuleType

import numpy as np
import pytest
import torch

from unified_marl import train, evaluate, load_policy
from unified_marl.api import _make_model
from unified_marl.algorithms.hmf.adapter import IMPLEMENTATIONS, ScalarCapture
from unified_marl.checkpoints import read_checkpoint
from unified_marl.config import load_config
from unified_marl.envs import make_env
from unified_marl.learners.base import UpdateContext
from unified_marl.runners import REGISTRY as RUNNERS
from unified_marl.utils.logging import Logger

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "marl平台代码/王思哲/code/large_scale"
EVIDENCE = ROOT / "docs/evidence/stage3"
EVIDENCE.mkdir(parents=True, exist_ok=True)


@pytest.fixture(autouse=True)
def cpu_threads(monkeypatch):
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@contextmanager
def original_modules(monkeypatch, folder, prefixes):
    with monkeypatch.context() as context:
        context.syspath_prepend(str(folder))
        for name in list(sys.modules):
            if name.split(".")[0] in prefixes:
                context.delitem(sys.modules, name)
        # utils.misc imports cv2.log without using it anywhere. Avoid installing
        # OpenCV only for this unused reference import; numerical code is intact.
        if folder.name == "hmf_replenishment":
            stub = ModuleType("cv2")
            stub.log = None
            context.setitem(sys.modules, "cv2", stub)
        try:
            yield
        finally:
            for name in list(sys.modules):
                if name.split(".")[0] in prefixes:
                    context.delitem(sys.modules, name)


def record(name, **values):
    (EVIDENCE / (name + ".json")).write_text(json.dumps(dict(result="PASS", **values), indent=2), encoding="utf-8")


def role_config(name):
    return load_config("wang/" + name, "toy", overrides={
        "runner": "parallel", "batch_size_run": 2, "batch_size": 2, "buffer_size": 8,
        "rnn_hidden_dim": 16, "role_interval": 2, "learner_log_interval": 1,
        "device": "cpu", "use_cuda": False})


def collect(cfg, mac, args, scheme, groups, preprocess):
    # A single local collector can assemble two deterministic episodes without
    # starting workers, then concatenate along B for the source HMF squeeze.
    from unified_marl.components.episode_buffer import EpisodeBatch
    runner = RUNNERS["episode"](SimpleNamespace(**dict(cfg, batch_size_run=1)), Logger())
    runner.args = SimpleNamespace(**dict(vars(args), batch_size_run=1))
    runner.setup(scheme, groups, preprocess, mac)
    try:
        episodes = [runner.run(test_mode=False), runner.run(test_mode=False)]
        result = EpisodeBatch(scheme, groups, 2, episodes[0].max_seq_length, preprocess=preprocess)
        for name in result.data.transition_data:
            result.data.transition_data[name].copy_(torch.cat([b[name] for b in episodes], dim=0))
        return result
    finally:
        runner.close_env()


def assert_modules_equal(a, b):
    error = 0.0
    for key, value in a.state_dict().items():
        other = b.state_dict()[key]
        torch.testing.assert_close(value, other, atol=1e-7, rtol=1e-6)
        if value.numel():
            error = max(error, float((value - other).abs().max()))
    return error


@pytest.mark.parametrize("name", ["rode", "rode_hmf"])
def test_role_source_outputs_losses_update_and_masks(name, monkeypatch):
    cfg = role_config(name)
    info = make_env("toy").get_env_info()
    torch.manual_seed(2026)
    args, scheme, groups, preprocess, mac, learner = _make_model(cfg, info, Logger())
    batch = collect(cfg, mac, args, scheme, groups, preprocess)
    assert batch["roles"].shape == (2, 5, 2, 1)
    assert batch["role_avail_actions"].shape == (2, 5, 2, 3)
    assert batch["agent_Q"].shape == (2, 5, 2, 3)
    assert batch["filled"].sum() == 10
    assert batch["terminated"][:, 3].all() and not batch["terminated"][:, 4].any()
    assert batch["avail_actions"].gather(-1, batch["actions"]).all()
    # The source role selector is called at 0,2,4 and held for the intervening t.
    assert torch.equal(batch["roles"][:, 0], batch["roles"][:, 1])
    assert torch.equal(batch["roles"][:, 2], batch["roles"][:, 3])
    prefixes = ("components", "controllers", "learners", "modules", "utils")
    with original_modules(monkeypatch, SOURCE / "RGHMF", prefixes):
        old_controllers = importlib.import_module("controllers")
        old_learners = importlib.import_module("learners")
        torch.manual_seed(2026)
        old_mac = old_controllers.REGISTRY["rode_mac"](batch.scheme, groups, args)
        old_logger = Logger()
        old_learner = old_learners.REGISTRY[cfg["learner"]](old_mac, batch.scheme, old_logger, args)
        mac.init_hidden(2)
        old_mac.init_hidden(2)
        output_error = 0.0
        for t in range(5):
            torch.manual_seed(900 + t)
            actual = mac.forward(batch, t)
            torch.manual_seed(900 + t)
            expected = old_mac.forward(batch, t)
            for a, b in zip(actual, expected):
                if a is not None:
                    torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-6)
                    output_error = max(output_error, float((a-b).abs().max().detach()))
            assert torch.equal(mac.selected_roles, old_mac.selected_roles)
            for a, b in zip(mac.action_repr_forward(batch, t), old_mac.action_repr_forward(batch, t)):
                torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-6)
        torch.manual_seed(301)
        old_learner.train(copy.deepcopy(batch), 8, 2)
        torch.manual_seed(301)
        metrics = learner.update(copy.deepcopy(batch), UpdateContext(8, 2, 0))
        expected_loss = old_logger.latest["loss"] + old_logger.latest["role_loss"]
        loss_error = abs(metrics["loss"] - expected_loss)
        assert loss_error <= 1e-6
        parameter_error = max(assert_modules_equal(getattr(learner, key), getattr(old_learner, key))
                              for key in ["mixer", "role_mixer"] if getattr(learner, key) is not None)
        for a, b in zip(mac.parameters(), old_mac.parameters()):
            torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-6)
            parameter_error = max(parameter_error, float((a-b).abs().max().detach()))
        if name == "rode_hmf":
            parameter_error = max(parameter_error, assert_modules_equal(learner.attention, old_learner.attention),
                                  assert_modules_equal(learner.mf_critic, old_learner.mf_critic))
            optimizer_ids = {id(p) for group in learner.optimiser.param_groups for p in group["params"]}
            assert not optimizer_ids.intersection(id(p) for p in learner.mf_critic.parameters())
            assert not optimizer_ids.intersection(id(p) for p in learner.attention.parameters())
    record("parity-" + name, network_max_abs_error=output_error, loss_abs_error=loss_error,
           updated_parameter_max_abs_error=parameter_error, source=str(SOURCE / "RGHMF"),
           scope="pre-clustering; includes source optimizer omissions for rode_hmf")


def hmf_config(name):
    return load_config("wang_hmf/" + name, "toy", overrides={
        "batch_size": 4, "buffer_size": 32, "hidden_dim": 16, "n_heads": 2, "embed_dim": 8,
        "hmf_norm_rewards": False, "hmf_steps_per_update": 1, "hmf_num_batch": 1,
        "env_args.n_agents": 4, "env_args.mask_actions": False})


def network_bundle(native):
    agent = native.agent
    return torch.nn.ModuleDict(dict(agent.named_children()) if isinstance(agent, torch.nn.Module) else {
        k: v for k, v in vars(agent).items() if isinstance(v, torch.nn.Module)})


@pytest.mark.parametrize("name", list(IMPLEMENTATIONS))
def test_hmf_native_source_loss_and_update_parity(name, monkeypatch):
    cfg = hmf_config(name)
    info = make_env("toy", n_agents=4, mask_actions=False).get_env_info()
    torch.manual_seed(2026)
    np.random.seed(2026)
    args, _, _, _, mac, _ = _make_model(cfg, info, Logger())
    from gym.spaces import Box, Discrete
    proxy = SimpleNamespace(n_agents=4, observation_space=[Box(-np.inf, np.inf, (6,), dtype=np.float32)] * 4,
                            action_space=[Discrete(3)] * 4)
    module, class_name = IMPLEMENTATIONS[name]
    with original_modules(monkeypatch, SOURCE / "hmf_replenishment", ("algorithms", "agent", "utils")):
        cls = getattr(importlib.import_module("algorithms." + module), class_name)
        supplied = dict(n_heads=2, embed_dim=8, bsz=4, num_groups=2, num_agents=4, add_agentID=False,
                        explore=args.explore, gamma=args.gamma, tau=args.tau, lr=args.lr, hidden_dim=16,
                        grad_norm_clip=args.grad_norm_clip)
        kwargs = {k: supplied[k] for k in inspect.signature(cls.init_from_env).parameters if k in supplied}
        torch.manual_seed(2026)
        np.random.seed(2026)
        with redirect_stdout(io.StringIO()):
            old = cls.init_from_env(proxy, **kwargs)
        old.device = torch.device("cpu")  # Delivery hardcodes CUDA, no numerical change.
        if name == "hierarchical_vdac_fixed":
            # PyTorch 2 rejects the original uint8 attention indexing. This
            # reference-only input conversion preserves exactly the masked set.
            attention_forward = old.agent.attention.forward
            def compatible_attention(*a, **kw):
                if kw.get("mask") is not None: kw["mask"] = kw["mask"].bool()
                return attention_forward(*a, **kw)
            monkeypatch.setattr(old.agent.attention, "forward", compatible_attention)
            def compatible_targets():
                misc = importlib.import_module("utils.misc")
                misc.soft_update(old.agent.target_local_critic, old.agent.local_critic, old.tau)
                misc.soft_update(old.agent.target_mf_critic, old.agent.mf_critic, old.tau)
                old.niter += 1
            monkeypatch.setattr(old, "update_all_targets", compatible_targets)
        old_bundle = network_bundle(old)
        assert_modules_equal(mac.agent, old_bundle)
        rng = torch.Generator().manual_seed(908)
        obs = [torch.randn(4, 6, generator=rng) for _ in range(4)]
        next_obs = [torch.randn(4, 6, generator=rng) for _ in range(4)]
        actions = [torch.nn.functional.one_hot(torch.randint(3, (4,), generator=rng), 3).float() for _ in range(4)]
        next_actions = [torch.nn.functional.one_hot(torch.randint(3, (4,), generator=rng), 3).float() for _ in range(4)]
        rewards = [torch.tensor([0.0, 1.0, -0.2, 0.3])] * 4
        dones = [torch.tensor([0.0, 0.0, 1.0, 0.0])] * 4
        if name == "mfq":
            mean = torch.stack(actions, 1).mean(1)
            next_mean = torch.stack(next_actions, 1).mean(1)
            sample = (obs, actions, [mean] * 4, rewards, next_obs, next_actions, [next_mean] * 4, dones)
        else:
            sample = (obs, actions, rewards, next_obs, next_actions, dones)
        actual_log, expected_log = ScalarCapture(), ScalarCapture()
        kwargs = {"use_next_acts": True}
        for key, value in [("mu_weight", args.hmf_mu_weight), ("high_weight", args.hmf_high_weight)]:
            if key in inspect.signature(old.update).parameters:
                kwargs[key] = value
        torch.manual_seed(701)
        np.random.seed(701)
        old.update(copy.deepcopy(sample), logger=expected_log, **kwargs)
        old.update_all_targets()
        torch.manual_seed(701)
        np.random.seed(701)
        mac.native.update(copy.deepcopy(sample), logger=actual_log, **kwargs)
        mac.native.update_all_targets()
        assert actual_log.values.keys() == expected_log.values.keys()
        loss_error = max(abs(actual_log.values[k] - expected_log.values[k]) for k in actual_log.values)
        assert loss_error <= 1e-6
        parameter_error = assert_modules_equal(mac.agent, old_bundle)
    record("parity-hmf-" + name, loss_max_abs_error=loss_error, updated_network_max_abs_error=parameter_error,
           source=str(SOURCE / "hmf_replenishment"),
           adaptations="CPU device selection; unused cv2.log import stub in reference only; identical transitions")


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_tree_equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("name", ["wang/rode", "wang/rode_hmf"] + ["wang_hmf/" + k for k in IMPLEMENTATIONS])
def test_wang_train_resume_policy_and_evaluation(name, tmp_path):
    hmf = name.startswith("wang_hmf/")
    opts = {"batch_size": 4 if hmf else 2, "buffer_size": 32, "t_max": 8, "max_updates": None,
            "runner": "episode", "batch_size_run": 1, "rnn_hidden_dim": 16, "role_interval": 2,
            "hidden_dim": 16, "n_heads": 2, "embed_dim": 8, "hmf_norm_rewards": False}
    opts.update(hmf_steps_per_update=1, hmf_num_batch=1)
    env_args = {"n_agents": 4, "mask_actions": False} if hmf else {}
    first = train(name, env_args=env_args, overrides=opts, output_dir=tmp_path / "first")
    resumed = train(resume=first.training_checkpoint, overrides={"t_max": 12}, output_dir=tmp_path / "resumed")
    whole = train(name, env_args=env_args, overrides=dict(opts, t_max=12), output_dir=tmp_path / "whole")
    assert first.parameter_delta_l2 > 0 and resumed.parameter_delta_l2 > 0
    a, b = read_checkpoint(resumed.training_checkpoint), read_checkpoint(whole.training_checkpoint)
    assert_tree_equal(a["learner"], b["learner"])
    assert a["progress"] == b["progress"]
    result = evaluate(resumed.checkpoint, episodes=2)
    assert result.episodes == 2
    policy = load_policy(resumed.checkpoint)
    env = make_env("toy", **env_args)
    try:
        env.reset()
        obs, available = env.get_obs(), np.asarray(env.get_avail_actions())
        first_action = policy.act(obs, available)
        for _ in range(3):
            action = policy.act(obs, available)
            assert available[np.arange(env.n_agents), action].all()
        policy.reset()
        np.testing.assert_array_equal(policy.act(obs, available), first_action)
    finally:
        env.close()
    record("resume-" + name.replace("/", "-"), scope="deterministic toy, episode boundary, pre-clustering",
           progress=a["progress"], training_checkpoint=resumed.training_checkpoint)


def test_mfq_mean_field_and_episode_boundaries(monkeypatch):
    cfg = hmf_config("mfq")
    info = make_env("toy", n_agents=4, mask_actions=False).get_env_info()
    args, scheme, groups, prep, mac, learner = _make_model(cfg, info, Logger())
    batch = collect(cfg, mac, args, scheme, groups, prep)
    captured = {}
    real_update = mac.native.update
    def capture(sample, **kwargs):
        captured["sample"] = sample
        return real_update(sample, **kwargs)
    monkeypatch.setattr(mac.native, "update", capture)
    learner.update(batch, UpdateContext(8, 2, 0))
    assert len(learner.records) == 8
    for ep in range(2):
        for t in range(4):
            row = learner.records[ep * 4 + t]
            torch.testing.assert_close(row["next_obs"], batch["obs"][ep, t+1])
            torch.testing.assert_close(row["next_actions"], batch["actions_onehot"][ep, t+1])
    obs, actions, means, rewards, nxt, next_actions, next_means, done = captured["sample"]
    for mean in means:
        torch.testing.assert_close(mean, torch.stack(actions, 1).mean(1))
    for mean in next_means:
        torch.testing.assert_close(mean, torch.stack(next_actions, 1).mean(1))
    record("mean-field-boundaries", valid_transitions=8, bootstrap_slots_excluded=True, mean_field="arithmetic mean across agents")


def test_source_clustering_and_role_latent_parity(monkeypatch):
    cfg = role_config("rode")
    cfg.update(n_actions=10, n_role_clusters=2, n_roles=2)
    info = dict(make_env("toy").get_env_info(), n_actions=10)
    torch.manual_seed(21)
    args, scheme, groups, prep, mac, _ = _make_model(cfg, info, Logger())
    template = __import__("unified_marl.components.episode_buffer", fromlist=["EpisodeBatch"]).EpisodeBatch(scheme, groups, 2, 5, preprocess=prep)
    with original_modules(monkeypatch, SOURCE / "RGHMF", ("components", "controllers", "learners", "modules", "utils")):
        cls = importlib.import_module("controllers").REGISTRY["rode_mac"]
        torch.manual_seed(21)
        old = cls(template.scheme, groups, args)
        # Two known well-separated clusters, SC2-specific first-six rule intact.
        representation = torch.cat([torch.zeros(5, args.action_latent_dim), torch.ones(5, args.action_latent_dim) * 10])
        monkeypatch.setattr(old.action_encoder, "forward", lambda: representation)
        monkeypatch.setattr(mac.action_encoder, "forward", lambda: representation)
        with redirect_stdout(io.StringIO()):
            old.update_role_action_spaces()
            mac.update_role_action_spaces()
        torch.testing.assert_close(mac.role_action_spaces, old.role_action_spaces, atol=0, rtol=0)
        torch.testing.assert_close(mac.role_latent, old.role_latent, atol=0, rtol=0)
        assert mac.n_roles == old.n_roles == 4
    record("clustering-parity", role_count=4, role_action_spaces=mac.role_action_spaces.tolist(),
           role_latent_max_abs_error=0, scope="controlled SC2-compatible ten-action embeddings; not post-clustering training")


def test_replenishment_adapter_contract(monkeypatch):
    from unified_marl.envs.replenishment import ReplenishmentAdapter
    captured = {}
    class Backend:
        config = {"env": {"horizon": 3}}
        action_space = [SimpleNamespace(n=3), SimpleNamespace(n=2)]
        observation_space = [SimpleNamespace(shape=(3,)), SimpleNamespace(shape=(2,))]
        closed = False
        def get_agent_count(self): return 2
        def reset(self): return [np.ones(3), np.ones(2)], {}
        def step(self, actions): return [np.ones(3)*2, np.ones(2)*2], [10000, 20000], [False]*2, [False]*2, {}
        def close(self): self.closed = True
    backend = Backend()
    module = ModuleType("ReplenishmentEnv")
    def factory(task, **kwargs):
        captured.update(task=task, **kwargs)
        return backend
    module.make_env = factory
    monkeypatch.setitem(sys.modules, "ReplenishmentEnv", module)
    env = ReplenishmentAdapter(time_limit=1, backend_args={"config_path": "external/task.yaml"})
    assert env.get_env_info() == {"state_shape": 6, "obs_shape": 3, "n_actions": 3, "n_agents": 2, "episode_limit": 1}
    np.testing.assert_array_equal(env.get_avail_actions(), [[1, 1, 1], [1, 1, 0]])
    reward, done, info = env.step([0, 1])
    assert reward == 3 and done and info["episode_limit"]
    assert captured["config_path"] == "external/task.yaml"
    assert captured["update_config"]["action"]["mode"] == "demand_mean_discrete"
    assert env.get_obs()[1, -1] == 0
    env.close()
    assert backend.closed
    record("replenishment-contract", scope="fake backend contract only; real simulator NOT validated", reward=reward)


def test_stage3_explicit_constraints():
    for name in ["wang/rode", "wang/qmix", "wang/coma", "wang_hmf/mfq"]:
        load_config(name)
    with pytest.raises(ValueError, match="Unsupported"):
        load_config("wang_hmf/hierarchical_vdn_roma")
    with pytest.raises(ValueError, match="batch_size"):
        load_config("wang/rode_hmf", overrides={"batch_size": 1})
    with pytest.raises(ValueError, match="at least two roles"):
        load_config("wang/rode", overrides={"n_roles": 1})
    with pytest.raises(ImportError, match="simulator"):
        make_env("replenishment")


def test_hmf_update_schedule_and_noise(tmp_path):
    result = train("wang_hmf/dqn", env_args={"n_agents": 4, "mask_actions": False},
        output_dir=tmp_path / "scheduled", overrides={"batch_size": 4, "buffer_size": 32, "t_max": 12,
        "hidden_dim": 16, "hmf_norm_rewards": False, "hmf_steps_per_update": 8, "hmf_num_batch": 2,
        "explore": 0.8, "hmf_noise_decay": 0.1, "hmf_final_noise_scale": 0.05})
    assert result.episodes == 3 and result.updates == 2  # no updates at steps 4 or 12
    payload = read_checkpoint(result.training_checkpoint)
    assert payload["controller"]["epsilon"] == pytest.approx(0.6)
    assert payload["controller"]["niter"] == 2
    record("hmf-schedule", steps=12, updates=2, update_step=8, epsilon_at_third_rollout=0.6)


def test_role_selector_masks_and_partial_batch():
    cfg = role_config("rode")
    args, scheme, groups, prep, mac, learner = _make_model(cfg, make_env("toy").get_env_info(), Logger())
    batch = collect(cfg, mac, args, scheme, groups, prep)
    mac.init_hidden(2)
    actions, roles, masks, q = mac.select_actions(batch, 0, 0, bs=[1], test_mode=True)
    assert actions.shape == roles.shape == (1, 2) and masks.shape == q.shape == (1, 2, 3)
    selector = mac.action_selector
    # Greedy role intersection excludes the highest-Q action. If all Qs are
    # masked, source argmax picks 0, which is retained when environment-legal.
    values = torch.tensor([[[3.0, 2.0, 100.0], [1.0, 9.0, 20.0]]])
    available = torch.tensor([[[1, 1, 0], [1, 0, 1]]])
    role_mask = torch.tensor([[[0, 1, 1], [0, 1, 0]]])
    chosen = selector.select_action(values, available, role_mask, 0, test_mode=True)
    torch.testing.assert_close(chosen, torch.tensor([[1, 0]]))
    record("role-masks", partial_batch=True, greedy_intersection_and_fallback=True,
           empty_intersection="source argmax picks 0; random legal fallback only if 0 is unavailable",
           exploration="source exploration respects environment mask and can ignore role mask")


@pytest.mark.parametrize("bootstrap", [False, True])
def test_hmf_source_timeout_done_semantics(bootstrap):
    cfg = hmf_config("dqn")
    cfg["hmf_bootstrap_time_limits"] = bootstrap
    cfg["env_args"]["truncate"] = True
    info = make_env("toy", n_agents=4, mask_actions=False, truncate=True).get_env_info()
    args, scheme, groups, prep, mac, learner = _make_model(cfg, info, Logger())
    batch = collect(cfg, mac, args, scheme, groups, prep)
    learner.observe(batch)
    assert len(learner.records) == 8
    assert learner.records[3]["done"].item() == int(not bootstrap)
    assert learner.records[7]["done"].item() == int(not bootstrap)
