"""Training, evaluation and stateful policy inference for packaged baselines."""
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import json
import random
import uuid
import numpy as np
import torch

from .checkpoints import read_checkpoint, restore_learner, restore_replay, restore_rng, rng_state, save_checkpoints
from .components.episode_buffer import EpisodeBatch, ReplayBuffer
from .components.transforms import OneHot
from .config import dotted_overrides, load_config, merge, resource, validate, read_yaml
from .registry import get_algorithm_spec, import_symbol, load_plugins
from .envs import environment_config, validate_compatibility
from .controllers import REGISTRY as CONTROLLERS
from .learners import REGISTRY as LEARNERS
from .learners.base import UpdateContext
from .runners import REGISTRY as RUNNERS
from .utils.logging import Logger


@dataclass(frozen=True)
class TrainResult:
    algorithm: str
    output_dir: str
    checkpoint: str
    training_checkpoint: str
    metrics_path: str
    config_path: str
    env_steps: int
    episodes: int
    updates: int
    parameter_delta_l2: float
    metrics: dict


@dataclass(frozen=True)
class EvaluationResult:
    episodes: int
    returns: list
    return_mean: float
    return_std: float
    episode_lengths: list = field(default_factory=list)
    env_steps: int = 0


def make_scheme(info, config=None):
    scheme = {
        "state": {"vshape": info["state_shape"]},
        "obs": {"vshape": info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
        "avail_actions": {"vshape": (info["n_actions"],), "group": "agents", "dtype": torch.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": torch.uint8},
        "truncated": {"vshape": (1,), "dtype": torch.uint8},
    }
    if config and config.get("mac") == "rode_mac":
        scheme.update(roles={"vshape": (1,), "group": "agents", "dtype": torch.long},
                      role_avail_actions={"vshape": (info["n_actions"],), "group": "agents", "dtype": torch.int},
                      agent_Q={"vshape": (info["n_actions"],), "group": "agents"})
    if config and config.get("hmf_algorithm") == "mfq":
        scheme["mean_actions"] = {"vshape": (info["n_actions"],), "group": "agents"}
    if config and config.get("sadma_algorithm"):
        scheme["hidden_states"]={"vshape":(config["hidden_dim"],),"group":"agents"}
        scheme["policy_version"]={"vshape":(1,),"dtype":torch.long}
        if config["sadma_algorithm"]=="ppo":
            scheme["action_log_probs"]={"vshape":(1,),"group":"agents"}
            scheme["behavior_values"]={"vshape":(1,),"group":"agents"}
        if config.get("use_individual_rewards"):
            scheme["individual_rewards"]={"vshape":(1,),"group":"agents"}
    spec = get_algorithm_spec((config or {}).get("algorithm"))
    if spec and spec.scheme_factory:
        extras = import_symbol(spec.scheme_factory)(info, config)
        if not isinstance(extras, dict) or set(extras) & (set(scheme) | {"filled", "actions_onehot"}):
            raise ValueError("Plugin scheme_factory must add fields without replacing standard transition fields")
        scheme.update(extras)
    return scheme, {"agents": info["n_agents"]}, {"actions": ("actions_onehot", [OneHot(info["n_actions"])])}


def make_args(config, info):
    result = dict(config)
    result.update(n_agents=info["n_agents"], n_actions=info["n_actions"], state_shape=info["state_shape"], obs_shape=info["obs_shape"])
    return SimpleNamespace(**result)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def check_device(device):
    parsed = torch.device(device)
    if parsed.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but this PyTorch runtime has no available CUDA device")
    if parsed.type not in {"cpu", "cuda"}:
        raise ValueError("This integration currently supports cpu and cuda devices")
    return str(parsed)


def _make_model(config, info, logger):
    validate_compatibility(config, info)
    if config.get("mac") == "rode_mac" and info["n_agents"] < 2:
        raise ValueError("Source RODE squeeze operations require at least two agents")
    if config.get("learner") == "rode_hmf_learner" and info["episode_limit"] < 2:
        raise ValueError("Source RODE-HMF grouping requires at least two transitions")
    args = make_args(config, info)
    scheme, groups, preprocess = make_scheme(info, config)
    template = EpisodeBatch(scheme, groups, 1, 2, preprocess=preprocess)
    controller_class, learner_class = components(config)
    mac = controller_class(template.scheme, groups, args)
    learner = learner_class(mac, template.scheme, logger, args).to(args.device)
    return args, scheme, groups, preprocess, mac, learner


def components(config):
    load_plugins(config.get("plugins", []))
    spec = get_algorithm_spec(config.get("algorithm"))
    if spec is not None:
        return import_symbol(spec.controller), import_symbol(spec.learner)
    if config.get("sadma_algorithm"):
        from .algorithms.sadma.adapter import SADMAController,SADMALearner
        return SADMAController,SADMALearner
    if config.get("kou_family"):
        from .algorithms.kou.adapter import KouMAC, KouLearner
        return KouMAC, KouLearner
    if config.get("mac") == "hmf_native":
        from .algorithms.hmf.adapter import HMFController, HMFLearner
        return HMFController, HMFLearner
    if config.get("mac") == "rode_mac":
        try:
            from .algorithms.rghmf.controllers.rode_controller import RODEMAC
            from .algorithms.rghmf.learners import REGISTRY
        except ImportError as exc:
            raise ImportError("RODE family requires optional role dependencies: install unified-marl[role]") from exc
        return RODEMAC, REGISTRY[config["learner"]]
    return CONTROLLERS[config["mac"]], LEARNERS[config["learner"]]


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def train(algorithm=None, env=None, env_args=None, *, execution="local", device="cpu", seed=None,
          output_dir=None, config=None, overrides=None, resume=None, dataset=None, plugins=None):
    if resume:
        stored_plugins = read_checkpoint(resume).get("config", {}).get("plugins", [])
        if plugins is not None and list(plugins) != stored_plugins:
            raise ValueError("Cannot change plugins during resume")
        load_plugins(stored_plugins)
    elif plugins is not None:
        load_plugins(plugins)
        config = read_yaml(config) if isinstance(config, (str, Path)) else dict(config or {})
        config["plugins"] = list(plugins)
    if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
        raise ValueError("seed must be an integer in [0, 2**32)")
    if env_args is not None and not isinstance(env_args, dict):
        raise ValueError("env_args must be a mapping")
    if execution=="zmq":
        if dataset is not None: raise ValueError("ZMQ execution currently supports online SADMA data only")
        from .execution.trainer import train_zmq
        return train_zmq(algorithm,env,env_args,device=device,seed=seed,output_dir=output_dir,config=config,overrides=overrides,resume=resume)
    sadma_resume=resume and read_checkpoint(resume).get("config",{}).get("sadma_algorithm")
    if sadma_resume or (algorithm and algorithm.startswith(("sadma/","sadma_yi/","sadma_yi_closed/"))):
        if execution!="local" or dataset is not None: raise ValueError("SADMA stage 5 supports local online synchronous training only")
        from .trainers.sadma import train_sadma
        return train_sadma(algorithm,env,env_args,device,seed,output_dir,config,overrides,resume)
    offline_resume = resume and read_checkpoint(resume).get("config",{}).get("kou_family")
    if dataset is not None or offline_resume or (algorithm and algorithm.startswith(("kou/", "kou_old/"))):
        if execution != "local": raise ValueError("Offline execution currently supports local only")
        from .trainers.offline import train_offline
        return train_offline(algorithm, dataset=dataset, env=env, env_args=env_args, device=device, seed=seed,
                             output_dir=output_dir, config=config, overrides=overrides, resume=resume)
    if execution != "local":
        raise ValueError("Only local execution is implemented in this stage")
    payload = read_checkpoint(resume, "cpu") if resume else None
    if payload:
        if payload["kind"] != "training":
            raise ValueError("Resume requires training.pt, not a policy-only checkpoint")
        if config is not None:
            raise ValueError("Use overrides for resume; a replacement user config is ambiguous")
        cfg = merge(payload["config"], dotted_overrides(overrides))
        if algorithm and algorithm not in {cfg["algorithm"], cfg["algorithm"].split("/")[-1]}:
            raise ValueError("Cannot change algorithm during resume")
        if env is not None and env != cfg["env_config"]:
            raise ValueError("Cannot change environment during resume")
        # Callers may omit defaults and the seed injected by train(). Restore
        # those from the checkpoint; every explicitly supplied value must agree.
        if env_args is not None and merge(cfg["env_args"], env_args) != cfg["env_args"]:
            raise ValueError("Cannot change environment arguments during resume")
        allowed = {"t_max", "max_updates", "log_interval", "runner_log_interval", "learner_log_interval", "test_nepisode", "test_interval", "device", "use_cuda", "worker_timeout"}
        for key in set(cfg) | set(payload["config"]):
            if key not in allowed and cfg.get(key) != payload["config"].get(key):
                raise ValueError(f"Cannot override {key} during training-state resume")
        if seed is not None and seed != cfg["seed"]:
            raise ValueError("Resume restores the stored RNG state; seed cannot change")
        validate(cfg)
    else:
        cfg = load_config(algorithm or "pymarl/qmix", env or "toy", config, overrides)
        if env_args:
            cfg["env_args"] = merge(cfg["env_args"], env_args)
        if seed is not None:
            cfg["seed"] = int(seed)
        validate(cfg)
    cfg["device"] = check_device(device)
    cfg["use_cuda"] = cfg["device"].startswith("cuda")
    cfg["env_args"] = dict(cfg["env_args"], seed=cfg["seed"])
    if cfg.get("max_updates") is not None and (not isinstance(cfg["max_updates"], int) or cfg["max_updates"] <= 0):
        raise ValueError("max_updates must be a positive integer or null")
    directory = Path(output_dir) if output_dir else Path.cwd() / "results" / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if any((directory / name).exists() for name in ["config.json", "metrics.jsonl", "policy.pt", "training.pt"]):
        raise FileExistsError(f"Output directory already contains a run: {directory}")
    seed_all(cfg["seed"])
    old_threads = torch.get_num_threads()
    torch.set_num_threads(cfg["num_threads"])
    logger = Logger(directory / "metrics.jsonl")
    runner = None
    try:
        runner_args = SimpleNamespace(**cfg)
        runner = RUNNERS[cfg["runner"]](runner_args, logger)
        info = runner.get_env_info()
        args, scheme, groups, preprocess, mac, learner = _make_model(cfg, info, logger)
        runner.args = args
        runner.setup(scheme, groups, preprocess, mac)
        spec = get_algorithm_spec(cfg["algorithm"])
        fresh = spec.training_mode == "on_policy" if spec else cfg["learner"] in {"coma_learner", "hmf_native"}
        buffer = None if fresh else ReplayBuffer(scheme, groups, cfg["buffer_size"], info["episode_limit"] + 1,
                                                 preprocess=preprocess, device="cpu" if cfg["buffer_cpu_only"] else cfg["device"])
        progress = {"env_steps": 0, "episodes": 0, "updates": 0}
        if payload:
            if info != payload["env_info"]:
                raise ValueError("Environment dimensions differ from training checkpoint")
            if cfg.get("mac") == "rode_mac" and len(payload["controller"]["roles"]) != cfg["n_roles"]:
                raise ValueError("Resume after source dynamic role expansion is not validated; optimizer parameter groups change")
            restore_learner(learner, payload["learner"])
            restore_replay(buffer, payload["replay"])
            progress.update(payload["progress"])
            runner.t_env = progress["env_steps"]
            restore_rng(payload["rng"])
        initial = [p.detach().clone() for p in mac.parameters()]
        _write_json(directory / "config.json", cfg)
        last_metrics = {}
        while runner.t_env < cfg["t_max"] and (cfg.get("max_updates") is None or progress["updates"] < cfg["max_updates"]):
            if hasattr(mac, "prepare_rollout"):
                mac.prepare_rollout(progress["episodes"])
            batch = runner.run(test_mode=False)
            progress["episodes"] += runner.batch_size
            progress["env_steps"] = runner.t_env
            if fresh:
                sample = batch
            else:
                buffer.insert_episode_batch(batch)
                if not buffer.can_sample(cfg["batch_size"]):
                    continue
                sample = buffer.sample(cfg["batch_size"])
            sample = sample[:, :int(sample.max_t_filled().item())]
            sample.to(cfg["device"])
            repeats = cfg["hmf_num_batch"] if cfg["learner"] == "hmf_native" else 1
            for repetition in range(repeats):
                if cfg.get("max_updates") is not None and progress["updates"] >= cfg["max_updates"]:
                    break
                current_metrics = learner.update(sample if repetition == 0 else None,
                    UpdateContext(runner.t_env, progress["episodes"], progress["updates"]))
                if current_metrics.get("warmup"):
                    break
                last_metrics = current_metrics
                progress["updates"] += 1
                logger.record(last_metrics, runner.t_env)
        if not last_metrics:
            raise RuntimeError("Training budget ended without a parameter update; increase t_max/max_updates or reduce batch_size")
        delta = sum(float((a - b.detach()).square().sum()) for a, b in zip(initial, mac.parameters())) ** 0.5
        if not np.isfinite(delta):
            raise FloatingPointError("Non-finite parameter change")
        policy_path, training_path = save_checkpoints(directory, cfg, info, learner, buffer, progress)
        result = TrainResult(cfg["algorithm"], str(directory), policy_path, training_path, str(directory / "metrics.jsonl"),
                             str(directory / "config.json"), progress["env_steps"], progress["episodes"], progress["updates"], delta, last_metrics)
        _write_json(directory / "result.json", asdict(result))
    finally:
        try:
            if runner is not None:
                runner.close_env()
        finally:
            torch.set_num_threads(old_threads)
    return result


class Policy:
    def __init__(self, payload, device):
        self.config, self.env_info = payload["config"], payload["env_info"]
        load_plugins(self.config.get("plugins", []))
        cfg = dict(self.config, device=check_device(device))
        args = make_args(cfg, self.env_info)
        self.scheme, self.groups, self.preprocess = make_scheme(self.env_info, cfg)
        template = EpisodeBatch(self.scheme, self.groups, 1, 2, preprocess=self.preprocess, device=device)
        self.mac = components(cfg)[0](template.scheme, self.groups, args).to(device)
        from .checkpoints import restore_controller
        restore_controller(self.mac, payload.get("controller", {"agent": payload["agent"]}))
        self.mac.agent.eval()
        self.device = device
        self.reset()

    def reset(self):
        self.mac.init_hidden(1)
        self.last_actions = torch.zeros((1, 1, self.env_info["n_agents"], self.env_info["n_actions"]), device=self.device)
        self.t = 0
        if self.config.get("mac") == "rode_mac":
            self.history = EpisodeBatch(self.scheme, self.groups, 1, self.env_info["episode_limit"] + 1,
                                        preprocess=self.preprocess, device=self.device)

    @torch.no_grad()
    def act(self, obs, avail_actions=None):
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape != (self.env_info["n_agents"], self.env_info["obs_shape"]) or not np.isfinite(obs).all():
            raise ValueError("Observations have incompatible shape or non-finite values")
        available = np.ones((self.env_info["n_agents"], self.env_info["n_actions"]), dtype=np.int64) if avail_actions is None else np.asarray(avail_actions)
        if available.shape != (self.env_info["n_agents"], self.env_info["n_actions"]) or not np.isin(available, [0, 1]).all() or np.any(available.sum(-1) == 0):
            raise ValueError("Invalid available-action mask")
        if self.config.get("mac") == "rode_mac":
            if self.t >= self.history.max_seq_length:
                raise ValueError("Policy history exhausted; call reset at the episode boundary")
            self.history.update({"obs": obs[None], "avail_actions": available[None]}, ts=self.t)
            actions, roles, role_mask, q = self.mac.select_actions(self.history, self.t, 0, test_mode=True)
            self.history.update({"actions": actions}, ts=self.t, mark_filled=False)
            self.t += 1
            return actions[0].cpu().numpy()
        batch = EpisodeBatch(self.scheme, self.groups, 1, 2, preprocess=self.preprocess, device=self.device)
        batch["actions_onehot"][:, :1] = self.last_actions
        batch.update({"obs": obs[None], "avail_actions": available[None]}, ts=1)
        actions = self.mac.select_actions(batch, t_ep=1, t_env=0, test_mode=True)
        self.last_actions = torch.nn.functional.one_hot(actions, self.env_info["n_actions"]).float().unsqueeze(1)
        return actions[0].cpu().numpy()


def load_policy(checkpoint, device="cpu"):
    return Policy(read_checkpoint(checkpoint, "cpu"), device)


def evaluate(checkpoint, *, env=None, env_args=None, episodes=20, device="cpu", seed=None):
    if not isinstance(episodes, int) or isinstance(episodes, bool) or episodes < 1:
        raise ValueError("episodes must be a positive integer")
    payload = read_checkpoint(checkpoint, "cpu")
    load_plugins(payload["config"].get("plugins", []))
    if env_args is not None and not isinstance(env_args, dict):
        raise ValueError("env_args must be a mapping")
    if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
        raise ValueError("seed must be an integer in [0, 2**32)")
    cfg = dict(payload["config"], device=check_device(device), batch_size_run=1, runner="episode", test_nepisode=episodes)
    if env is not None:
        env = "sc2" if env == "smac" else env
        preset = environment_config(env)
        if env != cfg["env"]:
            cfg["env_args"] = preset.get("env_args", {})
        cfg["env"] = preset.get("env", env)
    cfg["env_args"] = merge(cfg["env_args"], env_args)
    if seed is not None:
        cfg["env_args"]["seed"] = seed
    saved_rng = rng_state()
    seed_all(cfg["seed"] if seed is None else seed)
    runner = None
    try:
        runner = RUNNERS["episode"](SimpleNamespace(**cfg), Logger())
        info = runner.get_env_info()
        if info != payload["env_info"]:
            raise ValueError("Evaluation environment metadata differs from checkpoint")
        policy = Policy(payload, device)
        runner.setup(policy.scheme, policy.groups, policy.preprocess, policy.mac)
        returns, lengths = [], []
        for _ in range(episodes):
            runner.run(test_mode=True)
            returns.extend(runner.last_returns)
            lengths.extend(int(length) for length in runner.last_lengths)
        return EvaluationResult(episodes, returns, float(np.mean(returns)), float(np.std(returns)),
                                lengths, sum(lengths))
    finally:
        try:
            if runner is not None:
                runner.close_env()
        finally:
            restore_rng(saved_rng)
