"""Standalone bounded-Gaussian PPO application for the SADMA PowerGrid.

The public SADMA sources provide a recurrent Normal actor and PPO update, but
their continuous branch sends unbounded Gaussian samples to a bounded Box and
does not account for a bound transform in its likelihood. This engine is a
separate bounded PPO adaptation: it uses a tanh-affine action map and the
matching physical-action log probability. It does not claim algorithm parity
with the source implementation.

The environment and pandapower are imported only when train/evaluate constructs
the simulator. Importing this module or loading an inference checkpoint does
not require pandapower or the external climate/building assets.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

__all__ = ["GridPolicy", "train", "evaluate", "load_policy"]

_CHECKPOINT_FORMAT = "unified_marl.powergrid.bounded_ppo.v1"
_ENV_ARG_NAMES = {
    "data_path",
    "climate_zone",
    "buildings_states_actions_file",
    "hourly_timesteps",
    "houses_per_node",
    "cluster_adjacent_bus_num",
    "save_memory",
    "building_ids",
    "building_sampling_weights",
    "nclusters",
    "randomseed",
    "max_num_houses",
    "percent_rl",
    "episode_limit",
}


def _plain(value: Any) -> Any:
    """Convert config/metrics values to portable Python values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, torch.device):
        return str(value)
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _effective_config(config: dict[str, Any] | None) -> dict[str, Any]:
    cfg = copy.deepcopy(config or {})
    if not isinstance(cfg, dict):
        raise TypeError("PowerGrid config must be a dictionary")
    nested_training = cfg.get("training", {})
    if nested_training is not None:
        if not isinstance(nested_training, dict):
            raise TypeError("config.training must be a dictionary")
        for key, value in nested_training.items():
            cfg.setdefault(key, copy.deepcopy(value))
    return cfg


def _device_from_config(value: Any) -> torch.device:
    device = torch.device(value or "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PowerGrid requested CUDA, but CUDA is unavailable")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported PowerGrid device: {device}")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sample_normal(distribution: Normal, generator: torch.Generator) -> torch.Tensor:
    noise = torch.randn(
        distribution.mean.shape,
        dtype=distribution.mean.dtype,
        device=distribution.mean.device,
        generator=generator,
    )
    return distribution.mean + distribution.stddev * noise


def _capture_rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["torch_cuda"]])


def _normalise_env_args(config: dict[str, Any], seed: int) -> dict[str, Any]:
    env_args = copy.deepcopy(config.get("env_args") or {})
    if not isinstance(env_args, dict):
        raise TypeError("config.env_args must be a dictionary")
    if "building_states_actions_file" in env_args:
        if "buildings_states_actions_file" in env_args:
            raise ValueError(
                "Use only env_args.buildings_states_actions_file; the singular "
                "alias conflicts with it"
            )
        env_args["buildings_states_actions_file"] = env_args.pop(
            "building_states_actions_file"
        )
    for unused in ("net_path", "agent_path"):
        if env_args.get(unused):
            raise ValueError(
                f"env_args.{unused} is not consumed by the public SADMA PowerGrid "
                "source; this adapter uses pandapower case33bw and constructs "
                "buildings from the climate/building assets"
            )
        env_args.pop(unused, None)
    unknown = sorted(set(env_args) - _ENV_ARG_NAMES)
    if unknown:
        raise ValueError("Unknown PowerGrid env_args: " + ", ".join(unknown))
    if "data_path" not in env_args or "buildings_states_actions_file" not in env_args:
        raise ValueError(
            "PowerGrid requires env_args.data_path (climate CSV/JSON directory) "
            "and env_args.buildings_states_actions_file (state/action JSON path)"
        )

    env_args["data_path"] = str(Path(env_args["data_path"]).expanduser().resolve())
    env_args["buildings_states_actions_file"] = str(
        Path(env_args["buildings_states_actions_file"]).expanduser().resolve()
    )
    if env_args.get("building_ids") is not None:
        if not isinstance(env_args["building_ids"], (list, tuple)):
            raise TypeError("env_args.building_ids must be a list of building IDs")
        env_args["building_ids"] = [str(uid) for uid in env_args["building_ids"]]
    if env_args.get("building_sampling_weights") is not None:
        weights = env_args["building_sampling_weights"]
        if not isinstance(weights, dict) or not weights:
            raise TypeError("env_args.building_sampling_weights must be a non-empty mapping")
        normalized_weights = {}
        for uid, weight in weights.items():
            numeric_weight = float(weight)
            if not math.isfinite(numeric_weight) or numeric_weight <= 0:
                raise ValueError("building_sampling_weights values must be positive and finite")
            normalized_weights[str(uid)] = numeric_weight
        env_args["building_sampling_weights"] = normalized_weights
    env_args.setdefault("randomseed", int(seed))
    if float(env_args.get("percent_rl", 1.0)) != 1.0:
        raise ValueError(
            "PowerGrid currently requires percent_rl=1 because the source returns "
            "observations/actions for every building"
        )
    if int(env_args.get("episode_limit", 92)) <= 0:
        raise ValueError("env_args.episode_limit must be positive")
    _validate_powergrid_assets(env_args)
    return env_args


def _validate_powergrid_assets(env_args: dict[str, Any]) -> None:
    data_path = Path(env_args["data_path"])
    action_path = Path(env_args["buildings_states_actions_file"])
    required = [
        data_path / "weather_data.csv",
        data_path / "solar_generation_1kW.csv",
        data_path / "building_attributes.json",
        action_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "PowerGrid cannot start because required external assets are missing: "
            + ", ".join(missing)
        )
    try:
        with action_path.open(encoding="utf-8") as stream:
            state_actions = json.load(stream)
        with (data_path / "building_attributes.json").open(
            encoding="utf-8"
        ) as stream:
            attributes = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read PowerGrid JSON assets: {exc}") from exc

    if not isinstance(state_actions, dict):
        raise ValueError("The PowerGrid state/action JSON must be a mapping of building IDs")
    selected_ids = env_args.get("building_ids") or list(state_actions)
    if not selected_ids:
        raise ValueError("The PowerGrid state/action JSON has no building IDs")
    if len(selected_ids) < 1:
        raise ValueError("PowerGrid requires at least one building ID")
    missing_specs = [uid for uid in selected_ids if uid not in state_actions]
    if missing_specs:
        raise ValueError(
            "PowerGrid building IDs are absent from the state/action JSON: "
            + ", ".join(map(str, missing_specs[:8]))
        )
    configured_weights = env_args.get("building_sampling_weights")
    if configured_weights is not None:
        unknown_weight_ids = [uid for uid in configured_weights if uid not in selected_ids]
        if unknown_weight_ids:
            raise ValueError(
                "PowerGrid sampling weights reference IDs outside the selected pool: "
                + ", ".join(map(str, unknown_weight_ids[:8]))
            )
    missing_ids = [uid for uid in selected_ids if uid not in attributes]
    if missing_ids:
        raise ValueError(
            "PowerGrid IDs in the state/action JSON are missing from "
            "building_attributes.json: "
            + ", ".join(map(str, missing_ids[:8]))
        )
    try:
        building_types = sorted(
            {attributes[uid]["Building_Type"] for uid in selected_ids}
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Each selected PowerGrid building needs a Building_Type attribute"
        ) from exc
    action_names = {
        "cooling_storage",
        "dhw_storage",
        "pv_curtail",
        "pv_phi",
        "electrical_storage",
    }
    for uid in selected_ids:
        spec = state_actions[uid]
        actions = spec.get("actions") if isinstance(spec, dict) else None
        if not isinstance(actions, dict) or not action_names.issubset(actions):
            raise ValueError(
                f"PowerGrid state/action entry {uid!r} must define all five action flags: "
                + ", ".join(sorted(action_names))
            )
    missing_csv = [
        str(data_path / f"Building_{building_type}.csv")
        for building_type in building_types
        if not (data_path / f"Building_{building_type}.csv").is_file()
    ]
    if missing_csv:
        raise FileNotFoundError(
            "PowerGrid cannot start because building simulation files are missing: "
            + ", ".join(missing_csv)
        )


def _load_env_class():
    """Import the optional simulator lazily and give a focused dependency error."""
    try:
        from unified_marl.envs.powergrid_native.smart_grid import GridEnv
    except ModuleNotFoundError as exc:
        optional = {"pandapower", "gym", "pandas", "sklearn", "scipy"}
        if exc.name in optional:
            raise ImportError(
                "PowerGrid runtime dependencies are missing. Install pandapower, "
                "gym 0.26, pandas, and scikit-learn (plus pandapower's dependencies) "
                "in the active environment."
            ) from exc
        raise
    return GridEnv


def _seed_action_spaces(env: Any, seed: int) -> None:
    for index, agent_id in enumerate(env.agents):
        action_space = env.action_spaces[agent_id]
        if hasattr(action_space, "seed"):
            action_space.seed(int(seed) + index + 1)


def _capture_env_state(env: Any) -> dict[str, Any]:
    spaces = {}
    for agent_id in env.agents:
        generator = getattr(env.action_spaces[agent_id], "np_random", None)
        if generator is not None and hasattr(generator, "bit_generator"):
            spaces[str(agent_id)] = copy.deepcopy(generator.bit_generator.state)
    return {"exipode": int(getattr(env, "exipode", 0)), "action_space_rng": spaces}


def _restore_env_state(env: Any, state: dict[str, Any]) -> None:
    if not state:
        return
    if hasattr(env, "exipode"):
        env.exipode = int(state.get("exipode", 0))
    states = state.get("action_space_rng", {})
    for agent_id in env.agents:
        generator = getattr(env.action_spaces[agent_id], "np_random", None)
        saved = states.get(str(agent_id))
        if saved is not None and generator is not None and hasattr(generator, "bit_generator"):
            generator.bit_generator.state = copy.deepcopy(saved)


def _build_env_contract(env: Any) -> dict[str, Any]:
    agent_ids = list(env.agents)
    if not agent_ids:
        raise ValueError("PowerGrid generated no agents from the supplied building assets")
    if len(getattr(env, "rl_agents", agent_ids)) != len(agent_ids):
        raise ValueError("PowerGrid percent_rl must include every generated building")
    obs_dim = int(env.get_obs_size())
    state_dim = int(env.get_state_size())
    if obs_dim <= 0 or state_dim <= 0:
        raise ValueError("PowerGrid observation and state dimensions must be positive")
    lows, highs = [], []
    action_names_by_agent = []
    action_shape = None
    for agent_id in agent_ids:
        space = env.action_spaces[agent_id]
        low = np.asarray(space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(space.high, dtype=np.float32).reshape(-1)
        if action_shape is None:
            action_shape = low.shape
        if low.shape != action_shape or high.shape != action_shape:
            raise ValueError("PowerGrid agents must share one continuous action shape")
        if not np.isfinite(low).all() or not np.isfinite(high).all():
            raise ValueError("PowerGrid requires finite continuous action bounds")
        if np.any(high <= low):
            raise ValueError("Every PowerGrid action upper bound must exceed its lower bound")
        lows.append(low)
        highs.append(high)
        building = env.buildings[agent_id]
        action_names = list(getattr(building, "action_names", ()))
        if len(action_names) != low.size:
            raise ValueError(
                f"PowerGrid action names for {agent_id!r} do not match its action dimension"
            )
        action_names_by_agent.append(action_names)
    low_by_agent = np.stack(lows)
    high_by_agent = np.stack(highs)
    action_dim = int(low_by_agent.shape[1])
    if action_dim <= 0:
        raise ValueError("PowerGrid action dimensions must be positive")
    if any(names != action_names_by_agent[0] for names in action_names_by_agent[1:]):
        raise ValueError(
            "PowerGrid buildings must use the same action names in each shared-policy dimension"
        )
    action_spec = {
        "type": "continuous",
        "shape": [action_dim],
        "low": low_by_agent.tolist(),
        "high": high_by_agent.tolist(),
        "names": action_names_by_agent[0],
        "names_by_agent": action_names_by_agent,
        "transform": "tanh_affine",
    }
    return {
        "state_shape": state_dim,
        "obs_shape": obs_dim,
        "n_actions": action_dim,
        "action_shape": [action_dim],
        "action_type": "continuous",
        "action_spec": action_spec,
        "action_low": low_by_agent.tolist(),
        "action_high": high_by_agent.tolist(),
        "agents_name": [str(agent_id) for agent_id in agent_ids],
        "n_agents": len(agent_ids),
        "episode_limit": int(env.episode_limit),
        "episode_limit_semantics": getattr(env, "episode_limit_semantics", "legacy_reset_included"),
    }


def _make_env(config: dict[str, Any], seed: int):
    env_args = _normalise_env_args(config, seed)
    if bool(config.get("asynchronous_env", False)) or bool(config.get("async_train", False)):
        raise ValueError("PowerGrid bounded_ppo supports local synchronous execution only")
    if int(config.get("env_batch_size", 1)) != 1:
        raise ValueError("PowerGrid bounded_ppo currently supports one local environment")
    env_class = _load_env_class()
    try:
        env = env_class(**env_args)
    except ModuleNotFoundError as exc:
        optional = {"pandapower", "gym", "pandas", "sklearn", "scipy"}
        if exc.name in optional:
            raise ImportError(
                "PowerGrid runtime dependencies are missing. Install pandapower, "
                "gym 0.26, pandas, and scikit-learn (plus pandapower's dependencies)."
            ) from exc
        raise
    try:
        if len(getattr(env, "rl_agents", env.agents)) != len(env.agents):
            raise ValueError("PowerGrid requires percent_rl=1")
        _seed_action_spaces(env, seed)
        contract = _build_env_contract(env)
    except Exception:
        env.close()
        raise
    return env, contract


def _actor_architecture(config: dict[str, Any], env_info: dict[str, Any]) -> dict[str, Any]:
    agent_inputs = config.get("agent_inputs", ["obs", "agent_id", "last_action"])
    if not isinstance(agent_inputs, (list, tuple)):
        raise TypeError("agent_inputs must be a list of input names")
    agent_inputs = set(agent_inputs)
    if "obs" not in agent_inputs:
        raise ValueError("The PowerGrid policy requires obs in agent_inputs")
    agent_type = str(config.get("agent_type", "rnn")).lower()
    if agent_type not in {"rnn", "mlp"}:
        raise ValueError("agent_type must be 'rnn' or 'mlp'")
    critic_inputs = config.get("critic_inputs", ["central_obs"])
    if not isinstance(critic_inputs, (list, tuple)) or "central_obs" not in critic_inputs:
        raise ValueError("PowerGrid bounded_ppo requires critic_inputs including central_obs")
    unsupported_critic_inputs = set(critic_inputs) - {"central_obs", "state"}
    if unsupported_critic_inputs:
        raise ValueError(
            "Unsupported central critic input(s): " + ", ".join(sorted(unsupported_critic_inputs))
        )
    critic_type = str(config.get("critic_type", "rnn")).lower()
    if critic_type not in {"rnn", "mlp"}:
        raise ValueError("critic_type must be 'rnn' or 'mlp'")
    return {
        "obs_dim": int(env_info["obs_shape"]),
        "state_dim": int(env_info["state_shape"]),
        "n_agents": int(env_info["n_agents"]),
        "action_dim": int(env_info["n_actions"]),
        "hidden_dim": int(config.get("hidden_dim", 64)),
        "actor_type": agent_type,
        "critic_type": critic_type,
        "critic_use_state": "state" in critic_inputs,
        "use_agent_id": "agent_id" in agent_inputs,
        "use_last_action": "last_action" in agent_inputs,
        "use_layer_norm": bool(config.get("use_layer_norm", True)),
        "use_orthogonal": bool(config.get("use_orthogonal", True)),
        "gain": float(config.get("gain", 0.01)),
    }


def _init_linear(layer: nn.Linear, gain: float = 1.0) -> None:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)


class _GaussianActor(nn.Module):
    def __init__(self, architecture: dict[str, Any]):
        super().__init__()
        self.architecture = dict(architecture)
        self.hidden_dim = int(architecture["hidden_dim"])
        self.n_agents = int(architecture["n_agents"])
        self.action_dim = int(architecture["action_dim"])
        input_dim = int(architecture["obs_dim"])
        if architecture["use_last_action"]:
            input_dim += self.action_dim
        if architecture["use_agent_id"]:
            input_dim += self.n_agents
        self.fc1 = nn.Linear(input_dim, self.hidden_dim)
        self.rnn = nn.GRUCell(self.hidden_dim, self.hidden_dim) if architecture["actor_type"] == "rnn" else None
        self.norm = nn.LayerNorm(self.hidden_dim) if architecture["use_layer_norm"] else None
        self.mean = nn.Linear(self.hidden_dim, self.action_dim)
        self.log_std = nn.Parameter(torch.zeros(self.action_dim))
        if architecture["use_orthogonal"]:
            _init_linear(self.fc1)
            _init_linear(self.mean, float(architecture["gain"]))

    def step(
        self,
        obs: torch.Tensor,
        last_action: torch.Tensor,
        agent_ids: torch.Tensor,
        hidden: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        pieces = [obs]
        if self.architecture["use_last_action"]:
            pieces.append(last_action)
        if self.architecture["use_agent_id"]:
            pieces.append(agent_ids)
        encoded = F.relu(self.fc1(torch.cat(pieces, dim=-1)))
        if self.rnn is not None:
            if hidden is None:
                hidden = torch.zeros(
                    obs.shape[0], self.hidden_dim, device=obs.device, dtype=obs.dtype
                )
            features = self.rnn(encoded, hidden)
            next_hidden = features
        else:
            features = encoded
            next_hidden = None
        if self.norm is not None:
            features = self.norm(features)
        return self.mean(features), next_hidden

    def distribution(self, mean: torch.Tensor) -> Normal:
        log_std = self.log_std.clamp(min=-20.0, max=2.0).expand_as(mean)
        return Normal(mean, torch.exp(log_std))


class _CentralCritic(nn.Module):
    def __init__(self, architecture: dict[str, Any]):
        super().__init__()
        self.architecture = dict(architecture)
        self.hidden_dim = int(architecture["hidden_dim"])
        self.use_state = bool(architecture["critic_use_state"])
        input_dim = int(architecture["obs_dim"]) * int(architecture["n_agents"])
        if self.use_state:
            input_dim += int(architecture["state_dim"])
        self.fc1 = nn.Linear(input_dim, self.hidden_dim)
        recurrent = architecture["critic_type"] == "rnn"
        self.rnn = nn.GRUCell(self.hidden_dim, self.hidden_dim) if recurrent else None
        self.fc2 = nn.Linear(self.hidden_dim, 1) if recurrent else nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = None if recurrent else nn.Linear(self.hidden_dim, 1)
        self.norm = nn.LayerNorm(self.hidden_dim) if architecture["use_layer_norm"] else None
        if architecture["use_orthogonal"]:
            _init_linear(self.fc1)
            _init_linear(self.fc2, float(architecture["gain"]) if recurrent else 1.0)
            if self.fc3 is not None:
                _init_linear(self.fc3, float(architecture["gain"]))

    def step(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        hidden: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        pieces = [obs.reshape(1, -1)]
        if self.use_state:
            pieces.append(state.reshape(1, -1))
        encoded = F.relu(self.fc1(torch.cat(pieces, dim=-1)))
        if self.rnn is not None:
            if hidden is None:
                hidden = torch.zeros(1, self.hidden_dim, device=obs.device, dtype=obs.dtype)
            features = self.rnn(encoded, hidden)
            next_hidden = features
            if self.norm is not None:
                features = self.norm(features)
            value = self.fc2(features)
        else:
            features = F.relu(encoded)
            if self.norm is not None:
                features = self.norm(features)
            features = F.relu(self.fc2(features))
            value = self.fc3(features)
            next_hidden = None
        return value.reshape(()), next_hidden


class _ValueNorm(nn.Module):
    """Small source-style exponential value normalizer with resumable buffers."""

    def __init__(self, beta: float = 0.99999, epsilon: float = 1e-5):
        super().__init__()
        self.beta = float(beta)
        self.epsilon = float(epsilon)
        self.register_buffer("running_mean", torch.zeros(()))
        self.register_buffer("running_mean_sq", torch.zeros(()))
        self.register_buffer("debiasing_term", torch.zeros(()))

    def moments(self) -> tuple[torch.Tensor, torch.Tensor]:
        debias = self.debiasing_term.clamp(min=self.epsilon)
        mean = self.running_mean / debias
        mean_sq = self.running_mean_sq / debias
        variance = (mean_sq - mean.square()).clamp(min=1e-2)
        return mean, variance

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        values = values.detach().reshape(-1)
        if values.numel() == 0:
            return
        mean = values.mean()
        mean_sq = values.square().mean()
        self.running_mean.mul_(self.beta).add_(mean * (1.0 - self.beta))
        self.running_mean_sq.mul_(self.beta).add_(mean_sq * (1.0 - self.beta))
        self.debiasing_term.mul_(self.beta).add_(1.0 - self.beta)

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        mean, variance = self.moments()
        return (values - mean) / torch.sqrt(variance)

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        mean, variance = self.moments()
        return values * torch.sqrt(variance) + mean


def _agent_one_hot(n_agents: int, device: torch.device) -> torch.Tensor:
    return torch.eye(n_agents, dtype=torch.float32, device=device)


def _transform_log_prob(
    distribution: Normal,
    latent_action: torch.Tensor,
    half_range: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return physical-space log probability and latent Gaussian entropy."""
    log_sech_squared = 2.0 * (
        math.log(2.0) - latent_action - F.softplus(-2.0 * latent_action)
    )
    log_abs_det = torch.log(half_range) + log_sech_squared
    physical_log_prob = (distribution.log_prob(latent_action) - log_abs_det).sum(
        dim=-1, keepdim=True
    )
    latent_entropy = distribution.entropy().sum(dim=-1, keepdim=True)
    return physical_log_prob, latent_entropy


def _as_obs(value: Any, n_agents: int, obs_dim: int) -> np.ndarray:
    obs = np.asarray(value, dtype=np.float32)
    if obs.ndim == 1 and n_agents == 1:
        obs = obs.reshape(1, -1)
    if obs.shape != (n_agents, obs_dim):
        raise ValueError(
            f"PowerGrid observation shape must be {(n_agents, obs_dim)}, got {obs.shape}"
        )
    if not np.isfinite(obs).all():
        raise ValueError("PowerGrid observation contains non-finite values")
    return obs


def _as_state(value: Any, state_dim: int) -> np.ndarray:
    state = np.asarray(value, dtype=np.float32).reshape(-1)
    if state.shape != (state_dim,):
        raise ValueError(f"PowerGrid state shape must be {(state_dim,)}, got {state.shape}")
    if not np.isfinite(state).all():
        raise ValueError("PowerGrid state contains non-finite values")
    return state


def _critic_sequence(
    critic: _CentralCritic,
    observations: list[np.ndarray] | np.ndarray,
    states: list[np.ndarray] | np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    obs_tensor = torch.as_tensor(np.asarray(observations), dtype=torch.float32, device=device)
    state_tensor = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=device)
    hidden = None
    values = []
    for timestep in range(obs_tensor.shape[0]):
        value, hidden = critic.step(obs_tensor[timestep], state_tensor[timestep], hidden)
        values.append(value)
    return torch.stack(values)


def _architecture_from_checkpoint(payload: dict[str, Any]) -> dict[str, Any]:
    architecture = payload.get("architecture")
    if not isinstance(architecture, dict):
        raise ValueError("PowerGrid checkpoint has no policy architecture metadata")
    return architecture


class GridPolicy:
    """Recurrent inference policy with physical-action bounds and reset state."""

    def __init__(self, payload: dict[str, Any], device: str | torch.device = "cpu"):
        self.device = _device_from_config(device)
        self.config = copy.deepcopy(payload.get("config", {}))
        self.env_info = copy.deepcopy(payload.get("env_info", {}))
        self.architecture = _architecture_from_checkpoint(payload)
        caller_rng_state = _capture_rng_state()
        try:
            self.actor = _GaussianActor(self.architecture).to(self.device)
        finally:
            _restore_rng_state(caller_rng_state)
        self.actor.load_state_dict(payload["policy_state"])
        self.actor.eval()
        self.n_agents = int(self.env_info.get("n_agents", self.architecture["n_agents"]))
        self.agent_ids = list(self.env_info.get("agents_name", range(self.n_agents)))
        self.action_dim = int(self.env_info.get("n_actions", self.architecture["action_dim"]))
        action_spec = self.env_info.get("action_spec", {})
        self.control_names = list(action_spec.get("names", []))
        self.action_low = np.asarray(action_spec.get("low"), dtype=np.float32)
        self.action_high = np.asarray(action_spec.get("high"), dtype=np.float32)
        expected_shape = (self.n_agents, self.action_dim)
        if self.action_low.shape != expected_shape or self.action_high.shape != expected_shape:
            raise ValueError(
                f"Checkpoint action bounds must have shape {expected_shape}; got "
                f"{self.action_low.shape} and {self.action_high.shape}"
            )
        if not np.isfinite(self.action_low).all() or not np.isfinite(self.action_high).all():
            raise ValueError("Checkpoint action bounds must be finite")
        if np.any(self.action_high <= self.action_low):
            raise ValueError("Checkpoint action upper bounds must exceed lower bounds")
        self._obs_dim = int(self.architecture["obs_dim"])
        self._agent_ids_tensor = _agent_one_hot(self.n_agents, self.device)
        self._low_tensor = torch.as_tensor(self.action_low, device=self.device)
        self._high_tensor = torch.as_tensor(self.action_high, device=self.device)
        self._half_range = (self._high_tensor - self._low_tensor) / 2.0
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(int(self.config.get("seed", 0)))
        self.reset()

    def reset(self) -> None:
        self._hidden = None
        if self.architecture["actor_type"] == "rnn":
            self._hidden = torch.zeros(
                self.n_agents,
                int(self.architecture["hidden_dim"]),
                dtype=torch.float32,
                device=self.device,
            )
        self._last_action = torch.zeros(
            self.n_agents, self.action_dim, dtype=torch.float32, device=self.device
        )

    def act(self, obs: Any, deterministic: bool = True) -> np.ndarray:
        observations = _as_obs(obs, self.n_agents, self._obs_dim)
        obs_tensor = torch.as_tensor(observations, device=self.device)
        with torch.inference_mode():
            mean, self._hidden = self.actor.step(
                obs_tensor,
                self._last_action,
                self._agent_ids_tensor,
                self._hidden,
            )
            distribution = self.actor.distribution(mean)
            latent_action = mean if deterministic else _sample_normal(distribution, self._generator)
            normalized_action = torch.tanh(latent_action)
            physical_action = self._low_tensor + (normalized_action + 1.0) * self._half_range
            self._last_action = normalized_action
        action = physical_action.cpu().numpy().astype(np.float32, copy=False)
        return np.clip(action, self.action_low, self.action_high).astype(np.float32, copy=False)


def _load_checkpoint(path: str | os.PathLike[str], map_location: torch.device | str = "cpu") -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser()
    if checkpoint_path.is_dir():
        candidates = ("training.pt", "policy.pt", "checkpoint.pt")
        checkpoint_path = next(
            (checkpoint_path / name for name in candidates if (checkpoint_path / name).is_file()),
            checkpoint_path / candidates[0],
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"PowerGrid checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != _CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported PowerGrid checkpoint format: {checkpoint_path}")
    return payload


def load_policy(checkpoint: str | os.PathLike[str], device: str = "cpu") -> GridPolicy:
    """Load an inference policy without importing pandapower or opening an env."""
    target_device = _device_from_config(device)
    payload = _load_checkpoint(checkpoint, map_location=target_device)
    return GridPolicy(payload, device=target_device)


def _checkpoint_contract_matches(
    expected: dict[str, Any], observed: dict[str, Any]
) -> None:
    if expected.get("episode_limit_semantics", "legacy_reset_included") != observed.get("episode_limit_semantics", "legacy_reset_included"):
        raise ValueError(
            "PowerGrid checkpoint uses different episode-limit semantics. "
            "0.8.2 counts policy control transitions separately from reset warmup; start a new training run."
        )
    for key in ("n_agents", "obs_shape", "state_shape", "n_actions"):
        if int(expected[key]) != int(observed[key]):
            raise ValueError(
                f"PowerGrid checkpoint {key}={expected[key]} does not match environment "
                f"{observed[key]}"
            )
    for key in ("action_low", "action_high"):
        if not np.allclose(
            np.asarray(expected[key], dtype=np.float32),
            np.asarray(observed[key], dtype=np.float32),
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError(f"PowerGrid checkpoint {key} does not match environment bounds")
    for key in ("agents_name",):
        if key in expected and expected.get(key) != observed.get(key):
            raise ValueError(f"PowerGrid checkpoint {key} does not match environment agent order")
    expected_names = expected.get("action_spec", {}).get("names")
    observed_names = observed.get("action_spec", {}).get("names")
    if expected_names is not None and expected_names != observed_names:
        raise ValueError("PowerGrid checkpoint action names do not match the environment")


def _env_snapshot(env: Any) -> dict[str, Any]:
    return {
        "exipode": int(getattr(env, "exipode", 0)),
        "action_space_rng": _capture_env_state(env).get("action_space_rng", {}),
    }


def _atomic_save(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _policy_payload(
    config: dict[str, Any],
    env_info: dict[str, Any],
    architecture: dict[str, Any],
    actor: _GaussianActor,
) -> dict[str, Any]:
    return {
        "format": _CHECKPOINT_FORMAT,
        "kind": "policy",
        "algorithm": "bounded_ppo",
        "config": _plain(config),
        "env_info": _plain(env_info),
        "architecture": _plain(architecture),
        "policy_state": actor.state_dict(),
    }


def _save_checkpoint(
    destination: Path,
    config: dict[str, Any],
    env_info: dict[str, Any],
    architecture: dict[str, Any],
    actor: _GaussianActor,
    critic: _CentralCritic,
    optimizer: torch.optim.Optimizer,
    value_norm: _ValueNorm | None,
    update: int,
    policy_version: int,
    episodes: int,
    env_steps: int,
    seed: int,
    env: Any,
) -> None:
    training_state = {
        "optimizer_state": optimizer.state_dict(),
        "value_norm_state": value_norm.state_dict() if value_norm is not None else None,
        "update": int(update),
        "policy_version": int(policy_version),
        "episodes": int(episodes),
        "env_steps": int(env_steps),
        "seed": int(seed),
        "rng_state": _capture_rng_state(),
        "environment_state": _env_snapshot(env),
    }
    payload = {
        "format": _CHECKPOINT_FORMAT,
        "kind": "training",
        "algorithm": "bounded_ppo",
        "config": _plain(config),
        "env_info": _plain(env_info),
        "architecture": _plain(architecture),
        "policy_state": actor.state_dict(),
        "critic_state": critic.state_dict(),
        "training_state": training_state,
    }
    _atomic_save(payload, destination)
    _atomic_save(
        _policy_payload(config, env_info, architecture, actor),
        destination.parent / "policy.pt",
    )


def _collect_episode(
    env: Any,
    actor: _GaussianActor,
    critic: _CentralCritic,
    value_norm: _ValueNorm | None,
    env_info: dict[str, Any],
    device: torch.device,
    max_steps_remaining: int | None,
) -> dict[str, Any]:
    reset_result = env.reset()
    initial_obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    n_agents = int(env_info["n_agents"])
    obs_dim = int(env_info["obs_shape"])
    state_dim = int(env_info["state_shape"])
    action_dim = int(env_info["n_actions"])
    obs = _as_obs(env.get_obs() if hasattr(env, "get_obs") else initial_obs, n_agents, obs_dim)
    state = _as_state(env.get_state(), state_dim)
    agent_ids = _agent_one_hot(n_agents, device)
    low = torch.as_tensor(env_info["action_low"], dtype=torch.float32, device=device)
    high = torch.as_tensor(env_info["action_high"], dtype=torch.float32, device=device)
    half_range = (high - low) / 2.0
    last_action = torch.zeros(n_agents, action_dim, dtype=torch.float32, device=device)
    hidden = None
    if actor.architecture["actor_type"] == "rnn":
        hidden = torch.zeros(n_agents, actor.hidden_dim, dtype=torch.float32, device=device)

    observations = []
    states = []
    previous_actions = []
    latent_actions = []
    old_log_probs = []
    rewards = []
    terminated = []
    truncated = []
    infos = []
    total_reward = 0.0
    complete = False
    length = 0

    with torch.no_grad():
        while True:
            if max_steps_remaining is not None and length >= max_steps_remaining:
                if length:
                    truncated[-1] = 1.0
                break
            observations.append(obs.copy())
            states.append(state.copy())
            previous_actions.append(last_action.cpu().numpy().copy())
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            mean, hidden = actor.step(obs_tensor, last_action, agent_ids, hidden)
            distribution = actor.distribution(mean)
            latent = distribution.sample()
            normalized = torch.tanh(latent)
            physical = low + (normalized + 1.0) * half_range
            physical_np = np.clip(
                physical.cpu().numpy(), env_info["action_low"], env_info["action_high"]
            ).astype(np.float32)
            log_prob, _ = _transform_log_prob(distribution, latent, half_range)
            latent_actions.append(latent.cpu().numpy().astype(np.float32))
            old_log_probs.append(log_prob.cpu().numpy().astype(np.float32))

            result = env.step(physical_np)
            if not isinstance(result, tuple) or len(result) < 3:
                raise TypeError("PowerGrid step must return (reward, terminated, info)")
            reward, done, info = result[0], bool(result[1]), result[2] or {}
            reward = float(np.asarray(reward).reshape(-1)[0])
            total_reward += reward
            length += 1
            next_obs = _as_obs(env.get_obs(), n_agents, obs_dim)
            next_state = _as_state(env.get_state(), state_dim)
            episode_limit = int(getattr(env, "episode_limit", env_info["episode_limit"]))
            env_limit = bool(done and int(getattr(env, "steps", length)) >= episode_limit)
            is_terminated = bool(done and not env_limit)
            is_truncated = bool(done and env_limit)
            limit_reached = length >= episode_limit and not done
            budget_reached = (
                max_steps_remaining is not None and length >= max_steps_remaining and not done
            )
            if limit_reached or budget_reached:
                is_truncated = True
            rewards.append(reward)
            terminated.append(float(is_terminated))
            truncated.append(float(is_truncated))
            infos.append(copy.deepcopy(info))
            obs, state = next_obs, next_state
            last_action = normalized
            if done or limit_reached or budget_reached:
                complete = bool(done or limit_reached)
                break

    observations.append(obs.copy())
    states.append(state.copy())
    old_values_norm, old_values_raw = _fresh_critic_values(critic, {"observations": observations, "states": states}, device)
    if value_norm is not None:
        with torch.no_grad():
            raw_tensor = value_norm.denormalize(torch.as_tensor(old_values_norm, device=device))
        old_values_raw = raw_tensor.cpu().numpy().astype(np.float32)
    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "states": np.asarray(states, dtype=np.float32),
        "previous_actions": np.asarray(previous_actions, dtype=np.float32),
        "latent_actions": np.asarray(latent_actions, dtype=np.float32),
        "old_log_probs": np.asarray(old_log_probs, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "terminated": np.asarray(terminated, dtype=np.float32),
        "truncated": np.asarray(truncated, dtype=np.float32),
        "old_values_norm": old_values_norm,
        "old_values_raw": old_values_raw,
        "return": float(total_reward),
        "length": int(length),
        "complete": bool(complete),
        "info": infos[-1] if infos else {},
    }


def _fresh_critic_values(
    critic: _CentralCritic,
    episode: dict[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        values = _critic_sequence(critic, episode["observations"], episode["states"], device)
    result = values.detach().cpu().numpy().astype(np.float32)
    return result.copy(), result.copy()


def _tensor_episode_batch(
    episodes: list[dict[str, Any]], device: torch.device
) -> dict[str, torch.Tensor]:
    batch_size = len(episodes)
    max_length = max(len(episode["rewards"]) for episode in episodes)
    first = episodes[0]
    n_agents, obs_dim = first["observations"].shape[1:]
    action_dim = first["latent_actions"].shape[-1]
    state_dim = first["states"].shape[-1]
    obs = np.zeros((batch_size, max_length + 1, n_agents, obs_dim), dtype=np.float32)
    states = np.zeros((batch_size, max_length + 1, state_dim), dtype=np.float32)
    previous_actions = np.zeros(
        (batch_size, max_length, n_agents, action_dim), dtype=np.float32
    )
    latent_actions = np.zeros_like(previous_actions)
    old_log_probs = np.zeros((batch_size, max_length, n_agents, 1), dtype=np.float32)
    rewards = np.zeros((batch_size, max_length, 1), dtype=np.float32)
    terminated = np.zeros_like(rewards)
    truncated = np.zeros_like(rewards)
    filled = np.zeros_like(rewards)
    old_values_norm = np.zeros((batch_size, max_length + 1), dtype=np.float32)
    old_values_raw = np.zeros_like(old_values_norm)
    for index, episode in enumerate(episodes):
        length = len(episode["rewards"])
        obs[index, : length + 1] = episode["observations"]
        states[index, : length + 1] = episode["states"]
        previous_actions[index, :length] = episode["previous_actions"]
        latent_actions[index, :length] = episode["latent_actions"]
        old_log_probs[index, :length] = episode["old_log_probs"]
        rewards[index, :length, 0] = episode["rewards"]
        terminated[index, :length, 0] = episode["terminated"]
        truncated[index, :length, 0] = episode["truncated"]
        filled[index, :length, 0] = 1.0
        old_values_norm[index, : length + 1] = episode["old_values_norm"]
        old_values_raw[index, : length + 1] = episode["old_values_raw"]
    return {
        "obs": torch.as_tensor(obs, device=device),
        "states": torch.as_tensor(states, device=device),
        "previous_actions": torch.as_tensor(previous_actions, device=device),
        "latent_actions": torch.as_tensor(latent_actions, device=device),
        "old_log_probs": torch.as_tensor(old_log_probs, device=device),
        "rewards": torch.as_tensor(rewards, device=device),
        "terminated": torch.as_tensor(terminated, device=device),
        "truncated": torch.as_tensor(truncated, device=device),
        "filled": torch.as_tensor(filled, device=device),
        "old_values_norm": torch.as_tensor(old_values_norm, device=device),
        "old_values_raw": torch.as_tensor(old_values_raw, device=device),
    }


def _gae_targets(
    batch: dict[str, torch.Tensor], gamma: float, gae_lambda: float
) -> tuple[torch.Tensor, torch.Tensor]:
    rewards = batch["rewards"]
    terminated = batch["terminated"]
    truncated = batch["truncated"]
    filled = batch["filled"]
    values = batch["old_values_raw"].unsqueeze(-1)
    batch_size, length = rewards.shape[:2]
    advantages = torch.zeros_like(rewards)
    next_advantage = torch.zeros(batch_size, 1, device=rewards.device)
    for timestep in reversed(range(length)):
        valid = filled[:, timestep]
        bootstrap_mask = 1.0 - terminated[:, timestep]
        trace_mask = (1.0 - terminated[:, timestep]) * (1.0 - truncated[:, timestep])
        delta = (
            rewards[:, timestep]
            + gamma * values[:, timestep + 1] * bootstrap_mask
            - values[:, timestep]
        )
        current = delta + gamma * gae_lambda * trace_mask * next_advantage
        advantages[:, timestep] = current * valid
        next_advantage = advantages[:, timestep]
    returns = advantages + values[:, :-1]
    return advantages, returns


def _evaluate_actor_batch(
    actor: _GaussianActor,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    obs = batch["obs"]
    previous_actions = batch["previous_actions"]
    latent_actions = batch["latent_actions"]
    batch_size, length, n_agents, _ = previous_actions.shape
    device = obs.device
    agent_ids = _agent_one_hot(n_agents, device)
    logprob_episodes, entropy_episodes = [], []
    for episode_index in range(batch_size):
        hidden = None
        if actor.architecture["actor_type"] == "rnn":
            hidden = torch.zeros(
                n_agents, actor.hidden_dim, dtype=obs.dtype, device=device
            )
        logprobs, entropies = [], []
        for timestep in range(length):
            mean, hidden = actor.step(
                obs[episode_index, timestep],
                previous_actions[episode_index, timestep],
                agent_ids,
                hidden,
            )
            distribution = actor.distribution(mean)
            # half_range is independent of policy parameters and the same for all
            # stored actions. It is passed through a tensor attached by the caller.
            logprob, entropy = _transform_log_prob(
                distribution,
                latent_actions[episode_index, timestep],
                batch["half_range"],
            )
            logprobs.append(logprob)
            entropies.append(entropy)
        logprob_episodes.append(torch.stack(logprobs))
        entropy_episodes.append(torch.stack(entropies))
    return torch.stack(logprob_episodes), torch.stack(entropy_episodes)


def _evaluate_critic_batch(
    critic: _CentralCritic,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    obs = batch["obs"]
    states = batch["states"]
    batch_size, length_plus_one = obs.shape[:2]
    values = []
    for episode_index in range(batch_size):
        hidden = None
        if critic.rnn is not None:
            hidden = torch.zeros(1, critic.hidden_dim, dtype=obs.dtype, device=obs.device)
        episode_values = []
        for timestep in range(length_plus_one - 1):
            value, hidden = critic.step(
                obs[episode_index, timestep], states[episode_index, timestep], hidden
            )
            episode_values.append(value)
        values.append(torch.stack(episode_values))
    return torch.stack(values).unsqueeze(-1)


def _ppo_update(
    episodes: list[dict[str, Any]],
    actor: _GaussianActor,
    critic: _CentralCritic,
    optimizer: torch.optim.Optimizer,
    value_norm: _ValueNorm | None,
    config: dict[str, Any],
    half_range: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, float], int]:
    batch = _tensor_episode_batch(episodes, device)
    batch["half_range"] = half_range
    advantages, returns = _gae_targets(
        batch,
        float(config.get("gamma", 0.99)),
        float(config.get("gae_lambda", 0.95)),
    )
    valid = batch["filled"].bool()
    if value_norm is not None:
        value_norm.update(returns[valid])
        targets = value_norm.normalize(returns)
    else:
        targets = returns

    n_agents = int(actor.architecture["n_agents"])
    mask = batch["filled"].unsqueeze(2).expand(-1, -1, n_agents, -1)
    denominator = mask.sum().clamp(min=1.0)
    advantage_agents = advantages.unsqueeze(2).expand(-1, -1, n_agents, -1)
    valid_advantages = advantage_agents[mask.bool()]
    if valid_advantages.numel():
        mean = valid_advantages.mean()
        std = valid_advantages.std(unbiased=False)
        advantage_agents = (advantage_agents - mean) / (std + 1e-6)
    advantage_agents = torch.where(mask.bool(), advantage_agents, torch.zeros_like(advantage_agents))
    old_log_probs = batch["old_log_probs"]
    old_values = batch["old_values_norm"][:, :-1].unsqueeze(-1).unsqueeze(-1)
    old_values = old_values.expand(-1, -1, n_agents, -1)
    targets_agents = targets.unsqueeze(2).expand(-1, -1, n_agents, -1)
    clip = float(config.get("eps_clip", 0.2))
    dual_clip = float(config.get("dual_clip", 0.0) or 0.0)
    entropy_coef = float(config.get("entropy_coef", config.get("entropy", 0.01)))
    critic_coef = float(config.get("critic_coef", 0.5))
    grad_clip = float(config.get("grad_norm_clip", 10.0))
    mini_epochs = int(config.get("mini_epochs", 4))
    if mini_epochs <= 0:
        raise ValueError("mini_epochs must be positive")

    metrics: dict[str, float] = {}
    for _ in range(mini_epochs):
        new_log_probs, entropy = _evaluate_actor_batch(actor, batch)
        values = _evaluate_critic_batch(critic, batch).unsqueeze(2).expand(
            -1, -1, n_agents, -1
        )
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantage_agents
        surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantage_agents
        if dual_clip:
            clipped = torch.minimum(surr1, surr2)
            dual = torch.maximum(clipped, dual_clip * advantage_agents)
            actor_terms = torch.where(advantage_agents < 0, dual, clipped)
        else:
            actor_terms = torch.minimum(surr1, surr2)
        actor_loss = -(actor_terms * mask).sum() / denominator

        clipped_values = old_values + torch.clamp(values - old_values, -clip, clip)
        td_error = torch.maximum(
            (values - targets_agents.detach()).square(),
            (clipped_values - targets_agents.detach()).square(),
        )
        critic_loss = 0.5 * (td_error * mask).sum() / denominator
        entropy_loss = (entropy * mask).sum() / denominator
        loss = actor_loss + critic_coef * critic_loss - entropy_coef * entropy_loss
        scalar_losses = (actor_loss, critic_loss, entropy_loss, loss)
        if not all(bool(torch.isfinite(item).all()) for item in scalar_losses):
            raise FloatingPointError("PowerGrid PPO produced a non-finite loss; optimizer was not stepped")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in list(actor.parameters()) + list(critic.parameters())
            if parameter.grad is not None
        ]
        if not gradients or not all(bool(torch.isfinite(grad).all()) for grad in gradients):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("PowerGrid PPO produced non-finite gradients; optimizer was not stepped")
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(actor.parameters()) + list(critic.parameters()),
            grad_clip,
            error_if_nonfinite=True,
        )
        if not bool(torch.isfinite(torch.as_tensor(grad_norm))):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("PowerGrid PPO gradient norm is non-finite; optimizer was not stepped")
        optimizer.step()
        metrics = {
            "actor_loss": float(actor_loss.detach().cpu()),
            "critic_loss": float(critic_loss.detach().cpu()),
            "entropy": float(entropy_loss.detach().cpu()),
            "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
            "approx_kl": float(((old_log_probs - new_log_probs).detach() * mask).sum().cpu() / denominator.cpu()),
            "clip_fraction": float((((ratio - 1.0).abs() > clip).float() * mask).sum().cpu() / denominator.cpu()),
            "advantage_mean": float((advantage_agents * mask).sum().detach().cpu() / denominator.cpu()),
            "target_mean": float((targets_agents * mask).sum().detach().cpu() / denominator.cpu()),
        }
    return metrics, mini_epochs


def _restore_training_payload(
    payload: dict[str, Any],
    actor: _GaussianActor,
    critic: _CentralCritic,
    optimizer: torch.optim.Optimizer,
    value_norm: _ValueNorm | None,
    env: Any,
    device: torch.device,
) -> tuple[int, int, int, int]:
    actor.load_state_dict(payload["policy_state"])
    critic.load_state_dict(payload["critic_state"])
    state = payload.get("training_state")
    if not isinstance(state, dict) or "optimizer_state" not in state:
        raise ValueError("Resume requires a full PowerGrid training checkpoint")
    optimizer.load_state_dict(state["optimizer_state"])
    if value_norm is not None:
        if state.get("value_norm_state") is None:
            raise ValueError("Resume checkpoint is missing the configured value normalizer")
        value_norm.load_state_dict(state["value_norm_state"])
    _restore_env_state(env, state.get("environment_state", {}))
    for optimizer_state in optimizer.state.values():
        for key, value in optimizer_state.items():
            if torch.is_tensor(value):
                optimizer_state[key] = value.to(device)
    if state.get("rng_state"):
        _restore_rng_state(state["rng_state"])
    return (
        int(state.get("update", 0)),
        int(state.get("policy_version", 0)),
        int(state.get("episodes", 0)),
        int(state.get("env_steps", 0)),
    )


def train(
    config: dict[str, Any],
    output_dir: str | os.PathLike[str],
    resume: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Train bounded continuous PPO and write a resumable full-state checkpoint.

    Config accepts the application manifest fields plus env_args and PPO options.
    ``updates`` is the target total update count when resuming; ``t_max`` or the
    source-style ``max_steps`` optionally caps total environment steps.
    """
    incoming = _effective_config(config)
    resume_payload = _load_checkpoint(resume, map_location="cpu") if resume else None
    if resume_payload is not None:
        incoming = _deep_merge(resume_payload.get("config", {}), incoming)
        incoming = _effective_config(incoming)
    if incoming.get("algorithm", "bounded_ppo") != "bounded_ppo":
        raise ValueError("PowerGrid engine algorithm must be 'bounded_ppo'")
    if incoming.get("execution", "local") != "local":
        raise ValueError("PowerGrid bounded_ppo currently supports execution='local' only")
    if bool(incoming.get("use_vtrace", False)):
        raise ValueError("PowerGrid bounded_ppo uses on-policy GAE; use_vtrace is unsupported")
    if resume_payload is not None and not isinstance(resume_payload.get("training_state"), dict):
        raise ValueError("Resume requires training.pt with optimizer, RNG, and environment state")

    training_state = (resume_payload or {}).get("training_state", {})
    saved_seed = training_state.get("seed")
    if saved_seed is not None and "seed" in incoming and int(incoming["seed"]) != int(saved_seed):
        raise ValueError(
            f"Resume seed {incoming['seed']} differs from the checkpoint seed {saved_seed}"
        )
    seed = int(incoming.get("seed", training_state.get("seed", 0)))
    device = _device_from_config(incoming.get("device", "cpu"))
    output_path = Path(output_dir).expanduser().resolve()
    training_checkpoint_path = output_path / "training.pt"
    policy_checkpoint_path = output_path / "policy.pt"
    metrics_path = output_path / "metrics.jsonl"
    caller_rng_state = _capture_rng_state()
    _seed_everything(seed)
    try:
        incoming["env_args"] = _normalise_env_args(incoming, seed)
        env, env_info = _make_env(incoming, seed)
    except Exception:
        _restore_rng_state(caller_rng_state)
        raise

    try:
        architecture = _actor_architecture(incoming, env_info)
        if resume_payload is not None:
            _checkpoint_contract_matches(resume_payload["env_info"], env_info)
            if resume_payload.get("architecture") != architecture:
                raise ValueError("Resume config changes the saved PowerGrid policy architecture")
        actor = _GaussianActor(architecture).to(device)
        critic = _CentralCritic(architecture).to(device)
        value_norm = (
            _ValueNorm(
                beta=float(incoming.get("value_norm_beta", 0.99999)),
                epsilon=float(incoming.get("value_norm_epsilon", 1e-5)),
            ).to(device)
            if bool(incoming.get("use_value_norm", True))
            else None
        )
        parameters = list(actor.parameters()) + list(critic.parameters())
        optimizer = torch.optim.Adam(
            parameters,
            lr=float(incoming.get("lr", 5e-4)),
            eps=float(incoming.get("optim_eps", 1e-5)),
        )

        updates_done = policy_version = episodes_done = env_steps = 0
        if resume_payload is not None:
            updates_done, policy_version, episodes_done, env_steps = _restore_training_payload(
                resume_payload, actor, critic, optimizer, value_norm, env, device
            )

        target_updates = int(incoming.get("updates", incoming.get("max_updates", 1)))
        if target_updates < 0:
            raise ValueError("updates must be non-negative")
        if resume_payload is None and target_updates == 0:
            raise ValueError("updates must be positive for a new PowerGrid training run")
        if target_updates < updates_done:
            raise ValueError(
                f"updates={target_updates} is below the checkpoint's completed update {updates_done}"
            )
        max_env_steps_value = incoming.get("t_max", incoming.get("max_steps"))
        max_env_steps = int(max_env_steps_value) if max_env_steps_value is not None else 0
        if max_env_steps < 0:
            raise ValueError("t_max/max_steps must be non-negative")
        episodes_per_update = int(
            incoming.get("episodes_per_update", incoming.get("update_rate", 8))
        )
        if episodes_per_update <= 0:
            raise ValueError("episodes_per_update/update_rate must be positive")
        mini_epochs = int(incoming.get("mini_epochs", 4))
        if mini_epochs <= 0:
            raise ValueError("mini_epochs must be positive")
        horizon_value = incoming.get("horizon")
        rollout_horizon = int(horizon_value) if horizon_value is not None else 0
        if rollout_horizon < 0:
            raise ValueError("horizon must be a positive transition budget when supplied")
        if horizon_value is not None and rollout_horizon == 0:
            raise ValueError("horizon must be positive when supplied")
        half_range = torch.as_tensor(
            (np.asarray(env_info["action_high"]) - np.asarray(env_info["action_low"])) / 2.0,
            dtype=torch.float32,
            device=device,
        )
        recent_metrics: dict[str, float] = {}
        if updates_done >= target_updates or (max_env_steps and env_steps >= max_env_steps):
            _save_checkpoint(
                training_checkpoint_path, incoming, env_info, architecture, actor, critic,
                optimizer, value_norm, updates_done, policy_version, episodes_done,
                env_steps, seed, env,
            )
        while updates_done < target_updates and (not max_env_steps or env_steps < max_env_steps):
            rollout = []
            rollout_steps = 0
            for _ in range(episodes_per_update):
                remaining = max_env_steps - env_steps if max_env_steps else None
                if rollout_horizon:
                    horizon_remaining = rollout_horizon - rollout_steps
                    remaining = (
                        horizon_remaining
                        if remaining is None
                        else min(remaining, horizon_remaining)
                    )
                if remaining is not None and remaining <= 0:
                    break
                episode = _collect_episode(
                    env, actor, critic, value_norm, env_info, device, remaining
                )
                if episode["length"] <= 0:
                    break
                rollout.append(episode)
                rollout_steps += int(episode["length"])
                env_steps += int(episode["length"])
                if episode["complete"]:
                    episodes_done += 1
                if remaining is not None and episode["length"] >= remaining:
                    break
            if not rollout:
                break

            update_metrics, epochs_applied = _ppo_update(
                rollout,
                actor,
                critic,
                optimizer,
                value_norm,
                incoming,
                half_range,
                device,
            )
            policy_version += epochs_applied
            updates_done += 1
            returns_list = [float(episode["return"]) for episode in rollout]
            lengths_list = [int(episode["length"]) for episode in rollout]
            recent_metrics = {
                **update_metrics,
                "update": float(updates_done),
                "policy_version": float(policy_version),
                "episodes": float(episodes_done),
                "env_steps": float(env_steps),
                "rollout_episodes": float(len(rollout)),
                "rollout_env_steps": float(rollout_steps),
                "mean_episode_return": float(np.mean(returns_list)),
                "mean_episode_length": float(np.mean(lengths_list)),
            }
            output_path.mkdir(parents=True, exist_ok=True)
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(_plain(recent_metrics), ensure_ascii=False) + "\n")
            _save_checkpoint(
                training_checkpoint_path, incoming, env_info, architecture, actor, critic,
                optimizer, value_norm, updates_done, policy_version, episodes_done,
                env_steps, seed, env,
            )

        return {
            "checkpoint": str(policy_checkpoint_path),
            "training_checkpoint": str(training_checkpoint_path),
            "training_state": str(training_checkpoint_path),
            "updates": updates_done,
            "policy_version": policy_version,
            "episodes": episodes_done,
            "env_steps": env_steps,
            "seed": seed,
            "metrics": recent_metrics,
            "metrics_path": str(metrics_path),
            "last_metrics": recent_metrics,
            "config": _plain(incoming),
            "env_info": _plain(env_info),
        }
    finally:
        try:
            env.close()
        finally:
            _restore_rng_state(caller_rng_state)


def _numeric_info_mean(records: list[dict[str, Any]]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for record in records:
        for key, value in record.items():
            try:
                numeric = float(np.asarray(value).reshape(-1)[0])
            except (TypeError, ValueError, IndexError):
                continue
            if math.isfinite(numeric):
                values.setdefault(str(key), []).append(numeric)
    return {key: float(np.mean(items)) for key, items in values.items() if items}


def evaluate(
    checkpoint: str | os.PathLike[str],
    config: dict[str, Any] | None = None,
    episodes: int = 1,
    seed: int | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Evaluate a saved policy on the supplied SADMA PowerGrid assets."""
    if int(episodes) <= 0:
        raise ValueError("episodes must be positive")
    device_obj = _device_from_config(device)
    payload = _load_checkpoint(checkpoint, map_location=device_obj)
    effective = _effective_config(_deep_merge(payload.get("config", {}), config or {}))
    chosen_seed = int(seed if seed is not None else effective.get("seed", 0))
    caller_rng_state = _capture_rng_state()
    _seed_everything(chosen_seed)
    try:
        env, env_info = _make_env(effective, chosen_seed)
    except Exception:
        _restore_rng_state(caller_rng_state)
        raise
    try:
        _checkpoint_contract_matches(payload["env_info"], env_info)
        policy = GridPolicy(payload, device=device_obj)
        returns: list[float] = []
        lengths: list[int] = []
        all_info: list[dict[str, Any]] = []
        for _ in range(int(episodes)):
            policy.reset()
            reset_result = env.reset()
            obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            obs = _as_obs(env.get_obs() if hasattr(env, "get_obs") else obs, policy.n_agents, policy._obs_dim)
            total_reward = 0.0
            length = 0
            final_info: dict[str, Any] = {}
            while length < int(env_info["episode_limit"]):
                actions = policy.act(obs, deterministic=True)
                result = env.step(actions)
                if not isinstance(result, tuple) or len(result) < 3:
                    raise TypeError("PowerGrid step must return (reward, terminated, info)")
                reward, done, final_info = float(np.asarray(result[0]).reshape(-1)[0]), bool(result[1]), result[2] or {}
                total_reward += reward
                length += 1
                obs = _as_obs(env.get_obs(), policy.n_agents, policy._obs_dim)
                if done:
                    break
            returns.append(total_reward)
            lengths.append(length)
            all_info.append(dict(final_info))
        return {
            "episodes": int(episodes),
            "seed": chosen_seed,
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "mean_length": float(np.mean(lengths)),
            "episode_returns": returns,
            "episode_lengths": lengths,
            "env_steps": sum(lengths),
            "mean_final_info": _numeric_info_mean(all_info),
            "checkpoint": str(Path(checkpoint).expanduser().resolve()),
            "env_info": _plain(env_info),
        }
    finally:
        try:
            env.close()
        finally:
            _restore_rng_state(caller_rng_state)
