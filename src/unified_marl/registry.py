"""Algorithm registration independent of environment names.

Plugins register an importable Controller/Learner bundle and its data contract.
The initial plugin protocol is fixed-agent, discrete online learning. Legacy
offline and SADMA trainers keep their own verified update semantics.
"""
from dataclasses import dataclass, field
from copy import deepcopy
from importlib import import_module
import re


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    controller: str
    learner: str
    defaults: dict
    training_mode: str = "off_policy"
    required_capabilities: tuple = ("discrete", "fixed_agents", "team_reward", "global_state", "action_mask")
    locked_fields: dict = field(default_factory=dict)
    scheme_factory: str | None = None
    description: str = ""


_PLUGINS = {}


def import_symbol(reference):
    if not isinstance(reference, str) or reference.count(":") != 1:
        raise ValueError("Components must be importable 'module:Class' references")
    module, name = reference.split(":")
    value = import_module(module)
    for part in name.split("."):
        value = getattr(value, part)
    return value


def load_plugins(modules):
    if modules is None:
        return
    if not isinstance(modules, (list, tuple)) or any(not isinstance(x, str) or not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", x) for x in modules):
        raise ValueError("plugins must be a list of importable Python module names")
    for module in modules:
        import_module(module)


def register_algorithm(spec):
    if not isinstance(spec, AlgorithmSpec):
        raise TypeError("register_algorithm expects AlgorithmSpec")
    if not re.fullmatch(r"[a-zA-Z_][\w-]*/[a-zA-Z_][\w-]*", spec.name):
        raise ValueError("Plugin algorithms need a namespaced name, e.g. lab/new_algorithm")
    if spec.name.split("/")[0] in {"pymarl", "wang", "wang_hmf", "kou", "kou_old", "sadma", "sadma_yi", "sadma_yi_closed"}:
        raise ValueError("Plugin cannot replace a built-in algorithm namespace")
    if spec.training_mode not in {"off_policy", "on_policy"}:
        raise ValueError("Plugin training_mode must be off_policy or on_policy")
    for reference in (spec.controller, spec.learner, spec.scheme_factory):
        if reference is not None and (not isinstance(reference, str) or reference.count(":") != 1):
            raise ValueError("Plugin component must use module:Symbol syntax")
    if not isinstance(spec.defaults, dict) or not isinstance(spec.locked_fields, dict):
        raise ValueError("Plugin defaults and locked_fields must be mappings")
    if {"env", "env_args", "env_config", "algorithm", "plugins"} & spec.defaults.keys():
        raise ValueError("Algorithm defaults cannot bind an environment or plugin loader")
    if "discrete" not in spec.required_capabilities:
        raise ValueError("The current online plugin protocol requires discrete actions")
    previous = _PLUGINS.get(spec.name)
    if previous is not None and previous != spec:
        raise ValueError(f"Algorithm already registered: {spec.name}")
    _PLUGINS[spec.name] = deepcopy(spec)
    return spec


def get_algorithm_spec(name):
    return _PLUGINS.get(name)


def algorithm_catalog():
    result = [dict(name="pymarl/" + name, training_mode="on_policy" if name == "coma" else "off_policy",
                   action_space="discrete", source="built-in") for name in ALGORITHMS]
    result += [dict(name="sadma/" + name, training_mode="on_policy" if name == "ppo" else "off_policy",
                    action_space="discrete", source="SADMA local") for name in ("qmix", "vdn", "ppo")]
    result += [dict(name=s.name, training_mode=s.training_mode, action_space="discrete", source="plugin",
                    description=s.description) for s in _PLUGINS.values()]
    return result
ALGORITHMS = {
    "qmix": ("q_learner", "qmix", "q", "epsilon_greedy"),
    "vdn": ("q_learner", "vdn", "q", "epsilon_greedy"),
    "iql": ("q_learner", None, "q", "epsilon_greedy"),
    "coma": ("coma_learner", None, "pi_logits", "multinomial"),
    "qtran": ("qtran_learner", "qtran_base", "q", "epsilon_greedy"),
}
WANG_ROLES = {
    "rode": ("rode_learner", "qmix", "q", "soft_epsilon_greedy"),
    "rode_hmf": ("rode_hmf_learner", "vdn", "q", "soft_epsilon_greedy"),
}
HMF_ALGORITHMS = {
    "dqn", "vdn", "qmix", "mfq", "mfvfd", "hierarchical", "hierarchical_vdn",
    "hierarchical_vdn_IPC", "hierarchical_vdn_without_high_level", "maddpg",
    "hierarchical_vdn_copy_fixed", "hierarchical_vdn_meanq", "hierarchical_vdn_k_means_fixed",
    "hierarchical_vdac_fixed",
}


def algorithm_config_name(name):
    if name.startswith("wang_hmf/"):
        if name.split("/", 1)[1] not in HMF_ALGORITHMS:
            raise ValueError(f"Unsupported fixed-group HMF algorithm {name!r}")
        return name
    if name.startswith("wang/"):
        short = name.split("/", 1)[1]
        if short in WANG_ROLES:
            return "wang/" + short
        if short in ALGORITHMS or short in {"qmix_beta", "vdn_beta", "iql_beta"}:
            return "wang/" + short
        raise ValueError(f"Unsupported Wang RGHMF algorithm {short!r}")
    if name.startswith("pymarl/"):
        name = name.split("/", 1)[1]
    base = name.removesuffix("_beta")
    if base not in ALGORITHMS or (name.endswith("_beta") and base not in {"qmix", "vdn", "iql"}):
        raise ValueError(f"Unsupported algorithm {name!r}; available: {', '.join(ALGORITHMS)}")
    return name
