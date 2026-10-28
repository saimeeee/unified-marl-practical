"""Environment factories import optional simulators only when selected."""
from .multiagentenv import MultiAgentEnv
from dataclasses import dataclass, field
from copy import deepcopy


@dataclass(frozen=True)
class EnvironmentSpec:
    name: str
    factory: object
    capabilities: tuple = ("discrete", "fixed_agents", "team_reward", "global_state", "action_mask")
    defaults: dict = field(default_factory=dict)
    config_resource: str | None = None
    dependency: str | None = None
    description: str = ""


SPECS = {}


def register_environment(spec):
    if not isinstance(spec, EnvironmentSpec) or not spec.name or not callable(spec.factory):
        raise ValueError("EnvironmentSpec requires a name and callable factory")
    if spec.name in SPECS and SPECS[spec.name] != spec:
        raise ValueError(f"Environment already registered: {spec.name}")
    if not isinstance(spec.defaults, dict):
        raise ValueError("Environment defaults must be a mapping")
    SPECS[spec.name] = deepcopy(spec)
    REGISTRY[spec.name] = spec.factory
    return spec


def environment_config(name):
    from unified_marl.config import resource, merge
    if name == "smac":
        name = "sc2"
    if name == "sc2_beta":
        return resource("envs/sc2_beta.yaml")
    try:
        spec = SPECS[name]
    except KeyError:
        raise ValueError(f"Unknown environment configuration {name!r}; register an EnvironmentSpec") from None
    result = resource(spec.config_resource) if spec.config_resource else {"env": name, "env_args": {}}
    return merge(result, spec.defaults)


def environment_catalog():
    return [dict(name=s.name, capabilities=list(s.capabilities), dependency=s.dependency,
                 description=s.description) for s in SPECS.values()]


def validate_compatibility(config, info=None):
    from unified_marl.registry import get_algorithm_spec
    name = config["env"]
    if name not in SPECS:
        raise ValueError(f"Unknown environment registry key {name!r}")
    spec = SPECS[name]
    algorithm = get_algorithm_spec(config.get("algorithm"))
    required = set(algorithm.required_capabilities if algorithm else
                   ("discrete", "fixed_agents", "team_reward", "global_state", "action_mask"))
    missing = required - set(spec.capabilities)
    if missing:
        raise ValueError(f"{config.get('algorithm')} requires environment capabilities {sorted(missing)}; {name} does not declare them")
    if info is not None:
        for key in ("n_agents", "n_actions", "obs_shape", "state_shape", "episode_limit"):
            value = info.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"The discrete online protocol requires a positive integer env_info.{key}")


def _toy(**kwargs):
    from .toy import CoordinationEnv
    return CoordinationEnv(**kwargs)


def _sc2(**kwargs):
    from .smac import make_smac
    return make_smac(**kwargs)


def _replenishment(**kwargs):
    from .replenishment import ReplenishmentAdapter
    return ReplenishmentAdapter(**kwargs)


def _cityflow(**kwargs):
    from .cityflow import CityFlowAdapter
    return CityFlowAdapter(**kwargs)


def _inventory(**kwargs):
    from .inventory import InventoryEnv
    return InventoryEnv(**kwargs)


def _traffic_engine(**kwargs):
    from .traffic_engine import CityFlowEngineEnv
    return CityFlowEngineEnv(**kwargs)


def _mpe(**kwargs):
    from .mpe import MPEEnv
    return MPEEnv(**kwargs)


REGISTRY = {}
for _name, _factory, _dependency, _description in (
    ("toy", _toy, None, "Small coordination fixture; not a domain benchmark"),
    ("sc2", _sc2, "StarCraft II + SMAC + maps", "SMAC cooperative micromanagement"),
    ("replenishment", _replenishment, "ReplenishmentEnv + scenario assets", "Original external inventory backend"),
    ("cityflow", _cityflow, "cityflowenv + CityFlow + scenario assets", "Original traffic wrapper"),
    ("inventory", _inventory, None, "Parameterized warehouse/SKU inventory simulation"),
    ("traffic_engine", _traffic_engine, "CityFlow + roadnet/flow JSON", "Native CityFlow traffic engine"),
    ("mpe", _mpe, None, "Delivered MPE simple_spread cooperative particle physics"),
):
    _caps = ("discrete", "fixed_agents", "team_reward", "global_state", "action_mask")
    if _name in {"inventory", "replenishment", "mpe"}:
        _caps += ("individual_rewards",)
    register_environment(EnvironmentSpec(_name, _factory, _caps,
        config_resource="sadma/wang/envs/cityflow.yaml" if _name == "cityflow" else f"envs/{_name}.yaml",
        dependency=_dependency, description=_description))


def make_env(name, **kwargs):
    try:
        factory = REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown environment {name!r}") from None
    return factory(**kwargs)
