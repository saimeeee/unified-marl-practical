"""Explicit YAML experiments and sequential matrices over the public train API.

Plugin modules are trusted Python code, imported only when a run is executed.
Only documented path fields and explicitly declared ``path_keys`` are resolved
against the YAML directory. The module never substitutes an environment.
"""
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import csv
import importlib
import itertools
import json
from pathlib import Path
import re
import statistics
import time
import uuid

import yaml

from .api import check_device, evaluate, train
from .config import dotted_overrides, load_config, merge


_EXPERIMENT_KEYS = {
    "schema_version", "name", "algorithm", "env", "env_args", "training",
    "device", "seed", "plugins", "evaluation", "path_keys",
}
_SUITE_KEYS = {
    "schema_version", "name", "algorithms", "envs", "seeds", "training",
    "device", "plugins", "evaluation",
}
_PATH_KEYS = {
    "agents_file", "demand_file", "cityflow_config_file", "config_file",
    "data_path", "dataset_path", "data_dir", "vis_path",
}
_ROUTING_KEYS = {"algorithm", "env", "env_config", "env_args", "seed", "device", "use_cuda", "plugins"}
# These legacy source options have no effect in the packaged train() loop.
_UNSUPPORTED_TRAINING = {
    "evaluate", "save_model", "save_model_interval", "checkpoint_path", "load_step",
    "save_replay", "local_results_path", "use_tensorboard", "repeat_id", "label",
    "test_interval", "test_nepisode", "test_greedy",
}


class ExperimentRunError(RuntimeError):
    """A run failed; ``result`` and its JSON file retain the failure details."""

    def __init__(self, result):
        self.result = result
        super().__init__(f"Experiment failed ({result['error']['type']}): {result['error']['message']}; "
                         f"details: {result['output_dir']}")


def _mapping(value, label):
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise ValueError(f"{label} must be a mapping with string keys")
    return deepcopy(value)


def _unknown(value, allowed, label):
    extra = set(value) - allowed
    if extra:
        raise ValueError(f"Unknown {label} key(s): {', '.join(sorted(extra))}")


def _name(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _seed(value, label):
    if type(value) is not int or not 0 <= value < 2**32:
        raise ValueError(f"{label} must be an integer in [0, 2**32)")
    return value


def _sequence(value, label):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a nonempty list")
    return value


def _read(path):
    source = Path(path).expanduser().resolve()
    # Reject duplicate keys rather than allowing YAML to silently overwrite one.
    class UniqueLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader, node, deep=False):
        loader.flatten_mapping(node)
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ValueError("YAML mapping keys must be strings")
            if key in result:
                raise ValueError(f"Duplicate YAML key: {key}")
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)
    value = yaml.load(source.read_text(encoding="utf-8-sig"), Loader=UniqueLoader)
    try:
        json.dumps(value, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ValueError("YAML values must be finite JSON-compatible scalars, lists or mappings") from exc
    return source, _mapping(value, "Experiment YAML")


def _plugins(value):
    if not isinstance(value, list) or any(not isinstance(v, str) or not v.strip() for v in value):
        raise ValueError("plugins must be a list of importable Python module names")
    if len(set(value)) != len(value):
        raise ValueError("plugins contains duplicate module names")
    return value


def _paths(env_args, path_keys, base):
    if not isinstance(path_keys, list) or any(not isinstance(k, str) or not k for k in path_keys):
        raise ValueError("path_keys must be a list of dot-separated keys inside env_args")
    result = deepcopy(env_args)
    keys = set(path_keys) | (_PATH_KEYS & set(result))
    for key in keys:
        parts = key.split(".")
        if any(not part for part in parts):
            raise ValueError(f"Invalid path_keys entry: {key!r}")
        cursor = result
        for part in parts[:-1]:
            if not isinstance(cursor.get(part), dict):
                raise ValueError(f"path_keys entry does not exist in env_args: {key}")
            cursor = cursor[part]
        if parts[-1] not in cursor:
            raise ValueError(f"path_keys entry does not exist in env_args: {key}")
        value = cursor[parts[-1]]
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"env_args.{key} must be a nonempty file path or null")
        path = Path(value).expanduser()
        cursor[parts[-1]] = str((path if path.is_absolute() else base / path).resolve())
    return result


def _evaluation(value, training_seeds):
    result = _mapping(value, "evaluation")
    _unknown(result, {"episodes", "seeds", "seed_count"}, "evaluation")
    episodes = result.get("episodes", 5)
    if type(episodes) is not int or episodes < 1:
        raise ValueError("evaluation.episodes must be a positive integer")
    if "seeds" in result and "seed_count" in result:
        raise ValueError("Specify evaluation.seeds or evaluation.seed_count, not both")
    if "seeds" in result:
        seeds = [_seed(v, "evaluation.seeds") for v in _sequence(result["seeds"], "evaluation.seeds")]
        if len(set(seeds)) != len(seeds) or set(seeds) & set(training_seeds):
            raise ValueError("Evaluation seeds must be unique and disjoint from all training seeds")
    else:
        count = result.get("seed_count", 1)
        if type(count) is not int or count < 1 or count > 10000:
            raise ValueError("evaluation.seed_count must be an integer from 1 to 10000")
        used = set(training_seeds)
        candidate = (min(training_seeds) + 1000003) % 2**32
        seeds = []
        while len(seeds) < count:
            if candidate not in used:
                seeds.append(candidate)
                used.add(candidate)
            candidate = (candidate + 1) % 2**32
    return {"episodes": episodes, "seeds": seeds}


def _normalise(value, base, training_seeds=None):
    value = _mapping(value, "experiment")
    _unknown(value, _EXPERIMENT_KEYS, "experiment")
    if type(value.get("schema_version", 1)) is not int or value.get("schema_version", 1) != 1:
        raise ValueError("Only schema_version: 1 is supported")
    seed = _seed(value.get("seed", 1), "seed")
    env_args = _mapping(value.get("env_args", {}), "env_args")
    if "seed" in env_args:
        raise ValueError("Set seed at the experiment top level, not inside env_args")
    return {
        "schema_version": 1,
        "name": _name(value.get("name", "experiment"), "name"),
        "algorithm": _name(value.get("algorithm"), "algorithm"),
        "env": _name(value.get("env"), "env"),
        "env_args": _paths(env_args, value.get("path_keys", []), base),
        "training": _mapping(value.get("training", {}), "training"),
        "device": _name(value.get("device", "cpu"), "device"),
        "seed": seed,
        "plugins": _plugins(value.get("plugins", [])),
        "path_keys": deepcopy(value.get("path_keys", [])),
        "evaluation": _evaluation(value.get("evaluation", {}), training_seeds or [seed]),
    }


def load_experiment(path):
    """Parse and normalise a single YAML without importing plugins or training."""
    source, value = _read(path)
    return _normalise(value, source.parent)


def _check_training(value, baseline, label):
    for key, child in value.items():
        if key in _ROUTING_KEYS or key in _UNSUPPORTED_TRAINING:
            raise ValueError(f"{label}.{key} is not supported here; use explicit experiment fields")
        if key not in baseline:
            raise ValueError(f"Unknown {label} key: {key}")
        if isinstance(child, dict) and isinstance(baseline[key], dict):
            _check_nested(child, baseline[key], f"{label}.{key}")


def _check_nested(value, baseline, label):
    for key, child in value.items():
        if key not in baseline:
            raise ValueError(f"Unknown {label} key: {key}")
        if isinstance(child, dict) and isinstance(baseline[key], dict):
            _check_nested(child, baseline[key], f"{label}.{key}")


def _jsonable(value):
    value = asdict(value) if is_dataclass(value) else value
    # Fail inside the recorded execution boundary if a backend returns invalid
    # metrics, rather than leaving a misleading running-status JSON behind.
    json.dumps(value, allow_nan=False)
    return value


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _reserve(output_dir, source, prefix):
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        directory = source.parent / "results" / f"{prefix}-{stamp}-{uuid.uuid4().hex[:8]}"
    else:
        directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _resolve(spec, source, overrides=None):
    for module in spec["plugins"]:
        importlib.import_module(module)
    baseline = load_config(spec["algorithm"], spec["env"], config={"plugins": spec["plugins"]})
    _check_training(spec["training"], baseline, "training")
    raw_overrides = _mapping({} if overrides is None else overrides, "overrides")
    try:
        json.dumps(raw_overrides, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("overrides must contain finite JSON-compatible values") from exc
    expanded = dotted_overrides(raw_overrides)
    extra_env = expanded.pop("env_args", {})
    if not isinstance(extra_env, dict) or "seed" in extra_env:
        raise ValueError("overrides.env_args must be a mapping and may not override seed")
    _check_training(expanded, baseline, "overrides")
    effective_env = _paths(merge(spec["env_args"], extra_env), spec["path_keys"], source.parent)
    effective_overrides = {key: value for key, value in raw_overrides.items()
                           if key != "env_args" and not key.startswith("env_args.")}
    device = check_device(spec["device"])
    resolved = load_config(spec["algorithm"], spec["env"], config={
        **spec["training"], "plugins": spec["plugins"], "env_args": effective_env,
        "seed": spec["seed"], "device": device, "use_cuda": device.startswith("cuda"),
    }, overrides=effective_overrides)
    if resolved["env"] != spec["env"] and (spec["env"], resolved["env"]) not in {
        ("smac", "sc2"), ("sc2_beta", "sc2"),
    }:
        raise ValueError(f"Environment configuration changed requested {spec['env']!r} "
                         f"to {resolved['env']!r}; implicit environment substitution is forbidden")
    resolved["env_args"] = dict(resolved["env_args"], seed=spec["seed"])
    return resolved, effective_env, effective_overrides, raw_overrides


def check_experiment(path, instantiate=False, overrides=None):
    """Resolve configuration and capabilities, optionally create/close an env.

    Imports the YAML's explicitly selected trusted plugin modules. Neither mode
    performs training or calls step(); construction may load external assets.
    """
    from .envs import SPECS, make_env, validate_compatibility
    from .registry import get_algorithm_spec

    if type(instantiate) is not bool:
        raise ValueError("instantiate must be a boolean")
    source = Path(path).expanduser().resolve()
    spec = load_experiment(source)
    config, _, _, _ = _resolve(spec, source, overrides)
    validate_compatibility(config)
    algorithm = get_algorithm_spec(config["algorithm"])
    required = (algorithm.required_capabilities if algorithm else
                ("discrete", "fixed_agents", "team_reward", "global_state", "action_mask"))
    result = {"status": "success", "algorithm": config["algorithm"], "env": config["env"],
              "config": config, "evaluation": spec["evaluation"],
              "validation_level": "environment_metadata" if instantiate else "configuration",
              "training_executed": False, "instantiated": instantiate,
              "capabilities": {"required": list(required), "available": list(SPECS[config["env"]].capabilities),
                               "compatible": True}}
    if instantiate:
        env = make_env(config["env"], **config["env_args"])
        try:
            info = env.get_env_info()
            validate_compatibility(config, info)
            result["env_info"] = info
        finally:
            env.close()
    return result


def _execute(spec, source, directory, overrides=None):
    result = {"status": "running", "algorithm": spec["algorithm"], "env": spec["env"],
              "seed": spec["seed"], "env_args": deepcopy(spec["env_args"]),
              "output_dir": str(directory), "source": str(source), "evaluations": []}
    started = time.monotonic()
    _write(directory / "experiment.json", spec)
    _write(directory / "experiment_result.json", result)
    try:
        resolved, effective_env, effective_overrides, raw_overrides = _resolve(spec, source, overrides)
        result["resolved_algorithm"] = resolved["algorithm"]
        result["resolved_env"] = resolved["env"]
        result["env_args"] = effective_env
        result["overrides"] = raw_overrides
        trained = train(spec["algorithm"], spec["env"], effective_env,
                        config={**spec["training"], "plugins": spec["plugins"]}, overrides=effective_overrides,
                        output_dir=directory, device=spec["device"], seed=spec["seed"])
        result["train"] = _jsonable(trained)
        checkpoint = result["train"]["checkpoint"]
        for eval_seed in spec["evaluation"]["seeds"]:
            evaluated = evaluate(checkpoint, env=spec["env"], env_args=effective_env,
                                 episodes=spec["evaluation"]["episodes"], device=spec["device"], seed=eval_seed)
            result["evaluations"].append({"seed": eval_seed, **_jsonable(evaluated)})
        seed_means = [item["return_mean"] for item in result["evaluations"]]
        result["evaluation_return_mean"] = statistics.mean(seed_means)
        result["evaluation_return_std_across_seeds"] = statistics.pstdev(seed_means)
        result["status"] = "success"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    except (KeyboardInterrupt, SystemExit) as exc:
        result["status"] = "interrupted"
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        result["elapsed_seconds"] = time.monotonic() - started
        _write(directory / "experiment_result.json", result)
        raise
    result["elapsed_seconds"] = time.monotonic() - started
    _write(directory / "experiment_result.json", result)
    return result


def run_experiment(path, output_dir=None, overrides=None):
    """Train and evaluate one experiment; failures raise with a saved result."""
    source = Path(path).expanduser().resolve()
    spec = load_experiment(source)
    directory = _reserve(output_dir, source, "experiment")
    result = _execute(spec, source, directory, overrides)
    if result["status"] != "success":
        raise ExperimentRunError(result)
    return result


def _axis(value, kind):
    allowed = {"name", "training"} if kind == "algorithms" else {"name", "env_args", "path_keys"}
    result = []
    for item in _sequence(value, kind):
        item = {"name": item} if isinstance(item, str) else _mapping(item, kind)
        _unknown(item, allowed, kind)
        _name(item.get("name"), f"{kind}.name")
        if kind == "algorithms":
            item["training"] = _mapping(item.get("training", {}), "algorithms.training")
        else:
            item["env_args"] = _mapping(item.get("env_args", {}), "envs.env_args")
        result.append(item)
    return result


def _suite_summary(results):
    groups = {}
    for result in results:
        key = (result["algorithm_index"], result["env_index"])
        group = groups.setdefault(key, {"algorithm": result["algorithm"], "env": result["env"],
                                        "algorithm_index": key[0], "env_index": key[1],
                                        "successful_runs": 0, "failed_runs": 0, "seed_means": []})
        if result["status"] == "success":
            group["successful_runs"] += 1
            group["seed_means"].append(result["evaluation_return_mean"])
        else:
            group["failed_runs"] += 1
    for group in groups.values():
        values = group.pop("seed_means")
        group["return_mean"] = statistics.mean(values) if values else None
        group["return_std_across_training_seeds"] = statistics.pstdev(values) if values else None
    return list(groups.values())


def _save_suite(directory, result):
    _write(directory / "suite_result.json", result)
    columns = ["algorithm", "env", "seed", "status", "algorithm_index", "env_index", "output_dir",
               "evaluation_return_mean", "evaluation_return_std_across_seeds", "elapsed_seconds", "error"]
    with (directory / "suite_result.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in result["runs"]:
            writer.writerow({**row, "error": json.dumps(row["error"], ensure_ascii=False) if "error" in row else ""})


def run_suite(path, output_dir=None):
    """Run a Cartesian matrix, continuing on individual failures.

    Returns success, partial_failure, or failed. Summary statistics are scoped
    to each algorithm/environment entry, never ranked across environments.
    """
    source, value = _read(path)
    _unknown(value, _SUITE_KEYS, "suite")
    if type(value.get("schema_version", 1)) is not int or value.get("schema_version", 1) != 1:
        raise ValueError("Only schema_version: 1 is supported")
    algorithms = _axis(value.get("algorithms"), "algorithms")
    envs = _axis(value.get("envs"), "envs")
    seeds = [_seed(v, "seeds") for v in _sequence(value.get("seeds"), "seeds")]
    if len(set(seeds)) != len(seeds):
        raise ValueError("suite.seeds must be unique")
    evaluation = _evaluation(value.get("evaluation", {}), seeds)
    common_training = _mapping(value.get("training", {}), "training")
    specs = []
    for (ai, algorithm), (ei, env), seed in itertools.product(enumerate(algorithms), enumerate(envs), seeds):
        spec = _normalise({"name": value.get("name", source.stem), "algorithm": algorithm["name"],
                           "env": env["name"], "env_args": env["env_args"], "seed": seed,
                           "training": merge(common_training, algorithm["training"]),
                           "plugins": value.get("plugins", []), "device": value.get("device", "cpu"),
                           "path_keys": env.get("path_keys", []), "evaluation": evaluation}, source.parent, seeds)
        specs.append((ai, ei, spec))
    directory = _reserve(output_dir, source, "suite")
    _write(directory / "suite.json", value)
    result = {"status": "running", "source": str(source), "output_dir": str(directory),
              "expected_runs": len(specs), "runs": [], "summary": [],
              "aggregation": "Per algorithm/environment entry; no cross-environment ranking"}
    _save_suite(directory, result)
    for index, (ai, ei, spec) in enumerate(specs):
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", f"{spec['algorithm']}-{spec['env']}").strip("-")[:70]
        child = directory / f"{index:04d}-{slug}-seed{spec['seed']}"
        child.mkdir(exist_ok=False)
        try:
            run = _execute(spec, source, child)
        except (KeyboardInterrupt, SystemExit):
            run = json.loads((child / "experiment_result.json").read_text(encoding="utf-8"))
            run.update(algorithm_index=ai, env_index=ei)
            result["runs"].append(run)
            result["summary"] = _suite_summary(result["runs"])
            result["status"] = "interrupted"
            _save_suite(directory, result)
            raise
        run.update(algorithm_index=ai, env_index=ei)
        result["runs"].append(run)
        result["summary"] = _suite_summary(result["runs"])
        _save_suite(directory, result)
    successes = sum(run["status"] == "success" for run in result["runs"])
    result["status"] = "success" if successes == len(specs) else "partial_failure" if successes else "failed"
    result["successful_runs"] = successes
    result["failed_runs"] = len(specs) - successes
    _save_suite(directory, result)
    return result
