"""Resource-backed configuration: default < algorithm < environment < user < overrides."""
from copy import deepcopy
from importlib.resources import files
from pathlib import Path
import yaml
from unified_marl.registry import ALGORITHMS, WANG_ROLES, algorithm_config_name, get_algorithm_spec, load_plugins
from unified_marl.envs import environment_config, validate_compatibility


def merge(base, updates):
    result = deepcopy(base)
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def read_yaml(path):
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError("Configuration must be a mapping")
    return value


def resource(name):
    return yaml.safe_load(files(__package__).joinpath(name).read_text(encoding="utf-8")) or {}


def dotted_overrides(values):
    result = {}
    for key, value in (values or {}).items():
        if not isinstance(key, str) or any(not part for part in key.split(".")):
            raise ValueError("Override keys must contain nonempty dot-separated names")
        cursor = result
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
            if not isinstance(cursor, dict):
                raise ValueError(f"Conflicting override {key!r}")
        if parts[-1] in cursor:
            raise ValueError(f"Conflicting override {key!r}")
        cursor[parts[-1]] = value
    return result


def load_config(algorithm="pymarl/qmix", env="toy", config=None, overrides=None):
    user = read_yaml(config) if isinstance(config, (str, Path)) else deepcopy(config or {})
    if not isinstance(user, dict):
        raise ValueError("Configuration must be a mapping")
    load_plugins(user.get("plugins", []))
    load_plugins((overrides or {}).get("plugins", []))
    env = "sc2" if env == "smac" else env
    spec = get_algorithm_spec(algorithm)
    if spec is not None:
        result = merge(resource("default.yaml"), spec.defaults)
        result = merge(result, environment_config(env))
        result = merge(merge(result, user), dotted_overrides(overrides))
        result.update(algorithm=algorithm, env_config=env)
        validate(result)
        return result
    if algorithm.startswith(("sadma/","sadma_yi/","sadma_yi_closed/")):
        prefix,short=algorithm.split("/",1)
        profile={"sadma":"wang","sadma_yi":"yi_open","sadma_yi_closed":"yi_closed"}[prefix]
        if short not in {"ppo","qmix","vdn"}: raise ValueError("SADMA implements PPO and Qmix/VDN; MADDPG policy source is empty")
        if env=="powergrid": raise ValueError("SADMA powergrid needs continuous float actions [B,T,N,D], bounds and a Gaussian policy protocol; it is not mapped to discrete actions in this stage")
        env="sc2" if env=="smac" else env
        environment_config(env)
        result=merge(resource("default.yaml"),resource(f"sadma/{profile}/default.yaml"))
        result=merge(result,resource(f"sadma/{profile}/{'qmix' if short=='vdn' else short}.yaml"))
        if env in {"sc2", "replenishment", "cityflow"}:
            source_env=resource(f"sadma/{profile}/envs/{'smac' if env=='sc2' else env}.yaml")
            if env=="replenishment":
                source_env["sadma_env_metadata"]={key:source_env["env_args"].pop(key) for key in ["key","map_name","reward"] if key in source_env["env_args"]}
                source_env["env_args"]["horizon"]=None  # SADMA reads the backend horizon, unlike HMF's fixed 21.
            result=merge(result,source_env)
        else:
            result=merge(result,environment_config(env))
        result=merge(result,resource("sadma_runtime.yaml"))
        result.setdefault("max_steps",20050000)
        if env=="toy": result=merge(result,resource("envs/toy.yaml"))
        if short=="vdn": result["mixer"]="vdn_mixer"
        result["env"]=env
        user=read_yaml(config) if isinstance(config,(str,Path)) else config or {}
        result=merge(merge(result,user),dotted_overrides(overrides))
        result.update(algorithm=algorithm,sadma_algorithm="ppo" if short=="ppo" else "qmix",sadma_profile=profile,env_config=env)
        validate(result)
        return result
    if algorithm.startswith(("kou/", "kou_old/")):
        from unified_marl.algorithms.kou.adapter import resolve
        family, short = resolve(algorithm)
        try:
            algorithm_values = resource(f"kou/{family}/{short}.yaml")
        except FileNotFoundError as exc:
            raise ValueError(f"No delivered Kou configuration for {algorithm}") from exc
        result = merge(resource("default.yaml"), resource(f"kou/{family}/default.yaml"))
        result = merge(result, algorithm_values)
        result = merge(result, resource("kou_runtime.yaml"))
        user = read_yaml(config) if isinstance(config,(str,Path)) else config or {}
        result = merge(merge(result,user),dotted_overrides(overrides))
        result.update(algorithm=algorithm,kou_family=family,kou_config=short,env_config=env)
        result["t_max"] = result["schedule_updates"]
        validate(result)
        return result
    name = algorithm_config_name(algorithm)
    environment_config(env)
    user = read_yaml(config) if isinstance(config, (str, Path)) else config or {}
    if name.startswith("wang_hmf/"):
        result = merge(resource("default.yaml"), resource("hmf.yaml"))
        result["hmf_algorithm"] = name.split("/")[-1]
        if result["hmf_algorithm"] in {"hierarchical_vdn_copy_fixed", "hierarchical_vdn_k_means_fixed"}:
            result["hmf_high_weight"] = 0.05  # No main branch; delivered update() default.
    elif name.startswith("wang/"):
        result = merge(resource("default.yaml"), resource("wang/default.yaml"))
        result = merge(result, resource(f"{name}.yaml"))
        if name == "wang/rode" and "n_roles" not in result:
            result["n_roles"] = 2  # Missing source config; explicit initial role count, like rode_hmf.
        if result["runner"] == "episode":
            # Delivered default says episode but batch_size_run=8. A single
            # environment collector has one rollout; explicit user overrides
            # below still undergo validation.
            result["batch_size_run"] = 1
        if name == "wang/coma":
            result["batch_size_run"] = result["batch_size"]
            result["runner"] = "parallel"
    else:
        result = merge(resource("default.yaml"), resource(f"algs/{name}.yaml"))
    result = merge(result, environment_config(env))
    result = merge(result, user)
    result = merge(result, dotted_overrides(overrides))
    result["algorithm"] = name if name.startswith(("wang/", "wang_hmf/")) else "pymarl/" + name
    result["env_config"] = env
    validate(result)
    return result


def validate(config):
    load_plugins(config.get("plugins", []))
    if not isinstance(config.get("seed"), int) or isinstance(config["seed"], bool) or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if type(config.get("num_threads")) is not int or config["num_threads"] < 1:
        raise ValueError("num_threads must be a positive integer")
    if not isinstance(config.get("env_args"), dict):
        raise ValueError("env_args must be a mapping")
    if not config.get("algorithm", "").startswith(("kou/", "kou_old/")):
        validate_compatibility(config)
    spec = get_algorithm_spec(config.get("algorithm"))
    if spec is not None:
        for key, value in spec.locked_fields.items():
            if config.get(key) != value:
                raise ValueError(f"{spec.name} requires {key}={value!r}")
        if config.get("runner") not in {"episode", "parallel"}:
            raise ValueError("runner must be episode or parallel")
        for key in ("batch_size", "batch_size_run", "buffer_size", "t_max", "test_nepisode"):
            if type(config.get(key)) is not int or config[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        if config["runner"] == "episode" and config["batch_size_run"] != 1:
            raise ValueError("episode runner requires batch_size_run=1")
        if spec.training_mode == "on_policy" and config["batch_size"] != config["batch_size_run"]:
            raise ValueError("on_policy plugins consume fresh rollouts; batch_size must equal batch_size_run")
        if config["buffer_size"] < config["batch_size"]:
            raise ValueError("buffer_size must be at least batch_size")
        return
    if config.get("sadma_algorithm"):
        if config.get("env_type")!="discrete": raise ValueError("Continuous SADMA actions are an explicit unimplemented schema extension; no discretization is performed")
        if config["runner"]!="episode" or config["env_batch_size"]!=1 or config["batch_size_run"]!=1 or config.get("async_train"):
            raise ValueError("This stage SADMA runner is local synchronous, one environment per collector; distributed/vector runners are not enabled")
        for key in ["train_batch_size","train_buffer_size","update_rate","t_max","max_steps","hidden_dim","mini_epochs" if config["sadma_algorithm"]=="ppo" else "target_update_interval"]:
            if not isinstance(config.get(key),int) or isinstance(config[key],bool) or config[key]<=0: raise ValueError(f"{key} must be a positive integer")
        if config["train_buffer_size"]<config["train_batch_size"]: raise ValueError("train_buffer_size must cover train_batch_size")
        if config["agent_type"] not in {"rnn","mlp"}: raise ValueError("SADMA agent_type must be rnn or mlp")
        if config["sadma_algorithm"]=="ppo":
            if config["critic_type"] not in {"rnn","mlp"}: raise ValueError("Unsupported SADMA critic_type")
            if sum(key in config["critic_inputs"] for key in ["central_obs","independent_obs"])!=1: raise ValueError("Specify exactly one critic observation layout")
            if not config.get("save_log_probs"): raise ValueError("PPO must retain behavior action_log_probs")
        if config.get("max_updates") is not None and (not isinstance(config["max_updates"],int) or config["max_updates"]<1): raise ValueError("max_updates must be positive")
        return
    if config.get("algorithm", "").startswith(("kou/", "kou_old/")):
        if config["kou_family"] == "icq" and config["learner"] == "q_learner":
            raise ValueError("Delivered ICQ QLearner requires Q/imitator/aux outputs, but its supplied BasicMAC returns one tensor; this BCQ bundle is not integrated")
        if config.get("kou_config") == "acpl_smac":
            raise ValueError("Delivered ACPL config lacks NLL_lamda/use_nll and binds Q outputs to a probability learner; a complete source configuration is required")
        for key in ["training_updates", "schedule_updates", "off_batch_size" if config["kou_family"] == "icq" else "batch_size"]:
            if not isinstance(config.get(key),int) or isinstance(config[key],bool) or config[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        if config["eval_interval_updates"] < 0: raise ValueError("eval_interval_updates cannot be negative")
        if config["learner"] in {"hacpl_learner","maicpl_learner","maidpo_learner"}:
            raise ValueError(f"Source configuration names missing learner {config['learner']}")
        for key in ["e2e","moderate_lambda","use_per","use_reward_hat"]:
            if config.get(key):
                raise ValueError(f"Source variant {key}=true requires an additional training/data protocol that is not integrated; no default path is substituted")
        return
    base = algorithm_config_name(config["algorithm"]).split("/")[-1].removesuffix("_beta")
    hmf = config["algorithm"].startswith("wang_hmf/")
    roles = base in WANG_ROLES
    expected = ("hmf_native", None, "q", "native") if hmf else WANG_ROLES[base] if roles else ALGORITHMS[base]
    for key, value in zip(("learner", "mixer", "agent_output_type", "action_selector"), expected):
        if config.get(key) != value:
            raise ValueError(f"{config['algorithm']} requires {key}={value!r}")
    if roles and (config.get("mac") != "rode_mac" or config.get("agent") != "rode"):
        raise ValueError("RODE bundles require rode_mac and rode agent")
    if hmf and (config.get("mac") != "hmf_native" or config.get("agent") != "hmf_native"):
        raise ValueError("HMF bundles require native controller and learner")
    if not roles and not hmf and (config.get("mac") != "basic_mac" or config.get("agent") != "rnn"):
        raise ValueError("PyMARL baselines require basic_mac and rnn")
    if config["runner"] not in {"episode", "parallel"}:
        raise ValueError("runner must be episode or parallel")
    for key in ["batch_size", "batch_size_run", "buffer_size", "t_max", "test_nepisode", "target_update_interval", "epsilon_anneal_time"]:
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["runner"] == "episode" and config["batch_size_run"] != 1:
        raise ValueError("episode runner requires batch_size_run=1")
    if config["buffer_size"] < config["batch_size"]:
        raise ValueError("buffer_size must be at least batch_size")
    if base == "coma" and config["batch_size"] != config["batch_size_run"]:
        raise ValueError("COMA uses fresh on-policy rollouts: batch_size must equal batch_size_run")
    if config["env"] == "toy":
        for key in ["n_agents", "episode_limit"]:
            v = config["env_args"].get(key, 2 if key == "n_agents" else 4)
            if not isinstance(v, int) or isinstance(v, bool) or v < 1:
                raise ValueError(f"env_args.{key} must be a positive integer")
    if roles:
        for key in ["n_roles", "role_interval", "n_role_clusters"]:
            if not isinstance(config[key], int) or config[key] <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if config["n_roles"] < 2:
            raise ValueError("Source RODE selectors require at least two roles")
        if base == "rode_hmf" and config["batch_size"] < 2:
            raise ValueError("Source RODE-HMF requires batch_size>=2 because its grouping squeezes singleton axes")
    if hmf:
        for key in ["num_groups", "n_heads", "embed_dim", "hmf_steps_per_update", "hmf_num_batch"]:
            if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        if config["embed_dim"] % config["n_heads"]:
            raise ValueError("HMF embed_dim must be divisible by n_heads")
