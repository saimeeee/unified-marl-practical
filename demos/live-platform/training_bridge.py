"""Portable traffic training exports and explicit local model registration.

Exports do not run training. Registration accepts locally trained, trusted
unified-marl policy checkpoints and never overwrites an existing model ID.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
import uuid
import zipfile

import yaml

from scenarios import prepare_traffic, validate_parameters

ALGORITHMS = ("pymarl/qmix", "pymarl/vdn", "sadma/ppo")
PROTOCOL = "generic_traffic_v1"


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _copy_traffic_assets(native_path, destination):
    """Copy exactly the scene inputs and use paths relative to the new JSON."""
    native_path = Path(native_path).resolve()
    native = json.loads(native_path.read_text(encoding="utf-8"))
    root = (native_path.parent / native.get("dir", ".")).resolve()
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for field, name in (("roadnetFile", "roadnet.json"), ("flowFile", "flow.json")):
        source = (root / native[field]).resolve()
        target = destination / name
        if source != target.resolve():
            shutil.copyfile(source, target)
        native[field] = name
    native.update(dir=".", saveReplay=False)
    # Replay outputs are not training inputs and may reference the old machine.
    native.pop("roadnetLogFile", None)
    native.pop("replayLogFile", None)
    _json(destination / "cityflow.json", native)


def _training_parameters(parameters):
    if parameters is None:
        return {}
    if not isinstance(parameters, dict):
        raise ValueError("场景参数必须为对象。")
    if not parameters:
        return {}
    if parameters.get("controller", "trained") not in ("trained", "pressure"):
        raise ValueError("请选择有效的展示控制器；训练算法需单独选择。")
    # Training a new checkpoint is allowed on any supported grid. The display
    # controller cannot impose the bundled checkpoint's 2x2 restriction here.
    return validate_parameters("traffic", dict(parameters, controller="pressure"))


def export_traffic_training(model, parameters=None, seed=101, algorithm="pymarl/qmix"):
    """Return ZIP bytes with a relocatable experiment and its exact scene.

    The caller selects model from its server-side catalog. No destination path
    or file names are accepted from the browser.
    """
    if not isinstance(model, dict) or model.get("scenario") != "traffic" or not model.get("config"):
        raise ValueError("请选择有场景配置的交通模型。")
    if algorithm not in ALGORITHMS:
        raise ValueError("训练算法必须为 pymarl/qmix、pymarl/vdn 或 sadma/ppo。")
    if type(seed) is not int or not 0 <= seed <= 2147483647:
        raise ValueError("环境随机种子需为 0 至 2147483647 的整数。")
    parameters = _training_parameters(parameters)
    if model.get("inference_protocol") == PROTOCOL and parameters:
        raise ValueError("已登记模型请导出其原训练场景；新网格请使用随包交通场景作为模板。")
    from unified_marl.applications import load_config
    config = load_config(model["config"])
    with TemporaryDirectory(prefix="marl-training-export-") as temp:
        folder = Path(temp)
        config, provenance = prepare_traffic(config, parameters, folder)
        _copy_traffic_assets(config["env_args"]["cityflow_config_file"], folder / "scenario")
        env_args = deepcopy(config["env_args"])
        env_args.pop("seed", None)
        env_args["cityflow_config_file"] = "scenario/cityflow.json"
        horizon = env_args.get("episode_limit", 32)
        training = dict(t_max=horizon * 8, max_updates=2)
        if algorithm == "sadma/ppo":
            training.update(train_batch_size=2, train_buffer_size=8, update_rate=2,
                            hidden_dim=32, mini_epochs=2)
        else:
            training.update(batch_size=2, buffer_size=8, rnn_hidden_dim=32)
        experiment = dict(schema_version=1, name="traffic-custom-smoke", algorithm=algorithm,
                          env="traffic_engine", seed=seed, device="cpu", env_args=env_args,
                          training=training, evaluation=dict(episodes=2, seed_count=2))
        (folder / "experiment.yaml").write_text(yaml.safe_dump(experiment, allow_unicode=True, sort_keys=False), encoding="utf-8")
        # Training provenance describes an untrained experiment, not the rule
        # policy used internally to generate arbitrary-sized scene assets.
        provenance.pop("controller", None)
        provenance.pop("policy_source", None)
        _json(folder / "scenario.json", dict(provenance, algorithm=algorithm,
              purpose="training_smoke_configuration", existing_checkpoint_reused=False,
              budget_note="Two updates check the pipeline only; increase the budget for experiments."))
        (folder / "README.zh-CN.md").write_text(
            "# 自定义交通训练配置包\n\n"
            "本包包含场景输入与训练配置，不包含训练完成的模型。默认只更新 2 次，供检查训练链路。\n\n"
            "在已安装 unified-marl 和原生 CityFlow 的 Python 环境中，从解压目录执行：\n\n"
            "```bash\npython -m unified_marl check experiment.yaml --instantiate\n"
            "python -m unified_marl run experiment.yaml --output-dir run\n```\n\n"
            "训练算法独立于网页的信号控制器选择。不同路网会分别训练对应模型。"
            "用于实际比较前，请增加训练预算，并配置独立测试场景与多个训练随机种子。\n\n"
            "训练后在完整仓库中执行（路径替换为实际位置）：\n\n"
            "```bash\npython demos/live-platform/training_bridge.py register --run <解压目录>/run "
            "--state demos/live-platform/local-state --id traffic-custom-qmix\n```\n\n"
            "登记会复制模型与场景；重新启动平台后模型进入下拉列表。"
            "登记时请保留本包中的场景文件，训练检查点会引用它们。\n", encoding="utf-8")
        result = io.BytesIO()
        with zipfile.ZipFile(result, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(folder.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(folder).as_posix())
        return result.getvalue()


def _traffic_contract(env_args):
    from unified_marl.envs.traffic_engine import read_traffic_spec
    spec = read_traffic_spec(**env_args)
    info = dict(state_shape=spec["state_size"], obs_shape=spec["obs_size"],
                n_actions=spec["n_actions"], n_agents=spec["n_agents"], episode_limit=spec["episode_limit"])
    semantics = {key: value for key, value in spec.items() if key not in {
        "cityflow_config_file", "cityflow_root", "roadnet_file", "flow_file", "seed", "thread_num"}}
    return dict(env_info=info, semantics=semantics, roadnet_sha256=_hash(spec["roadnet_file"]),
                flow_sha256=_hash(spec["flow_file"]))


def _mpe_info(env_args):
    from unified_marl.envs.mpe import MPEEnv
    env = MPEEnv(**env_args)
    try:
        return env.get_env_info()
    finally:
        env.close()


@contextmanager
def _registration_lock(state):
    path = state / ".model-registration.lock"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ValueError("另一个模型登记正在进行；完成后请重试。") from exc
    try:
        os.close(fd)
        yield
    finally:
        path.unlink(missing_ok=True)


def register_experiment(run_dir, state_dir, model_id, label=None):
    """Copy a trusted core experiment policy and register a portable local model.

    Checkpoints retain their original bytes; registered traffic inference uses
    its copied config instead of absolute training-machine paths in payloads.
    """
    if not isinstance(model_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", model_id):
        raise ValueError("模型 ID 仅支持 1–80 个字母、数字、下划线或连字符，且不能以符号开头。")
    if label is not None and (not isinstance(label, str) or not label.strip() or len(label) > 120):
        raise ValueError("模型名称需为 1–120 个字符。")
    source = Path(run_dir).resolve() / "policy.pt"
    from unified_marl.checkpoints import read_checkpoint
    from unified_marl import load_policy
    payload = read_checkpoint(source)
    if payload["kind"] != "policy":
        raise ValueError("请登记实验目录中的 policy.pt，而不是训练状态。")
    cfg = payload["config"]
    environment = cfg.get("env")
    if environment not in ("traffic_engine", "mpe"):
        raise ValueError("目前只支持登记 traffic_engine 和 mpe 实验模型。")
    algorithm = cfg.get("algorithm")
    if algorithm not in ALGORITHMS or cfg.get("plugins"):
        raise ValueError("目前支持 pymarl/qmix、pymarl/vdn、sadma/ppo 的标准推理模型。")
    env_args = deepcopy(cfg["env_args"])
    contract = _traffic_contract(env_args) if environment == "traffic_engine" else dict(env_info=_mpe_info(env_args))
    if contract["env_info"] != payload["env_info"]:
        raise ValueError("场景观测/动作/智能体或回合元数据与模型不一致。")
    # Loading catches incompatible or incomplete network state before catalog
    # mutation. This constructs no native CityFlow engine and starts no worker.
    load_policy(source, device="cpu")
    state = Path(state_dir).resolve()
    state.mkdir(parents=True, exist_ok=True)
    custom = state / "custom"
    custom.mkdir(exist_ok=True)
    if not custom.resolve().is_relative_to(state):
        raise ValueError("模型目录不能指向 local-state 之外。")
    with _registration_lock(state):
        catalog_path = state / "extra-models.json"
        catalog = json.loads(catalog_path.read_text(encoding="utf-8")) if catalog_path.exists() else {"models": []}
        if not isinstance(catalog, dict) or not isinstance(catalog.get("models"), list):
            raise ValueError("extra-models.json 格式无效。")
        known_ids = {row.get("id") for row in catalog["models"] if isinstance(row, dict)}
        # The generated catalog includes bundled IDs, so reject them too.
        generated_catalog = state / "models.json"
        if generated_catalog.exists():
            known_ids.update(row.get("id") for row in json.loads(generated_catalog.read_text(encoding="utf-8"))["models"])
        bundled = Path(__file__).resolve().parent / "bundle/catalog.json"
        if bundled.exists():
            known_ids.update(row.get("id") for row in json.loads(bundled.read_text(encoding="utf-8"))["models"])
        destination = custom / model_id
        if model_id in known_ids or destination.exists():
            raise ValueError("模型 ID 已存在；请使用新的 ID，不覆盖已有模型。")
        with TemporaryDirectory(prefix=".register-", dir=custom) as temp:
            staging = Path(temp) / model_id
            staging.mkdir()
            shutil.copyfile(source, staging / "policy.pt")
            scenario = "traffic" if environment == "traffic_engine" else "mpe"
            config_path = None
            if scenario == "traffic":
                _copy_traffic_assets(env_args["cityflow_config_file"], staging / "scenario")
                portable_args = deepcopy(env_args)
                portable_args.pop("seed", None)
                portable_args["cityflow_config_file"] = "scenario/cityflow.json"
                application = dict(schema_version=1, application="traffic", backend="cityflow_engine",
                                   algorithm=algorithm, seed=cfg.get("seed", 1), device="cpu", execution="local",
                                   env_args=portable_args, training={}, output={})
                (staging / "config.yaml").write_text(yaml.safe_dump(application, allow_unicode=True, sort_keys=False), encoding="utf-8")
                config_path = f"custom/{model_id}/config.yaml"
            manifest = dict(schema_version=1, inference_protocol=PROTOCOL if scenario == "traffic" else "core_mpe_v1",
                            environment=environment, algorithm=algorithm, checkpoint_sha256=_hash(source), **contract)
            _json(staging / "registration.json", manifest)
            row = dict(id=model_id, scenario=scenario, label=label or ("交通信号" if scenario == "traffic" else "MPE 协作"),
                       algorithm=algorithm, training="本地实验训练产物 · 请结合独立评估判断效果",
                       detail=f"{contract['env_info']['n_agents']} 个智能体 · 已登记训练场景",
                       note="使用登记时的训练场景与原始权重；模型训练预算和效果请查看实验记录。",
                       checkpoint=f"custom/{model_id}/policy.pt", config=config_path,
                       inference_protocol=manifest["inference_protocol"])
            catalog["models"].append(row)
            temporary_catalog = state / (".extra-models-" + uuid.uuid4().hex + ".json")
            moved = False
            try:
                _json(temporary_catalog, catalog)
                staging.replace(destination)
                moved = True
                temporary_catalog.replace(catalog_path)
            except Exception:
                # Only remove the newly-created registration, whose resolved
                # absolute path is checked to remain inside the custom folder.
                if moved and destination.resolve().is_relative_to(custom.resolve()):
                    shutil.rmtree(destination)
                raise
            finally:
                temporary_catalog.unlink(missing_ok=True)
    return row


def load_registered_traffic_policy(model, env, config):
    """Load a registered generic policy with the live worker's action protocol."""
    checkpoint = Path(model["checkpoint"])
    saved = json.loads((checkpoint.parent / "registration.json").read_text(encoding="utf-8"))
    if saved.get("inference_protocol") != PROTOCOL or saved.get("checkpoint_sha256") != _hash(checkpoint):
        raise ValueError("登记元数据或模型摘要不匹配，请重新登记原实验模型。")
    actual = _traffic_contract(config["env_args"])
    if any(actual[key] != saved.get(key) for key in actual) or env.get_env_info() != saved["env_info"]:
        raise ValueError("当前交通场景与登记模型不兼容，请使用原场景或重新训练。")
    from unified_marl import load_policy
    raw = load_policy(checkpoint, device="cpu")

    class TrafficPolicy:
        def reset(self):
            raw.reset()

        def act(self, observation, avail_actions):
            actions = raw.act(observation, avail_actions).tolist()
            return dict(raw_actions=actions, decisions=env.actions_to_phase_ids(actions))

    return TrafficPolicy()


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    register = commands.add_parser("register", help="Register a trusted local experiment model")
    register.add_argument("--run", required=True, type=Path)
    register.add_argument("--state", required=True, type=Path)
    register.add_argument("--id", required=True)
    register.add_argument("--label")
    args = parser.parse_args()
    row = register_experiment(args.run, args.state, args.id, args.label)
    print(json.dumps(row, ensure_ascii=False, indent=2))
    print("Registered. Restart the local platform to refresh its model catalog.")


if __name__ == "__main__":
    main()
