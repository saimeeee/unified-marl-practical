from dataclasses import asdict
from pathlib import Path
import uuid
import numpy as np
import torch
from unified_marl.datasets import load_dataset
from unified_marl.config import load_config, merge, dotted_overrides, validate
from unified_marl.learners.base import UpdateContext
from unified_marl.checkpoints import read_checkpoint, restore_learner, restore_rng, save_checkpoints
from unified_marl.utils.logging import Logger


def train_offline(algorithm, *, dataset, env=None, env_args=None, device="cpu", seed=None,
                  output_dir=None, config=None, overrides=None, resume=None):
    from unified_marl.api import TrainResult, _make_model, _write_json, check_device, seed_all, evaluate
    payload = read_checkpoint(resume) if resume else None
    if payload:
        if payload["kind"] != "training" or not payload["config"].get("kou_family"):
            raise ValueError("Offline resume requires a Kou training.pt")
        if config is not None:
            raise ValueError("Use overrides for offline resume; replacement config is not supported")
        if algorithm is not None and algorithm != payload["config"]["algorithm"]:
            raise ValueError("Cannot change algorithm during offline resume")
        cfg = merge(payload["config"], dotted_overrides(overrides))
        for key in set(cfg) | set(payload["config"]):
            if key not in {"training_updates", "eval_interval_updates", "eval_episodes"} and cfg.get(key) != payload["config"].get(key):
                raise ValueError(f"Cannot change {key} during offline resume")
        algorithm = cfg["algorithm"]
        if dataset is None: dataset = cfg["dataset_path"]
    else:
        cfg = load_config(algorithm or "kou/bc", env or "sc2", config, overrides)
    if not cfg.get("kou_family"): raise ValueError("This stage offline trainer requires a Kou source algorithm")
    if dataset is None: dataset = cfg.get("dataset_path")
    if dataset is None: raise ValueError("Supply dataset explicitly; original author's absolute data paths are never executed")
    data = load_dataset(dataset, **cfg["dataset_options"])
    if data.reward_semantics != "team":
        raise ValueError("These delivered policy learners require team reward; individual reward is retained separately and never summed/copied")
    cfg["dataset_path"] = data.source
    cfg["device"] = check_device(device)
    cfg["use_cuda"] = cfg["device"].startswith("cuda")
    if seed is not None:
        if payload and seed != cfg["seed"]: raise ValueError("Resume restores RNG; cannot change seed")
        cfg["seed"] = int(seed)
    if env is not None: cfg["env_config"] = env
    if env_args is not None: cfg["env_args"] = merge(cfg["env_args"],env_args)
    validate(cfg)
    cfg["offline_clock"] = "parameter updates; env_steps=0; native t_env/episode_num receive update index"
    directory = Path(output_dir or Path.cwd() / "results" / ("offline-" + uuid.uuid4().hex[:8])).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if any((directory / name).exists() for name in ["config.json","training.pt","policy.pt","metrics.jsonl"]):
        raise FileExistsError(f"Output directory already contains a run: {directory}")
    seed_all(cfg["seed"])
    before_threads = torch.get_num_threads()
    torch.set_num_threads(cfg["num_threads"])
    logger = Logger(directory / "metrics.jsonl")
    try:
        args, scheme, groups, prep, mac, learner = _make_model(cfg, data.env_info, logger)
        reward_model=None
        if cfg.get("reward_model_checkpoint"):
            from unified_marl.rewards import load_reward_model
            reward_model=load_reward_model(cfg["reward_model_checkpoint"],device=device)
            if reward_model.env_info != data.env_info: raise ValueError("Reward model and dataset dimensions differ")
            if reward_model.output_semantics != "team":
                raise ValueError("Native learner needs team reward_hat; no implicit aggregation of individual predictions is performed")
        progress = {"env_steps":0,"episodes":0,"updates":0}
        if payload:
            if payload["env_info"] != data.env_info: raise ValueError("Dataset dimensions differ from checkpoint")
            restore_learner(learner,payload["learner"])
            restore_rng(payload["rng"])
            progress.update(payload["progress"])
        initial = [p.detach().clone() for p in mac.parameters()]
        _write_json(directory / "config.json",cfg)
        _write_json(directory / "dataset.json",dict(source=data.source,env_info=data.env_info,
            reward_semantics=data.reward_semantics,pairing="explicit" if cfg["dataset_options"].get("pair_indices") is not None else cfg["dataset_options"].get("pairing"),
            label_key=data.label_key,episodes=len(data),time_slots=data.time))
        pair_training = bool(cfg.get("use_pref",False)) or "batch1" in __import__("inspect").signature(learner.native.train).parameters
        count = cfg.get("off_batch_size",cfg["batch_size"])
        if pair_training:
            if count % 2: raise ValueError("Source off_batch_size counts both sides and must be even")
            count //= 2
            if data.pairs is None or count > len(data.pairs): raise ValueError("Not enough explicit preference pairs for this batch")
        elif count > len(data): raise ValueError("Not enough episodes for configured batch without replacement")
        metrics = {}
        for update in range(progress["updates"], cfg["training_updates"]):
            indices = np.random.choice(len(data.pairs) if pair_training else len(data), count, replace=False)
            batch = (data.preference(indices) if pair_training else data.batch(indices)).to(device)
            if reward_model is not None:
                if pair_training: raise ValueError("Reward relabeling here supports ordinary TD batches; individual preference weights need explicit agent_reward")
                prediction=reward_model.predict(batch)
                if prediction.shape != batch["reward"].shape: raise ValueError("Predicted team reward does not match the batch time axis")
                batch["reward_hat"]=prediction
            metrics = learner.update(batch,UpdateContext(0,0,update))
            progress["updates"] = update + 1
            logger.record(metrics,update+1)
            if cfg["eval_interval_updates"] and (update+1) % cfg["eval_interval_updates"] == 0:
                if env is None: raise ValueError("Periodic environment evaluation requires explicit env and compatible environment arguments")
                checkpoint, _ = save_checkpoints(directory / f"update-{update+1}",cfg,data.env_info,learner,None,progress)
                evaluation = evaluate(checkpoint,env=env,env_args=env_args,episodes=cfg["eval_episodes"],device=device)
                _write_json(directory / f"evaluation-{update+1}.json",asdict(evaluation))
        if not metrics: raise ValueError("Requested update budget is already exhausted")
        delta = sum(float((a-b.detach()).square().sum()) for a,b in zip(initial,mac.parameters())) ** 0.5
        if not np.isfinite(delta): raise FloatingPointError("Non-finite policy parameter change")
        policy, training = save_checkpoints(directory,cfg,data.env_info,learner,None,progress)
        result = TrainResult(cfg["algorithm"],str(directory),policy,training,str(directory / "metrics.jsonl"),
                            str(directory / "config.json"),0,0,progress["updates"],delta,metrics)
        _write_json(directory / "result.json",asdict(result))
        return result
    finally:
        torch.set_num_threads(before_threads)
