"""Synchronous SADMA: episodic replay for Q, one-use current-policy rolls for PPO."""
from dataclasses import asdict
from pathlib import Path
import uuid
import numpy as np
import torch
from unified_marl.checkpoints import read_checkpoint,restore_learner,restore_rng,save_checkpoints,atomic_save
from unified_marl.config import load_config,merge,dotted_overrides,validate
from unified_marl.components.episode_buffer import EpisodeBatch
from unified_marl.learners.base import UpdateContext
from unified_marl.runners.episode_runner import EpisodeRunner
from unified_marl.utils.logging import Logger
from unified_marl.algorithms.sadma.adapter import native_batch
from unified_marl.algorithms.sadma.native.utils.buffer import ReplayBuffer


def concatenate(episodes,scheme,groups,prep,device):
    length=max(e.max_seq_length for e in episodes)
    result=EpisodeBatch(scheme,groups,len(episodes),length,preprocess=prep,device=device)
    result["avail_actions"][...,0]=1
    for i,episode in enumerate(episodes):
        for name,value in episode.data.transition_data.items():
            result[name][i,:episode.max_seq_length]=value[0]
    return result


def train_sadma(algorithm,env,env_args,device,seed,output_dir,config,overrides,resume):
    from unified_marl.api import TrainResult,_make_model,_write_json,check_device,seed_all
    from types import SimpleNamespace
    payload=read_checkpoint(resume) if resume else None
    if payload:
        if payload["kind"]!="training" or not payload["config"].get("sadma_algorithm"): raise ValueError("SADMA resume requires its full training checkpoint")
        if config is not None: raise ValueError("Use overrides for SADMA resume")
        cfg=merge(payload["config"],dotted_overrides(overrides))
        if algorithm is not None and algorithm!=cfg["algorithm"]: raise ValueError("Cannot change SADMA algorithm during resume")
        if env is not None and env!=cfg["env_config"]: raise ValueError("Cannot change SADMA environment during resume")
        if env_args is not None and merge(cfg["env_args"],env_args)!=cfg["env_args"]: raise ValueError("Cannot change SADMA environment arguments during resume")
        if seed is not None and seed!=cfg["seed"]: raise ValueError("SADMA resume restores RNG; cannot change seed")
        for key in set(cfg)|set(payload["config"]):
            if key not in {"t_max","max_updates"} and cfg.get(key)!=payload["config"].get(key): raise ValueError(f"Cannot change {key} during SADMA resume")
    else:
        cfg=load_config(algorithm,env or "toy",config,overrides)
        cfg["env_args"]=merge(cfg["env_args"],env_args)
        if seed is not None: cfg["seed"]=int(seed)
    cfg["device"]=check_device(device)
    cfg["train_device"]=cfg["device"]
    cfg["use_cuda"]=cfg["device"].startswith("cuda")
    cfg["env_args"]["seed"]=cfg["seed"]
    validate(cfg)
    directory=Path(output_dir or Path.cwd()/"results"/("sadma-"+uuid.uuid4().hex[:8])).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    if any((directory/name).exists() for name in ["config.json","metrics.jsonl","policy.pt","training.pt"]): raise FileExistsError(directory)
    if cfg["env"]=="cityflow": cfg["env_args"].setdefault("log_path",str(directory/"environment"))
    seed_all(cfg["seed"])
    old_threads=torch.get_num_threads()
    torch.set_num_threads(cfg["num_threads"])
    logger=Logger(directory/"metrics.jsonl")
    runner=None
    try:
        runner=EpisodeRunner(SimpleNamespace(**cfg),logger)
        info=runner.get_env_info()
        args,scheme,groups,prep,mac,learner=_make_model(cfg,info,logger)
        runner.args=args
        runner.setup(scheme,groups,prep,mac)
        fresh=cfg["sadma_algorithm"]=="ppo"
        # Native replay stores source fields. Canonical field/time conversion is
        # completed once before insertion; no arrival-time shift is repeated.
        native_scheme={name:{"vshape":tuple(tensor.shape[1:]),"dtype":tensor.dtype} for name,tensor in native_batch(EpisodeBatch(scheme,groups,1,info["episode_limit"]+1,preprocess=prep)).items()}
        replay=None if fresh else ReplayBuffer(args,native_scheme)
        progress={"updates":0,"env_steps":0,"episodes":0,"last_update_episode":0}
        if payload:
            if payload["env_info"]!=info: raise ValueError("SADMA environment metadata differs from checkpoint")
            restore_learner(learner,payload["learner"])
            progress.update(payload["progress"])
            if replay is not None:
                state=payload["sadma_replay"]
                for name,value in state["datas"].items(): replay.datas[name].copy_(value)
                replay.index_count=state["index_count"]
                replay.episodes_in_buffer=state["episodes_in_buffer"]
            restore_rng(payload["rng"])
            runner.t_env=progress["env_steps"]
        before=[p.detach().clone() for p in mac.parameters()]
        _write_json(directory/"config.json",cfg)
        pending=[]
        metrics={}
        while runner.t_env<cfg["t_max"] and (cfg["max_updates"] is None or progress["updates"]<cfg["max_updates"]):
            episode=runner.run()
            progress["env_steps"]=runner.t_env
            progress["episodes"]+=1
            if fresh: pending.append(episode)
            else:
                padded=concatenate([episode],scheme,groups,prep,"cpu")
                if padded.max_seq_length!=info["episode_limit"]+1:
                    padded_full=EpisodeBatch(scheme,groups,1,info["episode_limit"]+1,preprocess=prep)
                    padded_full["avail_actions"][...,0]=1
                    for key,value in padded.data.transition_data.items(): padded_full[key][:,:padded.max_seq_length]=value
                    padded=padded_full
                replay.insert_episodes(1,native_batch(padded))
            enough=len(pending)>=cfg["train_batch_size"] if fresh else replay.can_sample()
            if not enough or progress["episodes"]-progress["last_update_episode"]<cfg["update_rate"]: continue
            if fresh:
                batch=concatenate(pending,scheme,groups,prep,device)
                # Every episode was collected by one unchanged policy; all are
                # consumed once. mini_epochs reuse this batch only inside PPO.
                old_size=args.train_batch_size
                args.train_batch_size=batch.batch_size
                metrics=learner.update(batch,UpdateContext(runner.t_env,progress["episodes"],progress["updates"]))
                args.train_batch_size=old_size
                pending.clear()
            else:
                sampled=replay.sample()
                native_metrics=learner.native.train(sampled,progress["episodes"],runner.t_env)
                metrics={key:float(value.detach().cpu()) if isinstance(value,torch.Tensor) else float(value) for key,value in native_metrics.items()}
                metrics["loss"]=float(learner.native._integration_loss)
                if not all(np.isfinite(v) for v in metrics.values()): raise FloatingPointError("Non-finite SADMA update")
            progress["last_update_episode"]=progress["episodes"]
            progress["updates"]+=1
            logger.record(metrics,runner.t_env)
        if not metrics: raise RuntimeError("SADMA budget ended without an update; increase t_max or reduce train_batch_size/update_rate")
        delta=sum(float((a-b.detach()).square().sum()) for a,b in zip(before,mac.parameters()))**0.5
        policy,training=save_checkpoints(directory,cfg,info,learner,None,progress)
        saved=read_checkpoint(training)
        saved["sadma_replay"]=None if fresh else dict(datas=replay.datas,index_count=replay.index_count,episodes_in_buffer=replay.episodes_in_buffer)
        saved["pending_rollouts"]=len(pending)
        saved["resume_scope"]="episode boundary after completed update; pending unconsumed PPO episodes are discarded; learner/replay/global RNG restored, simulator state excluded"
        atomic_save(saved,training)
        result=TrainResult(cfg["algorithm"],str(directory),policy,training,str(directory/"metrics.jsonl"),str(directory/"config.json"),runner.t_env,progress["episodes"],progress["updates"],delta,metrics)
        _write_json(directory/"result.json",asdict(result))
        return result
    finally:
        try:
            if runner is not None: runner.close_env()
        finally:
            torch.set_num_threads(old_threads)
