"""Bounded ROUTER coordinator/learner, spawn samplers, independent inference roles."""
from collections import deque
from dataclasses import asdict
import json
import multiprocessing as mp
from pathlib import Path
import time
import traceback
from types import SimpleNamespace
import uuid
import numpy as np
import torch
from unified_marl.algorithms.sadma.adapter import native_batch
from unified_marl.algorithms.sadma.native.utils.buffer import ReplayBuffer
from unified_marl.checkpoints import atomic_save,read_checkpoint,restore_learner,restore_rng,save_checkpoints
from unified_marl.components.episode_buffer import EpisodeBatch
from unified_marl.config import load_config,merge,dotted_overrides,validate
from unified_marl.learners.base import UpdateContext
from unified_marl.trainers.sadma import concatenate
from unified_marl.utils.logging import Logger
from .batches import trace_step,validate_episode,validate_transition
from .inference import inference_worker,snapshot
from .protocol import RPCClient,bind,configure,decode,encode,schema_id,zmq_module
from .sampler import sampler_worker
from .settings import ZMQSettings
from .evaluation import PeriodicEvaluator

class Cluster:
    def __init__(self,settings,directory):
        self.settings,self.directory=settings,directory
        self.context=mp.get_context("spawn")
        self.children=[]
        self.events=[]
    def start(self,role,target,args):
        reader,writer=self.context.Pipe(duplex=False)
        process=self.context.Process(target=target,args=(*args,writer),name=f"unified-marl-{role}")
        process.start()
        writer.close()
        item=dict(role=role,process=process,pipe=reader,stopped=False)
        self.children.append(item)
        return item
    def startup(self,item):
        if not item["pipe"].poll(self.settings.startup_timeout_s): raise TimeoutError(f"{item['role']} startup timeout")
        event=item["pipe"].recv()
        if event.get("kind")!="READY": raise RuntimeError(f"{item['role']} startup failed: {event}")
        return event
    def check(self,stopping=False):
        for item in self.children:
            pipe=item["pipe"]
            while not item.get("pipe_eof",False):
                try:
                    if not pipe.poll(): break
                    event=pipe.recv()
                except (EOFError,OSError):
                    item["pipe_eof"]=True
                    break
                self.events.append(dict(role=item["role"],**event))
                if event["kind"]=="STOPPED": item["stopped"]=True
                if event["kind"]=="ERROR" and not stopping: raise RuntimeError(f"{item['role']} failed: {event['error']}\n{event.get('traceback','')}")
            code=item["process"].exitcode
            if code is not None and (code!=0 or not item["stopped"]) and not stopping:
                raise RuntimeError(f"{item['role']} exited unexpectedly with exit code {code}")
    def close(self):
        deadline=time.monotonic()+self.settings.shutdown_timeout_s
        forced=[]
        for item in self.children:
            process=item["process"]
            process.join(max(0,deadline-time.monotonic()))
        self.check(stopping=True)
        for item in self.children:
            process=item["process"]
            if process.is_alive():
                forced.append(dict(role=item["role"],pid=process.pid))
                process.terminate()
                process.join(1)
                if process.is_alive(): process.kill(); process.join(1)
            item["pipe"].close()
        return dict(processes=[dict(role=item["role"],pid=item["process"].pid,exitcode=item["process"].exitcode,alive=item["process"].is_alive()) for item in self.children],
                    forced=forced,events=self.events)

def train_zmq(algorithm,env=None,env_args=None,*,device="cpu",seed=None,output_dir=None,config=None,overrides=None,resume=None):
    from unified_marl.api import TrainResult,_make_model,_write_json,check_device,seed_all
    from unified_marl.envs import make_env
    payload=read_checkpoint(resume) if resume else None
    if payload:
        if payload["kind"]!="training" or not payload["config"].get("sadma_algorithm"): raise ValueError("ZMQ resume requires full SADMA training state")
        if config is not None: raise ValueError("Use overrides to resume ZMQ")
        cfg=merge(payload["config"],dotted_overrides(overrides))
        if algorithm and algorithm!=cfg["algorithm"]: raise ValueError("Cannot change algorithm during ZMQ resume")
        for key in set(cfg)|set(payload["config"]):
            if key not in {"t_max","max_updates","zmq","execution"} and cfg.get(key)!=payload["config"].get(key): raise ValueError(f"Cannot change {key} during ZMQ resume")
        if seed is not None and seed!=cfg["seed"]: raise ValueError("Resume restores learner RNG; cannot change seed")
        if env is not None and env!=cfg["env_config"]: raise ValueError("Cannot change environment during ZMQ resume")
        if env_args is not None and merge(cfg["env_args"],env_args)!=cfg["env_args"]: raise ValueError("Cannot change environment arguments during ZMQ resume")
    else:
        if not algorithm or not algorithm.startswith(("sadma/","sadma_yi/","sadma_yi_closed/")): raise ValueError("ZMQ is implemented only for SADMA Qmix/VDN and PPO bundles")
        cfg=load_config(algorithm,env or "toy",config,overrides)
        cfg["env_args"]=merge(cfg["env_args"],env_args)
        if seed is not None: cfg["seed"]=int(seed)
    settings=ZMQSettings.parse(cfg.get("zmq"))
    if cfg["sadma_algorithm"]=="ppo" and settings.mode=="async" and not cfg.get("use_vtrace"):
        raise ValueError("Asynchronous PPO requires explicit use_vtrace=true; ordinary PPO supports ZMQ sync only")
    if payload and payload.get("execution_mode",settings.mode)!=settings.mode: raise ValueError("Cannot change data lifecycle mode during ZMQ resume")
    cfg.update(device=check_device(device),train_device=check_device(device),execution="zmq",zmq=settings.dictionary(),zmq_mode=settings.mode)
    cfg["env_args"]["seed"]=cfg["seed"]
    validate(cfg)
    directory=Path(output_dir or Path.cwd()/"results"/("zmq-"+uuid.uuid4().hex[:8])).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    if any((directory/name).exists() for name in ["config.json","execution.json","policy.pt","training.pt"]): raise FileExistsError(directory)
    if cfg["env"]=="cityflow": cfg["env_args"].setdefault("log_path",str(directory/"environment"))
    evaluator=PeriodicEvaluator(cfg,directory)
    _write_json(directory/"config.json",cfg)
    zmq=zmq_module()
    seed_all(cfg["seed"])
    threads=torch.get_num_threads()
    torch.set_num_threads(cfg["num_threads"])
    cluster=Cluster(settings,directory)
    context=router=None
    clients=[]
    endpoint=None
    failure=None
    lifecycle=None
    success=False
    summary={"mode":settings.mode,"job_id":settings.job_id,"cross_machine_verified":False,"platform":"Windows spawn tested; other platforms NOT VALIDATED"}
    stats=dict(env_steps=0,episodes=0,updates=0,last_update_episode=0,episodes_accepted=0,episodes_dropped_stale=0,completed_steps=0,inference_calls=0,snapshot_syncs=0,mixed_version_episodes=0)
    events=(directory/"execution-events.jsonl").open("w",encoding="utf-8")
    workers={}
    sync_recent=deque(maxlen=64)
    def event(kind,**values):
        events.write(json.dumps(dict(kind=kind,**values),ensure_ascii=False,allow_nan=False)+"\n")
        events.flush()
    def reply(identity,request,kind,**values):
        router.send_multipart([identity,*encode(dict(kind=kind,request_id=request.get("request_id"),**values),settings.max_message_bytes)])
    try:
        context=zmq.Context()
        router=context.socket(zmq.ROUTER)
        configure(router,settings)
        endpoint=bind(router,settings.bind_endpoint)
        advertised=settings.trainer_endpoint or endpoint
        probe=make_env(cfg["env"],**cfg["env_args"])
        try: info=probe.get_env_info()
        finally: probe.close()
        logger=Logger(directory/"metrics.jsonl")
        args,scheme,groups,prep,mac,learner=_make_model(cfg,info,logger)
        fresh=cfg["sadma_algorithm"]=="ppo" and settings.mode=="sync"
        full=EpisodeBatch(scheme,groups,1,info["episode_limit"]+1,preprocess=prep)
        replay=None if fresh else ReplayBuffer(args,{k:dict(vshape=tuple(v.shape[1:]),dtype=v.dtype) for k,v in native_batch(full).items()})
        if payload:
            if payload["env_info"]!=info: raise ValueError("ZMQ environment dimensions differ from checkpoint")
            restore_learner(learner,payload["learner"])
            state=payload.get("sadma_replay")
            if replay is not None and state is not None:
                for key,value in state["datas"].items(): replay.datas[key].copy_(value)
                replay.index_count,replay.episodes_in_buffer=state["index_count"],state["episodes_in_buffer"]
            stats.update({key:value for key,value in payload["progress"].items() if key in stats})
            restore_rng(payload["rng"])
        initial=[p.detach().clone() for p in mac.parameters()]
        schema=schema_id(cfg,info)
        active_digest={"value":None}
        for i in range(settings.num_inference_workers):
            address=settings.inference_endpoints[i] if settings.inference_endpoints else "tcp://127.0.0.1:0"
            if settings.role=="managed":
                child=cluster.start(f"infer-{i}",inference_worker,(address,settings.dictionary()))
                address=cluster.startup(child)["endpoint"]
            client=RPCClient(address,settings)
            clients.append(client)
        def synchronize(initializing=False):
            state=snapshot(learner)
            for i,client in enumerate(clients):
                body=dict(kind="INIT" if initializing else "SYNC",job_id=settings.job_id,schema_id=schema,snapshot=state)
                if initializing: body.update(config=cfg,env_info=info,seed=cfg["seed"]+(100+i)*settings.seed_stride)
                response=client.call(body)
                if response.get("version")!=state["version"] or response.get("digest")!=state["digest"]: raise ValueError("Inference worker did not acknowledge the exact parameter snapshot")
                ack=dict(worker=i,version=state["version"],digest=state["digest"])
                sync_recent.append(ack)
                stats["snapshot_syncs"]+=1
                event("SYNC",**ack)
            active_digest["value"]=state["digest"]
        synchronize(True)
        if settings.role=="managed":
            for i in range(settings.num_sampling_workers): cluster.start(f"sampler-{i}",sampler_worker,(advertised,i,settings.dictionary()))
        _write_json(directory/"execution-started.json",dict(trainer_pid=__import__("os").getpid(),trainer_endpoint=advertised,inference_endpoints=[client.endpoint for client in clients],
            job_id=settings.job_id,children=[dict(role=item["role"],pid=item["process"].pid) for item in cluster.children]))
        waiting={}
        round_done=set()
        pending=[]
        draining=False
        stop_started=None
        started=time.monotonic()
        last_activity=started
        metrics={}
        def budget(): return stats["env_steps"]>=cfg["t_max"] or (cfg["max_updates"] is not None and stats["updates"]>=cfg["max_updates"])
        def grant(worker,request):
            if draining:
                reply(worker["identity"],request,"STOP")
                return
            session=dict(id=uuid.uuid4().hex,next_t=0,trace=[],transitions=[],versions=[],previous=None,bootstrap=False)
            worker["session"]=session
            reply(worker["identity"],request,"COLLECT",session=session["id"],version=int(mac.native.agent_version.item()))
        def update():
            if draining or stats["episodes"]-stats["last_update_episode"]<cfg["update_rate"]: return
            if fresh:
                if len(pending)<cfg["train_batch_size"]: return
                batch=concatenate(pending,scheme,groups,prep,device)
                metrics_now=learner.update(batch,UpdateContext(stats["env_steps"],stats["episodes"],stats["updates"]))
                pending.clear()
            else:
                if not replay.can_sample(): return
                batch=replay.sample()
                if settings.max_policy_lag is not None:
                    values=batch["policy_version"][batch["filled"].bool()]
                    if values.min().item()<int(mac.native.agent_version.item())-settings.max_policy_lag:
                        event("REPLAY_SKIPPED_STALE",version=int(mac.native.agent_version.item()))
                        return
                source=learner.native.train(batch,stats["episodes"],stats["env_steps"])
                metrics_now={k:float(v.detach().cpu()) if isinstance(v,torch.Tensor) else float(v) for k,v in source.items()}
                metrics_now["loss"]=float(learner.native._integration_loss)
                if not all(np.isfinite(v) for v in metrics_now.values()): raise FloatingPointError("Nonfinite distributed learner metrics")
            metrics.clear(); metrics.update(metrics_now)
            stats["updates"]+=1
            stats["last_update_episode"]=stats["episodes"]
            logger.record(metrics,stats["env_steps"])
            synchronize()
            event("UPDATE",updates=stats["updates"],env_steps=stats["env_steps"],version=int(mac.native.agent_version.item()))
            evaluator.submit(lambda: save_checkpoints(directory/f"evaluation-snapshot-{stats['updates']}",cfg,info,learner,None,stats)[0],
                             stats["updates"],int(mac.native.agent_version.item()))
        while True:
            cluster.check()
            evaluator.poll()
            if not draining and budget() and all(w["session"] is None for w in workers.values()):
                draining=True
                stop_started=time.monotonic()
                for identity,request in list(waiting.items()): grant(workers[identity],request)
                waiting.clear()
            if draining and len(workers)==settings.num_sampling_workers and all(w["closed"] for w in workers.values()): break
            now=time.monotonic()
            startup_complete=len(workers)==settings.num_sampling_workers and all(w["ready"] for w in workers.values())
            if not startup_complete and now-started>settings.startup_timeout_s: raise TimeoutError("Samplers did not initialize before startup deadline")
            if startup_complete and now-last_activity>settings.timeout_ms/1000: raise TimeoutError("Sampler communication timeout; no progress within configured deadline")
            for w in workers.values():
                if w["session"] is not None and now-w["last_seen"]>settings.timeout_ms/1000: raise TimeoutError(f"Sampler {w['id']} communication timeout")
            if draining and now-stop_started>settings.shutdown_timeout_s: raise TimeoutError("Sampling drain/stop timeout")
            if not router.poll(20,zmq.POLLIN): continue
            frames=router.recv_multipart()
            identity=frames[0]
            request=decode(frames[1:],settings.max_message_bytes)
            last_activity=time.monotonic()
            try:
                kind=request.get("kind")
                if request.get("job_id")!=settings.job_id: raise ValueError("Job id mismatch")
                sequence=request.get("request_id")
                if type(sequence) is not int or sequence<1: raise ValueError("Invalid request id")
                if kind=="HELLO":
                    index=request.get("worker_id")
                    if type(index) is not int or not 0<=index<settings.num_sampling_workers or identity in workers or any(w["id"]==index for w in workers.values()): raise ValueError("Unknown or duplicate sampler id")
                    workers[identity]=dict(id=index,identity=identity,sequence=sequence,session=None,ready=False,closed=False,last_seen=time.monotonic())
                    reply(identity,request,"WELCOME",config=cfg,env_info=info,schema_id=schema)
                    continue
                if identity not in workers: raise ValueError("Sampler is not registered")
                worker=workers[identity]
                if request.get("worker_id")!=worker["id"] or sequence!=worker["sequence"]+1 or request.get("schema_id")!=schema: raise ValueError("Sampler sequence/schema mismatch")
                worker["sequence"]=sequence
                worker["last_seen"]=time.monotonic()
                if kind=="ENV_READY":
                    if worker["ready"] or request["env_info"]!=info or request["seed"]!=cfg["seed"]+(worker["id"]+1)*settings.seed_stride: raise ValueError("Sampler environment metadata/seed mismatch")
                    worker["ready"]=True
                    event("SAMPLER_READY",worker=worker["id"],seed=request["seed"])
                    reply(identity,request,"READY_ACK")
                elif kind=="READY":
                    if not worker["ready"] or worker["session"] is not None: raise ValueError("Sampler requested a permit with an active/unready session")
                    if settings.mode=="sync" and worker["id"] in round_done and not draining: waiting[identity]=request
                    else: grant(worker,request)
                elif kind=="INFER":
                    session=worker["session"]
                    t=request.get("t")
                    if session is None or request.get("session")!=session["id"] or type(t) is not int or t!=session["next_t"] or t>info["episode_limit"] or session["bootstrap"]: raise ValueError("Invalid inference session/time order")
                    if type(request.get("bootstrap")) is not bool or (t==0 and request["bootstrap"]): raise ValueError("Invalid bootstrap marker")
                    data=request["data"]
                    previous=session["previous"]
                    expected=torch.zeros(1,info["n_agents"],cfg["hidden_dim"]) if previous is None else previous["hidden_states"]
                    last=torch.zeros(1,info["n_agents"],info["n_actions"]) if previous is None else torch.nn.functional.one_hot(previous["actions"],info["n_actions"]).float()
                    if not torch.equal(data["hidden_states"],expected) or not torch.equal(data["last_actions"],last): raise ValueError("Recurrent state/last action does not match prior inference")
                    if cfg["sadma_algorithm"]=="ppo":
                        shape=(1,info["n_agents"],cfg["hidden_dim"]) if "independent_obs" in cfg["critic_inputs"] else (1,cfg["hidden_dim"])
                        expected_critic=torch.zeros(shape) if previous is None else previous["critic_hidden"]
                        if not torch.equal(data["critic_hidden"],expected_critic): raise ValueError("Critic recurrent state does not match prior inference")
                    if t>0:
                        session["transitions"].append(validate_transition(request.get("transition"),info,cfg.get("use_individual_rewards",False),request["bootstrap"]))
                        stats["env_steps"]+=1
                    elif request.get("transition") is not None: raise ValueError("Initial observation cannot acknowledge a previous transition")
                    infer_index=worker["id"]%len(clients)
                    response=clients[infer_index].call(dict(kind="INFER",job_id=settings.job_id,schema_id=schema,data=data,steps=stats["env_steps"]))
                    if response.get("version")!=int(mac.native.agent_version.item()) or response.get("digest")!=active_digest["value"]: raise ValueError("Inference parameters are not current")
                    session["trace"].append(trace_step(data,response))
                    session["versions"].append(response["version"])
                    session["previous"]=response["data"]
                    session["next_t"]+=1
                    session["bootstrap"]=request["bootstrap"]
                    stats["inference_calls"]+=1
                    event("INFER",worker=worker["id"],t=t,version=response["version"],digest=response["digest"],inference_worker=infer_index)
                    reply(identity,request,"ACTIONS",data=response["data"],version=response["version"],digest=response["digest"])
                elif kind=="EPISODE":
                    session=worker["session"]
                    if session is None or request.get("session")!=session["id"] or not session["bootstrap"]: raise ValueError("Episode has no completed inference session")
                    batch=validate_episode(request["tensors"],request["length"],scheme,groups,prep,info,session["trace"],session["transitions"])
                    versions=session["versions"]
                    current=int(mac.native.agent_version.item())
                    if max(versions)>current: raise ValueError("Future policy version in sample")
                    if settings.mode=="sync" and min(versions)!=max(versions): raise ValueError("Synchronous episode mixed policy versions")
                    stats["episodes"]+=1
                    stats["completed_steps"]+=request["length"]
                    stats["mixed_version_episodes"]+=int(min(versions)!=max(versions))
                    stale=settings.max_policy_lag is not None and current-min(versions)>settings.max_policy_lag
                    if stale: stats["episodes_dropped_stale"]+=1
                    else:
                        stats["episodes_accepted"]+=1
                        if fresh: pending.append(batch)
                        else:
                            padded=EpisodeBatch(scheme,groups,1,info["episode_limit"]+1,preprocess=prep)
                            padded["avail_actions"][...,0]=1
                            for key,value in batch.data.transition_data.items(): padded[key][:,:batch.max_seq_length]=value
                            replay.insert_episodes(1,native_batch(padded))
                    worker["session"]=None
                    round_done.add(worker["id"])
                    event("EPISODE",worker=worker["id"],length=request["length"],min_version=min(versions),max_version=max(versions),accepted=not stale)
                    if settings.mode=="async" or len(round_done)==settings.num_sampling_workers:
                        update()
                        round_done.clear()
                        # Budget is checked before granting another cohort.
                        if budget(): draining=True; stop_started=time.monotonic()
                        for route,waiting_request in list(waiting.items()): grant(workers[route],waiting_request)
                        waiting.clear()
                    reply(identity,request,"EPISODE_ACK",accepted=not stale)
                elif kind=="BYE":
                    if request.get("error"): raise RuntimeError(f"Sampler {worker['id']} failed: {request['error']}")
                    worker["closed"]=True
                    worker["session"]=None
                    event("SAMPLER_CLOSED",worker=worker["id"],env_closed=request.get("env_closed"))
                    reply(identity,request,"BYE_ACK")
                else: raise ValueError("Unknown trainer message")
            except BaseException as exc:
                reply(identity,request,"ERROR",error=f"{type(exc).__name__}: {exc}")
                raise
        evaluator.finish()
        if not metrics: raise RuntimeError("ZMQ training budget ended without an effective parameter update")
        delta=sum(float((a-b.detach()).square().sum()) for a,b in zip(initial,mac.parameters()))**0.5
        if not np.isfinite(delta): raise FloatingPointError("Nonfinite distributed parameter change")
        policy,training=save_checkpoints(directory,cfg,info,learner,None,stats)
        saved=read_checkpoint(training)
        saved["sadma_replay"]=None if fresh else dict(datas=replay.datas,index_count=replay.index_count,episodes_in_buffer=replay.episodes_in_buffer)
        saved["execution_mode"]=settings.mode
        saved["resume_scope"]="Learner/replay/RNG at drained episode boundary; restarted worker RNG/scheduling and external simulator state are not exact replay"
        atomic_save(saved,training)
        result=TrainResult(cfg["algorithm"],str(directory),policy,training,str(directory/"metrics.jsonl"),str(directory/"config.json"),stats["env_steps"],stats["episodes"],stats["updates"],delta,metrics)
        _write_json(directory/"result.json",asdict(result))
        success=True
    except BaseException as exc:
        failure=dict(error=f"{type(exc).__name__}: {exc}",traceback=traceback.format_exc())
        raise
    finally:
        # Answer pending sampler calls before closing ROUTER; no worker can
        # indefinitely keep a REQ/DEALER call or an environment alive locally.
        deadline=time.monotonic()+min(settings.shutdown_timeout_s,2.0)
        while time.monotonic()<deadline and any(item["role"].startswith("sampler") and item["process"].is_alive() for item in cluster.children):
            if router.poll(20,zmq.POLLIN):
                frames=router.recv_multipart()
                try:
                    request=decode(frames[1:],settings.max_message_bytes)
                    reply(frames[0],request,"STOP")
                except Exception: pass
        inference_stops=[]
        for client in clients:
            try: inference_stops.append(client.call(dict(kind="STOP",job_id=settings.job_id,schema_id=locals().get("schema")),timeout_ms=min(settings.timeout_ms,1000)))
            except Exception: pass
            finally: client.close()
        if router is not None: router.close(0)
        if context is not None: context.term()
        lifecycle=cluster.close()
        evaluation_lifecycle=evaluator.close()
        events.close()
        summary.update(status="PASS" if success else "FAILED",error=failure,trainer_endpoint=endpoint,inference_endpoints=[client.endpoint for client in clients],
                       counters=stats,sync_acknowledgements=list(sync_recent),lifecycle=lifecycle,evaluation=evaluation_lifecycle,
                       partial_confirmed_steps=stats["env_steps"]-stats["completed_steps"],samplers_closed=sum(w["closed"] for w in workers.values()),inference_stop_reports=inference_stops)
        _write_json(directory/"execution.json",summary)
        torch.set_num_threads(threads)
    return result
