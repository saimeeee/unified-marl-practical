import copy
from dataclasses import asdict
import json
import multiprocessing
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import numpy as np
import pytest
import torch
from unified_marl import train,evaluate,load_policy
from unified_marl.algorithms.sadma.adapter import native_batch
from unified_marl.checkpoints import read_checkpoint
from unified_marl.execution.batches import STEP_FIELDS,step_digest,validate_episode,validate_transition
from unified_marl.execution.inference import InferenceEngine,snapshot
from unified_marl.execution.protocol import decode,digest,encode,RPCClient,zmq_module
from unified_marl.execution.settings import ZMQSettings
from unified_marl.trainers.sadma import concatenate
from unified_marl.learners.base import UpdateContext
from test_sadma import setup

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/"docs/evidence/stage6"

@pytest.fixture(autouse=True)
def cpu():
    threads=torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)

def options(**extra):
    return {"train_batch_size":2,"train_buffer_size":8,"update_rate":1,"hidden_dim":16,"mini_epochs":2,"t_max":40,"max_updates":2,
            "zmq.num_sampling_workers":1,"zmq.startup_timeout_s":40,"zmq.timeout_ms":8000,**extra}

def record(name,value):
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    (EVIDENCE/(name+".json")).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")

def released(report):
    zmq=zmq_module()
    context=zmq.Context()
    sockets=[]
    try:
        for address in [report["trainer_endpoint"],*report["inference_endpoints"]]:
            if not address: continue
            handle=context.socket(zmq.REP)
            handle.setsockopt(zmq.LINGER,0)
            handle.bind(address)
            sockets.append(handle)
    finally:
        for handle in sockets: handle.close(0)
        context.term()

@pytest.mark.parametrize("dtype",[torch.float32,torch.float64,torch.int32,torch.int64,torch.uint8,torch.bool])
def test_wire_roundtrip_without_pickle(dtype):
    value=torch.tensor([[0,1],[1,0]],dtype=dtype)
    message=dict(kind="EXAMPLE",worker_id=0,nested={"array":value,"nothing":None},numbers=[1,2.5,True])
    frames=encode(message)
    result=decode(frames)
    torch.testing.assert_close(result["nested"]["array"],value,atol=0,rtol=0)
    assert result["numbers"]==message["numbers"] and result["nested"]["nothing"] is None
    assert frames[0].startswith(b'{"protocol":1')
    scalar=torch.tensor(1,dtype=dtype)
    torch.testing.assert_close(decode(encode(dict(scalar=scalar)))["scalar"],scalar,atol=0,rtol=0)

@pytest.mark.parametrize("bad",["protocol","shape","length","dtype","nonfinite","extra_frame","oversize"])
def test_malformed_wire_rejected(bad):
    frames=encode(dict(value=torch.ones(2,3)))
    header=json.loads(frames[0])
    if bad=="protocol": header["protocol"]=99
    elif bad=="shape": header["message"]["value"]["shape"]=[9999999999,3]
    elif bad=="length": frames[1]=b"x"
    elif bad=="dtype": header["message"]["value"]["dtype"]="object"
    elif bad=="nonfinite": frames[1]=np.full((2,3),np.nan,dtype=np.float32).tobytes()
    elif bad=="extra_frame": frames.append(b"extra")
    frames[0]=json.dumps(header).encode()
    with pytest.raises(ValueError): decode(frames,limit=1 if bad=="oversize" else 1024*1024)

@pytest.mark.parametrize("settings",[{"timeout_ms":0},{"hwm":0},{"startup_timeout_s":float("nan")},{"shutdown_timeout_s":float("inf")},
    {"num_sampling_workers":0},{"role":"trainer"},{"bind_endpoint":"ipc://socket"},{"bind_endpoint":"tcp://0.0.0.0:12000"}])
def test_execution_configuration_limits(settings):
    with pytest.raises(ValueError): ZMQSettings.parse(settings)

@pytest.mark.parametrize("name",["sadma/qmix","sadma/ppo"])
def test_inference_snapshot_outputs_equal_source_controller(name):
    runner,args,scheme,groups,prep,mac,learner=setup(name)
    try: batch=runner.run()
    finally: runner.close_env()
    config=vars(args).copy()
    engine=InferenceEngine(config,runner.get_env_info())
    state=decode(encode(snapshot(learner)))
    engine.synchronize(state)
    n,h=args.n_agents,args.hidden_dim
    data=dict(obs=batch["obs"][:,0],state=batch["state"][:,0],available_actions=batch["avail_actions"][:,0],hidden_states=torch.zeros(1,n,h),last_actions=torch.zeros(1,n,args.n_actions))
    if name.endswith("ppo"): data["critic_hidden"]=torch.zeros(1,h)
    torch.manual_seed(123)
    actual=engine.infer(decode(encode(dict(data=data)))["data"],0)
    torch.manual_seed(123)
    expected=mac.native.select_actions({k:v for k,v in data.items() if k not in {"state","critic_hidden"}},0)
    for key in expected: torch.testing.assert_close(actual["data"][key],expected[key],atol=0,rtol=0)
    bad=dict(state,version=1)
    with pytest.raises(ValueError,match="snapshot"): engine.synchronize(bad)
    record("inference-parity-"+name.split("/")[1],dict(result="PASS",source="Stage5 source-verified SADMA controller",output_max_abs_error=0,version=actual["version"],digest=actual["digest"]))

@pytest.mark.parametrize("mutation",["actions","probability","reward_shift","terminal_shift","dtype","filled","version"])
def test_episode_trace_rejects_corrupt_or_misaligned_data(mutation):
    runner,args,scheme,groups,prep,mac,learner=setup()
    try: batch=runner.run()
    finally: runner.close_env()
    tensors={key:value.clone() for key,value in batch.data.transition_data.items()}
    trace=[step_digest({key:batch[key][:,t] for key in STEP_FIELDS if key in batch.scheme}) for t in range(5)]
    transitions=[digest({key:batch[key][:,t] for key in ["reward","terminated","truncated"]}) for t in range(4)]
    validate_episode(tensors,4,scheme,groups,prep,runner.get_env_info(),trace,transitions)
    if mutation=="actions": tensors["actions"][0,0,0]=2
    elif mutation=="probability": tensors["action_log_probs"][0,0,0]+=0.1
    elif mutation=="reward_shift": tensors["reward"][0,1]+=0.5
    elif mutation=="terminal_shift": tensors["terminated"][0,1]=1
    elif mutation=="dtype": tensors["actions"]=tensors["actions"].float()
    elif mutation=="filled": tensors["filled"][0,1]=0
    else: tensors["policy_version"][0,0]=999
    with pytest.raises(ValueError): validate_episode(tensors,4,scheme,groups,prep,runner.get_env_info(),trace,transitions)

@pytest.mark.parametrize("name,mode,extra",[
 ("sadma/qmix","sync",{"zmq.num_sampling_workers":2}),("sadma/ppo","sync",{}),("sadma/qmix","async",{}),
 ("sadma/ppo","async",{"use_vtrace":True}),("sadma/vdn","sync",{}),("sadma_yi/qmix","async",{}),("sadma_yi_closed/ppo","sync",{})])
def test_managed_training_modes_and_port_release(tmp_path,name,mode,extra):
    before={child.pid for child in multiprocessing.active_children()}
    result=train(name,"toy",execution="zmq",output_dir=tmp_path,overrides=options(**{"zmq.mode":mode,**extra}))
    report=json.loads((tmp_path/"execution.json").read_text(encoding="utf-8"))
    assert result.updates==2 and result.parameter_delta_l2>0 and np.isfinite(result.metrics["loss"])
    assert report["status"]=="PASS" and not report["lifecycle"]["forced"]
    assert all(not child["alive"] and child["exitcode"]==0 for child in report["lifecycle"]["processes"])
    counters=report["counters"]
    assert counters["env_steps"]==counters["completed_steps"]==4*counters["episodes"]
    assert counters["inference_calls"]==5*counters["episodes"]
    assert counters["episodes_accepted"]==counters["episodes"]
    assert sum(item["calls"] for item in report["inference_stop_reports"])==counters["inference_calls"]
    assert report["sync_acknowledgements"][-1]["version"]>0
    assert len({item["digest"] for item in report["sync_acknowledgements"]})>=3
    assert evaluate(result.checkpoint,episodes=1).episodes==1
    load_policy(result.checkpoint).reset()
    assert {child.pid for child in multiprocessing.active_children()}==before
    released(report)
    record("managed-"+name.replace("/","-")+"-"+mode,dict(result="PASS",train=asdict(result),report=report,ports_rebound=True))

def test_two_inference_workers_and_resume(tmp_path):
    first=train("sadma/qmix","toy",execution="zmq",output_dir=tmp_path/"first",overrides=options(**{"zmq.num_sampling_workers":2,"zmq.num_inference_workers":2,"zmq.mode":"async"}))
    report=json.loads((Path(first.output_dir)/"execution.json").read_text(encoding="utf-8"))
    acknowledgements=report["sync_acknowledgements"]
    versions={item["version"] for item in acknowledgements}
    for version in versions:
        rows=[row for row in acknowledgements if row["version"]==version]
        assert {row["worker"] for row in rows}=={0,1} and len({row["digest"] for row in rows})==1
    continued=train(resume=first.training_checkpoint,execution="zmq",output_dir=tmp_path/"continued",overrides={"max_updates":3})
    assert continued.updates==3 and continued.parameter_delta_l2>0
    released(report)
    released(json.loads((Path(continued.output_dir)/"execution.json").read_text(encoding="utf-8")))
    record("multi-inference-resume",dict(result="PASS",first=asdict(first),continued=asdict(continued),all_snapshot_hashes_match=True))

@pytest.mark.parametrize("kind",["raise","stall"])
def test_worker_failure_or_stall_is_reported_and_cleaned(tmp_path,kind):
    started=time.monotonic()
    with pytest.raises((RuntimeError,TimeoutError),match="failed|failure|timeout"):
        train("sadma/qmix","toy",execution="zmq",output_dir=tmp_path,overrides=options(**{"zmq.fault_worker":0,"zmq.fault_kind":kind,"zmq.timeout_ms":1200}))
    elapsed=time.monotonic()-started
    report=json.loads((tmp_path/"execution.json").read_text(encoding="utf-8"))
    assert elapsed<30 and report["status"]=="FAILED" and all(not child["alive"] for child in report["lifecycle"]["processes"])
    released(report)
    record("fault-"+kind,dict(result="PASS",elapsed_seconds=elapsed,report=report,ports_rebound=True))

def unused_endpoint():
    with socket.socket() as server:
        server.bind(("127.0.0.1",0))
        return "tcp://127.0.0.1:"+str(server.getsockname()[1])

def test_connection_timeout_does_not_wait_forever(tmp_path):
    endpoint=unused_endpoint()
    before=torch.get_num_threads()
    started=time.monotonic()
    with pytest.raises(TimeoutError,match="Communication timeout"):
        train("sadma/qmix","toy",execution="zmq",output_dir=tmp_path,overrides=options(**{"zmq.role":"trainer","zmq.inference_endpoints":[endpoint],"zmq.timeout_ms":250}))
    report=json.loads((tmp_path/"execution.json").read_text(encoding="utf-8"))
    assert time.monotonic()-started<5 and torch.get_num_threads()==before
    released(report)
    record("connection-timeout",dict(result="PASS",elapsed_seconds=time.monotonic()-started,report=report,ports_rebound=True))

def test_inference_startup_bind_failure_cleans_own_processes(tmp_path):
    zmq=zmq_module()
    context=zmq.Context()
    occupied=context.socket(zmq.REP)
    occupied.setsockopt(zmq.LINGER,0)
    occupied.bind_to_random_port("tcp://127.0.0.1")
    endpoint=occupied.getsockopt_string(zmq.LAST_ENDPOINT)
    try:
        with pytest.raises(RuntimeError,match="startup failed"):
            train("sadma/qmix","toy",execution="zmq",output_dir=tmp_path,overrides=options(**{"zmq.inference_endpoints":[endpoint]}))
        report=json.loads((tmp_path/"execution.json").read_text(encoding="utf-8"))
        assert all(not row["alive"] for row in report["lifecycle"]["processes"])
        assert occupied.getsockopt(zmq.TYPE)==zmq.REP
    finally: occupied.close(0); context.term()
    record("startup-failure",dict(result="PASS",report=report))

def test_unsupported_algorithm_execution_combinations(tmp_path):
    with pytest.raises(ValueError,match="use_vtrace=true"):
        train("sadma/ppo","toy",execution="zmq",output_dir=tmp_path/"ppo",overrides={"zmq.mode":"async"})
    with pytest.raises(ValueError,match="only for SADMA"):
        train("pymarl/qmix","toy",execution="zmq",output_dir=tmp_path/"base")

def test_context_creation_failure_restores_threads(monkeypatch,tmp_path):
    zmq=zmq_module()
    def failure(*args,**kwargs): raise OSError("context denied")
    monkeypatch.setattr(zmq,"Context",failure)
    before=torch.get_num_threads()
    with pytest.raises(OSError,match="context denied"):
        train("sadma/qmix","toy",execution="zmq",output_dir=tmp_path,overrides=options())
    assert torch.get_num_threads()==before
    assert json.loads((tmp_path/"execution.json").read_text(encoding="utf-8"))["status"]=="FAILED"

def test_separate_cli_roles_on_loopback(tmp_path):
    inference,trainer=unused_endpoint(),unused_endpoint()
    job="test-"+str(time.time_ns())
    base=[sys.executable,"-B","-m","unified_marl"]
    create=getattr(subprocess,"CREATE_NO_WINDOW",0)
    processes=[]
    files=[]
    def start(name,arguments):
        output=(tmp_path/(name+".log")).open("w",encoding="utf-8")
        files.append(output)
        process=subprocess.Popen([*base,*arguments],stdout=output,stderr=subprocess.STDOUT,creationflags=create)
        processes.append(process)
        return process
    try:
        infer=start("infer",["zmq-infer","--endpoint",inference,"--idle-timeout","60"])
        arguments=["train","--algorithm","sadma/qmix","--env","toy","--execution","zmq","--output-dir",str(tmp_path/"run")]
        settings=options(**{"zmq.role":"trainer","zmq.bind_endpoint":trainer,"zmq.inference_endpoints":[inference],"zmq.job_id":job})
        for key,value in settings.items(): arguments.extend(["--set",key+"="+json.dumps(value)])
        learner=start("trainer",arguments)
        deadline=time.monotonic()+45
        while not (tmp_path/"run/execution-started.json").exists():
            if learner.poll() is not None: pytest.fail((tmp_path/"trainer.log").read_text(encoding="utf-8"))
            if time.monotonic()>deadline: pytest.fail("Standalone trainer startup timeout")
            time.sleep(0.05)
        sampler=start("sampler",["zmq-sample","--trainer-endpoint",trainer,"--worker-id","0","--job-id",job])
        assert learner.wait(timeout=45)==0,(tmp_path/"trainer.log").read_text(encoding="utf-8")
        assert sampler.wait(timeout=10)==infer.wait(timeout=10)==0
        report=json.loads((tmp_path/"run/execution.json").read_text(encoding="utf-8"))
        assert report["status"]=="PASS" and report["cross_machine_verified"] is False
        released(report)
        record("separate-roles",dict(result="PASS",cross_machine_verified=False,report=report,ports_rebound=True))
    finally:
        for process in processes:
            if process.poll() is None: process.terminate()
        for process in processes:
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=5)
        for file in files: file.close()

def test_async_vtrace_keeps_old_behavior_probabilities_and_source_loss(monkeypatch):
    from test_sadma import original,adapt_reference
    import importlib
    runner,args,scheme,groups,prep,mac,learner=setup(extra={"use_vtrace":True})
    try: batch=concatenate([runner.run(),runner.run()],scheme,groups,prep,"cpu")
    finally: runner.close_env()
    old_log_prob=batch["action_log_probs"].clone()
    learner.update(batch,UpdateContext(8,2,0))
    version=int(mac.native.agent_version.item())
    assert version==2 and batch["policy_version"].max().item()==0
    with original(monkeypatch):
        module=importlib.import_module("policy.mappo.ppo_controller")
        reference_mac=adapt_reference(module,"PPOController")(args)
        reference_mac.load_agent_state_dict(mac.agent.state_dict())
        reference_mac.agent_version.fill_(version)
        policy=importlib.import_module("policy.mappo.ppo_policy")
        reference=adapt_reference(policy,"PPOPolicy")(args,reference_mac)
        for key,value in vars(learner.native).items():
            if isinstance(value,torch.nn.Module): getattr(reference,key).load_state_dict(value.state_dict())
        reference.optimizer.load_state_dict(copy.deepcopy(learner.native.optimizer.state_dict()))
        reference.lr=learner.native.lr
        losses=[]
        backward=torch.Tensor.backward
        with monkeypatch.context() as context:
            def capture(self,*a,**kw):
                losses.append(float(self.detach()))
                return backward(self,*a,**kw)
            context.setattr(torch.Tensor,"backward",capture)
            reference.train(native_batch(batch),4,16)
        learner.native.train(native_batch(batch),4,16)
        error=abs(losses[-1]-learner.native._integration_loss)
        network_error=max(float((a-b).abs().max()) for a,b in zip(reference_mac.agent.parameters(),mac.agent.parameters()))
        assert error==network_error==0
        torch.testing.assert_close(batch["action_log_probs"],old_log_prob,atol=0,rtol=0)
    record("async-vtrace-source-parity",dict(result="PASS",behavior_version=0,learner_version_before=version,loss_abs_error=error,
        updated_actor_max_abs_error=network_error,old_log_probabilities_preserved=True,source="original Wang SADMA PPOPolicy with existing compatibility fixes only"))

def test_transport_hwm_and_dead_peer_timeout():
    settings=ZMQSettings.parse(dict(hwm=1,timeout_ms=150))
    client=RPCClient(unused_endpoint(),settings)
    try:
        zmq=zmq_module()
        assert client.socket.getsockopt(zmq.SNDHWM)==client.socket.getsockopt(zmq.RCVHWM)==1
        with pytest.raises(TimeoutError): client.call(dict(kind="NO_PEER"))
    finally: client.close()
