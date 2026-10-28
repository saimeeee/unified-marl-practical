"""Installed-wheel, outside-source verification of local TCP distributed execution."""
from dataclasses import asdict
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid
import numpy as np
import zmq
import unified_marl
from unified_marl import train,evaluate,load_policy
from unified_marl.envs.toy import CoordinationEnv

def main():
    root=Path(__file__).resolve().parents[1]
    assert Path.cwd().resolve()!=root and "site-packages" in unified_marl.__file__ and unified_marl.__version__=="0.6.0"
    evidence=root/"docs/evidence/stage6"
    runs=evidence/("installed-runs-"+uuid.uuid4().hex[:8])
    rows=[]
    options={"train_batch_size":2,"train_buffer_size":8,"update_rate":1,"hidden_dim":16,"mini_epochs":2,"t_max":40,"max_updates":2,"zmq.num_sampling_workers":1}
    children={child.pid for child in multiprocessing.active_children()}
    def ports(report):
        context=zmq.Context()
        handles=[]
        try:
            for endpoint in [report["trainer_endpoint"],*report["inference_endpoints"]]:
                handle=context.socket(zmq.REP)
                handle.setsockopt(zmq.LINGER,0)
                handle.bind(endpoint)
                handles.append(handle)
        finally:
            for handle in handles: handle.close(0)
            context.term()
    names=["sadma/qmix","sadma/ppo","sadma/vdn","sadma_yi/qmix","sadma_yi/ppo","sadma_yi_closed/qmix","sadma_yi_closed/ppo"]
    cases=[(name,"sync",{}) for name in names]+[("sadma/qmix","async",{"zmq.num_sampling_workers":2,"zmq.num_inference_workers":2}),
                                                ("sadma/ppo","async",{"use_vtrace":True,"zmq.num_sampling_workers":2})]
    for name,mode,extra in cases:
        result=train(name,"toy",execution="zmq",output_dir=runs/str(len(rows)),overrides={**options,"zmq.mode":mode,**extra})
        assert result.updates==2 and result.parameter_delta_l2>0
        report=json.loads((Path(result.output_dir)/"execution.json").read_text(encoding="utf-8"))
        assert report["status"]=="PASS" and report["lifecycle"]["forced"]==[]
        assert all(not child["alive"] and child["exitcode"]==0 for child in report["lifecycle"]["processes"])
        assert report["counters"]["env_steps"]==report["counters"]["completed_steps"]
        assert sum(item["calls"] for item in report["inference_stop_reports"])==report["counters"]["inference_calls"]
        ports(report)
        policy=load_policy(result.checkpoint)
        env=CoordinationEnv()
        obs,available=np.asarray(env.get_obs()),np.asarray(env.get_avail_actions())
        actions=policy.act(obs,available)
        assert available[np.arange(2),actions].all()
        policy.reset()
        np.testing.assert_array_equal(actions,policy.act(obs,available))
        env.close()
        score=evaluate(result.checkpoint,episodes=2)
        rows.append(dict(algorithm=name,mode=mode,result=asdict(result),evaluation=asdict(score),execution=report,ports_rebound=True))
        print(name,mode,"PASS",flush=True)
    continued=train(resume=rows[0]["result"]["training_checkpoint"],execution="zmq",output_dir=runs/"resumed",overrides={"max_updates":3})
    assert continued.updates==3 and continued.parameter_delta_l2>0
    ports(json.loads((Path(continued.output_dir)/"execution.json").read_text(encoding="utf-8")))
    cli=[]
    for name in ["sadma/qmix","sadma/ppo"]:
        arguments=[str(Path(sys.executable).with_name("unified-marl.exe")),"train","--algorithm",name,"--env","toy","--execution","zmq","--output-dir",str(runs/("cli-"+name.replace("/","-")))]
        for key,value in options.items(): arguments.extend(["--set",key+"="+json.dumps(value)])
        process=subprocess.run(arguments,text=True,capture_output=True,timeout=60,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        assert process.returncode==0,process.stderr
        cli.append(dict(algorithm=name,returncode=process.returncode,stdout=process.stdout,stderr=process.stderr))
    failures=[]
    for kind in ["worker_failure","connection_timeout"]:
        directory=runs/kind
        extra={"zmq.fault_worker":0} if kind=="worker_failure" else {"zmq.role":"trainer","zmq.timeout_ms":250}
        if kind=="connection_timeout":
            with socket.socket() as temporary:
                temporary.bind(("127.0.0.1",0))
                extra["zmq.inference_endpoints"]=["tcp://127.0.0.1:"+str(temporary.getsockname()[1])]
        started=time.monotonic()
        try:
            train("sadma/qmix","toy",execution="zmq",output_dir=directory,overrides={**options,**extra})
            raise AssertionError("Expected execution failure was not reported")
        except (RuntimeError,TimeoutError) as exc:
            report=json.loads((directory/"execution.json").read_text(encoding="utf-8"))
            assert report["status"]=="FAILED" and all(not child["alive"] for child in report["lifecycle"]["processes"])
            ports(report)
            failures.append(dict(case=kind,error=str(exc),elapsed_seconds=time.monotonic()-started,report=report,ports_rebound=True))
    regressions=[]
    for name,settings in [
        ("pymarl/qmix",dict(batch_size=2,buffer_size=8,t_max=16,max_updates=2)),
        ("sadma/ppo",dict(train_batch_size=2,train_buffer_size=8,update_rate=1,hidden_dim=16,t_max=40,max_updates=2)),
        ("wang/rode_hmf",dict(batch_size=2,buffer_size=8,t_max=16,max_updates=2,role_interval=2)),
        ("wang_hmf/mfq",dict(batch_size=4,buffer_size=32,hidden_dim=16,n_heads=2,embed_dim=8,t_max=16,max_updates=2,hmf_norm_rewards=False,hmf_steps_per_update=1,hmf_num_batch=1))]:
        result=train(name,"toy",output_dir=runs/("regression-"+name.replace("/","-")),overrides=settings,env_args={"n_agents":4,"mask_actions":False} if name.startswith("wang_hmf") else None)
        assert result.parameter_delta_l2>0
        regressions.append(dict(name=name,result=asdict(result)))
    data=root.parent/"marl平台代码/寇谦/论文代码及技术文档/论文代码及技术文档/MADiversePreferDatasets/3m/3m.th"
    for name in ["kou/bc","kou/cpl"]:
        result=train(name,dataset=str(data),output_dir=runs/("regression-"+name.replace("/","-")),overrides={"training_updates":2,"off_batch_size":4,"dataset_options.max_length":12,"rnn_hidden_dim":16})
        assert result.parameter_delta_l2>0
        regressions.append(dict(name=name,result=asdict(result)))
    assert {child.pid for child in multiprocessing.active_children()}==children
    (evidence/"installed_validation.json").write_text(json.dumps(dict(version=unified_marl.__version__,package=unified_marl.__file__,python=sys.executable,
        cwd=str(Path.cwd()),pyzmq=importlib.metadata.version("pyzmq"),libzmq=zmq.zmq_version(),runs=rows,resume=asdict(continued),cli=cli,failures=failures,
        local_regressions=regressions,child_processes_before=sorted(children),child_processes_after=sorted(child.pid for child in multiprocessing.active_children()),
        cross_machine_verified=False,real_environments_verified=False,gpu_verified=False,policy_effectiveness="NOT_VALIDATED"),ensure_ascii=False,indent=2),encoding="utf-8")

if __name__=="__main__": main()
