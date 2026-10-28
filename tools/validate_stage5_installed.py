"""Outside-source, installed-wheel short tasks; never substitutes for real simulators."""
from dataclasses import asdict
import importlib.util
import json
import multiprocessing
from pathlib import Path
import subprocess
import sys
import traceback
import uuid
import numpy as np
import unified_marl
from unified_marl import train,evaluate,load_policy
from unified_marl.envs import REGISTRY
from unified_marl.envs.toy import CoordinationEnv

def main():
    root=Path(__file__).resolve().parents[1]
    evidence=root/"docs/evidence/stage5"
    assert Path.cwd().resolve()!=root and "site-packages" in unified_marl.__file__ and unified_marl.__version__=="0.5.0"
    runs=evidence/("installed-runs-"+uuid.uuid4().hex[:8])
    children=len(multiprocessing.active_children())
    made=[]
    def factory(**kwargs):
        env=CoordinationEnv(**kwargs)
        made.append(env)
        return env
    original=REGISTRY["toy"]
    REGISTRY["toy"]=factory
    policies=[]
    options=dict(train_batch_size=2,train_buffer_size=8,update_rate=1,hidden_dim=16,mini_epochs=2,t_max=40,max_updates=2)
    names=["sadma/qmix","sadma/ppo","sadma/vdn","sadma_yi/qmix","sadma_yi/ppo","sadma_yi_closed/qmix","sadma_yi_closed/ppo"]
    try:
        for name in names:
            result=train(name,"toy",output_dir=runs/name.replace("/","-"),overrides=options)
            assert result.updates==2 and result.parameter_delta_l2>0
            score=evaluate(result.checkpoint,episodes=2)
            policy=load_policy(result.checkpoint)
            env=CoordinationEnv()
            obs,available=np.asarray(env.get_obs()),np.asarray(env.get_avail_actions())
            actions=policy.act(obs,available)
            assert available[np.arange(2),actions].all()
            policy.reset()
            np.testing.assert_array_equal(actions,policy.act(obs,available))
            env.close()
            continued=train(resume=result.training_checkpoint,output_dir=runs/(name.replace("/","-")+"-resumed"),overrides={"max_updates":3})
            assert continued.updates==3 and continued.parameter_delta_l2>0
            policies.append(dict(name=name,result=asdict(result),resume=asdict(continued),evaluation=asdict(score)))
            print(name,"PASS",flush=True)
        assert all(env.closed for env in made)
    finally: REGISTRY["toy"]=original
    cli=[]
    for name in ["sadma/qmix","sadma/ppo"]:
        arguments=[str(Path(sys.executable).with_name("unified-marl.exe")),"train","--algorithm",name,"--env","toy","--output-dir",str(runs/("cli-"+name.replace("/","-")))]
        for key,value in options.items(): arguments.extend(["--set",f"{key}={value}"])
        result=subprocess.run(arguments,text=True,capture_output=True)
        assert result.returncode==0,result.stderr
        cli.append(dict(algorithm=name,argv=arguments,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
    regressions=[]
    for name,settings in [
        ("pymarl/qmix",dict(batch_size=2,buffer_size=8,t_max=16,max_updates=2)),
        ("wang/rode_hmf",dict(batch_size=2,buffer_size=8,t_max=16,max_updates=2,role_interval=2)),
        ("wang_hmf/mfq",dict(batch_size=4,buffer_size=32,hidden_dim=16,n_heads=2,embed_dim=8,t_max=16,max_updates=2,hmf_norm_rewards=False,hmf_steps_per_update=1,hmf_num_batch=1))]:
        result=train(name,"toy",output_dir=runs/("regression-"+name.replace("/","-")),overrides=settings,
                     env_args={"n_agents":4,"mask_actions":False} if name.startswith("wang_hmf") else None)
        assert result.parameter_delta_l2>0
        regressions.append(dict(name=name,result=asdict(result)))
        print(name,"REGRESSION_PASS",flush=True)
    data=root.parent/"marl平台代码/寇谦/论文代码及技术文档/论文代码及技术文档/MADiversePreferDatasets/3m/3m.th"
    for name in ["kou/bc","kou/cpl"]:
        result=train(name,dataset=str(data),output_dir=runs/("regression-"+name.replace("/","-")),
            overrides={"training_updates":2,"off_batch_size":4,"dataset_options.max_length":12,"rnn_hidden_dim":16})
        assert result.parameter_delta_l2>0 and result.env_steps==0
        regressions.append(dict(name=name,result=asdict(result)))
        print(name,"REAL_DATA_REGRESSION_PASS",flush=True)
    blocked=[]
    for env in ["sc2","replenishment","cityflow","powergrid"]:
        try:
            result=train("sadma/ppo",env,output_dir=runs/("real-"+env),overrides=options)
            blocked.append(dict(env=env,status="RAN_REQUIRES_REVIEW",result=asdict(result)))
        except (ImportError,ValueError,OSError) as exc:
            blocked.append(dict(env=env,status="BLOCKED",error=str(exc),traceback=traceback.format_exc()))
    assert all(row["status"]=="BLOCKED" for row in blocked)
    assert len(multiprocessing.active_children())==children
    audit=json.loads((evidence/"audit.json").read_text(encoding="utf-8"))
    grid=Path(audit["sources"]["yi_open"])/"envs/powergrid/data"
    grid_assets={"directory":str(grid),"exists":grid.exists(),"files":sum(1 for path in grid.rglob("*") if path.is_file()) if grid.exists() else 0}
    dependencies={name:importlib.util.find_spec(name) is not None for name in ["zmq","smac","cityflowenv","ReplenishmentEnv","pandapower","pandas"]}
    payload=dict(version=unified_marl.__version__,package=unified_marl.__file__,python=sys.executable,cwd=str(Path.cwd()),
        algorithms=policies,cli=cli,regressions=regressions,real_environments=blocked,optional_dependencies=dependencies,
        lifecycle=dict(sequential_tasks=len(policies),closed_envs=len(made),all_closed=all(env.closed for env in made),children_before=children,children_after=len(multiprocessing.active_children())),
        continuous_grid_assets=grid_assets,policy_effectiveness="NOT_VALIDATED",distributed_execution="NOT_IMPLEMENTED_STAGE6")
    (evidence/"installed_validation.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")

if __name__=="__main__": main()
