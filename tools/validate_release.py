"""Standalone wheel acceptance. Copy this file and demo generator outside source."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import math
import multiprocessing
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
import torch
import unified_marl
from unified_marl import train, evaluate, load_policy, train_reward_model, load_reward_model
from unified_marl.datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["base", "full"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--real-data")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    assert "site-packages" in unified_marl.__file__ and unified_marl.__version__ == "0.7.0"
    assert not any(Path(item or '.').resolve().name in {"src", "pymarl-master", "marl平台代码"} for item in sys.path)
    report = dict(version=unified_marl.__version__, mode=args.mode, package=unified_marl.__file__, cwd=str(Path.cwd()),
                  python=sys.version, platform=platform.platform(), dependencies={x.metadata['Name']:x.version for x in metadata.distributions()},
                  source_imports=[], runs=[], optional_errors=[], real_environment_errors=[], cross_machine_verified=False,
                  gpu_verified=False, policy_effect_verified=False)
    before_children = {child.pid for child in multiprocessing.active_children()}
    def reject_original_code(event, values):
        if event == "open" and isinstance(values[0], (str, bytes)):
            path = str(values[0]).replace('\\','/')
            if ("/marl平台代码/" in path or "/pymarl-master/" in path) and Path(path).suffix.lower() in {".py", ".pyc", ".yaml", ".yml"}:
                raise RuntimeError("Installed package attempted to read original code/config: "+path)
    sys.addaudithook(reject_original_code)
    from make_demo_data import make_demo_data
    demo = make_demo_data(output/"合成轨迹.th")
    data = load_dataset(demo)
    def policy_check(result, dataset=None):
        assert result.updates > 0 and result.parameter_delta_l2 > 0 and math.isfinite(result.metrics['loss'])
        policy = load_policy(result.checkpoint)
        if dataset is None:
            from unified_marl.envs import make_env
            env = make_env("toy", **policy.config["env_args"])
            try:
                env.reset()
                obs, mask = env.get_obs(), np.asarray(env.get_avail_actions())
            finally: env.close()
        else:
            batch = dataset.batch([0]); obs, mask = batch['obs'][0,0].numpy(), batch['avail_actions'][0,0].numpy()
        first = policy.act(obs, mask)
        assert mask[np.arange(len(first)), first].all()
        policy.reset()
        np.testing.assert_array_equal(policy.act(obs,mask),first)
        if dataset is None: return asdict(evaluate(result.checkpoint,episodes=2,seed=11))
        return dict(inference=True, environment_evaluation="NOT_VALIDATED")
    def run(name, kind="core", **extra):
        options = dict(t_max=40, max_updates=2, batch_size=2, buffer_size=8, rnn_hidden_dim=16)
        env_args = None
        if kind == "sadma": options.update(train_batch_size=2, train_buffer_size=8, hidden_dim=16, mini_epochs=2, update_rate=1)
        if kind == "hmf":
            options.update(batch_size=4,buffer_size=32,hidden_dim=16,n_heads=2,embed_dim=8,hmf_norm_rewards=False,hmf_steps_per_update=1,hmf_num_batch=1)
            env_args = dict(n_agents=4,mask_actions=False)
        if kind == "role": options.update(role_interval=2)
        if name.endswith("coma"): options.update(batch_size=1, batch_size_run=1, runner="episode")
        options.update(extra)
        result = train(name,"toy",env_args=env_args,output_dir=output/name.replace('/','-'),overrides=options)
        score = policy_check(result)
        report['runs'].append(dict(algorithm=name,result=asdict(result),evaluation=score))
        return result
    if args.mode == "base":
        absent = ["sklearn", "gym", "cloudpickle", "ml_collections", "transformers", "scipy", "zmq", "smac"]
        assert all(importlib.util.find_spec(name) is None for name in absent)
        report['confirmed_absent'] = absent
        for name in ["qmix","vdn","iql","coma","qtran","qmix_beta","vdn_beta","iql_beta"]:
            run("pymarl/"+name)
        for name in ["sadma/qmix","sadma/ppo"]: run(name,"sadma")
        for name in ["kou/bc","kou/cpl"]:
            result = train(name,dataset=demo,output_dir=output/name.replace('/','-'),overrides={"training_updates":2,"off_batch_size":4,"rnn_hidden_dim":16})
            policy_check(result,data)
            report['runs'].append(dict(algorithm=name,result=asdict(result),evaluation=asdict(evaluate(result.checkpoint,env="toy",env_args={"mask_actions":False},episodes=2))))
        probes=[("role",lambda:train("wang/rode",output_dir=output/"missing-role",overrides={"t_max":8,"batch_size":2})),
                ("hmf",lambda:train("wang_hmf/mfq",output_dir=output/"missing-hmf")),
                ("reward",lambda:train_reward_model("MR",demo,updates=1,batch_size=2,output_dir=output/"missing-reward")),
                ("distributed",lambda:train("sadma/qmix",execution="zmq",output_dir=output/"missing-zmq"))]
        for capability, call in probes:
            try: call()
            except ImportError as exc:
                assert capability in str(exc).lower() or (capability=="hmf" and "gym" in str(exc))
                report['optional_errors'].append(dict(capability=capability,error=str(exc)))
            else: raise AssertionError("Missing optional capability unexpectedly succeeded: "+capability)
    else:
        for name in ["wang/rode","wang/rode_hmf"]: run(name,"role")
        for name in ["wang_hmf/mfq","wang_hmf/hierarchical_vdac_fixed"]: run(name,"hmf")
        for name in ["sadma/qmix","sadma/ppo","sadma/vdn","sadma_yi/qmix","sadma_yi/ppo","sadma_yi_closed/qmix","sadma_yi_closed/ppo"]:
            result = run(name,"sadma")
            continued = train(resume=result.training_checkpoint,output_dir=output/(name.replace('/','-')+'-resume'),overrides={"max_updates":3})
            assert continued.updates==3 and continued.parameter_delta_l2>0
        assert args.real_data
        real=load_dataset(args.real_data,max_length=12)
        report['real_data']=dict(path=str(Path(args.real_data).resolve()),sha256=hashlib.sha256(Path(args.real_data).read_bytes()).hexdigest(),episodes=len(real),env_info=real.env_info)
        for name,agent in [("kou/bc","rnn_ns"),("kou_old/qmix","mlp"),("kou_old/qmix","mlp_ns")]:
            result=train(name,dataset=args.real_data,output_dir=output/('variant-'+agent),overrides={"agent":agent,"hidden_dim":16,"training_updates":2,"off_batch_size":4,"batch_size":4,"rnn_hidden_dim":16,"dataset_options.max_length":12})
            policy_check(result,real)
            continued=train(resume=result.training_checkpoint,dataset=args.real_data,output_dir=output/('variant-resume-'+agent),overrides={"training_updates":3})
            assert continued.updates==3 and continued.parameter_delta_l2>0
            report['runs'].append(dict(algorithm=name,agent=agent,result=asdict(result),resume=asdict(continued),data="REAL_3M"))
        for name in ["kou/bc","kou/cpl","kou/icq_ma","kou_old/qmix"]:
            result=train(name,dataset=args.real_data,output_dir=output/('real-'+name.replace('/','-')),
                         overrides={"training_updates":2,"off_batch_size":4,"batch_size":4,"rnn_hidden_dim":16,"dataset_options.max_length":12})
            policy_check(result,real)
            continued=train(resume=result.training_checkpoint,dataset=args.real_data,output_dir=output/('real-resume-'+name.replace('/','-')),overrides={"training_updates":3})
            assert continued.updates==3 and continued.parameter_delta_l2>0
            report['runs'].append(dict(algorithm=name,result=asdict(result),resume=asdict(continued),data="REAL_3M",evaluation="SMAC_NOT_VALIDATED"))
        for name in ["MR","NMR","PrefTransformer","MultiPrefTransformerDivide","old_global","old_reward_global"]:
            options={"reward_hidden_size":16,"ensemble_size":2} if name.startswith('old_') else {"inner_dim":32,"embd_dim":32,"action_embd_dim":8,"pref_attn_embd_dim":32,"n_head":4,"n_layer":1}
            result=train_reward_model(name,args.real_data,updates=1,batch_size=4,output_dir=output/('reward-'+name),model_config=options,dataset_options={"max_length":61 if name=="NMR" else 12})
            assert result.parameter_delta_l2>0
            model=load_reward_model(result.checkpoint)
            torch.manual_seed(4); prediction=model.predict(real.batch([0,1],trim=False))
            clone=load_reward_model(result.checkpoint)
            torch.manual_seed(4); torch.testing.assert_close(clone.predict(real.batch([0,1],trim=False)),prediction,atol=0,rtol=0)
            continued=load_reward_model(result.training_checkpoint,for_training=True)
            metrics=continued.train_step(load_dataset(args.real_data,max_length=61 if name=="NMR" else 12).preference([0,1],trim=False))
            assert continued.steps==2 and all(math.isfinite(float(v)) for v in metrics.values())
            report['runs'].append(dict(reward_model=name,result=asdict(result),resumed_steps=continued.steps,data="REAL_3M"))
        import zmq
        for name, mode, more in [("sadma/qmix","sync",{"evaluation_interval_updates":1}), ("sadma/qmix","async",{"zmq.num_inference_workers":2}), ("sadma/ppo","sync",{}), ("sadma/ppo","async",{"use_vtrace":True})]:
            result=train(name,"toy",execution="zmq",output_dir=output/('zmq-'+name.replace('/','-')+'-'+mode),overrides={"train_batch_size":2,"train_buffer_size":8,"hidden_dim":16,"mini_epochs":2,"t_max":64,"max_updates":2,"zmq.mode":mode,**more})
            score=policy_check(result)
            execution=json.loads((Path(result.output_dir)/'execution.json').read_text(encoding='utf-8'))
            assert execution['status']=='PASS' and not execution['lifecycle']['forced'] and not execution['evaluation']['forced']
            context=zmq.Context()
            for endpoint in [execution['trainer_endpoint'],*execution['inference_endpoints']]:
                sock=context.socket(zmq.REP);sock.setsockopt(zmq.LINGER,0);sock.bind(endpoint);sock.close(0)
            context.term()
            report['runs'].append(dict(algorithm=name,mode=mode,result=asdict(result),execution=execution,evaluation=score,ports_rebound=True))
    from unified_marl.envs import make_env
    for name in ["sc2","replenishment","cityflow"]:
        try: make_env(name)
        except (ImportError,ValueError,OSError) as exc: report['real_environment_errors'].append(dict(environment=name,error=str(exc),verified=False))
        else: raise AssertionError("Unexpected external backend; perform real short acceptance before claiming support")
    cli=[str(Path(sys.executable).with_name('unified-marl.exe')),'train','--algorithm','pymarl/qmix','--env','toy','--output-dir',str(output/'CLI 中文 空格'),'--seed','3','--set','batch_size=2','--set','t_max=16','--set','max_updates=1']
    child=subprocess.run(cli,capture_output=True,text=True,encoding="utf-8",timeout=60)
    assert child.returncode==0,child.stderr
    result=json.loads(child.stdout)
    child2=subprocess.run([str(Path(sys.executable).with_name('unified-marl.exe')),'evaluate',result['checkpoint'],'--episodes','2','--seed','3','--env-args','{mask_actions: true}'],capture_output=True,text=True,encoding="utf-8",timeout=60)
    assert child2.returncode==0,child2.stderr
    report['cli']=[dict(command=cli,returncode=child.returncode,result=result),dict(returncode=child2.returncode,result=json.loads(child2.stdout))]
    assert {child.pid for child in multiprocessing.active_children()}==before_children
    for module in list(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path and ('marl平台代码' in path or 'pymarl-master' in path): report['source_imports'].append(path)
    assert not report['source_imports']
    assert subprocess.run([sys.executable,'-m','pip','check'],capture_output=True).returncode==0
    report['status']='PASS'; report['children_after']=[child.pid for child in multiprocessing.active_children()]
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='PASS',mode=args.mode,runs=len(report['runs']),report=str(output/'report.json')),ensure_ascii=False))


if __name__=='__main__': main()
