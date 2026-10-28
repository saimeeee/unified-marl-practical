"""Wheel-only verification, launched outside the project/source directory."""
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import traceback
import uuid
import numpy as np
import torch
import unified_marl
from unified_marl import train,load_policy,train_reward_model,load_reward_model,evaluate
from unified_marl.datasets import load_dataset

POLICIES=["kou/bc","kou/cpl","kou/icq_ma","kou/iql","kou/cql","kou/dppo","kou/madpo","kou/madpo_aux","kou/bcmapt",
    "kou_old/qmix","kou_old/qplex","kou_old/cw_qmix","kou_old/ow_qmix","kou_old/cw_qmix_w_globalRM","kou_old/ow_qmix_w_globalRM"]
MODELS=["MR","NMR","PrefTransformer","MultiPrefTransformerDivide","old_global","old_reward_global"]


def main():
    root=Path(__file__).resolve().parents[1]
    assert Path.cwd().resolve()!=root
    assert "site-packages" in unified_marl.__file__ and unified_marl.__version__=="0.4.0"
    evidence=root/"docs/evidence/stage4"
    run_root=evidence/("installed-runs-"+uuid.uuid4().hex[:8])
    data_root=root.parent/"marl平台代码/寇谦/论文代码及技术文档/论文代码及技术文档/MADiversePreferDatasets/3m"
    source=str(data_root/"3m.th")
    reward_path=str(data_root/"reward.th")
    data=load_dataset(source,max_length=12,agent_reward_path=reward_path)
    policy_runs=[]
    def run_policy(name,reward_checkpoint=None):
        opts={"training_updates":2,"off_batch_size":4,"batch_size":4,"rnn_hidden_dim":16,
              "dataset_options.max_length":12,"dataset_options.agent_reward_path":reward_path}
        if reward_checkpoint: opts["reward_model_checkpoint"]=reward_checkpoint
        run=train(name,dataset=source,output_dir=run_root/name.replace("/","-"),overrides=opts)
        assert run.updates==2 and run.env_steps==0 and run.parameter_delta_l2>0
        policy=load_policy(run.checkpoint)
        batch=data.batch([0])
        obs=batch["obs"][0,0].numpy()
        available=batch["avail_actions"][0,0].numpy()
        actions=policy.act(obs,available)
        assert available[np.arange(3),actions].all()
        policy.reset()
        np.testing.assert_array_equal(actions,policy.act(obs,available))
        resumed=train(resume=run.training_checkpoint,dataset=source,output_dir=run_root/(name.replace("/","-")+"-resumed"),overrides={"training_updates":3})
        assert resumed.updates==3 and resumed.parameter_delta_l2>0
        policy_runs.append({"algorithm":name,"train":asdict(run),"resume":asdict(resumed)})
        print(name,"PASS",flush=True)
    for name in POLICIES: run_policy(name)
    reward_runs=[]
    for name in MODELS:
        options={"reward_hidden_size":32,"ensemble_size":2} if name.startswith("old_") else {
            "inner_dim":32,"action_embd_dim":8,"embd_dim":32,"pref_attn_embd_dim":32,"n_head":4,"n_layer":1}
        result=train_reward_model(name,source,updates=1,batch_size=4,output_dir=run_root/("reward-"+name),model_config=options,
                                  dataset_options={"max_length":61 if name=="NMR" else 12})
        assert result.parameter_delta_l2>0
        policy,full=load_reward_model(result.checkpoint),load_reward_model(result.training_checkpoint)
        batch=data.batch([0,1],trim=False)
        torch.manual_seed(60)
        actual=policy.predict(batch)
        torch.manual_seed(60)
        expected=full.predict(batch)
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        assert torch.isfinite(actual).all()
        reward_runs.append({"model":name,"result":asdict(result),"prediction_shape":list(actual.shape)})
        if name=="old_global":
            for algorithm in ["kou_old/qmix_w_globalRM","kou_old/qplex_w_globalRM"]: run_policy(algorithm,result.checkpoint)
        print("reward",name,"PASS",flush=True)
    diverse=train_reward_model("MultiPrefTransformerDivide",source,updates=1,batch_size=4,output_dir=run_root/"diverse",
        dataset_options={"max_length":12},mode="diverse",model_config={"embd_dim":32,"pref_attn_embd_dim":32,"n_head":4,"action_embd_dim":8})
    assert diverse.parameter_delta_l2>0
    reward_runs.append({"model":"MultiPrefTransformerDivide/diverse","result":asdict(diverse)})
    # Online regression uses its own environment, not any recorded SMAC data.
    regression=train("pymarl/qmix","toy",output_dir=run_root/"qmix-regression",overrides={"batch_size":2,"buffer_size":8,"t_max":16,"max_updates":2})
    score=evaluate(regression.checkpoint,episodes=2)
    cli=subprocess.run([str(Path(sys.executable).with_name("unified-marl.exe")),"train","--algorithm","kou/bc","--dataset",source,
        "--output-dir",str(run_root/"cli-bc"),"--set","training_updates=1","--set","off_batch_size=4","--set","dataset_options.max_length=12"],
        text=True,capture_output=True)
    assert cli.returncode==0,cli.stderr
    settings=run_root/"cli-reward.yaml"
    settings.write_text("model_config:\n  inner_dim: 32\ndataset_options:\n  max_length: 12\n",encoding="utf-8")
    cli_reward=subprocess.run([str(Path(sys.executable).with_name("unified-marl.exe")),"train-reward","--model","MR","--dataset",source,
        "--output-dir",str(run_root/"cli-reward"),"--updates","1","--batch-size","4","--config",str(settings)],
        text=True,capture_output=True)
    assert cli_reward.returncode==0,cli_reward.stderr
    blocked=[]
    for name in ["hacpl_smac","maicpl_smac","maidpo_smac"]:
        try:
            train("kou/"+name,dataset=source,output_dir=run_root/name)
            raise AssertionError("Missing source learner unexpectedly ran")
        except ValueError as exc: blocked.append({"name":name,"status":"SOURCE_BLOCKED","error":str(exc)})
    try:
        evaluate(policy_runs[0]["train"]["checkpoint"],env="sc2",env_args={"map_name":"3m"},episodes=1)
        blocked.append({"name":"real_smac_evaluation","status":"RAN_REQUIRES_REVIEW"})
    except Exception as exc:
        blocked.append({"name":"real_smac_evaluation","status":"ENVIRONMENT_BLOCKED","error":str(exc),"traceback":traceback.format_exc()})
    (evidence/"installed_validation.json").write_text(json.dumps({"version":unified_marl.__version__,"package":unified_marl.__file__,
        "python":sys.executable,"cwd":str(Path.cwd()),"data_source":source,"policies":policy_runs,"reward_models":reward_runs,
        "qmix_regression":{"train":asdict(regression),"evaluation":asdict(score)},"blocked":blocked,
        "cli":{"returncode":cli.returncode,"stdout":cli.stdout,"stderr":cli.stderr},
        "cli_reward":{"returncode":cli_reward.returncode,"stdout":cli_reward.stdout,"stderr":cli_reward.stderr}},ensure_ascii=False,indent=2),encoding="utf-8")


if __name__=="__main__": main()
