from contextlib import contextmanager
import copy
from dataclasses import asdict
import importlib
import inspect
import json
from pathlib import Path
import sys
from types import ModuleType
import numpy as np
import pytest
import torch
from unified_marl import train,evaluate,load_policy
from unified_marl.api import _make_model
from unified_marl.algorithms.sadma.adapter import native_batch
from unified_marl.algorithms.sadma.native.utils.buffer import EpisodeBuffer
from unified_marl.checkpoints import read_checkpoint
from unified_marl.config import load_config
from unified_marl.envs.toy import CoordinationEnv
from unified_marl.learners.base import UpdateContext
from unified_marl.runners.episode_runner import EpisodeRunner
from unified_marl.trainers.sadma import concatenate
from unified_marl.utils.logging import Logger

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT.parent/"marl平台代码/王思哲/code/sadma"
EVIDENCE=ROOT/"docs/evidence/stage5"
NAMES=["sadma/qmix","sadma/ppo","sadma/vdn","sadma_yi/qmix","sadma_yi/ppo","sadma_yi_closed/qmix","sadma_yi_closed/ppo"]
CASES=[(name,{}) for name in NAMES]+[
 ("sadma/ppo",{"agent_type":"mlp","critic_type":"mlp","use_value_norm":False}),
 ("sadma/ppo",{"critic_inputs":["independent_obs","agent_id","last_action"]}),
 ("sadma/ppo",{"critic_inputs":["central_obs","state"]}),
 ("sadma/ppo",{"use_vtrace":True,"dual_clip":3}),
 ("sadma/qmix",{"q_lambda":True}),
 ("sadma/qmix",{"agent_type":"mlp"})]

@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    threads=torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(sys,"dont_write_bytecode",True)
    yield
    torch.set_num_threads(threads)

def options(**extra):
    return dict(train_batch_size=2,train_buffer_size=8,update_rate=1,t_max=40,max_updates=2,hidden_dim=16,mini_epochs=2,**extra)

def setup(name="sadma/ppo",extra=None):
    config=load_config(name,overrides={**options(),**(extra or {})})
    config["device"]=config["train_device"]="cpu"
    from types import SimpleNamespace
    runner=EpisodeRunner(SimpleNamespace(**config),Logger())
    args,scheme,groups,prep,mac,learner=_make_model(config,runner.get_env_info(),Logger())
    runner.setup(scheme,groups,prep,mac)
    runner.args=args
    return runner,args,scheme,groups,prep,mac,learner

@contextmanager
def original(monkeypatch):
    with monkeypatch.context() as context:
        context.syspath_prepend(str(SOURCE))
        for key in list(sys.modules):
            if key.split(".")[0] in {"policy","utils"}: context.delitem(sys.modules,key)
        try: yield
        finally:
            for key in list(sys.modules):
                if key.split(".")[0] in {"policy","utils"}: context.delitem(sys.modules,key)

def adapt_reference(module,name):
    text=inspect.getsource(getattr(module,name))
    text=text.replace('torch.gather(target_mac_out, 3, batch["actions"])','torch.gather(target_mac_out, 3, batch["actions"].unsqueeze(-1))')
    text=text.replace('            if not self.args.use_vtrace:', '            else:\n                norm_old_values = old_values\n\n            if not self.args.use_vtrace:')
    text=text.replace('inputs["state"] = states[:, t]','inputs["states"] = states[:, t]')
    text=text.replace('available_actions = data["available_actions"].reshape(b * a, -1)','available_actions = data["available_actions"].reshape_as(x)')
    text=text.replace('for k, v in data.items()}','for k, v in data.items() if v is not None}')
    namespace=dict(vars(module))
    exec(compile(text,str(SOURCE/"reference-runtime-compatibility"),"exec"),namespace)
    return namespace[name]

@pytest.mark.parametrize("index,case",list(enumerate(CASES)))
def test_source_loss_outputs_and_update_parity(monkeypatch,index,case):
    name,extra=case
    torch.manual_seed(51)
    runner,args,scheme,groups,prep,mac,learner=setup(name,extra)
    try: batch=concatenate([runner.run(),runner.run()],scheme,groups,prep,"cpu")
    finally: runner.close_env()
    legacy=native_batch(batch)
    before=[p.detach().clone() for p in mac.parameters()]
    with original(monkeypatch):
        kind="mappo.ppo" if args.sadma_algorithm=="ppo" else "qmix.q"
        controller_module=importlib.import_module("policy."+kind+"_controller")
        net=importlib.import_module("policy."+kind+"_net")
        if args.sadma_algorithm=="ppo" and args.agent_type=="mlp":
            controller_module.agent_REGISTRY["pi_mlp"]=adapt_reference(net,"PiMlpAgent")
        cls=adapt_reference(controller_module,"PPOController" if args.sadma_algorithm=="ppo" else "QController")
        reference_mac=cls(args)
        reference_mac.load_agent_state_dict(mac.agent.state_dict())
        policy_module=importlib.import_module("policy."+kind+"_policy")
        reference=adapt_reference(policy_module,"PPOPolicy" if args.sadma_algorithm=="ppo" else "Qpolicy")(args,reference_mac)
        for key,value in vars(learner.native).items():
            if isinstance(value,torch.nn.Module): getattr(reference,key).load_state_dict(value.state_dict())
        if hasattr(reference,"target_mac"): reference.target_mac.load_agent_state_dict(learner.native.target_mac.agent_state_dict)
        inputs={"obs":legacy["obs"][:,0],"available_actions":legacy["available_actions"][:,0],"hidden_states":reference_mac.init_hidden_states(2),
                "last_actions":torch.zeros(2,args.n_agents,args.n_actions)}
        expected=reference_mac.agent(inputs)[0]
        actual=mac.native.agent(inputs)[0]
        expected=expected.probs if hasattr(expected,"probs") else expected
        actual=actual.probs if hasattr(actual,"probs") else actual
        output_error=float((expected-actual).abs().max())
        losses=[]
        backward=torch.Tensor.backward
        with monkeypatch.context() as context:
            def capture(self,*a,**kw):
                losses.append(float(self.detach()))
                return backward(self,*a,**kw)
            context.setattr(torch.Tensor,"backward",capture)
            reference.train(copy.deepcopy(legacy),2,8)
        metrics=learner.update(batch,UpdateContext(8,2,0))
        errors=[]
        for a,b in zip(reference_mac.agent.parameters(),mac.agent.parameters()): errors.append(float((a-b).abs().max()))
        for key,value in vars(learner.native).items():
            if isinstance(value,torch.nn.Module):
                for field,tensor in value.state_dict().items(): errors.append(float((tensor-getattr(reference,key).state_dict()[field]).abs().max()))
        loss_error=abs(losses[-1]-metrics["loss"])
        delta=sum(float((a-b.detach()).square().sum()) for a,b in zip(before,mac.parameters()))**0.5
        assert output_error==0 and max(errors)==0 and loss_error<=1e-7 and delta>0
        EVIDENCE.mkdir(parents=True,exist_ok=True)
        (EVIDENCE/f"parity-{index:02d}.json").write_text(json.dumps(dict(result="PASS",algorithm=name,overrides=extra,
            source=str(SOURCE),seed=51,output_max_abs_error=output_error,loss_abs_error=loss_error,updated_network_max_abs_error=max(errors),parameter_delta_l2=delta,
            reference_adaptations="shape index, None MLP hidden state, state alias and norm-off fallback only; objectives unchanged; reference source files unmodified")),encoding="utf-8")

@pytest.mark.parametrize("name",NAMES)
def test_save_evaluate_and_exact_update_boundary_resume(tmp_path,name):
    run=train(name,"toy",output_dir=tmp_path/"short",overrides=options())
    assert run.updates==2 and run.parameter_delta_l2>0 and np.isfinite(run.metrics["loss"])
    policy=load_policy(run.checkpoint)
    env=CoordinationEnv()
    obs=np.asarray(env.get_obs())
    available=np.asarray(env.get_avail_actions())
    action=policy.act(obs,available)
    assert available[np.arange(2),action].all()
    policy.reset()
    np.testing.assert_array_equal(action,policy.act(obs,available))
    assert evaluate(run.checkpoint,episodes=2).episodes==2
    resumed=train(resume=run.training_checkpoint,output_dir=tmp_path/"resumed",overrides={"max_updates":3})
    continuous=train(name,"toy",output_dir=tmp_path/"continuous",overrides={**options(),"max_updates":3})
    a,b=read_checkpoint(resumed.training_checkpoint),read_checkpoint(continuous.training_checkpoint)
    for key,tensor in a["agent"].items(): torch.testing.assert_close(tensor,b["agent"][key],atol=0,rtol=0)
    for module_name,states in a["learner"]["extra"]["modules"].items():
        for key,tensor in states.items(): torch.testing.assert_close(tensor,b["learner"]["extra"]["modules"][module_name][key],atol=0,rtol=0)
    assert a["progress"]==b["progress"]
    if name.endswith("ppo"): assert a["sadma_replay"] is None

def test_time_positions_old_probabilities_values_and_hidden_states():
    runner,args,scheme,groups,prep,mac,learner=setup()
    try: batch=runner.run()
    finally: runner.close_env()
    assert batch["filled"].sum()==5 and batch["terminated"][0,3]==1
    assert batch["reward"][0,4]==0 and batch["obs"][0,4,0,0]==1
    hidden=mac.native.init_hidden_states(1)
    critic_hidden=torch.zeros(1,args.hidden_dim)
    prob_error=value_error=hidden_error=0.0
    for t in range(5):
        data={"obs":batch["obs"][:,t],"available_actions":batch["avail_actions"][:,t],"hidden_states":hidden,
              "last_actions":torch.zeros(1,2,3) if t==0 else batch["actions_onehot"][:,t-1]}
        dist,hidden=mac.native.agent(data)
        prob_error=max(prob_error,float((dist.log_prob(batch["actions"][:,t].squeeze(-1))-batch["action_log_probs"][:,t].squeeze(-1)).abs().max()))
        hidden_error=max(hidden_error,float((hidden-batch["hidden_states"][:,t]).abs().max()))
        value,critic_hidden=learner.native.critic(dict(obs=batch["obs"][:,t],states=batch["state"][:,t],hidden_states=critic_hidden))
        value_error=max(value_error,float((value-batch["behavior_values"][:,t]).abs().max()))
    assert prob_error==value_error==hidden_error==0
    from types import SimpleNamespace
    legacy=native_batch(batch)
    native_scheme={k:{"vshape":tuple(v.shape[1:]),"dtype":v.dtype} for k,v in legacy.items()}
    args.num_sample_worker=args.env_batch_size=args.num_env_runner=1
    args.evaluate=False
    original_buffer=EpisodeBuffer(args,native_scheme)
    for t in range(5):
        arrival={k:legacy[k][:,t] for k in ["obs","states","available_actions","actions","action_log_probs","hidden_states"]}
        arrival["rewards"]=legacy["rewards"][:,max(t-1,0)] if t else torch.zeros(1,1)
        arrival["terminated"]=legacy["terminated"][:,max(t-1,0)] if t else torch.zeros(1,1,dtype=torch.uint8)
        original_buffer.insert_data(arrival,np.array([0]),np.array([t]))
    for key in ["obs","states","available_actions","rewards","terminated","actions","action_log_probs","hidden_states"]:
        torch.testing.assert_close(original_buffer.datas[key],legacy[key],atol=0,rtol=0)
    pair=concatenate([batch,batch],scheme,groups,prep,"cpu")
    learner.update(pair,UpdateContext(8,2,0))
    with pytest.raises(ValueError,match="old policy"): learner.update(pair,UpdateContext(8,2,1))
    (EVIDENCE/"time_semantics.json").write_text(json.dumps(dict(result="PASS",log_probability_error=prob_error,behavior_value_error=value_error,
        hidden_state_error=hidden_error,source_arrival_buffer_equivalence=True,ppo_stale_replay_rejected=True)),encoding="utf-8")

def test_success_failure_cleanup_and_two_tasks_same_process(monkeypatch,tmp_path):
    from unified_marl.envs import REGISTRY
    made=[]
    def factory(**kwargs):
        env=CoordinationEnv(**kwargs)
        made.append(env)
        return env
    monkeypatch.setitem(REGISTRY,"toy",factory)
    before=torch.get_num_threads()
    for i in range(2): train("sadma/ppo","toy",output_dir=tmp_path/str(i),overrides=options())
    assert len(made)==2 and all(env.closed for env in made)
    with pytest.raises(ValueError,match="individual_rewards"):
        train("sadma/ppo","toy",output_dir=tmp_path/"failure",overrides={**options(),"use_individual_rewards":True})
    assert made[-1].closed and torch.get_num_threads()==before
    (EVIDENCE/"lifecycle.json").write_text(json.dumps(dict(result="PASS",sequential_tasks=2,closed_envs=len(made),error_cleanup=True,torch_threads_restored=True)),encoding="utf-8")

def test_individual_rewards_are_written_at_transition_not_arrival(monkeypatch):
    from unified_marl.envs import REGISTRY
    class Individual(CoordinationEnv):
        def step(self,actions):
            reward,done,info=super().step(actions)
            info["individual_rewards"]=[10+self.t,20+self.t]
            return reward,done,info
    monkeypatch.setitem(REGISTRY,"toy",lambda **kwargs:Individual(**kwargs))
    runner,args,scheme,groups,prep,mac,learner=setup(extra={"use_individual_rewards":True})
    try: batch=runner.run()
    finally: runner.close_env()
    np.testing.assert_array_equal(batch["individual_rewards"][0,:4,:,0],[[11,21],[12,22],[13,23],[14,24]])
    assert batch["individual_rewards"][0,4].sum()==0
    learner.update(concatenate([batch,batch],scheme,groups,prep,"cpu"),UpdateContext(8,2,0))

def test_cityflow_optional_backend_close(monkeypatch,tmp_path):
    class Backend(CoordinationEnv):
        made=[]
        def __init__(self,**kwargs):
            super().__init__(seed=kwargs.get("seed",1))
            self.made.append(self)
    module=ModuleType("cityflowenv.env")
    module.CityflowEnv=Backend
    monkeypatch.setitem(sys.modules,"cityflowenv",ModuleType("cityflowenv"))
    monkeypatch.setitem(sys.modules,"cityflowenv.env",module)
    result=train("sadma/qmix","cityflow",output_dir=tmp_path,overrides=options())
    assert result.parameter_delta_l2>0 and Backend.made[-1].closed

@pytest.mark.parametrize("algorithm,extra,message",[
 ("sadma/maddpg",{},"empty"),("sadma/ppo",{"env_type":"continuous"},"Continuous"),
 ("sadma/ppo",{"async_train":True},"synchronous"),("sadma/ppo",{"save_log_probs":False},"log_probs")])
def test_unsupported_protocols_are_explicit(algorithm,extra,message):
    with pytest.raises(ValueError,match=message): load_config(algorithm,overrides=extra)

def test_powergrid_cannot_be_silently_discretized():
    with pytest.raises(ValueError,match="continuous float actions"): load_config("sadma/ppo","powergrid")

def test_version_selection_and_source_formula_identity():
    wang=load_config("sadma/qmix")
    yi=load_config("sadma_yi/qmix")
    closed=load_config("sadma_yi_closed/ppo")
    assert wang["mixer"]=="q_mixer" and yi["mixer"]=="vdn_mixer" and closed["mini_epochs"]==8
    import ast
    paths=[SOURCE,ROOT.parent/"marl平台代码/易才润/论文/SADMA/sadma源代码开源版本"]
    for name,cls in [("policy/qmix/q_policy.py","Qpolicy"),("policy/mappo/ppo_policy.py","PPOPolicy")]:
        functions=[]
        for path in paths:
            tree=ast.parse((path/name).read_text(encoding="utf-8"))
            node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
            functions.append({n.name:ast.dump(n) for n in node.body if isinstance(n,ast.FunctionDef) and n.name in {"__init__","train","_update_targets"}})
        assert functions[0]==functions[1]

@pytest.mark.parametrize("truncate",[False,True])
def test_terminal_and_timeout_positions_remain_separate(truncate,tmp_path):
    runner,args,scheme,groups,prep,mac,learner=setup("sadma/qmix")
    runner.env.truncate=truncate
    try: batch=runner.run()
    finally: runner.close_env()
    assert batch["terminated"][0,3].item()==int(not truncate)
    assert batch["truncated"][0,3].item()==int(truncate)
    assert batch["terminated"][0,4].item()==batch["truncated"][0,4].item()==0
    result=train("sadma/qmix","toy",env_args={"truncate":truncate},output_dir=tmp_path,overrides=options())
    assert result.updates==2 and result.env_steps==12 and result.parameter_delta_l2>0

def test_constructor_metadata_failure_closes_environment(monkeypatch,tmp_path):
    from unified_marl.envs import REGISTRY
    created=[]
    class Bad(CoordinationEnv):
        def get_env_info(self): raise RuntimeError("metadata failure")
    def factory(**kwargs):
        env=Bad(**kwargs)
        created.append(env)
        return env
    monkeypatch.setitem(REGISTRY,"toy",factory)
    with pytest.raises(RuntimeError,match="metadata failure"):
        train("sadma/qmix","toy",output_dir=tmp_path,overrides=options())
    assert created[-1].closed

def test_source_gae_terminal_bootstrap_quirk_is_retained():
    from unified_marl.algorithms.sadma.native.utils.utils import build_gae_targets
    values=torch.tensor([[[[2.]],[[5.]]]])
    advantage,targets=build_gae_targets(torch.ones(1,1,1,1),torch.ones(1,1,1,1),values,0.9,0.95)
    # Original GAE consumes filled masks, not an explicit (1-terminated) mask.
    # This is documented as a source limitation, not presented as corrected GAE.
    torch.testing.assert_close(targets,torch.full_like(targets,5.5))

def test_native_replay_padding_and_wrap():
    from unified_marl.algorithms.sadma.native.utils.buffer import ReplayBuffer
    from types import SimpleNamespace
    args=SimpleNamespace(train_buffer_size=3,train_batch_size=2,train_device="cpu")
    scheme={"obs":{"vshape":(5,2,4),"dtype":torch.float32},"filled":{"vshape":(5,1),"dtype":torch.int32},"rewards":{"vshape":(5,1),"dtype":torch.float32}}
    buffer=ReplayBuffer(args,scheme)
    for i in range(5):
        data={"obs":torch.full((1,5,2,4),float(i)),"filled":torch.tensor([[[1],[1],[1],[0],[0]]]),"rewards":torch.ones(1,5,1)}
        buffer.insert_episodes(1,data)
    assert buffer.episodes_in_buffer==3 and buffer.index_count==2
    sample=buffer.sample()
    assert sample["obs"].shape==(2,3,2,4)
    assert set(sample["obs"][:,0,0,0].tolist())<={2.,3.,4.}

def test_sadma_replenishment_configuration_preserves_backend_horizon():
    config=load_config("sadma/ppo","replenishment")
    assert config["env_args"]["horizon"] is None
    assert config["env_args"]["time_limit"]==21
    assert config["sadma_env_metadata"]=={"key":"replenishment_efficient_env-v0","map_name":"sku1000.standard","reward":"reward2"}
    assert all(key not in config["env_args"] for key in ["key","map_name","reward"])

def test_sadma_replenishment_backend_contract_and_individual_updates(monkeypatch,tmp_path):
    from types import SimpleNamespace
    captured={}
    class Backend:
        config={"env":{"horizon":3}}
        action_space=[SimpleNamespace(n=3),SimpleNamespace(n=2)]
        observation_space=[SimpleNamespace(shape=(3,)),SimpleNamespace(shape=(2,))]
        closed=False
        def get_agent_count(self): return 2
        def reset(self): return [np.ones(3),np.ones(2)],{}
        def step(self,actions): return [np.ones(3)*2,np.ones(2)*2],[10000,20000],[False]*2,[False]*2,{}
        def close(self): self.closed=True
    backend=Backend()
    module=ModuleType("ReplenishmentEnv")
    def factory(task,**kwargs):
        captured.update(task=task,**kwargs)
        return backend
    module.make_env=factory
    monkeypatch.setitem(sys.modules,"ReplenishmentEnv",module)
    result=train("sadma/ppo","replenishment",output_dir=tmp_path,overrides={**options(),"use_individual_rewards":True})
    assert result.updates==2 and result.parameter_delta_l2>0 and backend.closed
    assert "env" not in captured["update_config"]
    assert len(captured["update_config"]["action"]["space"])==34
