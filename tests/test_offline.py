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

from unified_marl import train,load_policy,evaluate
from unified_marl.api import _make_model
from unified_marl.checkpoints import read_checkpoint
from unified_marl.config import load_config
from unified_marl.datasets import EpisodeDataset
from unified_marl.learners.base import UpdateContext
from unified_marl.utils.logging import Logger

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT.parent/"marl平台代码/寇谦/论文代码及技术文档/论文代码及技术文档"
EVIDENCE=ROOT/"docs/evidence/stage4"
POLICIES=["kou/bc","kou/cpl","kou/icq_ma","kou/iql","kou/cql","kou/dppo","kou/madpo","kou/madpo_aux","kou/bcmapt",
          "kou_old/qmix","kou_old/qplex","kou_old/cw_qmix","kou_old/ow_qmix","kou_old/qmix_w_globalRM",
          "kou_old/qplex_w_globalRM","kou_old/cw_qmix_w_globalRM","kou_old/ow_qmix_w_globalRM"]


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    before=torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(sys,"dont_write_bytecode",True)
    yield
    torch.set_num_threads(before)


@pytest.fixture
def raw_data():
    g=torch.Generator().manual_seed(75)
    e,t,n,a=8,6,3,3
    data={"obs":torch.randn(e,t,n,5,generator=g),"state":torch.randn(e,t,4,generator=g),
          "action":torch.randint(a,(e,t,n,1),generator=g),"avail_action":torch.ones(e,t,n,a,dtype=torch.int),
          "reward":torch.randn(e,t,1,generator=g),"filled":torch.zeros(e,t,1,dtype=torch.long),
          "terminated":torch.zeros(e,t,1,dtype=torch.uint8),"agent_ids":torch.arange(n).expand(e,-1).double(),
          "agent_reward":torch.randn(e,t,n,1,generator=g),"reward_hat":torch.randn(e,t,1,generator=g)}
    for i,length in enumerate([2,3,4,5,5,4,3,2]):
        data["filled"][i,:length+1]=1
        data["terminated"][i,length-1]=1
    data["labels"]=torch.tensor([[0.],[1.],[0.],[1.]])
    data["global_labels"]=1-data["labels"]
    data["between_labels"]=torch.tensor([[0.,1.,0.]]*4)
    data["in_labels"]=torch.tensor([[[1.,2.,3.]]*4,[[3.,1.,2.]]*4])
    return data


@contextmanager
def original(monkeypatch,family):
    folder=SOURCE/("ICQ-MA/src" if family=="icq" else "old_src")
    prefixes={"components","controllers","learners","modules","utils"}
    with monkeypatch.context() as context:
        context.syspath_prepend(str(folder))
        for key in list(sys.modules):
            if key.split(".")[0] in prefixes: context.delitem(sys.modules,key)
        context.setitem(sys.modules,"wandb",ModuleType("wandb"))  # unused import only
        try: yield folder
        finally:
            for key in list(sys.modules):
                if key.split(".")[0] in prefixes: context.delitem(sys.modules,key)


def config(name):
    return load_config(name,overrides={"training_updates":2,"off_batch_size":4,"batch_size":4,"rnn_hidden_dim":16})


def call_reference(learner,batch,step):
    kwargs={}
    for key in inspect.signature(learner.train).parameters:
        if key in {"t_env","episode_num"}: kwargs[key]=step
        elif key in {"log","running_log"}: kwargs[key]={}
    from unified_marl.datasets import PreferenceBatch
    if isinstance(batch,PreferenceBatch):
        if "labels" in inspect.signature(learner.train).parameters: kwargs["labels"]=batch.labels
        learner.train(batch.trajectory0,batch.trajectory1,**kwargs)
    else:
        if learner.args.learner=="offpg_learner": learner.train_critic(batch,best_batch=None,running_log={},t_env=step)
        learner.train(batch,**kwargs)


def native_networks(native):
    result={}
    for key,value in vars(native).items():
        if isinstance(value,torch.nn.Module): result[key]=value
        elif key.endswith("mac") and hasattr(value,"agent"): result[key]=value.agent
    return result


@pytest.mark.parametrize("name",POLICIES)
def test_source_policy_loss_and_parameter_parity(name,raw_data,monkeypatch):
    cfg=config(name)
    data=EpisodeDataset(raw_data)
    torch.manual_seed(2026)
    args,scheme,groups,prep,mac,learner=_make_model(cfg,data.env_info,Logger())
    paired=bool(cfg.get("use_pref",False))
    batch=data.preference([0,1],trim=False) if paired else data.batch([0,1,2,3],trim=False)
    torch_before=[p.detach().clone() for p in mac.parameters()]
    with original(monkeypatch,cfg["kou_family"]) as folder:
        controllers=importlib.import_module("controllers")
        learners=importlib.import_module("learners")
        torch.manual_seed(2026)
        old_mac=controllers.REGISTRY[cfg["mac"]](dict(scheme,actions_onehot={"vshape":(data.n_actions,),"group":"agents"}),groups,args)
        old_log=Logger()
        old=learners.REGISTRY[cfg["learner"]](old_mac,dict(scheme,actions_onehot={"vshape":(data.n_actions,),"group":"agents"}),old_log,args)
        # QPLEX and DPPO allocate intermediate tensors on hardcoded CUDA in the
        # original. Reference-only CPU redirection changes allocation, not math.
        tensor_to=torch.Tensor.to
        with monkeypatch.context() as allocation:
            allocation.setattr(torch.Tensor,"cuda",lambda tensor,*a,**k:tensor)
            def cpu_to(tensor,*args,**kwargs):
                if args and isinstance(args[0],str) and args[0]=="cuda": args=("cpu",*args[1:])
                return tensor_to(tensor,*args,**kwargs)
            allocation.setattr(torch.Tensor,"to",cpu_to)
            torch.manual_seed(880)
            np.random.seed(880)
            call_reference(old,copy.deepcopy(batch),1)
        torch.manual_seed(880)
        np.random.seed(880)
        metrics=learner.update(copy.deepcopy(batch),UpdateContext(0,0,0))
        errors=[]
        for key,value in old_log.latest.items():
            if "loss" in key and key in learner.logger.latest:
                error=abs(float(value)-float(learner.logger.latest[key]))
                assert error <= 1e-6
                errors.append(error)
        assert errors,old_log.latest
        maximum=0.0
        current,reference=native_networks(learner.native),native_networks(old)
        assert current.keys()==reference.keys()
        for key in current:
            for weight,value in current[key].state_dict().items():
                expected=reference[key].state_dict()[weight]
                torch.testing.assert_close(value,expected,atol=1e-7,rtol=1e-6)
                maximum=max(maximum,float((value-expected).abs().max()))
    delta=sum(float((x-y.detach()).square().sum()) for x,y in zip(torch_before,mac.parameters()))**0.5
    assert delta>0 and all(np.isfinite(v) for v in metrics.values())
    (EVIDENCE/("parity-policy-"+name.replace("/","-")+".json")).write_text(json.dumps({"result":"PASS",
        "source":str(folder),"algorithm":name,"seed":2026,"loss_max_abs_error":max(errors),
        "updated_network_max_abs_error":maximum,"optimized_actor_parameter_delta_l2":delta,
        "scope":"fixed synthetic inputs; CPU allocation adaptation in reference only"},indent=2),encoding="utf-8")


def test_aliases_pairing_padding_labels_and_rewards(raw_data):
    saved={k:v.clone() for k,v in raw_data.items()}
    data=EpisodeDataset(raw_data)
    pair=data.preference([0,2])
    assert pair.indices0.tolist()==[0,2] and pair.indices1.tolist()==[4,6]
    torch.testing.assert_close(pair.labels,raw_data["labels"][[0,2]])
    assert pair.lengths0.tolist()==[2,4] and pair.lengths1.tolist()==[5,3]
    assert pair.mask0[0,1]==1 and pair.mask0[0,2]==0
    assert pair.trajectory0["actions_onehot"].shape[-1]==3
    assert pair.extra_labels["in_labels"].shape==(2,2,3)
    # Sampling and native-style in-place mask edits cannot mutate the source.
    pair.trajectory0["filled"].zero_()
    assert all(torch.equal(v,saved[k]) for k,v in raw_data.items())
    for field in ["reward","agent_reward"]:
        torch.testing.assert_close(data.batch([0],trim=False)[field],raw_data[field][:1])
    conflict=copy.deepcopy(raw_data)
    conflict["actions"]=conflict["action"]+1
    with pytest.raises(ValueError,match="Conflicting"): EpisodeDataset(conflict)
    individual=copy.deepcopy(raw_data)
    individual["reward"]=individual["agent_reward"]
    with pytest.raises(ValueError,match="no implicit"): EpisodeDataset(individual)
    loaded=EpisodeDataset(individual,reward_semantics="individual")
    assert loaded.data["reward"].shape[-2]==3
    ordinary={k:v[:5] for k,v in raw_data.items() if "label" not in k}
    assert len(EpisodeDataset(ordinary))==5
    (EVIDENCE/"data_semantics.json").write_text(json.dumps({"result":"PASS","source":"synthetic",
        "halves_pairing":True,"terminal_transition_included":True,"bootstrap_excluded_from_preference_mask":True,
        "raw_data_unmodified":True,"no_reward_sum_or_copy":True}),encoding="utf-8")


def test_ties_and_label_fields_are_not_interchanged(raw_data):
    raw_data["labels"][0]=0.5
    data=EpisodeDataset(raw_data)
    assert data.preference([0]).labels.item()==0.5
    assert data.preference([0],label_key="global_labels").labels.item()==1
    with pytest.raises(ValueError,match="scalar"): data.preference([0],label_key="between_labels")
    from unified_marl.rewards.service import reward_batch
    pair=data.preference([0,1])
    mapped=reward_batch(pair)
    torch.testing.assert_close(mapped["labels"][0],torch.tensor([0.5,0.5]))
    torch.testing.assert_close(mapped["global_labels"][0],torch.tensor([0.,1.]))


@pytest.mark.parametrize("name",["kou/bc","kou/cpl","kou/icq_ma","kou/iql","kou/cql","kou_old/cw_qmix"])
def test_offline_no_sampler_and_resume(name,raw_data,tmp_path,monkeypatch):
    import unified_marl.runners.episode_runner as runner
    monkeypatch.setattr(runner,"make_env",lambda *a,**k:(_ for _ in ()).throw(AssertionError("Offline created an environment")))
    opts={"training_updates":2,"schedule_updates":20,"off_batch_size":4,"batch_size":4,"rnn_hidden_dim":16}
    first=train(name,dataset=EpisodeDataset(raw_data),output_dir=tmp_path/"first",overrides=opts)
    resumed=train(resume=first.training_checkpoint,dataset=EpisodeDataset(raw_data),output_dir=tmp_path/"resumed",overrides={"training_updates":3})
    whole=train(name,dataset=EpisodeDataset(raw_data),output_dir=tmp_path/"whole",overrides=dict(opts,training_updates=3))
    assert first.env_steps==first.episodes==resumed.env_steps==0
    a,b=read_checkpoint(resumed.training_checkpoint),read_checkpoint(whole.training_checkpoint)
    for key,network in a["learner"]["extra"]["networks"].items():
        for weight,value in network.items(): torch.testing.assert_close(value,b["learner"]["extra"]["networks"][key][weight],atol=0,rtol=0)
    for weight,value in a["agent"].items(): torch.testing.assert_close(value,b["agent"][weight],atol=0,rtol=0)
    policy=load_policy(resumed.checkpoint)
    obs=raw_data["obs"][0,0].numpy()
    available=raw_data["avail_action"][0,0].numpy()
    action=policy.act(obs,available)
    policy.reset()
    np.testing.assert_array_equal(action,policy.act(obs,available))


def test_real_data_small_read_and_update(tmp_path):
    path=SOURCE/"MADiversePreferDatasets/3m/3m.th"
    data=EpisodeDataset(path,max_length=12)
    assert len(data)==400 and len(data.pairs)==200 and data.env_info["n_agents"]==3
    records=[]
    for name in ["kou/bc","kou/cpl"]:
        run=train(name,dataset=data,output_dir=tmp_path/name.replace("/","-"),overrides={"training_updates":2,"off_batch_size":4,"rnn_hidden_dim":16})
        assert run.updates==2 and run.env_steps==0 and run.parameter_delta_l2>0
        records.append(asdict(run))
    original=torch.load(path,map_location="cpu",mmap=True,weights_only=True)
    matching=(original["labels"]==1.0*(original["reward"][:200].sum(1)<original["reward"][200:].sum(1))).float().mean().item()
    (EVIDENCE/"real_data_validation.json").write_text(json.dumps({"result":"PASS","source":str(path),
        "read_episodes":400,"read_pairs":200,"small_batch":4,"policy_runs":records,
        "labels_match_unmasked_return_comparison_fraction":matching,"policy_effectiveness":"NOT_VALIDATED"},indent=2),encoding="utf-8")


def test_periodic_evaluation_requires_explicit_environment(raw_data,tmp_path):
    with pytest.raises(ValueError,match="explicit env"):
        train("kou/bc",dataset=EpisodeDataset(raw_data),output_dir=tmp_path/"missing",overrides={"training_updates":1,"off_batch_size":4,"eval_interval_updates":1})
    result=train("kou/bc",env="toy",env_args={"n_agents":3,"episode_limit":5,"state_last_action":False},dataset=EpisodeDataset(raw_data),output_dir=tmp_path/"evaluated",
        overrides={"training_updates":1,"off_batch_size":4,"eval_interval_updates":1,"rnn_hidden_dim":16})
    assert result.env_steps==0 and (Path(result.output_dir)/"evaluation-1.json").is_file()


@pytest.mark.parametrize("name",["hacpl_smac","maicpl_smac","maidpo_smac"])
def test_missing_source_learners_are_explicit(name):
    with pytest.raises(ValueError,match="missing learner"): load_config("kou/"+name)

@pytest.mark.parametrize("variant",["e2e","moderate_lambda","use_per","use_reward_hat"])
def test_unmigrated_training_protocols_do_not_silently_fall_back(variant):
    with pytest.raises(ValueError,match="additional training/data protocol"):
        load_config("kou/cql",overrides={variant:True})
