import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from test_offline import raw_data,cpu,original,EVIDENCE,SOURCE
from unified_marl.datasets import EpisodeDataset
from unified_marl.rewards.service import RewardModel,reward_batch,load_reward_model,train_reward_model

MODELS=["MR","NMR","PrefTransformer","MultiPrefTransformerDivide","old_global","old_reward_global"]


def model_options(name):
    return {"reward_hidden_size":16,"ensemble_size":2} if name.startswith("old_") else {
        "inner_dim":32,"embd_dim":32,"action_embd_dim":8,"pref_attn_embd_dim":32,"n_head":4,"n_layer":1}


def reference_model(name,model):
    package="components.MAPT.mat.algorithms.reward_model.models"
    if name.startswith("old_"):
        module=importlib.import_module("components.reward_model")
        args=SimpleNamespace(**model.resolved_model_config)
        from unified_marl.utils.logging import Logger
        return module.GlobalRewardModel(args,None) if name=="old_global" else module.RewardModel(args,None,Logger())
    wrapper_name="MultiPrefTransformer" if name=="MultiPrefTransformerDivide" else name
    wrapper=getattr(importlib.import_module(package+"."+wrapper_name),wrapper_name)
    config=copy.deepcopy(model.native.config)
    info=model.env_info
    supplied=dict(observation_dim=info["obs_shape"],action_dim=info["n_actions"],action_type="Discrete",device="cpu")
    if name=="MR":
        cls=getattr(importlib.import_module(package+".q_function"),"FullyConnectedQFunction")
        net=cls(**supplied,inner_dim=config.inner_dim,action_embd_dim=config.action_embd_dim,
                orthogonal_init="False",activations="relu",activation_final="none")
    elif name=="NMR":
        cls=getattr(importlib.import_module(package+".lstm"),"LSTMRewardModel")
        net=cls(config=config,**supplied,activation="relu",activation_final="none",max_episode_steps=info["episode_limit"])
    elif name=="PrefTransformer":
        cls=getattr(importlib.import_module(package+".trajectory_gpt2"),"TransRewardModel")
        net=cls(config=config,**supplied,activation="relu",activation_final="none",max_episode_steps=info["episode_limit"])
    else:
        cls=getattr(importlib.import_module(package+".encoder_decoder_divide"),"MultiTransRewardDivideModel")
        net=cls(config=config,**supplied,n_agent=info["n_agents"],max_episode_steps=info["episode_limit"])
    return wrapper(config,net,"cpu")


@pytest.mark.parametrize("name",MODELS+["MultiPrefTransformerDivide/diverse"])
def test_reward_source_loss_update_and_prediction_parity(name,raw_data,monkeypatch):
    diverse=name.endswith("/diverse")
    model_type=name.split("/")[0]
    data=EpisodeDataset(raw_data)
    pair=data.preference([2,3],trim=False)
    torch.manual_seed(2026)
    model=RewardModel(model_type,data.env_info,model_config=model_options(model_type),mode="diverse" if diverse else "preference")
    before=[p.detach().clone() for p in model.network.parameters()]
    with original(monkeypatch,"old" if name.startswith("old_") else "icq") as folder:
        torch.manual_seed(2026)
        old=reference_model(model_type,model)
        values=reward_batch(pair)
        torch.manual_seed(31)
        np.random.seed(31)
        if model_type.startswith("old_"):
            size=pair.trajectory0.batch_size
            old.buffer_index=size
            old.buffer_full=False
            old.train_batch_size=size
            for side,batch,mask in [(1,pair.trajectory0,pair.mask0),(2,pair.trajectory1,pair.mask1)]:
                old.buffer[f"state_segment{side}"]=list(model._old_inputs(batch))
                old.buffer[f"mask{side}"]=[m.unsqueeze(0) for m in mask]
            old.buffer["label"]=list(values["labels"])
            with torch.no_grad():
                reference_loss=0.0
                for net in old.ensemble:
                    left=(net(model._old_inputs(pair.trajectory0))*pair.mask0).sum(1)
                    right=(net(model._old_inputs(pair.trajectory1))*pair.mask1).sum(1)
                    reference_loss+=float(old.loss_func(torch.cat([left,right],-1),values["labels"]))
            old.train_reward() if model_type=="old_global" else old.train_global_reward()
            old_net=torch.nn.ModuleList(old.ensemble)
        else:
            old_metrics=old.train_diverse(copy.deepcopy(values)) if diverse else old.train(copy.deepcopy(values))
            reference_loss=float(old_metrics["loss"])
            old_net=getattr(old,{"MR":"rf","NMR":"lstm"}.get(model_type,"trans"))
        torch.manual_seed(31)
        np.random.seed(31)
        metrics=model.train_step(copy.deepcopy(pair))
        error=abs(metrics["loss"]-reference_loss)
        assert error<=1e-6
        parameter_error=0.0
        for key,value in model.network.state_dict().items():
            expected=old_net.state_dict()[key]
            torch.testing.assert_close(value,expected,atol=1e-7,rtol=1e-6)
            parameter_error=max(parameter_error,float((value-expected).abs().max()))
        torch.manual_seed(200)
        actual=model.predict(pair.trajectory0)
        torch.manual_seed(200)
        if model_type.startswith("old_"): expected=old.r_hat(model._old_inputs(pair.trajectory0))
        else: expected=old.get_reward_for_offline(reward_batch(type(pair)(pair.trajectory0,pair.trajectory0,pair.labels,pair.mask0,pair.mask0,pair.indices0,pair.indices0)))[0]
        torch.testing.assert_close(actual,expected,atol=1e-7,rtol=1e-6)
        prediction_error=float((actual-expected).abs().max())
    delta=sum(float((a-b.detach()).square().sum()) for a,b in zip(before,model.network.parameters()))**0.5
    assert delta>0
    (EVIDENCE/("parity-reward-"+name.replace("/","-")+".json")).write_text(json.dumps({"result":"PASS",
        "model":name,"source":str(folder),"loss_abs_error":error,"updated_network_max_abs_error":parameter_error,
        "source_offline_prediction_max_abs_error":prediction_error,"parameter_delta_l2":delta,
        "source_semantics":model.output_semantics},indent=2),encoding="utf-8")


@pytest.mark.parametrize("name",MODELS)
def test_reward_save_load_preserves_prediction_and_optimizer(name,raw_data,tmp_path):
    result=train_reward_model(name,EpisodeDataset(raw_data),updates=2,batch_size=2,output_dir=tmp_path/name,
                              model_config=model_options(name))
    assert result.parameter_delta_l2>0
    model=load_reward_model(result.checkpoint)
    full=load_reward_model(result.training_checkpoint)
    batch=EpisodeDataset(raw_data).batch([0,1],trim=False)
    torch.manual_seed(200)
    actual=model.predict(batch)
    torch.manual_seed(200)
    expected=full.predict(batch)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    assert full.steps==2
    assert len(full.native.optimizer.state)>0
    expected_ndim=4 if name=="MultiPrefTransformerDivide" else 3
    assert model.predict(batch).ndim==expected_ndim


def test_diverse_curriculum_uses_separate_labels(raw_data,tmp_path):
    run=train_reward_model("MultiPrefTransformerDivide",EpisodeDataset(raw_data),updates=2,batch_size=2,
        output_dir=tmp_path/"diverse",model_config=model_options("MultiPrefTransformerDivide"),mode="diverse")
    assert run.parameter_delta_l2>0
    history=json.loads((Path(run.output_dir)/"trust_epochs.json").read_text())
    assert len(history)==1 and len(history[0]["pair_ids"])==4
    assert history[0]["thresholds"]==[1000.,1000.,1000.]
    with pytest.raises(RuntimeError,match="zero pairs"):
        train_reward_model("MultiPrefTransformerDivide",EpisodeDataset(raw_data),updates=3,batch_size=2,
            output_dir=tmp_path/"source-zero",model_config=model_options("MultiPrefTransformerDivide"),mode="diverse")
    (EVIDENCE/"diverse_curriculum.json").write_text(json.dumps({"result":"PASS","scope":"synthetic; original KL and threshold equations; initial epoch only",
        "history":history,"second_epoch":"source filter selected zero pairs; explicit error, no fallback"}),encoding="utf-8")


def test_old_new_reward_reference_asset_is_not_substituted(raw_data):
    with pytest.raises(ValueError,match="reference policy"):
        RewardModel("old_new_global",EpisodeDataset(raw_data).env_info)


def test_nmr_source_degenerate_short_mask_retained(raw_data):
    # Source reuses the left prediction for the right loss. Equal masks make
    # both preference logits identical and cancel gradients. Do not "fix" it.
    raw_data["filled"].fill_(1)
    raw_data["terminated"].zero_()
    data=EpisodeDataset(raw_data)
    model=RewardModel("NMR",data.env_info,model_config=model_options("NMR"))
    before=[p.detach().clone() for p in model.network.parameters()]
    model.train_step(data.preference([0,1],trim=False))
    assert all(torch.equal(a,b) for a,b in zip(before,model.network.parameters()))
