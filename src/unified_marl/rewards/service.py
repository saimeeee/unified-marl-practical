from dataclasses import dataclass, asdict
import importlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
import uuid
import numpy as np
import torch
from unified_marl.datasets import load_dataset, PreferenceBatch
from unified_marl.datasets.episodes import valid_mask
from unified_marl.checkpoints import atomic_save, rng_state, restore_rng

MODELS = {"MR", "NMR", "PrefTransformer", "MultiPrefTransformerDivide", "old_global", "old_reward_global", "old_new_global"}


def pad_time(value, length):
    if value.shape[1] == length: return value
    result = value.new_zeros((value.shape[0],length,*value.shape[2:]))
    result[:,:value.shape[1]] = value
    return result


def reward_batch(pair):
    a,b = pair.trajectory0,pair.trajectory1
    time = max(a.max_seq_length,b.max_seq_length)
    n = a["obs"].shape[2]
    result = {"labels": torch.cat([1-pair.labels,pair.labels],-1)}
    for side,batch,mask in [(0,a,pair.mask0),(1,b,pair.mask1)]:
        result[f"observations{side}"] = pad_time(batch["obs"],time)
        result[f"actions{side}"] = pad_time(batch["actions"],time)
        result[f"masks{side}"] = pad_time(mask,time).repeat(1,1,n)
        result[f"timesteps{side}"] = torch.arange(time,device=mask.device).unsqueeze(0).expand(batch.batch_size,-1)
        if "agent_ids" in batch: result[f"agent_ids{side}"] = batch["agent_ids"]
    for key in ["global_labels","between_labels"]:
        if key in pair.extra_labels:
            scalar = pair.extra_labels[key].float()
            result[key] = torch.cat([1-scalar,scalar],-1) if key == "global_labels" else torch.stack([1-scalar,scalar],-1)
    if "in_labels" in pair.extra_labels:
        result["in_labels0"],result["in_labels1"] = pair.extra_labels["in_labels"]
    for side in [0,1]:
        result[f"obs{side}"] = result[f"observations{side}"]
        result[f"action{side}"] = result[f"actions{side}"]
    return result


class RewardModel:
    def __init__(self, model_type, env_info, *, device="cpu", model_config=None, mode="preference"):
        if model_type not in MODELS: raise ValueError(f"Unsupported delivered reward model {model_type}")
        from unified_marl.api import check_device
        device = check_device(device)
        required = ["scipy"] if model_type.startswith("old_") else ["ml_collections", "transformers", "scipy"]
        missing = [name for name in required if importlib.util.find_spec(name) is None]
        if missing:
            raise ImportError("Reward models require optional dependencies: install unified-marl[reward]; missing " + ", ".join(missing))
        self.model_type,self.env_info,self.device,self.mode = model_type,env_info,str(device),mode
        self.model_config = dict(model_config or {})
        if mode not in {"preference","diverse"}: raise ValueError("Reward mode must be preference or diverse")
        if mode == "diverse" and model_type != "MultiPrefTransformerDivide":
            raise ValueError("Delivered diverse-label loss belongs to MultiPrefTransformerDivide only")
        self.steps = 0
        if model_type.startswith("old_"):
            self._construct_old()
            return
        package = "unified_marl.algorithms.kou.reward_models"
        wrapper_name = {"MultiPrefTransformerDivide":"MultiPrefTransformer"}.get(model_type,model_type)
        wrapper_class = getattr(importlib.import_module(package+"."+wrapper_name),wrapper_name)
        config = wrapper_class.get_default_config(self.model_config)
        self.resolved_model_config = dict(config)
        supplied = dict(observation_dim=env_info["obs_shape"],action_dim=env_info["n_actions"],action_type="Discrete",device=self.device)
        if model_type == "MR":
            cls = getattr(importlib.import_module(package+".q_function"),"FullyConnectedQFunction")
            self.network = cls(**supplied,inner_dim=config.inner_dim,action_embd_dim=config.action_embd_dim,
                               orthogonal_init="False",activations="relu",activation_final="none")
            self.native = wrapper_class(config,self.network,self.device)
        else:
            from transformers import GPT2Config
            config = GPT2Config(**dict(config))  # Configuration only; no remote model or tokenizer.
            length = self.model_config.get("max_episode_steps",env_info["episode_limit"])
            if model_type == "NMR":
                cls = getattr(importlib.import_module(package+".lstm"),"LSTMRewardModel")
                self.network = cls(config=config,**supplied,activation="relu",activation_final="none",max_episode_steps=length)
            elif model_type == "PrefTransformer":
                cls = getattr(importlib.import_module(package+".trajectory_gpt2"),"TransRewardModel")
                self.network = cls(config=config,**supplied,activation="relu",activation_final="none",max_episode_steps=length)
            else:
                cls = getattr(importlib.import_module(package+".encoder_decoder_divide"),"MultiTransRewardDivideModel")
                self.network = cls(config=config,**supplied,n_agent=env_info["n_agents"],max_episode_steps=length)
            self.native = wrapper_class(config,self.network,self.device)
        self.output_semantics = "individual" if model_type == "MultiPrefTransformerDivide" else "team"

    def _construct_old(self):
        from unified_marl.config import resource, merge
        source = "reward_model_new" if self.model_type == "old_new_global" else "reward_model"
        if source.endswith("new"):
            raise ValueError("The delivered new reward constructor unconditionally loads its reference policy; that checkpoint and agent-state labeling integration are unavailable, so no reference policy is substituted")
        module = importlib.import_module("unified_marl.algorithms.kou.old.components."+source)
        cfg = merge(resource("kou/old/qmix_w_globalRM.yaml"), resource("kou/old/pref_qmix_new.yaml") if source.endswith("new") else {})
        cfg = merge(cfg,self.model_config)
        cfg.update(self.env_info)
        cfg.update(device=self.device,use_cuda=self.device.startswith("cuda"), action_shape=1,
            agent_state_shape=cfg.get("agent_state_shape",self.env_info["obs_shape"]),
            global_preference_type="provided_labels",local_preference_type="provided_labels")
        if source.endswith("new"):
            cfg.update(use_global_reward=True,use_local_reward=False,actions_onehot=False)
        args = SimpleNamespace(**cfg)
        from unified_marl.utils.logging import Logger
        if self.model_type == "old_global": self.native = module.GlobalRewardModel(args,None)
        else: self.native = module.RewardModel(args,None,Logger())
        self.network = torch.nn.ModuleList(self.native.global_ensemble if source.endswith("new") else self.native.ensemble)
        self.resolved_model_config = cfg
        self.output_semantics = "team"

    def _old_inputs(self, batch):
        args = self.native.args
        if getattr(args,"actions_onehot",False):
            actions = batch["actions_onehot"].reshape(batch.batch_size,batch.max_seq_length,-1)
        else:
            actions = batch["actions"].squeeze(-1).float()
        if getattr(args,"state_or_obs",True): return torch.cat([batch["state"],actions],-1)
        ids = torch.eye(args.n_agents,device=self.device)[None,None].expand(batch.batch_size,batch.max_seq_length,-1,-1)
        local_actions = batch["actions_onehot"] if args.actions_onehot else batch["actions"].float()
        return torch.cat([batch["obs"],local_actions,ids],-1).reshape(batch.batch_size,batch.max_seq_length,-1)

    def train_step(self,pair):
        pair = pair.to(self.device)
        if self.model_type.startswith("old_"):
            self.native._integration_losses={}
            a,b=pair.trajectory0,pair.trajectory1
            labels=torch.cat([1-pair.labels,pair.labels],-1)
            if self.model_type == "old_new_global":
                queries=[]
                for batch,mask in [(a,pair.mask0),(b,pair.mask1)]:
                    queries.append({"state":[batch["state"]],"actions":[batch["actions"]],"mask":[mask]})
                accuracy=self.native.train_global_reward(labels,queries)
            else:
                # Fill the original segment buffer with the provided pairs;
                # no scripted labels or reference policies are substituted.
                size=a.batch_size
                self.native.buffer_index=size
                self.native.buffer_full=False
                self.native.train_batch_size=size
                self.native.buffer["state_segment1"]=list(self._old_inputs(a))
                self.native.buffer["state_segment2"]=list(self._old_inputs(b))
                self.native.buffer["label"]=list(labels)
                self.native.buffer["mask1"]=[m.unsqueeze(0) for m in pair.mask0]
                self.native.buffer["mask2"]=[m.unsqueeze(0) for m in pair.mask1]
                method=self.native.train_reward if self.model_type=="old_global" else self.native.train_global_reward
                accuracy=method()
            self.steps+=1
            # Native methods expose accuracy; compute their identical CE/KL
            # objective for a finite scalar report after the update.
            scores=[]
            with torch.no_grad():
                for member in range(len(self.network)):
                    left=(self.network[member](self._old_inputs(a))*pair.mask0).sum(1)
                    right=(self.network[member](self._old_inputs(b))*pair.mask1).sum(1)
                    scores.append(float(self.native.loss_func(torch.cat([left,right],-1),labels)))
            metrics={"loss":self.native._integration_losses["loss"],"loss_after_update":sum(scores),"accuracy":float(np.mean(accuracy))}
        else:
            values=reward_batch(pair)
            if self.mode=="diverse":
                if not {"global_labels","between_labels","in_labels0","in_labels1"} <= values.keys():
                    raise ValueError("Diverse reward training requires separate global, between and within-trajectory ranking labels")
                metrics=self.native.train_diverse(values)
            else:
                if "agent_ids0" not in values: raise ValueError("Delivered reward objective requires the dataset's actual agent_ids")
                metrics=self.native.train(values)
            metrics={k:float(v) for k,v in metrics.items()}
            self.steps+=1
        if not all(math.isfinite(v) for v in metrics.values()): raise FloatingPointError("Non-finite reward metrics")
        return metrics

    @torch.no_grad()
    def predict(self,batch,*,mode="source_offline"):
        batch=batch.to(self.device)
        self.network.eval()
        if self.model_type.startswith("old_"):
            return torch.stack([net(self._old_inputs(batch)) for net in self.network]).mean(0)
        mask=valid_mask(batch)
        pair=PreferenceBatch(batch,batch,torch.zeros(batch.batch_size,1,device=self.device),mask,mask,
                             torch.arange(batch.batch_size),torch.arange(batch.batch_size))
        values=reward_batch(pair)
        if mode=="source_offline":
            output=self.native.get_reward_for_offline(values)[0]
            if output.shape[1] != batch.max_seq_length: raise ValueError("Source offline predictor collapsed the time dimension")
            return output
        if mode!="native": raise ValueError("Prediction mode must be source_offline or native")
        if self.model_type=="MR": return self.network(batch["obs"],batch["actions"]).unsqueeze(-1)
        if self.model_type=="NMR": return self.network(batch["obs"],batch["actions"])[0]
        result=self.network(batch["obs"],batch["actions"],values["timesteps0"],attn_mask=values["masks0"])
        return result[0]["value"] if self.model_type=="PrefTransformer" else result

    def save(self,path,*,training=False):
        payload={"reward_format":1,"kind":"reward-training" if training else "reward", "model_type":self.model_type,
            "env_info":self.env_info,"model_config":self.model_config,"mode":self.mode,"steps":self.steps,
            "network":self.network.state_dict(),"output_semantics":self.output_semantics}
        # PrefTransformer keeps GPT2 blocks in a Python list. Preserve their
        # weights/modes without registering them in the optimizer (which would
        # change the delivered training algorithm).
        payload["unregistered_modules"]={}
        for module_path,module in self.network.named_modules():
            for name,value in vars(module).items():
                if isinstance(value,list) and value and all(isinstance(item,torch.nn.Module) for item in value):
                    payload["unregistered_modules"][module_path+":"+name]={"states":[item.state_dict() for item in value],
                        "training":[item.training for item in value]}
        if training:
            payload["network_training"] = self.network.training
            payload["optimizer"]=(self.native.global_optimizer if self.model_type=="old_new_global" else self.native.optimizer).state_dict()
            scheduler=getattr(self.native,"scheduler",None)
            payload["scheduler"]=scheduler.state_dict() if scheduler is not None else None
            payload["rng"]=rng_state()
        return atomic_save(payload,path)


def load_reward_model(path,device="cpu",*,for_training=False):
    payload=torch.load(path,map_location="cpu",weights_only=True)
    if payload.get("reward_format")!=1: raise ValueError("Unsupported reward checkpoint")
    if for_training and payload["kind"] != "reward-training":
        raise ValueError("Reward continuation requires reward-training.pt, not a prediction checkpoint")
    model=RewardModel(payload["model_type"],payload["env_info"],device=device,model_config=payload["model_config"],mode=payload["mode"])
    model.network.load_state_dict(payload["network"])
    modules=dict(model.network.named_modules())
    for key,value in payload.get("unregistered_modules",{}).items():
        module_path,name=key.rsplit(":",1)
        for module,state,training in zip(getattr(modules[module_path],name),value["states"],value["training"]):
            module.load_state_dict(state)
            module.train(training)
    model.steps=payload["steps"]
    model.network.eval()
    if payload["kind"]=="reward-training":
        optimizer=model.native.global_optimizer if model.model_type=="old_new_global" else model.native.optimizer
        optimizer.load_state_dict(payload["optimizer"])
        if payload["scheduler"] is not None: model.native.scheduler.load_state_dict(payload["scheduler"])
        if hasattr(model.native,"_total_steps"): model.native._total_steps=model.steps
    if for_training:
        model.network.train(payload.get("network_training", True))
        restore_rng(payload["rng"])
    return model


@dataclass(frozen=True)
class RewardTrainResult:
    model_type:str
    updates:int
    output_dir:str
    checkpoint:str
    training_checkpoint:str
    metrics_path:str
    parameter_delta_l2:float
    output_semantics:str
    metrics:dict


def train_reward_model(model_type, dataset, *, updates=100,batch_size=32,output_dir=None,device="cpu",seed=1,
                       model_config=None,dataset_options=None,mode="preference",trust_filter=True):
    from unified_marl.api import seed_all,check_device,_write_json
    from unified_marl.utils.logging import Logger
    if type(updates) is not int or updates<1 or type(batch_size) is not int or batch_size<1: raise ValueError("Positive integer reward updates and pair batch_size required")
    if type(seed) is not int or not 0 <= seed < 2**32: raise ValueError("seed must be an integer in [0, 2**32)")
    data=load_dataset(dataset,**(dataset_options or {}))
    if data.pairs is None or batch_size>len(data.pairs): raise ValueError("Insufficient preference pairs")
    directory=Path(output_dir or Path.cwd()/"results"/("reward-"+uuid.uuid4().hex[:8])).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/"reward.pt").exists() or (directory/"metrics.jsonl").exists(): raise FileExistsError(directory)
    seed_all(seed)
    old_threads=torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model=RewardModel(model_type,data.env_info,device=check_device(device),model_config=model_config,mode=mode)
        before=[p.detach().clone() for p in model.network.parameters()]
        logger=Logger(directory/"metrics.jsonl")
        selected_ids=[]
        trust_history=[]
        epoch=0
        interval=0
        trust_pool=[]
        for step in range(updates):
            if mode=="diverse" and trust_filter:
                if not selected_ids:
                    all_pairs=data.preference(np.arange(len(data.pairs)),trim=False).to(device)
                    with torch.no_grad():
                        divergences=model.native.get_KL(reward_batch(all_pairs))
                        thresholds=[]
                        for divergence in divergences:
                            if epoch==0: threshold=1000.0
                            else:
                                beta=max(1,-(3-1)/30*epoch+3)
                                threshold=-torch.log(divergence.max()+1e-8)+0.5*divergence.max()+min(beta*divergence.var(),3.0)
                            thresholds.append(threshold)
                        trusted=torch.ones(len(data.pairs),dtype=torch.bool,device=device)
                        for divergence,threshold in zip(divergences,thresholds): trusted &= divergence<threshold
                        trust_pool=torch.where(trusted)[0].cpu().numpy().tolist()
                    if not trust_pool: raise RuntimeError("Source diverse trust filter selected zero pairs; no synthetic fallback is used")
                    selected_ids=list(range(math.ceil(len(trust_pool)/batch_size)))
                    trust_history.append({"epoch":epoch,"pair_ids":trust_pool,"thresholds":[float(x) for x in thresholds]})
                    epoch+=1
                # The source reshuffles inside each minibatch iteration; retain
                # that behavior instead of silently changing the curriculum.
                position=selected_ids.pop(0)
                shuffled=np.random.permutation(trust_pool)
                ids=shuffled[position*batch_size:min((position+1)*batch_size,len(trust_pool))]
            else:
                ids=np.random.choice(len(data.pairs),batch_size,replace=False)
            pair=data.preference(ids,trim=False)
            metrics=model.train_step(pair)
            logger.record(metrics,step+1)
        delta=sum(float((a-b.detach()).square().sum()) for a,b in zip(before,model.network.parameters()))**0.5
        if not math.isfinite(delta): raise FloatingPointError("Non-finite reward parameter change")
        checkpoint=model.save(directory/"reward.pt")
        training=model.save(directory/"reward-training.pt",training=True)
        result=RewardTrainResult(model_type,updates,str(directory),checkpoint,training,str(directory/"metrics.jsonl"),delta,model.output_semantics,metrics)
        _write_json(directory/"config.json",dict(model_type=model_type,model_config=model_config,dataset_options=dataset_options,
            dataset=data.source,updates=updates,pair_batch_size=batch_size,mode=mode,trust_filter=trust_filter,device=device,seed=seed,env_info=data.env_info))
        if trust_history: _write_json(directory/"trust_epochs.json",trust_history)
        _write_json(directory/"result.json",asdict(result))
        return result
    finally:
        torch.set_num_threads(old_threads)
