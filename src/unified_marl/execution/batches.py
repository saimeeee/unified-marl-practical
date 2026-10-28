"""Strict episode and inference trace contracts at the learner boundary."""
import torch
from unified_marl.components.episode_buffer import EpisodeBatch
from .protocol import digest

STEP_FIELDS=["obs","state","avail_actions","actions","actions_onehot","hidden_states","policy_version","action_log_probs","behavior_values"]

def step_digest(data): return digest({key:data[key] for key in STEP_FIELDS if key in data})

def validate_episode(tensors,length,scheme,groups,prep,info,trace,transitions=None):
    if type(length) is not int or not 1<=length<=info["episode_limit"]: raise ValueError("Invalid episode length")
    if len(trace)!=length+1: raise ValueError("Episode length does not match acknowledged inference steps")
    batch=EpisodeBatch(scheme,groups,1,length+1,preprocess=prep)
    if not isinstance(tensors,dict) or set(tensors)!=set(batch.data.transition_data): raise ValueError("Episode fields differ from negotiated schema")
    for key,value in tensors.items():
        target=batch[key]
        if not isinstance(value,torch.Tensor) or value.shape!=target.shape or value.dtype!=target.dtype or not torch.isfinite(value).all(): raise ValueError(f"Episode tensor {key} has wrong shape/dtype/value")
        target.copy_(value)
    if not torch.all(batch["filled"]==1): raise ValueError("Transported episodes must be cropped occupied slots, including bootstrap")
    for key in ["terminated","truncated"]:
        values=batch[key]
        if not torch.all((values==0)|(values==1)) or values[0,:length-1].any() or values[0,length].any(): raise ValueError("Invalid terminal time positions")
    if (batch["terminated"][0,length-1]+batch["truncated"][0,length-1]).item()!=1: raise ValueError("Episode must end in exactly one terminal/truncation flag")
    if batch["reward"][0,length].item()!=0: raise ValueError("Bootstrap slot cannot carry a transition reward")
    available=batch["avail_actions"]
    actions=batch["actions"]
    if not torch.all((available==0)|(available==1)) or not torch.all(available.sum(-1)>0): raise ValueError("Invalid episode availability")
    if not torch.all((actions>=0)&(actions<info["n_actions"])) or not torch.all(available.gather(-1,actions)): raise ValueError("Invalid episode actions")
    expected=torch.nn.functional.one_hot(actions.squeeze(-1),info["n_actions"]).float()
    if not torch.equal(expected,batch["actions_onehot"]): raise ValueError("One-hot actions disagree with action indices")
    for t,expected_hash in enumerate(trace):
        if step_digest({key:batch[key][:,t] for key in STEP_FIELDS if key in batch.scheme})!=expected_hash: raise ValueError("Episode observation/action/behavior data disagrees with inference trace")
    if transitions is not None:
        if len(transitions)!=length: raise ValueError("Transition acknowledgement count mismatch")
        for t,expected_hash in enumerate(transitions):
            if digest({key:batch[key][:,t] for key in ["reward","terminated","truncated","individual_rewards"] if key in batch.scheme})!=expected_hash: raise ValueError("Episode reward/termination time positions disagree with acknowledged transitions")
    return batch

def validate_transition(data,info,individual,bootstrap):
    keys={"reward","terminated","truncated"}|({"individual_rewards"} if individual else set())
    if not isinstance(data,dict) or set(data)!=keys: raise ValueError("Invalid transition fields")
    for key,value in data.items():
        shape=(1,info["n_agents"],1) if key=="individual_rewards" else (1,1)
        dtype=torch.uint8 if key in {"terminated","truncated"} else torch.float32
        if not isinstance(value,torch.Tensor) or tuple(value.shape)!=shape or value.dtype!=dtype or not torch.isfinite(value).all(): raise ValueError("Invalid transition dtype/shape/value")
    flags=[data[key].item() for key in ["terminated","truncated"]]
    if any(value not in {0,1} for value in flags) or sum(flags)!=int(bootstrap): raise ValueError("Transition terminal/truncation does not match bootstrap marker")
    return digest(data)

def trace_step(request,result):
    output=result["data"]
    actions=output["actions"]
    record=dict(obs=request["obs"],state=request["state"],avail_actions=request["available_actions"],actions=actions.unsqueeze(-1),
                actions_onehot=torch.nn.functional.one_hot(actions,request["available_actions"].shape[-1]).float(),
                hidden_states=output["hidden_states"],policy_version=torch.tensor([[result["version"]]],dtype=torch.long))
    if "action_log_probs" in output: record["action_log_probs"]=output["action_log_probs"].unsqueeze(-1)
    if "behavior_values" in output: record["behavior_values"]=output["behavior_values"]
    return step_digest(record)
