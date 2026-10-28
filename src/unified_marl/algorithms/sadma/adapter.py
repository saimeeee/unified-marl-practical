"""SADMA controller/policy over canonical EpisodeBatch, with explicit field mapping."""
import math
from types import SimpleNamespace
import torch
from unified_marl.learners.base import LearnerAPI
from .native.policy import policy_REGISTRY


def native_batch(batch):
    aliases={"state":"states","reward":"rewards","avail_actions":"available_actions"}
    data={aliases.get(key,key):batch[key].clone() for key in batch.scheme}
    data["actions"]=data["actions"].squeeze(-1)
    if "action_log_probs" in data: data["action_log_probs"]=data["action_log_probs"].squeeze(-1)
    return data


class SADMAController:
    def __init__(self,scheme,groups,args):
        self.args=args
        self.native=policy_REGISTRY[args.sadma_algorithm]["mac"](args)
        self.action_selector=SimpleNamespace(epsilon=0.0)
        self.critic=None

    @property
    def agent(self): return self.native.agent
    def parameters(self): return self.agent.parameters()
    def to(self,device):
        self.native.to(device)
        return self
    def init_hidden(self,batch_size):
        self.hidden_states=self.native.init_hidden_states(batch_size)
        if self.critic is not None:
            shape=(batch_size,self.args.n_agents,self.args.hidden_dim) if "independent_obs" in self.args.critic_inputs else (batch_size,self.args.hidden_dim)
            self.critic_hidden=torch.zeros(shape,device=self.native.device)
    def state_dict(self):
        return {"agent":self.agent.state_dict(),"agent_version":self.native.agent_version.clone()}
    def load_state_dict(self,state):
        self.agent.load_state_dict(state["agent"])
        self.native.agent_version.copy_(state["agent_version"])
    @torch.no_grad()
    def select_actions(self,batch,t_ep,t_env,bs=slice(None),test_mode=False):
        data={"obs":batch["obs"][bs,t_ep],"available_actions":batch["avail_actions"][bs,t_ep],"hidden_states":self.hidden_states[bs]}
        if not torch.all(data["available_actions"].sum(-1)>0): raise ValueError("SADMA received an empty action mask")
        if "last_action" in self.args.agent_inputs:
            data["last_actions"]=torch.zeros_like(data["available_actions"]) if t_ep==0 else batch["actions_onehot"][bs,t_ep-1]
        result=self.native.select_actions(data,t_env,test_mode)
        actions=result["actions"].to(self.native.device)
        if not torch.all(data["available_actions"].gather(-1,actions.unsqueeze(-1))): raise RuntimeError("SADMA selected an unavailable action")
        self.rollout_data={"policy_version":torch.full((actions.shape[0],1),int(self.native.agent_version.item()),device=actions.device,dtype=torch.long)}
        if "hidden_states" in result:
            self.hidden_states[bs]=result["hidden_states"].to(self.native.device)
        self.rollout_data["hidden_states"]=self.hidden_states[bs].clone()
        if "action_log_probs" in result:
            self.rollout_data["action_log_probs"]=result["action_log_probs"].to(self.native.device).unsqueeze(-1)
        if self.critic is not None:
            critic_data={"obs":data["obs"],"states":batch["state"][bs,t_ep],"hidden_states":self.critic_hidden[bs]}
            if "last_action" in self.args.critic_inputs:
                critic_data["last_actions"]=torch.zeros_like(data["available_actions"]) if t_ep==0 else batch["actions_onehot"][bs,t_ep-1]
            values,hidden=self.critic(critic_data)
            if hidden is not None: self.critic_hidden[bs]=hidden
            self.rollout_data["behavior_values"]=values
        return actions


class SADMALearner(LearnerAPI):
    def __init__(self,mac,scheme,logger,args):
        self.mac,self.args,self.logger=mac,args,logger
        self.native=policy_REGISTRY[args.sadma_algorithm]["policy"](args,mac.native)
        if args.sadma_algorithm=="ppo": mac.critic=self.native.critic
    def to(self,device):
        self.mac.to(device)
        return self
    def update(self,batch,context):
        if self.args.sadma_algorithm=="ppo":
            mask=batch["filled"].bool().expand_as(batch["policy_version"])
            versions=batch["policy_version"][mask]
            if not torch.all(versions==self.mac.native.agent_version.item()):
                raise ValueError("PPO batch belongs to an old policy; synchronous updates require fresh rollouts")
        metrics=self.native.train(native_batch(batch),context.episodes,context.env_steps)
        result={k:float(v.detach().cpu()) if isinstance(v,torch.Tensor) else float(v) for k,v in metrics.items()}
        result["loss"]=float(self.native._integration_loss)
        result["policy_version"]=int(self.mac.native.agent_version.item())
        if not all(math.isfinite(value) for value in result.values()): raise FloatingPointError("Non-finite SADMA metrics")
        return result
    def extra_state(self):
        modules={key:value.state_dict() for key,value in vars(self.native).items() if isinstance(value,torch.nn.Module)}
        controllers={"target_mac":self.native.target_mac.agent_state_dict} if hasattr(self.native,"target_mac") else {}
        return {"modules":modules,"controllers":controllers,"optimizer":self.native.optimizer.state_dict(),
                "lr":self.native.lr,"last_target_update_episode":getattr(self.native,"last_target_update_episode",0)}
    def load_extra_state(self,state):
        for name,value in state["modules"].items(): getattr(self.native,name).load_state_dict(value)
        if "target_mac" in state["controllers"]: self.native.target_mac.load_agent_state_dict(state["controllers"]["target_mac"])
        self.native.optimizer.load_state_dict(state["optimizer"])
        self.native.lr=state["lr"]
        if hasattr(self.native,"last_target_update_episode"): self.native.last_target_update_episode=state["last_target_update_episode"]
