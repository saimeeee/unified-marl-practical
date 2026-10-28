"""Environment-only role. Inference and all parameter ownership stay remote."""
import time
import traceback
import torch
from .protocol import RPCClient,zmq_module
from .settings import ZMQSettings

class StopRequested(Exception): pass

class RemoteController:
    def __init__(self,client,settings,config,info,schema,worker_id):
        self.client,self.settings,self.config,self.info,self.schema,self.worker_id=client,settings,config,info,schema,worker_id
        self.session=None
        self.infers=0
    def call(self,kind,**data):
        reply=self.client.call(dict(kind=kind,job_id=self.settings.job_id,schema_id=self.schema,worker_id=self.worker_id,**data))
        if reply["kind"]=="STOP": raise StopRequested()
        return reply
    def init_hidden(self,batch_size):
        if batch_size!=1: raise ValueError("Remote sampler collects one environment")
        n,h=self.info["n_agents"],self.config["hidden_dim"]
        self.hidden=torch.zeros(1,n,h)
        shape=(1,n,h) if "independent_obs" in self.config.get("critic_inputs",[]) else (1,h)
        self.critic_hidden=torch.zeros(shape)
    @torch.no_grad()
    def select_actions(self,batch,t_ep,t_env,bs=slice(None),test_mode=False):
        if test_mode: raise ValueError("Distributed evaluation uses the saved policy through evaluate(), not a training sampler")
        if self.settings.fault_worker==self.worker_id and self.infers>=self.settings.fault_after_infers:
            if self.settings.fault_kind=="stall": time.sleep(self.settings.timeout_ms/1000*3)
            elif self.settings.fault_kind=="exit": raise SystemExit(23)
            else: raise RuntimeError("Injected sampler failure for lifecycle validation")
        data=dict(obs=batch["obs"][:,t_ep],state=batch["state"][:,t_ep],available_actions=batch["avail_actions"][:,t_ep],hidden_states=self.hidden,
                  last_actions=torch.zeros_like(batch["actions_onehot"][:,t_ep]) if t_ep==0 else batch["actions_onehot"][:,t_ep-1])
        if self.config["sadma_algorithm"]=="ppo": data["critic_hidden"]=self.critic_hidden
        bootstrap=bool(t_ep>0 and (batch["terminated"][0,t_ep-1].item() or batch["truncated"][0,t_ep-1].item()))
        transition=None if t_ep==0 else {key:batch[key][:,t_ep-1] for key in ["reward","terminated","truncated","individual_rewards"] if key in batch.scheme}
        reply=self.call("INFER",session=self.session,t=t_ep,bootstrap=bootstrap,data=data,transition=transition)
        result=reply["data"]
        self.hidden=result["hidden_states"]
        if "critic_hidden" in result: self.critic_hidden=result["critic_hidden"]
        self.rollout_data={"hidden_states":self.hidden,"policy_version":torch.tensor([[reply["version"]]],dtype=torch.long)}
        if "action_log_probs" in result: self.rollout_data["action_log_probs"]=result["action_log_probs"].unsqueeze(-1)
        if "behavior_values" in result: self.rollout_data["behavior_values"]=result["behavior_values"]
        self.infers+=1
        return result["actions"]

def sampler_worker(trainer_endpoint,worker_id,settings_values=None,ready=None):
    from types import SimpleNamespace
    from unified_marl.api import make_scheme,seed_all
    from unified_marl.runners.episode_runner import EpisodeRunner
    from unified_marl.utils.logging import Logger
    settings=ZMQSettings.parse(dict(settings_values or {},trainer_endpoint=trainer_endpoint))
    client=RPCClient(trainer_endpoint,settings,dealer=True,identity=f"sampler-{worker_id}-{time.time_ns()}")
    runner=None
    error=None
    controller=None
    try:
        if ready: ready.send(dict(kind="PROCESS_STARTED",worker_id=worker_id))
        hello=client.call(dict(kind="HELLO",job_id=settings.job_id,worker_id=worker_id))
        if hello["kind"]!="WELCOME": raise ValueError("Unexpected trainer greeting")
        config,info,schema=hello["config"],hello["env_info"],hello["schema_id"]
        torch.set_num_threads(config.get("num_threads",1))
        settings.seed_stride=config["zmq"]["seed_stride"]
        seed=config["seed"]+(worker_id+1)*settings.seed_stride
        seed_all(seed)
        config=dict(config,device="cpu",env_args=dict(config["env_args"],seed=seed),batch_size_run=1)
        runner=EpisodeRunner(SimpleNamespace(**config),Logger())
        if runner.get_env_info()!=info: raise ValueError("Sampler environment metadata differs from trainer")
        scheme,groups,prep=make_scheme(info,config)
        controller=RemoteController(client,settings,config,info,schema,worker_id)
        runner.setup(scheme,groups,prep,controller)
        controller.call("ENV_READY",env_info=info,seed=seed)
        while True:
            permit=controller.call("READY")
            if permit["kind"]!="COLLECT": raise ValueError("Unexpected collection permit")
            controller.session=permit["session"]
            batch=runner.run()
            controller.call("EPISODE",session=controller.session,length=runner.last_lengths[0],tensors=batch.data.transition_data)
    except StopRequested:
        pass
    except BaseException as exc:
        error=f"{type(exc).__name__}: {exc}"
        if ready:
            try: ready.send(dict(kind="ERROR",error=error,traceback=traceback.format_exc(),worker_id=worker_id))
            except (OSError,BrokenPipeError): pass
        raise
    finally:
        try:
            if runner is not None: runner.close_env()
            if controller is not None:
                try: controller.call("BYE",error=error,env_closed=runner is not None)
                except Exception: pass
            if ready:
                try: ready.send(dict(kind="STOPPED",worker_id=worker_id,env_closed=runner is not None,error=error))
                except (OSError,BrokenPipeError): pass
        finally:
            client.close()
            if ready: ready.close()
