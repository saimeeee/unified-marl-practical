"""Single-owner ZMQ inference worker with acknowledged parameter snapshots."""
import traceback
from collections import deque
import torch
from .protocol import RPCClient,bind,configure,decode,digest,encode,schema_id,zmq_module
from .settings import ZMQSettings

def snapshot(learner):
    native=learner.native
    body={"actor":learner.mac.agent.state_dict(),"critic":native.critic.state_dict() if hasattr(native,"critic") else None,
          "version":int(learner.mac.native.agent_version.item())}
    return dict(body,digest=digest(body))

class InferenceEngine:
    def __init__(self,config,info,device="cpu",seed=None):
        from unified_marl.api import make_args,check_device,seed_all
        from unified_marl.algorithms.sadma.native.policy import policy_REGISTRY
        from unified_marl.algorithms.sadma.native.policy.mappo.ppo_net import critic_REGISTRY
        if not config.get("sadma_algorithm") in {"ppo","qmix"}: raise ValueError("Only SADMA policies are accepted by this inference role")
        torch.set_num_threads(config.get("num_threads",1))
        seed_all(config["seed"] if seed is None else seed)
        self.config,self.info=config,info
        self.device=check_device(device)
        self.args=make_args(dict(config,device=self.device,train_device=self.device),info)
        self.mac=policy_REGISTRY[config["sadma_algorithm"]]["mac"](self.args)
        self.mac.to(self.device)
        self.critic=critic_REGISTRY[self.args.critic_type](self.args).to(self.device) if config["sadma_algorithm"]=="ppo" else None
        self.version=-1
        self.digest=None
        self.calls=0
        self.syncs=deque(maxlen=64)
        self.sync_count=0

    def synchronize(self,state):
        body={key:state[key] for key in ["actor","critic","version"]}
        if type(body["version"]) is not int or body["version"]<self.version or digest(body)!=state["digest"]: raise ValueError("Invalid/regressing parameter snapshot")
        if body["version"]==self.version and state["digest"]!=self.digest: raise ValueError("Same policy version has different parameters")
        self.mac.agent.load_state_dict(body["actor"],strict=True)
        self.mac.agent_version.fill_(body["version"])
        if self.critic is not None: self.critic.load_state_dict(body["critic"],strict=True)
        self.version,self.digest=body["version"],state["digest"]
        self.syncs.append(dict(version=self.version,digest=self.digest))
        self.sync_count+=1
        return dict(kind="SYNCED",version=self.version,digest=self.digest)

    @torch.no_grad()
    def infer(self,data,steps):
        n,a,h=self.info["n_agents"],self.info["n_actions"],self.args.hidden_dim
        shapes={"obs":(1,n,self.info["obs_shape"]),"state":(1,self.info["state_shape"]),"available_actions":(1,n,a),
                "hidden_states":(1,n,h),"last_actions":(1,n,a)}
        for key,shape in shapes.items():
            if key not in data or not isinstance(data[key],torch.Tensor) or tuple(data[key].shape)!=shape: raise ValueError(f"Inference {key} shape mismatch")
            expected=torch.int32 if key=="available_actions" else torch.float32
            if data[key].dtype!=expected or not torch.isfinite(data[key]).all(): raise ValueError(f"Inference {key} dtype/value mismatch")
        available=data["available_actions"]
        if not torch.all((available==0)|(available==1)) or not torch.all(available.sum(-1)>0): raise ValueError("Invalid inference action mask")
        if type(steps) is not int or steps<0: raise ValueError("Invalid inference step counter")
        result=self.mac.select_actions({k:v for k,v in data.items() if k not in {"state","critic_hidden"}},steps)
        result["hidden_states"]=result.get("hidden_states",data["hidden_states"]).cpu()
        if self.critic is not None:
            shape=(1,n,h) if "independent_obs" in self.args.critic_inputs else (1,h)
            if "critic_hidden" not in data or tuple(data["critic_hidden"].shape)!=shape or data["critic_hidden"].dtype!=torch.float32: raise ValueError("Critic hidden-state shape/dtype mismatch")
            critic_data=dict(obs=data["obs"].to(self.device),states=data["state"].to(self.device),hidden_states=data["critic_hidden"].to(self.device),last_actions=data["last_actions"].to(self.device))
            values,hidden=self.critic(critic_data)
            result["behavior_values"]=values.cpu()
            result["critic_hidden"]=(hidden if hidden is not None else data["critic_hidden"]).cpu()
        actions=result["actions"]
        if actions.dtype!=torch.long or tuple(actions.shape)!=(1,n) or not torch.all((actions>=0)&(actions<a)): raise ValueError("Inference produced invalid actions")
        if not torch.all(available.gather(-1,actions.unsqueeze(-1))): raise ValueError("Inference produced an unavailable action")
        self.calls+=1
        return dict(kind="ACTIONS",data=result,version=self.version,digest=self.digest)

def inference_worker(endpoint,settings_values=None,ready=None):
    settings=ZMQSettings.parse(dict(settings_values or {},bind_endpoint=endpoint))
    zmq=zmq_module()
    context=zmq.Context()
    socket=context.socket(zmq.REP)
    configure(socket,settings)
    engine=None
    job=None
    schema=None
    try:
        endpoint=bind(socket,endpoint)
        if ready: ready.send(dict(kind="READY",endpoint=endpoint))
        while True:
            if not socket.poll(int(settings.startup_timeout_s*1000),zmq.POLLIN): raise TimeoutError("Inference worker idle timeout")
            request=decode(socket.recv_multipart(),settings.max_message_bytes)
            request_id=request.get("request_id")
            stop=False
            try:
                kind=request.get("kind")
                if kind=="INIT":
                    if engine is not None: raise ValueError("Inference role already initialized")
                    job=request["job_id"]
                    schema=request["schema_id"]
                    if schema_id(request["config"],request["env_info"])!=schema: raise ValueError("Inference initialization schema mismatch")
                    engine=InferenceEngine(request["config"],request["env_info"],settings.inference_device,request["seed"])
                    reply=engine.synchronize(request["snapshot"])
                else:
                    if engine is None or request.get("job_id")!=job or request.get("schema_id")!=schema: raise ValueError("Inference job/schema mismatch")
                    if kind=="SYNC": reply=engine.synchronize(request["snapshot"])
                    elif kind=="INFER": reply=engine.infer(request["data"],request["steps"])
                    elif kind=="STOP":
                        reply=dict(kind="STOPPED",calls=engine.calls,syncs=list(engine.syncs),sync_count=engine.sync_count)
                        stop=True
                    else: raise ValueError("Unknown inference message")
            except Exception as exc:
                reply=dict(kind="ERROR",error=f"{type(exc).__name__}: {exc}",traceback=traceback.format_exc())
            socket.send_multipart(encode(dict(reply,request_id=request_id),settings.max_message_bytes))
            if stop: break
    except BaseException as exc:
        if ready:
            try: ready.send(dict(kind="ERROR",error=str(exc),traceback=traceback.format_exc()))
            except (BrokenPipeError,OSError): pass
        raise
    finally:
        socket.close(0)
        context.term()
        if ready: ready.close()
