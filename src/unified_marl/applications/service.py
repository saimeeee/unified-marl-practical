"""Application configuration, dispatch, metadata and business action conversion."""
from dataclasses import asdict, dataclass, is_dataclass
from importlib.resources import files
from pathlib import Path
from copy import deepcopy
import csv
import json
import uuid
import yaml
from unified_marl.config import merge, dotted_overrides

APPS = {"warehouse", "traffic", "powergrid"}
PATH_KEYS = {"agents_file", "demand_file", "cityflow_config_file", "data_path",
    "buildings_states_actions_file", "net_path", "agent_path", "vis_path"}


def _jsonable(value):
    if is_dataclass(value): return _jsonable(asdict(value))
    if isinstance(value, dict): return {str(k): _jsonable(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"): return value.tolist()
    if isinstance(value, Path): return str(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def load_config(config, *, overrides=None):
    if isinstance(config, (str, Path)):
        path = Path(config).resolve()
        value = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
        base = path.parent
    else:
        value, base = deepcopy(config), Path.cwd()
    if not isinstance(value, dict): raise ValueError("Application config must be a YAML mapping")
    value = merge(value, dotted_overrides(overrides))
    if value.get("application") not in APPS: raise ValueError("application must be warehouse, traffic or powergrid")
    unknown = set(value)-{"application","algorithm","backend","env_args","training","seed","device","execution","output","schema_version"}
    if unknown: raise ValueError("Unknown application config keys: " + ", ".join(sorted(unknown)))
    for key in ("env_args", "training", "output"):
        if not isinstance(value.get(key, {}),dict): raise ValueError(f"{key} must be a mapping")
        value.setdefault(key,{})
    for key,item in value["env_args"].items():
        if key in PATH_KEYS and item is not None:
            if not isinstance(item,str) or not item: raise ValueError(f"env_args.{key} must be a path string")
            path = Path(item).expanduser()
            value["env_args"][key] = str((base/path).resolve() if not path.is_absolute() else path.resolve())
    value.setdefault("seed",1)
    if type(value["seed"]) is not int or not 0 <= value["seed"] < 2**32: raise ValueError("seed must be an integer in [0, 2**32)")
    value.setdefault("device","cpu")
    value.setdefault("execution","local")
    value.setdefault("schema_version",1)
    if value["schema_version"]!=1: raise ValueError("Unsupported application schema_version")
    value.setdefault("algorithm",{"warehouse":"mfvfd", "traffic":"ppo", "powergrid":"bounded_ppo"}[value["application"]])
    reserved = {"env","env_args","algorithm","seed","device","output_dir","execution","dataset","resume","config"} & set(value["training"])
    if reserved: raise ValueError("Put application settings outside training: " + ", ".join(sorted(reserved)))
    return value


def _bundle(cfg):
    app, algorithm = cfg["application"],cfg["algorithm"]
    if app=="powergrid":
        if algorithm!="bounded_ppo" or cfg["execution"]!="local": raise ValueError("PowerGrid currently exposes local bounded_ppo only")
        return algorithm,"powergrid"
    backend = cfg.get("backend", "inventory" if app=="warehouse" else "cityflow_engine")
    env = {"inventory":"inventory","replenishment":"replenishment","cityflow_engine":"traffic_engine","cityflowenv":"cityflow"}.get(backend)
    allowed = {"warehouse":{"inventory","replenishment"},"traffic":{"cityflow_engine","cityflowenv"}}
    if backend not in allowed[app]: raise ValueError(f"Unsupported {app} backend {backend!r}")
    if app=="warehouse":
        from unified_marl.registry import HMF_ALGORITHMS
        bundles = {name:"wang_hmf/"+name for name in HMF_ALGORITHMS}
        bundles.update(rode_hmf="wang/rode_hmf",rode="wang/rode",ppo="sadma/ppo",pymarl_qmix="pymarl/qmix")
    else:
        bundles={"ppo":"sadma/ppo","qmix":"sadma/qmix","vdn":"sadma/vdn"}
    if algorithm in bundles: return bundles[algorithm],env
    if algorithm in bundles.values(): return algorithm,env
    raise ValueError(f"Unsupported {app} algorithm {algorithm!r}; choose from {sorted(bundles)}")


def validate(config, *, overrides=None):
    """Check parameter/data contracts only. Never create a simulator or run a policy."""
    cfg=load_config(config,overrides=overrides)
    bundle,env=_bundle(cfg)
    args=cfg["env_args"]
    metadata={}
    required = {"inventory":["agents_file","demand_file"],"traffic_engine":["cityflow_config_file","phase_ids"],
        "powergrid":["data_path","buildings_states_actions_file"]}.get(env,[])
    missing=[k for k in required if k not in args or args[k] is None]
    if missing: raise ValueError("Required scenario parameters: " + ", ".join(missing))
    for key in required:
        if key in PATH_KEYS and not Path(args[key]).exists(): raise FileNotFoundError(f"env_args.{key}: {args[key]}")
    if env=="inventory":
        from unified_marl.envs.inventory import read_problem
        rows,demands,multipliers=read_problem(args["agents_file"],args["demand_file"],args.get("horizon",30),args.get("action_multipliers"))
        metadata={"agent_ids":[r["agent_id"] for r in rows],"agents":rows,"action_multipliers":multipliers.tolist(),
            "horizon":len(demands),"max_lead_time":max(r["lead_time"] for r in rows),"obs_schema":"stock/capacity,backlog/capacity,mean_demand/capacity,last_demand/capacity,period/horizon,pipeline_by_due/capacity"}
        if bundle.startswith("wang_hmf/"):
            groups=cfg["training"].get("num_groups",2)
            if type(groups) is not int or groups < 1 or len(rows)%groups: raise ValueError("Inventory agent count must be divisible by training.num_groups for fixed HMF")
    elif env=="traffic_engine":
        from unified_marl.envs.traffic_engine import read_traffic_spec
        metadata=read_traffic_spec(**dict(args,seed=cfg["seed"]))
        import hashlib
        metadata["roadnet_sha256"]=hashlib.sha256(Path(metadata["roadnet_file"]).read_bytes()).hexdigest()
    elif env=="powergrid":
        for name in ["weather_data.csv","solar_generation_1kW.csv"]:
            if not (Path(args["data_path"])/name).is_file(): raise FileNotFoundError(str(Path(args["data_path"])/name))
        json.loads(Path(args["buildings_states_actions_file"]).read_text(encoding="utf-8-sig"))
        import hashlib
        assets={}
        for path in sorted(Path(args["data_path"]).rglob("*")):
            if path.is_file() and path.suffix.lower() in {".json",".csv"}:
                assets[str(path.relative_to(args["data_path"]))]=hashlib.sha256(path.read_bytes()).hexdigest()
        metadata={"env_args":deepcopy(args),"asset_sha256":assets,
            "building_spec_sha256":hashlib.sha256(Path(args["buildings_states_actions_file"]).read_bytes()).hexdigest()}
    if cfg["execution"] not in {"local","zmq"}: raise ValueError("execution must be local or zmq")
    if cfg["execution"]=="zmq" and not bundle.startswith("sadma/"): raise ValueError("ZMQ currently supports SADMA bundles only")
    return {"application":cfg["application"],"algorithm":bundle,"environment":env,"config":cfg,
        "metadata":_jsonable(metadata),"status":"configuration_checked","experiments_run":False}


@dataclass
class ApplicationResult:
    application: str
    algorithm: str
    output_dir: str
    checkpoint: str
    training_checkpoint: str
    metrics: dict
    updates: int
    env_steps: int
    raw: dict


def train(application, config, output_dir=None, *, overrides=None, resume=None):
    report=validate(config,overrides=overrides)
    cfg=report["config"]
    if cfg["application"]!=application: raise ValueError("Config application differs from selected API")
    directory=Path(output_dir or Path.cwd()/"results"/(application+"-"+uuid.uuid4().hex[:10])).resolve()
    if directory.exists() and any(directory.iterdir()): raise FileExistsError(f"Output must be empty: {directory}")
    directory.mkdir(parents=True,exist_ok=True)
    if resume:
        previous=_manifest(resume)
        for key in ["application","algorithm","backend","env_args"]:
            if previous["config"].get(key)!=cfg.get(key): raise ValueError(f"Cannot change {key} during resume")
    if application=="powergrid":
        from . import powergrid_engine
        raw=powergrid_engine.train(dict(cfg,**cfg["training"]),directory,resume=resume)
    else:
        from unified_marl import train as core_train
        raw=core_train(report["algorithm"],report["environment"],env_args=cfg["env_args"],
            execution=cfg["execution"],device=cfg["device"],seed=cfg["seed"],output_dir=directory,
            config=cfg["training"] if not resume else None,overrides=cfg["training"] if resume else None,resume=resume)
        raw=asdict(raw)
    write_json(directory/"application.json",dict(schema_version=1,config=cfg,metadata=report["metadata"],algorithm=report["algorithm"],environment=report["environment"]))
    result=ApplicationResult(application,report["algorithm"],str(directory),str(raw["checkpoint"]),
        str(raw.get("training_checkpoint",raw.get("training_state",""))),raw.get("metrics",{}),int(raw.get("updates",0)),int(raw.get("env_steps",0)),_jsonable(raw))
    write_json(directory/"application-result.json",result)
    return result


def _manifest(checkpoint):
    file=Path(checkpoint).resolve().parent/"application.json"
    if not file.is_file(): raise FileNotFoundError("Keep application.json beside the application checkpoint: "+str(file))
    return json.loads(file.read_text(encoding="utf-8"))


def _semantic_metadata(application, metadata):
    result=deepcopy(metadata)
    if application=="traffic":
        for key in ["cityflow_config_file","cityflow_root","roadnet_file","flow_file","seed","thread_num","episode_limit"]:
            result.pop(key,None)
    elif application=="powergrid":
        for key in PATH_KEYS:
            result.get("env_args",{}).pop(key,None)
        result.get("env_args",{}).pop("randomseed",None)
        result.get("env_args",{}).pop("episode_limit",None)
    return result


def _inference_scene(application,checkpoint,config=None):
    saved=_manifest(checkpoint)
    if saved["config"]["application"]!=application: raise ValueError("Checkpoint belongs to another application")
    if config is not None:
        report=validate(config)
        if report["application"]!=application or report["algorithm"]!=saved["algorithm"]: raise ValueError("Application/algorithm differs from checkpoint")
        if report["environment"]!=saved["environment"]: raise ValueError("Backend differs from checkpoint")
        # IDs, physical parameters and action semantics are part of model compatibility.
        if _semantic_metadata(application,report["metadata"])!=_semantic_metadata(application,saved["metadata"]):
            raise ValueError("Agent order, dimensions, network or action semantics differ from checkpoint; retrain for the new scenario")
        saved=dict(saved,config=report["config"])
    return saved


def evaluate(application,checkpoint,*,config=None,episodes=1,seed=None,device="cpu"):
    saved=_inference_scene(application,checkpoint,config)
    if application=="powergrid":
        from . import powergrid_engine
        cfg=saved["config"]
        return powergrid_engine.evaluate(checkpoint,config=dict(cfg,**cfg["training"]),episodes=episodes,seed=seed,device=device)
    from unified_marl import evaluate as core_evaluate
    result=core_evaluate(checkpoint,env=saved["environment"],env_args=saved["config"]["env_args"],episodes=episodes,seed=seed,device=device)
    return _jsonable(result)


class ApplicationPolicy:
    def __init__(self,application,raw,saved):
        self.application,self.raw,self.saved=application,raw,saved
        self.config,self.metadata=saved["config"],saved["metadata"]
        self.agent_ids=self.metadata.get("agent_ids",getattr(raw,"agent_ids",[]))
    def reset(self): self.raw.reset()
    def encode(self,state):
        """Warehouse named state to the exact inventory observation layout."""
        import numpy as np
        if self.application=="traffic" and self.metadata.get("phase_ids"):
            records={r["intersection_id"]:r for r in state["intersections"]}
            if set(records)!=set(self.agent_ids) or len(records)!=len(state["intersections"]): raise ValueError("Expected each configured intersection exactly once")
            size=self.metadata["obs_size"];n_actions=self.metadata["n_actions"]
            obs=np.zeros((len(self.agent_ids),size),dtype=np.float32)
            for n,i in enumerate(self.agent_ids):
                item=records[i];phases=self.metadata["phase_ids"][i]
                if item["phase_id"] not in phases: raise ValueError("Named traffic input must be sampled at a green decision boundary")
                obs[n,phases.index(item["phase_id"])]=1.
                lanes=self.metadata["inbound_lane_ids"][i]
                for field in ["waiting_vehicles","vehicles"]:
                    if set(item[field])!=set(lanes): raise ValueError(f"{field} must contain every incoming lane exactly once")
                for j,lane in enumerate(lanes):
                    waiting=float(item["waiting_vehicles"][lane]);vehicles=float(item["vehicles"][lane])
                    if min(waiting,vehicles)<0 or waiting>vehicles: raise ValueError("Invalid vehicle/queue count")
                    obs[n,n_actions+2*j:n_actions+2*j+2]=[waiting,vehicles]
            if not np.isfinite(obs).all(): raise ValueError("Non-finite traffic counts")
            return obs
        if self.application!="warehouse" or "agents" not in self.metadata: raise ValueError("Named-state encoding requires inventory or cityflow_engine backend")
        records={r["agent_id"]:r for r in state["agents"]}
        if set(records)!=set(self.agent_ids): raise ValueError("state.agents must contain each configured agent exactly once")
        if len(records)!=len(state["agents"]): raise ValueError("Duplicate state agent IDs")
        period=state["period"]
        horizon=self.metadata["horizon"]
        if type(period) is not int or not 0<=period<horizon: raise ValueError("period must be within the current episode")
        values=[]
        for spec in self.metadata["agents"]:
            row=records[spec["agent_id"]];cap=spec["capacity"]
            stock=float(row["stock"]);backlog=float(row.get("backlog",0));last=float(row.get("last_demand",0))
            if min(stock,backlog,last)<0 or stock>cap: raise ValueError("Invalid stock/backlog/last_demand")
            due=[0.]*(self.metadata["max_lead_time"]+1)
            for item in row.get("pipeline",[]):
                t=item["arrives_in"];q=float(item["quantity"])
                if type(t) is not int or not 0<=t<len(due) or q<0: raise ValueError("Invalid pipeline due period/quantity")
                due[t]+=q/cap
            values.append([stock/cap,backlog/cap,spec["mean_demand"]/cap,last/cap,period/horizon]+due)
        array=np.asarray(values,dtype=np.float32)
        if not np.isfinite(array).all(): raise ValueError("Non-finite business state")
        return array
    def act(self,observation,avail_actions=None,*,deterministic=True):
        import numpy as np
        obs=self.encode(observation) if isinstance(observation,dict) and self.application in {"warehouse","traffic"} else observation
        if self.application=="powergrid":
            actions=self.raw.act(obs,deterministic=deterministic)
            spec=self.raw.env_info.get("action_spec",{})
            by_agent=spec.get("names_by_agent",{})
            decisions=[]
            for index,(i,a) in enumerate(zip(self.raw.agent_ids,actions)):
                names=by_agent[index] if isinstance(by_agent,list) and index<len(by_agent) else by_agent.get(str(i),spec.get("names",[])) if isinstance(by_agent,dict) else spec.get("names",[])
                row={"agent_id":i,"controls":_jsonable(a),"control_names":_jsonable(names)}
                if names and len(names)==len(a): row["named_controls"]={str(k):float(v) for k,v in zip(names,a)}
                decisions.append(row)
            return {"application":"powergrid","decisions":decisions,"raw_actions":_jsonable(actions)}
        if not deterministic: raise ValueError("Discrete application inference currently uses deterministic evaluation mode")
        if self.application=="traffic" and self.metadata.get("phase_ids"):
            count=self.raw.env_info["n_actions"]
            physical=np.asarray([[1]*len(self.metadata["phase_ids"][i])+[0]*(count-len(self.metadata["phase_ids"][i])) for i in self.agent_ids],dtype=np.int32)
            if avail_actions is not None:
                supplied=np.asarray(avail_actions)
                if supplied.shape!=physical.shape or not np.isin(supplied,[0,1]).all(): raise ValueError("Invalid supplied traffic mask")
                physical=physical*supplied
            if np.any(physical.sum(-1)==0): raise ValueError("No available traffic phase")
            avail_actions=physical
        actions=np.asarray(self.raw.act(obs,avail_actions)).reshape(-1)
        if len(actions)!=self.raw.env_info["n_agents"]: raise ValueError("Unexpected policy action count")
        decisions=[]
        if self.application=="warehouse" and "agents" in self.metadata:
            observations=np.asarray(obs)
            for spec,a,o in zip(self.metadata["agents"],actions,observations):
                requested=self.metadata["action_multipliers"][int(a)]*spec["mean_demand"]
                reserved=(float(o[0])+float(o[5:].sum()))*spec["capacity"]
                decisions.append(dict(agent_id=spec["agent_id"],warehouse_id=spec["warehouse_id"],sku_id=spec["sku_id"],
                    action_index=int(a),requested_order=requested,order_quantity=max(0.,min(requested,spec["capacity"]-reserved))))
        elif self.application=="traffic" and self.metadata.get("phase_ids"):
            for agent,a in zip(self.agent_ids,actions):
                phases=self.metadata["phase_ids"][agent]
                if not 0<=int(a)<len(phases): raise ValueError("Policy selected an unavailable traffic phase")
                decisions.append(dict(intersection_id=agent,action_index=int(a),phase_id=phases[int(a)]))
        else:
            ids=self.config["output"].get("agent_ids",list(range(len(actions))))
            if len(ids)!=len(actions): raise ValueError("output.agent_ids differs from model")
            decisions=[dict(agent_id=i,action_index=int(a)) for i,a in zip(ids,actions)]
        return {"application":self.application,"decisions":decisions,"raw_actions":actions.tolist()}


def load_policy(application,checkpoint,*,config=None,device="cpu"):
    saved=_inference_scene(application,checkpoint,config)
    if application=="powergrid":
        from .powergrid_engine import load_policy as loader
    else:
        from unified_marl import load_policy as loader
    return ApplicationPolicy(application,loader(checkpoint,device=device),saved)


def export_template(application,destination):
    if application not in APPS: raise ValueError("Unknown application")
    target=Path(destination).resolve()
    items=[p for p in files("unified_marl.applications").joinpath("templates").iterdir() if p.name.startswith(application+".") or p.name.startswith(application+"_")]
    if any((target/p.name).exists() for p in items): raise FileExistsError("Template files already exist at destination")
    target.mkdir(parents=True,exist_ok=True)
    for item in items: (target/item.name).write_bytes(item.read_bytes())
    return [str(target/p.name) for p in items]
