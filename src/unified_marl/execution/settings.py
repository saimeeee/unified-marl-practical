"""Explicit, bounded execution settings shared by managed and separate roles."""
from dataclasses import asdict,dataclass
import math
import re
import uuid

@dataclass
class ZMQSettings:
    role:str="managed"
    mode:str="sync"
    bind_endpoint:str="tcp://127.0.0.1:0"
    trainer_endpoint:str|None=None
    inference_endpoints:list|None=None
    num_sampling_workers:int=2
    num_inference_workers:int=1
    inference_device:str="cpu"
    sampler_device:str="cpu"
    seed_stride:int=1000
    timeout_ms:int=10000
    startup_timeout_s:float=45.0
    shutdown_timeout_s:float=5.0
    hwm:int=2
    max_message_bytes:int=64*1024*1024
    max_policy_lag:int|None=None
    job_id:str|None=None
    allow_remote:bool=False
    # Deterministic failure checks; defaults never inject a failure.
    fault_worker:int|None=None
    fault_after_infers:int=1
    fault_kind:str="raise"

    @classmethod
    def parse(cls,values=None):
        settings=cls(**(values or {}))
        if settings.role not in {"managed","trainer"} or settings.mode not in {"sync","async"}: raise ValueError("Invalid ZMQ role/mode")
        for key in ["num_sampling_workers","num_inference_workers","seed_stride","timeout_ms","hwm","max_message_bytes","fault_after_infers"]:
            value=getattr(settings,key)
            if not isinstance(value,int) or isinstance(value,bool) or value<1: raise ValueError(f"zmq.{key} must be a positive integer")
        if settings.num_sampling_workers>64 or settings.num_inference_workers>16: raise ValueError("Worker count exceeds the configured bounded topology")
        if settings.timeout_ms>300000 or settings.max_message_bytes>256*1024*1024: raise ValueError("ZMQ timeout/message bound is too large")
        if any(not isinstance(x,(int,float)) or isinstance(x,bool) or not math.isfinite(x) or x<=0 or x>300 for x in [settings.startup_timeout_s,settings.shutdown_timeout_s]): raise ValueError("Startup/shutdown deadlines must be finite, positive and at most 300 seconds")
        if settings.sampler_device!="cpu": raise ValueError("Sampler transport and environment tensors use CPU; inference/train devices are configured separately")
        if settings.max_policy_lag is not None and (type(settings.max_policy_lag) is not int or settings.max_policy_lag<0): raise ValueError("max_policy_lag must be nonnegative or null")
        if settings.fault_kind not in {"raise","exit","stall"}: raise ValueError("Invalid diagnostic fault kind")
        settings.job_id=settings.job_id or uuid.uuid4().hex
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,96}",settings.job_id): raise ValueError("Invalid job_id")
        settings.inference_endpoints=list(settings.inference_endpoints or [])
        if settings.inference_endpoints and len(settings.inference_endpoints)!=settings.num_inference_workers: raise ValueError("inference_endpoints count differs from num_inference_workers")
        if settings.role=="trainer" and not settings.inference_endpoints: raise ValueError("Standalone trainer requires explicit inference_endpoints")
        for endpoint in [settings.bind_endpoint,settings.trainer_endpoint,*settings.inference_endpoints]:
            if endpoint is None: continue
            match=re.fullmatch(r"tcp://([A-Za-z0-9_.-]+):(\d{1,5})",endpoint)
            if not match or not 0<=int(match[2])<=65535: raise ValueError("Only explicit TCP IPv4/hostname endpoints are supported")
            if not settings.allow_remote and match[1] not in {"127.0.0.1","localhost"}: raise ValueError("Non-loopback endpoints require zmq.allow_remote=true")
        return settings

    def dictionary(self): return asdict(self)
