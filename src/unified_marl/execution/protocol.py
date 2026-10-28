"""Versioned JSON + primitive ndarray frames. Never pickle network input."""
import hashlib
import json
import time
import numpy as np
import torch

PROTOCOL=1
DTYPES={"float32","float64","int32","int64","uint8","bool"}

def encode(message,limit=64*1024*1024):
    frames=[]
    def visit(value):
        if isinstance(value,torch.Tensor): value=value.detach().cpu().numpy()
        if isinstance(value,np.ndarray):
            if not value.flags.c_contiguous: value=np.ascontiguousarray(value)
            if value.dtype.name not in DTYPES or (value.dtype.kind=="f" and not np.isfinite(value).all()): raise ValueError("Unsupported/nonfinite wire tensor")
            index=len(frames)
            frames.append(value.tobytes())
            return {"__array__":index,"dtype":value.dtype.name,"shape":list(value.shape)}
        if isinstance(value,dict):
            if not all(isinstance(k,str) and k!="__array__" for k in value): raise ValueError("Wire mappings require ordinary string keys")
            return {key:visit(item) for key,item in value.items()}
        if isinstance(value,(list,tuple)): return [visit(item) for item in value]
        if value is None or isinstance(value,(str,bool,int,float)): return value
        raise ValueError(f"Unsupported wire object {type(value).__name__}")
    header=json.dumps({"protocol":PROTOCOL,"message":visit(message)},allow_nan=False,separators=(",",":")).encode()
    if len(header)>256*1024 or len(frames)>2048 or len(header)+sum(map(len,frames))>limit: raise ValueError("Message exceeds bounded wire size")
    return [header,*frames]

def decode(frames,limit=64*1024*1024):
    if not frames or len(frames)>2049 or len(frames[0])>256*1024 or sum(map(len,frames))>limit: raise ValueError("Invalid/bounded message size")
    try: header=json.loads(frames[0])
    except (ValueError,UnicodeError) as exc: raise ValueError("Malformed JSON header") from exc
    if not isinstance(header,dict) or header.get("protocol")!=PROTOCOL: raise ValueError("Incompatible protocol version")
    used=set()
    def visit(value,depth=0):
        if depth>32: raise ValueError("Message nesting limit exceeded")
        if isinstance(value,dict) and "__array__" in value:
            if set(value)!={"__array__","dtype","shape"}: raise ValueError("Invalid tensor descriptor")
            index=value["__array__"]
            shape=value["shape"]
            if type(index) is not int or index in used or index<0 or index+1>=len(frames): raise ValueError("Invalid tensor frame index")
            if value["dtype"] not in DTYPES or not isinstance(shape,list) or len(shape)>8 or any(type(n) is not int or n<0 for n in shape): raise ValueError("Invalid dtype/shape")
            count=1
            for n in shape:
                count*=n
                if count>limit: raise ValueError("Tensor shape exceeds bound")
            dtype=np.dtype(value["dtype"])
            if count*dtype.itemsize!=len(frames[index+1]): raise ValueError("Tensor frame length differs from shape")
            result=np.frombuffer(frames[index+1],dtype=dtype).reshape(shape).copy()
            if dtype.kind=="f" and not np.isfinite(result).all(): raise ValueError("Nonfinite received tensor")
            used.add(index)
            return torch.from_numpy(result)
        if isinstance(value,dict): return {key:visit(item,depth+1) for key,item in value.items()}
        if isinstance(value,list): return [visit(item,depth+1) for item in value]
        if value is not None and not isinstance(value,(str,bool,int,float)): raise ValueError("Invalid metadata value")
        if isinstance(value,float) and not np.isfinite(value): raise ValueError("Nonfinite metadata")
        return value
    result=visit(header.get("message"))
    if len(used)!=len(frames)-1 or not isinstance(result,dict): raise ValueError("Unreferenced frames or nonmapping message")
    return result

def digest(value):
    result=hashlib.sha256()
    for frame in encode(value): result.update(frame)
    return result.hexdigest()

def schema_id(config,info):
    keys=["algorithm","sadma_algorithm","agent_type","critic_type","agent_inputs","critic_inputs","hidden_dim","mixer","use_layer_norm","use_orthogonal","gain","use_individual_rewards"]
    return digest({"config":{key:config.get(key) for key in keys},"env_info":info})

def zmq_module():
    try: import zmq
    except ImportError as exc: raise ImportError("ZMQ execution requires unified-marl[distributed] (pyzmq)") from exc
    return zmq

def configure(socket,settings):
    zmq=zmq_module()
    socket.setsockopt(zmq.LINGER,0)
    socket.setsockopt(zmq.SNDHWM,settings.hwm)
    socket.setsockopt(zmq.RCVHWM,settings.hwm)
    socket.setsockopt(zmq.SNDTIMEO,settings.timeout_ms)
    socket.setsockopt(zmq.MAXMSGSIZE,settings.max_message_bytes)

def bind(socket,endpoint):
    if endpoint.endswith(":0"): socket.bind_to_random_port(endpoint.rsplit(":",1)[0])
    else: socket.bind(endpoint)
    return socket.getsockopt_string(zmq_module().LAST_ENDPOINT)

class RPCClient:
    def __init__(self,endpoint,settings,*,dealer=False,identity=None):
        self.zmq=zmq_module()
        self.settings=settings
        self.context=self.zmq.Context()
        self.socket=self.context.socket(self.zmq.DEALER if dealer else self.zmq.REQ)
        configure(self.socket,settings)
        if identity: self.socket.setsockopt(self.zmq.IDENTITY,identity.encode())
        self.socket.connect(endpoint)
        self.endpoint=endpoint
        self.sequence=0
    def call(self,message,timeout_ms=None):
        self.sequence+=1
        request=dict(message,request_id=self.sequence)
        try:
            self.socket.send_multipart(encode(request,self.settings.max_message_bytes))
            timeout=self.settings.timeout_ms if timeout_ms is None else timeout_ms
            if not self.socket.poll(timeout,self.zmq.POLLIN): raise TimeoutError(f"Communication timeout ({timeout} ms): {self.endpoint}")
            reply=decode(self.socket.recv_multipart(),self.settings.max_message_bytes)
        except self.zmq.Again as exc: raise TimeoutError(f"ZMQ backpressure timeout: {self.endpoint}") from exc
        if reply.get("request_id")!=self.sequence: raise ValueError("Reply request_id mismatch")
        if reply.get("kind")=="ERROR": raise RuntimeError(f"Remote worker error: {reply.get('error')}")
        return reply
    def close(self):
        self.socket.close(0)
        self.context.term()
