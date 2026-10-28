"""Source communication inventory and observed local platform, read-only."""
import ast
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import sys

root=Path(__file__).resolve().parents[1]
evidence=root/"docs/evidence/stage6"
evidence.mkdir(parents=True,exist_ok=True)
previous=json.loads((root/"docs/evidence/stage5/audit.json").read_text(encoding="utf-8"))
items=[]
for family,source in previous["sources"].items():
    base=Path(source)
    for relative in ["worker/zmq_worker.py","worker/infer_worker.py","worker/train_worker.py","worker/sample_worker.py","worker/evaluate_worker.py","worker/base_worker.py","runner/async_runner/train_runner.py","runner/async_runner/sample_runner.py","utils/queue.py"]:
        path=base/relative
        text=path.read_text(encoding="utf-8")
        tree=ast.parse(text)
        items.append(dict(family=family,source=path.relative_to(root.parent).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            definitions=[node.name for node in tree.body if isinstance(node,(ast.ClassDef,ast.FunctionDef))],lines=len(text.splitlines())))
import zmq
payload=dict(sources=items,python=sys.version,platform=platform.platform(),pyzmq=importlib.metadata.version("pyzmq"),libzmq=zmq.zmq_version(),
    source_behavior=dict(roles=["train","sample","inference","communication","evaluate"],transport="REQ/REP structured native NumPy records + JSON info",
        version_in_action_message=False,old_log_probability_in_remote_reply=False,parameter_sync="inference copies shared MAC if agent_version increases; no snapshot acknowledgement",
        shared_slots="bounded shared_idx and episode_idx queues, but blocking Queue.get/put have no deadlines",socket_ownership="InferServer receives/sends the same REP sockets from different threads",
        stop="flags plus force kill; no systematic socket/context close, join or sentinel unblocking",client_poll_ms=500000),
    integration_decisions=dict(reuse="Stage5 algorithm classes, EpisodeBatch, source ReplayBuffer, source PPO/vtrace losses",
        replace="wire ABI, thread/socket ownership, shared queues and lifecycle with versioned bounded ZMQ execution",
        modes={"Qmix/VDN":["local","zmq sync","zmq async"],"PPO":["local fresh synchronous","zmq sync"],"PPO use_vtrace=true":["local","zmq sync","zmq async replay"]},
        message_compatibility="new protocol=1; not compatible with original InferClient binary records",platform="Windows spawn; no sh/pgrep/kill script or fork requirement",
        cross_machine="NOT_VALIDATED: no remote host supplied",distributed_evaluation="Not migrated as a continuous worker; saved policy evaluate() remains available"))
(evidence/"audit.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({"source_files":len(items),"pyzmq":payload["pyzmq"],"libzmq":payload["libzmq"]}))

if __name__=="__main__": pass
