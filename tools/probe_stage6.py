"""Managed spawn probe. Run as a saved main module on Windows."""
from dataclasses import asdict
import json
from pathlib import Path
import traceback
import uuid
from unified_marl import train,evaluate

def main():
    root=Path(__file__).resolve().parents[1]
    evidence=root/"docs/evidence/stage6"
    evidence.mkdir(parents=True,exist_ok=True)
    runs=root/".work"/("stage6-probes-"+uuid.uuid4().hex[:8])
    rows=[]
    for name,mode,extra in [("sadma/qmix","sync",{}),("sadma/ppo","sync",{}),("sadma/qmix","async",{}),("sadma/ppo","async",{"use_vtrace":True})]:
        try:
            result=train(name,"toy",execution="zmq",output_dir=runs/str(len(rows)),overrides={"train_batch_size":2,"train_buffer_size":8,
                "update_rate":1,"hidden_dim":16,"mini_epochs":2,"t_max":40,"max_updates":2,"zmq.mode":mode,**extra})
            score=evaluate(result.checkpoint,episodes=2)
            report=json.loads((Path(result.output_dir)/"execution.json").read_text(encoding="utf-8"))
            assert result.parameter_delta_l2>0 and result.updates==2 and report["lifecycle"]["forced"]==[]
            rows.append(dict(name=name,mode=mode,status="PASS",result=asdict(result),evaluation=asdict(score),execution=report))
        except Exception as exc:
            rows.append(dict(name=name,mode=mode,status="FAIL",error=str(exc),traceback=traceback.format_exc()))
        print(name,mode,rows[-1]["status"],rows[-1].get("error",""),flush=True)
    (evidence/"probes.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding="utf-8")
    assert all(row["status"]=="PASS" for row in rows)

if __name__=="__main__": main()
