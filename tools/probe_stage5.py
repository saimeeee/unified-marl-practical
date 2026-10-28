"""Short local synchronous probes with structured outcomes, never simulator claims."""
from dataclasses import asdict
import json
from pathlib import Path
import traceback
import uuid
from unified_marl import train,evaluate,load_policy

def main():
    root=Path(__file__).resolve().parents[1]
    runs=root/".work"/("stage5-probes-"+uuid.uuid4().hex[:8])
    rows=[]
    for name,extra in [(n,{}) for n in ["sadma/qmix","sadma/ppo","sadma/vdn","sadma_yi/qmix","sadma_yi/ppo","sadma_yi_closed/qmix","sadma_yi_closed/ppo"]]+[
        ("sadma/ppo",{"agent_type":"mlp","critic_type":"mlp","use_value_norm":False}),
        ("sadma/ppo",{"critic_inputs":["independent_obs","agent_id","last_action"]}),
        ("sadma/ppo",{"critic_inputs":["central_obs","state"]}),
        ("sadma/ppo",{"use_vtrace":True,"dual_clip":3}),
        ("sadma/qmix",{"q_lambda":True}),
        ("sadma/qmix",{"agent_type":"mlp"})]:
        try:
            options=dict(train_batch_size=2,train_buffer_size=8,update_rate=1,t_max=32,max_updates=2,hidden_dim=16,mini_epochs=2)
            options.update(extra)
            run=train(name,"toy",output_dir=runs/str(len(rows)),overrides=options)
            assert run.parameter_delta_l2>0 and run.updates==2
            score=evaluate(run.checkpoint,episodes=2)
            load_policy(run.checkpoint).reset()
            rows.append(dict(name=name,extra=extra,status="PASS",train=asdict(run),evaluation=asdict(score)))
        except Exception as exc:
            rows.append(dict(name=name,extra=extra,status="FAIL",error=str(exc),traceback=traceback.format_exc()))
        print(name,extra,rows[-1]["status"],flush=True)
    evidence=root/"docs/evidence/stage5"
    (evidence/"probes.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding="utf-8")
    assert all(row["status"]=="PASS" for row in rows)

if __name__=="__main__": main()
