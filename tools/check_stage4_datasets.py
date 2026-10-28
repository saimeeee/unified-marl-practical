"""Read small real batches and check label roles, without rewriting any data."""
import json
from pathlib import Path
import traceback
from unified_marl.datasets import load_dataset

root=Path(__file__).resolve().parents[1]
audit=json.loads((root/"docs/evidence/stage4/audit.json").read_text(encoding="utf-8"))
records=[]
for row in audit["data"]:
    if "obs" not in row.get("fields",{}): continue
    path=root.parent/row["path"]
    try:
        data=load_dataset(path,max_length=12)
        batch=data.batch([0,1])
        pair=data.preference([0,1])
        records.append({"path":row["path"],"status":"SMALL_BATCH_READ_PASS","env_info":data.env_info,
            "batch_shape":list(batch["obs"].shape),"pair_indices":[pair.indices0.tolist(),pair.indices1.tolist()],
            "label_key":pair.label_key,"labels":pair.labels.tolist(),"lengths0":pair.lengths0.tolist(),"lengths1":pair.lengths1.tolist()})
    except Exception as exc:
        records.append({"path":row["path"],"status":"READ_BLOCKED","error":str(exc),"traceback":traceback.format_exc()})
(root/"docs/evidence/stage4/dataset_batches.json").write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({"episode_files":len(records),"passed":sum(r["status"]=="SMALL_BATCH_READ_PASS" for r in records),"failures":[r["error"] for r in records if r["status"]!="SMALL_BATCH_READ_PASS"]},ensure_ascii=False))
