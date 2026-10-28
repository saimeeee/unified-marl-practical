"""Repeat actual remaining fixed-entry failures; do not expose failed bundles."""
import ast
import json
from pathlib import Path
import traceback
import uuid
from unified_marl import train
from unified_marl.algorithms.hmf.adapter import IMPLEMENTATIONS
from unified_marl.registry import HMF_ALGORITHMS


def main():
    root=Path(__file__).resolve().parents[1]
    results=[]
    candidates={"hierarchical_qmix":("hierarchical_qmix","Hierarchical_Qmix"),
                "hierarchical_vdn_dynamic_ipc":("hierarchical_vdn_IPC","Hierarchical_VDN"),
                "hierarchical_vdn_roma":("hierarchical_vdn_roma","Hierarchical_VDN_ROMA"),
                "mfac":("mfac","MFAC")}
    for name,implementation in candidates.items():
        IMPLEMENTATIONS[name]=implementation
        HMF_ALGORITHMS.add(name)
        try:
            result=train("wang_hmf/"+name,"toy",env_args={"n_agents":4,"mask_actions":False},
                         output_dir=root/".work/stage7-remaining"/(name+'-'+uuid.uuid4().hex[:8]),
                         overrides={"batch_size":4,"buffer_size":32,"t_max":16,"max_updates":1,"hidden_dim":16,
                                    "n_heads":2,"embed_dim":8,"hmf_norm_rewards":False,"hmf_steps_per_update":1,"hmf_num_batch":1})
            results.append(dict(name=name,status="UNEXPECTED_PROGRESS_REQUIRES_REVIEW",updates=result.updates))
        except Exception as exc:
            results.append(dict(name=name,status="BLOCKER_RECONFIRMED",error=str(exc),traceback=traceback.format_exc()))
        finally:
            IMPLEMENTATIONS.pop(name);HMF_ALGORITHMS.discard(name)
    mfac=(root/"src/unified_marl/algorithms/hmf/algorithms/mfac.py").read_text(encoding="utf-8")
    results[-1]['further_source_findings']=[line.strip() for line in mfac.splitlines() if any(key in line for key in ['detacj','self.agent.critic','self.agent.eval_optimizer','next_state, rews, log_probs'])]
    evidence=root/"docs/evidence/stage7/remaining-probes.json"
    evidence.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps([{k:v for k,v in row.items() if k!='traceback'} for row in results],ensure_ascii=False,indent=2))
    assert all(row['status']=='BLOCKER_RECONFIRMED' for row in results)


if __name__=='__main__': main()
