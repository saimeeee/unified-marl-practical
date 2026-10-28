from dataclasses import asdict
from pathlib import Path
import json
import traceback
import uuid
from unified_marl import train, load_policy


def main():
    root = Path(__file__).resolve().parents[1]
    base = root.parent / "marl平台代码/寇谦/论文代码及技术文档/论文代码及技术文档"
    data = base / "MADiversePreferDatasets/3m/3m.th"
    rewards = base / "MADiversePreferDatasets/3m/reward.th"
    run_root = root / ".work/stage4-probes" / uuid.uuid4().hex[:8]
    names = ["kou/bc","kou/cpl","kou/icq_ma","kou/iql","kou/cql","kou/dppo","kou/madpo_aux","kou/madpo","kou/bcmapt","kou/acpl","kou/bcq_qmix","kou/bcq_vdn"]
    names += ["kou_old/"+n for n in ["qmix","qmix_differ","qplex","cw_qmix","ow_qmix","qmix_w_globalRM","qmix_new","pref_qmix","pref_qmix_new","qplex_w_globalRM","cw_qmix_w_globalRM","ow_qmix_w_globalRM"]]
    result = []
    for name in names:
        opts = {"training_updates":1,"off_batch_size":4,"batch_size":4,"rnn_hidden_dim":16,"dataset_options.max_length":12}
        if name in {"kou/madpo","kou/bcmapt","kou/madpo_aux"}: opts["dataset_options.agent_reward_path"] = str(rewards)
        try:
            run = train(name,dataset=str(data),output_dir=run_root / name.replace("/","-"),overrides=opts)
            assert run.updates > 0 and run.parameter_delta_l2 > 0
            policy = load_policy(run.checkpoint)
            policy.reset()
            result.append({"algorithm":name,"status":"PASS","result":asdict(run),"data_kind":"real delivered 3m"})
        except Exception as exc:
            result.append({"algorithm":name,"status":"BLOCKED","error":f"{type(exc).__name__}: {exc}","traceback":traceback.format_exc()})
        print(name,result[-1]["status"],result[-1].get("error",""),flush=True)
    (root / "docs/evidence/stage4/core_probes.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__ == "__main__": main()
