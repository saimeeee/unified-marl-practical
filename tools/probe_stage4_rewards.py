from dataclasses import asdict
import json
from pathlib import Path
import traceback
import uuid
import torch
from unified_marl import train_reward_model,load_reward_model,train
from unified_marl.datasets import load_dataset


def main():
    root=Path(__file__).resolve().parents[1]
    path=root.parent/"marl平台代码/寇谦/论文代码及技术文档/论文代码及技术文档/MADiversePreferDatasets/3m/3m.th"
    runs=root/".work/stage4-reward-probes"/uuid.uuid4().hex[:8]
    records=[]
    for name in ["MR","NMR","PrefTransformer","MultiPrefTransformerDivide","old_global","old_reward_global","old_new_global"]:
        options={"inner_dim":32,"action_embd_dim":8,"embd_dim":32,"pref_attn_embd_dim":32,"n_head":4,"n_layer":1}
        if name.startswith("old_"): options={"reward_hidden_size":32,"ensemble_size":2}
        try:
            result=train_reward_model(name,str(path),updates=1,batch_size=4,output_dir=runs/name,
                model_config=options,dataset_options={"max_length":61 if name=="NMR" else 16})
            model=load_reward_model(result.checkpoint)
            batch=load_dataset(str(path),max_length=16).batch([0,1],trim=False)
            output=model.predict(batch)
            assert torch.isfinite(output).all()
            records.append({"model":name,"status":"PASS" if result.parameter_delta_l2>0 else "NO_PARAMETER_CHANGE",
                "result":asdict(result),"prediction_shape":list(output.shape),"data_kind":"real delivered 3m"})
            if name=="old_global":
                for algorithm in ["kou_old/qmix_w_globalRM","kou_old/qplex_w_globalRM"]:
                    try:
                        policy=train(algorithm,dataset=str(path),output_dir=runs/algorithm.replace("/","-"),
                            overrides={"training_updates":1,"batch_size":4,"rnn_hidden_dim":16,
                                "dataset_options.max_length":16,"reward_model_checkpoint":result.checkpoint})
                        records.append({"algorithm":algorithm,"status":"PASS","result":asdict(policy)})
                    except Exception as exc:
                        records.append({"algorithm":algorithm,"status":"BLOCKED","error":str(exc),"traceback":traceback.format_exc()})
        except Exception as exc:
            records.append({"model":name,"status":"BLOCKED","error":str(exc),"traceback":traceback.format_exc()})
        print(name,records[-1]["status"],records[-1].get("error",""),flush=True)
    try:
        result=train_reward_model("MultiPrefTransformerDivide",str(path),updates=1,batch_size=4,output_dir=runs/"diverse",
            model_config={"embd_dim":32,"pref_attn_embd_dim":32,"n_head":4,"action_embd_dim":8},
            dataset_options={"max_length":12},mode="diverse")
        records.append({"model":"MultiPrefTransformerDivide/diverse","status":"PASS","result":asdict(result)})
    except Exception as exc:
        records.append({"model":"MultiPrefTransformerDivide/diverse","status":"BLOCKED","error":str(exc),"traceback":traceback.format_exc()})
    print("diverse",records[-1]["status"],records[-1].get("error",""),flush=True)
    (root/"docs/evidence/stage4/reward_probes.json").write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__=="__main__": main()
