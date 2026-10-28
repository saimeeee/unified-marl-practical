"""Record measured stage-4 evidence and preserve all previous inventory rows."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

root=Path(__file__).resolve().parents[1]
evidence=root/"docs/evidence/stage4"
installed=json.loads((evidence/"installed_validation.json").read_text(encoding="utf-8"))
assert len(installed["policies"])==17 and len(installed["reward_models"])==7
assert all(r["train"]["parameter_delta_l2"]>0 and r["train"]["env_steps"]==0 for r in installed["policies"])
assert all(r["result"]["parameter_delta_l2"]>0 for r in installed["reward_models"])
suite=ET.parse(evidence/"tests.xml").getroot().find("testsuite")
assert int(suite.attrib["failures"])==int(suite.attrib["errors"])==0
parity=[json.loads(p.read_text(encoding="utf-8")) for p in evidence.glob("parity-*.json")]
assert len(parity)==24 and all(p["result"]=="PASS" for p in parity)
datasets=json.loads((evidence/"dataset_batches.json").read_text(encoding="utf-8"))
assert len(datasets)==12 and all(p["status"]=="SMALL_BATCH_READ_PASS" for p in datasets)
source_count=0
for name in ["source_manifest.json","supplemental_manifest.json"]:
    for item in json.loads((root/"docs/evidence"/name).read_text(encoding="utf-8")):
        assert hashlib.sha256((root.parent/item["path"]).read_bytes()).hexdigest()==item["sha256"],item["path"]
        source_count+=1
matrix=root/"docs/MIGRATION_MATRIX.csv"
snapshot=root/"docs/evidence/stage3_matrix.csv"
if not snapshot.exists(): snapshot.write_bytes(matrix.read_bytes())
with matrix.open(encoding="utf-8-sig",newline="") as file:
    reader=csv.DictReader(file)
    fields,rows=list(reader.fieldnames),list(reader)
for name in ["synthetic_test_verified","real_data_verified","policy_effect_verified"]:
    if name not in fields: fields.append(name)
    for row in rows: row.setdefault(name,"NOT_RECORDED")
proof="docs/evidence/stage4/tests.xml; installed_validation.json; 24 parity reports; dataset_batches.json; STAGE4_REPORT.md"
common="CPU short updates only; real SMAC evaluation blocked; no policy effectiveness/convergence claim"
icq_missing={"hacpl_smac":"hacpl_learner","maicpl_smac":"maicpl_learner","maidpo_smac":"maidpo_learner"}
icq_broken={"qmix","qmix_smac","vdn_smac"}
old_blocked={"qmix_differ":"Requires original per-agent indi_terminated; dataset only supplies team terminated",
    "qmix_new":"Requires the actual reference policy checkpoint, per-agent termination and reward-model integration",
    "pref_qmix":"Requires actual reference policy checkpoint and per-agent termination/labels",
    "pref_qmix_new":"Requires actual reference policy checkpoint, agent_state and compatible new local reward model",
    "qmix_make_datasets":"Online source data-collection runner not migrated; real SMAC and author source .npy assets unavailable"}
for row in rows:
    family=row["source_project"]
    if family not in {"kou_icq","kou_old","kou_data","kou_tools"}: continue
    source=row["source_path"]
    capability=row["capability"]
    row.update(migration_status="INTEGRATED_OFFLINE_LOCAL",implementation_verified="YES",local_test_verified="YES",
        real_environment_verified="NO",synthetic_test_verified="YES",real_data_verified="UPDATE_PASS",policy_effect_verified="NO",blockers=common)
    if proof not in row["evidence"]: row["evidence"]+="; "+proof
    if family=="kou_data":
        row["target_module"]="src/unified_marl/datasets/episodes.py"
        row.update(migration_status="REAL_DATA_READ_VERIFIED",synthetic_test_verified="NOT_APPLICABLE",real_data_verified="READ_PASS")
        if source.endswith("MADiversePreferDatasets/3m/3m.th"):
            row.update(migration_status="REAL_DATA_UPDATE_VERIFIED",real_data_verified="UPDATE_PASS")
        elif source.endswith("MADiversePreferDatasets/3m/reward.th"):
            row.update(migration_status="REAL_INDIVIDUAL_REWARD_INPUT_VERIFIED",real_data_verified="UPDATE_INPUT_PASS",
                blockers=common+"; learned individual reward [400,60,3,1] kept separate from team reward; 61st slot is not fabricated")
        continue
    if family=="kou_tools":
        if capability=="get_preferences":
            row.update(migration_status="PURE_FEATURE_HELPERS_INTEGRATED",target_module="src/unified_marl/datasets/prompts.py",real_data_verified="READ_PASS",
                blockers="3m source feature comparison and explicit template read passed; other map encodings not validated; no remote calls")
        elif capability=="make_diverse_preference":
            row.update(migration_status="PARTIAL_PURE_SEGMENT_HELPER",target_module="src/unified_marl/datasets/segments.py",real_data_verified="NO",
                blockers="Only pure clip_to_segment_length is migrated and source-compared; LLM label generation/retries and automatic file rewriting are not executed")
        else:
            row.update(migration_status="EXTERNAL_SERVICE_NOT_EXECUTED",implementation_verified="NO",local_test_verified="NO",synthetic_test_verified="NO",real_data_verified="NO",
                blockers="LLM client is an external-service boundary; no new-label generation/API calls requested or executed; existing labels are read explicitly")
        continue
    row["target_module"]="src/unified_marl/algorithms/kou/"+("icq" if family=="kou_icq" else "old")+" + trainers/offline.py"
    if "/config/algs/" in source:
        name=Path(source).stem
        row["target_module"]=("kou/" if family=="kou_icq" else "kou_old/")+name
        if family=="kou_icq" and name in icq_missing:
            row.update(migration_status="BLOCKED_MISSING_SOURCE_LEARNER",implementation_verified="NO",local_test_verified="NO",synthetic_test_verified="NO",real_data_verified="NO",
                blockers="Configuration references absent/unregistered "+icq_missing[name])
        elif family=="kou_icq" and name in icq_broken:
            row.update(migration_status="BLOCKED_INCOMPATIBLE_SOURCE_CONTROLLER",implementation_verified="NO",local_test_verified="NO",synthetic_test_verified="NO",real_data_verified="NO",
                blockers="Delivered QLearner expects Q/imitator/auxiliary triple; registered BasicMAC returns one tensor. No matching imitation agent/controller supplied; not replaced by PyMARL QMIX")
        elif family=="kou_icq" and name=="acpl_smac":
            row.update(migration_status="BLOCKED_INCOMPLETE_SOURCE_CONFIG",implementation_verified="NO",local_test_verified="NO",synthetic_test_verified="NO",real_data_verified="NO",
                blockers="ACPL config lacks NLL_lamda/use_nll etc and selects Q outputs while bound CPL expects policy probabilities; original dispatcher has no ACPL branch")
        elif family=="kou_old" and name in old_blocked:
            row.update(migration_status="BLOCKED_SOURCE_ASSETS_OR_PROTOCOL",implementation_verified="NO",local_test_verified="NO",synthetic_test_verified="NO",real_data_verified="NO",blockers=old_blocked[name])
        elif family=="kou_old" and name in {"cw_qmix_w_globalRM","ow_qmix_w_globalRM"}:
            row.update(migration_status="PARTIAL_BASE_TD_METHOD_VERIFIED",blockers=common+"; source train() uses original team reward; full online unsupervised/relabel/train_reward schedule is not migrated")
    elif capability.startswith("REGISTRY/"):
        symbol=capability.split("/",1)[1]
        unsupported={"grnn","rnn_ns","mlp","mlp_ns","maddpg_mac","coma_critic","maddpg_critic_ns"}
        if symbol in unsupported:
            row.update(migration_status="SOURCE_ONLY_COMPONENT_UNVALIDATED",implementation_verified="NO",local_test_verified="NO",synthetic_test_verified="NO",real_data_verified="NO",
                blockers="No verified delivered discrete offline bundle uses this component; GRNN additionally needs original SMAC feature metadata/communication settings")
        elif family=="kou_icq" and symbol=="q_learner":
            row.update(migration_status="BLOCKED_INCOMPATIBLE_SOURCE_CONTROLLER",implementation_verified="NO",local_test_verified="NO",synthetic_test_verified="NO",real_data_verified="NO",
                blockers="BCQ-like triple-output contract mismatches all supplied active basic-controller configurations")
        elif symbol in {"q_learner_new","qdiffer_learner","pref_qmix_learner","pref_qmix_learner_new"}:
            row.update(migration_status="BLOCKED_SOURCE_ASSETS_OR_PROTOCOL",implementation_verified="NO",local_test_verified="NO",synthetic_test_verified="NO",real_data_verified="NO",
                blockers="Missing actual reference policy, individual termination and/or agent-state reward protocol; source presence alone not counted integrated")
        elif symbol in {"episode","parallel","pref_episode","pref_episode_new","makedata_runner"}:
            row.update(migration_status="NOT_MIGRATED_SOURCE_ONLINE_EXECUTION",implementation_verified="NO",local_test_verified="NO",synthetic_test_verified="NO",real_data_verified="NO",
                blockers="Stage 4 uses an independent offline update trainer; historical online preference/collection runner lifecycle and its extra fields are not migrated")
        elif symbol=="sc2":
            row.update(migration_status="ENVIRONMENT_BLOCKED",target_module="src/unified_marl/envs/smac.py",synthetic_test_verified="NOT_APPLICABLE",real_data_verified="NOT_APPLICABLE",
                blockers="Shared optional SMAC factory exists, but actual old/ICQ SMAC wrapper and map metadata were not validated against a running simulator")
        elif family=="kou_old" and symbol=="max_q_learner_w_globalRM":
            row.update(migration_status="PARTIAL_BASE_TD_METHOD_VERIFIED",blockers=common+"; source online reward/unsupervised methods remain unconnected")
    elif capability.startswith("env_REGISTRY/"):
        row.update(migration_status="OUTSIDE_REWARD_IMPORT_CLOSURE",implementation_verified="NO",local_test_verified="NO",synthetic_test_verified="NO",real_data_verified="NO",
            blockers="Embedded MuJoCo/swimmer environment is not a reward-model dependency and is not integrated by this stage")
    elif capability=="MR/NMR/PrefTransformer/MultiPrefTransformerDivide":
        row.update(migration_status="PARTIAL_REWARD_MODELS_AND_DIVERSE_CURRICULUM",target_module="src/unified_marl/rewards/service.py",
            blockers=common+"; source NMR reuses left prediction; PrefTransformer unregistered blocks/optimizer omission preserved; calibration train_predicter/e2e not integrated; diverse filter can select zero pairs")
    elif capability=="embedded MAPT library incl unrelated environment assets":
        row.update(migration_status="PARTIAL_REWARD_IMPORT_CLOSURE_ONLY",target_module="src/unified_marl/algorithms/kou/reward_models",
            blockers="Only actual reward-model Python dependency closure is bundled; embedded simulators, other MAPT policy algorithms and unused JAX helpers are not integrated")
    elif capability=="reward model":
        row.update(migration_status="PARTIAL_GLOBAL_MODELS_VERIFIED",target_module="rewards/service.py:old_global,old_reward_global",
            blockers=common+"; local/individual reward training and policy-oracle label generation require missing masks/reference assets")
    elif capability in {"reward model variant","active preference training entry","alternate preference implementation (not current main import)","preference construction"}:
        row.update(migration_status="BLOCKED_REFERENCE_POLICY_OR_AGENT_FIELDS",implementation_verified="NO",local_test_verified="NO",synthetic_test_verified="NO",real_data_verified="NO",
            blockers="Active old_src workflow preserved in audit; actual reference policy checkpoint plus agent_state/indi_terminated and original local-label protocol are missing")
    elif capability=="offline/preference dispatch and evaluation":
        row.update(migration_status="PARTIAL_OFFLINE_DISPATCH_REPLACED",target_module="src/unified_marl/trainers/offline.py",
            blockers=common+"; e2e, predictor calibration, behavior pretraining/moderate CQL and PER are explicitly rejected rather than silently ignored")
    elif capability in {"run/globalRM/pref/dataset dispatch","global reward model training","offline dataset generation"}:
        row.update(migration_status="PARTIAL_OFFLINE_COMPONENTS_ONLY",real_data_verified="READ_PASS" if capability=="offline dataset generation" else "UPDATE_PASS",
            blockers="Offline model/pairing components are integrated; original interactive feedback, unsupervised reset, online collection and hand-filtered .npy pipeline are not executed")

model_sources={"MR":"MR.py","NMR":"NMR.py","PrefTransformer":"PrefTransformer.py","MultiPrefTransformerDivide":"MultiPrefTransformer.py"}
additions=[
    ("integration","canonical offline tensor dataset","unified-marl/src/unified_marl/datasets/episodes.py","DATA_SCHEMA","INTEGRATED_LOCAL", "YES","READ_PASS",common),
    ("integration","PreferenceBatch with independent trajectory masks and label roles","unified-marl/src/unified_marl/datasets/episodes.py","DATA_SCHEMA","INTEGRATED_LOCAL","YES","UPDATE_PASS",common),
    ("integration","independent offline update clock and periodic explicit evaluation","unified-marl/src/unified_marl/trainers/offline.py","TRAINING_EXECUTION","INTEGRATED_LOCAL","YES","UPDATE_PASS",common),
    ("integration","reward checkpoint includes source unregistered GPT2 blocks","unified-marl/src/unified_marl/rewards/service.py","CHECKPOINT","INTEGRATED_LOCAL","YES","UPDATE_PASS","PrefTransformer retains original stochastic dropout and optimizer omission; use identical RNG state to compare predictions"),
]
for name,filename in model_sources.items():
    source="marl平台代码/寇谦/论文代码及技术文档/论文代码及技术文档/ICQ-MA/src/components/MAPT/mat/algorithms/reward_model/models/"+filename
    limitations={"NMR":"Source reuses first prediction in second trajectory objective; equal masks can cancel gradients; full real 3m trajectories have verified nonzero update",
        "PrefTransformer":"Source Python-list blocks omitted from optimizer and eval propagation; full weights additionally checkpointed; stochastic inference retained"}.get(name,common)
    additions.append(("kou_reward",name,source,"REWARD_MODEL","INTEGRATED_WITH_SOURCE_LIMITATIONS","YES","UPDATE_PASS",limitations))
for name in ["old_global","old_reward_global"]:
    additions.append(("kou_reward",name,"marl平台代码/寇谦/论文代码及技术文档/论文代码及技术文档/old_src/components/reward_model.py",
        "REWARD_MODEL","PARTIAL_GLOBAL_ONLY_VERIFIED","YES","UPDATE_PASS",common+"; individual/local reward methods not migrated"))
additions.append(("kou_reward","MultiPrefTransformerDivide/diverse","marl平台代码/寇谦/论文代码及技术文档/论文代码及技术文档/ICQ-MA/src/components/reward_model.py",
    "REWARD_MODEL_VARIANT","PARTIAL_INITIAL_EPOCH_VERIFIED","YES","UPDATE_PASS","Original global/between/in losses and KL curriculum integrated; first real epoch update passed; next synthetic epoch selects zero pairs and explicitly blocks"))
for family,capability,source,classification,status,synthetic,real,blockers in additions:
    if any(r["source_project"]==family and r["capability"]==capability for r in rows): continue
    item=dict.fromkeys(fields,"")
    item.update(id=f"M{max(int(r['id'][1:]) for r in rows)+1:04d}",source_project=family,source_path=source,capability=capability,
        classification=classification,target_module="src/unified_marl/rewards/service.py" if family=="kou_reward" else source,
        planned_stage="4",dependencies="torch|numpy; reward extra for MAPT; no simulator required",
        migration_status=status,implementation_verified="YES",local_test_verified="YES",real_environment_verified="NO",
        synthetic_test_verified=synthetic,real_data_verified=real,policy_effect_verified="NO",evidence=proof,blockers=blockers)
    rows.append(item)
with matrix.open("w",encoding="utf-8-sig",newline="") as file:
    writer=csv.DictWriter(file,fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
summary={"stage":4,"version":"0.4.0","status":"LOCAL_AND_REAL_DATA_UPDATES_VERIFIED_WITH_SOURCE_AND_ENVIRONMENT_BLOCKERS",
    "python":sys.version,"tests":int(suite.attrib["tests"]),"failed":0,"policy_source_parity_reports":17,"reward_source_parity_reports":7,
    "real_episode_files_small_batch_read":12,"installed_policy_entries":17,"installed_reward_model_entries":7,
    "original_source_hashes_unchanged":source_count,"matrix_rows":len(rows),"real_smac_evaluation":"BLOCKED",
    "policy_effectiveness":"NOT_VALIDATED","stopped_after_stage":4,"matrix_stage4_statuses":dict(Counter(r["migration_status"] for r in rows if r["planned_stage"]=="4")),
    "wheel":str(root/"dist/unified_marl-0.4.0-py3-none-any.whl")}
(evidence/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({k:summary[k] for k in ["tests","failed","original_source_hashes_unchanged","matrix_rows","installed_policy_entries","installed_reward_model_entries"]}))
