"""Update the inventory from measured SADMA evidence, preserving historical rows."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

root=Path(__file__).resolve().parents[1]
evidence=root/"docs/evidence/stage5"
installed=json.loads((evidence/"installed_validation.json").read_text(encoding="utf-8"))
assert installed["version"]=="0.5.0" and len(installed["algorithms"])==7
assert all(item["result"]["parameter_delta_l2"]>0 and item["resume"]["updates"]==3 for item in installed["algorithms"])
assert all(item["returncode"]==0 for item in installed["cli"])
suite=ET.parse(evidence/"tests.xml").getroot().find("testsuite")
assert int(suite.attrib["tests"])==149 and int(suite.attrib["failures"])==int(suite.attrib["errors"])==0
parity=[json.loads(path.read_text(encoding="utf-8")) for path in evidence.glob("parity-*.json")]
assert len(parity)==13 and all(item["result"]=="PASS" for item in parity)
audit=json.loads((evidence/"audit.json").read_text(encoding="utf-8"))
assert not audit["compared_to_yi_open"]["yi_zmq"]
hashes=0
for name in ["source_manifest.json","supplemental_manifest.json"]:
    for row in json.loads((root/"docs/evidence"/name).read_text(encoding="utf-8")):
        assert hashlib.sha256((root.parent/row["path"]).read_bytes()).hexdigest()==row["sha256"],row["path"]
        hashes+=1
for row in audit["source_hashes"]:
    assert hashlib.sha256((root.parent/row["source"]).read_bytes()).hexdigest()==row["sha256"],row["source"]
matrix=root/"docs/MIGRATION_MATRIX.csv"
snapshot=root/"docs/evidence/stage4_matrix.csv"
if not snapshot.exists(): snapshot.write_bytes(matrix.read_bytes())
with matrix.open(encoding="utf-8-sig",newline="") as file:
    reader=csv.DictReader(file)
    fields,rows=list(reader.fieldnames),list(reader)
proof="docs/STAGE5_REPORT.md; SADMA_VERSION_SELECTION.md; docs/evidence/stage5/tests.xml; 13 parity JSON; installed_validation.json; time_semantics.json; lifecycle.json"
common="CPU toy short updates only; no real simulator, GPU or policy effectiveness validation; source PPO GAE/entropy/LR quirks retained"
profiles={"wang_sadma":"wang","yi_sadma_open":"yi_open","yi_sadma_closed":"yi_closed","yi_sadma_zmq":"yi_open"}
for row in rows:
    family=row["source_project"]
    if family not in profiles: continue
    cap,path=row["capability"],row["source_path"]
    profile=profiles[family]
    row.update(real_environment_verified="NO",real_data_verified="NOT_APPLICABLE",policy_effect_verified="NO")
    row["version_relation"]+="; stage5: "+("Yi open/ZMQ byte-identical comparison; same yi_open profile" if family=="yi_sadma_zmq" else "primary Wang formulas, preserved "+profile+" config profile")
    if proof not in row["evidence"]: row["evidence"]+="; "+proof
    if row["planned_stage"]=="6":
        row.update(migration_status="DEFERRED_STAGE6_NOT_EXECUTED",implementation_verified="NO",local_test_verified="NO",synthetic_test_verified="NO",
                   blockers="Distributed workers, ZMQ framing, shared IPC buffers, parameter exchange and asynchronous execution are not integrated in local stage 5")
        continue
    row.update(migration_status="INTEGRATED_SADMA_LOCAL_SYNC",implementation_verified="YES",local_test_verified="YES",synthetic_test_verified="YES",blockers=common)
    row["target_module"]="src/unified_marl/algorithms/sadma/native + algorithms/sadma/adapter.py"
    if "/maddpg" in path:
        row.update(migration_status="BLOCKED_EMPTY_SOURCE_POLICY",implementation_verified="NO",local_test_verified="NO",synthetic_test_verified="NO",
            blockers="All four delivered maddpg_policy.py files are empty and policy_REGISTRY has no MADDPG entry; actor/critic networks alone are not a training algorithm")
    elif cap.startswith(("env_REGISTRY/","env_runner_REGISTRY/")):
        env=cap.split("/",1)[1]
        row["target_module"]="src/unified_marl/envs/"+{"smac":"smac.py","replenishment":"replenishment.py","cityflow":"cityflow.py","powergrid":"CONTINUOUS_SCHEMA_BLOCKED"}[env]
        if env=="powergrid":
            row.update(migration_status="BLOCKED_CONTINUOUS_ACTION_SCHEMA_AND_ASSETS",implementation_verified="NO",local_test_verified="NO",synthetic_test_verified="GUARD_ONLY",
                blockers="Continuous [B,T,N,D] float actions, Gaussian log-prob semantics and bounds must extend canonical scheme/Policy/Runner; no discretization. pandapower/backend data absent in inspected setup")
        elif env=="smac":
            row.update(migration_status="SHARED_ADAPTER_REAL_ENV_BLOCKED",local_test_verified="NO",synthetic_test_verified="GUARD_ONLY",
                blockers="Optional shared SMAC factory implemented; no running SMAC/SC2/map validation; source scheme translated to canonical fields")
        else:
            row.update(migration_status="ADAPTER_MOCK_BACKEND_ONLY",blockers=common+"; only simulated backend contract/update/close tested; actual "+env+" backend absent")
            if env=="replenishment": row["blockers"]+="; source backend horizon preserved; unused key/map_name/reward retained as metadata; source profit queries not integrated"
    elif cap in {"shared/episode/replay buffers","transition/episode/buffer schemes"}:
        row.update(migration_status="PARTIAL_LOCAL_BUFFER_AND_SCHEMA_VERIFIED",target_module="src/unified_marl/algorithms/sadma/native/utils/buffer.py + adapter.py",
            blockers=common+"; native ReplayBuffer and arrival-time EpisodeBuffer equivalence verified; IPC SharedBuffer and distributed transition framing not executed")
    elif cap.startswith("scheme_REGISTRY/"):
        row.update(migration_status="CANONICAL_SCHEMA_BRIDGE_VERIFIED",target_module="src/unified_marl/api.py:make_scheme + algorithms/sadma/adapter.py:native_batch",
            blockers=common+"; transition fields used locally; original ZMQ send metadata not enabled")
    elif cap=="runner_REGISTRY/sync":
        row.update(migration_status="LOCAL_SYNC_REPLACEMENT_VERIFIED",target_module="src/unified_marl/trainers/sadma.py + runners/episode_runner.py",
            blockers=common+"; fresh PPO lifecycle replaces source stale replay; actual steps exclude resets; one environment collector, no distributed workers")
    elif cap=="save_models/load_models":
        row.update(migration_status="PACKAGE_FULL_CHECKPOINT_VERIFIED",target_module="src/unified_marl/checkpoints.py + algorithms/sadma/adapter.py:extra_state",
            blockers=common+"; policy.pt/training.pt preserve target lag/ValueNorm/optimizer/LR/version/replay; legacy agent.pth folder import is not verified")
    elif "/config/alg_config/" in path:
        name=Path(path).stem
        prefix={"wang":"sadma","yi_open":"sadma_yi","yi_closed":"sadma_yi_closed"}[profile]
        row["target_module"]=prefix+"/"+name
        if name=="qmix" and profile=="yi_open": row["blockers"]+="; original qmix name actually selects VDN Mixer, explicitly retained"

additions=[
 ("SADMA local synchronous trainer","src/unified_marl/trainers/sadma.py","TRAINING_EXECUTION","INTEGRATED_LOCAL_SYNC","YES","YES",common),
 ("SADMA canonical transition bridge and native replay","src/unified_marl/algorithms/sadma/adapter.py","DATA_SCHEMA","INTEGRATED_LOCAL_SYNC","YES","YES",common),
 ("PPO fresh-rollout lifecycle and behavior metadata","src/unified_marl/trainers/sadma.py","ON_POLICY_DATA_LIFECYCLE","INTEGRATED_LOCAL_SYNC","YES","YES",common+"; old probabilities/values/versions stored; stale policy data rejected"),
 ("SADMA versioned full checkpoint","src/unified_marl/algorithms/sadma/adapter.py","CHECKPOINT","INTEGRATED_LOCAL_SYNC","YES","YES",common+"; exact continuation tested only at completed-update toy boundaries; pending PPO rollouts discarded and counted"),
 ("CityFlow optional backend bridge","src/unified_marl/envs/cityflow.py","ENVIRONMENT_ADAPTER","MOCK_BACKEND_ONLY","YES","YES",common+"; actual engine/traffic maps absent; close forwarded only when external backend exposes it"),
 ("Continuous action schema extension design","docs/INTEGRATION_SPEC.md","PLANNED_SCHEMA_EXTENSION","BLOCKED_CONTINUOUS_ACTION_IMPLEMENTATION","NO","GUARD_ONLY","Design and explicit rejection only; no Gaussian float-action Runner/Policy/checkpoint integration or real grid assets validation"),
 ("SADMA explicit VDN mixer variant","src/unified_marl/config/sadma/wang/qmix.yaml","ALGORITHM_VARIANT","INTEGRATED_LOCAL_SYNC","YES","YES",common+"; sadma/vdn explicitly overrides mixer, not an independent delivered algorithm"),
 ("SADMA preserved source configuration profiles","src/unified_marl/config/sadma","CONFIGURATION","INTEGRATED_LOCAL_SYNC","YES","YES",common+"; Wang, Yi open and Yi closed; Yi ZMQ duplicate tracked in provenance"),
]
for capability,target,classification,status,implemented,synthetic,blockers in additions:
    if any(row["source_project"]=="integration" and row["capability"]==capability for row in rows): continue
    item=dict.fromkeys(fields,"")
    item.update(id=f"M{max(int(row['id'][1:]) for row in rows)+1:04d}",source_project="integration",source_path="unified-marl/"+target,
        capability=capability,classification=classification,target_module=target,planned_stage="5",dependencies="torch|numpy|PyYAML; external simulators optional",
        migration_status=status,implementation_verified=implemented,local_test_verified="YES" if implemented=="YES" else "NO",
        real_environment_verified="NO",synthetic_test_verified=synthetic,real_data_verified="NOT_APPLICABLE",policy_effect_verified="NO",evidence=proof,blockers=blockers)
    rows.append(item)
with matrix.open("w",encoding="utf-8-sig",newline="") as file:
    writer=csv.DictWriter(file,fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
summary=dict(stage=5,version="0.5.0",status="LOCAL_SYNCHRONOUS_VERIFIED_WITH_SOURCE_AND_ENVIRONMENT_LIMITATIONS",tests=int(suite.attrib["tests"]),failures=0,
    source_parity_reports=len(parity),installed_configs=len(installed["algorithms"]),tested_config_and_branch_cases=13,
    source_hash_records_unchanged=hashes,sadma_compared_hash_records_unchanged=len(audit["source_hashes"]),matrix_rows=len(rows),
    real_environments="BLOCKED",policy_effectiveness="NOT_VALIDATED",distributed_stage6="NOT_EXECUTED",stopped_after_stage=5,
    matrix_sadma_statuses=dict(Counter(row["migration_status"] for row in rows if row["source_project"] in profiles)),
    wheel=str(root/"dist/unified_marl-0.5.0-py3-none-any.whl"))
(evidence/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(summary,ensure_ascii=False))
