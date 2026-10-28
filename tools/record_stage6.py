"""Record measured loopback execution without claiming remote validation."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

root=Path(__file__).resolve().parents[1]
evidence=root/"docs/evidence/stage6"
installed=json.loads((evidence/"installed_validation.json").read_text(encoding="utf-8"))
assert len(installed["runs"])==9 and all(row["result"]["updates"]==2 and row["result"]["parameter_delta_l2"]>0 and row["ports_rebound"] for row in installed["runs"])
assert installed["cross_machine_verified"] is False and len(installed["failures"])==2 and len(installed["local_regressions"])==6
suite=ET.parse(evidence/"tests.xml").getroot().find("testsuite")
assert int(suite.attrib["tests"])==196 and int(suite.attrib["failures"])==int(suite.attrib["errors"])==0
for filename in ["separate-roles.json","fault-raise.json","fault-stall.json","connection-timeout.json","startup-failure.json","async-vtrace-source-parity.json"]:
    assert json.loads((evidence/filename).read_text(encoding="utf-8"))["result"]=="PASS"
count=0
for filename in ["source_manifest.json","supplemental_manifest.json"]:
    for row in json.loads((root/"docs/evidence"/filename).read_text(encoding="utf-8")):
        assert hashlib.sha256((root.parent/row["path"]).read_bytes()).hexdigest()==row["sha256"],row["path"]
        count+=1
audit=json.loads((evidence/"audit.json").read_text(encoding="utf-8"))
for row in audit["sources"]:
    assert hashlib.sha256((root.parent/row["source"]).read_bytes()).hexdigest()==row["sha256"]
matrix=root/"docs/MIGRATION_MATRIX.csv"
snapshot=root/"docs/evidence/stage5_matrix.csv"
if not snapshot.exists(): snapshot.write_bytes(matrix.read_bytes())
with matrix.open(encoding="utf-8-sig",newline="") as file:
    reader=csv.DictReader(file)
    fields,rows=list(reader.fieldnames),list(reader)
for column in ["cross_machine_verified","execution_modes"]:
    if column not in fields: fields.append(column)
    for row in rows: row.setdefault(column,"NOT_RECORDED")
proof="docs/STAGE6_REPORT.md; docs/ZMQ_PROTOCOL.md; docs/evidence/stage6/tests.xml; installed_validation.json; separate-roles.json; fault-raise.json; fault-stall.json; startup-failure.json; connection-timeout.json"
families={"wang_sadma","yi_sadma_open","yi_sadma_closed","yi_sadma_zmq"}
common="Windows loopback TCP only; cross-machine, real simulators, GPU and policy effects NOT verified; new protocol=1 replaces original binary ABI"
for row in rows:
    if row["source_project"] not in families: continue
    cap=row["capability"]
    if "/config/alg_config/" in row["source_path"] or cap in {"policy_REGISTRY/qmix","policy_REGISTRY/ppo"}:
        is_ppo=cap.endswith("ppo")
        row["execution_modes"]="local|zmq_sync|zmq_async_if_use_vtrace" if is_ppo else "local|zmq_sync|zmq_async"
        row["cross_machine_verified"]="NO"
        if proof not in row["evidence"]: row["evidence"]+="; "+proof
    if row["planned_stage"]!="6": continue
    row.update(implementation_verified="YES",local_test_verified="YES",synthetic_test_verified="YES",real_data_verified="NOT_APPLICABLE",
        real_environment_verified="NO",policy_effect_verified="NO",cross_machine_verified="NO",migration_status="ZMQ_LOOPBACK_ROLES_VERIFIED",
        target_module="src/unified_marl/execution",execution_modes="SADMA Q/VDN:sync,async; PPO:sync; PPO+vtrace:async",blockers=common)
    if proof not in row["evidence"]: row["evidence"]+="; "+proof
    if cap in {"runner_REGISTRY/train","worker_REGISTRY/train_worker"}:
        row["target_module"]="src/unified_marl/execution/trainer.py"
        row["blockers"]+="; one learner; source loss/target/replay reused, ordinary PPO stale replay explicitly disallowed"
    elif cap in {"runner_REGISTRY/sample","worker_REGISTRY/remote_sample_worker"}:
        row.update(migration_status="SEPARATE_ROLE_LOOPBACK_ONLY",target_module="src/unified_marl/execution/sampler.py",
            blockers=common+"; independent process roles verified locally; network endpoints implemented but no supplied remote host")
    elif cap=="worker_REGISTRY/local_sample_worker":
        row.update(migration_status="REPLACED_MANAGED_ZMQ_SAMPLER",target_module="src/unified_marl/execution/sampler.py",
            blockers=common+"; managed loopback RPC replaces original shared-memory local worker; old IPC queue ABI not supported")
    elif cap=="worker_REGISTRY/infer_worker": row["target_module"]="src/unified_marl/execution/inference.py"
    elif cap in {"worker_REGISTRY/infer_client","worker_REGISTRY/infer_server"}:
        row["target_module"]="src/unified_marl/execution/protocol.py + trainer.py"
        row["blockers"]+="; bounded JSON/ndarray messages, strict schema/version checks, per-socket owner and finite deadlines; no original InferClient ABI compatibility"
    elif cap=="worker_REGISTRY/evaluate_worker":
        row.update(migration_status="NOT_MIGRATED_CONTINUOUS_EVALUATE_WORKER",implementation_verified="NO",local_test_verified="NO",synthetic_test_verified="NO",
            target_module="Existing evaluate(checkpoint) API only",execution_modes="local post-training evaluation only",
            blockers="Original long-running EvaluateWorker/shared buffer trigger is not migrated; local saved-policy evaluate() passed; no distributed evaluation worker claim")
    elif cap=="ZMQ workers and parameter exchange":
        row.update(migration_status="PARTIAL_ZMQ_ROLES_LOOPBACK_VERIFIED",blockers=common+"; persistent evaluator and original shared-memory queue path remain unintegrated")
additions=[
 ("ZMQ role/mode/device/seed configuration","src/unified_marl/execution/settings.py","EXECUTION_CONFIGURATION"),
 ("Versioned JSON ndarray message protocol","src/unified_marl/execution/protocol.py","COMMUNICATION_PROTOCOL"),
 ("Acknowledged actor and critic parameter snapshots","src/unified_marl/execution/inference.py","PARAMETER_SYNCHRONIZATION"),
 ("Behavior and reward-time trace validation","src/unified_marl/execution/batches.py","DATA_VALIDATION"),
 ("ZMQ synchronous cohort barrier and fresh PPO","src/unified_marl/execution/trainer.py","ON_POLICY_EXECUTION"),
 ("ZMQ asynchronous Q and explicit vtrace PPO replay","src/unified_marl/execution/trainer.py","ASYNCHRONOUS_EXECUTION"),
 ("Bounded credits HWM and communication deadlines","src/unified_marl/execution/protocol.py","BACKPRESSURE"),
 ("Spawn worker startup stop and failure cleanup","src/unified_marl/execution/trainer.py","PROCESS_LIFECYCLE"),
 ("Separate train inference and sample CLI roles","src/unified_marl/cli.py","CLI_EXECUTION"),
 ("Distributed learner replay continuation boundary","src/unified_marl/execution/trainer.py","CHECKPOINT_EXECUTION"),
]
for capability,target,classification in additions:
    if any(row["source_project"]=="integration" and row["capability"]==capability for row in rows): continue
    item=dict.fromkeys(fields,"")
    item.update(id=f"M{max(int(row['id'][1:]) for row in rows)+1:04d}",source_project="integration",source_path="unified-marl/"+target,
        capability=capability,classification=classification,target_module=target,planned_stage="6",dependencies="torch|numpy|PyYAML|pyzmq==25.1.2",
        migration_status="ZMQ_LOOPBACK_VERIFIED",implementation_verified="YES",local_test_verified="YES",synthetic_test_verified="YES",real_data_verified="NOT_APPLICABLE",
        real_environment_verified="NO",policy_effect_verified="NO",cross_machine_verified="NO",execution_modes="SADMA Q/VDN sync,async; PPO sync; PPO+vtrace async",
        evidence=proof,blockers=common+("; restarted worker RNG, external simulator and async scheduling not exact replay" if classification=="CHECKPOINT_EXECUTION" else ""))
    rows.append(item)
with matrix.open("w",encoding="utf-8-sig",newline="") as file:
    writer=csv.DictWriter(file,fieldnames=fields)
    writer.writeheader();writer.writerows(rows)
summary=dict(stage=6,version="0.6.0",status="WINDOWS_LOOPBACK_VERIFIED_REMOTE_AND_ENVIRONMENT_UNVERIFIED",tests=196,failures=0,
    installed_training_cases=9,installed_policy_profiles_sync=7,installed_async_cases=2,cli_managed_cases=2,independent_roles_loopback=True,
    failure_checks=["sampler_exception","sampler_stall","connection_timeout","inference_startup_bind_failure","context_permission_failure"],
    source_hash_records_unchanged=count,extra_communication_source_hashes_unchanged=len(audit["sources"]),matrix_rows=len(rows),
    local_regressions=6,cross_machine_verified=False,real_environment_verified=False,gpu_verified=False,policy_effectiveness="NOT_VALIDATED",stopped_after_stage=6,
    source_stage6_statuses=dict(Counter(row["migration_status"] for row in rows if row["source_project"] in families and row["planned_stage"]=="6")),
    wheel=str(root/"dist/unified_marl-0.6.0-py3-none-any.whl"))
(evidence/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(summary,ensure_ascii=False))
