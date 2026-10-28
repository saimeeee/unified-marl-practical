"""Review every retained migration row; package relative source provenance."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/stage7"
EVIDENCE.mkdir(parents=True, exist_ok=True)
matrix = ROOT / "docs/MIGRATION_MATRIX.csv"
snapshot = ROOT / "docs/evidence/stage6_matrix.csv"
if not snapshot.exists(): shutil.copyfile(matrix, snapshot)
rows = list(csv.DictReader(snapshot.open(encoding="utf-8-sig")))
license_root = ROOT / "licenses"
license_root.mkdir(exist_ok=True)
mit = ROOT.parent / "marl平台代码/易才润/论文/SADMA/sadma源代码开源版本/LICENSE"
shutil.copyfile(mit, license_root / "SADMA-Yi-MIT.txt")
shutil.copyfile(ROOT / "LICENSE", license_root / "PyMARL-Apache-2.0.txt")

for row in rows:
    assert (ROOT.parent / row["source_path"]).exists(), row["id"]
    state = row["migration_status"]
    if row["id"] in {"M0188", "M0189", "M0217"}:
        row.update(migration_status="INTEGRATED_SOURCE_AGENT_VARIANT", implementation_verified="YES",
            local_test_verified="YES", synthetic_test_verified="YES",
            target_module="src/unified_marl/algorithms/kou/" + ("icq" if row["id"]=="M0217" else "old") + "/modules/agents",
            blockers="CPU fixed-input loss/update source parity and offline save/load/resume passed for BC rnn_ns or old QMIX mlp/mlp_ns only; other bundle combinations and real environment/effectiveness are not validated",
            evidence=row["evidence"]+"; stage7/extra-agents-tests.xml; stage7/agent-*/parity-policy-*.json; release-full.json")
        if (EVIDENCE/"release-full.json").exists(): row["real_data_verified"]="YES"
    if row["id"] in {"M0278","M0284","M0286","M0289"}:
        row["evidence"] += "; stage7/remaining-probes.json: actual failure reconfirmed"
        if row["id"]=="M0289":
            row["blockers"] += "; deeper source inspection: update needs next_state/rews/log_probs/values/entropy/done, calls missing Agent.critic/eval_optimizer and misspelled adv.detacj(); supplying num_evaders alone cannot restore the actor-critic protocol"
    if row["id"] == "M0279":
        row.update(migration_status="PARTIAL_FIXED_CRITIC_UPDATE_VERIFIED",
            target_module="src/unified_marl/algorithms/hmf/algorithms/hierarchical_vdac.py + adapter.py",
            implementation_verified="YES", local_test_verified="YES", synthetic_test_verified="YES",
            blockers="Only fixed-group update() and existing local/mf critic targets integrated as wang_hmf/hierarchical_vdac_fixed; source Agent has no mf_policy/target_mf_policy, while update_high_level/prep_* reference them. Full VDAC/actor and alternate schedule remain unimplemented; no real replenishment backend/assets.",
            evidence=row["evidence"]+"; stage7/vdac-tests.xml; stage7 fixed variant source parity with explicit uint8->bool and nonexistent target-policy guard")
    if state == "NOT_MIGRATED_CONTINUOUS_EVALUATE_WORKER":
        row.update(migration_status="PARTIAL_PERIODIC_EVALUATION_REPLACEMENT_VERIFIED",
            target_module="src/unified_marl/execution/evaluation.py + trainer.py",
            implementation_verified="YES", local_test_verified="YES", synthetic_test_verified="YES",
            evidence=row["evidence"]+"; stage7/periodic-evaluation.json; stage7/evaluation-tests.xml",
            blockers="Bounded spawn policy-snapshot evaluator integrated and tested locally, with update-based trigger, one pending task, timeout/error propagation and cleanup. Original perpetual EvaluateWorker/shared-buffer trigger, vector evaluation and remote evaluation not reproduced; real simulator unavailable.")
    if row["local_test_verified"] == "YES":
        decision = "ACCEPTED_ONLY_WITH_RECORDED_LIMITS"
    elif row["planned_stage"] == "8":
        decision = "OPTIONAL_STAGE8_RETAINED_NOT_EXECUTED"
    else:
        decision = "UNACCEPTED_RETAINED_WITH_BLOCKER"
    row["stage7_review"] = decision
    if row["local_test_verified"] != "YES":
        assert row["blockers"], row["id"]

def add(capability, source, target, blockers, evidence):
    row = {key: "" for key in rows[0]}
    row.update(id=f"M{len(rows)+1:04d}", source_project="unified_stage7", source_path=source,
        capability=capability, classification="INTEGRATION_CAPABILITY", target_module=target, planned_stage="7",
        migration_status="LOCAL_RELEASE_ACCEPTANCE_VERIFIED", implementation_verified="YES", local_test_verified="YES",
        synthetic_test_verified="YES", real_environment_verified="NO", real_data_verified="NO",
        policy_effect_verified="NO", cross_machine_verified="NO", blockers=blockers, evidence=evidence,
        stage7_review="ACCEPTED_ONLY_WITH_RECORDED_LIMITS")
    rows.append(row)

add("hierarchical_vdac_fixed existing-critic variant", rows[278]["source_path"],
    "src/unified_marl/algorithms/hmf/adapter.py", "Fixed update only; source VDAC actor/high-level methods still blocked by absent policy networks; CPU toy, no real environment", "stage7/vdac-tests.xml; stage7/release-full.json")
add("reward fixed-pair next-update continuation", "unified-marl/src/unified_marl/rewards/service.py",
    "rewards/service.py:load_reward_model(for_training=True)", "Six preference-mode models on explicit pairs; high-level diverse trust curriculum/pool state is not resumed", "stage7/api-tests.xml; stage7/release-full.json")
add("seeded evaluate/CLI and strict configuration errors", "unified-marl/src/unified_marl/cli.py",
    "api.py + cli.py + config/__init__.py", "Only recorded Windows/Python/CPU stack verified", "stage7/api-tests.xml; stage7/release-base.json")
add("bounded independent periodic distributed evaluation", "unified-marl/src/unified_marl/execution/evaluation.py",
    "execution/evaluation.py + trainer.py", "Loopback toy only, one evaluator, skipped-busy triggers recorded; source vector/perpetual evaluator ABI and real/remote evaluation not validated", "stage7/periodic-evaluation.json; stage7/release-full.json")
with matrix.open("w", encoding="utf-8-sig", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

provenance = {}
for manifest in sorted((ROOT / "docs/evidence").glob("stage*/migration_sources.json")):
    for item in json.loads(manifest.read_text(encoding="utf-8")):
        source, target = item["source"], item["target"]
        source_path, target_path = ROOT.parent/source, ROOT/target
        assert source_path.is_file() and target_path.is_file(), item
        provenance[(source, target)] = dict(source=source, source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
            target=target, target_sha256=hashlib.sha256(target_path.read_bytes()).hexdigest(),
            evidence=manifest.relative_to(ROOT).as_posix())
for path in (ROOT / "src/unified_marl").rglob("*"):
    if path.suffix not in {".py", ".yaml"}: continue
    target = path.relative_to(ROOT).as_posix()
    if not any(key[1] == target for key in provenance):
        provenance[("unified-marl/"+target,target)] = dict(source="unified-marl/"+target,
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), target=target,
            target_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), evidence="New integration/adapter or resource; source-family component trace remains in MIGRATION_MATRIX.csv")
trace = list(provenance.values())
with (ROOT/"docs/MIGRATION_TRACEABILITY.csv").open("w",encoding="utf-8-sig",newline="") as stream:
    writer=csv.DictWriter(stream,fieldnames=list(trace[0]));writer.writeheader();writer.writerows(trace)
(ROOT/"src/unified_marl/PROVENANCE.json").write_text(json.dumps(trace,ensure_ascii=False,indent=2),encoding="utf-8")
audit = dict(matrix_rows=len(rows), unique_ids=len({row["id"] for row in rows}), source_paths_exist=True,
    retained_previous_rows=len(rows)-4, original_rows_removed=0, trace_records=len(trace),
    review_decisions={value:sum(r["stage7_review"]==value for r in rows) for value in {r["stage7_review"] for r in rows}},
    unaccepted=[{k:r[k] for k in ["id","source_project","source_path","capability","migration_status","blockers"]} for r in rows if r["local_test_verified"]!="YES"],
    partial=[{k:r[k] for k in ["id","capability","migration_status","blockers"]} for r in rows if r["migration_status"].startswith("PARTIAL")],
    licenses={"pymarl":"Apache-2.0; preserved", "yi_open_and_zmq":"MIT XJTU 2024; preserved", "wang":"No overall grant found; confirmation required", "kou":"No overall grant found; confirmation required", "yi_closed":"No grant inferred from open distribution; confirmation required"})
(EVIDENCE/"migration-audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in audit.items() if k not in {"unaccepted","partial"}},ensure_ascii=False))
