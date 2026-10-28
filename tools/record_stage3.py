"""Update source inventory only from recorded successful validation evidence."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import xml.etree.ElementTree as ET
import numpy
import torch
from unified_marl.algorithms.hmf.adapter import IMPLEMENTATIONS

root = Path(__file__).resolve().parents[1]
evidence = root / "docs/evidence/stage3"
installed = json.loads((evidence / "installed_validation.json").read_text(encoding="utf-8"))
probes = json.loads((evidence / "public_api_probes.json").read_text(encoding="utf-8"))
assert len(probes) == 23 and all(p["status"] == "PASS" and p["train"]["parameter_delta_l2"] > 0 for p in probes)
assert len(installed["algorithms"]) == 24 and all(a["train"]["updates"] == 2 for a in installed["algorithms"])
suite = ET.parse(evidence / "tests.xml").getroot().find("testsuite")
assert suite is not None and int(suite.attrib["failures"]) == int(suite.attrib["errors"]) == 0
parity = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(evidence.glob("parity-*.json"))]
assert len(parity) == 15 and all(p["result"] == "PASS" for p in parity)
verified = 0
for name in ["source_manifest.json", "supplemental_manifest.json"]:
    for row in json.loads((root / "docs/evidence" / name).read_text(encoding="utf-8")):
        assert hashlib.sha256((root.parent / row["path"]).read_bytes()).hexdigest() == row["sha256"], row["path"]
        verified += 1
path = root / "docs/MIGRATION_MATRIX.csv"
snapshot = root / "docs/evidence/stage2_matrix.csv"
if not snapshot.exists():
    snapshot.write_bytes(path.read_bytes())
with path.open(encoding="utf-8-sig", newline="") as file:
    reader = csv.DictReader(file)
    fields, rows = reader.fieldnames, list(reader)
proof = "docs/evidence/stage3/tests.xml; public_api_probes.json; installed_validation.json; source_comparisons.json"
common_blocker = "CPU deterministic toy only; real SMAC/ReplenishmentEnv/data and CUDA training not validated"
remaining = {p["name"]: p for p in json.loads((evidence / "remaining_variant_probes.json").read_text(encoding="utf-8"))}
module_to_public = {module: name for name, (module, cls) in IMPLEMENTATIONS.items()}
dynamic_files = {"hierarchical_vdn copy", "hierarchical_vdn_k_means"}
for row in rows:
    family = row["source_project"]
    if family not in {"wang_rghmf", "wang_hmf"}:
        continue
    row.update(implementation_verified="YES", local_test_verified="YES", real_environment_verified="NO",
               migration_status="INTEGRATED_LOCAL", blockers=common_blocker)
    if proof not in row["evidence"]:
        row["evidence"] += "; " + proof
    if family == "wang_rghmf":
        component = row["capability"]
        source_path = row["source_path"]
        row["target_module"] = "src/unified_marl/algorithms/rghmf"
        if component in {"REGISTRY/sc2", "REGISTRY/replenishment"}:
            env = component.split("/")[-1]
            row.update(target_module="src/unified_marl/envs/" + ("smac.py" if env == "sc2" else "replenishment.py"),
                       migration_status="IMPLEMENTED_ENV_BLOCKED")
        elif component == "REGISTRY/q" and ("/roles/" in source_path or "/role_selectors/" in source_path):
            row.update(migration_status="SOURCE_ONLY_UNVALIDATED", implementation_verified="NO", local_test_verified="NO",
                blockers="Alternative q role/selector has no delivered active configuration; selector hardcodes 3 roles while default is 2; post-clustering resizing/optimizer schedule unresolved")
        elif "/config/algs/" in source_path:
            row["target_module"] = "wang/" + component
        elif component in {"REGISTRY/multinomial", "REGISTRY/epsilon_greedy", "REGISTRY/basic_mac", "REGISTRY/rnn",
                            "REGISTRY/q_learner", "REGISTRY/coma_learner", "REGISTRY/qtran_learner"}:
            row["target_module"] = "shared PyMARL core + wang configuration"
            relation = "network/learner files byte identical; delivered basic selectors fail missing schedule arguments; validated core selectors used"
            if relation not in row["version_relation"]:
                row["version_relation"] += "; " + relation
        elif component in {"REGISTRY/episode", "REGISTRY/parallel"}:
            row["target_module"] = "src/unified_marl/runners + role fields"
        if component in {"REGISTRY/rode_learner", "REGISTRY/rode_hmf_learner", "rode", "rode_hmf"}:
            row["migration_status"] = "PARTIAL_PRE_CLUSTERING_VALIDATED"
            row["blockers"] += "; dynamic role expansion training/resume not validated; source optimizer omissions and grouping indices preserved (STAGE3_REPORT.md)"
    else:
        module = Path(row["source_path"]).stem
        public = module_to_public.get(module)
        if public:
            row["target_module"] = "wang_hmf/" + public
            if module in dynamic_files:
                row.update(migration_status="PARTIAL_FIXED_GROUP_VALIDATED", classification="CONFIG_VARIANT",
                    blockers=common_blocker + "; default fixed-group update only; update_dynamic_group and clustering schedule not migrated")
            elif module == "hierarchical_vdn_02_formulate_meanQ":
                row["classification"] = "ALGORITHM_VARIANT"
                relation = "no main branch, explicit default update integrated and compared against source"
                if relation not in row["version_relation"]:
                    row["version_relation"] += "; " + relation
        else:
            key = {"hierarchical_vdn_IPC": "hierarchical_vdn_dynamic_ipc"}.get(module, module)
            row.update(migration_status="BLOCKED_SOURCE_IMPLEMENTATION", implementation_verified="NO", local_test_verified="NO",
                target_module="unregistered source under algorithms/hmf/algorithms/" + module + ".py",
                blockers=remaining[key]["error"] + "; no confirmed full training schedule; source namespaced only, not counted migrated")

new = [
    ("role EpisodeBatch fields and bootstrap alignment", "src/unified_marl/api.py + runners/episode_runner.py", "BATCH_SCHEMA", "INTEGRATED_LOCAL", "YES", "YES", common_blocker),
    ("HMF fixed-group transition and MFQ mean-field bridge", "src/unified_marl/algorithms/hmf/adapter.py", "TRAINING_EXECUTION", "INTEGRATED_LOCAL", "YES", "YES", common_blocker + "; only explicit same-episode next-action path"),
    ("replenishment external factory, padding and reward adapter", "src/unified_marl/envs/replenishment.py", "ENVIRONMENT_ADAPTER", "IMPLEMENTED_ENV_BLOCKED", "YES", "YES", "Fake backend contract passed only; external simulator, task and data absent; seed consumption unverified"),
    ("dynamic IPC/K-means grouping collection and high-level scheduling", "marl平台代码/王思哲/code/large_scale/hmf_replenishment/main_replenishment_IPC.py", "UNWIRED_EXPERIMENT", "NOT_MIGRATED", "NO", "NO", "Source dynamic-group sample includes agents_group/next_agents_group; fixed-group registration does not implement this collection/update schedule"),
    ("post-clustering role expansion training", "marl平台代码/王思哲/code/large_scale/RGHMF/controllers/rode_controller.py", "ALGORITHM_VARIANT", "SOURCE_LIMITATION", "NO", "NO", "New role parameters missing from source optimizer; RODE-HMF n_groups fixed; standalone clustering parity does not validate post-expansion updates"),
]
for capability, source_path, category, status, impl, local, blockers in new:
    canonical_source = source_path.split(" + ", 1)[0]
    if canonical_source.startswith("src/"):
        canonical_source = "unified-marl/" + canonical_source
    if any(r["source_project"] == "integration" and r["capability"] == capability for r in rows):
        for row in rows:
            if row["source_project"] == "integration" and row["capability"] == capability:
                row["source_path"] = canonical_source
        continue
    item = dict.fromkeys(fields, "")
    item.update(id=f"M{max(int(r['id'][1:]) for r in rows)+1:04d}", source_project="integration", source_path=canonical_source,
        capability=capability, classification=category, target_module=source_path, planned_stage="3",
        dependencies="torch|numpy|scikit-learn (role)|gym (HMF)|external ReplenishmentEnv (real env)",
        migration_status=status, implementation_verified=impl, local_test_verified=local, real_environment_verified="NO",
        evidence=proof, blockers=blockers)
    rows.append(item)
with path.open("w", encoding="utf-8-sig", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
summary = {"stage": 3, "status": "LOCAL_INTEGRATIONS_PASSED_WITH_EXPLICIT_REMAINING_SOURCE_AND_ENV_BLOCKERS",
    "python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "numpy": numpy.__version__,
    "tests": int(suite.attrib["tests"]), "failed": 0, "source_parity_reports": len(parity),
    "public_algorithms_and_configurations": len(probes), "installed_algorithms_including_qmix_regression": len(installed["algorithms"]),
    "real_environments": installed["real_environments"], "source_hash_records_unchanged": verified,
    "matrix_rows": len(rows), "wang_statuses": dict(Counter(r["migration_status"] for r in rows if r["source_project"].startswith("wang_"))),
    "wheel": str(root / "dist/unified_marl-0.3.0-py3-none-any.whl"), "stopping_after_stage": 3}
(evidence / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: summary[k] for k in ["tests", "failed", "source_parity_reports", "source_hash_records_unchanged", "matrix_rows"]}))
