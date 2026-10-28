"""Derive a stage-1 migration ledger from audit evidence, without importing sources."""
from __future__ import annotations

import ast
import collections
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import pickletools
import re
import sys
import zipfile

from audit_sources import WORKSPACE, PROJECT, OUT, ROOTS, KOU, read, dump, rel


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8-sig"))


def main():
    symbols = load("python_symbols.json")
    regs = load("registries.json")
    configs = load("configs.json")
    by_path = {s["path"]: s for s in symbols}
    supplemental = [WORKSPACE / "pymarl-master/pymarl-master" / f for f in ["requirements.txt", "README.md", "LICENSE"]]
    supplemental += [WORKSPACE / KOU / "pymarl.yml", WORKSPACE / KOU / "ICQ-MA/requirements.txt", WORKSPACE / KOU / "ICQ-MA/README.md"]
    supplemental += list((WORKSPACE / KOU).glob("*.py"))
    supplemental += list((WORKSPACE / "marl平台代码/易才润/在线代码编辑调试微服务系统/tc-web-bf源代码闭源版").glob("*.yml"))
    extra_manifest = [{"path": rel(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size} for p in supplemental if p.is_file()]
    dump("supplemental_manifest.json", extra_manifest)

    def resolve_module(source, module, level=0):
        root = WORKSPACE / ROOTS[source["project"]]
        base = (WORKSPACE / source["path"]).parent if level else root
        for _ in range(max(0, level - 1)):
            base = base.parent
        target = base.joinpath(*(module or "").split(".")) if module else base
        for candidate in [target.with_suffix(".py"), target / "__init__.py"]:
            if candidate.is_file():
                return candidate
        if target.is_dir():
            return target
        return None

    unresolved = []
    for s in symbols:
        root = WORKSPACE / ROOTS[s["project"]]
        local_names = {p.stem for p in root.iterdir() if p.is_dir() or p.suffix == ".py"}
        for imp in s["imports"]:
            mod = imp["module"] or ""
            if (imp["level"] or mod.split(".")[0] in local_names) and resolve_module(s, mod, imp["level"]) is None:
                unresolved.append({"project": s["project"], "source": s["path"], "line": imp["line"], "module": mod,
                                   "level": imp["level"], "interpretation": "Static resolution only; dynamic sys.path or external namespaces need review."})
    dump("local_import_findings.json", unresolved)

    rows = []
    stages = {"pymarl": 2, "wang_rghmf": 3, "wang_hmf": 3, "kou_old": 4, "kou_icq": 4,
              "wang_sadma": 5, "yi_sadma_closed": 5, "yi_sadma_open": 5, "yi_sadma_zmq": 5}

    def add(pid, path, capability, category, target, evidence, blockers="", stage=None, relation="", deps=""):
        rows.append({"id": f"M{len(rows)+1:04d}", "source_project": pid, "source_path": path,
                     "capability": capability, "classification": category, "target_module": target,
                     "planned_stage": stage or stages.get(pid, 8), "dependencies": deps,
                     "migration_status": "NOT_STARTED", "implementation_verified": "NO", "local_test_verified": "NO",
                     "real_environment_verified": "NO", "evidence": evidence, "blockers": blockers,
                     "version_relation": relation})

    # Every registry assignment remains traceable, including components rather than only algorithm names.
    for r in regs:
        pid, path, key = r["project"], r["path"], r["key"]
        if pid not in stages:
            continue
        source = by_path[path]
        is_learner = "/learners/" in path
        is_policy = path.endswith("policy/__init__.py")
        if is_learner or is_policy:
            category, target = "ALGORITHM_IMPLEMENTATION", f"learners/{pid}"
        elif "/envs/" in path:
            category, target = "ENVIRONMENT_ADAPTER", "envs"
        elif "/runner/" in path or "/worker/" in path or "/runners/" in path:
            category, target = "TRAINING_EXECUTION", "execution/zmq" if key in {"train", "sample"} or "/worker/" in path else "runners"
        else:
            category, target = "ALGORITHM_COMPONENT", "controllers" if "/controllers/" in path else "modules"
        deps, impls = set(), []
        try:
            names = {n.id for n in ast.walk(ast.parse(r["value"], mode="eval")) if isinstance(n, ast.Name)}
        except SyntaxError:
            names = set()
        for imp in source["imports"]:
            if names.intersection(imp.get("names", [])):
                p = resolve_module(source, imp["module"], imp["level"])
                if p is not None and p.is_file():
                    impls.append(rel(p))
                    for i in by_path.get(rel(p), {}).get("imports", []):
                        mod = (i["module"] or "").split(".")[0]
                        if mod in {"torch", "numpy", "gym", "smac", "pysc2", "zmq", "transformers", "wandb", "sklearn", "ReplenishmentEnv", "cityflow", "cityflowenv", "pandapower", "cloudpickle"}:
                            deps.add(mod)
        relation = ""
        if pid == "yi_sadma_zmq":
            category = "DUPLICATE_IMPLEMENTATION" if is_policy else category
            relation = "51 inventoried files byte-identical to yi_sadma_open; retain provenance"
        elif pid == "yi_sadma_closed":
            relation = "41 Python ASTs equal to yi_sadma_open; config/scripts differ"
        elif pid == "wang_sadma":
            relation = "27 common Python ASTs equal to yi variants; remaining behavior/config/checkpoint differences retained"
        elif pid == "wang_rghmf" and key == "q_learner":
            category = "DUPLICATE_IMPLEMENTATION"
            relation = "learners/q_learner.py byte-identical to pymarl; whole pipelines not assumed equivalent"
        blocker = "No training baseline: torch/sacred/environment dependencies unavailable in audited interpreter"
        if pid in {"kou_old", "kou_icq"}:
            blocker += "; data path/label semantics and checkpoint round-trip unverified"
        if pid.startswith("yi_sadma") and is_policy:
            blocker += "; policy checkpoint path incomplete relative to wang variant"
        evidence = f"{path}:{r['line']} -> {r['value']}; implementations=" + "|".join(impls)
        add(pid, path, r["registry"] + "/" + key, category, target, evidence, blocker,
            stage=6 if target == "execution/zmq" else None, relation=relation, deps="|".join(sorted(deps)))

    config_findings = []
    for c in configs:
        pid = c["project"]
        if pid not in stages or not ("/algs/" in c["path"] or "/alg_config/" in c["path"]):
            continue
        vals = c["top_level_scalars"]
        learner = vals.get("learner", {}).get("value")
        known = {r["key"] for r in regs if r["project"] == pid and "/learners/" in r["path"]}
        missing = learner is not None and learner not in known
        if missing:
            config_findings.append({"project": pid, "path": c["path"], "learner": learner,
                                    "line": vals["learner"]["line"], "issue": "Configured learner is not registered"})
        add(pid, c["path"], Path(c["path"]).stem, "UNWIRED_EXPERIMENT" if missing else "CONFIG_VARIANT",
            "config/algs", "configs.json top-level scalar scan; learner=" + str(learner) + "; run=" + str(vals.get("run", {}).get("value")),
            "Learner missing from registry; do not fabricate implementation" if missing else "Config path/dispatch/runtime not validated",
            deps=learner or "policy_REGISTRY")
    dump("config_findings.json", config_findings)

    hmf_root = WORKSPACE / ROOTS["wang_hmf"]
    used = set()
    for entry in hmf_root.glob("main*.py"):
        tree = ast.parse(read(entry))
        used.update(n.func.value.id for n in ast.walk(tree) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute) and n.func.attr == "init_from_env" and isinstance(n.func.value, ast.Name))
    hmf_connections = []
    for s in symbols:
        if s["project"] != "wang_hmf" or "/algorithms/" not in s["path"]:
            continue
        for cls in s["classes"]:
            called = cls["name"] in used
            # Resolve name to module as well: several files intentionally define the same class name.
            module = Path(s["path"]).stem
            imported = any(re.search(r"from algorithms\." + re.escape(module) + r" import\b", read(p)) for p in hmf_root.glob("main*.py"))
            connected = called and imported
            hmf_connections.append({"path": s["path"], "class": cls["name"], "direct_main_dispatch": connected})
            add("wang_hmf", s["path"], cls["name"], "ALGORITHM_IMPLEMENTATION" if connected else "UNWIRED_EXPERIMENT",
                "learners/wang_hmf", f"Class at line {cls['line']}; direct main init_from_env dispatch={connected}",
                "Missing torch/yaml/gym/ReplenishmentEnv; hardcoded output root; transition/mean-field semantics unverified" +
                ("; no direct main dispatch confirmed (not proof of unused code)" if not connected else ""),
                deps="torch|numpy|gym|ReplenishmentEnv|tensorboard")
    dump("hmf_entry_connections.json", hmf_connections)

    for pid in ["wang_sadma", "yi_sadma_closed", "yi_sadma_open", "yi_sadma_zmq"]:
        p = WORKSPACE / ROOTS[pid] / "policy/maddpg"
        if p.is_dir():
            add(pid, rel(p), "MADDPG source without policy registration", "UNWIRED_EXPERIMENT", "learners/sadma/maddpg",
                "maddpg_controller.py and maddpg_net.py present; policy/__init__.py registers only ppo and qmix",
                "No registered Policy training entry; do not count as callable algorithm", stage=5, deps="torch")

    support = [
        ("pymarl", "components/episode_buffer.py", "EpisodeBatch/ReplayBuffer", "components", 2),
        ("pymarl", "run.py", "online training and process lifecycle", "trainers/online_off_policy", 2),
        ("kou_old", "main.py", "run/globalRM/pref/dataset dispatch", "trainers", 4),
        ("kou_old", "run_pref_new.py", "active preference training entry", "trainers/preference", 4),
        ("kou_old", "run_pref.py", "alternate preference implementation (not current main import)", "trainers/preference", 4),
        ("kou_old", "run_globalRM.py", "global reward model training", "trainers/reward", 4),
        ("kou_old", "run_make_datasets.py", "offline dataset generation", "datasets/generation", 4),
        ("kou_old", "components/reward_model.py", "reward model", "modules/reward_models", 4),
        ("kou_old", "components/reward_model_new.py", "reward model variant", "modules/reward_models", 4),
        ("kou_old", "components/preference.py", "preference construction", "datasets/preference", 4),
        ("kou_icq", "run.py", "offline/preference dispatch and evaluation", "trainers", 4),
        ("kou_icq", "components/reward_model.py", "MR/NMR/PrefTransformer/MultiPrefTransformerDivide", "modules/reward_models", 4),
        ("kou_icq", "components/MAPT", "embedded MAPT library incl unrelated environment assets", "modules/reward_models", 4),
        ("yi_sadma_open", "utils/buffer.py", "shared/episode/replay buffers", "adapters/sadma", 5),
        ("yi_sadma_open", "utils/scheme.py", "transition/episode/buffer schemes", "adapters/sadma", 5),
        ("yi_sadma_open", "worker", "ZMQ workers and parameter exchange", "execution/zmq", 6),
        ("wang_sadma", "policy/qmix/q_policy.py", "save_models/load_models", "checkpoints", 5),
        ("wang_sadma", "policy/mappo/ppo_policy.py", "save_models/load_models", "checkpoints", 5),
    ]
    for pid, suffix, cap, target, stage in support:
        p = WORKSPACE / ROOTS[pid] / suffix
        add(pid, rel(p), cap, "SUPPORTING_LIBRARY" if "components" in suffix or "utils" in suffix else "TRAINING_EXECUTION",
            target, "Source present=" + str(p.exists()) + "; see python_symbols.json", "Runtime dependencies and semantic validation pending", stage=stage)
    for p in (WORKSPACE / KOU).glob("*.py"):
        tree = ast.parse(read(p))
        imports = sorted({n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module} |
                         {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names})
        add("kou_tools", rel(p), p.stem, "EXTERNAL_SERVICE_CLIENT" if p.name == "use_llm.py" else "DATA_PREPARATION",
            "datasets/labeling", "AST inventory only; script not executed", "Model/service dependencies and local path assumptions unverified; do not initiate API calls", stage=4,
            deps="|".join(imports))
    for d in load("data_inventory.json"):
        add("kou_data", d["path"], Path(d["path"]).name, "DATASET", "datasets/external", "data_inventory.json; bytes=" + str(d["bytes"]) + "; archive directory readable; tensor content not loaded",
            "Torch unavailable; tensor shapes/pair ordering/labels/reward semantics not validated; original absolute paths need explicit mapping", stage=4, deps="torch")
    for pid, cap, deps in [("wang_operator", "TCJob operator (Go 1.19)", "Go|Kubernetes|controller-runtime"),
                           ("yi_operator", "TCJob operator + scheduling extensions", "Go|Kubernetes|controller-runtime"),
                           ("yi_backend", "FastAPI task/code/tensorboard backend", "FastAPI|SQLAlchemy|MySQL|Docker|Kubernetes"),
                           ("yi_frontend", "Vue3/ElementPlus/Vite frontend", "Node|pnpm|backend")]:
        add(pid, ROOTS[pid], cap, "EXTERNAL_SERVICE", "deployment/" + pid, "Source inventory; service not started",
            "Outside core Python package; no service/cluster/database baseline verified", stage=8, deps=deps)

    # Safe archive metadata inspection, without pickle.load or importing torch.
    metadata_rows = []
    for d in load("data_inventory.json"):
        p = WORKSPACE / d["path"]
        if not d.get("archive"):
            continue
        with zipfile.ZipFile(p) as z:
            name = next((m for m in z.namelist() if m.endswith("data.pkl")), None)
            if not name or z.getinfo(name).file_size > 2_000_000:
                continue
            strings = []
            for op, value, pos in pickletools.genops(z.read(name)):
                if op.name in {"BINUNICODE", "SHORT_BINUNICODE", "UNICODE"} and isinstance(value, str) and len(value) < 120:
                    strings.append(value)
            metadata_rows.append({"path": d["path"], "pickle_string_tokens": list(dict.fromkeys(strings)),
                                  "method": "pickletools.genops; no deserialization; tokens are not verified schema or tensor shapes"})
    dump("data_metadata_tokens.json", metadata_rows)

    smoke_source = WORKSPACE / ROOTS["pymarl"] / "components/epsilon_schedules.py"
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("audit_original_schedule", smoke_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    observations = []
    for decay in ["linear", "exp"]:
        sched = module.DecayThenFlatSchedule(1.0, 0.05, 100, decay=decay)
        for t in [0, 50, 100, 200]:
            value = float(sched.eval(t))
            assert 0.05 <= value <= 1.0
            observations.append({"decay": decay, "t": t, "epsilon": value})
        assert abs(float(sched.eval(0)) - 1.0) < 1e-9
        assert abs(float(sched.eval(200)) - 0.05) < 1e-9
    dump("component_smoke.json", {"source": rel(smoke_source), "result": "PASS", "cases": observations,
                                   "scope": "Original epsilon schedule component only; NOT a training baseline"})
    target = PROJECT / "docs/MIGRATION_MATRIX.csv"
    with target.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    dump("inventory_summary.json", {"matrix_rows": len(rows), "classifications": dict(collections.Counter(r["classification"] for r in rows)),
                                    "planned_stages": dict(collections.Counter(r["planned_stage"] for r in rows)),
                                    "unregistered_config_learners": config_findings, "local_import_findings": len(unresolved),
                                    "scope": "Rows include configs, components and provenance duplicates; NOT independent algorithm count"})
    print(json.dumps(load("inventory_summary.json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
