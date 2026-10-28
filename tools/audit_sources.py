"""Read-only source audit. Writes evidence only under unified-marl/docs/evidence.

Run using a known interpreter; no dependencies or source modules are imported.
"""
from __future__ import annotations

import ast
import collections
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tokenize
import zipfile
from datetime import datetime, timezone, timedelta

WORKSPACE = Path(__file__).resolve().parents[2]
PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "docs" / "evidence"
KOU = "marl平台代码/寇谦/论文代码及技术文档/论文代码及技术文档"
YI = "marl平台代码/易才润"
WANG = "marl平台代码/王思哲/code"
ROOTS = {
    "pymarl": "pymarl-master/pymarl-master/src",
    "wang_rghmf": WANG + "/large_scale/RGHMF",
    "wang_hmf": WANG + "/large_scale/hmf_replenishment",
    "wang_sadma": WANG + "/sadma",
    "yi_sadma_closed": YI + "/论文/SADMA/sadma源代码闭源版本",
    "yi_sadma_open": YI + "/论文/SADMA/sadma源代码开源版本",
    "yi_sadma_zmq": YI + "/基于ZMQ的分布式多智能体强化学习算法/sadma源代码公开版本",
    "kou_old": KOU + "/old_src",
    "kou_icq": KOU + "/ICQ-MA/src",
    "wang_operator": WANG + "/operator",
    "yi_operator": YI + "/基于K8s的集群资源调度管理系统/operator源代码闭源版本",
    "yi_backend": YI + "/在线代码编辑调试微服务系统/tc-web-bf源代码闭源版/taichu-web-backend",
    "yi_frontend": YI + "/在线代码编辑调试微服务系统/tc-web-bf源代码闭源版/vue-element-plus-admin",
}
SKIP = {".git", "__pycache__", "__MACOSX", "node_modules", "wandb", "results", "bin", ".idea"}
SOURCE_EXT = {".py", ".yaml", ".yml", ".sh", ".go", ".mod", ".sum", ".toml", ".json", ".vue", ".ts", ".js", ".md"}


def rel(p):
    return p.relative_to(WORKSPACE).as_posix()


def read(p):
    if p.suffix == ".py":
        with tokenize.open(p) as f:
            return f.read()
    return p.read_text(encoding="utf-8-sig", errors="replace")


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def files(root):
    for base, dirs, names in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP)
        for name in sorted(names):
            if not name.startswith("._"):
                yield Path(base) / name


def signature(node):
    return ast.unparse(node.args)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source_manifest, details, registries, configs, invalid, projects = [], [], [], [], [], []
    for project_id, location in ROOTS.items():
        root = WORKSPACE / location
        ps = list(files(root))
        projects.append({"id": project_id, "root": location, "exists": root.exists(),
                         "files": len(ps), "extensions": dict(collections.Counter(p.suffix for p in ps))})
        for p in ps:
            if p.suffix in SOURCE_EXT or p.name in {"LICENSE", "Dockerfile", "Makefile", "requirement.txt", "requirements.txt"}:
                source_manifest.append({"project": project_id, "path": rel(p), "bytes": p.stat().st_size,
                                        "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
            if p.suffix in {".yaml", ".yml"} and "config" in p.parts:
                values = {}
                for i, line in enumerate(read(p).splitlines(), 1):
                    m = re.match(r"^([A-Za-z_][\w-]*):\s*(.*?)\s*(?:#.*)?$", line)
                    if m:
                        values[m[1]] = {"value": m[2].strip().strip("\"'"), "line": i}
                configs.append({"project": project_id, "path": rel(p), "top_level_scalars": values})
            if p.suffix != ".py":
                continue
            try:
                text = read(p)
                tree = ast.parse(text, filename=rel(p))
            except (SyntaxError, UnicodeError) as exc:
                invalid.append({"path": rel(p), "error": str(exc)})
                continue
            imports, methods, classes, hazards = [], [], [], []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend({"module": a.name, "line": node.lineno, "level": 0} for a in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imports.append({"module": node.module, "names": [a.name for a in node.names],
                                    "line": node.lineno, "level": node.level})
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Subscript) and "registry" in ast.unparse(target.value).lower():
                            registries.append({"project": project_id, "path": rel(p), "line": node.lineno,
                                               "registry": ast.unparse(target.value), "key": ast.unparse(target.slice).strip("\"'"),
                                               "value": ast.unparse(node.value)})
                elif isinstance(node, ast.Call):
                    func = ast.unparse(node.func)
                    if func in {"os._exit", "sys.exit", "yaml.load", "torch.save", "torch.load", "th.save", "th.load"}:
                        hazards.append({"call": func, "line": node.lineno})
                elif isinstance(node, ast.ClassDef):
                    classes.append({"name": node.name, "line": node.lineno,
                                    "methods": [{"name": n.name, "signature": signature(n), "line": n.lineno}
                                                for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]})
            methods = [{"name": n.name, "signature": signature(n), "line": n.lineno}
                       for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            details.append({"project": project_id, "path": rel(p), "imports": imports, "classes": classes,
                            "functions": methods, "notable_calls": hazards,
                            "ast_sha256": hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()})

    comparisons = []
    by_project = collections.defaultdict(dict)
    for f in source_manifest:
        path = str(Path(f["path"]).relative_to(Path(ROOTS[f["project"]]))).replace("\\", "/")
        by_project[f["project"]][path] = f
    ast_by_path = {f["path"]: f["ast_sha256"] for f in details}
    for a, b in [("wang_sadma", "yi_sadma_closed"), ("yi_sadma_open", "yi_sadma_zmq"),
                 ("yi_sadma_closed", "yi_sadma_open"), ("wang_sadma", "yi_sadma_open"),
                 ("wang_operator", "yi_operator"), ("pymarl", "wang_rghmf"),
                 ("pymarl", "kou_old"), ("pymarl", "kou_icq"), ("kou_old", "kou_icq")]:
        rows = []
        for name in sorted(set(by_project[a]) | set(by_project[b])):
            x, y = by_project[a].get(name), by_project[b].get(name)
            status = "only_right" if x is None else "only_left" if y is None else "identical" if x["sha256"] == y["sha256"] else "different"
            ast_equal = None
            if x and y and name.endswith(".py"):
                ast_equal = ast_by_path.get(x["path"]) is not None and ast_by_path.get(x["path"]) == ast_by_path.get(y["path"])
            rows.append({"relative_path": name, "status": status, "ast_equal": ast_equal})
        comparisons.append({"left": a, "right": b, "summary": dict(collections.Counter(r["status"] for r in rows)),
                            "python_summary": dict(collections.Counter(r["status"] for r in rows if r["relative_path"].endswith(".py"))),
                            "ast_equal_python_files": sum(r["ast_equal"] is True for r in rows), "files": rows})

    data = []
    for p in files(WORKSPACE / KOU):
        if p.suffix.lower() in {".th", ".pt", ".pth", ".h5", ".hdf5", ".pkl", ".pickle", ".npy", ".npz"}:
            row = {"path": rel(p), "bytes": p.stat().st_size, "content_loaded": False}
            if p.suffix.lower() in {".th", ".pt", ".pth", ".npz"} and zipfile.is_zipfile(p):
                with zipfile.ZipFile(p) as z:
                    members = z.infolist()
                    row.update({"archive": True, "members": len(members),
                                "uncompressed_bytes": sum(m.file_size for m in members),
                                "metadata_members": [m.filename for m in members if not re.search(r"/data/\d+$", m.filename)][:10]})
            data.append(row)

    modules = ["numpy", "torch", "yaml", "sacred", "smac", "pysc2", "zmq", "gym", "gymnasium", "cityflow", "cityflowenv",
               "ReplenishmentEnv", "pandapower", "wandb", "h5py", "sklearn", "tensorboard_logger", "kubernetes", "fastapi",
               "tensorboard", "cloudpickle", "transformers", "einops", "ml_collections", "pandas", "scipy"]
    from importlib import metadata
    distnames = {"yaml": "PyYAML", "zmq": "pyzmq", "sklearn": "scikit-learn"}
    env = {"time": datetime.now(timezone(timedelta(hours=8))).isoformat(), "python": sys.version,
           "executable": sys.executable, "platform": platform.platform(), "modules": {}}
    for name in modules:
        info = {"find_spec": importlib.util.find_spec(name) is not None}
        try:
            info["distribution_version"] = metadata.version(distnames.get(name, name))
        except metadata.PackageNotFoundError:
            pass
        env["modules"][name] = info
    env["env_paths"] = {k: os.environ.get(k) for k in ["SC2PATH", "CUDA_PATH", "CONDA_PREFIX", "VIRTUAL_ENV"]}
    try:
        proc = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=15)
        env["nvidia_smi"] = {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
    except Exception as exc:
        env["nvidia_smi"] = {"error": str(exc)}

    # These original entry probes fail before training in the audited dependency environment.
    probes = []
    entries = [("pymarl", "main.py", ["--config=qmix", "--env-config=sc2", "with", "env_args.map_name=3m", "use_cuda=False", "t_max=1"]),
               ("wang_rghmf", "main.py", ["--config=rode_hmf", "--env-config=sc2", "with", "env_args.map_name=3m", "use_cuda=False", "t_max=1"]),
               ("wang_hmf", "main_replenishment.py", ["--help"]),
               ("wang_sadma", "entry.py", ["--help"]),
               ("yi_sadma_closed", "entry.py", ["--help"]),
               ("yi_sadma_open", "entry.py", ["--help"]),
               ("yi_sadma_zmq", "entry.py", ["--help"]),
               ("kou_old", "main.py", ["--config=qmix", "--env-config=sc2", "with", "use_cuda=False", "t_max=1"]),
               ("kou_icq", "main.py", ["--config=offpg_smac", "--env-config=sc2", "with", "use_cuda=False", "t_max=1"])]
    if not env["modules"]["torch"]["find_spec"] and not env["modules"]["sacred"]["find_spec"]:
        for pid, entry, args in entries:
            path = WORKSPACE / ROOTS[pid] / entry
            command = [sys.executable, "-B", str(path), *args]
            child_env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
            try:
                proc = subprocess.run(command, cwd=path.parent, env=child_env, capture_output=True,
                                      text=True, encoding="utf-8", errors="replace", timeout=20)
                probes.append({"project": pid, "command": command, "cwd": str(path.parent), "returncode": proc.returncode,
                               "stdout": proc.stdout[-6000:], "stderr": proc.stderr[-6000:], "training_verified": False})
            except subprocess.TimeoutExpired:
                probes.append({"project": pid, "command": command, "timeout_seconds": 20, "training_verified": False})
    else:
        probes.append({"skipped": "Dependencies differ from audited missing-dependency precondition; manual baseline required."})

    manifests = {"source_manifest.json": source_manifest, "source_inventory.json": projects,
                 "python_symbols.json": details, "registries.json": registries, "configs.json": configs,
                 "syntax_findings.json": invalid, "version_comparisons.json": comparisons,
                 "data_inventory.json": data, "environment.json": env, "baseline_probes.json": probes}
    for name, obj in manifests.items():
        dump(name, obj)
    # Validate preservation of all source files hashed before subprocess probes.
    changed = [r["path"] for r in source_manifest if hashlib.sha256((WORKSPACE / r["path"]).read_bytes()).hexdigest() != r["sha256"]]
    dump("source_preservation.json", {"hashed_files": len(source_manifest), "changed_files": changed,
                                      "scope": "Manifested source/resource files; not a full byte snapshot of all deliveries."})
    print(json.dumps({"projects": projects, "python_files": len(details), "source_files_hashed": len(source_manifest),
                      "registrations": len(registries), "configs": len(configs), "syntax_findings": len(invalid),
                      "data_files": len(data), "data_bytes": sum(r["bytes"] for r in data),
                      "comparisons": [{k: v for k, v in c.items() if k != "files"} for c in comparisons],
                      "baseline_failures": [{"project": p.get("project"), "last_error": p.get("stderr", "").splitlines()[-1:]}
                                            for p in probes], "changed_source_files": changed}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
