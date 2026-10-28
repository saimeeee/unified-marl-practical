"""Read-only source/data audit for Kou delivery. Never execute legacy mains."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import torch
import yaml

sys.dont_write_bytecode = True
root = Path(__file__).resolve().parents[1]
base = root.parent / "marl平台代码/寇谦/论文代码及技术文档/论文代码及技术文档"
out = root / "docs/evidence/stage4"
out.mkdir(parents=True, exist_ok=True)
sources = []
for family, folder in [("icq", base / "ICQ-MA/src"), ("old", base / "old_src")]:
    for p in sorted(folder.rglob("*.py")):
        if "MAPT" in p.parts or "__pycache__" in p.parts:
            continue
        tree = ast.parse(p.read_text(encoding="utf-8"))
        if p.parent.name not in {"learners", "controllers"} and "reward_model" not in p.name:
            continue
        attrs = sorted({n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and
            ((isinstance(n.value, ast.Name) and n.value.id == "args") or
             (isinstance(n.value, ast.Attribute) and n.value.attr == "args"))})
        sources.append({"family": family, "file": str(p.relative_to(base)).replace("\\", "/"),
            "classes": [{"name": c.name, "methods": {n.name: ast.unparse(n.args) for n in c.body if isinstance(n, ast.FunctionDef)}}
                        for c in tree.body if isinstance(c, ast.ClassDef)], "args": attrs,
            "imports": [ast.unparse(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]})
configs = {}
for family, folder in [("icq", base / "ICQ-MA/src"), ("old", base / "old_src")]:
    configs[family] = {p.name: yaml.safe_load(p.read_text(encoding="utf-8")) for p in (folder / "config/algs").glob("*.yaml")}
data = []
for p in sorted(base.rglob("*.th")):
    if "MAPT" in p.parts:
        continue
    row = {"path": str(p.relative_to(root.parent)).replace("\\", "/"), "bytes": p.stat().st_size}
    try:
        value = torch.load(p, map_location="cpu", weights_only=True, mmap=True)
        entries = value if isinstance(value, dict) else {"<root>": value}
        row["fields"] = {k: {"type": type(v).__name__, "shape": list(v.shape) if isinstance(v, torch.Tensor) else None,
            "dtype": str(v.dtype) if isinstance(v, torch.Tensor) else None} for k, v in entries.items()}
        row["label_summary"] = {k: {"values": torch.unique(v).tolist()[:20], "mean": v.float().mean().item()}
             for k, v in entries.items() if isinstance(v, torch.Tensor) and "label" in k}
        row["status"] = "TENSORS_READ"
        del entries, value
    except Exception as exc:
        row.update(status="READ_BLOCKED", error=f"{type(exc).__name__}: {exc}")
    data.append(row)
(out / "audit.json").write_text(json.dumps({"sources": sources, "configs": configs, "data": data}, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"source_modules": len(sources), "data": [{"path": Path(d["path"]).name,
    "status": d["status"], "fields": d.get("fields", {}), "labels": d.get("label_summary", {})} for d in data]}, ensure_ascii=False))
