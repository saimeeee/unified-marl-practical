"""Source comparisons and exact blockers, without executing legacy mains."""
import ast
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

root = Path(__file__).resolve().parents[1]
base = root.parent / "marl平台代码/王思哲/code/large_scale"
pymarl = root.parent / "pymarl-master/pymarl-master/src"
same, different = [], []
for p in sorted(pymarl.rglob("*.py")):
    relative = p.relative_to(pymarl)
    q = base / "RGHMF" / relative
    if q.exists():
        (same if p.read_bytes() == q.read_bytes() else different).append(relative.as_posix())
native = []
for p in sorted((base / "hmf_replenishment/algorithms").glob("*.py")):
    tree = ast.parse(p.read_text(encoding="utf-8"))
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    if not classes:
        continue
    native.append({"file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "classes": [{"name": c.name, "methods": [n.name for n in c.body if isinstance(n, ast.FunctionDef)]} for c in classes]})
a = base / "hmf_replenishment/algorithms/hierarchical_vdn.py"
b = base / "hmf_replenishment/algorithms/hierarchical_vdn copy.py"
def methods(p):
    return {n.name: ast.dump(n, include_attributes=False) for c in ast.parse(p.read_text(encoding="utf-8")).body
            if isinstance(c, ast.ClassDef) for n in c.body if isinstance(n, ast.FunctionDef)}
am, bm = methods(a), methods(b)
selector = {}
saved = {k: v for k, v in sys.modules.items() if k.split(".")[0] == "components"}
for key in saved:
    del sys.modules[key]
sys.path.insert(0, str(base / "RGHMF"))
try:
    source = importlib.import_module("components.action_selectors")
    for name in ["epsilon_greedy", "multinomial"]:
        try:
            source.REGISTRY[name](SimpleNamespace(epsilon_start=1.0, epsilon_finish=0.05, epsilon_anneal_time=100))
            selector[name] = {"status": "PASS"}
        except Exception as exc:
            selector[name] = {"status": "SOURCE_BLOCKED", "type": type(exc).__name__, "message": str(exc)}
finally:
    sys.path.pop(0)
    for key in list(sys.modules):
        if key.split(".")[0] == "components":
            del sys.modules[key]
    sys.modules.update(saved)
result = {"rghmf_vs_pymarl": {"byte_identical": same, "different": different},
    "hmf_algorithm_files": native,
    "copy_vs_vdn": {"byte_identical": a.read_bytes() == b.read_bytes(),
                    "equal_methods": sorted(k for k in am.keys() & bm.keys() if am[k] == bm[k]),
                    "different_methods": sorted(k for k in am.keys() & bm.keys() if am[k] != bm[k]),
                    "only_main": sorted(am.keys() - bm.keys()), "only_copy": sorted(bm.keys() - am.keys())},
    "source_basic_selectors": selector}
(root / "docs/evidence/stage3/source_comparisons.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"same": len(same), "different": len(different), "hmf_files": len(native), "copy": result["copy_vs_vdn"],
                  "source_basic_selectors": selector}, ensure_ascii=False))
