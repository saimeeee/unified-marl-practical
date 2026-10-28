"""Namespace Wang source implementations without executing original entry scripts."""
import hashlib
import json
from pathlib import Path
import re

project = Path(__file__).resolve().parents[1]
base = project.parent / "marl平台代码/王思哲/code/large_scale"
records = []
for family, directories in [("RGHMF", ["components", "controllers", "learners", "modules", "utils"]),
                            ("hmf_replenishment", ["agent", "algorithms", "utils"])]:
    namespace = "rghmf" if family == "RGHMF" else "hmf"
    package = project / "src/unified_marl/algorithms" / namespace
    for directory in directories:
        for path in (base / family / directory).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            relative = path.relative_to(base / family)
            dest = package / relative
            if dest.exists():
                raise FileExistsError(dest)
            text = path.read_text(encoding="utf-8")
            text = re.sub(r"(?m)^from (components|controllers|learners|modules|utils|agent|algorithms)([.\w]*) import ",
                          rf"from unified_marl.algorithms.{namespace}.\1\2 import ", text)
            if namespace == "rghmf" and directory == "learners" and path.name != "__init__.py":
                text = "from unified_marl.learners.base import LearnerAPI\n" + text
                text = re.sub(r"class (\w+):", r"class \1(LearnerAPI):", text)
                if path.name in {"rode_learner.py", "rode_hmf_learner.py"}:
                    loss_name = "total_loss" if "hmf" in path.name else "loss"
                    text = text.replace("        # Optimise\n", f"        self.check_finite({loss_name})\n        # Optimise\n")
                    returned = f"\n        return {{'loss': float({loss_name}.detach()), 'role_loss': float(role_loss.detach()), 'grad_norm': float(grad_norm),\n                'encoder_loss': float((pred_obs_loss + 10 * pred_r_loss).detach()) if pred_obs_loss is not None else 0.0}}\n"
                    text = text.replace("\n    def _update_targets(self):", returned + "\n    def _update_targets(self):")
                else:
                    # Shared validated baseline implementations are selected by the bundle adapter.
                    pass
            if namespace == "hmf":
                # Source hard-codes CUDA, including save(). Device selection is operational,
                # not a loss/target change. CPU is the portable construction default.
                text = text.replace('torch.device("cuda")', 'torch.device("cpu")')
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(text, encoding="utf-8")
            records.append({"family": namespace, "source": str(path.relative_to(project.parent)).replace("\\", "/"),
                            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "target": dest.relative_to(project).as_posix()})
    for parent in [package, *[p for p in package.rglob("*") if p.is_dir() and p.name != "__pycache__"]]:
        init = parent / "__init__.py"
        if not init.exists():
            init.write_text("", encoding="utf-8")
    if namespace == "rghmf":
        config_target = project / "src/unified_marl/config/wang"
        for path in (base / family / "config/algs").glob("*.yaml"):
            config_target.mkdir(parents=True, exist_ok=True)
            (config_target / path.name).write_bytes(path.read_bytes())
        (config_target / "default.yaml").write_bytes((base / family / "config/default.yaml").read_bytes())
(project / "src/unified_marl/algorithms/__init__.py").write_text("\"\"\"Source-specific algorithm families, imported only when selected.\"\"\"\n", encoding="utf-8")
out = project / "docs/evidence/stage3"
out.mkdir(parents=True, exist_ok=True)
(out / "migration_sources.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Namespaced {len(records)} files; this alone does not count as integrated/validated algorithms.")
