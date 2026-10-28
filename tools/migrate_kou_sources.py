"""Initial source namespacing; files copied alone are NOT integrated algorithms."""
import hashlib
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
base = root.parent / "marl平台代码/寇谦/论文代码及技术文档/论文代码及技术文档"
records = []
for family, source in [("icq", base / "ICQ-MA/src"), ("old", base / "old_src")]:
    selected = []
    for directory in ["learners", "controllers", "modules"]:
        selected.extend((source / directory).rglob("*.py"))
    for name in ["action_selectors.py", "epsilon_schedules.py", "preference.py", "reward_model.py", "reward_model_new.py"]:
        p = source / "components" / name
        if p.exists(): selected.append(p)
    for p in (source / "utils").glob("*.py"):
        if p.name not in {"logging.py", "timehelper.py"}: selected.append(p)
    for p in selected:
        target = root / "src/unified_marl/algorithms/kou" / family / p.relative_to(source)
        if target.exists(): raise FileExistsError(target)
        text = p.read_text(encoding="utf-8")
        text = text.replace("components.MAPT.mat.algorithms.reward_model.models", "unified_marl.algorithms.kou.reward_models")
        text = text.replace("from components.episode_buffer import", "from unified_marl.components.episode_buffer import")
        text = re.sub(r"(?m)^from (components|controllers|learners|modules|utils)([.\w]*) import ",
            rf"from unified_marl.algorithms.kou.{family}.\1\2 import ", text)
        text = re.sub(r"(?m)^import wandb\s*$", "# Unused remote logging import removed; metrics use the package logger.", text)
        # Source import is unused except in the legacy reward wrapper, which is
        # retained for provenance. Its actual models are imported lazily below.
        if p.name.startswith("reward_model"):
            text = text.replace(f"from unified_marl.algorithms.kou.{family}.utils.logging import Logger, get_logger",
                                "import logging\nget_logger = lambda: logging.getLogger('unified_marl.reward')")
        if family == "icq":
            for symbol in ["mac_out", "mac_out0", "mac_out1"]:
                text = text.replace(f"{symbol}/{symbol}.sum(dim=-1, keepdim=True)", f"{symbol}/{symbol}.sum(dim=-1, keepdim=True).clamp_min(1e-12)")
            text = text.replace("mac_out / mac_out.sum(dim=-1, keepdim=True)", "mac_out / mac_out.sum(dim=-1, keepdim=True).clamp_min(1e-12)")
        text = text.replace(".to('cuda')", ".to(self.args.device)") if p.name == "madpo_learner.py" else text
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        records.append({"source": p.relative_to(root.parent).as_posix(), "source_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                        "target": target.relative_to(root).as_posix()})
    dest = root / "src/unified_marl/config/kou" / family
    dest.mkdir(parents=True, exist_ok=True)
    for p in [source / "config/default.yaml", *(source / "config/algs").glob("*.yaml")]:
        (dest / p.name).write_bytes(p.read_bytes())
reward_root = base / "ICQ-MA/src/components/MAPT/mat/algorithms/reward_model/models"
for p in reward_root.glob("*.py"):
    target = root / "src/unified_marl/algorithms/kou/reward_models" / p.name
    if target.exists(): raise FileExistsError(target)
    text = p.read_text(encoding="utf-8").replace("components.MAPT.mat.algorithms.reward_model.models", "unified_marl.algorithms.kou.reward_models")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    records.append({"source": p.relative_to(root.parent).as_posix(), "source_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                    "target": target.relative_to(root).as_posix()})
package = root / "src/unified_marl/algorithms/kou"
for folder in [package, *[p for p in package.rglob("*") if p.is_dir()]]:
    init = folder / "__init__.py"
    if not init.exists(): init.write_text("", encoding="utf-8")
(root / "docs/evidence/stage4/migration_sources.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Namespaced {len(records)} files. Source-only files remain unvalidated until public API tests pass.")
