"""Operational adaptation only: CPU allocation, package imports, detached metrics."""
from pathlib import Path
import re
root = Path(__file__).resolve().parents[1] / "src/unified_marl/algorithms/kou"
count = 0
for family in ["icq", "old"]:
    for path in (root / family).rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"(?m)^(\s*)from (components|controllers|learners|modules|utils)([.\w]*) import ",
            rf"\1from unified_marl.algorithms.kou.{family}.\2\3 import ", text)
        if path.parent.name == "learners" or (family=="old" and path.name.startswith("reward_model")):
            if "# integration: detached loss capture" not in text:
                def instrument(match):
                    indent, loss, arguments = match.groups()
                    tensor_name="torch" if path.name.startswith("reward_model") else "th"
                    return (f"{indent}# integration: detached loss capture, no change to the optimized expression\n"
                        f"{indent}if not {tensor_name}.isfinite({loss}).all(): raise FloatingPointError('Non-finite source loss: {loss}')\n"
                        f"{indent}if not hasattr(self, '_integration_losses'): self._integration_losses = {{}}\n"
                        f"{indent}self._integration_losses['{loss}'] = float({loss}.detach())\n"
                        f"{indent}{loss}.backward({arguments})")
                text = re.sub(r"(?m)^([ \t]+)(\w+)\.backward\(([^\n]*)\)", instrument, text)
            if path.name in {"dppo_learner.py", "madpo_learner.py"}:
                text = text.replace(".to('cuda')", ".to(self.args.device)")
            text = text.replace('th.zeros(cur_max_actions.squeeze(3).shape + (self.n_actions,)).cuda()',
                                'th.zeros(cur_max_actions.squeeze(3).shape + (self.n_actions,)).to(self.args.device)')
            if family == "icq" and path.name == "q_learner.py":
                text = text.replace('batch["actions"].view(-1)', 'batch["actions"].reshape(-1)')
        if path.read_text(encoding="utf-8") != text:
            path.write_text(text,encoding="utf-8")
            count += 1
print(f"Adapted {count} package files; reference source was not modified.")
