"""Stage 2 one-time source migration; loss equations retain PyMARL semantics.

Refuses to overwrite an existing target. Later edits belong in src/unified_marl.
"""
import hashlib
import json
from pathlib import Path
import re

project = Path(__file__).resolve().parents[1]
source = project.parent / "pymarl-master/pymarl-master/src"
target = project / "src/unified_marl"
records = []
paths = []
for directory in ["components", "controllers", "learners", "modules"]:
    paths.extend((source / directory).rglob("*.py"))
paths += [source / "utils/rl_utils.py", source / "envs/multiagentenv.py"]
for path in paths:
    relative = path.relative_to(source)
    dest = target / relative
    if dest.exists():
        raise FileExistsError(dest)
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"(?m)^from (components|controllers|learners|modules|utils)([.\w]*) import ",
                  r"from unified_marl.\1\2 import ", text)
    if relative.as_posix().startswith("learners/") and path.name != "__init__.py":
        text = "from .base import LearnerAPI\n" + text
        text = re.sub(r"class (\w+):", r"class \1(LearnerAPI):", text)
        text = text.replace("        # Optimise\n", "        self.check_finite(loss)\n        # Optimise\n")
        text = text.replace("        # Optimise agents\n", "        self.check_finite(coma_loss)\n        # Optimise agents\n")
        marker = "\n    def _update_targets(self):"
        if path.name == "coma_learner.py":
            returned = "\n        return {\"loss\": float(coma_loss.detach()), \"coma_loss\": float(coma_loss.detach()),\n                \"critic_loss\": sum(critic_train_stats[\"critic_loss\"]) / len(critic_train_stats[\"critic_loss\"]),\n                \"grad_norm\": float(grad_norm)}\n"
            # Return belongs at end of train, before _train_critic.
            text = text.replace("\n    def _train_critic(", returned + "\n    def _train_critic(")
            text = text.replace("            loss.backward()", "            self.check_finite(loss)\n            loss.backward()")
            text = text.replace("mac_out/mac_out.sum(dim=-1, keepdim=True)", "mac_out/mac_out.sum(dim=-1, keepdim=True).clamp_min(1e-12)")
        else:
            returned = "\n        return {\"loss\": float(loss.detach()), \"grad_norm\": float(grad_norm)}\n"
            if path.name == "qtran_learner.py":
                returned = returned.replace("float(grad_norm)}", "float(grad_norm), \"td_loss\": float(td_loss.detach()),\n                \"opt_loss\": float(opt_loss.detach()), \"nopt_loss\": float(nopt_loss.detach())}")
            text = text.replace(marker, returned + marker)
    if relative.as_posix() == "controllers/basic_controller.py":
        text = text.replace("self.action_selector.epsilon/epsilon_action_num", "self.action_selector.epsilon/epsilon_action_num.clamp_min(1)")
        # Device transfer is useful to checkpoint/policy callers without CUDA-specific methods.
        text += "\n    def to(self, device):\n        self.agent.to(device)\n        if self.hidden_states is not None:\n            self.hidden_states = self.hidden_states.to(device)\n        return self\n"
    if relative.as_posix() == "components/episode_buffer.py":
        text = text.replace("v = th.tensor(v, dtype=dtype, device=self.device)",
                            "v = th.as_tensor(np.asarray(v) if isinstance(v, (list, tuple)) else v, dtype=dtype, device=self.device)")
        text = text.replace("self.device = device\n\n    def update", "self.device = device\n        return self\n\n    def update")
        old = "        self._setup_data(scheme, self.groups if groups is None else groups, self.batch_size, self.max_seq_length)"
        new = """        merged_groups = dict(self.groups)
        merged_groups.update(groups or {})
        extra = EpisodeBatch(scheme, merged_groups, self.batch_size, self.max_seq_length, device=self.device)
        self.scheme.update({k: v for k, v in extra.scheme.items() if k != 'filled'})
        self.groups = merged_groups
        self.data.transition_data.update({k: v for k, v in extra.data.transition_data.items() if k != 'filled'})
        self.data.episode_data.update(extra.data.episode_data)"""
        text = text.replace(old, new)
        anchor = "        if self.buffer_index + ep_batch.batch_size <= self.buffer_size:\n"
        text = text.replace(anchor, anchor + "            for value in self.data.transition_data.values():\n                value[self.buffer_index:self.buffer_index + ep_batch.batch_size].zero_()\n")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    records.append({"source": str(path.relative_to(project.parent)).replace("\\", "/"),
                    "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "target": dest.relative_to(project).as_posix(),
                    "initial_target_sha256": hashlib.sha256(dest.read_bytes()).hexdigest()})
for path in (source / "config").rglob("*.yaml"):
    dest = target / "config" / path.relative_to(source / "config")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(path.read_bytes())
(project / "LICENSE").write_bytes((source.parent / "LICENSE").read_bytes())
out = project / "docs/evidence/stage2"
out.mkdir(parents=True, exist_ok=True)
(out / "migration_sources.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Migrated {len(records)} Python source files and source YAML resources.")
