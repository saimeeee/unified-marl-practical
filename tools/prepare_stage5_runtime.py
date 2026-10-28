"""Explicit runtime-only adaptations; no delivered source is edited."""
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
audit=json.loads((root/"docs/evidence/stage5/audit.json").read_text(encoding="utf-8"))
config=root/"src/unified_marl/config/sadma"
for profile in ["wang","yi_open","yi_closed"]:
    base=Path(audit["sources"][profile])/"config"
    output=config/profile
    output.mkdir(parents=True,exist_ok=True)
    for path in [base/"default.yaml",*sorted((base/"alg_config").glob("*.yaml"))]:
        (output/path.name).write_bytes(path.read_bytes())
    (output/"envs").mkdir(exist_ok=True)
    for path in sorted((base/"env_config").glob("*.yaml")): (output/"envs"/path.name).write_bytes(path.read_bytes())
native=root/"src/unified_marl/algorithms/sadma/native"
changes=[]
for relative in ["policy/qmix/q_policy.py","policy/mappo/ppo_policy.py"]:
    path=native/relative
    text=path.read_text(encoding="utf-8")
    text=text.replace("        loss.backward()", "        self._integration_loss = float(loss.detach().cpu())\n        if not torch.isfinite(loss): raise FloatingPointError('Non-finite SADMA source loss')\n        loss.backward()") if relative.startswith("policy/qmix") else text.replace("            loss.backward()", "            self._integration_loss = float(loss.detach().cpu())\n            if not torch.isfinite(loss): raise FloatingPointError('Non-finite SADMA source loss')\n            loss.backward()")
    if relative.endswith("q_policy.py"):
        text=text.replace('torch.gather(target_mac_out, 3, batch["actions"])','torch.gather(target_mac_out, 3, batch["actions"].unsqueeze(-1))')
        changes.append("q_lambda gather index receives explicit action dimension; target formula unchanged")
    else:
        text=text.replace('            if not self.args.use_vtrace:', '            else:\n                norm_old_values = old_values\n\n            if not self.args.use_vtrace:')
        text=text.replace('inputs["state"] = states[:, t]','inputs["states"] = states[:, t]')
        changes.append("PPO unnormalized values explicit fallback and state/states input typo; source GAE and entropy objective unchanged")
    path.write_text(text,encoding="utf-8")
path=native/"policy/mappo/ppo_net.py"
text=path.read_text(encoding="utf-8")
text=text.replace('available_actions = data["available_actions"].reshape(b * a, -1)\n            x[available_actions==0]', 'available_actions = data["available_actions"].reshape_as(x)\n            x[available_actions==0]')
path.write_text(text,encoding="utf-8")
changes.append("PiMlp action mask reshapes to logits [B,N,A]; PiRnn remains same flattened shape")
(root/"docs/evidence/stage5/runtime_adaptations.json").write_text(json.dumps(changes,indent=2),encoding="utf-8")
