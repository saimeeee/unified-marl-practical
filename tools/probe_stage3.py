"""Short public-API stage-3 probes. Records actual failures without calling them passes."""
from dataclasses import asdict
from pathlib import Path
import json
import traceback
import uuid
from unified_marl import train, evaluate


def main():
    project = Path(__file__).resolve().parents[1]
    evidence = project / "docs/evidence/stage3"
    run_root = project / ".work/stage3-probes" / uuid.uuid4().hex[:8]
    algorithms = ["wang/rode_hmf", "wang/rode", "wang/qmix", "wang/vdn", "wang/iql", "wang/coma", "wang/qtran"]
    algorithms += ["wang/" + name + "_beta" for name in ["qmix", "vdn", "iql"]]
    from unified_marl.algorithms.hmf.adapter import IMPLEMENTATIONS
    algorithms += ["wang_hmf/" + name for name in IMPLEMENTATIONS]
    records = []
    for name in algorithms:
        options = {"batch_size": 2, "buffer_size": 8, "t_max": 16, "max_updates": 1,
                   "runner": "episode", "batch_size_run": 1, "rnn_hidden_dim": 16, "learner_log_interval": 1}
        if name == "wang/coma":
            options.update(batch_size=1)
        if name.startswith("wang/rode"):
            options.update(role_interval=2)
        if name.startswith("wang_hmf/"):
            options.update(batch_size=4, buffer_size=32, t_max=16, hmf_norm_rewards=False,
                           hidden_dim=16, n_heads=2, embed_dim=8, hmf_steps_per_update=1, hmf_num_batch=1)
            env_args = {"n_agents": 4, "mask_actions": False}
        else:
            env_args = None
        try:
            result = train(name, "toy", env_args=env_args, output_dir=run_root / name.replace("/", "-"), overrides=options)
            assert result.parameter_delta_l2 > 0 and result.updates > 0
            evaluation = evaluate(result.checkpoint, episodes=2)
            records.append({"algorithm": name, "status": "PASS", "train": asdict(result), "evaluation": asdict(evaluation)})
        except Exception as exc:
            records.append({"algorithm": name, "status": "FAIL", "error": str(exc), "traceback": traceback.format_exc()})
        print(name + ": " + records[-1]["status"] + (" " + records[-1].get("error", "")), flush=True)
    (evidence / "public_api_probes.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
