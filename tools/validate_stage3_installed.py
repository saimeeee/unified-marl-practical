"""Run from outside project after installing the wheel in .venv-check."""
from dataclasses import asdict
from pathlib import Path
import json
import subprocess
import sys
import traceback
import uuid
import unified_marl
from unified_marl import train, evaluate, load_policy
from unified_marl.algorithms.hmf.adapter import IMPLEMENTATIONS


def main():
    project = Path(__file__).resolve().parents[1]
    evidence = project / "docs/evidence/stage3"
    assert Path.cwd().resolve() != project
    assert "site-packages" in unified_marl.__file__
    root = evidence / ("installed-runs-" + uuid.uuid4().hex[:8])
    names = ["wang/rode_hmf", "wang/rode", "wang/qmix", "wang/vdn", "wang/iql", "wang/coma", "wang/qtran"]
    names += ["wang/" + name + "_beta" for name in ["qmix", "vdn", "iql"]]
    names += ["wang_hmf/" + name for name in IMPLEMENTATIONS]
    names += ["pymarl/qmix"]
    results = []
    for name in names:
        opts = {"batch_size": 2, "buffer_size": 8, "t_max": 32, "max_updates": 2,
                "runner": "episode", "batch_size_run": 1, "rnn_hidden_dim": 16, "role_interval": 2}
        env_args = {}
        if name == "wang/coma":
            opts["batch_size"] = 1
        if name.startswith("wang_hmf/"):
            env_args = {"n_agents": 4, "mask_actions": False}
            opts.update(batch_size=4, buffer_size=32, hidden_dim=16, n_heads=2, embed_dim=8,
                        hmf_norm_rewards=False, hmf_steps_per_update=1, hmf_num_batch=1)
        result = train(name, env_args=env_args, overrides=opts, output_dir=root / name.replace("/", "-"))
        assert result.updates == 2 and result.parameter_delta_l2 > 0
        score = evaluate(result.checkpoint, episodes=2)
        policy = load_policy(result.checkpoint)
        policy.reset()
        results.append({"algorithm": name, "train": asdict(result), "evaluation": asdict(score)})
        print(name, "PASS", flush=True)
    external = []
    for env, env_args in [("sc2", {"map_name": "3m"}), ("replenishment", {})]:
        try:
            train("pymarl/qmix", env, env_args, output_dir=root / ("real-" + env), overrides={"t_max": 16})
            external.append({"env": env, "status": "RAN_REQUIRES_REVIEW"})
        except Exception as exc:
            external.append({"env": env, "status": "BLOCKED", "exception": type(exc).__name__,
                             "error": str(exc), "traceback": traceback.format_exc()})
    cli = subprocess.run([str(Path(sys.executable).with_name("unified-marl.exe")), "train", "--algorithm", "wang/rode",
        "--env", "toy", "--output-dir", str(root / "cli-rode"), "--set", "batch_size=2", "--set", "buffer_size=8",
        "--set", "runner=episode", "--set", "batch_size_run=1", "--set", "t_max=16", "--set", "max_updates=1"],
        text=True, capture_output=True)
    assert cli.returncode == 0, cli.stderr
    (evidence / "installed_validation.json").write_text(json.dumps({"version": unified_marl.__version__,
        "python": sys.executable, "cwd": str(Path.cwd()), "package": unified_marl.__file__,
        "algorithms": results, "real_environments": external,
        "cli": {"returncode": cli.returncode, "stdout": cli.stdout, "stderr": cli.stderr}}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
