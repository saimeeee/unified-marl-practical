"""Optional user-run cross-process reset diagnostic. No policy training."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


PREFIX = "SEED_PROBE_RESULT="


def child(config_file, seed, steps, mode):
    import numpy as np
    import unified_marl
    from unified_marl.applications import load_config
    from unified_marl.applications import powergrid_engine as engine
    config = load_config(config_file, overrides={"seed": seed, "device": "cpu"})
    if config["application"] != "powergrid":
        raise ValueError("A prepared PowerGrid scenario is required")
    if mode == "fully-seeded":
        # Public train/evaluate seed Python/NumPy/Torch before env construction.
        engine._seed_everything(seed)
    env, contract = engine._make_env(config, seed)
    try:
        env.reset()
        trace = []
        def record():
            obs = np.asarray(env.get_obs(), dtype="<f4")
            state = np.asarray(env.get_state(), dtype="<f4")
            if not np.isfinite(obs).all() or not np.isfinite(state).all():
                raise ValueError("Non-finite observations")
            trace.append({"obs_sha256": hashlib.sha256(obs.tobytes()).hexdigest(),
                          "state_sha256": hashlib.sha256(state.tobytes()).hexdigest()})
        record()
        actions = (np.asarray(contract["action_low"]) + np.asarray(contract["action_high"])) / 2
        for _ in range(min(steps, contract["episode_limit"])):
            reward, done, info = env.step(actions)
            record()
            trace[-1].update(reward=float(reward), done=bool(done))
            if done:
                break
        return {"mode": mode, "seed": seed, "version": unified_marl.__version__,
                "package_path": unified_marl.__file__, "n_agents": contract["n_agents"], "trace": trace}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--mode", choices=["fully-seeded", "factory-only", "both"], default="fully-seeded")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 0 <= args.seed < 2**32 or args.steps < 0 or args.timeout <= 0:
        parser.error("Invalid seed, step count or timeout")
    if args.child:
        print(PREFIX + json.dumps(child(args.config, args.seed, args.steps, args.mode), allow_nan=False))
        return
    if args.output is None:
        parser.error("--output is required")
    if args.output.exists():
        raise FileExistsError("Use a new output file")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    logs = args.output.parent / (args.output.stem + "-logs")
    logs.mkdir(exist_ok=False)
    modes = ["fully-seeded", "factory-only"] if args.mode == "both" else [args.mode]
    comparisons = {}
    for mode in modes:
        records = []
        for index in range(2):
            command = [sys.executable, "-X", "utf8", "-I", str(Path(__file__).resolve()), "--child",
                       "--config", str(args.config.resolve()), "--mode", mode,
                       "--seed", str(args.seed), "--steps", str(args.steps)]
            run = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=args.timeout)
            (logs / f"{mode}-{index}.stdout.log").write_text(run.stdout, encoding="utf-8")
            (logs / f"{mode}-{index}.stderr.log").write_text(run.stderr, encoding="utf-8")
            if run.returncode:
                raise RuntimeError(f"Seed probe {mode}/{index} failed with exit {run.returncode}; see {logs}")
            records.append(json.loads(next(line[len(PREFIX):] for line in reversed(run.stdout.splitlines()) if line.startswith(PREFIX))))
        equal = records[0]["trace"] == records[1]["trace"]
        comparisons[mode] = {"equal": equal, "records": records}
    result = {"comparisons": comparisons,
              "scope": "controlled environment diagnostic, not a policy-effect or bitwise resume test",
              "factory_only_note": "Private factory invocation omits full global RNG initialization; mismatch alone does not prove public train/evaluate seed failure"}
    if "fully-seeded" in comparisons:
        result["status"] = "PASS" if comparisons["fully-seeded"]["equal"] else "NEEDS_INVESTIGATION"
    else:
        result["status"] = "OBSERVATION_ONLY"
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output)}, ensure_ascii=False))
    if result["status"] == "NEEDS_INVESTIGATION":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
