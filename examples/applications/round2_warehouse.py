"""User-run short regression; nothing runs on import. Requires the hmf extra."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--previous-run", type=Path, help="Optional existing 0.8.0 application output directory")
    args = parser.parse_args()
    root = args.work_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    import unified_marl
    from unified_marl.applications import export_template, load_config, warehouse
    import numpy as np

    if unified_marl.__version__ not in {"0.8.1", "0.8.2"}:
        raise RuntimeError("Install the 0.8.1/0.8.2 hotfix in this interpreter before running this test")
    export_template("warehouse", root / "scene")
    config = load_config(root / "scene/warehouse.yaml", overrides={
        "training.max_updates": 1, "training.t_max": 32,
        "training.batch_size": 2, "training.buffer_size": 32,
    })
    result = warehouse.train(config, output_dir=root / "first")

    def check_update(item, prior_updates, prior_steps):
        if item.updates <= prior_updates or item.env_steps <= prior_steps:
            raise AssertionError("Expected another effective update and more environment transitions")
        if not math.isfinite(item.raw["parameter_delta_l2"]) or item.raw["parameter_delta_l2"] <= 0:
            raise AssertionError("Policy parameters did not change by a finite positive amount")
        if not all(math.isfinite(float(v)) for v in item.metrics.values()):
            raise AssertionError("Non-finite training metrics")
        for name in [item.checkpoint, item.training_checkpoint,
                     str(Path(item.output_dir) / "application.json")]:
            if not Path(name).is_file():
                raise AssertionError("Missing output: " + name)

    check_update(result, 0, 0)
    resumed = warehouse.train(config, output_dir=root / "resume-api",
                              resume=result.training_checkpoint, overrides={"training.max_updates": 2})
    check_update(resumed, result.updates, result.env_steps)
    if resumed.updates != 2:
        raise AssertionError("Expected target total updates=2")
    cli = subprocess.run([
        sys.executable, "-m", "unified_marl.applications.cli", "train",
        "--config", str(root / "scene/warehouse.yaml"),
        "--output-dir", str(root / "resume-cli"), "--resume", result.training_checkpoint,
        "--set", "training.t_max=32", "--set", "training.batch_size=2",
        "--set", "training.buffer_size=32", "--set", "training.max_updates=2",
    ], capture_output=True, text=True, encoding="utf-8", timeout=180)
    (root / "cli.stdout.log").write_text(cli.stdout, encoding="utf-8")
    (root / "cli.stderr.log").write_text(cli.stderr, encoding="utf-8")
    if cli.returncode:
        raise RuntimeError(f"CLI resume exit {cli.returncode}: {cli.stderr}")
    cli_result = json.loads((root / "resume-cli/application-result.json").read_text(encoding="utf-8"))
    if cli_result["updates"] != 2 or cli_result["env_steps"] <= result.env_steps:
        raise AssertionError("CLI did not advance the stored progress")
    if cli_result["raw"]["parameter_delta_l2"] <= 0:
        raise AssertionError("CLI resume did not change policy parameters")

    evaluation = warehouse.evaluate(resumed.checkpoint, episodes=1)
    if evaluation["episode_lengths"] != [4] or evaluation["env_steps"] != 4:
        raise AssertionError("Evaluation should record four transitions, excluding the bootstrap state")
    state = json.loads((root / "scene/warehouse_state.json").read_text(encoding="utf-8"))
    decisions = []
    for _ in range(2):
        policy = warehouse.load_policy(resumed.checkpoint)
        policy.reset()
        decisions.append(policy.act(state))
    if json.dumps(decisions[0], sort_keys=True) != json.dumps(decisions[1], sort_keys=True):
        raise AssertionError("Reloaded reset policies differ on the same deterministic input")
    agents = {row["agent_id"]: row for row in state["agents"]}
    from unified_marl.applications import validate
    metadata = validate(config)["metadata"]
    capacities = {row["agent_id"]: row["capacity"] for row in metadata["agents"]}
    for decision in decisions[0]["decisions"]:
        row = agents[decision["agent_id"]]
        free = max(0, capacities[decision["agent_id"]] - row["stock"] - sum(x["quantity"] for x in row["pipeline"]))
        assert 0 <= decision["action_index"] < len(metadata["action_multipliers"])
        assert np.isclose(decision["order_quantity"], min(decision["requested_order"], free))

    failures = []
    for label, checkpoint, override in [
        ("policy-only", result.checkpoint, {"training.max_updates": 2}),
        ("changed-scene", result.training_checkpoint, {"training.max_updates": 2, "env_args.backlog": True}),
        ("changed-seed", result.training_checkpoint, {"training.max_updates": 2, "seed": 2}),
    ]:
        try:
            warehouse.train(config, output_dir=root / ("reject-" + label), resume=checkpoint, overrides=override)
        except ValueError as exc:
            failures.append({"case": label, "status": "rejected", "message": str(exc)})
        else:
            raise AssertionError("Unsafe resume was not rejected: " + label)

    old_result = None
    if args.previous_run:
        previous = args.previous_run.expanduser().resolve()
        old_scene = json.loads((previous / "application.json").read_text(encoding="utf-8"))["config"]
        stored = json.loads((previous / "application-result.json").read_text(encoding="utf-8"))
        resumed_old = warehouse.train(old_scene, output_dir=root / "resume-old-080",
            resume=previous / "training.pt", overrides={"training.max_updates": stored["updates"] + 1,
                                                         "training.t_max": stored["env_steps"] + 32})
        check_update(resumed_old, stored["updates"], stored["env_steps"])
        old_result = asdict(resumed_old)

    report = {"status": "PASS", "version": unified_marl.__version__, "python": sys.version,
              "package_path": unified_marl.__file__, "first": asdict(result),
              "resume_api": asdict(resumed), "resume_cli": cli_result,
              "evaluation": evaluation, "decision": decisions[0], "negative_cases": failures,
              "previous_080_run": old_result,
              "scope": "synthetic inventory functional regression; no policy effectiveness claim"}
    (root / "round2-results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": "PASS", "report": str(root / "round2-results.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
