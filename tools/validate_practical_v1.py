"""Short end-to-end validation on domain simulators and the real MPE backend.

No simulator is replaced with a toy or mock.  Inventory uses synthetic demand
fixtures, CityFlow uses the existing generated road network, and MPE is a
research simulation; none is evidence of real-world policy effectiveness.
Outputs always go to a new run directory.  Optional simulators must already
be installed in the selected interpreter/container; this script installs none.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
import uuid


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_SCENES = PROJECT / "demos/three-algorithm-videos/参数模板"
ALGORITHMS = ("pymarl/qmix", "pymarl/vdn", "sadma/ppo", "example/huber_iql")
ENVIRONMENTS = ("inventory", "traffic_engine", "mpe")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")


def runtime_evidence():
    import unified_marl
    package_path = Path(unified_marl.__file__).resolve()
    expected_source = (PROJECT / "src/unified_marl").resolve()
    if package_path.parent != expected_source:
        raise RuntimeError(
            f"Validation must use this checkout's src, not an older installed/demo copy: "
            f"loaded {package_path}; expected {expected_source}. Set PYTHONPATH to {PROJECT / 'src'}."
        )
    versions = {}
    for name in ("unified-marl", "torch", "numpy", "PyYAML", "cityflow", "mpe2",
                 "pettingzoo", "gymnasium"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    modules = {}
    for name in ("cityflow", "mpe2", "pettingzoo", "gymnasium"):
        spec = importlib.util.find_spec(name)
        modules[name] = None if spec is None else spec.origin
    return dict(python=sys.executable, python_version=sys.version,
                module_version=unified_marl.__version__,
                platform=platform.platform(), package_file=str(package_path),
                source_sha256={str(relative): sha256(expected_source / relative)
                               for relative in (Path("api.py"), Path("registry.py"),
                                                Path("config/__init__.py"))},
                versions=versions, optional_module_paths=modules,
                command=[sys.executable, *sys.argv], cwd=str(Path.cwd()),
                script_sha256=sha256(__file__), device="cpu")


def create_run_directory(output_root):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = Path(output_root).resolve() / (stamp + "-" + uuid.uuid4().hex[:8])
    target.mkdir(parents=True, exist_ok=False)
    return target


def copy_asset(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Required existing scene asset missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copy2(source, destination)
    original = sha256(source)
    if sha256(destination) != original:
        raise AssertionError("Scene copy changed input bytes")
    return dict(source=str(source), copy=str(destination), sha256=original)


def prepare_environment(name, scene_root, directory, seed, traffic_horizon=12):
    """Copy fixtures byte-for-byte and return documented effective env args."""
    import yaml
    scene_root, directory = Path(scene_root), Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    evidence = dict(name=name, assets=[], synthetic_data=True,
                    real_world_effectiveness="NOT_VALIDATED")
    if name == "inventory":
        for filename in ("agents.csv", "demand.csv"):
            evidence["assets"].append(copy_asset(scene_root / "warehouse" / filename,
                                                 directory / filename))
        args = dict(agents_file=str(directory / "agents.csv"),
                    demand_file=str(directory / "demand.csv"), horizon=24,
                    backlog=False, reward_scale=1000., seed=seed)
        evidence.update(evidence_level="PARAMETERIZED_DOMAIN_SIMULATOR",
                        description="InventoryEnv stock/order/lead-time simulation; existing synthetic 2-warehouse x 2-SKU demand fixture, not original ReplenishmentEnv")
    elif name == "traffic_engine":
        template = scene_root / "traffic.yaml"
        config = yaml.safe_load(template.read_text(encoding="utf-8"))
        args = dict(config["env_args"])
        native_source = scene_root / args["cityflow_config_file"]
        native_config = json.loads(native_source.read_text(encoding="utf-8"))
        # The checked-in demonstration deliberately uses self-contained relative paths.
        if native_config.get("dir") != "." or any(
                Path(native_config[key]).name != native_config[key]
                for key in ("roadnetFile", "flowFile")):
            raise ValueError("Expected existing self-contained CityFlow demo; refuse to rewrite native asset paths silently")
        for filename in (native_source.name, native_config["roadnetFile"], native_config["flowFile"]):
            evidence["assets"].append(copy_asset(native_source.parent / filename,
                                                 directory / filename))
        args.update(cityflow_config_file=str(directory / native_source.name),
                    episode_limit=traffic_horizon, thread_num=1, seed=seed)
        evidence.update(evidence_level="NATIVE_CITYFLOW_ENGINE",
                        description="Native CityFlow Engine with existing generated 2x2 roadnet/flow; source service constraints preserved",
                        template=str(template), template_sha256=sha256(template),
                        original_episode_limit=config["env_args"]["episode_limit"])
    elif name == "mpe":
        args = dict(scenario="simple_spread", n_agents=3, episode_limit=12, seed=seed)
        evidence.update(evidence_level="MIGRATED_MPE_PARTICLE_SIMULATOR",
                        description="NumPy MPE particle physics and simple_spread scenario migrated from Kou's embedded MAPT delivery; not CoordinationEnv and not an MPE2/PettingZoo installation")
    else:
        raise ValueError(f"No real-environment validation recipe for {name!r}")
    evidence["env_args"] = args
    save_json(directory / "scene-evidence.json", evidence)
    return args, evidence


def training_options(algorithm, updates):
    common = dict(t_max=100000, max_updates=updates, num_threads=1,
                  test_interval=1000000, test_nepisode=1)
    if algorithm.startswith("sadma/"):
        common.update(train_batch_size=2, train_buffer_size=8, update_rate=2,
                      hidden_dim=16, mini_epochs=2)
    else:
        common.update(batch_size=2, buffer_size=8, rnn_hidden_dim=16,
                      mixing_embed_dim=8, hypernet_embed=16)
    return common


def validate_saved_policy(checkpoint, env_name, env_args):
    import numpy as np
    from unified_marl import load_policy
    from unified_marl.checkpoints import read_checkpoint
    from unified_marl.envs import make_env

    policy = load_policy(checkpoint, device="cpu")
    payload = read_checkpoint(checkpoint)
    for key, value in policy.mac.agent.state_dict().items():
        if not value.cpu().equal(payload["agent"][key].cpu()):
            raise AssertionError(f"Reloaded parameter mismatch: {key}")
    env = make_env(env_name, **env_args)
    try:
        env.reset()
        obs, mask = np.asarray(env.get_obs()), np.asarray(env.get_avail_actions())
        policy.reset()
        actions = np.asarray(policy.act(obs, mask)).reshape(-1)
        policy.reset()
        np.testing.assert_array_equal(actions, np.asarray(policy.act(obs, mask)).reshape(-1))
        returns = 0.
        observed_steps = 0
        for _ in range(min(4, env.get_env_info()["episode_limit"])):
            if not np.issubdtype(actions.dtype, np.integer):
                raise AssertionError("Expected discrete integer actions")
            if not mask[np.arange(len(actions)), actions].all():
                raise AssertionError("Saved policy chose unavailable actions")
            reward, done, _ = env.step(actions)
            if not math.isfinite(float(reward)):
                raise AssertionError("Non-finite simulator reward")
            returns += float(reward)
            observed_steps += 1
            if done:
                break
            obs, mask = np.asarray(env.get_obs()), np.asarray(env.get_avail_actions())
            actions = np.asarray(policy.act(obs, mask)).reshape(-1)
        return dict(parameters_match_checkpoint=True, reset_action_repeatable=True,
                    available_actions_respected=True, rollout_steps=observed_steps,
                    rollout_return=returns, environment_type=f"{type(env).__module__}.{type(env).__name__}")
    finally:
        env.close()


def independent_evaluate(checkpoint, output, episodes, seed, timeout):
    """Run public evaluate in a fresh interpreter, with zero training calls."""
    before = sha256(checkpoint)
    argv = [sys.executable, "-B", str(Path(__file__).resolve()),
            "--evaluate-checkpoint", str(checkpoint), "--evaluation-output", str(output),
            "--episodes", str(episodes), "--seed", str(seed)]
    completed = subprocess.run(argv, text=True, encoding="utf-8", errors="replace",
                               capture_output=True, timeout=timeout)
    command_record = dict(argv=argv, returncode=completed.returncode,
                          stdout=completed.stdout, stderr=completed.stderr)
    save_json(Path(output).with_suffix(".command.json"), command_record)
    if completed.returncode:
        raise RuntimeError(f"Independent evaluation failed: {completed.stderr[-3000:]}")
    if sha256(checkpoint) != before:
        raise AssertionError("Evaluation mutated the policy checkpoint")
    result = json.loads(Path(output).read_text(encoding="utf-8"))
    result["command_record"] = str(Path(output).with_suffix(".command.json"))
    result["checkpoint_unchanged"] = True
    return result


def run_case(algorithm, env_name, env_args, directory, args):
    from unified_marl import train
    from unified_marl.checkpoints import read_checkpoint
    directory.mkdir(parents=True, exist_ok=False)
    options = training_options(algorithm, args.updates)
    plugin_options = ({"config": {"plugins": ["unified_marl.contrib.huber_iql"]}}
                      if algorithm == "example/huber_iql" else {})
    save_json(directory / "requested-training.json", dict(algorithm=algorithm,
              env=env_name, env_args=env_args, overrides=options, seed=args.seed,
              **plugin_options))
    trained = train(algorithm, env_name, env_args=env_args, seed=args.seed,
                    device="cpu", execution="local", output_dir=directory / "trained",
                    overrides=options, **plugin_options)
    if trained.updates != args.updates or not trained.parameter_delta_l2 > 0:
        raise AssertionError("Short training did not make the requested nonzero parameter updates")
    if not all(math.isfinite(float(value)) for value in trained.metrics.values()):
        raise AssertionError("Non-finite training metrics")
    hashes = {str(path): sha256(path) for path in
              (trained.checkpoint, trained.training_checkpoint)}
    load_check = validate_saved_policy(trained.checkpoint, env_name, env_args)
    evaluation = independent_evaluate(trained.checkpoint, directory / "evaluation.json",
                                      args.episodes, args.seed + 1, args.timeout)
    original = read_checkpoint(trained.training_checkpoint)
    resumed = train(resume=trained.training_checkpoint, device="cpu",
                    output_dir=directory / "resumed",
                    overrides={"max_updates": args.updates + args.resume_updates})
    if resumed.updates != args.updates + args.resume_updates or not resumed.parameter_delta_l2 > 0:
        raise AssertionError("Resume did not continue with nonzero parameter updates")
    if resumed.env_steps <= trained.env_steps or resumed.episodes <= trained.episodes:
        raise AssertionError("Resume did not increase cumulative environment counters")
    restored = read_checkpoint(resumed.training_checkpoint)
    if original["config"]["algorithm"] != restored["config"]["algorithm"]:
        raise AssertionError("Resume changed algorithm")
    for path, digest in hashes.items():
        if sha256(path) != digest:
            raise AssertionError("Load/evaluate/resume mutated an earlier checkpoint")
    resume_load = validate_saved_policy(resumed.checkpoint, env_name, env_args)
    resumed_evaluation = independent_evaluate(resumed.checkpoint,
        directory / "resumed-evaluation.json", args.episodes, args.seed + 1, args.timeout)
    return dict(status="PASS", algorithm=algorithm, environment=env_name,
                training=asdict(trained), resume=asdict(resumed),
                saved_policy=load_check, resumed_policy=resume_load,
                independent_evaluation=evaluation, resumed_evaluation=resumed_evaluation,
                original_checkpoint_hashes=hashes,
                resumed_checkpoint_sha256=sha256(resumed.checkpoint),
                resume_scope=restored.get("resume_scope"),
                policy_effectiveness="NOT_VALIDATED")


def evaluate_only(args):
    from unified_marl import evaluate
    if args.evaluation_output is None:
        raise ValueError("--evaluation-output is required for standalone evaluate")
    output = args.evaluation_output.resolve()
    if output.exists():
        raise FileExistsError(output)
    result = evaluate(args.evaluate_checkpoint, episodes=args.episodes,
                      device="cpu", seed=args.seed)
    if len(result.returns) != args.episodes or result.env_steps <= 0:
        raise AssertionError("Standalone evaluation did not complete requested episodes")
    if not all(math.isfinite(float(value)) for value in result.returns):
        raise AssertionError("Non-finite evaluation return")
    save_json(output, dict(result=asdict(result), python=sys.executable,
                          evaluation_only=True, training_called=False))
    print(json.dumps(asdict(result), ensure_ascii=False), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environments", nargs="+", choices=ENVIRONMENTS,
                        default=["inventory", "traffic_engine", "mpe"])
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS))
    parser.add_argument("--scene-root", type=Path, default=DEFAULT_SCENES)
    parser.add_argument("--output-root", type=Path, default=PROJECT / "results/practical-v1-env")
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--resume-updates", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--traffic-horizon", type=int, default=12)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--probe-only", action="store_true", help="Record runtime/assets without creating or running a simulator")
    parser.add_argument("--evaluate-checkpoint", type=Path)
    parser.add_argument("--evaluation-output", type=Path)
    args = parser.parse_args(argv)
    if min(args.updates, args.resume_updates, args.episodes, args.traffic_horizon, args.timeout) < 1:
        parser.error("update/episode/horizon/timeout counts must be positive")
    if not 0 <= args.seed < 2**32 - 1:
        parser.error("seed must allow an independent evaluation seed seed+1")
    if args.evaluate_checkpoint:
        return evaluate_only(args)
    output = create_run_directory(args.output_root)
    report = dict(runtime=runtime_evidence(), output_dir=str(output), environments=[], cases=[],
                  scope="Short training, parameter update, checkpoint reload/resume and independent evaluation; no convergence or field-effectiveness claim")
    save_json(output / "runtime.json", report["runtime"])
    for env_name in args.environments:
        try:
            env_args, evidence = prepare_environment(env_name, args.scene_root,
                output / "scenes" / env_name, args.seed, args.traffic_horizon)
            report["environments"].append(evidence)
        except Exception as exc:
            report["cases"].append(dict(status="BLOCKED", environment=env_name,
                                         error=str(exc), traceback=traceback.format_exc()))
            continue
        if args.probe_only:
            continue
        for algorithm in args.algorithms:
            target = output / "cases" / env_name / algorithm.replace("/", "-")
            started = time.perf_counter()
            try:
                result = run_case(algorithm, env_name, env_args, target, args)
            except (ImportError, ModuleNotFoundError) as exc:
                result = dict(status="BLOCKED", algorithm=algorithm, environment=env_name,
                              error=str(exc), traceback=traceback.format_exc())
            except Exception as exc:
                result = dict(status="FAIL", algorithm=algorithm, environment=env_name,
                              error=str(exc), traceback=traceback.format_exc())
            result["elapsed_seconds"] = time.perf_counter() - started
            target.mkdir(parents=True, exist_ok=True)
            save_json(target / "validation.json", result)
            report["cases"].append(result)
            save_json(output / "summary.json", report)
            print(f"{env_name} / {algorithm}: {result['status']} {result.get('error', '')}", flush=True)
    unchanged = all(sha256(asset["source"]) == asset["sha256"]
                    and sha256(asset["copy"]) == asset["sha256"]
                    for env in report["environments"] for asset in env["assets"])
    report["scene_assets_unchanged"] = unchanged
    report["probe_only"] = args.probe_only
    save_json(output / "summary.json", report)
    print(f"Evidence: {output / 'summary.json'}", flush=True)
    return 0 if unchanged and all(case["status"] == "PASS" for case in report["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
