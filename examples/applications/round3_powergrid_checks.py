"""User-run F1/F2 checks on prepared real assets; no policy training on import."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output file")
    import numpy as np
    import unified_marl
    from unified_marl.applications import load_config, validate
    from unified_marl.applications.powergrid_engine import _make_env
    if unified_marl.__version__ != "0.8.2":
        raise RuntimeError("Install the 0.8.2 F1/F2/N1 patch first")
    config = load_config(args.config)
    validate(config)
    action_file = Path(config["env_args"]["buildings_states_actions_file"])
    original = hashlib.sha256(action_file.read_bytes()).hexdigest()
    env, contract = _make_env(config, config["seed"])
    try:
        samples = []
        selected = []
        for agent in env.agents:
            building = env.buildings[agent]
            if not building.enabled_states.get("net_electricity_consumption"):
                raise AssertionError("Enable net_electricity_consumption in a COPY of the scenario JSON for every selected building")
            if float(building.pv_installed) > 0 and len(samples) < 3:
                solar = np.asarray(building.weather.data["solar_gen"], dtype=np.float64)
                profiles = [np.asarray(building.sim_results[name], dtype=np.float64)
                            for name in ["non_shiftable_load", "dhw_demand", "cooling_demand"]]
                assert all(p.shape == solar.shape and np.isfinite(p).all() for p in profiles)
                expected = np.max(profiles[0] - float(building.pv_installed) * solar + profiles[1] / .8
                                  + profiles[2] + building.dhw_storage.capacity / .8 + building.cooling_storage.capacity / 2)
                assert np.isclose(expected, building._net_elec_cons_upper_bound)
                samples.append({"agent": str(agent), "pv_installed": float(building.pv_installed),
                                "profile_length": len(solar), "expected_upper_bound": float(expected)})
            selected.append(str(agent))
        assert samples, "Include a building with positive PV capacity"
        assert contract["episode_limit_semantics"] == "control_transitions_v2"
        cases = []
        for limit in [1, 4]:
            env.episode_limit = limit
            env.reset()
            assert env.steps == 0 and env.initialization_steps == 1
            actions = (np.asarray(contract["action_low"]) + np.asarray(contract["action_high"])) / 2
            for step in range(1, limit + 1):
                reward, done, info = env.step(actions)
                obs, state = np.asarray(env.get_obs()), np.asarray(env.get_state())
                assert np.isfinite(obs).all() and np.isfinite(state).all() and np.isfinite(reward)
                assert env.steps == step and done == (step == limit)
                assert info["control_steps"] == step and info["initialization_steps"] == 1
                assert bool(info["episode_limit"]) == bool(done)
            cases.append({"episode_limit": limit, "control_steps": env.steps,
                          "initialization_steps": env.initialization_steps, "obs_shape": list(obs.shape)})
        assert hashlib.sha256(action_file.read_bytes()).hexdigest() == original
        result = {"status": "PASS", "version": unified_marl.__version__, "package_path": unified_marl.__file__,
                  "F1": {"status": "PASS", "pv_samples": samples, "n_agents": len(selected)},
                  "F2": {"status": "PASS", "cases": cases},
                  "scope": "environment/profile checks only; PPO update and training resume require separate tests"}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False))
    finally:
        env.close()


if __name__ == "__main__":
    main()
