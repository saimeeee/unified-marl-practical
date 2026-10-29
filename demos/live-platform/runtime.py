"""Standalone live-rollout helpers, extracted without changing simulator semantics.

Derived from demos/three-algorithm-videos/{demo_common,recorders}.py and
remote-practical/{run_application,run_mpe}.py. Kept here so a local platform
installation does not depend on development-only training/video directories.
"""
from copy import deepcopy
import csv
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
import random
import shutil


def plain(value):
    import numpy as np
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)


def write_config(path, config):
    import yaml
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")


def evaluation_config(config, seed, directory):
    """Separate stochastic demand from training without modifying source files."""
    derived = deepcopy(config)
    derived["seed"] = seed
    derived["device"] = "cpu"
    if config["application"] == "warehouse":
        import numpy as np
        from unified_marl.envs.inventory import read_problem
        env_args = derived["env_args"]
        agents, original, _ = read_problem(env_args["agents_file"], env_args["demand_file"],
                                         env_args.get("horizon", 30), env_args.get("action_multipliers"))
        source_agents, source_demand = Path(env_args["agents_file"]), Path(env_args["demand_file"])
        demand = np.random.default_rng(seed).poisson(
            [agent["mean_demand"] for agent in agents], size=original.shape
        )
        if np.array_equal(demand, original):
            raise ValueError("Generated evaluation demand equals the training demand; choose another training dataset")
        shutil.copy2(source_agents, directory / "agents.csv")
        destination = directory / "demand.csv"
        with destination.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["period", "agent_id", "demand"])
            writer.writeheader()
            for period, row in enumerate(demand):
                for agent, amount in zip(agents, row):
                    writer.writerow(dict(period=period, agent_id=agent["agent_id"], demand=int(amount)))
        env_args.update(agents_file=str(directory / "agents.csv"), demand_file=str(destination))
        save_json(directory / "demand-generation.json", dict(
            seed=seed, distribution="Independent Poisson(mean_demand) per agent per period; synthetic evaluation demand",
            source_agents=str(source_agents), source_agents_sha256=sha256(source_agents),
            source_demand=str(source_demand), source_demand_sha256=sha256(source_demand),
            generated_demand_sha256=sha256(destination), differs_from_training=True,
            note="New synthetic demand, not a held-out observed business dataset or an inferred demand model.",
        ))
    elif config["application"] == "powergrid":
        # The engine merges checkpoint config; override its nested training seed too.
        # Global RNGs are also explicitly seeded before every evaluate/record call.
        derived["env_args"]["randomseed"] = seed
    write_config(directory / "config.yaml", derived)
    return derived


def _parse_native_replay(line):
    sections = line.rstrip('\n').split(';')
    vehicles=[]
    for entry in sections[0].split(','):
        if not entry.strip():
            continue
        tokens=entry.split()
        if len(tokens)!=7:
            raise ValueError('Unexpected native CityFlow replay vehicle format')
        x,y,angle,uid,change,length,width=tokens
        vehicles.append(dict(x=float(x),y=float(y),angle=float(angle),id=uid,length=float(length),width=float(width)))
    lights={}
    if len(sections)>1:
        for entry in sections[1].split(','):
            tokens=entry.split()
            if tokens:
                lights[tokens[0]]=tokens[1:]
    return vehicles,lights


def world_snapshot(env, step, reward, cumulative_reward, actions, radius, collision_sum):
    import numpy as np
    positions = np.asarray([agent.state.p_pos.copy() for agent in env.world.agents])
    landmarks = np.asarray([landmark.state.p_pos.copy() for landmark in env.world.landmarks])
    distances = np.linalg.norm(landmarks[:, None, :] - positions[None, :, :], axis=2)
    nearest = distances.min(axis=1)
    collisions = [[i, j] for i, left in enumerate(env.world.agents)
                  for j, right in enumerate(env.world.agents) if j > i and
                  np.linalg.norm(left.state.p_pos - right.state.p_pos) < left.size + right.size]
    if step and len(collisions) != int(env.get_stats()["collisions"]):
        raise AssertionError("Recorded pair collisions disagree with simulator statistics")
    if not np.isfinite(positions).all() or not np.isfinite(nearest).all():
        raise FloatingPointError("Non-finite physical coordinates")
    return dict(step=step, physical_time=float(step * env.world.dt),
                reward=float(reward), cumulative_reward=float(cumulative_reward),
                actions=None if actions is None else [int(value) for value in actions],
                agents=[dict(id=i, name=agent.name, position=positions[i].tolist(),
                             velocity=np.asarray(agent.state.p_vel).tolist(), radius=float(agent.size))
                        for i, agent in enumerate(env.world.agents)],
                landmarks=[dict(id=i, position=landmarks[i].tolist(), radius=float(landmark.size))
                           for i, landmark in enumerate(env.world.landmarks)],
                nearest_landmark_distances=nearest.tolist(),
                mean_nearest_landmark_distance=float(nearest.mean()),
                covered_landmarks=int(np.count_nonzero(nearest < radius)),
                coverage_fraction=float(np.count_nonzero(nearest < radius) / len(landmarks)),
                collision_pairs=collisions, collision_pair_steps=collision_sum)
