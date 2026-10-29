"""One fresh process per live rollout; every frame follows real simulator state."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import traceback

from runtime import evaluation_config, plain, save_json, seed_all, _parse_native_replay, world_snapshot


class Stop(Exception):
    pass


def metric(label, value, unit=""):
    return dict(label=label, value=float(value), unit=unit)


def warehouse(request, emit, folder):
    from unified_marl.applications import warehouse as api, load_config
    from unified_marl.envs.inventory import InventoryEnv
    config = evaluation_config(load_config(request["model"]["config"]), request["seed"], folder)
    seed_all(request["seed"])
    env = InventoryEnv(**dict(config["env_args"], seed=request["seed"]))
    try:
        env.reset()
        policy = api.load_policy(request["model"]["checkpoint"], config=config, device="cpu")
        policy.reset()
        for step in range(env.episode_limit + 1):
            state = dict(period=step, agents=[dict(agent_id=row["agent_id"], stock=float(env.stock[i]),
                backlog=float(env.backlog[i]), last_demand=float(env.last_demand[i]),
                pipeline=[dict(arrives_in=due-step, quantity=float(q)) for due,q in env.pipeline[i]]) for i,row in enumerate(env.rows)])
            if step == 0:
                last, total, reward = [], 0., 0.
            stats = env.get_stats()
            emit(dict(scene="warehouse", step=step, limit=env.episode_limit, done=step == env.episode_limit,
                clock=f"第 {step} / {env.episode_limit} 个补货周期", view=dict(agents=[dict(**row,
                    capacity=float(env.capacity[i]), in_transit=sum(q for _,q in env.pipeline[i]),
                    **({k:v for k,v in last[i].items() if k not in row} if last else {})) for i,row in enumerate(state["agents"])]),
                metrics=[metric("累计利润", stats.get("profit", stats.get("total_profit", 0))),
                         metric("需求满足率", 100*stats.get("sales", 0)/max(1, stats.get("sales", 0)+stats.get("unserved_units", 0)), "%"),
                         metric("累计奖励", total), metric("当前库存", sum(env.stock), "件")],
                chart=dict(label="总库存", value=float(sum(env.stock))),
                actions=[dict(agent=d.get("agent_id", i), action=f"补货 {d['executed_order']:.0f} 件") for i,d in enumerate(last)]))
            if step == env.episode_limit:
                break
            decision = policy.act(state, env.get_avail_actions())
            reward, done, info = env.step(decision["raw_actions"])
            total += float(reward)
            last = info["decisions"]
    finally:
        env.close()


def mpe(request, emit, folder):
    import numpy as np
    from unified_marl import load_policy
    from unified_marl.envs import make_env
    from unified_marl.checkpoints import read_checkpoint
    seed_all(request["seed"])
    payload = read_checkpoint(request["model"]["checkpoint"], "cpu")
    env = make_env("mpe", **dict(payload["config"]["env_args"], seed=request["seed"]))
    try:
        env.reset()
        policy = load_policy(request["model"]["checkpoint"], device="cpu")
        policy.reset()
        reward, total, collisions, actions = 0., 0., 0, None
        for step in range(env.episode_limit + 1):
            row = world_snapshot(env, step, reward, total, actions, .1, collisions)
            emit(dict(scene="mpe", step=step, limit=env.episode_limit, done=step == env.episode_limit,
                clock=f"物理时间 {row['physical_time']:.1f} s", view=row,
                metrics=[metric("累计奖励", total), metric("地标距离", row["mean_nearest_landmark_distance"]),
                         metric("覆盖地标", row["covered_landmarks"], "/ 3"), metric("累计碰撞", collisions)],
                chart=dict(label="最近地标平均距离", value=row["mean_nearest_landmark_distance"]),
                actions=[] if actions is None else [dict(agent=f"智能体 {i+1}", action=["不施力", "向右", "向左", "向上", "向下"][int(a)]) for i,a in enumerate(actions)]))
            if step == env.episode_limit:
                break
            actions = np.asarray(policy.act(env.get_obs(), env.get_avail_actions())).reshape(-1)
            reward, done, info = env.step(actions)
            total += float(reward)
            collisions += int(info["collisions"])
    finally:
        env.close()


def powergrid(request, emit, folder):
    import numpy as np
    from unified_marl.applications import powergrid as api, powergrid_engine, load_config
    config = evaluation_config(load_config(request["model"]["config"]), request["seed"], folder)
    seed_all(request["seed"])
    env, contract = powergrid_engine._make_env(config, request["seed"])
    try:
        env.reset()
        policy = api.load_policy(request["model"]["checkpoint"], config=config, device="cpu")
        policy.reset()
        buses = list(env.net.bus.index)
        topology = dict(buses=[dict(id=int(i), x=float(env.net.bus_geodata.loc[i,"x"]), y=float(env.net.bus_geodata.loc[i,"y"])) for i in buses],
                        lines=[dict(id=int(i), source=int(r.from_bus), target=int(r.to_bus)) for i,r in env.net.line.iterrows()])
        total, actions = 0., None
        for step in range(env.episode_limit + 1):
            voltage = env.net.res_bus.loc[buses,"vm_pu"].to_numpy(copy=True)
            storage = [dict(agent=str(uid), bus=int(env.buildings[uid].bus), soc=[float(getattr(env.buildings[uid],name)._soc / getattr(env.buildings[uid],name).capacity) for name in ["cooling_storage","dhw_storage","electrical_storage"]]) for uid in env.agents]
            loss = float(env.net.res_line.pl_mw.sum())
            emit(dict(scene="powergrid", step=step, limit=env.episode_limit, done=step == env.episode_limit,
                clock=f"控制时间 {step/config['env_args']['hourly_timesteps']:.2f} h · 初始化预热另计",
                view=dict(**topology, voltage=voltage, storage=storage,
                          line_power=env.net.res_line.p_from_mw.to_numpy(copy=True)),
                metrics=[metric("最低电压", min(voltage), "p.u."), metric("最高电压", max(voltage), "p.u."),
                         metric("当前网损", loss*1000, "kW"), metric("越限节点", np.count_nonzero((voltage < .95)|(voltage > 1.05)))],
                chart=dict(label="当前网损 / kW", value=loss*1000), cumulative_reward=total,
                actions=[] if actions is None else [dict(agent=str(uid), action="冷/热/电: "+" / ".join(f"{float(a):+.2f}" for a in actions[i])) for i,uid in enumerate(env.agents)]))
            if step == env.episode_limit:
                break
            decision = policy.act(env.get_obs())
            actions = np.asarray(decision["raw_actions"], dtype=np.float32)
            reward, done, info = env.step(actions)
            total += float(reward)
    finally:
        env.close()


def traffic(request, emit, folder):
    import os
    from unified_marl.applications import traffic as api, load_config
    from unified_marl.envs.traffic_engine import CityFlowEngineEnv
    config = evaluation_config(load_config(request["model"]["config"]), request["seed"], folder)
    seed_all(request["seed"])
    source = Path(config["env_args"]["cityflow_config_file"])
    native = json.loads(source.read_text(encoding="utf-8"))
    root = (source.parent / native.get("dir", ".")).resolve()
    native.update(dir=str(root)+os.sep, saveReplay=True,
                  roadnetLogFile=os.path.relpath(folder / "native-roadnet.json", root),
                  replayLogFile=os.path.relpath(folder / "native-replay.txt", root))
    save_json(folder / "live-cityflow.json", native)
    roadnet = json.loads((root / native["roadnetFile"]).read_text())
    env = CityFlowEngineEnv(**dict(config["env_args"], cityflow_config_file=str(folder / "live-cityflow.json"), seed=request["seed"]))
    replay = None
    try:
        policy = api.load_policy(request["model"]["checkpoint"], config=config, device="cpu")
        policy.reset()
        replay = (folder / "native-replay.txt").open(encoding="utf-8")
        original = env._advance
        decisions, total, vehicles, lights = 0, 0., [], {}
        actual_actions = []

        def snapshot(done=False):
            queue = sum(env._queue_by_lane.values())
            emit(dict(scene="traffic", step=int(env.get_current_time()), limit=0, done=done,
                decision_step=decisions, decision_limit=env.episode_limit,
                clock=f"仿真 {env.get_current_time():.0f} s · 决策 {decisions} / {env.episode_limit}",
                view=dict(roads=roadnet["roads"], intersections=roadnet["intersections"], vehicles=vehicles,
                          lights=lights, phases=dict(env._current_phase)),
                metrics=[metric("排队车辆", queue, "辆"), metric("在网车辆", env._engine.get_vehicle_count(), "辆"),
                         metric("平均行程时间", env._engine.get_average_travel_time(), "s"), metric("已完成决策累计奖励", total)],
                chart=dict(label="排队车辆", value=queue), actions=actual_actions))

        def advance(count):
            nonlocal vehicles, lights
            for _ in range(count):
                original(1)
                line = replay.readline()
                if not line.endswith("\n"):
                    raise RuntimeError("CityFlow did not flush a complete native replay record")
                vehicles, lights = _parse_native_replay(line)
                snapshot()

        env._advance = advance
        snapshot()
        for step in range(env.episode_limit):
            decision = policy.act(env.get_obs(), env.get_avail_actions())
            decisions = step + 1
            actual_actions = [dict(agent=d["intersection_id"], action=f"选择相位 {d['phase_id']}") for d in decision["decisions"]]
            reward, done, info = env.step(decision["raw_actions"])
            total += float(reward)
            if done:
                break
        snapshot(done=True)
    finally:
        if replay:
            replay.close()
        env.close()


ADAPTERS = dict(warehouse=warehouse, mpe=mpe, traffic=traffic, powergrid=powergrid)


def main():
    request_path = Path(sys.argv[1]).resolve()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    # Preserve one dedicated protocol FD; imports and native-library diagnostics
    # must never corrupt the JSON line stream, including C/C++ stdout writes.
    import os
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    import torch
    torch.set_num_threads(1)
    digest = hashlib.sha256(Path(request["model"]["checkpoint"]).read_bytes()).hexdigest()

    def emit(frame):
        frame["checkpoint_sha256"] = digest
        protocol.write(json.dumps(plain(frame), ensure_ascii=False, allow_nan=False) + "\n")
        protocol.flush()
        if frame.get("done"):
            return
        if sys.stdin.readline().strip() != "advance":
            raise Stop()

    try:
        ADAPTERS[request["model"]["scenario"]](request, emit, request_path.parent)
    except Stop:
        pass
    except Exception as exc:
        traceback.print_exc()
        protocol.write(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False) + "\n")
        protocol.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()
