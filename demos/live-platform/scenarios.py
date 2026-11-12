"""Validated, per-run traffic scenes. Bundled training assets stay unchanged."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
TRAFFIC_DEFAULTS = dict(rows=2, cols=2, vehicle_count=400, decision_limit=32, controller="trained")
SCENARIO_SCHEMA = {"traffic": {
    "defaults": TRAFFIC_DEFAULTS,
    "fields": {
        "rows": {"type": "integer", "min": 1, "max": 4},
        "cols": {"type": "integer", "min": 1, "max": 4},
        "vehicle_count": {"type": "integer", "min": 0, "max": 3000},
        "decision_limit": {"type": "integer", "min": 8, "max": 180},
        "controller": {"enum": ["trained", "pressure"]},
    },
    "trained_grid": [2, 2],
    "vehicle_count_meaning": "Total scheduled vehicle departures; not simultaneous on-road vehicles",
}}


def validate_parameters(scenario, raw):
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("场景参数必须为对象。")
    if not raw:
        return {}
    if scenario != "traffic":
        raise ValueError("当前可自定义交通环境；其他场景请使用已训练配置。")
    if set(raw) - set(TRAFFIC_DEFAULTS):
        raise ValueError("包含不支持的交通参数。")
    result = {**TRAFFIC_DEFAULTS, **raw}
    for key in ("rows", "cols", "vehicle_count", "decision_limit"):
        bounds = SCENARIO_SCHEMA["traffic"]["fields"][key]
        if type(result[key]) is not int or not bounds["min"] <= result[key] <= bounds["max"]:
            raise ValueError(f"{key} 必须为 {bounds['min']} 至 {bounds['max']} 的整数。")
    if result["controller"] not in ("trained", "pressure"):
        raise ValueError("请选择已训练策略或压力规则控制器。")
    if result["controller"] == "trained" and (result["rows"], result["cols"]) != (2, 2):
        raise ValueError("现有 PPO 模型仅兼容 2×2 路网；其他路口数量请选择压力规则控制器，或训练匹配的新模型。")
    return result


def _write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _grid(rows, cols):
    source = HERE / "bundle/vendor/CityFlow/tools/generator/generate_json_from_grid.py"
    spec = importlib.util.spec_from_file_location("marl_local_grid_generator", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The upstream helper indexes spacing by both axes; homogeneous spacing
    # with enough entries also supports rectangular grids.
    spacing = [300] * (max(rows, cols) - 1)
    roadnet = module.gridToRoadnet(rows, cols, spacing[:], spacing[:], 300, 300,
        [[30] * cols for _ in range(rows)], laneMaxSpeed=16.67, tlPlan=True)
    for intersection in roadnet["intersections"]:
        if not intersection.get("virtual"):
            intersection["trafficLight"]["lightphases"].append({"time": 1, "availableRoadLinks": []})
    return roadnet


def straight_routes(rows, cols):
    routes = []
    for y in range(1, rows + 1):
        routes.append([f"road_{x}_{y}_0" for x in range(cols + 1)])
        routes.append([f"road_{x}_{y}_2" for x in range(cols + 1, 0, -1)])
    for x in range(1, cols + 1):
        routes.append([f"road_{x}_{y}_1" for y in range(rows + 1)])
        routes.append([f"road_{x}_{y}_3" for y in range(rows + 1, 0, -1)])
    return routes


def prepare_traffic(config, raw, folder):
    """Return application config and provenance; {} preserves the original scene."""
    parameters = validate_parameters("traffic", raw)
    if not parameters:
        return config, {"custom": False, "controller": "trained", "policy_source": "trained_checkpoint"}
    config = deepcopy(config)
    folder = Path(folder).resolve()
    scene = folder / "scenario"
    scene.mkdir(parents=True, exist_ok=True)
    source = Path(config["env_args"]["cityflow_config_file"])
    native = json.loads(source.read_text(encoding="utf-8"))
    original_root = (source.parent / native.get("dir", ".")).resolve()
    rows, cols = parameters["rows"], parameters["cols"]
    if (rows, cols) == (2, 2):
        # Keep exact roadnet bytes and therefore checkpoint compatibility.
        shutil.copyfile(original_root / native["roadnetFile"], scene / "roadnet.json")
        roadnet = json.loads((scene / "roadnet.json").read_text(encoding="utf-8"))
    else:
        roadnet = _grid(rows, cols)
        _write(scene / "roadnet.json", roadnet)
    template = json.loads((original_root / native["flowFile"]).read_text(encoding="utf-8"))[0]["vehicle"]
    routes = straight_routes(rows, cols)
    total = parameters["vehicle_count"]
    nominal_seconds = parameters["decision_limit"] * float(config["env_args"]["control_interval"])
    departure_window = max(1, int(nominal_seconds * 0.8))
    # One native flow per departure makes the requested total exact (including 0).
    # CityFlow's inclusive start=end emits one vehicle, irrespective of interval.
    flow = [{"vehicle": template, "route": routes[index % len(routes)], "interval": 1.0,
             "startTime": int((index // len(routes)) * departure_window / max(1, (total + len(routes)-1)//len(routes))),
             "endTime": int((index // len(routes)) * departure_window / max(1, (total + len(routes)-1)//len(routes)))}
            for index in range(total)]
    _write(scene / "flow.json", flow)
    native.update(dir=str(scene) + os.sep, roadnetFile="roadnet.json", flowFile="flow.json", saveReplay=False)
    _write(scene / "cityflow.json", native)
    agents = [i["id"] for i in roadnet["intersections"] if not i.get("virtual")]
    args = config["env_args"]
    args.update(cityflow_config_file=str(scene / "cityflow.json"), episode_limit=parameters["decision_limit"])
    if (rows, cols) != (2, 2):
        args["phase_ids"] = {agent: [0, 2, 4, 6] for agent in agents}
        args["yellow_phase_ids"] = {agent: 8 for agent in agents}
    provenance = {**parameters, "custom": True, "intersection_count": len(agents),
                  "planned_vehicle_count": len(flow), "departure_window_seconds": departure_window,
                  "policy_source": "pressure_rule" if parameters["controller"] == "pressure" else "trained_checkpoint",
                  "note": "Scheduled departures differ from simultaneous vehicle count; changed demand is outside the training scenario."}
    _write(folder / "scenario.json", provenance)
    return config, provenance


class PressurePolicy:
    """Queue-pressure rule respecting the same legal-action/service masks."""
    def __init__(self, env, roadnet):
        self.env = env
        self.movements = {}
        for it in roadnet["intersections"]:
            if it.get("virtual"):
                continue
            links = it["roadLinks"]
            self.movements[it["id"]] = {}
            for phase in env.phase_ids[it["id"]]:
                movements = set()
                for link_index in it["trafficLight"]["lightphases"][phase]["availableRoadLinks"]:
                    link = links[link_index]
                    for lane in link["laneLinks"]:
                        movements.add((f"{link['startRoad']}_{lane['startLaneIndex']}",
                                       f"{link['endRoad']}_{lane['endLaneIndex']}"))
                self.movements[it["id"]][phase] = movements

    def reset(self):
        pass

    def act(self, observation, avail_actions):
        waiting = self.env._engine.get_lane_waiting_vehicle_count()
        actions = []
        for agent, mask in zip(self.env.agent_ids, avail_actions):
            valid = [index for index, allowed in enumerate(mask) if allowed]
            if not valid:
                raise ValueError("No legal traffic phase available")
            def score(index):
                phase = self.env.phase_ids[agent][index]
                pressure = sum(waiting.get(start, 0) - waiting.get(end, 0)
                               for start, end in self.movements[agent][phase])
                return pressure, phase == self.env._current_phase[agent], -index
            actions.append(max(valid, key=score))
        return {"raw_actions": actions, "decisions": self.env.actions_to_phase_ids(actions)}
