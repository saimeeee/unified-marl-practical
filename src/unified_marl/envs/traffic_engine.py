"""Direct optional CityFlow.Engine adapter for discrete multi-intersection control.

read_traffic_spec validates local scenario metadata without importing CityFlow,
so applications can validate/checkpoint the intersection-to-phase action map.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import json
import math
import os
import tempfile

import numpy as np

from .multiagentenv import MultiAgentEnv


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _finite_number(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    if positive and number <= 0:
        raise ValueError(f"{name} must be greater than zero")
    if nonnegative and number < 0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _load_cityflow_files(cityflow_config_file):
    config_path = Path(cityflow_config_file).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"CityFlow config file does not exist: {config_path}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read native CityFlow JSON config: {config_path}") from exc
    if not isinstance(config, dict):
        raise ValueError("CityFlow config must contain a JSON object")

    raw_dir = config.get("dir", ".")
    if not isinstance(raw_dir, str) or not raw_dir.strip():
        raise ValueError("CityFlow config field 'dir' must be a non-empty path")
    root = Path(raw_dir).expanduser()
    if not root.is_absolute():
        root = config_path.parent / root
    root = root.resolve()

    resolved_files = {}
    for field in ("roadnetFile", "flowFile"):
        raw_path = config.get(field)
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"CityFlow config must set a non-empty {field}")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"CityFlow config field {field!r} does not resolve to a local file: {path}"
            )
        resolved_files[field] = path

    interval = _finite_number(config.get("interval"), "CityFlow config interval", positive=True)
    try:
        roadnet = json.loads(resolved_files["roadnetFile"].read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not read CityFlow roadnet JSON: {resolved_files['roadnetFile']}"
        ) from exc
    if not isinstance(roadnet, dict):
        raise ValueError("CityFlow roadnet must contain a JSON object")
    return config_path, config, root, resolved_files, interval, roadnet


def _polyline_length(points, road_id):
    if not isinstance(points, list) or len(points) < 2:
        raise ValueError(
            f"Road {road_id!r} needs at least two geometry points or explicit lane lengths"
        )
    coordinates = []
    for point in points:
        if not isinstance(point, Mapping) or "x" not in point or "y" not in point:
            raise ValueError(f"Road {road_id!r} has an invalid geometry point")
        coordinates.append((
            _finite_number(point["x"], f"Road {road_id!r} point x"),
            _finite_number(point["y"], f"Road {road_id!r} point y"),
        ))
    length = sum(
        math.hypot(right[0] - left[0], right[1] - left[1])
        for left, right in zip(coordinates, coordinates[1:])
    )
    if not math.isfinite(length) or length <= 0:
        raise ValueError(f"Road {road_id!r} must have positive geometric length")
    return length


def _phase_road_links(intersection, intersection_id):
    road_links = intersection.get("roadLinks")
    traffic_light = intersection.get("trafficLight")
    phases = traffic_light.get("lightphases") if isinstance(traffic_light, Mapping) else None
    if not isinstance(road_links, list) or not isinstance(phases, list) or not phases:
        raise ValueError(
            f"Non-virtual intersection {intersection_id!r} must define roadLinks "
            "and trafficLight.lightphases"
        )
    phase_links = []
    for phase_index, phase in enumerate(phases):
        if not isinstance(phase, Mapping):
            raise ValueError(
                f"Intersection {intersection_id!r} phase {phase_index} must be an object"
            )
        available = phase.get("availableRoadLinks")
        if not isinstance(available, list):
            raise ValueError(
                f"Intersection {intersection_id!r} phase {phase_index} must define "
                "availableRoadLinks as a list"
            )
        for link_index in available:
            if type(link_index) is not int or not 0 <= link_index < len(road_links):
                raise ValueError(
                    f"Intersection {intersection_id!r} phase {phase_index} references "
                    f"invalid road link {link_index!r}"
                )
        phase_links.append(tuple(available))
    return phase_links


def _validate_id_mapping(mapping, agent_ids, name):
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{name} must map every non-virtual intersection ID to a value")
    expected, supplied = set(agent_ids), set(mapping)
    if supplied != expected:
        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)
        raise ValueError(f"{name} keys must match agent_ids; missing={missing}, extra={extra}")


def read_traffic_spec(
    cityflow_config_file,
    phase_ids,
    yellow_phase_ids=None,
    *,
    control_interval=10.0,
    yellow_time=0.0,
    episode_limit=360,
    seed=1,
    thread_num=1,
    queue_reward_scale=1.0,
    max_green_time=None,
    max_red_time=None,
):
    """Parse local CityFlow config/roadnet files without importing the simulator.

    phase_ids maps each non-virtual intersection ID to selectable green phase
    indices from its native trafficLight.lightphases array. Positive yellow_time
    requires an explicit all-red clearance phase for every controlled intersection.
    Optional maximum green/red times restrict the available action mask using
    actual inbound lane service and simulator time. They are environment rules,
    not decisions learned by the policy. With both unset the old contract is kept.
    """
    config_path, config, root, files, interval, roadnet = _load_cityflow_files(
        cityflow_config_file
    )
    if config.get("rlTrafficLight", True) is not True:
        raise ValueError(
            "CityFlow config must set rlTrafficLight=true for external signal control"
        )
    control_interval = _finite_number(
        control_interval, "control_interval", positive=True
    )
    yellow_time = _finite_number(yellow_time, "yellow_time", nonnegative=True)
    episode_limit = _positive_int(episode_limit, "episode_limit")
    seed = _nonnegative_int(seed, "seed")
    thread_num = _positive_int(thread_num, "thread_num")
    queue_reward_scale = _finite_number(
        queue_reward_scale, "queue_reward_scale", positive=True
    )
    service_limits_enabled = max_green_time is not None or max_red_time is not None
    if max_green_time is not None:
        max_green_time = _finite_number(max_green_time, "max_green_time", positive=True)
    if max_red_time is not None:
        max_red_time = _finite_number(max_red_time, "max_red_time", positive=True)

    intersections = roadnet.get("intersections")
    roads = roadnet.get("roads")
    if not isinstance(intersections, list) or not isinstance(roads, list):
        raise ValueError("CityFlow roadnet must define intersections and roads lists")

    all_intersections = {}
    for intersection in intersections:
        if not isinstance(intersection, Mapping):
            raise ValueError("Every CityFlow intersection entry must be an object")
        intersection_id = intersection.get("id")
        if not isinstance(intersection_id, str) or not intersection_id:
            raise ValueError("Every CityFlow intersection must have a non-empty string ID")
        if intersection_id in all_intersections:
            raise ValueError(f"Duplicate CityFlow intersection ID: {intersection_id!r}")
        all_intersections[intersection_id] = intersection

    agent_ids = tuple(sorted(
        intersection_id
        for intersection_id, intersection in all_intersections.items()
        if not bool(intersection.get("virtual", False))
    ))
    if not agent_ids:
        raise ValueError("CityFlow roadnet has no non-virtual intersections to control")

    roads_by_end = {intersection_id: [] for intersection_id in agent_ids}
    seen_road_ids = set()
    for road in roads:
        if not isinstance(road, Mapping):
            raise ValueError("Every CityFlow road entry must be an object")
        road_id = road.get("id")
        if not isinstance(road_id, str) or not road_id:
            raise ValueError("Every CityFlow road must have a non-empty string ID")
        if road_id in seen_road_ids:
            raise ValueError(f"Duplicate CityFlow road ID: {road_id!r}")
        seen_road_ids.add(road_id)
        end_intersection = road.get("endIntersection")
        if end_intersection not in roads_by_end:
            continue
        lanes = road.get("lanes")
        if not isinstance(lanes, list) or not lanes:
            raise ValueError(f"Incoming road {road_id!r} must define at least one lane")
        geometric_length = None
        for lane_index, lane in enumerate(lanes):
            if not isinstance(lane, Mapping):
                raise ValueError(f"Road {road_id!r} lane {lane_index} must be an object")
            lane_id = f"{road_id}_{lane_index}"
            if "length" in lane:
                lane_length = _finite_number(
                    lane["length"], f"Lane {lane_id!r} length", positive=True
                )
            else:
                if geometric_length is None:
                    geometric_length = _polyline_length(road.get("points"), road_id)
                lane_length = geometric_length
            roads_by_end[end_intersection].append((road_id, lane_index, lane_id, lane_length))

    _validate_id_mapping(phase_ids, agent_ids, "phase_ids")
    normalized_phase_ids = {}
    phase_links_by_agent = {}
    phase_inbound_lane_ids = {}
    for intersection_id in agent_ids:
        selected = phase_ids[intersection_id]
        if not isinstance(selected, Sequence) or isinstance(selected, (str, bytes)) or not selected:
            raise ValueError(
                f"phase_ids[{intersection_id!r}] must be a non-empty sequence of phase indices"
            )
        phase_links = _phase_road_links(all_intersections[intersection_id], intersection_id)
        normalized = []
        for phase_index in selected:
            if type(phase_index) is not int or not 0 <= phase_index < len(phase_links):
                raise ValueError(
                    f"phase_ids[{intersection_id!r}] contains an invalid phase index "
                    f"{phase_index!r}"
                )
            if phase_index in normalized:
                raise ValueError(
                    f"phase_ids[{intersection_id!r}] contains duplicate phase {phase_index}"
                )
            if not phase_links[phase_index]:
                raise ValueError(
                    f"phase_ids[{intersection_id!r}] phase {phase_index} has no available "
                    "road links; selectable actions must be real green phases"
                )
            normalized.append(phase_index)
        normalized_phase_ids[intersection_id] = normalized
        phase_links_by_agent[intersection_id] = phase_links
        inbound = sorted(roads_by_end[intersection_id], key=lambda row: (row[0], row[1]))
        if not inbound:
            raise ValueError(
                f"Non-virtual intersection {intersection_id!r} has no incoming road lanes"
            )
        lane_ids = [row[2] for row in inbound]
        if len(lane_ids) != len(set(lane_ids)):
            raise ValueError(f"Intersection {intersection_id!r} has duplicate incoming lane IDs")
        if service_limits_enabled:
            if len(normalized) < 2:
                raise ValueError(
                    "Signal service limits require at least two selectable green phases "
                    f"at {intersection_id!r}"
                )
            phase_inbound_lane_ids[intersection_id] = {}
            road_links = all_intersections[intersection_id]["roadLinks"]
            for phase_index in normalized:
                served = set()
                for link_index in phase_links[phase_index]:
                    link = road_links[link_index]
                    if not isinstance(link, Mapping):
                        raise ValueError("Signal service limits require valid roadLinks objects")
                    start_road = link.get("startRoad")
                    lane_links = link.get("laneLinks")
                    if not isinstance(lane_links, list) or not lane_links:
                        raise ValueError(
                            "Signal service limits require non-empty roadLink.laneLinks"
                        )
                    for lane_link in lane_links:
                        start_lane = lane_link.get("startLaneIndex") if isinstance(lane_link, Mapping) else None
                        lane_id = f"{start_road}_{start_lane}"
                        if type(start_lane) is not int or lane_id not in lane_ids:
                            raise ValueError(
                                "Signal service limits found a roadLink lane that is not "
                                f"an incoming lane at {intersection_id!r}: {lane_id!r}"
                            )
                        served.add(lane_id)
                phase_inbound_lane_ids[intersection_id][str(phase_index)] = sorted(served)

    if yellow_phase_ids is None:
        normalized_yellow_ids = None
        if yellow_time > 0:
            raise ValueError(
                "yellow_phase_ids is required when yellow_time is positive; "
                "the adapter never guesses a clearance or all-red phase"
            )
    else:
        _validate_id_mapping(yellow_phase_ids, agent_ids, "yellow_phase_ids")
        normalized_yellow_ids = {}
        for intersection_id in agent_ids:
            phase_index = yellow_phase_ids[intersection_id]
            phase_links = phase_links_by_agent[intersection_id]
            if type(phase_index) is not int or not 0 <= phase_index < len(phase_links):
                raise ValueError(
                    f"yellow_phase_ids[{intersection_id!r}] is not a valid roadnet phase index"
                )
            if phase_index in normalized_phase_ids[intersection_id]:
                raise ValueError(
                    f"yellow phase {phase_index} at {intersection_id!r} is also selectable green"
                )
            if phase_links[phase_index]:
                raise ValueError(
                    f"yellow_phase_ids[{intersection_id!r}] phase {phase_index} must have "
                    "an empty availableRoadLinks list (an explicit all-red clearance phase)"
                )
            normalized_yellow_ids[intersection_id] = phase_index

    control_steps = max(1, math.ceil(control_interval / interval - 1e-12))
    yellow_steps = math.ceil(yellow_time / interval - 1e-12)
    inbound_lane_ids = {}
    inbound_lane_lengths = {}
    for intersection_id in agent_ids:
        inbound = sorted(roads_by_end[intersection_id], key=lambda row: (row[0], row[1]))
        inbound_lane_ids[intersection_id] = [row[2] for row in inbound]
        inbound_lane_lengths[intersection_id] = [float(row[3]) for row in inbound]

    n_actions = max(len(normalized_phase_ids[intersection_id]) for intersection_id in agent_ids)
    max_inbound_lanes = max(len(inbound_lane_ids[intersection_id]) for intersection_id in agent_ids)
    obs_size = n_actions + 2 * max_inbound_lanes
    spec = {
        "cityflow_config_file": str(config_path),
        "cityflow_root": str(root),
        "roadnet_file": str(files["roadnetFile"]),
        "flow_file": str(files["flowFile"]),
        "interval": interval,
        "control_interval": control_interval,
        "control_steps": control_steps,
        "yellow_time": yellow_time,
        "yellow_steps": yellow_steps,
        "episode_limit": episode_limit,
        "seed": seed,
        "thread_num": thread_num,
        "queue_reward_scale": queue_reward_scale,
        "agent_ids": list(agent_ids),
        "phase_ids": normalized_phase_ids,
        "yellow_phase_ids": normalized_yellow_ids,
        "inbound_lane_ids": inbound_lane_ids,
        "inbound_lane_lengths": inbound_lane_lengths,
        "n_agents": len(agent_ids),
        "n_actions": n_actions,
        "max_inbound_lanes": max_inbound_lanes,
        "obs_size": obs_size,
        "state_size": len(agent_ids) * obs_size,
        "rl_traffic_light": True,
    }
    if service_limits_enabled:
        spec.update(
            max_green_time=max_green_time,
            max_red_time=max_red_time,
            phase_inbound_lane_ids=phase_inbound_lane_ids,
        )
    return spec


class CityFlowEngineEnv(MultiAgentEnv):
    """Fixed-order discrete signal control over native CityFlow.Engine.

    control_interval and yellow_time are seconds of simulated time and are
    rounded up to whole CityFlow steps. episode_limit counts decision cycles.
    """

    def __init__(
        self,
        cityflow_config_file,
        control_interval=10.0,
        yellow_time=0.0,
        episode_limit=360,
        phase_ids=None,
        yellow_phase_ids=None,
        seed=1,
        thread_num=1,
        queue_reward_scale=1.0,
        max_green_time=None,
        max_red_time=None,
    ):
        self.spec = read_traffic_spec(
            cityflow_config_file,
            phase_ids,
            yellow_phase_ids,
            control_interval=control_interval,
            yellow_time=yellow_time,
            episode_limit=episode_limit,
            seed=seed,
            thread_num=thread_num,
            queue_reward_scale=queue_reward_scale,
            max_green_time=max_green_time,
            max_red_time=max_red_time,
        )
        self.agent_ids = tuple(self.spec["agent_ids"])
        self.phase_ids = {
            intersection_id: list(self.spec["phase_ids"][intersection_id])
            for intersection_id in self.agent_ids
        }
        self.yellow_phase_ids = (
            None if self.spec["yellow_phase_ids"] is None
            else dict(self.spec["yellow_phase_ids"])
        )
        self.inbound_lane_ids = {
            intersection_id: list(self.spec["inbound_lane_ids"][intersection_id])
            for intersection_id in self.agent_ids
        }
        self.inbound_lane_lengths = {
            intersection_id: list(self.spec["inbound_lane_lengths"][intersection_id])
            for intersection_id in self.agent_ids
        }
        self.n_agents = self.spec["n_agents"]
        self.n_actions = self.spec["n_actions"]
        self.obs_size = self.spec["obs_size"]
        self.episode_limit = self.spec["episode_limit"]
        self.control_interval = self.spec["control_interval"]
        self.control_steps = self.spec["control_steps"]
        self.yellow_time = self.spec["yellow_time"]
        self.yellow_steps = self.spec["yellow_steps"]
        self.seed_value = self.spec["seed"]
        self.thread_num = self.spec["thread_num"]
        self.queue_reward_scale = self.spec["queue_reward_scale"]
        self.max_green_time = self.spec.get("max_green_time")
        self.max_red_time = self.spec.get("max_red_time")
        self._service_limits_enabled = (
            self.max_green_time is not None or self.max_red_time is not None
        )
        self._phase_inbound_lanes = {
            intersection_id: {
                int(phase): set(lanes) for phase, lanes in mapping.items()
            }
            for intersection_id, mapping in self.spec.get("phase_inbound_lane_ids", {}).items()
        }
        self._green_started = {}
        self._lane_last_service = {}
        self._service_limit_counts = {"max_green_enforcements": 0, "max_red_enforcements": 0}

        self._engine = None
        self._temp_config = None
        self.closed = False
        self.t = 0
        self._episode_index = -1
        self._current_phase = {}
        self._queue_by_lane = {}
        self._vehicle_by_lane = {}
        self._queue_sum = 0.0
        self._queue_samples = 0
        try:
            self._prepare_engine_config()
            try:
                import cityflow
            except ImportError as exc:
                raise ImportError(
                    "CityFlowEngineEnv needs the optional native 'cityflow' package "
                    "and the user's local roadnet/flow assets; nothing is bundled"
                ) from exc
            self._engine = cityflow.Engine(
                self._engine_config_file,
                thread_num=self.thread_num,
            )
            self.reset()
        except BaseException:
            self.close()
            raise

    def _prepare_engine_config(self):
        self._temp_config = tempfile.TemporaryDirectory(prefix="unified-marl-cityflow-")
        temp_path = Path(self._temp_config.name) / "config.json"
        _, config, root, files, _, _ = _load_cityflow_files(
            self.spec["cityflow_config_file"]
        )
        normalized = dict(config)
        normalized["dir"] = str(root) + os.sep
        for field in ("roadnetFile", "flowFile"):
            try:
                normalized[field] = os.path.relpath(files[field], root)
            except ValueError as exc:
                raise ValueError(
                    "CityFlow roadnetFile and flowFile must be on the same filesystem "
                    "volume as the configured dir so native relative paths remain valid"
                ) from exc
        normalized["seed"] = self.seed_value
        normalized["rlTrafficLight"] = True
        temp_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
        self._engine_config_file = str(temp_path)

    def _check_open(self):
        if self.closed or self._engine is None:
            raise RuntimeError("CityFlow environment is closed")

    def _set_phase(self, intersection_id, phase_id):
        self._engine.set_tl_phase(intersection_id, int(phase_id))

    def _refresh_counts(self):
        waiting = self._engine.get_lane_waiting_vehicle_count()
        vehicles = self._engine.get_lane_vehicle_count()
        self._queue_by_lane = {
            lane_id: max(0, int(waiting.get(lane_id, 0)))
            for intersection_id in self.agent_ids
            for lane_id in self.inbound_lane_ids[intersection_id]
        }
        self._vehicle_by_lane = {
            lane_id: max(0, int(vehicles.get(lane_id, 0)))
            for intersection_id in self.agent_ids
            for lane_id in self.inbound_lane_ids[intersection_id]
        }
        return sum(self._queue_by_lane.values())

    def _advance(self, steps):
        for _ in range(steps):
            self._engine.next_step()
            if self._service_limits_enabled:
                self._mark_service(self.get_current_time())
            queue_total = self._refresh_counts()
            self._queue_sum += queue_total
            self._queue_samples += 1

    def reset(self):
        self._check_open()
        self._episode_index += 1
        set_seed = getattr(self._engine, "set_random_seed", None)
        if callable(set_seed):
            set_seed(self.seed_value + self._episode_index)
        self._engine.reset()
        self.t = 0
        self._queue_sum = 0.0
        self._queue_samples = 0
        if self._service_limits_enabled:
            now = self.get_current_time()
            self._green_started = {intersection_id: now for intersection_id in self.agent_ids}
            self._lane_last_service = {
                intersection_id: {lane: now for lane in self.inbound_lane_ids[intersection_id]}
                for intersection_id in self.agent_ids
            }
            self._service_limit_counts = {"max_green_enforcements": 0, "max_red_enforcements": 0}
        for intersection_id in self.agent_ids:
            first_phase = self.phase_ids[intersection_id][0]
            self._set_phase(intersection_id, first_phase)
            self._current_phase[intersection_id] = first_phase
        self._refresh_counts()
        return self.get_obs(), self.get_state()

    def _mark_service(self, now):
        """Record actual green service; clearance phases serve no inbound lanes."""
        for intersection_id in self.agent_ids:
            served = self._phase_inbound_lanes[intersection_id].get(
                self._current_phase.get(intersection_id), set()
            )
            for lane_id in served:
                self._lane_last_service[intersection_id][lane_id] = now

    def _signal_action_mask(self, intersection_id, now):
        phases = self.phase_ids[intersection_id]
        mask = [1] * len(phases) + [0] * (self.n_actions - len(phases))
        current = self._current_phase[intersection_id]
        served = self._phase_inbound_lanes[intersection_id].get(current, set())
        waiting = []
        for index, phase in enumerate(phases):
            if phase == current:
                continue
            # Shared lanes already have green; their queues must not count as
            # red starvation for another overlapping phase.
            lanes = self._phase_inbound_lanes[intersection_id][phase] - served
            queued = [lane for lane in lanes if self._queue_by_lane.get(lane, 0) > 0]
            if queued:
                age = max(now - self._lane_last_service[intersection_id][lane] for lane in queued)
                waiting.append((index, age))
        if self.max_red_time is not None:
            expired = [item for item in waiting if item[1] >= self.max_red_time - 1e-12]
            if expired:
                forced, _ = max(expired, key=lambda item: (item[1], -item[0]))
                return [int(index == forced) for index in range(self.n_actions)], "max_red_enforcements"
        if (
            self.max_green_time is not None and waiting
            and now - self._green_started[intersection_id] >= self.max_green_time - 1e-12
        ):
            mask[phases.index(current)] = 0
            return mask, "max_green_enforcements"
        return mask, None

    def _signal_action_masks(self):
        self._check_open()
        self._refresh_counts()
        now = self.get_current_time()
        masks, reasons = [], []
        for intersection_id in self.agent_ids:
            mask, reason = self._signal_action_mask(intersection_id, now)
            masks.append(mask)
            reasons.append(reason)
        return masks, reasons

    def _actions(self, actions):
        raw = np.asarray(actions)
        if raw.size != self.n_agents:
            raise ValueError(f"Expected {self.n_agents} actions in agent_ids order")
        try:
            numeric = raw.astype(np.float64).reshape(self.n_agents)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Actions must be finite integer phase-choice indices") from exc
        if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError("Actions must be finite integer phase-choice indices")
        result = numeric.astype(np.int64)
        for index, intersection_id in enumerate(self.agent_ids):
            if not 0 <= result[index] < len(self.phase_ids[intersection_id]):
                raise ValueError(
                    f"Action {int(result[index])} is unavailable for {intersection_id!r}"
                )
        return result

    def actions_to_phase_ids(self, actions):
        """Convert policy action indices to intersection_id/phase_id records."""
        choices = self._actions(actions)
        return [
            {"intersection_id": intersection_id,
             "phase_id": int(self.phase_ids[intersection_id][choices[index]])}
            for index, intersection_id in enumerate(self.agent_ids)
        ]

    def step(self, actions):
        self._check_open()
        if self.t >= self.episode_limit:
            raise RuntimeError("Call reset before stepping a finished episode")
        choices = self._actions(actions)
        if self._service_limits_enabled:
            masks, reasons = self._signal_action_masks()
            for index, intersection_id in enumerate(self.agent_ids):
                if not masks[index][int(choices[index])]:
                    raise ValueError(
                        f"Action {int(choices[index])} is unavailable for {intersection_id!r} "
                        "under the configured signal service-limit action mask"
                    )
            for reason in reasons:
                if reason is not None:
                    self._service_limit_counts[reason] += 1
        targets = {
            intersection_id: self.phase_ids[intersection_id][int(choices[index])]
            for index, intersection_id in enumerate(self.agent_ids)
        }
        changed = [
            intersection_id for intersection_id in self.agent_ids
            if targets[intersection_id] != self._current_phase[intersection_id]
        ]
        if changed and self.yellow_steps:
            for intersection_id in changed:
                safe_phase = self.yellow_phase_ids[intersection_id]
                self._set_phase(intersection_id, safe_phase)
                self._current_phase[intersection_id] = safe_phase
            self._advance(self.yellow_steps)
        for intersection_id in changed:
            self._set_phase(intersection_id, targets[intersection_id])
            self._current_phase[intersection_id] = targets[intersection_id]
            if self._service_limits_enabled:
                self._green_started[intersection_id] = self.get_current_time()
        self._advance(self.control_steps)

        self.t += 1
        current_queue = self._refresh_counts()
        reward = -self.queue_reward_scale * float(current_queue)
        done = self.t >= self.episode_limit
        info = {
            "episode_limit": bool(done),
            "TimeLimit.truncated": bool(done),
            "current_time": self.get_current_time(),
            "current_queue_vehicles": int(current_queue),
            "avg_queue_vehicles": (
                float(self._queue_sum / self._queue_samples)
                if self._queue_samples else float(current_queue)
            ),
            "actions": self.actions_to_phase_ids(choices),
        }
        return float(reward), bool(done), info

    def _observation_array(self):
        self._check_open()
        self._refresh_counts()
        observations = np.zeros((self.n_agents, self.obs_size), dtype=np.float32)
        for agent_index, intersection_id in enumerate(self.agent_ids):
            phases = self.phase_ids[intersection_id]
            observations[agent_index, phases.index(self._current_phase[intersection_id])] = 1.0
            for lane_index, lane_id in enumerate(self.inbound_lane_ids[intersection_id]):
                offset = self.n_actions + 2 * lane_index
                observations[agent_index, offset] = self._queue_by_lane[lane_id]
                observations[agent_index, offset + 1] = self._vehicle_by_lane[lane_id]
        return observations

    def get_obs(self):
        return self._observation_array()

    def get_obs_agent(self, agent_id):
        if isinstance(agent_id, str):
            try:
                agent_id = self.agent_ids.index(agent_id)
            except ValueError:
                raise IndexError(f"Unknown intersection ID {agent_id!r}") from None
        if type(agent_id) is not int or not 0 <= agent_id < self.n_agents:
            raise IndexError("agent_id must be a valid agent index or intersection ID")
        return self._observation_array()[agent_id].copy()

    def get_state(self):
        return self._observation_array().reshape(-1).copy()

    def get_obs_size(self):
        return self.obs_size

    def get_state_size(self):
        return self.n_agents * self.obs_size

    def get_total_actions(self):
        return self.n_actions

    def get_avail_agent_actions(self, agent_id):
        if isinstance(agent_id, str):
            try:
                agent_id = self.agent_ids.index(agent_id)
            except ValueError:
                raise IndexError(f"Unknown intersection ID {agent_id!r}") from None
        if type(agent_id) is not int or not 0 <= agent_id < self.n_agents:
            raise IndexError("agent_id must be a valid agent index or intersection ID")
        count = len(self.phase_ids[self.agent_ids[agent_id]])
        if self._service_limits_enabled:
            return self._signal_action_masks()[0][agent_id]
        return [1] * count + [0] * (self.n_actions - count)

    def get_avail_actions(self):
        if self._service_limits_enabled:
            return self._signal_action_masks()[0]
        return [self.get_avail_agent_actions(index) for index in range(self.n_agents)]

    def get_current_time(self):
        self._check_open()
        return float(self._engine.get_current_time())

    def get_avg_travel_time(self):
        self._check_open()
        return float(self._engine.get_average_travel_time())

    def get_stats(self):
        self._check_open()
        current_queue = self._refresh_counts()
        avg_queue = (
            float(self._queue_sum / self._queue_samples)
            if self._queue_samples else float(current_queue)
        )
        lane_count = sum(len(self.inbound_lane_ids[item]) for item in self.agent_ids)
        stats = {
            "avg_travel_time": self.get_avg_travel_time(),
            "avg_queue_length": avg_queue,
            "avg_queue_vehicles": avg_queue,
            "avg_queue_per_inbound_lane": avg_queue / max(1, lane_count),
            "current_queue_vehicles": int(current_queue),
            "simulation_time": self.get_current_time(),
        }
        if self._service_limits_enabled:
            stats.update(self._service_limit_counts)
        return stats

    def seed(self):
        return self.seed_value

    def render(self):
        raise NotImplementedError("CityFlow rendering is configured by the native scenario")

    def save_replay(self):
        raise NotImplementedError(
            "Configure native CityFlow replay output in the supplied JSON config"
        )

    def close(self):
        if self.closed:
            return
        self.closed = True
        engine = self._engine
        self._engine = None
        try:
            close = getattr(engine, "close", None)
            if callable(close):
                close()
        finally:
            if self._temp_config is not None:
                self._temp_config.cleanup()
                self._temp_config = None


TrafficEngineEnv = CityFlowEngineEnv

__all__ = ["CityFlowEngineEnv", "TrafficEngineEnv", "read_traffic_spec"]
