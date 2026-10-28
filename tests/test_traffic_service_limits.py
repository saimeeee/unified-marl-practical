"""Signal-mask contract tests with a deterministic native-engine substitute.

These are unit tests, not CityFlow experiments or policy-quality evidence.
"""
import json
import sys
from types import SimpleNamespace

import pytest

from unified_marl.envs.traffic_engine import CityFlowEngineEnv, read_traffic_spec


class FakeEngine:
    def __init__(self, config_path, thread_num):
        self.interval = json.loads(open(config_path, encoding="utf-8").read())["interval"]
        self.time = 0.0
        self.waiting = {}
        self.phases = {}
        self.phase_history = []
        self.closed = False

    def set_random_seed(self, seed):
        self.seed = seed

    def reset(self):
        self.time = 0.0
        self.waiting = {}
        self.phase_history = []

    def set_tl_phase(self, intersection, phase):
        self.phases[intersection] = phase
        self.phase_history.append((self.time, intersection, phase))

    def next_step(self):
        self.time += self.interval

    def get_current_time(self):
        return self.time

    def get_lane_waiting_vehicle_count(self):
        return dict(self.waiting)

    def get_lane_vehicle_count(self):
        return dict(self.waiting)

    def get_average_travel_time(self):
        return 0.0

    def close(self):
        self.closed = True


@pytest.fixture
def scene(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "cityflow", SimpleNamespace(Engine=FakeEngine))
    roads = [
        {"id": name, "startIntersection": "outside_" + name,
         "endIntersection": "junction", "lanes": [{"length": 100.0}]}
        for name in ("a", "b", "c")
    ]
    road_links = [
        {"startRoad": name, "endRoad": "outgoing", "laneLinks": [
            {"startLaneIndex": 0, "endLaneIndex": 0}
        ]}
        for name in ("a", "b", "c")
    ]
    roadnet = {"roads": roads, "intersections": [
        {"id": "junction", "virtual": False, "roadLinks": road_links,
         "trafficLight": {"lightphases": [
             {"availableRoadLinks": [0]}, {"availableRoadLinks": [1]},
             {"availableRoadLinks": [2]}, {"availableRoadLinks": []},
         ]}}
    ]}
    roadnet_path = tmp_path / "roadnet.json"
    roadnet_path.write_text(json.dumps(roadnet), encoding="utf-8")
    (tmp_path / "flow.json").write_text("[]", encoding="utf-8")
    config_path = tmp_path / "cityflow.json"
    native = {"interval": 1.0, "dir": ".", "roadnetFile": "roadnet.json",
              "flowFile": "flow.json", "rlTrafficLight": True}
    config_path.write_text(json.dumps(native), encoding="utf-8")
    environments = []

    def create(**kwargs):
        defaults = dict(cityflow_config_file=str(config_path), phase_ids={"junction": [0, 1, 2]},
                        control_interval=10.0, episode_limit=20)
        defaults.update(kwargs)
        env = CityFlowEngineEnv(**defaults)
        environments.append(env)
        return env

    yield SimpleNamespace(create=create, config=config_path, native=native,
                          roadnet=roadnet, roadnet_path=roadnet_path)
    for env in environments:
        env.close()


def test_default_metadata_and_behavior_remain_compatible(scene):
    original = read_traffic_spec(scene.config, {"junction": [0, 1, 2]})
    explicit_none = read_traffic_spec(scene.config, {"junction": [0, 1, 2]},
                                      max_green_time=None, max_red_time=None)
    assert original == explicit_none
    assert not {"max_green_time", "max_red_time", "phase_inbound_lane_ids"} & original.keys()
    env = scene.create()
    env._engine.waiting = {"b_0": 5}
    for _ in range(5):
        env.step([0])
    assert env.get_avail_actions() == [[1, 1, 1]]
    assert not {"max_green_enforcements", "max_red_enforcements"} & env.get_stats().keys()


def test_enabled_metadata_records_limits_and_true_lane_service(scene):
    spec = read_traffic_spec(scene.config, {"junction": [0, 1, 2]}, max_green_time=30)
    assert spec["max_green_time"] == 30.0
    assert spec["max_red_time"] is None
    assert spec["phase_inbound_lane_ids"] == {
        "junction": {"0": ["a_0"], "1": ["b_0"], "2": ["c_0"]}
    }


@pytest.mark.parametrize("old,new", [(None, 20), (20, 40)])
def test_application_checkpoint_rejects_changed_mask_contract(scene, tmp_path, old, new):
    from copy import deepcopy
    from unified_marl.applications.service import validate, _inference_scene
    config = {"application": "traffic", "backend": "cityflow_engine", "algorithm": "ppo",
              "env_args": {"cityflow_config_file": str(scene.config),
                           "phase_ids": {"junction": [0, 1, 2]}}}
    if old is not None:
        config["env_args"]["max_green_time"] = old
    report = validate(config)
    saved = {key: report[key] for key in ("config", "metadata", "algorithm", "environment")}
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "application.json").write_text(json.dumps(saved), encoding="utf-8")
    changed = deepcopy(config)
    changed["env_args"]["max_green_time"] = new
    with pytest.raises(ValueError, match="action semantics differ"):
        _inference_scene("traffic", model_dir / "policy.pt", changed)


@pytest.mark.parametrize("name,value", [
    ("max_green_time", 0), ("max_green_time", -1), ("max_green_time", True),
    ("max_red_time", float("nan")), ("max_red_time", float("inf")),
])
def test_invalid_limit_is_rejected(scene, name, value):
    with pytest.raises(ValueError, match=name):
        scene.create(**{name: value})


def test_single_phase_limits_are_rejected_but_legacy_is_allowed(scene):
    assert scene.create(phase_ids={"junction": [0]}).get_avail_actions() == [[1]]
    with pytest.raises(ValueError, match="at least two"):
        scene.create(phase_ids={"junction": [0]}, max_red_time=10)


def test_enabled_limits_require_valid_lane_links(scene):
    scene.roadnet["intersections"][0]["roadLinks"][0]["laneLinks"][0]["startLaneIndex"] = 7
    scene.roadnet_path.write_text(json.dumps(scene.roadnet), encoding="utf-8")
    with pytest.raises(ValueError, match="not an incoming lane"):
        scene.create(max_red_time=10)
    # Legacy contracts never inspected laneLinks; preserve their validation.
    scene.create()


def test_max_green_requires_other_queued_service_and_counts_only_steps(scene):
    env = scene.create(max_green_time=20)
    env.step([0])
    env.step([0])
    assert env.get_avail_actions() == [[1, 1, 1]]  # no competing demand
    env._engine.waiting = {"b_0": 3}
    assert env.get_avail_actions() == [[0, 1, 1]]
    assert env.get_avail_agent_actions("junction") == [0, 1, 1]
    assert env.get_stats()["max_green_enforcements"] == 0
    env.step([1])
    assert env.get_stats()["max_green_enforcements"] == 1


def test_dynamic_mask_rejects_action_before_simulation_and_mapping_stays_static(scene):
    env = scene.create(max_green_time=10)
    env._engine.waiting = {"b_0": 2}
    # This valid action becomes forbidden at its END. Returning info must still
    # convert it without incorrectly applying the next decision's dynamic mask.
    _, _, info = env.step([0])
    assert info["actions"] == [{"intersection_id": "junction", "phase_id": 0}]
    assert env.actions_to_phase_ids([0]) == info["actions"]
    before = (env.t, env.get_current_time(), list(env._engine.phase_history))
    with pytest.raises(ValueError, match="service-limit action mask"):
        env.step([0])
    assert before == (env.t, env.get_current_time(), env._engine.phase_history)
    assert env.get_stats()["max_green_enforcements"] == 0


def test_max_red_forces_oldest_unserved_queued_phase(scene):
    env = scene.create(max_red_time=15)
    env.step([1])  # B served until t=10; C last served at reset t=0.
    env.step([0])  # A served until t=20.
    env._engine.waiting = {"b_0": 4, "c_0": 2}
    assert env.get_avail_actions() == [[0, 0, 1]]
    env.step([2])
    assert env.get_stats()["max_red_enforcements"] == 1
    assert env.get_avail_actions() == [[0, 1, 0]]  # B is now oldest.


def test_max_red_tie_is_deterministic_and_takes_priority_over_max_green(scene):
    env = scene.create(max_green_time=10, max_red_time=10)
    env._engine.waiting = {"b_0": 3, "c_0": 7}
    env.step([0])
    assert env.get_avail_actions() == [[0, 1, 0]]
    env.step([1])
    assert env.get_stats()["max_red_enforcements"] == 1
    assert env.get_stats()["max_green_enforcements"] == 0


def test_shared_currently_served_lane_is_not_treated_as_red_starvation(scene):
    # Phase 2 also serves A, but its only actual waiting lane already has green.
    scene.roadnet["intersections"][0]["trafficLight"]["lightphases"][2]["availableRoadLinks"] = [0, 2]
    scene.roadnet_path.write_text(json.dumps(scene.roadnet), encoding="utf-8")
    env = scene.create(max_green_time=10, max_red_time=10)
    env._engine.waiting = {"a_0": 5}
    env.step([0])
    assert env.get_avail_actions() == [[1, 1, 1]]


def test_limits_use_engine_seconds_and_clearance_does_not_count_as_green(scene):
    scene.native["interval"] = 0.6
    scene.config.write_text(json.dumps(scene.native), encoding="utf-8")
    env = scene.create(control_interval=2.0, yellow_time=1.0,
                       yellow_phase_ids={"junction": 3}, max_green_time=3.0)
    env._engine.waiting = {"a_0": 1, "c_0": 2}
    env.step([1])  # 1.2s clearance, then 2.4s green: real t=3.6.
    assert env.get_current_time() == pytest.approx(3.6)
    assert env._green_started["junction"] == pytest.approx(1.2)
    assert env._lane_last_service["junction"]["b_0"] == pytest.approx(3.6)
    assert env._lane_last_service["junction"]["a_0"] == 0.0
    assert env.get_avail_actions() == [[1, 1, 1]]
    assert any(phase == 3 for _, _, phase in env._engine.phase_history)
    env.step([1])
    assert env.get_avail_actions() == [[1, 0, 1]]


def test_reset_clears_timers_masks_and_enforcement_counters(scene):
    env = scene.create(max_red_time=10)
    env._engine.waiting = {"b_0": 2}
    env.step([0])
    env.step([1])
    assert env.get_stats()["max_red_enforcements"] == 1
    env.reset()
    env._engine.waiting = {"b_0": 2}
    assert env.get_current_time() == 0
    assert env.get_avail_actions() == [[1, 1, 1]]
    assert env._green_started == {"junction": 0.0}
    assert all(value == 0 for value in env._lane_last_service["junction"].values())
    assert env.get_stats()["max_red_enforcements"] == 0
