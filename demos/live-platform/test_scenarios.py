"""Meaningful checks for generated networks, policy provenance and exact demand."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import yaml

from scenarios import HERE, PressurePolicy, prepare_traffic, straight_routes, validate_parameters


def source_config():
    cfg = yaml.safe_load((HERE / "bundle/configs/traffic.yaml").read_text(encoding="utf-8"))
    cfg["env_args"]["cityflow_config_file"] = str(HERE / "bundle/assets/traffic/cityflow.json")
    return cfg


class ScenarioChecks(unittest.TestCase):
    def test_input_boundaries_and_explicit_controller(self):
        for invalid in ([], "traffic", 1):
            with self.assertRaises(ValueError): validate_parameters("traffic", invalid)
        for key, values in {"rows": [0, 5, True, 2.0], "cols": [0, 5],
                            "vehicle_count": [-1, 3001, False], "decision_limit": [7, 181]}.items():
            for value in values:
                with self.assertRaises(ValueError): validate_parameters("traffic", {key: value})
        with self.assertRaises(ValueError): validate_parameters("traffic", {"controller": "random"})
        with self.assertRaises(ValueError): validate_parameters("traffic", {"arbitrary_file": "/tmp/a"})
        with self.assertRaises(ValueError): validate_parameters("mpe", {"rows": 2})
        with self.assertRaisesRegex(ValueError, "2×2"): validate_parameters("traffic", {"rows": 3})
        self.assertEqual(validate_parameters("traffic", {}), {})
        self.assertEqual(validate_parameters("traffic", {"rows": 1, "controller": "pressure"})["vehicle_count"], 400)

    def test_empty_parameters_keep_original_config(self):
        config = source_config()
        with TemporaryDirectory() as folder:
            result, receipt = prepare_traffic(config, {}, folder)
            self.assertIs(result, config)
            self.assertFalse(receipt["custom"])
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_every_grid_has_exact_intersections_and_connected_routes(self):
        from unified_marl.envs.traffic_engine import read_traffic_spec
        for rows in range(1, 5):
            for cols in range(1, 5):
                with self.subTest(rows=rows, cols=cols), TemporaryDirectory() as folder:
                    result, receipt = prepare_traffic(source_config(),
                        dict(rows=rows, cols=cols, controller="pressure", vehicle_count=37), folder)
                    spec = read_traffic_spec(**result["env_args"])
                    self.assertEqual(spec["n_agents"], rows * cols)
                    self.assertEqual(receipt["intersection_count"], rows * cols)
                    roadnet = json.loads((Path(folder)/"scenario/roadnet.json").read_text())
                    roads = {r["id"]: r for r in roadnet["roads"]}
                    for route in straight_routes(rows, cols):
                        for road in route: self.assertIn(road, roads)
                        for a, b in zip(route, route[1:]):
                            self.assertEqual(roads[a]["endIntersection"], roads[b]["startIntersection"])
                    for it in roadnet["intersections"]:
                        if not it.get("virtual"):
                            self.assertEqual(it["trafficLight"]["lightphases"][8]["availableRoadLinks"], [])

    def test_precise_departures_and_finite_window_include_zero(self):
        for count in (0, 1, 7, 400, 3000):
            with self.subTest(count=count), TemporaryDirectory() as folder:
                _, receipt = prepare_traffic(source_config(), dict(vehicle_count=count, decision_limit=8), folder)
                flow = json.loads((Path(folder)/"scenario/flow.json").read_text())
                self.assertEqual(len(flow), count)
                self.assertEqual(receipt["planned_vehicle_count"], count)
                for departure in flow:
                    self.assertEqual(departure["startTime"], departure["endTime"])
                    self.assertGreaterEqual(departure["startTime"], 0)
                    self.assertLess(departure["endTime"], 64)

    def test_same_grid_keeps_exact_roadnet_and_model_contract(self):
        from unified_marl.applications.service import _inference_scene
        source = HERE / "bundle/assets/traffic/roadnet.json"
        checksum = hashlib.sha256(source.read_bytes()).hexdigest()
        original = source_config()
        before = deepcopy(original)
        with TemporaryDirectory() as folder:
            result, receipt = prepare_traffic(original, dict(vehicle_count=73, decision_limit=8), folder)
            generated = Path(folder)/"scenario/roadnet.json"
            self.assertEqual(hashlib.sha256(generated.read_bytes()).hexdigest(), checksum)
            saved = _inference_scene("traffic", HERE / "bundle/models/traffic/policy.pt", result)
            self.assertEqual(saved["config"]["env_args"]["episode_limit"], 8)
            self.assertEqual(receipt["policy_source"], "trained_checkpoint")
        self.assertEqual(original, before)

    def test_pressure_never_selects_masked_out_phase(self):
        class Engine:
            def get_lane_waiting_vehicle_count(self): return {"in_0": 15, "out_0": 1, "in_1": 3}
        class Env:
            agent_ids = ["i"]
            phase_ids = {"i": [0, 2]}
            _current_phase = {"i": 0}
            _engine = Engine()
            def actions_to_phase_ids(self, actions): return [{"intersection_id": "i", "phase_id": self.phase_ids["i"][actions[0]]}]
        network = {"intersections": [{"id": "i", "roadLinks": [
            {"startRoad": "in", "endRoad": "out", "laneLinks": [{"startLaneIndex": 0, "endLaneIndex": 0}]},
            {"startRoad": "in", "endRoad": "out", "laneLinks": [{"startLaneIndex": 1, "endLaneIndex": 0}]}],
            "trafficLight": {"lightphases": [{"availableRoadLinks": [0]}, {}, {"availableRoadLinks": [1]}]}}]}
        policy = PressurePolicy(Env(), network)
        self.assertEqual(policy.act(None, [[1, 1]])["raw_actions"], [0])
        self.assertEqual(policy.act(None, [[0, 1]])["raw_actions"], [1])
        with self.assertRaises(ValueError): policy.act(None, [[0, 0]])


if __name__ == "__main__":
    unittest.main()
