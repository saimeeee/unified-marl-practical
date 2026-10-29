"""Portable helper checks; no simulator installation or training is required."""
from copy import deepcopy
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from runtime import evaluation_config, plain, save_json, seed_all, _parse_native_replay, world_snapshot


class RuntimeChecks(unittest.TestCase):
    def test_nested_serialization_and_nonfinite_rejection(self):
        @dataclass
        class Sample:
            count: object
            location: Path
        value = {1: Sample(np.int64(3), Path("relative/path")), "array": np.array([[.25, 2.]])}
        encoded = plain(value)
        self.assertEqual(encoded, {"1": {"count": 3, "location": str(Path("relative/path"))}, "array": [[.25, 2.]]})
        with TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "snapshot.json"
            save_json(path, value)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), encoded)
            with self.assertRaises(ValueError):
                save_json(path, {"voltage": np.float64(float("nan"))})

    def test_native_replay_preserves_coordinates_ids_and_lights(self):
        vehicles, lights = _parse_native_replay("12.5 -2 1.57 flow_3_9 0 5 2,;road_1 G r,road_2 r G,\n")
        self.assertEqual(vehicles, [{"x": 12.5, "y": -2., "angle": 1.57, "id": "flow_3_9", "length": 5., "width": 2.}])
        self.assertEqual(lights, {"road_1": ["G", "r"], "road_2": ["r", "G"]})
        self.assertEqual(_parse_native_replay(";\n"), ([], {}))

    def test_native_replay_rejects_incomplete_vehicle(self):
        with self.assertRaises(ValueError):
            _parse_native_replay("12.5 -2 1.57 flow_3_9 0 5;road_1 G\n")

    @staticmethod
    def world(collision_stat=1):
        def entity(name, position, radius=.08):
            return SimpleNamespace(name=name, size=radius, state=SimpleNamespace(
                p_pos=np.asarray(position, dtype=float), p_vel=np.array([.5, -.2])))
        world = SimpleNamespace(dt=.1,
            agents=[entity("a0", [0, 0]), entity("a1", [.1, 0]), entity("a2", [1, 1])],
            landmarks=[entity("l0", [0, 0]), entity("l1", [.2, 0]), entity("l2", [1, 1])])
        return SimpleNamespace(world=world, get_stats=lambda: {"collisions": collision_stat})

    def test_snapshot_uses_strict_coverage_and_actual_collision_statistics(self):
        env = self.world()
        before = [a.state.p_pos.copy() for a in env.world.agents]
        row = world_snapshot(env, 2, -1, -3, np.array([1, 0, 4]), .1, 4)
        self.assertEqual(row["covered_landmarks"], 2)  # Distance exactly .1 is outside.
        self.assertEqual(row["collision_pairs"], [[0, 1]])
        self.assertEqual(row["collision_pair_steps"], 4)
        self.assertEqual(row["actions"], [1, 0, 4])
        self.assertEqual(row["physical_time"], .2)
        for original, agent in zip(before, env.world.agents):
            np.testing.assert_array_equal(original, agent.state.p_pos)

    def test_snapshot_rejects_disagreement_and_nonfinite_physics(self):
        with self.assertRaises(AssertionError):
            world_snapshot(self.world(0), 1, 0, 0, None, .1, 0)
        env = self.world()
        env.world.agents[0].state.p_pos[0] = float("nan")
        with self.assertRaises(FloatingPointError):
            world_snapshot(env, 0, 0, 0, None, .1, 0)

    @staticmethod
    def inventory_config(root):
        agents = root / "agents.csv"
        agents.write_text("agent_id,warehouse_id,sku_id,initial_stock,capacity,mean_demand,lead_time,unit_price,unit_cost,holding_cost,shortage_cost\nA,W,S,20,100,10,1,10,4,.15,5\n", encoding="utf-8")
        demand = root / "demand.csv"
        demand.write_text("period,agent_id,demand\n0,A,0\n1,A,0\n2,A,0\n3,A,0\n", encoding="utf-8")
        return dict(application="warehouse", seed=11, device="cuda", env_args=dict(
            agents_file=str(agents), demand_file=str(demand), horizon=4))

    def test_inventory_evaluation_keeps_source_and_reproduces_new_demand(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.inventory_config(root)
            original = deepcopy(config)
            source_files = {p: p.read_bytes() for p in root.glob("*.csv")}
            outputs = []
            for name in ("run-one", "run-two"):
                target = root / name
                target.mkdir()
                derived = evaluation_config(config, 101, target)
                self.assertEqual(derived["seed"], 101)
                self.assertEqual(derived["device"], "cpu")
                self.assertTrue((target / "config.yaml").is_file())
                with (target / "demand.csv").open(newline="") as stream:
                    outputs.append([int(row["demand"]) for row in csv.DictReader(stream)])
            self.assertEqual(outputs[0], outputs[1])
            self.assertTrue(any(outputs[0]))
            self.assertEqual(config, original)
            for path, content in source_files.items():
                self.assertEqual(path.read_bytes(), content)

    def test_grid_evaluation_overrides_both_seeds_without_mutating_config(self):
        config = dict(application="powergrid", seed=11, device="cuda", env_args={"randomseed": 11})
        original = deepcopy(config)
        with TemporaryDirectory() as directory:
            derived = evaluation_config(config, 303, Path(directory))
        self.assertEqual(derived["seed"], 303)
        self.assertEqual(derived["env_args"]["randomseed"], 303)
        self.assertEqual(derived["device"], "cpu")
        self.assertEqual(config, original)

    def test_seed_resets_python_numpy_and_torch_generators(self):
        import torch
        thread_count = torch.get_num_threads()
        try:
            samples = []
            for _ in range(2):
                seed_all(303)
                samples.append((random.random(), float(np.random.random()), float(torch.rand(()))))
            self.assertEqual(samples[0], samples[1])
        finally:
            torch.set_num_threads(thread_count)


if __name__ == "__main__":
    unittest.main()
