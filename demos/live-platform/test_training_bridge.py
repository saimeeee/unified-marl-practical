"""Portable export, registration, and scene/model compatibility regressions."""
from copy import deepcopy
import io
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
import zipfile

import numpy as np
import torch

from scenarios import HERE
from training_bridge import (ALGORITHMS, PROTOCOL, _traffic_contract,
    export_traffic_training, load_registered_traffic_policy, register_experiment)


class TrainingBridgeChecks(unittest.TestCase):
    def source_model(self):
        return dict(id="traffic", scenario="traffic", config=str(HERE / "bundle/configs/traffic.yaml"))

    def extract(self, content, folder):
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            self.assertTrue(all(not Path(name).is_absolute() and ".." not in Path(name).parts for name in archive.namelist()))
            archive.extractall(folder)

    def test_training_algorithm_is_independent_and_all_exports_pass_preflight(self):
        from unified_marl.experiments import check_experiment
        for algorithm in ALGORITHMS:
            with self.subTest(algorithm=algorithm), TemporaryDirectory() as temp:
                # A new policy can be trained on this grid even though the
                # original checkpoint selected in the webpage cannot run it.
                content = export_traffic_training(self.source_model(),
                    dict(rows=1, cols=3, vehicle_count=11, decision_limit=8, controller="trained"),
                    seed=37, algorithm=algorithm)
                folder = Path(temp) / "moved export"
                self.extract(content, folder)
                report = check_experiment(folder / "experiment.yaml")
                cfg = report["config"]
                self.assertEqual(cfg["algorithm"], algorithm)
                self.assertEqual(cfg["env"], "traffic_engine")
                self.assertEqual(cfg["seed"], 37)
                contract = _traffic_contract(cfg["env_args"])
                self.assertEqual(contract["env_info"]["n_agents"], 3)
                self.assertEqual(contract["env_info"]["episode_limit"], 8)
                flow = json.loads((folder / "scenario/flow.json").read_text())
                self.assertEqual(len(flow), 11)
                native = json.loads((folder / "scenario/cityflow.json").read_text())
                self.assertEqual(native["dir"], ".")
                self.assertFalse(native["saveReplay"])
                provenance = json.loads((folder / "scenario.json").read_text())
                self.assertFalse(provenance["existing_checkpoint_reused"])
                self.assertNotIn("policy_source", provenance)
                self.assertEqual(cfg["max_updates"], 2)

    def test_default_export_preserves_original_scene_and_zero_vehicles_are_explicit(self):
        for parameters, expected in (({}, None), ({"vehicle_count": 0}, 0)):
            with self.subTest(parameters=parameters), TemporaryDirectory() as temp:
                folder = Path(temp)
                self.extract(export_traffic_training(self.source_model(), parameters), folder)
                self.assertEqual((folder / "scenario/roadnet.json").read_bytes(),
                                 (HERE / "bundle/assets/traffic/roadnet.json").read_bytes())
                if expected is None:
                    self.assertEqual((folder / "scenario/flow.json").read_bytes(),
                                     (HERE / "bundle/assets/traffic/flow.json").read_bytes())
                else:
                    self.assertEqual(json.loads((folder / "scenario/flow.json").read_text()), [])

    def test_invalid_exports_reject_without_accepting_paths_or_unknown_algorithms(self):
        for options in (dict(algorithm="arbitrary/module"), dict(seed=True), dict(seed=-1),
                        dict(parameters={"destination": "../escape"}), dict(parameters=[]),
                        dict(parameters={"controller": "unknown"}), dict(parameters={"rows": 5})):
            with self.subTest(options=options), self.assertRaises(ValueError):
                export_traffic_training(self.source_model(), **options)

    def _run(self, folder, scene):
        run = folder / "run"
        run.mkdir()
        source = HERE / "bundle/models" / scene / "policy.pt"
        shutil.copyfile(source, run / "policy.pt")
        if scene == "traffic":
            from unified_marl.checkpoints import read_checkpoint
            payload = read_checkpoint(source)
            payload["config"]["env_args"]["cityflow_config_file"] = str(HERE / "bundle/assets/traffic/cityflow.json")
            torch.save(payload, run / "policy.pt")
        return run

    def test_register_mpe_preserves_model_and_rejects_duplicate_or_unsafe_ids(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            run = self._run(root, "mpe")
            original = (run / "policy.pt").read_bytes()
            state = root / "state"
            row = register_experiment(run, state, "mpe-qmix-new")
            self.assertEqual(row["scenario"], "mpe")
            self.assertEqual(row["algorithm"], "pymarl/qmix")
            self.assertEqual((state / row["checkpoint"]).read_bytes(), original)
            self.assertIsNone(row["config"])
            catalog = (state / "extra-models.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "已存在"):
                register_experiment(run, state, "mpe-qmix-new")
            with self.assertRaises(ValueError):
                register_experiment(run, state, "../../outside")
            self.assertEqual((state / "extra-models.json").read_bytes(), catalog)
            self.assertFalse((state / ".model-registration.lock").exists())

    def test_register_rejects_checkpoint_metadata_mismatch_before_catalog_write(self):
        from unified_marl.checkpoints import read_checkpoint
        with TemporaryDirectory() as temp:
            root = Path(temp)
            run = self._run(root, "mpe")
            payload = read_checkpoint(run / "policy.pt")
            payload["env_info"]["n_agents"] += 1
            torch.save(payload, run / "policy.pt")
            with self.assertRaisesRegex(ValueError, "元数据"):
                register_experiment(run, root / "state", "invalid")
            self.assertFalse((root / "state/extra-models.json").exists())

    def test_registered_traffic_is_portable_and_enforces_scene_contract(self):
        from unified_marl.applications import load_config
        with TemporaryDirectory() as temp:
            root = Path(temp)
            run = self._run(root, "traffic")
            state = root / "original state"
            row = register_experiment(run, state, "traffic-ppo-new")
            self.assertEqual(row["inference_protocol"], PROTOCOL)
            # Registration copies its own scene. It can move as a complete
            # local-state directory without depending on its old location.
            moved = root / "relocated state"
            shutil.copytree(state, moved)
            model = dict(row, checkpoint=str(moved / row["checkpoint"]), config=str(moved / row["config"]))
            cfg = load_config(model["config"])
            info = _traffic_contract(cfg["env_args"])["env_info"]

            class Env:
                def get_env_info(self):
                    return info
                def actions_to_phase_ids(self, actions):
                    return [dict(intersection_id=str(i), phase_id=int(a) * 2) for i, a in enumerate(actions)]

            policy = load_registered_traffic_policy(model, Env(), cfg)
            policy.reset()
            actions = policy.act(np.zeros((info["n_agents"], info["obs_shape"])),
                                 np.ones((info["n_agents"], info["n_actions"])))
            self.assertEqual(len(actions["raw_actions"]), info["n_agents"])
            self.assertEqual(len(actions["decisions"]), info["n_agents"])
            # Re-exporting the registered scene can train another algorithm.
            self.assertTrue(export_traffic_training(model, {}, algorithm="pymarl/vdn").startswith(b"PK"))
            with self.assertRaises(ValueError):
                export_traffic_training(model, {"rows": 1, "controller": "pressure"})
            native = Path(cfg["env_args"]["cityflow_config_file"])
            (native.parent / "flow.json").write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "不兼容"):
                load_registered_traffic_policy(model, Env(), cfg)


if __name__ == "__main__":
    unittest.main()
