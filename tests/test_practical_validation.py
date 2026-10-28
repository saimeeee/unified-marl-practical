"""Validation-harness checks; these are not real-environment training evidence."""
import importlib.util
import json
from pathlib import Path

import pytest
import yaml


SCRIPT = Path(__file__).resolve().parents[1] / "tools/validate_practical_v1.py"
SPEC = importlib.util.spec_from_file_location("practical_validation", SCRIPT)
validation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validation)


def test_inventory_uses_unchanged_existing_assets(tmp_path):
    source = tmp_path / "source/warehouse"
    source.mkdir(parents=True)
    data = {"agents.csv": b"agent_id\r\na\r\n", "demand.csv": b"period,agent_id,demand\n0,a,3\n"}
    for name, content in data.items():
        (source / name).write_bytes(content)
    args, evidence = validation.prepare_environment("inventory", source.parent, tmp_path / "copied", 7)
    assert evidence["synthetic_data"] is True
    assert "not original ReplenishmentEnv" in evidence["description"]
    for asset in evidence["assets"]:
        assert Path(asset["source"]).read_bytes() == Path(asset["copy"]).read_bytes()
        assert validation.sha256(asset["source"]) == asset["sha256"]
    assert args["seed"] == 7
    with pytest.raises(FileExistsError):
        validation.prepare_environment("inventory", source.parent, tmp_path / "copied", 7)


def test_missing_asset_is_not_replaced_with_synthetic_fallback(tmp_path):
    with pytest.raises(FileNotFoundError, match="Required existing scene asset missing"):
        validation.prepare_environment("inventory", tmp_path / "absent", tmp_path / "copy", 1)
    assert not (tmp_path / "copy/demand.csv").exists()


def test_traffic_retains_service_constraints_and_native_bytes(tmp_path):
    source = tmp_path / "source"
    traffic = source / "traffic"
    traffic.mkdir(parents=True)
    native = {"dir": ".", "roadnetFile": "roadnet.json", "flowFile": "flow.json"}
    (traffic / "cityflow.json").write_text(json.dumps(native), encoding="utf-8")
    (traffic / "roadnet.json").write_text('{"intersections": []}', encoding="utf-8")
    (traffic / "flow.json").write_text("[]", encoding="utf-8")
    (source / "traffic.yaml").write_text(yaml.safe_dump({"env_args": {
        "cityflow_config_file": "traffic/cityflow.json", "episode_limit": 32,
        "max_green_time": 20., "max_red_time": 40., "phase_ids": {"i": [0, 2]}}}), encoding="utf-8")
    args, evidence = validation.prepare_environment("traffic_engine", source, tmp_path / "copy", 4, 12)
    assert args["max_green_time"] == 20. and args["max_red_time"] == 40.
    assert args["phase_ids"] == {"i": [0, 2]}
    assert args["episode_limit"] == 12 and evidence["original_episode_limit"] == 32
    assert json.loads(Path(args["cityflow_config_file"]).read_text()) == native
    assert all(validation.sha256(asset["copy"]) == asset["sha256"] for asset in evidence["assets"])


def test_each_validation_run_has_a_new_directory(tmp_path):
    one = validation.create_run_directory(tmp_path)
    (one / "keep.txt").write_text("previous evidence", encoding="utf-8")
    two = validation.create_run_directory(tmp_path)
    assert one != two
    assert (one / "keep.txt").read_text() == "previous evidence"


def test_mpe_recipe_cannot_accept_a_toy_backend(tmp_path):
    args, evidence = validation.prepare_environment("mpe", tmp_path, tmp_path / "mpe", 3)
    assert args == {"scenario": "simple_spread", "n_agents": 3, "episode_limit": 12, "seed": 3}
    assert evidence["evidence_level"] == "MIGRATED_MPE_PARTICLE_SIMULATOR"
    with pytest.raises(ValueError, match="No real-environment validation recipe"):
        validation.prepare_environment("toy", tmp_path, tmp_path / "toy", 3)
