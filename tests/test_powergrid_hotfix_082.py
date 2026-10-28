"""Execute production calculations and installer transactions without a simulator."""
import ast
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "src/unified_marl/envs/powergrid_native"


def method(file, cls, name, namespace):
    source = ast.parse(file.read_text(encoding="utf-8"))
    class_node = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == cls)
    function = next(node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(file), "exec"), namespace)
    return namespace[name]


def building():
    return SimpleNamespace(
        enabled_states={"net_electricity_consumption": True},
        sim_results={"non_shiftable_load": [100., 110., 120., 130.], "dhw_demand": [1., 2., 3., 4.],
                     "cooling_demand": [3., 4., 5., 6.]},
        weather=SimpleNamespace(data={"solar_gen": [0., .1, .2, .3]}),
        pv_installed=25, solar_power_capacity=25,
        dhw_storage=SimpleNamespace(capacity=8), cooling_storage=SimpleNamespace(capacity=4),
    )


@pytest.mark.parametrize("pv", [0, 1, 25, 2.5])
def test_f1_actual_state_bounds_use_elementwise_pv_scaling(pv):
    item = building()
    item.pv_installed = item.solar_power_capacity = pv
    calculate = method(NATIVE / "energy_models.py", "Building", "set_state_space", {
        "np": np, "spaces": SimpleNamespace(Box=lambda **kwargs: kwargs),
    })
    calculate(item)
    expected = max(load - pv * solar + dhw / .8 + cooling + 8 / .8 + 4 / 2
                   for load, solar, dhw, cooling in zip([100, 110, 120, 130], [0, .1, .2, .3], [1, 2, 3, 4], [3, 4, 5, 6]))
    assert item._net_elec_cons_upper_bound == pytest.approx(expected)
    assert item.normalization_range.shape == (1,) and np.isfinite(item.normalization_range).all()
    assert item.weather.data["solar_gen"] == [0., .1, .2, .3]


@pytest.mark.parametrize("solar", [[0, 1], [0, 1, float("nan"), 1], [[0, 1], [2, 3]]])
def test_f1_rejects_bad_profile_shape_or_nonfinite_values(solar):
    item = building()
    item.weather.data["solar_gen"] = solar
    calculate = method(NATIVE / "energy_models.py", "Building", "set_state_space", {"np": np})
    with pytest.raises(ValueError, match="profile solar_gen"):
        calculate(item)


def test_f1_generation_history_also_uses_elementwise_scaling():
    file = NATIVE / "energy_models.py"
    source = ast.parse(file.read_text(encoding="utf-8"))
    assignment = next(node for node in ast.walk(source) if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Attribute) and target.attr == "electric_generation" for target in node.targets)
                      and isinstance(node.value, ast.BinOp))
    item = SimpleNamespace(pv_installed=25, time_step=3, weather=SimpleNamespace(data={"solar_gen": [0., .1, .2, .3]}))
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(file), "exec"), {"self": item, "np": np})
    np.testing.assert_allclose(item.electric_generation, [0, 2.5, 5])


@pytest.mark.parametrize("limit", [1, 4])
def test_f2_production_reset_and_step_count_control_transitions_only(limit):
    calls = {"physics": 0, "flow": 0}
    def physics(_):
        calls["physics"] += 1
    def flow(*args, **kwargs):
        calls["flow"] += 1
    ns = {"np": np, "runpp": flow}
    reset = method(NATIVE / "smart_grid.py", "GridEnv", "reset", ns)
    step = method(NATIVE / "smart_grid.py", "GridEnv", "step", ns)
    done = method(NATIVE / "smart_grid.py", "GridEnv", "_get_done", ns)
    item = SimpleNamespace(
        episode_limit=limit, exipode=0, hourly_timesteps=4,
        aspace={"a": SimpleNamespace(sample=lambda: np.array([0.]))}, agents=["a"],
        net=SimpleNamespace(shunt=SimpleNamespace(at={})),
        buildings={"a": SimpleNamespace(step=physics, reset_timestep=lambda *args: None)},
        voltage_data=[], load_data=[], gen_data=[],
        _select_timestamp=lambda _: 192, _update_grid=lambda: None,
        get_obs=lambda: np.zeros((1, 1)), get_state=lambda: np.zeros(1),
        _get_reward=lambda: ([1.], [0.]), _get_info=lambda: {},
    )
    item._get_done = lambda: done(item)
    item.net.res_bus = {"vm_pu": [1.]}
    item.net.load, item.net.sgen = {"p_mw": [0.]}, {"p_mw": [0.]}
    reset(item)
    assert item.steps == 0 and item.initialization_steps == 1
    assert calls == {"physics": 1, "flow": 1}  # real warmup path is preserved
    for index in range(limit):
        reward, finished, info = step(item, [[0.]])
        assert item.steps == index + 1 and reward == 1.
        assert finished == (index + 1 == limit)
    assert calls == {"physics": limit + 1, "flow": limit + 1}
    reset(item)
    assert item.steps == 0 and item.initialization_steps == 1


def test_old_powergrid_episode_semantics_are_explicitly_rejected():
    from unified_marl.applications.powergrid_engine import _checkpoint_contract_matches
    with pytest.raises(ValueError, match="episode-limit semantics"):
        _checkpoint_contract_matches({}, {"episode_limit_semantics": "control_transitions_v2"})


def helper():
    return runpy.run_path(str(ROOT / "delivery/hotfix/feature_install.py"))


def test_n1_filters_old_core_and_preserves_installed_tooling():
    hash_text = "a" * 64
    text = "\n".join(f"{req} --hash=sha256:{hash_text}" for req in [
        "unified-marl[hmf]==0.8.0", "setuptools==84.2.0", "numpy==1.26.4", "gym==0.26.2"])
    def installed(name):
        if name == "gym":
            raise metadata.PackageNotFoundError(name)
        return {"setuptools": "80.10.2", "numpy": "1.26.4"}[name]
    lines = helper()["missing_lock_lines"](text, installed)
    assert len(lines) == 1 and lines[0].startswith("gym==")


def make_bundle(root, edition="linux-x86_64-cpu-py312-engineering"):
    target = "linux/install.sh" if edition.startswith("linux") else "windows/install.ps1"
    (root / target).parent.mkdir(parents=True)
    (root / target).write_bytes(b"original installer\n")
    data = (root / target).read_bytes()
    manifest = {"edition": edition, "version": "0.8.0", "files": [{
        "path": target, "size": len(data), "sha256": hashlib.sha256(data).hexdigest(),
    }]}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return target


@pytest.mark.parametrize("edition", ["linux-x86_64-cpu-py312-engineering", "windows-x64-cpu-py312-engineering"])
def test_n1_installer_update_is_verified_backed_up_and_repeatable(tmp_path, edition):
    root = tmp_path / "bundle"
    target = make_bundle(root, edition)
    original = (root / "manifest.json").read_bytes()
    patch_root = tmp_path / "patch"
    patch_root.mkdir()
    for name in ["feature_install.py", "linux_install.sh", "windows_install.ps1"]:
        (patch_root / name).write_bytes((ROOT / "delivery/hotfix" / name).read_bytes())
    (patch_root / "manifest.json").write_text('{"version":"0.8.2"}')
    updater = runpy.run_path(str(ROOT / "delivery/hotfix/install_patch.py"))["update_installers"]
    updater(root, patch_root, {})
    after = (root / "manifest.json").read_bytes()
    manifest = json.loads(after)
    assert manifest["version"] == "0.8.2"
    assert set(manifest["applied_hotfixes"][-1]["fixes"]) == {"F1", "F2", "N1"}
    assert (root / ".unified-marl-hotfix/backups/pre-0.8.2/manifest.json").read_bytes() == original
    assert (root / ".unified-marl-hotfix/backups/pre-0.8.2" / Path(target).name).read_bytes() == b"original installer\n"
    helper()["verify_files"](root, manifest)
    updater(root, patch_root, {})
    assert (root / "manifest.json").read_bytes() == after
    (root / target).write_bytes(b"changed after update")
    with pytest.raises(ValueError, match="Integrity"):
        helper()["verify_files"](root, manifest)


def test_n1_refuses_corrupted_base_instead_of_refreshing_its_hashes(tmp_path):
    root = tmp_path / "bundle"
    target = make_bundle(root)
    (root / target).write_bytes(b"unrecognized modification")
    before = (root / "manifest.json").read_bytes()
    updater = runpy.run_path(str(ROOT / "delivery/hotfix/install_patch.py"))["update_installers"]
    with pytest.raises(ValueError, match="Integrity"):
        updater(root, ROOT / "delivery/hotfix", {})
    assert (root / "manifest.json").read_bytes() == before


def test_patch_manifest_covers_helpers_before_executing_them(tmp_path):
    verify = runpy.run_path(str(ROOT / "delivery/hotfix/install_patch.py"))["verify_patch"]
    file = tmp_path / "feature_install.py"
    file.write_bytes(b"bad file")
    manifest = {"files": [{"path": file.name, "bytes": 8, "sha256": "a" * 64}]}
    with pytest.raises(ValueError, match="Patch integrity"):
        verify(tmp_path, manifest)
