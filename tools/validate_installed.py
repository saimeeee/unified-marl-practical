"""Run from an external cwd against a wheel-installed interpreter."""
from dataclasses import asdict
from pathlib import Path
import importlib.util
import json
import sys
import uuid


def main():
    import unified_marl
    root = Path(sys.argv[1]).resolve()
    run_root = root / ("installed-runs-" + uuid.uuid4().hex[:8])
    assert "site-packages" in str(Path(unified_marl.__file__).resolve())
    records = {"python": sys.version, "executable": sys.executable, "package_file": unified_marl.__file__,
               "cwd": str(Path.cwd()), "algorithms": {}}
    for algorithm in ["qmix", "vdn", "iql", "coma", "qtran", "qmix_beta", "iql_beta", "vdn_beta"]:
        overrides = {"batch_size": 2, "buffer_size": 8, "t_max": 128, "max_updates": 2}
        if algorithm == "coma":
            overrides.update(runner="parallel", batch_size_run=2)
        result = unified_marl.train(algorithm, "toy", output_dir=run_root / algorithm, overrides=overrides)
        assert result.updates == 2 and result.parameter_delta_l2 > 0
        evaluation = unified_marl.evaluate(result.checkpoint, episodes=2)
        records["algorithms"][algorithm] = {"train": asdict(result), "evaluate": asdict(evaluation)}
    try:
        unified_marl.train("qmix", "sc2", output_dir=run_root / "smac-probe", env_args={"map_name": "3m"},
                           overrides={"batch_size": 2, "t_max": 8})
    except ImportError as exc:
        records["smac"] = {"status": "BLOCKED", "reason": str(exc), "training_verified": False}
    else:
        records["smac"] = {"status": "PASS", "training_verified": True}
    (root / "installed_validation.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"result": "PASS", "package_file": records["package_file"], "algorithm_count": len(records["algorithms"]), "smac": records["smac"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
