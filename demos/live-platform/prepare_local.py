"""Verify portable assets and resolve paths on this computer/container."""
import argparse
import hashlib
import json
from pathlib import Path

import yaml

PATH_KEYS = {"agents_file", "demand_file", "cityflow_config_file", "data_path", "buildings_states_actions_file"}


def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Bundle path escapes its directory: {relative}")
    return path


def prepare(bundle, output):
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = inside(bundle, item["path"])
        data = path.read_bytes()
        if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError(f"Bundle checksum mismatch: {item['path']}")
    source = json.loads((bundle / "catalog.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    (output / "configs").mkdir(exist_ok=True)
    models = []
    for model in source["models"]:
        row = dict(model)
        row["checkpoint"] = str(inside(bundle, model["checkpoint"]))
        if model.get("config"):
            origin = inside(bundle, model["config"])
            cfg = yaml.safe_load(origin.read_text(encoding="utf-8"))
            cfg["device"] = "cpu"
            cfg["execution"] = "local"
            for key in PATH_KEYS:
                if key in cfg["env_args"]:
                    value = (origin.parent / cfg["env_args"][key]).resolve()
                    if not value.is_relative_to(bundle) or not value.exists():
                        raise ValueError(f"Missing or invalid scene asset: {key}")
                    cfg["env_args"][key] = str(value)
            target = output / "configs" / (model["scenario"] + ".yaml")
            target.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
            row["config"] = str(target)
        models.append(row)
    # User models live in the mounted state folder and survive container rebuilds.
    extras = output / "extra-models.json"
    if extras.exists():
        for model in json.loads(extras.read_text(encoding="utf-8"))["models"]:
            row = dict(model)
            for key in ("checkpoint", "config"):
                if row.get(key):
                    row[key] = str(inside(output, row[key]))
                    if not Path(row[key]).is_file():
                        raise FileNotFoundError(row[key])
            models.append(row)
    if len({row["id"] for row in models}) != len(models):
        raise ValueError("Duplicate model ids in catalog/extra-models.json")
    (output / "models.json").write_text(json.dumps(dict(schema_version=1, models=models), ensure_ascii=False, indent=2), encoding="utf-8")
    receipt = dict(mode="local_cpu", bundle_files_verified=len(manifest["files"]), models=len(models),
                   source_training_date="2026-10-28", network_required_for_inference=False)
    (output / "prepared.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt), flush=True)
    return output / "models.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.bundle, args.output)
