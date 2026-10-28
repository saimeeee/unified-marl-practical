"""Feature-only installer for an already updated engineering venv (no core downgrade)."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile


FEATURES = ("core", "warehouse", "warehouse-role", "traffic", "powergrid", "reward", "role", "hmf", "distributed")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_files(root, manifest, size_key="size"):
    root = Path(root).resolve()
    entries = manifest.get("files", [])
    if not entries:
        raise ValueError("Empty integrity manifest")
    seen = set()
    for entry in entries:
        name = entry["path"]
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name or name.casefold() in seen:
            raise ValueError("Unsafe or duplicate manifest path: " + name)
        seen.add(name.casefold())
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Missing or escaping manifest file: " + name)
        if path.stat().st_size != entry[size_key] or file_hash(path) != entry["sha256"]:
            raise ValueError("Integrity check failed: " + name)


def stable_version(value):
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", value)
    if not match:
        raise ValueError("A stable engineering core version is required: " + value)
    return tuple(map(int, match.groups()))


def missing_lock_lines(text, installed_version=None):
    """Retain installed dependencies and tooling, install only missing locked wheels.

    The core and its extras are supplied by the new wheel, never by an old lock.
    pip check and current wheel requirements validate compatibility separately.
    """
    installed_version = installed_version or metadata.version
    result = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z0-9_.-]+)(?:\[[^]]+\])?==[^\s]+\s+--hash=sha256:[0-9a-fA-F]{64}(?:\s+--hash=sha256:[0-9a-fA-F]{64})*$", stripped)
        if not match:
            raise ValueError("Expected a pinned hashed local requirement: " + stripped)
        name = re.sub(r"[-_.]+", "-", match.group(1)).lower()
        if name == "unified-marl":
            continue
        try:
            installed_version(name)
        except metadata.PackageNotFoundError:
            result.append(stripped)
    return result


def check_requirements(feature):
    # packaging may be absent from the base bundle; pip always carries its parser.
    try:
        from packaging.requirements import Requirement
    except ImportError:
        from pip._vendor.packaging.requirements import Requirement
    extras = {"warehouse": ["hmf"], "warehouse-role": ["hmf", "role"],
              "powergrid": ["powergrid"]}.get(feature, [feature])
    for text in metadata.requires("unified-marl") or []:
        req = Requirement(text)
        if req.marker and not any(req.marker.evaluate({"extra": extra}) for extra in ["", *extras]):
            continue
        current = metadata.version(req.name)
        if req.specifier and not req.specifier.contains(current, prereleases=True):
            raise RuntimeError(f"Feature {feature} needs {req}; installed {current}. Dependency compatibility must be resolved.")


def install_features(bundle, feature, dry_run=False):
    bundle = Path(bundle).expanduser().resolve()
    if Path(sys.prefix).resolve() != (bundle / ".venv").resolve() or sys.prefix == sys.base_prefix:
        raise RuntimeError("Use this bundle's existing .venv Python")
    current = metadata.version("unified-marl")
    if stable_version(current) < (0, 8, 2):
        raise RuntimeError("Apply the 0.8.2 update before using the feature-only installer")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    verify_files(bundle, manifest)
    edition = manifest.get("edition", "")
    if edition.startswith("linux-"):
        if sys.platform != "linux":
            raise RuntimeError("Linux bundle requires Linux")
        lock_names = ["requirements-core-linux.txt"]
        selected = {"warehouse": ["hmf"], "warehouse-role": ["hmf", "role"], "powergrid": ["hmf"]}.get(feature, [])
        if feature in {"reward", "role", "hmf", "distributed"}:
            selected = [feature]
        lock_names += [f"requirements-{item}-linux.txt" for item in selected]
    elif edition.startswith("windows-"):
        if sys.platform != "win32":
            raise RuntimeError("Windows bundle requires Windows")
        lock_names = ["requirements-lock.txt"]
    else:
        raise RuntimeError("Unknown engineering bundle edition")
    lines = []
    for name in lock_names:
        lines.extend(missing_lock_lines((bundle / name).read_text(encoding="utf-8")))
    lines = list(dict.fromkeys(lines))
    network = None
    if feature == "powergrid":
        network = bundle / "requirements-powergrid.txt"
    elif edition.startswith("linux-") and feature in {"reward", "role", "hmf", "distributed"}:
        candidate = bundle / f"requirements-network-{feature}.txt"
        if candidate.is_file():
            network = candidate
    if network and not network.is_file():
        raise FileNotFoundError(network)
    plan = {"core_version_preserved": current, "feature": feature,
            "missing_local_requirements": lines, "network_requirements": str(network) if network else None}
    if dry_run:
        print(json.dumps(dict(plan, dry_run=True), ensure_ascii=False, indent=2))
        return plan
    env = {k: v for k, v in os.environ.items() if not k.startswith("PIP_")}
    env.update(PIP_CONFIG_FILE=os.devnull, PIP_NO_CACHE_DIR="1", PIP_NO_INPUT="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
    if not (bundle / ".unified-marl-hotfix").resolve().is_relative_to(bundle):
        raise ValueError("Feature work directory escapes the bundle")
    with tempfile.TemporaryDirectory(prefix="feature-", dir=bundle / ".unified-marl-hotfix") as temp:
        temp = Path(temp)
        constraint = temp / "preserve-core.txt"
        constraint.write_text(f"unified-marl=={current}\n", encoding="utf-8")
        pip = [sys.executable, "-X", "utf8", "-I", "-m", "pip"]
        if lines:
            local = temp / "missing-lock.txt"
            local.write_text("\n".join(lines) + "\n", encoding="utf-8")
            subprocess.run([*pip, "install", "--no-index", "--no-cache-dir", "--no-deps", "--only-binary=:all:",
                            "--find-links", str(bundle / "wheelhouse"), "--require-hashes", "-r", str(local)], check=True, env=env)
        if network:
            subprocess.run([*pip, "install", "--no-cache-dir", "--index-url", "https://pypi.org/simple",
                            "--find-links", str(bundle / "wheelhouse"), "--constraint", str(constraint),
                            "-r", str(network)], check=True, env=env)
        if metadata.version("unified-marl") != current:
            raise RuntimeError("Core version changed unexpectedly during feature installation")
        check_requirements(feature)
        subprocess.run([*pip, "check"], check=True, env=env)
        subprocess.run([sys.executable, "-X", "utf8", "-I", "-c", "import unified_marl,sys; assert unified_marl.__version__==sys.argv[1]; print(unified_marl.__version__)", current], check=True, env=env)
    if feature == "traffic":
        print("CityFlow engine and traffic assets remain external; this command does not build or run a simulator.")
    print(json.dumps(dict(plan, status="FEATURE_INSTALLED", core_version_preserved=current), ensure_ascii=False))
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--feature", "--Feature", choices=FEATURES, default="core")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    install_features(args.bundle, args.feature, args.dry_run)


if __name__ == "__main__":
    main()
