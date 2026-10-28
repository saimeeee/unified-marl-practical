"""Apply the 0.8.2 wheel and a verified, backed-up installer update."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys


VERSION = "0.8.2"


def verify_patch(root, manifest):
    entries = manifest.get("files", [])
    if not entries:
        raise ValueError("Empty patch manifest")
    seen = set()
    for entry in entries:
        name = entry["path"]
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name or name.casefold() in seen:
            raise ValueError("Unsafe patch path: " + name)
        seen.add(name.casefold())
        file = (root / name).resolve()
        if not file.is_relative_to(root.resolve()) or not file.is_file():
            raise ValueError("Missing or escaping patch file: " + name)
        if file.stat().st_size != entry["bytes"] or hashlib.sha256(file.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Patch integrity check failed: " + name)


def update_installers(bundle, patch_root, patch_manifest):
    helper = runpy.run_path(str(patch_root / "feature_install.py"))
    verify_files, file_hash = helper["verify_files"], helper["file_hash"]
    manifest_path = bundle / "manifest.json"
    original_manifest = manifest_path.read_bytes()
    manifest = json.loads(original_manifest)
    verify_files(bundle, manifest)
    edition = manifest.get("edition", "")
    if edition.startswith("linux-"):
        target, source = "linux/install.sh", "linux_install.sh"
    elif edition.startswith("windows-"):
        target, source = "windows/install.ps1", "windows_install.ps1"
    else:
        raise RuntimeError("Expected an existing Windows/Linux engineering bundle")
    replacements = {target: source, f".unified-marl-hotfix/{VERSION}/feature_install.py": "feature_install.py"}
    for name in replacements:
        if not (bundle / name).resolve().is_relative_to(bundle.resolve()):
            raise ValueError("Installer target escapes the bundle")
    if manifest.get("hotfix_version") == VERSION:
        for name, source_name in replacements.items():
            if (bundle / name).read_bytes() != (patch_root / source_name).read_bytes():
                raise RuntimeError("The same hotfix version is installed with different installer bytes")
        return
    backup = bundle / ".unified-marl-hotfix/backups/pre-0.8.2"
    if not backup.resolve().is_relative_to(bundle.resolve()):
        raise ValueError("Backup directory escapes the bundle")
    backup.mkdir(parents=True, exist_ok=True)
    original_installer = (bundle / target).read_bytes()
    for name, data in {"manifest.json": original_manifest, Path(target).name: original_installer}.items():
        file = backup / name
        if file.exists() and file.read_bytes() != data:
            raise FileExistsError("Different prior update backup already exists: " + str(file))
        file.write_bytes(data)
    previous = {name: (bundle / name).read_bytes() if (bundle / name).exists() else None for name in replacements}
    try:
        for name, source_name in replacements.items():
            file = bundle / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes((patch_root / source_name).read_bytes())
        entries = {entry["path"]: entry for entry in manifest["files"]}
        changed = [*replacements, *[p.relative_to(bundle).as_posix() for p in backup.iterdir()]]
        for name in changed:
            file = bundle / name
            entries[name] = {"path": name, "size": file.stat().st_size, "sha256": file_hash(file)}
        manifest["files"] = sorted(entries.values(), key=lambda entry: entry["path"])
        manifest["base_bundle_version"] = manifest.get("base_bundle_version", manifest.get("version"))
        manifest["version"] = manifest["hotfix_version"] = VERSION
        manifest.setdefault("applied_hotfixes", []).append({
            "version": VERSION, "fixes": ["F1", "F2", "N1"],
            "previous_manifest_sha256": hashlib.sha256(original_manifest).hexdigest(),
            "patch_manifest_sha256": hashlib.sha256((patch_root / "manifest.json").read_bytes()).hexdigest(),
            "backup": backup.relative_to(bundle).as_posix(), "updated_files": list(replacements),
        })
        verify_files(bundle, manifest)
        temporary = manifest_path.with_suffix(".json.hotfix-tmp")
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, manifest_path)
    except BaseException:
        for name, data in previous.items():
            path = bundle / name
            if data is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(data)
        manifest_path.write_bytes(original_manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    bundle = args.bundle.expanduser().resolve()
    venv = bundle / ".venv"
    if Path(sys.prefix).resolve() != venv.resolve() or sys.prefix == sys.base_prefix:
        raise RuntimeError("Run this script with the target bundle .venv Python, not the system Python")
    installed = importlib.metadata.version("unified-marl")
    if installed not in {"0.8.0", "0.8.1", VERSION}:
        raise RuntimeError(f"Expected an existing 0.8.0/0.8.1/{VERSION} engineering installation; found {installed}")
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != VERSION or set(manifest.get("fixes", [])) != {"F1", "F2", "N1"}:
        raise ValueError("Expected the 0.8.2 F1/F2/N1 update manifest")
    verify_patch(root, manifest)
    helper = runpy.run_path(str(root / "feature_install.py"))
    helper["verify_files"](bundle, json.loads((bundle / "manifest.json").read_text(encoding="utf-8")))
    wheel = root / f"unified_marl-{VERSION}-py3-none-any.whl"
    subprocess.run([sys.executable, "-X", "utf8", "-I", "-m", "pip", "install", "--no-index", "--no-deps",
                    "--no-cache-dir", "--force-reinstall", str(wheel)], check=True)
    subprocess.run([sys.executable, "-X", "utf8", "-I", "-m", "pip", "check"], check=True)
    subprocess.run([sys.executable, "-X", "utf8", "-I", "-c",
                    "import unified_marl; assert unified_marl.__version__=='0.8.2'; "
                    "print('Installed:', unified_marl.__version__, unified_marl.__file__)"], check=True)
    update_installers(bundle, root, manifest)
    print("Updated core and feature installer to 0.8.2. Existing install.sh/install.ps1 now preserve the core version.")
    print("Original installers and manifest are retained under .unified-marl-hotfix/backups/pre-0.8.2.")
    print("No environment, training or simulation was started.")


if __name__ == "__main__":
    main()
