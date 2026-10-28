"""Build a small, offline core update for existing 0.8.0 engineering bundles."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib
import zipfile


ROOT = Path(__file__).resolve().parents[1]
VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
STEM = f"unified-marl-{VERSION}-engineering-hotfix"


def main():
    dist = ROOT / "dist"
    destination = dist / STEM
    archive = dist / (STEM + ".zip")
    if destination.exists() or archive.exists():
        raise FileExistsError("Refusing to overwrite an existing hotfix delivery")
    subprocess.run([sys.executable, "-I", "-c",
                    "from setuptools.build_meta import build_wheel, build_sdist; "
                    "build_wheel('dist'); build_sdist('dist')"], cwd=ROOT, check=True)
    destination.mkdir()
    mappings = {
        f"unified_marl-{VERSION}-py3-none-any.whl": dist / f"unified_marl-{VERSION}-py3-none-any.whl",
        "install_patch.py": ROOT / "delivery/hotfix/install_patch.py",
        "feature_install.py": ROOT / "delivery/hotfix/feature_install.py",
        "linux_install.sh": ROOT / "delivery/hotfix/linux_install.sh",
        "windows_install.ps1": ROOT / "delivery/hotfix/windows_install.ps1",
        "round2_warehouse.py": ROOT / "examples/applications/round2_warehouse.py",
        "round3_powergrid_checks.py": ROOT / "examples/applications/round3_powergrid_checks.py",
        "README.zh-CN.md": ROOT / "docs/ENGINEERING_HOTFIX_082.md",
        "LINUX_ROUND3_PROMPTS.md": ROOT / "docs/LINUX_ROUND3_PROMPTS.md",
        "ENGINEERING_HOTFIX_082.md": ROOT / "docs/ENGINEERING_HOTFIX_082.md",
        "unit-tests.xml": ROOT / "docs/evidence/engineering-hotfix-082/unit-tests.xml",
    }
    # Include external evidence only when present; never claim it is a local test.
    report = ROOT / "docs/evidence/engineering-hotfix-082/linux-round2-user-report.md"
    if report.exists():
        mappings["linux-round2-user-report.md"] = report
    entries = []
    for name, source in mappings.items():
        shutil.copyfile(source, destination / name)
        data = (destination / name).read_bytes()
        entries.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    (destination / "manifest.json").write_text(json.dumps({
        "version": VERSION, "base_installation": "0.8.0 engineering with required feature dependencies; 0.8.1 accepted",
        "fixes": ["F1", "F2", "N1"],
        "fix_details": {"F1": "numeric elementwise PV scaling and aligned finite profile validation",
                        "F2": "control_transitions_v2 excludes reset warmup; requires new PowerGrid training run",
                        "N1": "backed-up verified bundle installer update; feature-only installation preserves core"},
        "files": entries, "linux_hotfix_runtime_test": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as output:
        for path in sorted(destination.iterdir()):
            output.write(path, arcname=path.name)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (dist / (STEM + ".zip.sha256")).write_text(f"{digest}  {archive.name}\n", encoding="ascii")
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size,
                      "sha256": digest, "files": len(entries) + 1}, ensure_ascii=False))


if __name__ == "__main__":
    main()
