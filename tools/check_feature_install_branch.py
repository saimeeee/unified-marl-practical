"""Exercise missing-local-wheel installation in a disposable venv, without a simulator.

The synthetic probe distribution is an installer fixture, not a PowerGrid
dependency. This does not claim clean Linux/PowerGrid dependency acceptance.
"""
import argparse
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--dependency-site", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    bundle = args.work_dir.resolve()
    bundle.mkdir(parents=True, exist_ok=False)
    wheelhouse = bundle / "wheelhouse"
    wheelhouse.mkdir()
    probe = wheelhouse / "unified_marl_installer_probe-0.0.1-py3-none-any.whl"
    info = "unified_marl_installer_probe-0.0.1.dist-info"
    files = {
        "unified_marl_installer_probe.py": b"VALUE = 1\n",
        info + "/METADATA": b"Metadata-Version: 2.1\nName: unified-marl-installer-probe\nVersion: 0.0.1\n\n",
        info + "/WHEEL": b"Wheel-Version: 1.0\nGenerator: installer-fixture\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    rows = []
    for name, data in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        rows.append([name, "sha256=" + digest, str(len(data))])
    rows.append([info + "/RECORD", "", ""])
    record = io.StringIO(newline="")
    csv.writer(record).writerows(rows)
    files[info + "/RECORD"] = record.getvalue().encode()
    with zipfile.ZipFile(probe, "w", zipfile.ZIP_DEFLATED) as output:
        for name, data in files.items():
            output.writestr(name, data)
    helper = bundle / ".unified-marl-hotfix/0.8.2/feature_install.py"
    helper.parent.mkdir(parents=True)
    shutil.copyfile(root / "delivery/hotfix/feature_install.py", helper)
    windows = sys.platform == "win32"
    lock = bundle / ("requirements-lock.txt" if windows else "requirements-core-linux.txt")
    lock.write_text("unified-marl==0.8.0 --hash=sha256:" + "0" * 64 + "\n"
                    + "unified-marl-installer-probe==0.0.1 --hash=sha256:"
                    + hashlib.sha256(probe.read_bytes()).hexdigest() + "\n", encoding="utf-8")
    entries = [{"path": p.relative_to(bundle).as_posix(), "size": p.stat().st_size,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in [probe, helper, lock]]
    (bundle / "manifest.json").write_text(json.dumps({
        "edition": "windows-x64-cpu-py312-engineering" if windows else "linux-x86_64-cpu-py312-engineering",
        "version": "0.8.2", "files": entries,
    }), encoding="utf-8")
    subprocess.run([sys.executable, "-X", "utf8", "-m", "venv", str(bundle / ".venv")], check=True)
    python = bundle / (".venv/Scripts/python.exe" if windows else ".venv/bin/python")
    site = bundle / (".venv/Lib/site-packages" if windows else f".venv/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")
    # Explicitly reuse only dependencies; the core wheel is installed locally.
    (site / "shared_test_dependencies.pth").write_text(str(args.dependency_site.resolve()) + "\n", encoding="utf-8")
    subprocess.run([str(python), "-X", "utf8", "-I", "-m", "pip", "install", "--no-index", "--no-deps",
                    "--no-cache-dir", str(args.wheel.resolve())], check=True)
    subprocess.run([str(python), "-X", "utf8", "-I", "-c",
                    "import importlib.util; assert importlib.util.find_spec('unified_marl_installer_probe') is None"], check=True)
    runs = []
    for index in range(2):
        result = subprocess.run([str(python), "-X", "utf8", "-I", str(helper), "--bundle", str(bundle), "--feature", "core"],
                                capture_output=True, text=True, encoding="utf-8", timeout=180)
        (bundle / f"run-{index}.stdout.log").write_text(result.stdout, encoding="utf-8")
        (bundle / f"run-{index}.stderr.log").write_text(result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"Installer fixture failed ({result.returncode}): {result.stderr}")
        runs.append(json.loads(next(line for line in reversed(result.stdout.splitlines()) if line.startswith("{"))))
    assert len(runs[0]["missing_local_requirements"]) == 1
    assert runs[1]["missing_local_requirements"] == []
    assert all(item["core_version_preserved"] == "0.8.2" for item in runs)
    subprocess.run([str(python), "-X", "utf8", "-I", "-c",
                    "import unified_marl,unified_marl_installer_probe; assert unified_marl.__version__=='0.8.2'; assert unified_marl_installer_probe.VALUE==1"], check=True)
    report = {"status": "PASS", "platform": sys.platform, "missing_probe_wheel_installed": True,
              "repeat_run_no_missing_dependency": True, "core_version_preserved": "0.8.2",
              "scope": "synthetic installer branch fixture with read-only dependency reuse; not clean Linux PowerGrid installation",
              "linux_powergrid_clean_install_verified": False, "training_run": False, "simulation_run": False}
    (bundle / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
