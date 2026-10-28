"""Prepare immutable tested assets plus current docs/source for the 0.8.2 release."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
TAG = "v0.8.2-engineering.1"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path):
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": digest(path)}


def main():
    payload = ROOT / "delivery/hotfix-release/0.8.2"
    output = ROOT / "dist/releases" / TAG
    if payload.exists() or output.exists():
        raise FileExistsError("Release preparation already exists; refusing to overwrite")
    report = json.loads((ROOT / "docs/evidence/release-082/results-round3-summary.json").read_text(encoding="utf-8"))
    zip_name = "unified-marl-0.8.2-engineering-hotfix.zip"
    wheel_name = "unified_marl-0.8.2-py3-none-any.whl"
    assert digest(ROOT / "dist" / zip_name) == report["environment"]["patch_zip_sha256"].split()[0]
    assert digest(ROOT / "dist" / wheel_name) == report["environment"]["wheel_sha256"]
    # Never rebuild the wheel/patch tested by the user. Rebuild only the source
    # archive to include current documentation and standalone diagnostic tools.
    output.mkdir(parents=True)
    subprocess.run([sys.executable, "-I", "-c", "from setuptools.build_meta import build_sdist; build_sdist(" + repr(str(output)) + ")"], cwd=ROOT, check=True)
    for name in [zip_name, zip_name + ".sha256", wheel_name]:
        shutil.copyfile(ROOT / "dist" / name, output / name)
    for name in ["ENGINEERING_RELEASE_082", "VALIDATION_REPORT_082"]:
        shutil.copyfile(ROOT / "docs" / (name + ".md"), output / (name + ".zh-CN.md"))
    supporting = output / "unified-marl-0.8.2-supporting-files.zip"
    items = {
        "ENGINEERING_RELEASE_082.md": ROOT / "docs/ENGINEERING_RELEASE_082.md",
        "VALIDATION_REPORT_082.md": ROOT / "docs/VALIDATION_REPORT_082.md",
        "ROUND3_REPORT_SUMMARY.md": ROOT / "docs/evidence/release-082/ROUND3_REPORT_SUMMARY.md",
        "results-round3-summary.json": ROOT / "docs/evidence/release-082/results-round3-summary.json",
        "check_powergrid_seed.py": ROOT / "tools/check_powergrid_seed.py",
        "check_feature_install_branch.py": ROOT / "tools/check_feature_install_branch.py",
        "missing-dependency-fixture.json": ROOT / "docs/evidence/engineering-hotfix-082/missing-dependency-fixture.json",
        "unit-tests.xml": ROOT / "docs/evidence/engineering-hotfix-082/unit-tests.xml",
        "publication-tests.xml": ROOT / "docs/evidence/engineering-hotfix-082/publication-tests.xml",
    }
    with zipfile.ZipFile(supporting, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, path in items.items():
            archive.write(path, name)
    package_root = ROOT / "src/unified_marl"
    with zipfile.ZipFile(output / wheel_name) as wheel, tarfile.open(output / "unified_marl-0.8.2.tar.gz") as source:
        count = 0
        for path in package_root.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix in {".py", ".yaml", ".json", ".csv"}:
                relative = path.relative_to(package_root).as_posix()
                assert wheel.read("unified_marl/" + relative) == path.read_bytes()
                assert source.extractfile("unified_marl-0.8.2/src/unified_marl/" + relative).read() == path.read_bytes()
                count += 1
    before_manifest = [record(path) for path in sorted(output.iterdir()) if path.is_file()]
    (output / "DELIVERY_MANIFEST.json").write_text(json.dumps({
        "version": "0.8.2", "tag": TAG, "wheel_patch_unchanged_from_user_round3": True,
        "runtime_files_matched": count, "assets": before_manifest,
        "linux_acceptance_source": "user report, scoped scenarios; not local reproduction",
        "seed_observation": "pending targeted diagnostic", "clean_linux_powergrid_install_verified": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    records = [record(path) for path in sorted(output.iterdir()) if path.is_file()]
    (output / "SHA256SUMS.txt").write_text("\n".join(item["sha256"] + "  " + item["name"] for item in records) + "\n", encoding="ascii")
    records = [record(path) for path in sorted(output.iterdir()) if path.is_file()]
    (payload / "assets").mkdir(parents=True)
    for path in output.iterdir():
        if path.is_file():
            shutil.copyfile(path, payload / "assets" / path.name)
    shutil.copyfile(ROOT / "docs/releases" / (TAG + ".md"), payload / "release-notes.md")
    (payload / "release-plan.json").write_text(json.dumps({
        "repository": "saimeeee/unified-marl", "tag": TAG, "assets": records,
        "runtime_files_matched": count, "published": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (payload.parent / ".gitattributes").write_text("* -text\n", encoding="ascii")
    print(json.dumps({"status": "PREPARED_NOT_PUBLISHED", "assets": len(records),
                      "output": str(output), "runtime_files_matched": count}, ensure_ascii=False))


if __name__ == "__main__":
    main()
