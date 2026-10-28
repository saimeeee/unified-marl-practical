"""Final artifact/evidence consistency check; no simulator claim or source edits."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

root = Path(__file__).resolve().parents[1]
evidence = root / "docs/evidence/stage3"
summary = json.loads((evidence / "summary.json").read_text(encoding="utf-8"))
installed = json.loads((evidence / "installed_validation.json").read_text(encoding="utf-8"))
suite = ET.parse(evidence / "tests.xml").getroot().find("testsuite")
assert int(suite.attrib["tests"]) == summary["tests"] == 59
assert int(suite.attrib["failures"]) == int(suite.attrib["errors"]) == 0
assert installed["version"] == "0.3.0" and len(installed["algorithms"]) == 24
assert all(x["train"]["updates"] == 2 and x["train"]["parameter_delta_l2"] > 0 for x in installed["algorithms"])
assert all(x["status"] == "BLOCKED" for x in installed["real_environments"])
for name in ["WORK_ORDER.md", "README.md", "docs/INTEGRATION_SPEC.md", "docs/PLAN.md", "docs/STATUS.md", "docs/STAGE2_REPORT.md", "docs/STAGE3_REPORT.md"]:
    assert (root / name).is_file(), name
with (root / "docs/MIGRATION_MATRIX.csv").open(encoding="utf-8-sig", newline="") as file:
    matrix = list(csv.DictReader(file))
assert len(matrix) == len({r["id"] for r in matrix}) == 350
for row in matrix:
    assert (root.parent / row["source_path"]).exists(), row["source_path"]
checked = 0
for name in ["source_manifest.json", "supplemental_manifest.json"]:
    for row in json.loads((root / "docs/evidence" / name).read_text(encoding="utf-8")):
        assert hashlib.sha256((root.parent / row["path"]).read_bytes()).hexdigest() == row["sha256"]
        checked += 1
wheel = root / "dist/unified_marl-0.3.0-py3-none-any.whl"
installed_root = root / ".venv-check/Lib/site-packages/unified_marl"
runtime_files = 0
with zipfile.ZipFile(wheel) as archive:
    for path in sorted((root / "src/unified_marl").rglob("*")):
        if path.suffix not in {".py", ".yaml"}:
            continue
        relative = path.relative_to(root / "src/unified_marl")
        payload = path.read_bytes()
        if path.suffix == ".py":
            ast.parse(payload.decode("utf-8"), filename=str(path))
        assert archive.read("unified_marl/" + relative.as_posix()) == payload, relative
        assert (installed_root / relative).read_bytes() == payload, relative
        runtime_files += 1
bytecode = sum(len(list(p.rglob("*.cpython-312.pyc"))) for p in [root.parent / "pymarl-master",
    root.parent / "marl平台代码/王思哲/code/large_scale/RGHMF", root.parent / "marl平台代码/王思哲/code/large_scale/hmf_replenishment"])
assert bytecode == 0
result = {"result": "PASS", "tests": summary["tests"], "matrix_rows": len(matrix), "source_hash_records_unchanged": checked,
    "runtime_source_wheel_installed_files_identical": runtime_files, "reference_python312_bytecode_files": bytecode,
    "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(), "wheel_bytes": wheel.stat().st_size,
    "real_simulators": "BLOCKED", "stopped_after_stage": 3}
(evidence / "final_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False))
