"""Final artifact, source-integrity, installed-package and evidence consistency."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import zipfile

root=Path(__file__).resolve().parents[1]
evidence=root/"docs/evidence/stage5"
summary=json.loads((evidence/"summary.json").read_text(encoding="utf-8"))
installed=json.loads((evidence/"installed_validation.json").read_text(encoding="utf-8"))
suite=ET.parse(evidence/"tests.xml").getroot().find("testsuite")
assert int(suite.attrib["tests"])==summary["tests"]==149
assert int(suite.attrib["failures"])==int(suite.attrib["errors"])==0
assert installed["version"]=="0.5.0" and "site-packages" in installed["package"]
assert Path(installed["cwd"]).resolve()!=root
assert len(installed["algorithms"])==7
assert all(item["result"]["updates"]==2 and item["result"]["parameter_delta_l2"]>0 and item["resume"]["updates"]==3 for item in installed["algorithms"])
assert len(installed["cli"])==2 and all(item["returncode"]==0 for item in installed["cli"])
assert len(installed["regressions"])==5 and all(item["result"]["parameter_delta_l2"]>0 for item in installed["regressions"])
assert len(installed["real_environments"])==4 and all(item["status"]=="BLOCKED" for item in installed["real_environments"])
assert installed["lifecycle"]["all_closed"] and installed["lifecycle"]["children_before"]==installed["lifecycle"]["children_after"]
parity=[json.loads(path.read_text(encoding="utf-8")) for path in evidence.glob("parity-*.json")]
assert len(parity)==13 and all(item["result"]=="PASS" and item["parameter_delta_l2"]>0 for item in parity)
errors=[value for item in parity for key,value in item.items() if "error" in key and isinstance(value,(float,int))]
assert errors and max(errors)==0
assert json.loads((evidence/"time_semantics.json").read_text(encoding="utf-8"))["source_arrival_buffer_equivalence"]
assert json.loads((evidence/"lifecycle.json").read_text(encoding="utf-8"))["error_cleanup"]
for name in ["WORK_ORDER.md","README.md","docs/INTEGRATION_SPEC.md","docs/MIGRATION_MATRIX.csv","docs/PLAN.md","docs/STATUS.md",
             "docs/STAGE4_REPORT.md","docs/STAGE5_REPORT.md","docs/SADMA_VERSION_SELECTION.md"]:
    assert (root/name).is_file(),name
with (root/"docs/MIGRATION_MATRIX.csv").open(encoding="utf-8-sig",newline="") as file: rows=list(csv.DictReader(file))
assert len(rows)==len({row["id"] for row in rows})==summary["matrix_rows"]==369
for row in rows:
    assert (root.parent/row["source_path"]).exists(),row["source_path"]
    if "sadma" in row["source_project"]:
        assert row["policy_effect_verified"]=="NO" and row["real_environment_verified"]=="NO"
        if row["planned_stage"]=="6": assert row["migration_status"]=="DEFERRED_STAGE6_NOT_EXECUTED"
source_count=0
for name in ["source_manifest.json","supplemental_manifest.json"]:
    for row in json.loads((root/"docs/evidence"/name).read_text(encoding="utf-8")):
        assert hashlib.sha256((root.parent/row["path"]).read_bytes()).hexdigest()==row["sha256"],row["path"]
        source_count+=1
assert source_count==1228
audit=json.loads((evidence/"audit.json").read_text(encoding="utf-8"))
assert len(audit["source_hashes"])==192
for row in audit["source_hashes"]:
    assert hashlib.sha256((root.parent/row["source"]).read_bytes()).hexdigest()==row["sha256"],row["source"]
assert audit["compared_to_yi_open"]["yi_zmq"]==[]
for source in audit["sources"].values(): assert (Path(source)/"policy/maddpg/maddpg_policy.py").stat().st_size==0
wheel=root/"dist/unified_marl-0.5.0-py3-none-any.whl"
installed_root=root/".venv-check/Lib/site-packages/unified_marl"
runtime_count=0
with zipfile.ZipFile(wheel) as archive:
    for path in sorted((root/"src/unified_marl").rglob("*")):
        if path.suffix not in {".py",".yaml"}: continue
        relative=path.relative_to(root/"src/unified_marl")
        payload=path.read_bytes()
        if path.suffix==".py": ast.parse(payload.decode("utf-8"),filename=str(path))
        assert archive.read("unified_marl/"+relative.as_posix())==payload,relative
        assert (installed_root/relative).read_bytes()==payload,relative
        runtime_count+=1
    for profile in ["wang","yi_open","yi_closed"]:
        for filename in ["default.yaml","qmix.yaml","ppo.yaml"]:
            prefix="unified_marl/config/sadma/"+profile+"/"+filename
            assert archive.read(prefix)==(Path(audit["sources"][profile])/"config"/(filename if filename=="default.yaml" else "alg_config/"+filename)).read_bytes()
reference_roots=[root.parent/"pymarl-master",root.parent/"marl平台代码"]
bytecode=sum(len(list(path.rglob("*.cpython-312.pyc"))) for path in reference_roots)
assert bytecode==0,bytecode
dependencies={}
for name in [".venv",".venv-check"]:
    executable=root/name/"Scripts/python.exe"
    check=subprocess.run([str(executable),"-B","-m","pip","check"],text=True,capture_output=True)
    packages=subprocess.run([str(executable),"-B","-m","pip","list","--format=json"],text=True,capture_output=True)
    assert check.returncode==packages.returncode==0,(name,check.stdout,check.stderr)
    dependencies[name]=dict(python=str(executable),pip_check=dict(returncode=check.returncode,stdout=check.stdout,stderr=check.stderr),packages=json.loads(packages.stdout))
(evidence/"dependencies.json").write_text(json.dumps(dependencies,ensure_ascii=False,indent=2),encoding="utf-8")
result=dict(result="PASS",version="0.5.0",tests=149,source_parity_reports=13,source_parity_max_error=max(errors),installed_configs=7,
    source_hash_records_unchanged=source_count,sadma_compared_source_hash_records_unchanged=192,matrix_rows=len(rows),
    runtime_source_wheel_installed_files_identical=runtime_count,reference_python312_bytecode_files=bytecode,
    wheel_bytes=wheel.stat().st_size,wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
    two_project_env_pip_checks="PASS",representative_regressions=5,real_environments="BLOCKED",continuous_actions="BLOCKED_SCHEMA_AND_ASSETS",
    policy_effectiveness="NOT_VALIDATED",distributed_stage6="NOT_EXECUTED",stopped_after_stage=5)
(evidence/"final_verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(result,ensure_ascii=False))
