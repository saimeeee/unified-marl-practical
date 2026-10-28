"""Final evidence/source/wheel validation; no implicit remote or simulator claim."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import zipfile

root=Path(__file__).resolve().parents[1]
evidence=root/"docs/evidence/stage6"
summary=json.loads((evidence/"summary.json").read_text(encoding="utf-8"))
installed=json.loads((evidence/"installed_validation.json").read_text(encoding="utf-8"))
suite=ET.parse(evidence/"tests.xml").getroot().find("testsuite")
assert int(suite.attrib["tests"])==summary["tests"]==196
assert int(suite.attrib["failures"])==int(suite.attrib["errors"])==0
assert installed["version"]=="0.6.0" and "site-packages" in installed["package"] and Path(installed["cwd"]).resolve()!=root
assert installed["pyzmq"]=="25.1.2" and installed["libzmq"]=="4.3.4"
assert len(installed["runs"])==9
for row in installed["runs"]:
    assert row["result"]["updates"]==2 and row["result"]["parameter_delta_l2"]>0 and row["ports_rebound"]
    report=row["execution"]
    assert report["status"]=="PASS" and report["cross_machine_verified"] is False
    assert report["lifecycle"]["forced"]==[] and all(not child["alive"] and child["exitcode"]==0 for child in report["lifecycle"]["processes"])
    counters=report["counters"]
    assert counters["env_steps"]==counters["completed_steps"]==4*counters["episodes"]
    assert counters["inference_calls"]==5*counters["episodes"]
    assert sum(item["calls"] for item in report["inference_stop_reports"])==counters["inference_calls"]
    assert report["samplers_closed"]>0 and report["partial_confirmed_steps"]==0
    assert len({ack["digest"] for ack in report["sync_acknowledgements"]})>=3
    for version in {ack["version"] for ack in report["sync_acknowledgements"]}:
        assert len({ack["digest"] for ack in report["sync_acknowledgements"] if ack["version"]==version})==1
    events=[json.loads(line) for line in (Path(row["result"]["output_dir"])/"execution-events.jsonl").read_text(encoding="utf-8").splitlines()]
    if row["mode"]=="sync": assert all(event["min_version"]==event["max_version"] for event in events if event["kind"]=="EPISODE")
assert installed["resume"]["updates"]==3 and installed["resume"]["parameter_delta_l2"]>0
assert len(installed["cli"])==2 and all(row["returncode"]==0 for row in installed["cli"])
assert len(installed["failures"])==2 and all(row["report"]["status"]=="FAILED" and row["ports_rebound"] for row in installed["failures"])
assert len(installed["local_regressions"])==6 and all(row["result"]["parameter_delta_l2"]>0 for row in installed["local_regressions"])
assert installed["child_processes_before"]==installed["child_processes_after"]
assert not installed["cross_machine_verified"] and not installed["real_environments_verified"] and not installed["gpu_verified"]
for filename in ["separate-roles.json","fault-raise.json","fault-stall.json","connection-timeout.json","startup-failure.json","multi-inference-resume.json"]:
    assert json.loads((evidence/filename).read_text(encoding="utf-8"))["result"]=="PASS"
for kind in ["qmix","ppo"]:
    assert json.loads((evidence/f"inference-parity-{kind}.json").read_text(encoding="utf-8"))["output_max_abs_error"]==0
vtrace=json.loads((evidence/"async-vtrace-source-parity.json").read_text(encoding="utf-8"))
assert vtrace["loss_abs_error"]==vtrace["updated_actor_max_abs_error"]==0 and vtrace["old_log_probabilities_preserved"]
for name in ["WORK_ORDER.md","README.md","docs/INTEGRATION_SPEC.md","docs/PLAN.md","docs/STATUS.md","docs/STAGE6_REPORT.md","docs/ZMQ_PROTOCOL.md",
             "examples/stage6_zmq.py","examples/zmq_managed.yaml","examples/zmq_roles.yaml"]:
    assert (root/name).is_file(),name
with (root/"docs/MIGRATION_MATRIX.csv").open(encoding="utf-8-sig",newline="") as file: rows=list(csv.DictReader(file))
assert len(rows)==len({row["id"] for row in rows})==summary["matrix_rows"]==379
for row in rows:
    assert (root.parent/row["source_path"]).exists(),row["source_path"]
    if row["planned_stage"]=="6":
        assert row["cross_machine_verified"]==row["real_environment_verified"]==row["policy_effect_verified"]=="NO"
source_count=0
for filename in ["source_manifest.json","supplemental_manifest.json"]:
    for row in json.loads((root/"docs/evidence"/filename).read_text(encoding="utf-8")):
        assert hashlib.sha256((root.parent/row["path"]).read_bytes()).hexdigest()==row["sha256"],row["path"]
        source_count+=1
assert source_count==1228
audit=json.loads((evidence/"audit.json").read_text(encoding="utf-8"))
assert len(audit["sources"])==36
for row in audit["sources"]:
    assert hashlib.sha256((root.parent/row["source"]).read_bytes()).hexdigest()==row["sha256"]
wheel=root/"dist/unified_marl-0.6.0-py3-none-any.whl"
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
    metadata=archive.read("unified_marl-0.6.0.dist-info/METADATA").decode()
    assert "Provides-Extra: distributed" in metadata and "pyzmq==25.1.2" in metadata
bytecode=sum(len(list(path.rglob("*.cpython-312.pyc"))) for path in [root.parent/"pymarl-master",root.parent/"marl平台代码"])
assert bytecode==0,bytecode
dependencies={}
for name in [".venv",".venv-check"]:
    executable=root/name/"Scripts/python.exe"
    check=subprocess.run([str(executable),"-B","-m","pip","check"],capture_output=True,text=True)
    packages=subprocess.run([str(executable),"-B","-m","pip","list","--format=json"],capture_output=True,text=True)
    assert check.returncode==packages.returncode==0,(name,check.stdout,check.stderr)
    dependencies[name]=dict(python=str(executable),pip_check=dict(returncode=check.returncode,stdout=check.stdout,stderr=check.stderr),packages=json.loads(packages.stdout))
(evidence/"dependencies.json").write_text(json.dumps(dependencies,ensure_ascii=False,indent=2),encoding="utf-8")
result=dict(result="PASS",version="0.6.0",tests=196,new_execution_tests=47,installed_training_cases=9,installed_cli_cases=2,local_regressions=6,
    separate_roles_loopback=True,worker_failure_stall_startup_timeout_and_port_release="PASS",source_parity_reports=3,source_parity_max_error=0,
    source_hash_records_unchanged=source_count,communication_source_hashes_unchanged=36,matrix_rows=len(rows),
    runtime_source_wheel_installed_files_identical=runtime_count,reference_python312_bytecode_files=bytecode,
    wheel_bytes=wheel.stat().st_size,wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),pip_checks="BOTH_PROJECT_ENVS_PASS",
    cross_machine_verified=False,real_environments_verified=False,gpu_verified=False,policy_effectiveness="NOT_VALIDATED",stopped_after_stage=6)
(evidence/"final_verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(result,ensure_ascii=False))
