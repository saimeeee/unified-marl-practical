"""Validate measured stage-4 artifacts without mutating delivery sources."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import zipfile

root=Path(__file__).resolve().parents[1]
evidence=root/"docs/evidence/stage4"
summary=json.loads((evidence/"summary.json").read_text(encoding="utf-8"))
installed=json.loads((evidence/"installed_validation.json").read_text(encoding="utf-8"))
suite=ET.parse(evidence/"tests.xml").getroot().find("testsuite")
assert int(suite.attrib["tests"])==summary["tests"]==112
assert int(suite.attrib["failures"])==int(suite.attrib["errors"])==0
assert installed["version"]=="0.4.0" and "site-packages" in installed["package"]
assert Path(installed["cwd"]).resolve()!=root
assert len(installed["policies"])==17 and len(installed["reward_models"])==7
for item in installed["policies"]:
    assert item["train"]["updates"]==2 and item["resume"]["updates"]==3
    assert item["train"]["env_steps"]==0 and item["train"]["parameter_delta_l2"]>0
    assert item["resume"]["parameter_delta_l2"]>0
for item in installed["reward_models"]:
    assert item["result"]["parameter_delta_l2"]>0
assert installed["cli"]["returncode"]==installed["cli_reward"]["returncode"]==0
assert installed["qmix_regression"]["train"]["parameter_delta_l2"]>0
assert installed["qmix_regression"]["evaluation"]["episodes"]==2
assert any(x["name"]=="real_smac_evaluation" and x["status"]=="ENVIRONMENT_BLOCKED" for x in installed["blocked"])
parity=[json.loads(path.read_text(encoding="utf-8")) for path in evidence.glob("parity-*.json")]
assert len(parity)==24 and all(item["result"]=="PASS" for item in parity)
errors=[value for item in parity for key,value in item.items() if "error" in key and isinstance(value,(int,float))]
assert errors and max(errors)<=1e-6
network_errors=[value for item in parity for key,value in item.items() if ("network" in key or "prediction" in key) and "error" in key]
assert max(network_errors)==0
datasets=json.loads((evidence/"dataset_batches.json").read_text(encoding="utf-8"))
assert len(datasets)==12 and all(x["status"]=="SMALL_BATCH_READ_PASS" for x in datasets)
for name in ["WORK_ORDER.md","README.md","docs/INTEGRATION_SPEC.md","docs/PLAN.md","docs/STATUS.md",
             "docs/STAGE2_REPORT.md","docs/STAGE3_REPORT.md","docs/STAGE4_REPORT.md"]:
    assert (root/name).is_file(),name
with (root/"docs/MIGRATION_MATRIX.csv").open(encoding="utf-8-sig",newline="") as file:
    matrix=list(csv.DictReader(file))
assert len(matrix)==len({x["id"] for x in matrix})==summary["matrix_rows"]==361
for row in matrix:
    assert (root.parent/row["source_path"]).exists(),row["source_path"]
    if row["source_project"] in {"kou_icq","kou_old","kou_data","kou_tools","kou_reward"}:
        assert row["policy_effect_verified"]=="NO" and row["real_environment_verified"]=="NO"
checked=0
for name in ["source_manifest.json","supplemental_manifest.json"]:
    for row in json.loads((root/"docs/evidence"/name).read_text(encoding="utf-8")):
        assert hashlib.sha256((root.parent/row["path"]).read_bytes()).hexdigest()==row["sha256"],row["path"]
        checked+=1
assert checked==1228
wheel=root/"dist/unified_marl-0.4.0-py3-none-any.whl"
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
    for prefix in ["unified_marl/config/kou/icq/","unified_marl/config/kou/old/"]:
        assert any(name.startswith(prefix) for name in archive.namelist()),prefix
reference_roots=[root.parent/"pymarl-master",root.parent/"marl平台代码/王思哲/code/large_scale/RGHMF",
                 root.parent/"marl平台代码/王思哲/code/large_scale/hmf_replenishment",
                 root.parent/"marl平台代码/寇谦/论文代码及技术文档/论文代码及技术文档/ICQ-MA",
                 root.parent/"marl平台代码/寇谦/论文代码及技术文档/论文代码及技术文档/old_src"]
bytecode=sum(len(list(path.rglob("*.cpython-312.pyc"))) for path in reference_roots)
assert bytecode==0,bytecode
dependencies={}
for name in [".venv",".venv-check"]:
    executable=root/name/"Scripts/python.exe"
    check=subprocess.run([str(executable),"-B","-m","pip","check"],capture_output=True,text=True)
    packages=subprocess.run([str(executable),"-B","-m","pip","list","--format=json"],capture_output=True,text=True)
    assert check.returncode==packages.returncode==0,(name,check.stdout,check.stderr)
    dependencies[name]={"python":str(executable),"pip_check":{"argv":[str(executable),"-B","-m","pip","check"],
        "returncode":check.returncode,"stdout":check.stdout,"stderr":check.stderr},"packages":json.loads(packages.stdout)}
(evidence/"dependencies.json").write_text(json.dumps(dependencies,ensure_ascii=False,indent=2),encoding="utf-8")
result={"result":"PASS","version":"0.4.0","tests":112,"matrix_rows":len(matrix),"policy_configs":17,
        "reward_models":6,"additional_diverse_mode":1,"source_parity_reports":24,"source_parity_max_error":max(errors),
        "real_episode_files_small_batch_read":12,"source_hash_records_unchanged":checked,
        "runtime_source_wheel_installed_files_identical":runtime_count,"reference_python312_bytecode_files":bytecode,
        "wheel_sha256":hashlib.sha256(wheel.read_bytes()).hexdigest(),"wheel_bytes":wheel.stat().st_size,
        "pip_checks":"PASS_BOTH_PROJECT_ENVS","real_smac_evaluation":"BLOCKED","policy_effectiveness":"NOT_VALIDATED","stopped_after_stage":4}
(evidence/"final_verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(result,ensure_ascii=False))
