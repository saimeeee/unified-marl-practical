"""Check final archives, installed bytes, source preservation and evidence."""
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import xml.etree.ElementTree as ET
import zipfile

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/'docs/evidence/stage7'
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    report={"version":"0.7.0"}
    rows=list(csv.DictReader((ROOT/'docs/MIGRATION_MATRIX.csv').open(encoding='utf-8-sig')))
    previous=list(csv.DictReader((ROOT/'docs/evidence/stage6_matrix.csv').open(encoding='utf-8-sig')))
    assert len(rows)==len({r['id'] for r in rows})==383
    assert {r['id'] for r in previous}<={r['id'] for r in rows}
    assert all((ROOT.parent/r['source_path']).exists() for r in rows)
    assert all(r['blockers'] for r in rows if r['local_test_verified']!='YES')
    report['matrix_rows']=len(rows);report['previous_rows_retained']=len(previous)
    count=0
    for filename in ['source_manifest.json','supplemental_manifest.json']:
        for row in json.loads((ROOT/'docs/evidence'/filename).read_text(encoding='utf-8')):
            assert sha(ROOT.parent/row['path'])==row['sha256'],row['path']
            count+=1
    assert count==1228
    bytecode=list((ROOT.parent/'marl平台代码').rglob('*.cpython-312.pyc'))+list((ROOT.parent/'pymarl-master').rglob('*.cpython-312.pyc'))
    assert not bytecode,bytecode
    report['source_hash_records_unchanged']=count;report['reference_py312_bytecode_files']=0
    tests=ET.parse(EVIDENCE/'tests.xml').getroot().find('testsuite')
    assert int(tests.attrib['tests'])==220 and int(tests.attrib['errors'])==int(tests.attrib['failures'])==0
    report['tests_passed']=220
    for name in ['release-base.json','release-full.json','sdist-validation.json','examples-validation.json']:
        value=json.loads((EVIDENCE/name).read_text(encoding='utf-8'))
        assert value['status']=='PASS',name
    archive=ROOT/'dist/unified_marl-0.7.0-py3-none-any.whl'
    sdist=ROOT/'dist/unified_marl-0.7.0.tar.gz'
    forbidden={'.pt','.pth','.th','.npy','.npz','.pkl','.pickle','.pyc'}
    with zipfile.ZipFile(archive) as wheel:
        names=wheel.namelist()
        for name in names:
            assert PurePosixPath(name).suffix.lower() not in forbidden,name
            assert not any(part in {'.venv','.venv-release','.venv-sdist','.work','results','__pycache__'} for part in PurePosixPath(name).parts),name
        assert 'unified_marl/PROVENANCE.json' in names
        assert any(n.endswith('/licenses/SADMA-Yi-MIT.txt') for n in names)
        count=0
        for path in (ROOT/'src/unified_marl').rglob('*'):
            if path.suffix not in {'.py','.yaml','.json'}:continue
            relative=path.relative_to(ROOT/'src').as_posix()
            assert path.read_bytes()==wheel.read(relative),relative
            for env in ['.venv-release','.venv-sdist']:
                assert (ROOT/env/'Lib/site-packages'/relative).read_bytes()==path.read_bytes(),(env,relative)
            count+=1
        report['runtime_files_source_wheel_two_installs_equal']=count
        with tarfile.open(sdist) as source:
            entries=source.getnames()
            for name in entries:
                assert PurePosixPath(name).suffix.lower() not in forbidden,name
                assert not any(part.startswith('.venv') or part in {'.work','results','__pycache__'} for part in PurePosixPath(name).parts),name
            for relative in ['README.md','examples/README.md','docs/VALIDATION_REPORT.md','docs/SUPPORT_MATRIX.md','docs/KNOWN_ISSUES.md','docs/MIGRATION_MATRIX.csv','docs/MIGRATION_TRACEABILITY.csv','WORK_ORDER.md']:
                assert source.extractfile('unified_marl-0.7.0/'+relative).read()==(ROOT/relative).read_bytes(),relative
            for path in (ROOT/'src/unified_marl').rglob('*'):
                if path.suffix not in {'.py','.yaml','.json'}:continue
                assert source.extractfile('unified_marl-0.7.0/'+path.relative_to(ROOT).as_posix()).read()==path.read_bytes()
            report['sdist_entries']=len(entries)
        report['wheel_entries']=len(names)
    pip={}
    for env in ['.venv','.venv-release','.venv-sdist']:
        result=subprocess.run([str(ROOT/env/'Scripts/python.exe'),'-m','pip','check'],text=True,capture_output=True)
        assert result.returncode==0,result.stdout+result.stderr
        pip[env]=result.stdout.strip()
    report.update(status='PASS',pip_check=pip,artifacts=[dict(file=p.name,size=p.stat().st_size,sha256=sha(p)) for p in [archive,sdist]],
        real_environments_verified=False,cross_machine_verified=False,gpu_verified=False,policy_effect_verified=False)
    (ROOT/'dist/SHA256SUMS.txt').write_text(''.join(f"{item['sha256']}  {item['file']}\n" for item in report['artifacts']),encoding='utf-8')
    (EVIDENCE/'final_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
