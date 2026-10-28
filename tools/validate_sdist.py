"""Rebuild final sdist outside source with a newly created base-only venv."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import uuid
import zipfile

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/'docs/evidence/stage7'

def main():
    work=ROOT.parent/'.release-check'/('sdist-'+uuid.uuid4().hex[:8])
    work.mkdir(parents=True)
    python=ROOT/'.venv-sdist/Scripts/python.exe'
    full_python=ROOT/'.venv-release/Scripts/python.exe'
    environment=os.environ.copy()
    environment.update(PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',TEMP=str(ROOT/'.work/tmp'),TMP=str(ROOT/'.work/tmp'))
    environment.pop('PYTHONPATH',None)
    commands=[]
    def run(arguments,cwd):
        result=subprocess.run([str(x) for x in arguments],cwd=cwd,env=environment,capture_output=True,text=True,encoding='utf-8',timeout=120)
        commands.append(dict(command=[str(x) for x in arguments],cwd=str(cwd),returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
        if result.returncode: raise RuntimeError(json.dumps(commands[-1],ensure_ascii=False))
        return result
    sdist=ROOT/'dist/unified_marl-0.7.0.tar.gz'
    wheel=ROOT/'dist/unified_marl-0.7.0-py3-none-any.whl'
    with tarfile.open(sdist) as archive:
        for member in archive.getmembers():
            target=(work/member.name).resolve()
            assert target.is_relative_to(work.resolve()) and not member.issym() and not member.islnk()
        archive.extractall(work,filter='data')
    extracted=work/'unified_marl-0.7.0'
    run([python,'-B','-m','build','--wheel','--no-isolation','--outdir',work/'rebuilt'],extracted)
    rebuilt=work/'rebuilt'/wheel.name
    with zipfile.ZipFile(wheel) as original, zipfile.ZipFile(rebuilt) as actual:
        files=[name for name in original.namelist() if name.startswith('unified_marl/')]
        assert files
        assert {name for name in actual.namelist() if name.startswith('unified_marl/')}==set(files)
        for name in files:
            assert original.read(name)==actual.read(name),name
            if name.endswith(('.py','.yaml')):
                assert (ROOT/'.venv-release/Lib/site-packages'/name).read_bytes()==original.read(name),('Untested runtime change',name)
    run([python,'-B','-m','pip','install','--no-index','--no-deps','--force-reinstall',rebuilt],work)
    run([full_python,'-B','-m','pip','install','--no-index','--no-deps','--force-reinstall',wheel],work)
    shutil.copyfile(extracted/'tools/validate_release.py',work/'validate_release.py')
    shutil.copyfile(extracted/'examples/make_demo_data.py',work/'make_demo_data.py')
    run([python,'-B',work/'validate_release.py','--mode','base','--output',work/'base-acceptance'],work)
    shutil.copyfile(work/'base-acceptance/report.json',EVIDENCE/'release-base.json')
    report=dict(status='PASS',sdist=str(sdist),input_sha256=hashlib.sha256(sdist.read_bytes()).hexdigest(),
        extracted=str(extracted),python=str(python),rebuilt_wheel=str(rebuilt),runtime_files_equal=len(files),
        full_validation_runtime_equal_before_reinstall=True,base_acceptance='PASS',commands=commands[:])
    (EVIDENCE/'sdist-validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    examples=extracted/'examples'
    demo=work/'examples-demo.th'
    run([python,'-B',examples/'make_demo_data.py',demo],work)
    run([python,'-B',examples/'online.py','--output',work/'examples-qmix'],work)
    run([python,'-B',examples/'online.py','--algorithm','sadma/ppo','--output',work/'examples-ppo'],work)
    run([python,'-B',examples/'inference.py',work/'examples-qmix/first/policy.pt'],work)
    for name in ['kou/bc','kou/cpl']:
        run([python,'-B',examples/'offline.py','--algorithm',name,'--dataset',demo,'--output',work/('examples-'+name.replace('/','-'))],work)
    run([full_python,'-B',examples/'reward.py','--dataset',demo,'--output',work/'examples-reward'],work)
    run([python,'-B',examples/'train_qmix.py'],work)
    run([full_python,'-B',examples/'stage6_zmq.py'],work)
    run([ROOT/'.venv-release/Scripts/unified-marl.exe','train','--algorithm','sadma/qmix','--env','toy','--execution','zmq','--config',examples/'zmq_managed.yaml',
         '--output-dir',work/'examples-zmq-cli','--set','evaluation_interval_updates=1','--set','evaluation_episodes=2'],work)
    example_report=dict(status='PASS',commands=commands[len(report['commands']):],runs=len(commands)-len(report['commands']),directory=str(work))
    (EVIDENCE/'examples-validation.json').write_text(json.dumps(example_report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='PASS',source_runtime_files=len(files),base_runs=12,example_commands=example_report['runs'],directory=str(work)),ensure_ascii=False))

if __name__=='__main__':main()
