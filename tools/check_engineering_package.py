"""Static/configuration checks only: never instantiate an environment or run training."""
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import warnings

ROOT=Path(__file__).resolve().parents[1]

def main():
    sources=list((ROOT/"src/unified_marl").rglob("*.py"))+list((ROOT/"examples/applications").glob("*.py"))
    syntax_warnings=[]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for file in sources: ast.parse(file.read_text(encoding="utf-8-sig"),filename=str(file))
        syntax_warnings=[str(w.message) for w in caught]
    sys.path.insert(0,str(ROOT/"src"))
    from unified_marl.applications import warehouse,traffic,powergrid,validate
    from unified_marl.applications import export_template
    from unified_marl.applications.cli import main as cli
    report=validate(ROOT/"examples/applications/warehouse.yaml")
    assert report["algorithm"]=="wang_hmf/mfvfd"
    with contextlib.redirect_stdout(io.StringIO()):
        try: cli(["--help"])
        except SystemExit as e: assert e.code==0
    template_files=list((ROOT/"src/unified_marl/applications/templates").iterdir())
    for file in template_files:
        example=ROOT/"examples/applications"/file.name
        assert example.read_bytes()==file.read_bytes(),f"Template and example differ: {file.name}"
        if file.suffix==".json": json.loads(file.read_text(encoding="utf-8"))
    docs=[ROOT/"README.md",*(ROOT/"docs").glob("*GUIDE.md"),ROOT/"docs/ENGINEERING_DISTRIBUTION.md"]
    # Historical docs may reference archived raw evidence; only new guides are required here.
    docs=[f for f in docs if f.name in {"README.md","APPLICATIONS_GUIDE.md","WAREHOUSE_GUIDE.md","TRAFFIC_GUIDE.md","POWERGRID_GUIDE.md","ENGINEERING_DISTRIBUTION.md"}]
    missing=[]
    for file in docs:
        text=file.read_text(encoding="utf-8")
        assert text.count("```")%2==0,str(file)
        for link in re.findall(r"\]\(([^)]+)\)",text):
            if link.startswith(("http:","https:","mailto:","#")):continue
            if not (file.parent/link.split("#")[0]).exists(): missing.append([str(file.relative_to(ROOT)),link])
    assert not missing,missing
    for name in ["cityflow","pandapower"]: assert name not in sys.modules,f"Optional simulator imported: {name}"
    result={"status":"STATIC_AND_CONFIG_CHECKS_PASS","source_files":len(sources),"template_files":len(template_files),
        "warehouse_parameters_checked":True,"application_imports":True,"cli_help":True,"new_document_links":True,
        "syntax_warnings":syntax_warnings,"training_run":False,"environment_created":False,"linux_runtime_test":False,
        "assets":{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in template_files}}
    out=ROOT/"docs/evidence/engineering";out.mkdir(parents=True,exist_ok=True)
    (out/"checks.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="assets"},ensure_ascii=False))

if __name__=="__main__":main()
