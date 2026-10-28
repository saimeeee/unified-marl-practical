"""Read-only SADMA behavior comparison and one-time namespaced source import."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import yaml

root=Path(__file__).resolve().parents[1]
workspace=root.parent
sources={"wang":workspace/"marl平台代码/王思哲/code/sadma",
 "yi_open":workspace/"marl平台代码/易才润/论文/SADMA/sadma源代码开源版本",
 "yi_closed":workspace/"marl平台代码/易才润/论文/SADMA/sadma源代码闭源版本",
 "yi_zmq":workspace/"marl平台代码/易才润/基于ZMQ的分布式多智能体强化学习算法/sadma源代码公开版本"}
evidence=root/"docs/evidence/stage5"
evidence.mkdir(parents=True,exist_ok=True)
files={key:{p.relative_to(base).as_posix():p for p in base.rglob("*") if p.suffix in {".py",".yaml",".sh"}} for key,base in sources.items()}
comparisons={}
for key in ["wang","yi_closed","yi_zmq"]:
    differences=[]
    base=files["yi_open"]
    for relative in sorted(set(base)|set(files[key])):
        left,right=base.get(relative),files[key].get(relative)
        if left and right and left.read_bytes()==right.read_bytes(): continue
        a=left.read_text(encoding="utf-8-sig") if left else ""
        b=right.read_text(encoding="utf-8-sig") if right else ""
        ast_equal=relative.endswith(".py") and ast.dump(ast.parse(a))==ast.dump(ast.parse(b))
        differences.append(dict(path=relative,ast_equal=ast_equal,diff="\n".join(difflib.unified_diff(a.splitlines(),b.splitlines(),fromfile="yi_open",tofile=key))))
    comparisons[key]=differences
payload={"sources":{k:str(p) for k,p in sources.items()},"file_counts":{k:len(v) for k,v in files.items()},"compared_to_yi_open":comparisons,
 "primary":"wang","reason":"Same core learning formulas as Yi plus explicit policy save/load; Yi powergrid environment/config retained separately; distributed code deferred",
 "source_hashes":[dict(source=p.relative_to(workspace).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for mapping in files.values() for p in mapping.values()],
 "configs":{key:{r:yaml.safe_load(p.read_text(encoding="utf-8")) for r,p in mapping.items() if p.suffix==".yaml"} for key,mapping in files.items()}}
(evidence/"audit.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({"files":payload["file_counts"],"differences":{k:len(v) for k,v in comparisons.items()}},ensure_ascii=False))

if __name__=="__main__":
    import sys
    if "--migrate" in sys.argv:
        base=sources["wang"]
        target=root/"src/unified_marl/algorithms/sadma/native"
        if target.exists(): raise FileExistsError(target)
        records=[]
        selection=[p for p in base.rglob("*.py") if p.relative_to(base).parts[0]=="policy" and "maddpg" not in p.parts]
        selection += [base/"utils/utils.py",base/"utils/buffer.py",base/"utils/scheme.py"]
        for p in selection:
            relative=p.relative_to(base)
            output=target/relative
            output.parent.mkdir(parents=True,exist_ok=True)
            text=p.read_text(encoding="utf-8")
            text=text.replace("from policy", "from unified_marl.algorithms.sadma.native.policy").replace("from utils", "from unified_marl.algorithms.sadma.native.utils")
            output.write_text(text,encoding="utf-8")
            records.append(dict(source=p.relative_to(workspace).as_posix(),target=output.relative_to(root).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        for folder in [target,*[p for p in target.rglob("*") if p.is_dir()]]:
            if not (folder/"__init__.py").exists(): (folder/"__init__.py").write_text("",encoding="utf-8")
        parent=target.parent
        (parent/"__init__.py").write_text('"""Namespaced local synchronous SADMA integration."""\n',encoding="utf-8")
        config=root/"src/unified_marl/config/sadma"
        config.mkdir(parents=True,exist_ok=True)
        for p in [base/"config/default.yaml",*sorted((base/"config/alg_config").glob("*.yaml"))]:
            (config/p.name).write_bytes(p.read_bytes())
        (evidence/"migration_sources.json").write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding="utf-8")
        print("Namespaced",len(records),"source files")
