"""Port pure feature/segmentation helpers without the LLM client or script mains."""
import ast
import hashlib
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
base=root.parent/"marl平台代码/寇谦/论文代码及技术文档/论文代码及技术文档"
records=[]
source=base/"get_preferences.py"
text=source.read_text(encoding="utf-8")
text=text.replace("['action']","['actions']")
text=text.replace("saved_exmple_path='prompt_example'", "saved_exmple_path=None")
text=text.replace("    base_prompt_path =", "    if saved_exmple_path is None:\n        raise ValueError('Supply the explicit prompt template directory')\n    base_prompt_path =",1)
target=root/"src/unified_marl/datasets/prompts.py"
if target.exists(): raise FileExistsError(target)
target.write_text(text,encoding="utf-8")
records.append({"source":source.relative_to(root.parent).as_posix(),"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"target":target.relative_to(root).as_posix(),"scope":"pure placeholders and explicit template path only"})
source=base/"make_diverse_preference.py"
text=source.read_text(encoding="utf-8")
function=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=="clip_to_segment_length")
target=root/"src/unified_marl/datasets/segments.py"
if target.exists(): raise FileExistsError(target)
target.write_text('"""Pure delivered random segment selection; no labels/services are invoked."""\nimport torch\n\n'+ast.get_source_segment(text,function)+'\n',encoding="utf-8")
records.append({"source":source.relative_to(root.parent).as_posix(),"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"target":target.relative_to(root).as_posix(),"scope":"clip_to_segment_length only; remote label generation excluded"})
path=root/"docs/evidence/stage4/migration_sources.json"
previous=json.loads(path.read_text(encoding="utf-8"))
path.write_text(json.dumps(previous+records,ensure_ascii=False,indent=2),encoding="utf-8")
print("Ported two pure data helpers; no remote labeling calls were made.")
