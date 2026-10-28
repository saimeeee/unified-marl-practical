import ast
import importlib.util
import json
from pathlib import Path
import pytest
import torch
from test_offline import raw_data,cpu,SOURCE,EVIDENCE
from unified_marl.datasets import EpisodeDataset
from unified_marl.datasets.labels import labels_from_team_returns
from unified_marl.datasets import prompts
from unified_marl.datasets.segments import clip_to_segment_length


def test_team_return_labels_source_direction_and_ties():
    source=SOURCE/"old_src/make_pref_datasets_104.py"
    tree=ast.parse(source.read_text(encoding="utf-8"))
    assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="labels" for t in n.targets))
    rewards=torch.tensor([[[1.],[2.]],[[1.],[1.]],[[2.],[1.]],[[3.],[0.]]])
    expected=eval(compile(ast.Expression(assignment.value),str(source),"eval"),{"labeled_reward":rewards,"num_segment_pairs":2})
    actual=labels_from_team_returns(rewards[:2],rewards[2:])
    torch.testing.assert_close(actual,expected)
    assert labels_from_team_returns(rewards[:2],rewards[:2]).eq(0).all()
    with pytest.raises(ValueError,match="individual"):
        labels_from_team_returns(rewards.unsqueeze(2).repeat(1,1,3,1),rewards.unsqueeze(2).repeat(1,1,3,1))


def test_segment_selection_matches_source(raw_data):
    source=SOURCE/"make_diverse_preference.py"
    text=source.read_text(encoding="utf-8")
    function=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=="clip_to_segment_length")
    scope={"torch":torch}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),"exec"),scope)
    values=[raw_data[k] for k in ["obs","state","action","reward","avail_action","filled","terminated","agent_ids"]]
    lengths=raw_data["filled"].sum(1).squeeze(-1)
    torch.manual_seed(58)
    actual=clip_to_segment_length(*values,lengths,2)
    torch.manual_seed(58)
    expected=scope["clip_to_segment_length"](*values,lengths,2)
    for a,b in zip(actual,expected): torch.testing.assert_close(a,b,atol=0,rtol=0)


def test_real_smac_prompt_features_match_source():
    source=SOURCE/"get_preferences.py"
    spec=importlib.util.spec_from_file_location("reference_kou_prompt_features",source)
    reference=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    data=EpisodeDataset(SOURCE/"MADiversePreferDatasets/3m/3m.th")
    a,b=data.batch([0],trim=False),data.batch([200],trim=False)
    checks=[("get_global_placeholder",(a,b)),("get_between_placeholder",(a,b,1)),("get_in_placeholder",(a,))]
    for name,args in checks:
        actual=getattr(prompts,name)(*args,scenario="3m")
        legacy=tuple(dict(x,action=x["actions"],avail_action=x["avail_actions"]) if isinstance(x,dict) else x for x in args)
        expected=getattr(reference,name)(*legacy,scenario="3m")
        assert json.dumps(actual,default=str,sort_keys=True)==json.dumps(expected,default=str,sort_keys=True)
    template=SOURCE/"prompt_example"
    actual=prompts.get_prompt({"test":1},saved_exmple_path=str(template))
    expected=reference.get_prompt({"test":1},saved_exmple_path=str(template))
    assert actual==expected
    with pytest.raises(ValueError,match="explicit"): prompts.get_prompt({})
    (EVIDENCE/"data_helper_parity.json").write_text(json.dumps({"result":"PASS","real_data":"3m",
        "source_feature_functions":3,"explicit_template_read":True,"segment_sampling":"same RNG gives exact tensors",
        "remote_llm_calls":0}),encoding="utf-8")
