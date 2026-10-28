"""Engineering application CLI. No training unless the train subcommand is requested."""
import argparse
import json
import sys
from pathlib import Path
import yaml
from .service import APPS, export_template, validate, train, evaluate, load_policy, _jsonable


def main(argv=None):
    parser=argparse.ArgumentParser(prog="unified-marl-app")
    sub=parser.add_subparsers(dest="command",required=True)
    init=sub.add_parser("init",help="Export editable scenario templates")
    init.add_argument("application",choices=sorted(APPS));init.add_argument("--directory",required=True)
    check=sub.add_parser("check",help="Check parameter files without creating an environment")
    check.add_argument("--config",required=True);check.add_argument("--set",action="append",default=[])
    training=sub.add_parser("train",help="Train on your scenario (runs only when explicitly invoked)")
    training.add_argument("--config",required=True);training.add_argument("--output-dir",required=True)
    training.add_argument("--resume");training.add_argument("--set",action="append",default=[])
    ev=sub.add_parser("evaluate");ev.add_argument("checkpoint");ev.add_argument("--config")
    ev.add_argument("--episodes",type=int,default=1);ev.add_argument("--seed",type=int)
    ev.add_argument("--device",default="cpu")
    act=sub.add_parser("act",help="Load a trained model and calculate one decision")
    act.add_argument("checkpoint");act.add_argument("--config");act.add_argument("--input",required=True,help="JSON observation or warehouse named state")
    act.add_argument("--output");act.add_argument("--device",default="cpu")
    act.add_argument("--steps",action="store_true",help="Input is a JSON list of chronological inputs in one episode")
    args=parser.parse_args(argv)
    try:
        from .service import load_config, _manifest
        overrides={}
        for item in getattr(args,"set",[]):
            if "=" not in item: raise ValueError("--set needs KEY=VALUE")
            key,value=item.split("=",1)
            if key in overrides: raise ValueError("Duplicate override key")
            overrides[key]=yaml.safe_load(value)
        if args.command=="init": result=export_template(args.application,args.directory)
        elif args.command=="check": result=validate(args.config,overrides=overrides)
        elif args.command=="train":
            cfg=load_config(args.config,overrides=overrides)
            result=train(cfg["application"],cfg,args.output_dir,resume=args.resume)
        else:
            application=_manifest(args.checkpoint)["config"]["application"]
            if args.command=="evaluate":
                result=evaluate(application,args.checkpoint,config=args.config,episodes=args.episodes,seed=args.seed,device=args.device)
            else:
                policy=load_policy(application,args.checkpoint,config=args.config,device=args.device)
                data=json.loads(Path(args.input).read_text(encoding="utf-8-sig"))
                policy.reset()
                def step(item):
                    if isinstance(item,dict) and "observation" in item:
                        return policy.act(item["observation"],item.get("avail_actions"))
                    return policy.act(item)
                result=[step(item) for item in data] if args.steps else step(data)
                if args.output:
                    Path(args.output).write_text(json.dumps(_jsonable(result),ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
        print(json.dumps(_jsonable(result),ensure_ascii=False,indent=2,allow_nan=False))
        return 0
    except Exception as exc:
        print(f"unified-marl-app: {type(exc).__name__}: {exc}",file=sys.stderr)
        return 1


if __name__=="__main__": raise SystemExit(main())
