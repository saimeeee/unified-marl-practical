"""Read trained application model and emit business decisions."""
import argparse,json
from unified_marl.applications import warehouse,traffic,powergrid

def main():
    p=argparse.ArgumentParser()
    p.add_argument("application",choices=["warehouse","traffic","powergrid"])
    p.add_argument("checkpoint");p.add_argument("input");p.add_argument("--config")
    args=p.parse_args()
    api={"warehouse":warehouse,"traffic":traffic,"powergrid":powergrid}[args.application]
    policy=api.load_policy(args.checkpoint,config=args.config)
    policy.reset()
    with open(args.input,encoding="utf-8") as f: data=json.load(f)
    result=policy.act(data.get("observation",data) if isinstance(data,dict) else data,data.get("avail_actions") if isinstance(data,dict) else None)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
