"""Explicit user-run training; importing this file does not train."""
import argparse
from unified_marl.applications import warehouse, traffic, powergrid, load_config

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("config");parser.add_argument("--output-dir",required=True)
    args=parser.parse_args()
    cfg=load_config(args.config)
    api={"warehouse":warehouse,"traffic":traffic,"powergrid":powergrid}[cfg["application"]]
    result=api.train(cfg,output_dir=args.output_dir)
    print(result)

if __name__=="__main__": main()
