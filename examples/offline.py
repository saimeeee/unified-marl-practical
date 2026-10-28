"""普通离线 BC / 偏好 CPL；预算为梯度更新次数。"""
import argparse
from dataclasses import asdict
from unified_marl import train


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--algorithm",choices=["kou/bc","kou/cpl"],default="kou/bc")
    args=parser.parse_args()
    result=train(args.algorithm,dataset=args.dataset,output_dir=args.output,seed=7,
        overrides={"training_updates":2,"off_batch_size":4,"rnn_hidden_dim":16,"dataset_options.max_length":12})
    print(asdict(result))


if __name__=="__main__": main()
