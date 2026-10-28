"""在线训练、模型保存/评估和完整状态继续训练。"""
import argparse
from dataclasses import asdict
from pathlib import Path
from unified_marl import train, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--algorithm", default="pymarl/qmix", choices=["pymarl/qmix", "sadma/ppo"])
    args = parser.parse_args()
    options = dict(t_max=64,max_updates=2,batch_size=2,buffer_size=8,rnn_hidden_dim=16)
    if args.algorithm == "sadma/ppo": options.update(train_batch_size=2,train_buffer_size=8,hidden_dim=16,mini_epochs=2,update_rate=1)
    result = train(args.algorithm,"toy",seed=7,output_dir=Path(args.output)/"first",overrides=options)
    print(asdict(result))
    print(asdict(evaluate(result.checkpoint,episodes=2,seed=7)))
    continued = train(resume=result.training_checkpoint,output_dir=Path(args.output)/"continued",overrides={"max_updates":3})
    print(asdict(continued))


if __name__ == "__main__": main()
