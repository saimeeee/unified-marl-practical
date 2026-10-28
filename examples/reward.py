"""MR 奖励训练、预测与固定偏好批次的完整状态继续更新。"""
import argparse
from dataclasses import asdict
from unified_marl import train_reward_model,load_reward_model
from unified_marl.datasets import load_dataset


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    data=load_dataset(args.dataset,max_length=12)
    result=train_reward_model("MR",data,updates=2,batch_size=2,seed=7,output_dir=args.output,
                              model_config={"inner_dim":32,"action_embd_dim":8})
    print(asdict(result))
    model=load_reward_model(result.checkpoint)
    print("团队奖励预测形状",tuple(model.predict(data.batch([0,1])).shape))
    resumed=load_reward_model(result.training_checkpoint,for_training=True)
    print(resumed.train_step(data.preference([0,1],trim=False)))
    resumed.save(result.output_dir+"/reward-continued.pt",training=True)


if __name__=="__main__": main()
