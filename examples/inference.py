"""在 toy 环境逐步推理；每个新回合重置 RNN 隐藏状态。"""
import argparse
from unified_marl import load_policy
from unified_marl.envs import make_env


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    args=parser.parse_args()
    policy=load_policy(args.checkpoint)
    if policy.config["env"]!="toy": raise ValueError("该最小示例仅运行 toy checkpoint；真实环境请提供匹配的观测和动作掩码")
    env=make_env("toy",**policy.config["env_args"])
    try:
        for _ in range(2):
            env.reset()
            policy.reset()
            total=0
            for _ in range(env.episode_limit):
                action=policy.act(env.get_obs(),env.get_avail_actions())
                reward,done,_=env.step(action)
                total+=reward
                if done: break
            print("回报",total)
    finally: env.close()


if __name__=="__main__": main()
