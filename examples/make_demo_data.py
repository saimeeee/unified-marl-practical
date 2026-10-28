"""生成 toy 合成轨迹；仅用于接口演示，不是真实 SMAC 数据。"""
import argparse
from pathlib import Path
import numpy as np
import torch
from unified_marl.envs.toy import CoordinationEnv


def make_demo_data(path):
    count, slots, agents, actions = 8, 5, 2, 3
    data = {"obs": torch.zeros(count, slots, agents, 4), "state": torch.zeros(count, slots, 3),
            "actions": torch.zeros(count, slots, agents, 1, dtype=torch.long),
            "avail_actions": torch.ones(count, slots, agents, actions, dtype=torch.int),
            "reward": torch.zeros(count, slots, 1), "filled": torch.ones(count, slots, 1, dtype=torch.long),
            "terminated": torch.zeros(count, slots, 1, dtype=torch.uint8),
            "agent_ids": torch.arange(agents).expand(count, -1).double()}
    rng = np.random.default_rng(7)
    for episode in range(count):
        env = CoordinationEnv(seed=episode, mask_actions=False)
        try:
            env.reset()
            for t in range(slots):
                data["obs"][episode, t] = torch.as_tensor(np.asarray(env.get_obs()))
                data["state"][episode, t] = torch.as_tensor(env.get_state())
                action = rng.integers(actions, size=agents)
                data["actions"][episode, t, :, 0] = torch.as_tensor(action)
                if t < slots-1:
                    reward, done, _ = env.step(action)
                    data["reward"][episode, t, 0] = reward
                    data["terminated"][episode, t, 0] = done
        finally:
            env.close()
    returns = data["reward"].sum(1)
    data["labels"] = torch.where(returns[4:] == returns[:4], 0.5, (returns[4:] > returns[:4]).float())
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): raise FileExistsError(path)
    torch.save(data, path)
    return str(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    print(make_demo_data(parser.parse_args().output))
