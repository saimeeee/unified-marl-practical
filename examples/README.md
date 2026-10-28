# 最小调用示例

先在隔离环境安装发行 wheel。以下命令从解压后的源码包根目录执行；示例没有向 sys.path 注入原始交付路径，也可复制到其他目录使用。每次换一个新的输出目录。

```powershell
python examples/online.py --output ./results/demo-online
python examples/online.py --algorithm sadma/ppo --output ./results/demo-ppo
python examples/inference.py ./results/demo-online/first/policy.pt
unified-marl evaluate ./results/demo-online/first/policy.pt --episodes 2 --seed 7
```

在线示例包括有效更新、保存、加载评估与 training.pt 恢复；推理示例逐回合调用 reset。

```powershell
python examples/make_demo_data.py ./my-data/demo.th
python examples/offline.py --dataset ./my-data/demo.th --output ./results/demo-bc
python examples/offline.py --algorithm kou/cpl --dataset ./my-data/demo.th --output ./results/demo-cpl
python examples/reward.py --dataset ./my-data/demo.th --output ./results/demo-reward
```

demo.th 为明确标记的合成 toy 数据，不是 SMAC 训练数据。替换为用户真实数据时，保留团队奖励、配对和标签方向：前后半段配对，labels=1 偏好后半段、0 偏好前半段、0.5 平局。奖励示例需安装 `[reward]`。需要真实个体奖励的其他算法见支持清单，不复制团队奖励。

```powershell
python examples/stage6_zmq.py
unified-marl train --algorithm sadma/qmix --env toy --execution zmq --config examples/zmq_managed.yaml --output-dir ./results/demo-zmq --set evaluation_interval_updates=1 --set evaluation_episodes=2
```

分布式示例需 `[distributed]` 并允许本机回环 TCP。所有 Windows 多进程示例均保护主入口。独立 trainer/infer/sample 三角色命令见主 README 和 docs/ZMQ_PROTOCOL.md。evaluation_interval_updates=0 为默认关闭；启用后最多一个独立评估进程，忙时跳过并记录触发次数，错误和超时会使任务失败。

真实环境示例仅是调用说明：SMAC 需要 SMAC/PySC2、StarCraft II、地图和 SC2PATH；补货需要 ReplenishmentEnv 及其任务数据；CityFlow 需要包装器、引擎及路网流量。当前没有这些环境的运行验收证据。
