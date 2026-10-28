# 工程应用中文说明书（0.8.0）

本版按仓储补货、交通信号、电网连续控制提供应用入口。用户要求自行测试，本轮没有执行任何训练、环境启动、真实场景或收敛实验。代码和发行包检查不能代替效果验证；0.7.0 的旧验证记录只适用于旧记录中的实现。

## 安装和选择

从 GitHub Releases 下载对应 Windows x64 或 Linux x86_64 工程 ZIP，先安装 CPython 3.12 64位，再按包内说明安装。基础依赖带入独立虚拟环境，CPU 版 PyTorch 与上一版相同。HMF、角色、奖励等依赖按安装功能选择。CityFlow 引擎、原 ReplenishmentEnv 和电网业务数据不随包提供；PowerGrid 扩展按 requirements-powergrid.txt 联网安装。

- [仓储补货参数与调用](WAREHOUSE_GUIDE.md)
- [交通控制参数与调用](TRAFFIC_GUIDE.md)
- [电网控制参数与调用](POWERGRID_GUIDE.md)
- [平台安装与外部依赖](ENGINEERING_DISTRIBUTION.md)
- [本轮实现与检查范围](ENGINEERING_REPORT.md)

## 统一操作

```sh
unified-marl-app init warehouse --directory ./my-warehouse
unified-marl-app init traffic --directory ./my-traffic
unified-marl-app init powergrid --directory ./my-grid
unified-marl-app check --config ./my-warehouse/warehouse.yaml
# 下列命令由你在准备真实参数后执行，本次交付没有执行：
unified-marl-app train --config ./my-warehouse/warehouse.yaml --output-dir ./results/warehouse
unified-marl-app evaluate ./results/warehouse/policy.pt --episodes 1
unified-marl-app act ./results/warehouse/policy.pt --input ./my-warehouse/warehouse_state.json --output ./decision.json
```

`init` 导出可编辑模板，不下载场景、不训练。`check` 读取配置和数据、检查契约，不创建仿真器。相对路径以 YAML 所在目录为基准；Python 传 dict 则以当前目录为基准。seed 为 [0, 2**32) 的整数。输出目录必须为空，不覆盖已有结果。

```python
from unified_marl.applications import warehouse, traffic, powergrid
result = warehouse.train("my-warehouse/warehouse.yaml", output_dir="results/run-001")
policy = warehouse.load_policy(result.checkpoint)
policy.reset()  # 每个回合开始调用
# decision = policy.act(current_state)
```

训练返回 ApplicationResult：application、algorithm、output_dir、checkpoint、training_checkpoint、metrics、updates、env_steps 和原始结果。完整场景说明保存在 application.json，必须与模型一起分发；仅保留 policy.pt 不足以做业务动作映射。继续训练用 `api.train(config, output_dir="新目录", resume=result.training_checkpoint)`；策略checkpoint与完整训练状态分开。

## 参数优先级

底层默认配置 < 所选算法/环境配置 < 应用 YAML 的 training/env_args < 调用 overrides。应用显式 seed/device/execution 传入核心接口。CLI 覆盖示例：`--set training.max_updates=200 --set seed=7`。`training` 中禁止再次写 env、algorithm、seed 等应用层字段，防止出现两个来源。

## 计算与模型

没有随包提供行业预训练权重。第一次使用要用自己的场景训练，或加载自己已有且匹配的应用模型。修改智能体数量、排序、动作定义、观测字段或物理参数可能需要重训；模型与场景检查不保证模型能迁移到任意业务数据。仓储用命名库存状态；交通/电网使用环境定义的观测数组和动作约束。不要直接把任意原始业务表格传给神经网络。

命令行 `act` 每次进程启动会重置隐藏状态，适合单步或使用 `--steps` 顺序处理同一回合的一组输入。长期在线控制应在一个 Python 进程中保留 policy，并逐步调用 act；新回合再 reset。`--steps` 输入是按时间排列的JSON列表，每项可含 observation 与 avail_actions；不跨回合自动重置。

## 能力边界

仓储内置模型覆盖库存补货，没有运输路径、车辆派单、共享仓库容量、调拨或供应商产能模型。交通直接控制 CityFlow 绿灯相位，黄灯/安全过渡由显式配置决定。电网是配电网建筑用能连续控制，不涵盖完整发电机组组合、电力交易或输电安全优化。新增能力均为实现待用户测试，原 HMF 动态分组、RODE扩角色等研究缺项继续保留。

## 0.8.1 热修复

根据Linux第一轮用户报告修复应用恢复中的自动seed误判，并为仓储/交通所用核心评估增加episode_lengths和env_steps。安装及二测见[热修复说明](ENGINEERING_HOTFIX_081.md)。原安装器添加功能后需再次应用补丁；当前补丁没有随0.8.0平台ZIP自动分发。

## 0.8.2

电网F1/F2及安装N1修复见[热修复说明](ENGINEERING_HOTFIX_082.md)与[第三轮提示词](LINUX_ROUND3_PROMPTS.md)。完整补丁升级原安装入口并保留备份，不能只安装wheel而声称N1已修复。
