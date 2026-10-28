# 实用首版：同一算法切换环境训练

这一版把算法选择、环境选择和实验组织分开：算法负责选动作与更新参数，环境提供统一的多智能体接口，YAML 决定组合、预算与随机种子。新增算法可以通过 Python 模块注册；不必为 MPE、库存等每个环境另写训练入口。

首版通用在线协议支持 **离散动作、固定智能体数、团队标量奖励、全局状态与动作掩码**。它不承诺每一种既有算法都兼容所有环境。连续动作电网保留原入口，尚未纳入这个通用实验矩阵；原有离线偏好学习和分布式训练也保留各自的数据与执行协议。

## 1. 从两个自带环境开始

在项目根目录操作，使用已经安装本项目的 Python 环境。以下命令中的 `python` 可替换为本机的 `.\.venv\Scripts\python.exe`。

```powershell
python -m pip install -e .
python -m unified_marl catalog --plugin unified_marl.contrib.huber_iql
python -m unified_marl check examples/experiments/mpe.yaml
python -m unified_marl check examples/experiments/mpe.yaml --instantiate
python -m unified_marl run examples/experiments/mpe.yaml
python -m unified_marl run examples/experiments/inventory.yaml
```

`check` 默认导入 YAML 明确选择的插件，校验配置与声明的能力，不构造仿真器。加 `--instantiate` 后会构造环境、读取维度并关闭环境；它不执行动作或训练。因此默认检查成功不代表外部仿真器及其数据已经安装，构造检查成功也不等于训练已通过。

| 环境名 | 实际内容 | 首版使用边界 |
|---|---|---|
| `mpe` | 交付源码中的 `simple_spread` 粒子物理与协同地标覆盖任务 | 自带 NumPy 实现；只开放该场景和五个离散动作；没有渲染器 |
| `inventory` | 仓库/SKU 库存补货仿真，读取 agents/demand CSV | 自带后端；示例 CSV 是演示数据；它与外部 `ReplenishmentEnv` 是两个不同后端 |
| `traffic_engine` | 原生 CityFlow 交通引擎 | 需要已安装的 CityFlow，以及实际路网和车流 JSON |
| `cityflow` | 原交付的 `cityflowenv` 包装器 | 需要对应外部包装器、引擎和场景资产 |
| `replenishment` | 外部 ReplenishmentEnv | 需要该包及其任务数据 |
| `sc2` | StarCraft II / SMAC 协作微操 | 需要游戏、SMAC 与地图；当前首版示例未验证该外部运行环境 |
| `toy` | 小型协调测试环境 | 供接口测试；不作为实际领域实验结果 |

MPE 的团队奖励沿用交付实现，将各智能体场景奖励求和，其中包含原实现的自碰撞常数项。比较结果时应使用相同实现、智能体数、地标数和回合长度，不应直接与其他 MPE 版本的原始回报比较。

## 2. 单实验 YAML

完整示例见 `examples/experiments/mpe.yaml`、`inventory.yaml` 和 `plugin_mpe.yaml`：

```yaml
schema_version: 1
name: huber-iql-mpe
plugins: [unified_marl.contrib.huber_iql]
algorithm: example/huber_iql
env: mpe
seed: 11
device: cpu
env_args:
  scenario: simple_spread
  n_agents: 3
  n_landmarks: 3
  episode_limit: 25
training:
  t_max: 96
  max_updates: 2
  batch_size: 2
  buffer_size: 8
  rnn_hidden_dim: 16
  huber_delta: 1.0
evaluation:
  episodes: 3
  seed_count: 2
```

| 字段 | 含义 |
|---|---|
| `schema_version` | 当前只接受整数 `1`，可省略 |
| `name` | 实验说明名称；不是输出目录路径 |
| `algorithm` / `env` | 必填，分别选择算法和环境注册名 |
| `env_args` | 原样传给该环境的参数；具体字段由环境接口决定 |
| `training` | 底层算法/训练配置；只接受该算法默认配置中已声明的字段 |
| `device` / `seed` | 默认 `cpu` / `1`；seed 必须是 `[0, 2**32)` 内整数 |
| `plugins` | 导入并执行的 Python 模块列表；只指定自己信任且可导入的模块 |
| `evaluation.episodes` | 每个评估 seed 的回合数，默认 `5` |
| `evaluation.seed_count` | 自动生成的评估 seed 数，默认 `1` |
| `evaluation.seeds` | 显式评估 seed 列表；与 `seed_count` 二选一 |
| `path_keys` | `env_args` 内额外的路径字段名，可使用点号访问嵌套字段 |

示例预算只用于快速检查端到端训练。`max_updates` 限制更新次数，`t_max` 限制采样步数；先到哪个就停止，回合采样可能使实际步数越过步数边界。预算不足以填满训练 batch 时不会产生更新，应增加预算或减小 batch，而不是把“没有训练”记作成功。

`example/huber_iql` 是 **IQL 的 Smooth-L1 TD 损失教学变体**，用于演示独立插件接入；它不是三位作者的原创算法，也没有优于基线的性能结论。

随机种子放在顶层 `seed`，不要再放入 `training` 或 `env_args`。输出目录用 CLI/API 参数指定。实验入口拒绝拼写错误的顶层和训练字段，也拒绝当前入口不会执行的源项目遗留控制项，例如 `training.save_model`、`training.evaluate`、`training.checkpoint_path`。本入口始终在成功训练后保存模型并按 `evaluation` 评估。

覆盖参数沿用底层配置名，不加 `training.` 前缀：

```powershell
python -m unified_marl run examples/experiments/mpe.yaml --set max_updates=4 --set t_max=160
```

`--set env_args.episode_limit=50` 可改环境参数。算法、环境、device、seed 和插件仍在 YAML 对应层级选择，不通过 training 覆盖。

## 3. 数据文件路径

YAML 中的路径相对于 **该 YAML 文件的父目录**，与启动命令所在目录无关。已知路径字段包括 `agents_file`、`demand_file`、`cityflow_config_file`、`config_file`、`data_path`、`dataset_path`、`data_dir`、`vis_path`。

未知字段不会因为字符串里包含斜杠而被猜成路径。自定义环境可以显式声明：

```yaml
env_args:
  task_type: Standard
  backend_args:
    weights: models/weights.pt
path_keys:
  - backend_args.weights
```

这里仅 `backend_args.weights` 会解析为绝对路径，`task_type` 保持普通字符串。文件是否存在或内容是否有效，由环境构造/运行校验。CityFlow JSON 内部再引用的路网和车流路径，继续按 CityFlow 后端自己的规则解析。

## 4. 算法 × 环境 × seed 矩阵

```powershell
python -m unified_marl run-suite examples/experiments/practical_suite.yaml
```

该文件执行 QMIX、VDN、HuberIQL × MPE、inventory × 两个训练 seed，共 12 次独立短跑。矩阵不是混合环境训练：每个组合拥有独立模型、配置、日志和评估。

```yaml
plugins: [unified_marl.contrib.huber_iql]
algorithms:
  - pymarl/qmix
  - name: example/huber_iql
    training:
      huber_delta: 0.5
envs:
  - name: mpe
    env_args:
      scenario: simple_spread
      n_agents: 3
      episode_limit: 25
seeds: [11, 23]
training:
  t_max: 96
  max_updates: 2
  batch_size: 2
  buffer_size: 8
evaluation:
  episodes: 3
  seed_count: 2
```

公共 `training` 与每个算法的 `training` 合并，后者优先。`algorithms` 和 `envs` 都可用字符串或带 `name` 的映射；每个环境可独立设置 `env_args/path_keys`。空矩阵和重复训练 seed 会报错。

自动评估 seeds 是确定性生成的，与整个 suite 的所有训练 seeds 都不同。各组合使用相同评估 seeds 便于同场景比较。显式评估 seeds 也必须唯一，并与全部训练 seeds 不相交。仅换 seed 不会自动将库存 CSV 划分为训练集/测试集；如果环境数据固定，独立 seed 不能代替独立测试数据。

矩阵汇总只计算同一算法条目、同一环境条目下的训练 seed 均值与总体标准差；不把 MPE 与库存的奖励混为一个总分，也不生成跨环境排名。统计范围和有效/失败运行数均写入结果。

## 5. 查看、复现和继续运行

未指定输出目录时，结果放在 YAML 父目录的 `results/` 下，使用时间和随机后缀生成新目录。显式指定 `--output-dir` 时目录必须尚不存在；重复运行请改目录或省略该参数。

| 文件 | 内容 |
|---|---|
| `experiment.json` | 规范化后的实验意图、绝对数据路径、训练/评估 seeds |
| `config.json` | 训练实际使用的完整合并配置，包括插件模块与环境参数 |
| `metrics.jsonl` | 训练统计 |
| `policy.pt` | 推理/评估模型及配置 |
| `training.pt` | 学习器、优化器、回放与 RNG 等训练状态 |
| `result.json` | 底层训练结果 |
| `experiment_result.json` | 单实验状态、覆盖参数、训练结果、各评估 seed 结果及耗时 |
| `suite.json` / `suite_result.json` / `suite_result.csv` | 原始矩阵配置、完整运行状态与可表格阅读的汇总 |

单实验失败会保存错误类型和消息并返回非零退出状态；矩阵继续执行剩余组合，最终标记 `success`、`partial_failure` 或 `failed`。中断会保存 `interrupted`。失败目录会保留已有证据，不会伪造 checkpoint 或换用另一个环境。

以下命令中的路径换成运行结果里的真实 checkpoint：

```powershell
python -m unified_marl evaluate path/to/policy.pt --episodes 20 --seed 9001
python -m unified_marl train --resume path/to/training.pt --set t_max=200 --set max_updates=10 --output-dir results/resumed-run
```

恢复训练延长的是总预算，不是追加预算；输出必须选择新目录。恢复的是回合边界的学习器、回放和全局 RNG，外部仿真器的内部状态不包含在 checkpoint 中。跨依赖版本、设备或仿真器实现的运行不保证逐位一致。

插件模块名随 checkpoint 保存，评估/恢复时会重新导入；插件代码本身不会打包进 checkpoint，因此换机器时仍需安装同版本模块。

Python 入口也可以使用：

```python
from unified_marl.experiments import check_experiment, run_experiment, run_suite
from unified_marl import evaluate, load_policy

checked = check_experiment("examples/experiments/plugin_mpe.yaml", instantiate=True)
run = run_experiment("examples/experiments/plugin_mpe.yaml")
score = evaluate(run["train"]["checkpoint"], episodes=20, seed=9001)
policy = load_policy(run["train"]["checkpoint"])
# 每个新回合调用 policy.reset()；随后 policy.act(obs, avail_actions)。
```

新算法实现接口见 [ADDING_ALGORITHM.zh-CN.md](ADDING_ALGORITHM.zh-CN.md)。
