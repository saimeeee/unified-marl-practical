# unified-marl

**当前本地实用首版：0.9.0.dev1**。算法与环境独立注册，通过 YAML 选择组合；新增实验矩阵、能力检查、MPE `simple_spread` 和可跨环境训练的插件示例。[开始使用](docs/PRACTICAL_V1_GUIDE.zh-CN.md) · [新增算法](docs/ADDING_ALGORITHM.zh-CN.md) · [本轮验收与边界](docs/PRACTICAL_V1_REPORT.md)。此前已发布的 0.8.2 及其附件保持历史状态，本轮没有发布远端。

```powershell
python -m unified_marl check examples/experiments/plugin_mpe.yaml --instantiate
python -m unified_marl run examples/experiments/plugin_mpe.yaml
python -m unified_marl run examples/experiments/ppo_mpe.yaml
python -m unified_marl run-suite examples/experiments/practical_suite.yaml
```

在已安装本轮包的独立 Python 环境中，从项目根目录运行。实用矩阵自带 MPE 与参数化库存环境，交通需要 CityFlow。已实际验证 QMIX、VDN、SADMA-PPO、Huber-IQL 示例在库存、MPE、原生 CityFlow 上的 12 个短训练组合，包含保存加载、独立评估与恢复续训；不表示收敛、论文结果或真实业务收益。Huber-IQL 是教学变体。新增算法无需在 learner 中识别环境名称；连续动作、全部特色方法与其余环境后续逐项接入。

**0.8.2 工程修复版**：[安装与升级](docs/ENGINEERING_RELEASE_082.md) · [三轮验收范围](docs/VALIDATION_REPORT_082.md) · [0.8.2正式发布](https://github.com/saimeeee/unified-marl/releases/tag/v0.8.2-engineering.1)。首次安装复用0.8.0平台依赖包，再应用完整0.8.2补丁。



[中文使用说明书](docs/USER_GUIDE.md) · [下载安装包（Releases）](https://github.com/saimeeee/unified-marl/releases/tag/v0.8.0-engineering.1) · [支持范围](docs/SUPPORT_MATRIX.md) · [已知问题](docs/KNOWN_ISSUES.md)

**给同事安装使用，请从 Releases 下载对应 Windows 或 Linux ZIP，按包内 README 安装。** GitHub 自动生成的源码 ZIP 不包含平台依赖。接收方需要预先安装 CPython 3.12；两版核心依赖可离线安装；Linux 奖励模型功能另需联网补齐 hf-xet，电网扩展依赖和交通引擎按功能另装。两版均为 CPU 版本，Python 解释器、真实数据和仿真器另行提供。

以 PyMARL 架构整合三份 MARL 交付的 Python 包。当前核心修复版本 **0.8.2**：重点整理角色与分层均场方法、离线／偏好训练、奖励建模，以及 SADMA 本地／ZMQ 执行系统；通用算法作为对比基线保留。调用名称包含配置和实现变体，不代表独立算法数。真实数据更新与真实环境效果分别记录。

使用前查阅 [中文示例](examples/README.md)、[支持清单](docs/SUPPORT_MATRIX.md)、[验收报告](docs/VALIDATION_REPORT.md) 和 [已知问题](docs/KNOWN_ISSUES.md)。原 383 项迁移记录保留，另增 10 项工程应用与 5 项修复记录，包含未通过项；阶段结束不代表全部原实验、环境或平台服务已经迁移。

本版工程安装包位于 Releases 的 `v0.8.0-engineering.1`：Windows 使用 `windows/install.ps1`，Linux 使用 `linux/install.sh`。先安装 CPython 3.12，再按功能选择依赖并填写真实场景参数；详见 [工程安装说明](docs/ENGINEERING_DISTRIBUTION.md)。原 0.7.0 安装包与历史验收记录继续保留。

## 三个工程应用入口

[工程应用中文说明书](docs/APPLICATIONS_GUIDE.md) · [仓储库存补货](docs/WAREHOUSE_GUIDE.md) · [交通信号控制](docs/TRAFFIC_GUIDE.md) · [电网连续控制](docs/POWERGRID_GUIDE.md) · [Windows/Linux 安装](docs/ENGINEERING_DISTRIBUTION.md)

| 方向 | 输入 | 输出 | 实现方式 |
|---|---|---|---|
| 仓储补货 | 仓库/SKU、库存、需求、提前期、成本CSV | 各仓各商品补货数量 | 新增参数化库存环境，接入MFVFD/HMF等既有实现；原补货后端仍可选 |
| 交通控制 | CityFlow路网、车流、路口及相位映射 | 各路口信号相位 | 新增原生CityFlow.Engine适配，不必依赖缺失的cityflowenv封装 |
| 电网控制 | 配电网/建筑/气候及设备配置数据 | 各建筑连续控制量 | 适配来源GridEnv，新增bounded Gaussian PPO训练、推理和保存恢复 |

0.8.2第三轮用户报告中的仓储合成小样例、CityFlow官方单路口及case33bw配合公开CityLearn替代数据的最小链路通过。报告来自用户提供的Grok Bot盒子执行记录，不是本机复现；测试范围见[0.8.2验收说明](docs/VALIDATION_REPORT_082.md)。没有附行业预训练模型，准备真实数据后训练或加载自己匹配的模型。参数模板中的数字仅为格式示例。

```sh
unified-marl-app init warehouse --directory ./my-warehouse
unified-marl-app check --config ./my-warehouse/warehouse.yaml
# 准备真实数据后自行执行训练：
unified-marl-app train --config ./my-warehouse/warehouse.yaml --output-dir ./results/warehouse
# 模型生成后，输入当前库存即可计算补货决策：
unified-marl-app act ./results/warehouse/policy.pt --input ./my-warehouse/warehouse_state.json
```

同样通过 `init traffic` 或 `init powergrid` 获取另外两个方向的参数模板。Python 使用 `from unified_marl.applications import warehouse, traffic, powergrid`，三者均提供 train/evaluate/load_policy；具体数据字段和应用边界见各自指南。模型与 application.json 必须一同保留。

## 交付代码中的主要工作

本节重点介绍三份交付中的方法扩展、训练流程和系统能力。条目采用“方法或系统名称（交付者）”；姓名用于定位交付来源。通用对比算法及内嵌第三方方法统一列在文末，不按交付者姓名单独宣传为原创算法。这里介绍的是源码中可追踪的实现工作，原创归属仍应以论文及来源说明为准。

“本地可用”指限定配置下有效参数更新、保存加载等已通过。早期在线测试主要使用 toy，离线／偏好训练另有真实 3m 数据短更新；0.8.2另有用户报告中的仓储小样例、CityFlow官方单路口和指定电网场景短链路通过。真实 SMAC、原补货后端及上述应用的收敛与业务效果仍未验证。详见 [支持范围](docs/SUPPORT_MATRIX.md)、[0.8.2验收说明](docs/VALIDATION_REPORT_082.md) 和 [未完成项](docs/KNOWN_ISSUES.md)。

### RODE-HMF（王思哲）

作用：将角色分工与分层均场信息结合，用于具有分组交互的大规模协作。

状态：部分完成，调用 `wang/rode_hmf`；聚类前路径可运行，动态扩角色及后续完整流程仍未验收，来源优化器等问题保留记录。

### Hierarchical／HMF 系列（王思哲）

作用：将智能体组织为层级或分组，通过组内及组间信息进行协作，面向大规模补货等任务。

状态：部分完成。`wang_hmf/hierarchical`、`hierarchical_vdn` 等默认路径及若干固定分组变体可用；动态分组、高层调度和完整 VDAC actor 流程未接通。Hierarchical-QMIX、独立 IPC、ROMA 等实验仍有来源错误，不能把固定组验证当作整个系列完成。

### MFVFD（王思哲）

作用：结合均场信息与价值分解，学习大规模智能体之间的团队协作。

状态：`wang_hmf/mfvfd` 本地可用；真实补货任务与长期效果未验证。

### 离线与偏好训练流程（寇谦）

作用：从已有轨迹训练策略，统一两段轨迹、偏好标签、padding、终止掩码和团队／个体奖励字段。训练按更新次数推进，无需在线采样；可以将已训练奖励模型的预测用于历史离线价值学习。

状态：普通离线及偏好训练的已接入路径有真实数据短更新、保存加载与损失对照证据；参考策略生成个体标签、原交互反馈、在线采集、PER／e2e 等流程仍有未完成项。BC、CQL、IQL、ICQ-MA、CPL、DPPO 等已有方法作为实现基础或对照列在文末。

### MADPO／MADPO-AUX（寇谦）

作用：利用多智能体轨迹偏好及个体奖励信息分配学习权重；AUX 是附加辅助训练的变体。

状态：`kou/madpo`、`kou/madpo_aux` 可用，但必须提供真实 `agent_reward`，不能复制团队奖励来代替。

### BCMAPT（寇谦）

作用：将行为模仿与个体奖励加权结合，在已有轨迹上训练偏向较好行为的策略。

状态：`kou/bcmapt` 可用，同样需要真实个体奖励字段。

### Pref-QMIX（寇谦）

作用：将轨迹偏好约束加入 QMIX 类价值学习，使学习结果反映轨迹之间的优劣关系。

状态：不可用。两份历史版本缺少实际参考策略、个体终止或 agent_state／局部奖励协议，未完成统一训练接入。

### SADMA 本地与分布式训练系统（王思哲、易才润）

作用：将策略训练、环境采样、策略推理、缓冲区及 ZMQ 通信组织成执行系统；区分 Q 学习的回放数据与 PPO 的新策略轨迹，并传递策略版本、旧动作概率及模型参数。这里的主要工作是训练执行和通信集成，PPO、QMIX、VDN 本身作为通用算法列在文末。

版本关系：保留王思哲交付实现，以及易才润开源／闭源配置；易才润开源和 ZMQ 交付中的已比较文件相同。不同配置分别注册，不把重复版本计作新算法。

状态：本地同步及 Windows 本机 ZMQ 多进程短运行通过，涵盖参数同步、样本校验、故障超时和端口释放。Q／VDN 支持 sync／async；普通 PPO 支持 sync，async 要求显式 V-trace。跨机器、真实环境及连续动作电网路径未验收。

### 补货与其他环境交互（王思哲、易才润）

作用：保留交付中补货、SMAC、交通等任务的观测、动作、奖励及时间位置约定，通过统一接口对接外部环境。补货路径保留任务配置、奖励缩放、观测补齐及 episode horizon 等行为。

状态：适配器和模拟后端契约已接入；真实 ReplenishmentEnv、SMAC／StarCraft II、CityFlow 引擎及数据需另行提供，尚未完成真实短训练。电网连续动作接口仍未实现。

## 奖励建模与策略训练衔接（寇谦）

作用：把轨迹偏好转换为团队或个体奖励，为离线策略提供训练信号。交付包含多种奖励网络、全局／个体标签处理、分解式奖励和 diverse 信任筛选；MR、NMR、PrefTransformer 等网络作为已有模型对照，统一在文末说明。

### MultiPrefTransformerDivide（寇谦）

作用：结合全局偏好、跨轨迹个体比较及轨迹内排序标签学习个体奖励，支持用 KL 信任筛选组织 diverse 训练课程。

状态：基础训练、预测、保存加载及固定批次恢复更新已验证；diverse 仅初始阶段通过，后续可能出现空样本池，完整课程及恢复尚未通过。

### 全局奖励模型与历史训练分支（寇谦）

作用：`old_global` 与 `old_reward_global` 保留两条历史全局奖励建模路径，并向离线策略提供团队奖励；两条路径单独保存其奖励语义。

状态：两分支的全局模式训练、预测、保存加载与恢复更新通过；局部／个体奖励方法、参考策略标签生成及完整在线反馈尚未完成。历史目录保留，不因名称为 old_src 就认定废弃。

## 安装

在项目隔离 Python 环境中安装。CPU PyTorch 可按其[官方历史版本说明](https://pytorch.org/get-started/previous-versions/)安装，例如本项目验证使用 torch 2.6.0 CPU。

```powershell
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install .
```

开发安装使用 `pip install -e .`；发行包位于 dist。0.7.0 历史验证环境为 Windows / Python 3.12.14 / torch 2.6.0+cpu / NumPy 1.26.4 / PyYAML 6.0.2。其他版本未声称通过。

以下原核心调用示例沿用 0.7.0 的接口；新版安装文件请将版本替换为 0.8.0。直接安装交付包：

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install ./dist/unified_marl-0.8.0-py3-none-any.whl
# 按需要选择功能，避免修改全局环境：
.venv/Scripts/python.exe -m pip install "./dist/unified_marl-0.8.0-py3-none-any.whl[role,hmf,reward,distributed]"
```

包的 Python/依赖元数据是安装约束，不代表已验兼容范围；已验版本以验收报告为准。wheel 不含数据、checkpoint、缓存和外部仿真器。源码包另含中文文档、示例与验证工具；原始源码对照测试需要用户保留的参考目录，正常包调用不需要它们。

角色算法额外安装 `pip install ".[role]"`；HMF 额外安装 `pip install ".[hmf]"`。两类依赖按选择的算法加载。HMF 当前明确只支持 CPU。

奖励模型额外安装 `pip install ".[reward]"`，使用交付中的 transformers 4.53.0 和 ml-collections 1.1.0。仅创建本地配置与网络，不下载预训练权重。普通离线策略训练无需奖励模型依赖。

## 最小训练与评估

```python
from unified_marl import train, evaluate, load_policy

result = train(
    algorithm="pymarl/qmix", env="toy", output_dir="./results/my-first-run",
    overrides={"batch_size": 2, "buffer_size": 8, "t_max": 32},
)
print(result.updates, result.metrics, result.checkpoint)
print(evaluate(result.checkpoint, episodes=2))
policy = load_policy(result.checkpoint)
# 每个回合先 policy.reset()；随后 policy.act(obs, avail_actions)。
```

toy 是确定性测试环境，验证参数更新和接口，不代表 SMAC 或实际场景效果。已有运行输出目录会被拒绝，避免覆盖模型。

```powershell
unified-marl train --algorithm pymarl/qmix --env toy --output-dir ./results/qmix `
  --set batch_size=2 --set buffer_size=8 --set t_max=32
unified-marl evaluate ./results/qmix/policy.pt --episodes 2
```

COMA 使用新鲜 rollout，不使用离策略回放；batch_size 必须等于 batch_size_run。默认 COMA 使用 8 个并行环境，短验证可同时设 batch_size=2、batch_size_run=2、buffer_size=8。在 Windows 脚本中调用 parallel runner 时，入口放在 `if __name__ == '__main__':` 内。

## 配置和输出

配置优先级为：公共默认 < 算法 YAML < 环境 YAML < 用户配置 < overrides < 显式 Python 调用参数（如 env_args、seed、device）。嵌套 overrides 支持 env_args.episode_limit；CLI --set 的值使用 YAML 标量语法。算法组件组合会校验。

每次训练写出 config.json、metrics.jsonl、result.json、policy.pt、training.pt。TrainResult 返回这些路径、更新数、环境步数和 agent 参数变化量。

- policy.pt：推理所需配置、维度和完整 Controller 状态，包含角色网络、动作编码器和角色动作空间。
- training.pt：额外包含在线/目标网络、Mixer/Critic/Attention、优化器、更新计数、回放内容、全局 RNG 和选择器调度状态。

```python
continued = train(resume=result.training_checkpoint, output_dir="./results/continued", overrides={"t_max": 64})
```

恢复发生在回合边界。确定性 toy 的 QMIX 恢复已与连续训练逐项对照；外部仿真器内部状态不在该格式中，不能据此承诺真实仿真逐步精确恢复。policy.pt 不能作为继续训练状态加载。

## RODE、RODE-HMF 与 HMF 调用

`wang/rode_hmf` 和 `wang/rode` 使用统一 EpisodeBatch、Runner、Controller、Learner 和 checkpoint。`wang/qmix` 等保留交付配置；与原版逐文件相同的基线网络复用核心实现。HMF 来源的 QMIX 使用 `wang_hmf/qmix`，与这两类分别注册。

```python
result = train("wang/rode_hmf", "toy", output_dir="./results/rode-hmf",
    overrides={"runner": "episode", "batch_size_run": 1, "batch_size": 2,
               "buffer_size": 8, "role_interval": 2, "t_max": 16, "max_updates": 2})
print(evaluate(result.checkpoint, episodes=2))
```

HMF 可调用名：`wang_hmf/` 加 `dqn`、`vdn`、`qmix`、`mfq`、`mfvfd`、`hierarchical`、`hierarchical_vdn`、`hierarchical_vdn_IPC`、`hierarchical_vdn_without_high_level`、`maddpg`、`hierarchical_vdn_copy_fixed`、`hierarchical_vdn_meanq`、`hierarchical_vdn_k_means_fixed`。末三项来自缺少主入口分支的实验文件，已验证默认固定分组更新；`*_fixed` 不包含动态聚类调度。

```python
result = train("wang_hmf/mfq", "toy", env_args={"n_agents": 4, "mask_actions": False},
    output_dir="./results/mfq", overrides={"batch_size": 4, "buffer_size": 32,
        "hidden_dim": 16, "n_heads": 2, "embed_dim": 8, "t_max": 16, "max_updates": 2,
        "hmf_norm_rewards": False, "hmf_steps_per_update": 1, "hmf_num_batch": 1})
```

该短验证显式关闭奖励标准化并缩短更新间隔。HMF 默认保留入口的每 84 环境步更新、每次 2 个 minibatch 和探索率衰减。原 HMF 动作选择器没有可用动作掩码；遇到非法动作会明确失败，不能通过此接口承诺任意掩码环境支持。

HMF 默认按来源将时间上限算作 done；如需在截断处继续 bootstrap，显式设置 `hmf_bootstrap_time_limits=True`。PyMARL/RODE 保留各自原有的 timeout bootstrap 规则。

补货环境使用 `env="replenishment"`。需要外部 `ReplenishmentEnv`、真实任务配置和数据；`env_args.task_type` 指定任务，`update_config` 递归传入后端配置，`backend_args` 原样传给后端工厂，可提供其实际支持的外部路径参数。适配器保留 34 档补货动作、默认 21 天 horizon、团队奖励缩放 1e4、观测补齐。尚未完成真实补货训练，种子是否被后端消费也未验证。

RODE 的 SC2 专用动作聚类及动态扩角色之后的完整训练尚未通过；RODE-HMF 的均场网络优化器遗漏、IPC 损失广播等来源问题按原样保留。详细限制和 5 个未接通实验的具体原因见 [第 3 阶段报告](docs/STAGE3_REPORT.md)。

## SMAC

SMAC 按需导入。需要独立安装 SMAC、PySC2、StarCraft II 和地图，并配置 SC2PATH。基础包和 toy 不依赖它们。

```powershell
unified-marl train --algorithm pymarl/qmix --env sc2 --output-dir ./results/sc2 `
  --set env_args.map_name=3m --set batch_size=2 --set t_max=100
```

当前机器尚未完成真实 SMAC 短训练；安装或启动失败会明确报错，不会回退成 toy。

## 离线与偏好学习调用

普通离线训练：`kou/bc`、`kou/icq_ma`、`kou/iql`、`kou/cql`。偏好训练：`kou/cpl`、`kou/dppo`、`kou/madpo`、`kou/madpo_aux`、`kou/bcmapt`。后 3 项需要真实个体奖励 `agent_reward`，不能用团队奖励复制得到。

```python
from unified_marl import train, train_reward_model, load_reward_model

data = "D:/marl-data/3m.th"  # 替换为用户自己的数据路径；发行包不包含数据。
run = train("kou/bc", dataset=data, output_dir="./results/offline-bc",
    overrides={"training_updates": 2, "off_batch_size": 4,
               "dataset_options.max_length": 12})
pref = train("kou/cpl", dataset=data, output_dir="./results/offline-cpl",
    overrides={"training_updates": 2, "off_batch_size": 4,
               "dataset_options.max_length": 12})
print(run.updates, run.env_steps)  # 2, 0：不创建环境或伪造在线步数。
```

```powershell
unified-marl train --algorithm kou/cpl --dataset "G:/path/to/data.th" `
  --output-dir ./results/cpl --set training_updates=2 --set off_batch_size=4
```

数据通过 `EpisodeDataset` 加载，action/actions、avail_action/avail_actions 等别名转换后检查冲突、动作、掩码和奖励形状。提供的文件采用前后半段配对，`labels=1` 表示偏好后一段；支持显式 pair_indices。global_labels、between_labels 和 in_labels 单独保存，不能混用；标签侧文件仅在 dataset_options.labels_path 显式给出时加载。另已验证 `kou/bc` 的 `agent=rnn_ns`、`kou_old/qmix` 的 `agent=mlp` 和 `agent=mlp_ns` 配置变体，需显式设置 `hidden_dim`；不据此承诺任意 Agent/Learner 组合可用。

`off_batch_size` 沿用源入口的“两段轨迹总数”，偏好对数为其一半。原生 Learner 的时间参数使用更新索引；训练预算为 training_updates，调度周期为 schedule_updates，评估频率为 eval_interval_updates。默认不评估；周期环境评估必须显式提供 env/env_args，数据训练本身不创建仿真器。

old_src 中的 `kou_old/qmix`、`qplex`、`cw_qmix`、`ow_qmix` 已支持记录轨迹更新。`qmix_w_globalRM` 和 `qplex_w_globalRM` 可通过 reward_model_checkpoint 读取已训练的团队奖励，写入独立 reward_hat；CW/OW 的 globalRM 文件目前只验证了基础 TD 方法，完整在线奖励采集/无监督阶段尚未接入。

## 奖励模型

```python
reward = train_reward_model("MR", data, updates=2, batch_size=4,
    output_dir="./results/reward-mr", dataset_options={"max_length": 12})
model = load_reward_model(reward.checkpoint)
# model.predict(EpisodeDataset(...).batch([0, 1]))
```

支持 MR、NMR、PrefTransformer、MultiPrefTransformerDivide，以及 old_global、old_reward_global 两种历史全局奖励模型。MultiPrefTransformerDivide 可用 `mode="diverse"` 消费全局、跨轨迹个体和轨迹内排序标签，并执行原 KL 信任筛选。`batch_size` 在奖励训练中表示偏好对数。

`predict(..., mode="source_offline")` 保留源离线预测后处理：MR/NMR/PrefTransformer 输出团队标量（源代码在个体维取均值），MultiPrefTransformerDivide 输出个体奖励，历史全局模型输出团队奖励。`mode="native"` 明确请求网络的个体输出。不会为对齐形状自动求和或复制奖励。现有 3m 个体奖励文件只有 60 个时间位置，需要显式限制训练窗口，不能补造第 61 格奖励。

源 NMR 有复用第一段预测的问题；PrefTransformer 的 GPT2 blocks 未注册到原优化器且推理仍含随机 dropout。迁移保留这些训练行为，额外保存遗漏的层权重；相同随机状态下可核对加载预测。多标签信任筛选后续轮次可能选不到样本，会明确停止。详细通过项和未迁移项见 [第 4 阶段报告](docs/STAGE4_REPORT.md)。

```powershell
unified-marl train-reward --model MR --dataset "G:/path/to/data.th" `
  --updates 2 --batch-size 4 --output-dir ./results/reward-mr
```

## SADMA 本地同步训练

`sadma/qmix` 与 `sadma/ppo` 以王思哲交付的真实训练实现为主；`sadma/vdn` 为显式 VDN Mixer 变体。`sadma_yi/qmix`、`sadma_yi/ppo` 保留易才润开源配置，`sadma_yi_closed/qmix`、`sadma_yi_closed/ppo` 保留闭源配置。易才润开源的 qmix 配置实际使用 VDN Mixer，不能因调用名就认定网络相同。开源/ZMQ 两份比对文件逐字节相同，不重复计为算法。

```python
from unified_marl import train, evaluate, load_policy

for name in ["sadma/qmix", "sadma/ppo"]:
    result = train(name, "toy", output_dir="./results/" + name.replace("/", "-"),
        overrides={"train_batch_size": 2, "train_buffer_size": 8, "update_rate": 1,
                   "hidden_dim": 16, "t_max": 40, "max_updates": 2})
    print(result.metrics, evaluate(result.checkpoint, episodes=2))
    policy = load_policy(result.checkpoint)
    policy.reset()  # 每个新回合调用。
```

```powershell
unified-marl train --algorithm sadma/ppo --env toy --output-dir ./results/sadma-cli `
  --set train_batch_size=2 --set train_buffer_size=8 --set update_rate=1 `
  --set t_max=40 --set max_updates=2
```

Q 学习使用来源 episodic ReplayBuffer；PPO 收集当前策略的新回合、保存旧动作 log-prob/原始 critic 值，完成 mini_epochs 后消费并清空，不回放旧策略轨迹。t_max 是实际采样预算，max_steps 是原学习率衰减 horizon，两者分开；实际环境步数不包含 reset。完整训练状态保留优化器、目标网络/ValueNorm、计数器和 Q 回放；已完成更新的回合边界恢复在 toy 逐项对照通过，尚未消费的 PPO 回合会记录数量并丢弃，外部仿真器状态不保存。

本阶段运行器每个采集器一个环境，本地同步运行，不创建 ZMQ 连接。支持离散动作；连续动作 powergrid 明确阻塞，所需浮点动作 schema 扩展写入规格，未强行离散化。SMAC/补货仍是可选外部后端；CityFlow 延迟加载其 cityflowenv/引擎与地图。真实环境均未验证。PPO 原 GAE 对终止 bootstrap 的处理、熵项自归一化和递归学习率衰减等行为保留，不能将来源一致解释为算法正确或已收敛。详情见 [第 5 阶段报告](docs/STAGE5_REPORT.md)。

## SADMA ZMQ 执行

可选依赖安装 `pip install ".[distributed]"`，使用来源指定的 pyzmq 25.1.2。基础包和本地训练按需加载，不要求安装 ZMQ。

```powershell
unified-marl train --algorithm sadma/qmix --env toy --execution zmq `
  --config examples/zmq_managed.yaml --output-dir ./results/zmq-qmix
```

Windows Python 脚本必须将调用放在 `if __name__ == "__main__":` 保护内；完整脚本见 examples/stage6_zmq.py。采样、推理为独立 spawn 进程，默认只监听回环地址，退出时有界关闭并回收。设备、seed、worker 数、endpoint、超时和高水位均在 zmq 配置中明确。

QMix/VDN 支持 ZMQ sync/async；普通 PPO 支持 sync。PPO async 必须显式设置 `--set zmq.mode=async --set use_vtrace=true`，保留来源旧概率、版本和 V-trace 回放语义。其他来源的算法尚未接到 ZMQ，不按同名算法自动启用。

独立角色启动示例（同机三个终端，依次启动推理、训练、采样）：

```powershell
unified-marl zmq-infer --endpoint tcp://127.0.0.1:18401 --idle-timeout 60
```

```powershell
unified-marl train --algorithm sadma/qmix --env toy --execution zmq `
  --config examples/zmq_roles.yaml --output-dir ./results/zmq-roles
```

```powershell
unified-marl zmq-sample --trainer-endpoint tcp://127.0.0.1:18400 `
  --worker-id 0 --job-id local-demo --timeout-ms 30000
```

只在指定端口空闲时使用固定端口示例；managed 默认自动分配。三个角色需相同版本/协议、job_id 和兼容环境；修改地址并不代表已经跨机器验证。本机多进程及独立角色回环路径已验证，没有远程主机、真实环境或 GPU 验证。默认 Codex 沙箱禁止创建 ZMQ context，本次本机测试通过授权提升执行；普通终端仍受本机网络策略约束。

输出新增 execution-started.json、execution-events.jsonl 和 execution.json，记录已确认样本数、策略版本/参数摘要、正常停止或失败、进程关闭和强制终止情况。正常结束会排空在途回合，采样预算可能超出一个在途批次；恢复保留 Learner/回放，但不承诺重新启动 worker 后逐步完全相同。详见 [第 6 阶段报告](docs/STAGE6_REPORT.md) 与 [协议和执行约定](docs/ZMQ_PROTOCOL.md)。

## 来源和后续阶段

原版 PyMARL 的 Apache-2.0 与易才润 SADMA 开源/ZMQ 版的 MIT 许可文件已保留；王思哲、寇谦及易才润闭源配置的授权仍待确认，不能把同名或相似代码自动置于另一份许可证下。详见 [许可与来源](docs/LICENSES.md)、NOTICE 和包内 PROVENANCE.json。本轮在第 7 阶段完成交付验收后停止，Web/K8s 是未执行的可选第 8 阶段。

## 0.7.0 收尾补充

- `wang_hmf/hierarchical_vdac_fixed` 接通来源固定组 critic 更新；只转换旧 uint8 掩码，并跳过来源中不存在的 mf_policy 目标更新。没有新增策略网络，也没有接通原高层/VDAC actor 调度。
- `evaluate(..., seed=7)` 及 CLI `--seed` 支持指定种子，评估后恢复调用方的 Python/NumPy/PyTorch RNG。切换环境时先加载目标环境参数，再覆盖 env_args，并核对 checkpoint 维度。
- `load_reward_model(path, for_training=True)` 要求 reward-training.pt，恢复优化器、调度器、计数、额外未注册层与 RNG；六种模型的固定偏好批次下一次更新与连续更新一致。高层 diverse 信任池/课程恢复仍未实现。
- ZMQ 设置 `evaluation_interval_updates=1` 可启用独立快照评估，`evaluation_episodes=2`、`evaluation_timeout_s=45` 可配置。一次最多一个评估任务，忙时跳过并记录；评估故障/超时明确失败并清理。它不复刻旧永久 EvaluateWorker/共享缓冲 ABI。
- 嵌套覆盖键冲突、非法 seed/线程数以及离线恢复时被忽略的替代 config/algorithm 现会明确拒绝。

- [执行约定](WORK_ORDER.md)
- [当前状态](docs/STATUS.md)
- [整合规格](docs/INTEGRATION_SPEC.md)
- [源码盘点](docs/CODE_AUDIT.md)
- [迁移矩阵](docs/MIGRATION_MATRIX.csv)
- [阶段计划](docs/PLAN.md)

## 通用对比基线与内嵌方法

以下为交付中采用的已有算法、基础方法或内嵌第三方代码，集中用于对照和复用，不标为三位交付者的原创算法。调用命名空间仅用于区分实现来源；同名算法的不同实现和配置不互相替代。详细出处、版本及限制见 [来源追踪](docs/MIGRATION_TRACEABILITY.csv) 和 [支持范围](docs/SUPPORT_MATRIX.md)。

| 方法 | 作用 | 当前支持 |
|---|---|---|
| DQN | 离散动作价值学习基础对照 | wang_hmf/dqn 本地通过 |
| IQL — Independent Q-Learning | 各智能体独立学习动作价值 | pymarl/iql 本地通过 |
| VDN | 将个体价值相加构成团队价值 | PyMARL／HMF／SADMA 的已验配置可用 |
| QMIX | 用混合网络学习团队协作价值 | PyMARL／HMF／SADMA 和历史离线路径有已验实现；ICQ-MA 中同名行为约束实现未接通 |
| QTRAN | 约束团队价值与个体动作选择的一致性 | pymarl/qtran 本地通过 |
| COMA | 用反事实基线估计智能体对团队的贡献 | pymarl/coma 本地通过；使用新轨迹 |
| QPLEX | 更丰富的团队优势价值分解 | kou_old/qplex 离线通过；团队奖励模型路径可用 |
| Weighted QMIX — CW-QMIX／OW-QMIX | 对价值拟合样本加权 | 基础离线更新通过；globalRM 完整在线交互未接通 |
| PPO | 限制策略变化幅度的策略梯度训练 | SADMA 本地及 ZMQ 支持范围见上文 |
| MADDPG | 集中式价值评估指导各智能体策略 | HMF 离散动作路径通过；SADMA Policy 为空，未接通 |
| MFQ | 用平均行为概括大规模智能体交互 | wang_hmf/mfq 本地通过；无通用动作掩码支持 |
| MFAC | 均场 Actor-Critic 策略学习 | 来源缺接口且存在训练错误，不可用 |
| RODE | 先选择角色，再按角色动作空间决策 | wang/rode 聚类前通过；动态扩角色未完成 |
| BC — Behavior Cloning | 模仿数据中的动作 | kou/bc 真实数据短更新通过 |
| ICQ-MA | 基于固定轨迹的多智能体离线策略学习 | kou/icq_ma 基础路径通过 |
| CQL | 保守估计离线数据之外动作的价值 | kou/cql 基础路径通过；moderate-CQL／PER／e2e 未接通 |
| IQL — Implicit Q-Learning | 从固定数据学习价值和策略 | kou/iql 通过；与独立 Q 学习不同 |
| CPL | 用轨迹偏好直接优化动作概率 | kou/cpl 真实数据短更新通过 |
| DPPO | 交付中用偏好及动作概率构建策略目标 | kou/dppo 短更新通过；不是 ZMQ 分布式 PPO |
| HAPPO | 依次更新不同智能体的策略 | 仅内嵌 MAPT 源码，未迁移 |
| HATRPO | 顺序策略更新与信赖域约束 | 仅内嵌 MAPT 源码，未迁移 |
| MAT 系列 | 用 Transformer 建模智能体协作 | 策略训练未迁移，仅迁移奖励模型依赖闭包 |

奖励模型对照另包括：

| 模型 | 作用 | 当前支持 |
|---|---|---|
| MR | 从轨迹偏好学习奖励评分 | 训练、预测、保存加载及恢复更新通过 |
| NMR | 使用另一种奖励结构拟合偏好 | 流程通过；原损失复用首段预测，部分输入可能梯度抵消 |
| PrefTransformer | 用 Transformer 建模时间关系并预测奖励 | 流程通过；保留原未注册层／优化器及 dropout 问题 |

加上前述 MultiPrefTransformerDivide、old_global、old_reward_global，共保留六个奖励模型分支。模型可运行不代表奖励质量或策略收敛已验证。
