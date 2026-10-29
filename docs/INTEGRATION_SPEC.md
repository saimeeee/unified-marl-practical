# 本机网页交付增补规格（2026-10-29）

用户最新要求为“GitHub 下载代码后在本机网页运行”。交付包含完整仓库、四场景 assets/models bundle、CityFlow 源码和资源校验清单；平台运行时不依赖 SSH、远程机器、外部训练目录或未随包交付的场景资产。既有远程运行说明保留为历史，不能作为当前安装前置。

Windows 使用 Docker Desktop 的 WSL 2/Linux 容器与 Compose v2；容器内运行 CPU 模型推理、环境和 HTTP 后端，主机仅映射 `127.0.0.1:18765`，网页由同一后端提供。首次构建联网安装依赖，镜像构建完成后可复用本机镜像离线仿真。平台不要求 GPU，不修改全局 Python。浅色宽屏界面和现有控制协议保持。

`bundle/catalog.json` 登记默认模型；`prepare_local.py` 校验 bundle 文件、转换本地路径并生成配置与模型清单。`local-state/` 通过本机目录挂载保存运行、导出和额外模型；停止容器不删除此目录。额外模型采用 `local-state/extra-models.json` 的 `models` 数组，`checkpoint` 与非空 `config` 相对于并限定于该目录。新模型仍须满足对应环境的推理、观测和动作协议，不能仅改算法名称；新环境仍需适配器和画布。

本轮只改变分发、启动与资源定位，复用原训练权重、奖励和环境语义。平台为单会话推理，不是网页训练调度系统。四场景实际运行、本地完整源码归档构建及 GitHub 发布分别记录证据：本次从待发布 Git 索引归档到干净中文目录构建 Docker，公开 Python 3.12 Bookworm 基础镜像、PyTorch 2.6 CPU、随包 CityFlow 源码以及 `pip check` 通过；禁用外网的容器中四场景完整回合与控制检查通过。671 个资源文件校验和四份模型 SHA-256 一致性验证完成，见 `LOCAL_PLATFORM_REPORT.md` 与 `evidence/local-platform-verification.json`。GitHub 发布仍进行中，旧服务器结果保留为独立历史证据。

# 历史：服务器实时可视化增补（2026-10-29）

`demos/live-platform`为核心算法包外的单会话Web服务：CPU推理工作进程、四场景适配器、HTTP长轮询状态、同源控制与轨迹导出。管理员模型JSON与场景适配器独立；前端只消费状态JSON。继续复用原推理及环境语义，不把任意算法注册为全部环境可用。当前通过SSH隧道使用，未实现公网认证、多用户调度、网页训练任务或K8s。实测与范围见STATUS及平台README。

# 实用首版增补规格（2026-10-28）

0.9.0.dev1 增加 AlgorithmSpec / EnvironmentSpec，取消通用在线算法对环境名称的硬编码白名单，改为离散动作、固定智能体、团队奖励、全局状态及动作掩码能力契约。插件选择由明确的Python模块配置驱动；只扩展同/离策略在线采集，不替换旧离线/偏好与连续电网协议。

新增 run_experiment / run_suite / check_experiment 及 CLI run/run-suite/check/catalog。配置schema、路径、种子、数据/模型恢复与插件协议以 PRACTICAL_V1_GUIDE.zh-CN.md、ADDING_ALGORITHM.zh-CN.md 为准；验收与后续边界见 PRACTICAL_V1_REPORT.md。旧实验缺项及不同来源算法语义保持独立追踪。

以下是原整合规格的历史记录。

# 多智能体强化学习统一包规格

2026-10-24本地演示补充：CityFlow环境可显式配置可选 `max_green_time`/`max_red_time`，默认None保持旧行为及metadata；启用后动态掩码基于真实服务车道/队列/原生时间，训练、推理和step校验一致，规则属于环境工程约束，不是新的PPO损失。max_red按决策边界触发，不能当严格等待上限；配置进入模型兼容元数据。桌面三算法运行、效果边界与实测栈详见THREE_ALGORITHM_DEMOS.md。

状态：0.7.0 完成支持子集的交付验收，补充固定 HMF critic/离线 Agent 变体、奖励恢复和独立周期评估；383 行清单全部保留。跨机、真实环境及其他未通过项见 KNOWN_ISSUES，未缩减原范围。日期：2026-10-21（UTC+08:00）。

## 目标与范围

目标是交付 `unified_marl`，可通过 Python 与命令行完成训练、评估、推理、模型保存及加载。采用现有 PyMARL 架构；迁移来源、配置和变体通过 `MIGRATION_MATRIX.csv` 逐项追踪。

阶段 2—7 范围：原版 PyMARL 基线；王思哲 RGHMF 和补货 HMF；寇谦 ICQ-MA、old_src 的离线/偏好学习、奖励模型及数据处理；SADMA 同步训练与 ZMQ 执行。存在但未接通的研究代码保留清单，先厘清入口，不虚构实现。

阶段 8 可选范围：Web 后端/前端、K8s Operator 与部署联调。它们是独立服务，核心包不启动数据库、集群或网页服务。

本轮不要求复现论文收敛曲线、全参数搜索、多随机种子统计或改进算法效果。短训练证明链路与参数更新，不证明论文效果。

## 内部分发补充

用户在阶段 7 之后要求把已交付支持子集整理成发给同事即可安装调用的 Windows x64 与 Linux x86_64 两版，当前目标为 CPU / CPython 3.12。核心 wheel 保持 0.7.0，不修改算法。分发包应含各平台的固定依赖 wheel、带哈希的安装清单、完整性验证、隔离环境安装/调用脚本、最小训练检查、示例、源码归档及原有支持/未支持文档。Python 解释器和系统运行库是明确先决条件，不把本机虚拟环境复制到接收方。真实数据与外部仿真器仍显式由使用方提供。分发目录和结果见 INTERNAL_DISTRIBUTION.md；Linux 静态依赖校验与实机运行必须分别记录。

后续用户明确 Linux 只准备文件、不在本机运行测试。Windows 为完整离线包；Linux 对未获本机下载许可的 hf-xet 保留接收方联网补齐步骤，包名/说明不得标为完整离线，不更改算法范围。接收方自行执行安装与可选的训练检查。

## 源码边界

完整路径别名见 `CODE_AUDIT.md`。初始盘点 13 个项目根、615 个 Python 文件（含嵌入 MAPT 库）、139 个 config 目录下 YAML 文件、220 条注册赋值。第 6 阶段迁移清单 379 行，包含算法、配置、组件、来源重复项、数据和服务，不能解释为 379 个算法。

- PyMARL：核心模块参考；默认先离散动作、固定智能体数量、团队奖励。
- RGHMF：PyMARL 衍生，增加角色字段和网络；其基础 QLearner 与 PyMARL 文件相同，但训练编排有差异。
- HMF：独立 transition buffer / `update` / `init_from_env` 架构，需要专门适配器。
- ICQ-MA：离线批数据实际使用字典，Controller 和 QLearner 已改动，不能当作原版 QMIX。
- old_src：主入口仍分发 globalRM、pref、dataset 流程，属于活跃参考来源。
- SADMA：统一计算接口，但保留同策略/离策略数据生命周期和来源特有能力。
- MAPT：先追踪 RewardModel 实际引用闭包；嵌入的机械手等环境资产不因位于目录内就宣称全部支持。

## 目标模块

```text
src/unified_marl/
  api.py / cli.py / registry.py / config/
  components/        EpisodeBatch、transform、buffer
  controllers/       动作选择、隐藏状态管理
  learners/          参数更新与算法实现
  modules/           agent、mixer、critic、role、reward model
  envs/              环境适配器与能力描述
  datasets/          离线、偏好数据与路径映射
  runners/           环境交互、采样
  trainers/          在线离策略、在线同策略、离线、偏好训练
  execution/         local、ZMQ
  adapters/          来源数据与调用接口转换
deployment/          可选服务与部署材料
```

阶段 1 未创建占位代码。第 2 阶段已实现 api、cli、registry、config、components、controllers、learners、modules、envs、runners、utils、checkpoints；未将后续模块空目录计为完成。

第 3 阶段增加 algorithms/roles、algorithms/hmf；第 4 阶段增加 algorithms/kou（ICQ、old 两个来源命名空间及奖励模型引用闭包）、datasets、trainers/offline.py 和 rewards。来源文件含尚未接通的分支，是否支持以清单和运行证据为准。

## 公共接口

```python
train(algorithm, env=None, env_args=None, dataset=None,
      execution="local", device="cpu", seed=None,
      output_dir=None, config=None, overrides=None, resume=None) -> TrainResult
evaluate(checkpoint, env=None, env_args=None, episodes=20,
         device="cpu", seed=None) -> EvaluationResult
load_policy(checkpoint, device="cpu") -> Policy
policy.reset()  # 回合边界重置隐藏状态
policy.act(obs, avail_actions=None) -> actions
```

`TrainResult` 至少包含实际最终模型位置、更新次数、采样环境步数、配置快照和指标路径。离线训练以更新次数终止，不伪造采样步数。算法名先保留如 `pymarl/qmix`、`wang/rode_hmf`、`kou/cpl`、`sadma/ppo` 的来源标识；精确清单在各阶段冻结。

命令行调用相同业务接口，不在库内解析外部 `sys.argv`。导入不创建环境、进程、连接或训练日志；取消和异常均释放资源。

## 环境与数据约定

初始环境能力：固定 N、离散动作数 A、团队标量奖励。支持矩阵按算法/环境/执行方式明确，连续动作或个体奖励需单独扩展和验证，不能强行压成团队标量。

沿用 MultiAgentEnv 的 `reset`、`step`、`get_obs`、`get_state`、`get_avail_actions`、`get_env_info`、`close`。step 的内部标准含 reward、terminated、info；适配器单独保留 truncated，训练器依据算法决定 bootstrap。不能把 timeout 全当真实终止。

EpisodeBatch 的公共布局为 B（批次）、T（转移）、N（智能体）、A（动作）。保留 T+1 状态/观测用于 bootstrap；最后一格不当作有效转移参与损失。

| 字段 | 单时间步形状 | 语义 |
|---|---|---|
| obs | B,N,O | 局部观测，含终态观测 |
| state | B,S | 全局状态，含终态状态 |
| actions | B,N,1 | 离散动作整数 |
| avail_actions | B,N,A | 可用动作掩码 |
| reward | B,1 | 团队奖励；个体奖励保留独立字段 |
| terminated | B,1 | 真实终止 |
| truncated | B,1 | 时间上限等截断 |
| filled | B,1 | 沿用 PyMARL 的已占用时间槽标记，含最终 bootstrap 状态槽；损失只使用 [:,:-1] 转移掩码 |

统一 schema 的扩展字段支持 roles、role_avail_actions、均场统计、行为策略版本、旧策略概率和价值估计。具体 dtype、hidden state reset、转换时间索引须写入适配测试。

SADMA 的 `rewards`、`available_actions`，寇谦数据的 `action`、`avail_action` 分别映射；映射不只重命名，还核对 t 与 t+1 位置、dtype、掩码和奖励含义。

PreferenceBatch 显式包含轨迹 0、轨迹 1、各自长度/掩码、标签类型、标签方向及智能体层级标签。现有 3m 数据元数据出现 labels、global_labels、between_labels、in_labels；不能默认为同一类二元标签。先验证原配对规则，再重排或切分。

第 4 阶段实现：EpisodeDataset 接受 tensor-only .th 或内存映射；统一 aliases 并拒绝冲突。团队 reward 为 [E,T,1]，agent_reward 保持 [E,T,N,1] 独立字段，允许真实文件时间轴 T-1 但必须显式裁切训练窗口，不补造末格。DataBatch 的 transition_mask 采用 filled/terminated 再截去末槽；偏好 mask 沿用原记录的占用槽与终止约定，含终止转移、排除之后槽，未终止/截断的最后 occupied slot 可能仍参与原 CPL 打分。前述“最后一格不参与转移”是 TD 约定，不能用于擅改 CPL 语义。

PreferenceBatch 包含 trajectory0、trajectory1、各自 mask/lengths、pair_indices、label_key、labels 及独立 extra 标签。默认前后半配对，可显式给 pair_indices；labels=1 偏好第二段、0 偏好第一段、0.5 为平局。in_labels 是名次排列，不能改造成全局二元标签。标签 sidecar 必须显式指定。现有 3m 的 agent_reward 是交付预测文件，不声称环境个体真值。

## Learner / Trainer / Execution

寇谦离线训练预算为 training_updates，调度 horizon 为 schedule_updates，来源 train 的 t_env/episode_num 参数接收更新索引；返回真实 env_steps=0、episodes=0。ICQ off_batch_size 为两段轨迹总量，奖励模型 batch_size 为偏好对数。Kou 配置优先级为公共默认 < 来源 default < 来源 alg YAML < kou_runtime 运行配置 < 用户 config < overrides < 显式调用参数。e2e、moderate_lambda、use_per、use_reward_hat 的额外协议未接通时明确拒绝，不静默降级。eval_interval_updates 默认 0，环境评估须显式提供 env/env_args。

奖励 API 为 train_reward_model(model_type, dataset, updates=..., batch_size=..., output_dir=..., dataset_options=..., model_config=..., mode=...) 和 load_reward_model(path, device="cpu")。保存 reward.pt 与 reward-training.pt，分别记录预测网络与完整来源优化器/调度器/计数器。predict(mode="source_offline") 保留源后处理：MR/NMR/PrefTransformer 由源代码按 agent 维平均为团队值；MultiPrefTransformerDivide 保持个体值；旧全局模型输出团队值。mode="native" 显式请求底层个体输出。个体模型不得自动聚合成 old 全局 Learner 的 reward_hat。

PrefTransformer 未注册 blocks 的权重额外保存，训练仍保持原优化器遗漏和随机 dropout；加载预测以同一随机状态比较。NMR 的首段预测复用、diverse 课程的空筛选等来源限制见 STAGE4_REPORT.md；不得按支持名单推断这些分支算法正确、可收敛或长训可用。

拟统一 `learner.update(batch, context) -> metrics`；`context` 区分 update_step、env_steps、episodes、policy_version。保存加载及设备迁移为明确方法。原损失实现经适配保留，不直接套用三参数 `train`。

- 在线离策略：采样与回放独立，目标网络/混合网络按原规则更新。
- 在线同策略：保留行为策略、旧概率、优势估计与 rollout 消费周期。
- 离线：使用数据迭代器，环境仅在显式评估时创建。
- 偏好/奖励：轨迹配对、奖励训练与策略训练各有步数与数据约定。
- ZMQ：通信和工作进程生命周期属于执行层；策略陈旧性、策略版本以及 on-policy 约束必须绑定到具体算法。

PyMARL 的 parallel Runner 是多环境执行，不等价于跨机器 ZMQ。

## 配置、依赖和产物

配置优先级拟定：公共默认 < 算法配置 < 环境配置 < 用户配置 < Python/CLI 显式覆盖。嵌套字典递归合并；加载后验证算法组合与字段；保存最终解析配置。

配置随包分发并按包资源读取。数据、checkpoint、SC2 安装、CityFlow 配置、补货和电网数据采用显式路径，不依赖 cwd 或作者 `/home`、`/share`、`/data` 目录。

训练环境建立在项目隔离环境中。第 2 阶段因当前可用解释器为 Python 3.12.14，改为实际验证该版本 + torch 2.6.0+cpu + NumPy 1.26.4 + PyYAML 6.0.2；.venv 用于开发，.venv-check 用于干净 wheel 安装。没有修改应用/全局 Python 包。寇谦 pymarl.yml 的 Python 3.10 / torch 2.1 仍仅是来源描述，未宣称兼容；CUDA/GPU 训练尚未验证。

### 第 2 阶段明确实现决策

- 为保留原算法时间语义，filled 沿用占用槽约定：T 个有效转移和 1 个 bootstrap 状态槽。Learner 使用 [:,:-1]，不会把末状态算成转移。truncated 独立记录，timeout 继续 bootstrap。
- COMA 仅消费当前策略的新鲜 rollout，batch_size 等于 batch_size_run；原版默认 batch_size=run=buffer=8 的更新语义保留。
- QTRAN 保留所提供源码的隐藏状态布局和损失；固定输入与原实现对照通过，不将迁移对照解释为论文公式审查。
- 数值兼容修改仅包括 padded policy 归一化下界、空动作选择报错、tensor 构造、buffer 覆盖时清理旧 padding 和 extend 修复；有效数据上的五算法对照误差为 0。
- 当前可选 SMAC 延迟导入，未安装/未配置时明确失败；测试不会偷偷切换成 toy。
- 完整训练状态明确恢复在线/目标网络、优化器、计数器、回放与全局 RNG；外部环境内部状态不保存。确定性 toy QMIX 的回合边界恢复已验证逐项完全一致。

保存分为推理模型与继续训练状态。后者明确包含在线/目标网络、Mixer/Critic/奖励模型、优化器、计数器、随机状态及必要算法状态。是否保存 replay buffer 和环境状态需声明；不承诺未经验证的逐步精确恢复。

## 验收标准

### 第 6 阶段：ZMQ 执行与兼容边界

execution 层复用阶段 5 稳定算法/数据接口，拆分单一 Learner 协调端、环境采样进程、参数副本推理进程和消息协议。`execution="local"` 仍走原本地同步路径；`execution="zmq"` 启用新协议。配置、消息、形状、生命周期完整约定见 [ZMQ_PROTOCOL.md](ZMQ_PROTOCOL.md)，启动示例与逐配置验收见 [STAGE6_REPORT.md](STAGE6_REPORT.md)。

| 算法/来源 | local | ZMQ sync | ZMQ async | 本阶段限制 |
|---|---|---|---|---|
| SADMA QMIX/VDN | 已有验证 | 7 个来源配置中的 Q 配置通过 | 王版 QMIX、易开源 Q 配置通过；其余配置接口允许，未逐一异步运行 | 原回放/TD-lambda/更新节奏保留 |
| SADMA 普通 PPO | 已有验证 | 王、易开源、易闭源配置通过 | 明确拒绝 | 新数据一次消费，回合屏障后同步参数 |
| SADMA PPO use_vtrace=true | 已有分支验证 | 允许；未单列该 ZMQ 组合验收 | 王版通过；其他配置未逐一异步运行 | 显式原 V-trace + 旧 log-prob + 回放，不自动转普通 PPO |
| PyMARL、RODE/HMF、寇谦离线/偏好 | 按前阶段已验范围 | 未接入 | 未接入 | 同名 QMIX 不自动映射 SADMA ZMQ |
| SADMA MADDPG/连续 PowerGrid | 仍阻塞 | 仍阻塞 | 仍阻塞 | 空 Policy / 未实现连续动作 schema，不以网络分支存在代替完整支持 |

新 protocol=1 用 JSON 与原始数值数组帧，含 job/schema/worker/request/session/time/version；不接收 pickle，不兼容旧 InferClient 二进制布局。训练前核对 dtype、shape、有限性、动作/onehot/availability、终止/padding、隐藏状态时序和逐步行为/转移摘要。参数 snapshot 含 actor/必要 critic，所有推理端确认版本与摘要后继续接收新动作请求。

sync 以完整回合屏障保证同一轮策略不变；async Q 保留离策略回放，async PPO 仅显式 V-trace 分支允许。max_policy_lag 默认 null 保留原历史行为；显式设置才丢弃/跳过过旧数据。行为版本按原 actor train/mini_epoch 计数，不用外层 updates 替代。source PPO 的 critic 重算、原 GAE/熵/LR 限制继续保留。

每个 worker 至多一个在途回合与一个 RPC；HWM、帧/消息总长、回放容量、worker 数和等待时间有界。Windows 使用 spawn，不修改全局启动方式；socket/context 不跨进程/线程共享。正常结束停止发许可并排空；异常先 STOP/关闭/join，强制终止自有进程会记录 forced，不能据此承诺后端 close 已执行。外部角色按停止确认和超时处理，不能跨机强杀。

env_steps 是已通过后续消息确认的物理转移，reset/未执行动作/纯推理不计；完整回合、接受/丢弃回合、部分确认步单列。正常排空可能超过目标一个在途批次。checkpoint 恢复 Learner/回放/计数后重新启动 worker，不承诺 worker RNG、异步调度或外部环境逐步精确一致。

当前验证为 Windows / Python 3.12.14 / PyTorch 2.6.0 CPU / pyzmq 25.1.2 / libzmq 4.3.4 的回环多进程和独立 CLI 角色。没有可用远程主机，不进行跨机器验证；GPU、Linux、真实仿真器、原持续 EvaluateWorker、多 Learner 和旧 IPC SharedBuffer 路径未验/未迁移。默认工具沙箱禁止创建 ZMQ context，本次本机 TCP 验证在任务授权提升后执行，不能宣称默认沙箱通过。

### 第 5 阶段：SADMA 本地同步约定

选型和逐版本差异见 [SADMA_VERSION_SELECTION.md](SADMA_VERSION_SELECTION.md)。核心类来自王思哲 SADMA，易才润开源/闭源配置分别按资源保存；开源/ZMQ 逐字节重复来源保持追踪。主调用为 sadma/qmix、sadma/ppo、显式 sadma/vdn，另有 sadma_yi/ 与 sadma_yi_closed/ 配置档案。未启动旧 entry.py 或分布式 worker。

- 共享 EpisodeBatch 使用 state/reward/avail_actions，原 SADMA Policy 输入由 native_batch 转成 states/rewards/available_actions；actions 从 [B,T,N,1] 显式 squeeze 为 [B,T,N]，旧动作 log-prob 同样转换。clone 防止来源原地 mask 修改污染回放。团队奖励保持 float32 标量；来源 SMAC scheme 曾用 float64，本次数值一致范围是已测输入与 dtype，不声称所有未测精度逐位一致。
- reward_t、terminated_t、truncated_t 是执行 action_t 后得到的转移字段；obs/state/action/可用动作/策略输出 hidden/log-prob 位于 t，bootstrap 观测与动作位于 T。原 EpisodeBuffer 的到达式 ts-1 写入只在来源对照中使用；规范化后不二次减一。individual_rewards 必须来自实际 info，写入相同 t，不用团队值补造。PPO 来源算法内部对团队奖励重复到各 agent 的目标计算保留为算法行为。
- hidden_states 为 float32 [B,T+1,N,H]，保存该时间步动作前向后输出的隐藏状态；下一步消费前一步隐藏状态，回合开始为零。action_log_probs 为 float32 [B,T+1,N,1]（离散动作），behavior_values 为 float32 [B,T+1,N,1]（原始 critic 输出），policy_version 为 int64 [B,T+1,1]。PPO 在更新前仍按来源重算冻结 old_values；同步未改变 critic 的前提下已与采样存值逐项一致。
- Q 使用来源 episodic ReplayBuffer：完整回合插入、环形覆盖、无放回抽样、按 filled 最大长度裁切；PPO 只积累当前策略的新回合，满足 train_batch_size/update_rate 后全部消费并清空，mini_epochs 仅在这次更新内重复使用。批中版本必须等于当前 actor 版本，不允许旧轨迹进入后续同步更新。PPO 版本按原每个 mini_epoch 增加，不等同外部 updates。
- t_max 为实际 step 预算，max_updates 为可选外部更新数上限；max_steps 独立保留源学习率调度 horizon。episodes 是完整回合数，update_rate 和 target_update_interval 继续按来源回合数触发。reset 不计入 env_steps。源递归 lr_decay、Q/TD-lambda、GAE/ValueNorm、clipping、dual_clip/vtrace 和熵目标不擅改；已知来源限制写入报告。
- 配置优先级：公共默认 < 对应来源 default < 对应来源算法 YAML < 来源环境 YAML（非 toy）< SADMA 运行默认 < 公共 toy 资源（仅 toy 的 env/env_args）< 用户 config < overrides < 显式 API 参数。运行默认只规定本地同步/接口/设备，不统一各来源数学超参数。agenet_type 不被当作 agent_type 使用。补货源包装忽略的 key/map_name/reward 保留为 metadata；horizon=None 使用实际后端值，HMF 既有默认保持 21。
- 当前 SADMA 采集器单环境、单进程本地同步；env_batch_size/batch_size_run 必须为 1，async_train=true 和非 local execution 明确拒绝。核心 PyMARL 的既有 parallel Runner 仍可单独使用，SADMA 分布式 worker 与 ZMQ 留第 6 阶段，不因文件已复制就算实现。
- policy.pt 包含 actor 和策略版本，training.pt 额外保存目标网络/critic/mixer、ValueNorm、优化器、递归 LR、目标计数、全局随机状态和 Q 回放。PPO 已完成更新的 toy 回合边界恢复与连续运行对照通过；预算结束时尚未消费的 PPO 回合记录数量并丢弃。外部环境内部状态和旧 agent.pth 文件夹的兼容导入未验证。
- 环境正常完成、训练异常和元信息构造异常均释放已创建环境；本地路径不改变全局 multiprocessing start method、不创建通信端口。CityFlow 按需加载外部 cityflowenv/引擎，并在 backend 提供 close 时转发；模拟后端通过不等于真实环境验证。

### 连续动作扩展方案（设计已记录，实现阻塞）

易才润 PowerGrid 返回 Box 动作空间，源 get_env_info 中 action_shape=(4,)，PPO 网络有 Normal 分支。此能力不进入离散 OneHot/Categorical 接口；当前 env=powergrid 或 env_type=continuous 明确报错。SADMA MADDPG policy 文件为空，不能借网络文件替代策略实现。

后续连续扩展需要同时实现：可序列化 action_spec={type:continuous, shape:[D], low:..., high:...}；float32 actions/last_actions [B,T,N,D]；action_log_probs [B,T,N,D] 并按源 PPO 对 D 求和一次；去除 OneHot 和离散 avail_actions 选择，另定义连续动作有效维掩码；Policy.act 返回浮点向量，Runner 校验实际 bounds、策略分布与后端预处理的一致性；保存 Gaussian 的 log_std、优化器与原行为概率；明确裁剪/缩放及 log-prob 修正语义，不能默认加 tanh 或离散化来改变算法。还需实际 pandapower/气候/网络资产、异构 agent 空间与 seed/关闭协议的验证。

该段仅为扩展约定，没有把设计或拒绝测试计作连续算法迁移完成。真实 SMAC、补货、CityFlow 与 PowerGrid 及策略效果仍单列未通过。

### 第 3 阶段已确定的适配约定

- RGHMF 使用 `wang/` 命名空间，HMF 补货使用 `wang_hmf/`。同名 QMIX 不覆盖 `pymarl/qmix`。复用原版基线的依据是源码字节对比，而非名称；来源基本动作选择器的调度器签名缺参问题通过选择已验证核心选择器解决，来源错误另留证据。
- 角色字段沿完整时间轴存储：roles 为 int64 `[B,T+1,N,1]`，role_avail_actions 为 int32 `[B,T+1,N,A]`，agent_Q 为 float32 `[B,T+1,N,A]`。t 状态选择的角色/动作/Q 写在 t；t=0 及 t%role_interval=0 重选，期间保持角色；最终 bootstrap 状态也保存角色和动作。回放按原字段整体保存，不在 Learner 重算采样时的 agent_Q。
- MFQ 的 mean_actions 为 float32 `[B,T+1,N,A]`，记录 t 原生策略输出的跨智能体均值并复制到 N 个智能体；推理 t 消费上一时刻均值，回合开始为零。训练按原 buffer 约定消费 mean[t] 和 mean[t+1]。若原生输出在并列最大值时为多热，保存原输出均值，不擅自重新归一化。
- HMF 原转移采样桥接 EpisodeBatch，固定分组按原智能体顺序连续划分；每组人数相等。团队奖励按来源 wrapper 复制给各智能体，奖励标准化使用当前回放均值/总体标准差，常量奖励明确报错。`hmf_norm_rewards=false` 是显式配置，不是偷偷加 epsilon。
- 补货工厂按需导入、数据/任务配置从调用参数传入。34 档动作、21 天默认 horizon、padding、奖励除以 1e4 保留。原 wrapper 记录 seed 但注释掉实际 seed 调用，当前适配器也不能承诺外部环境确定性。
- 原自动 reset 后跨回合读取 next_obs/next_action 的行为改为显式同回合 t+1；timeout 单列 truncated。PyMARL/RODE 继续 bootstrap；HMF 默认保留来源将时间上限当 done 的目标掩码，仅在显式 hmf_bootstrap_time_limits=true 时继续 bootstrap。这是有记录的转移边界适配，不宣称复刻旧 main 的跨回合错误采样。
- HMF 暴露原固定入口的更新间隔、每轮 minibatch 数和探索衰减。GPU、动态分组主程序、MFAC 的同策略协议及 VDAC 的多种高层更新调度尚未接通。
- RODE-HMF 均场网络未加入优化器、下一分组索引使用当前分组、目标均场使用在线 critic，以及 IPC 的广播损失均按来源保留，数值对照不等于理论正确性验证。动态扩角色的新增参数未纳入原优化器，扩展后完整训练恢复明确拒绝；长期训练验收不通过。
- 原 PyMARL/角色/HMF 支持的推理模型和完整训练状态分开保存。角色保存附加网络、角色动作空间/嵌入、两类选择器调度；HMF 保存原生全部网络、优化器和转移回放。确定性 toy 的阶段 3 恢复范围为回合边界、聚类前。

所有迁移项保留迁移状态、实现验证、本地测试验证、真实环境验证。第 4 阶段另增合成测试、真实数据、策略效果三列；历史未记录的新列不追溯标为通过。

1. 干净环境中构建/安装，在源码目录外导入和调用，配置资源可读，不引用桌面或原交付目录。
2. 小型确定性环境中执行采样→有效批次→有限损失→参数更新；检查非法动作、padding、终止和隐藏状态。
3. 固定输入下对照原实现关键输出和损失，记录种子、误差界与合理差异。
4. checkpoint 重新加载后能够评估；声称支持训练恢复的项验证必要状态。
5. 真实环境短训练/评估单列证据；模拟环境、静态检查或组件冒烟不能替代。
6. 真实数据读取、配对、标签、奖励含义有证据；第 4 阶段已验证 12 份轨迹小批读取和 3m 短更新。其他地图参数更新、真实环境及效果仍须分别验收，归档元数据扫描本身不能算通过。
7. ZMQ 本机多进程、故障、停止、资源释放分别验证；跨机器单列，未运行不得标通过。
8. 未接通或缺失实现、缺失资产和外部服务保留阻塞项，不能用占位实现消除。

## 第 7 阶段最终接口与交付约定

1. 配置仍按已记录的各家族来源默认合并；显式 env_args/seed/device 优先。device 默认 cpu，seed=None 采用最终配置（默认 1）。seed 要求整数 [0,2**32)，num_threads 为正整数，覆盖键不允许空层级或父/子冲突；离线恢复拒绝替代 config 和不同 algorithm。
2. evaluate 可传 seed；不传时采用 checkpoint 的 seed。保存并恢复调用方三类 RNG，切换 env 时读取目标环境默认参数，再合并显式 env_args，并严格核对 metadata。CLI 同样提供 --env、--env-args 和 --seed。
3. wang_hmf/hierarchical_vdac_fixed 仅调用来源 update 的固定 critic 路径与实际存在的两个目标 critic。uint8 转 bool 保留被掩码元素；long 索引分支不转换。缺失 mf_policy 的目标更新被明确跳过，原高层/actor 方法仍不支持；此变体不等于完整 VDAC。
4. 三个离线 Agent 配置变体已补验：kou/bc+rnn_ns，kou_old/qmix+mlp/mlp_ns。保留来源拼接和损失行为，不擅自推广至 GRNN、MADDPG 或其他 Learner 组合。
5. load_reward_model(...,for_training=True) 只接受 reward-training，恢复完整来源状态与 RNG；六模型的固定 pair 下一更新已对照。高层 train_reward_model 的 diverse 信任池/课程没有高层 resume 接口，不声明完整恢复。
6. ZMQ 可选 evaluation_interval_updates（默认 0）、evaluation_episodes（默认 2）、evaluation_timeout_s（默认 45）和 evaluation_device（默认 cpu）。不可变策略快照由一个独立 spawn 进程评估；最多一个 pending，忙时跳过并记录。错误/超时使任务失败，自有进程退出有界。这是周期能力替代，不复刻旧共享缓冲/vector/perpetual worker ABI。
7. 发行包带资源和 PROVENANCE；源码包另带规格、追踪、中文示例、测试工具及紧凑证据，不带模型/数据/结果/缓存。许可证分别保留，未明权利继续标记。SMAC extra 采用交付 requirements 的官方 Git 地址，但没有测试该外部安装/运行版本。
8. 已验平台仅 Windows 11 / Python 3.12.14 / torch 2.6.0+cpu；元数据约束不扩展已验范围。383 行矩阵不删除未通过项，阶段 7 结束后停止，全部未通过项见 KNOWN_ISSUES。

## 0.8.0 工程应用授权与接口

用户后续授权 warehouse / traffic / powergrid 三个应用入口，完成打包后自行测试。本轮新增参数化库存模拟器、直接 CityFlow.Engine 适配、来源 GridEnv 包内适配与独立 bounded Gaussian PPO，并提供应用train/evaluate/load_policy、业务动作转换、模板、CLI与双平台包。不执行训练、真实/合成环境实验；仅静态、配置、导入、构建和归档检查。旧算法动态分组、扩角色和缺失来源项不会因本轮新增实现自动转为完成。完整接口与输入单位见 APPLICATIONS_GUIDE.md。

## 0.8.1 恢复及评估契约补充

core train恢复时，省略env_args字段沿用checkpoint默认值（包括训练时注入的seed），显式值必须与已保存值一致；应用层仍要求相同场景。显式改seed、场景或用推理checkpoint恢复仍拒绝。EvaluationResult新增episode_lengths与env_steps，计数Runner实际转移，不含bootstrap/reset/纯推理。PowerGrid独立评估接口未改变。

## 0.8.2电网与分发契约

PowerGrid episode_limit计外部控制transition，reset预热保留但不占控制预算；info明确control_steps、initialization_steps、episode_limit。checkpoint.env_info记录control_transitions_v2，旧语义恢复/环境评估拒绝，纯推理不依赖该环境计数。新加装入口不得请求旧核心条目或无意改写已装构建工具；升级校验原清单、备份原脚本和清单，并对更新登记精确SHA256，不通过刷新损坏文件哈希绕过验证。
