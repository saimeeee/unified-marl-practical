# 第 1 阶段源码盘点结论

盘点日期：2026-10-20（UTC+08:00）。本报告依据工作区副本的源码、AST、SHA-256、归档元数据和当前会话入口探测。没有运行论文实验，也没有安装训练依赖。

## 来源路径及边界

以下路径相对于 `G:\AI\Codex\多智能体代码整合`。路径包含中文，后续工具须使用 UTF-8 和 LiteralPath。

| 项目标识 | 来源路径 | 角色 | Python 文件数 |
|---|---|---|---:|
| pymarl | pymarl-master/pymarl-master/src | 核心架构与基线 | 32 |
| wang_rghmf | marl平台代码/王思哲/code/large_scale/RGHMF | PyMARL 角色/均场扩展 | 51 |
| wang_hmf | marl平台代码/王思哲/code/large_scale/hmf_replenishment | 独立补货算法训练库 | 53 |
| wang_sadma | marl平台代码/王思哲/code/sadma | 同步/异步训练系统及算法 | 38 |
| yi_sadma_closed | marl平台代码/易才润/论文/SADMA/sadma源代码闭源版本 | SADMA 交付变体 | 41 |
| yi_sadma_open | marl平台代码/易才润/论文/SADMA/sadma源代码开源版本 | SADMA 开源来源 | 41 |
| yi_sadma_zmq | marl平台代码/易才润/基于ZMQ的分布式多智能体强化学习算法/sadma源代码公开版本 | SADMA 重复来源 | 41 |
| kou_old | marl平台代码/寇谦/论文代码及技术文档/论文代码及技术文档/old_src | 在线、偏好、全局奖励和数据生成 | 69 |
| kou_icq | marl平台代码/寇谦/论文代码及技术文档/论文代码及技术文档/ICQ-MA/src | 离线/偏好/奖励模型，含 MAPT 库 | 227 |
| wang_operator | marl平台代码/王思哲/code/operator | Go K8s TCJob Operator | 0 |
| yi_operator | marl平台代码/易才润/基于K8s的集群资源调度管理系统/operator源代码闭源版本 | Go Operator 与 scheduler 扩展 | 0 |
| yi_backend | marl平台代码/易才润/在线代码编辑调试微服务系统/tc-web-bf源代码闭源版/taichu-web-backend | FastAPI、SQL、任务服务 | 22 |
| yi_frontend | 同一 tc-web-bf源代码闭源版/vue-element-plus-admin | Vue3/ElementPlus/Vite 前端 | 0 |

共 615 个 Python 文件通过 Python 3.12 AST 语法解析；未检出 SyntaxError。该结论不证明依赖、API 或运行兼容。timehelper.py 中存在旧字符串转义警告，未修改原文件。

排除 `.git`、`__pycache__`、`__MACOSX`、`node_modules`、results、wandb、bin、.idea 等产物目录；源码 JSON、YAML 和第三方库仍在清单内。数据目录另行扫描。纸质/电子论文、软著和说明书只登记其存在，未提取全文，也未将其中说明当执行指令。本阶段以源码为主要依据。

## 入口、算法和配置

### PyMARL 与 RGHMF

- PyMARL 的 `main.py` 使用 Sacred；`run.py` 组装 registry、scheme、buffer、MAC、learner。注册 q_learner、coma_learner、qtran_learner，配置提供 QMIX、VDN、IQL、COMA、QTRAN 及 beta 变体。
- RGHMF 注册上述 3 个基线 Learner，以及 rode_learner、rode_hmf_learner。Controller 注册 basic_mac、rode_mac；额外动作编码/角色/角色选择网络和 scheme 字段需一并迁移。
- `qmix_beta` 等是配置变体，不逐个当成独立算法。旧 SC2 beta 配置是否仍有可用环境需明确支持边界。

### HMF 补货

- 两个入口为 `main_replenishment.py` 和 `main_replenishment_IPC.py`；通过 alg_type 分支实例化算法，没有统一算法注册表。
- algorithms 下有 18 个带算法类的 Python 文件。10 个文件在这两个 main 中确认存在匹配模块导入与直接 init_from_env 调用：DQN、Hierarchical、Hierarchical_VDN、Hierarchical_VDN_02_Formulate_MeanQ_IPC、Hierarchical_VDN_Without_High_Level、MADDPG、MFQ、MFVFD、QMIX、VDN。
- 另 8 个文件未确认该直接入口：hierarchical_qmix、hierarchical_vdac、hierarchical_vdn copy、hierarchical_vdn_02_formulate_meanQ、hierarchical_vdn_IPC、hierarchical_vdn_k_means、hierarchical_vdn_roma、mfac。属于“未接通待核实”，不能据此认定无用或补造算法。
- init_from_env/update/save/init_from_save 与 PyMARL 不同；输出根在 main 内硬编码为作者 Linux 目录。补货 wrapper 最终仍依赖外部 `ReplenishmentEnv.make_env`，交付中的同名 wrapper 不代表仿真器已包含。

### 寇谦 ICQ-MA 与 old_src

- ICQ-MA 有 10 个 Learner 注册项：q_learner、offpg_learner、bc_learner、bcmapt_learner、cpl_learner、dppo_learner、madpo_learner、madpo_aux_learner、cql_learner、iql_learner。
- `offpg_smac.yaml` 的 name 为 ICQ-MA，使用 offpg_learner；不能因目录名就额外虚构 icq_learner。
- 16 个算法 YAML 中，hacpl_smac、maicpl_smac、maidpo_smac 分别引用未注册的 hacpl_learner、maicpl_learner、maidpo_learner；当前 learners 目录也未发现对应实现文件。配置注册检查证据见 config_findings.json。
- acpl_smac 与 cpl_smac 都引用 cpl_learner；配置名字不等于独立实现。配置、args.name 与 run.py 分派是否匹配仍需第 4 阶段验证。
- old_src 有 10 个 Learner 注册项，包括 q_learner_new、qdiffer、pref_qmix、globalRM、QPLEX、weighted QMIX 等变体。主入口明确导入并分派 run_globalRM、run_pref_new、run_make_datasets；`old_src` 不能丢弃。
- old_src 的 run_pref.py 与 run_pref_new.py 均保留，主入口当前导入后者，前者不自动判作废弃。
- 外层 get_preferences.py、make_diverse_preference.py、use_llm.py 是数据/标签工具。仅分析其导入和结构，没有运行或调用外部模型服务。
- RewardModel 的实际分支包括 MultiPrefTransformerDivide、PrefTransformer、MR、NMR；依赖嵌入 MAPT 的 reward_model 子树与 transformers 等外部包。MAPT 包含大量其他环境资源，源码存在不代表它们属于已接通能力。

### SADMA

- entry.py 根据 async_train 和 train/sample 角色分派，默认同步 runner 另有分支。policy_REGISTRY 实际只注册 ppo 与 qmix。
- policy/maddpg 下存在网络和 Controller，但未被顶层 policy_REGISTRY 接入。已登记为未接通实验，不能当作可直接运行的第三个算法。
- Wang 注册 SMAC、Replenishment、CityFlow；Yi 各版增加 PowerGrid 及 runner。默认环境包全量导入，可能使只用 SMAC 的用户也受其他仿真依赖影响。
- worker 负责推理、训练、评估、本地/远程采样；utils.scheme 定义共享 buffer 和 ZMQ 发送数据。它们属于执行系统，不是另一个 PyMARL parallel Runner。

## 版本比较（静态结论）

| 比较 | 字节相同的 Python | 字节不同的共同 Python | 单侧额外 Python | 共同 Python AST 相同 |
|---|---:|---:|---|---:|
| Wang SADMA → Yi closed | 1 | 37 | Yi 多 3 | 27 |
| Yi open → Yi ZMQ public | 41 | 0 | 0 | 41 |
| Yi closed → Yi open | 40 | 1 | 0 | 41 |
| PyMARL → RGHMF | 22 | 10 | RGHMF 多 19 | 22 |
| PyMARL → old_src | 16 | 14 | 原版独有 2，old 多 39 | 16 |
| old_src → ICQ-MA | 8 | 18 | old 独有 43，ICQ 多 201 | 9 |

Yi open 与 Yi ZMQ public 本次扫描的全部 51 个源/配置/说明文件 SHA-256 相同，可建立共享实现与双来源记录；不删除原目录。Yi closed 与 open 全部 41 个 Python AST 相同，但文件集合、README、config 等仍有差异，不能将整个交付未经核实视为等价。

Wang 与 Yi 的差异不是只有换行：entry 的 spawn 设置时机、config、powergrid、replenishment runner、Controller、Q/PPO Policy、同步/异步 runner、worker、logger 等需评审。Wang Q/PPO 的 save_models/load_models 是重要增量。

PyMARL 与 RGHMF 的 `learners/q_learner.py` 字节完全相同。old_src 同名 QLearner 文件与原版主要可见差异为新增说明字符串，尚未据此断言完整行为等价。ICQ-MA 同名 QLearner 明确加入模仿损失、行为约束目标动作、额外网络输出和正则，不能合并为原版 QMIX。

两份 Operator 共同相对路径的 40 个已扫描源/配置文件均字节不同；Yi 多 39 项，包括 scheduler 和文档扩展。原始算法不受其部署逻辑约束；Go 文件字节不同尚不等于语义全部不同。

依据：version_comparisons.json、source_manifest.json、python_symbols.json；AST 对比忽略位置和注释，不证明依赖配置与数值等价。

## 数据与环境资源

- 发现 18 个 .th 数据/标签/奖励文件，合计 3,736,627,170 字节（约 3.74 GB / 3.48 GiB）。均位于寇谦数据交付，包括 MADiversePreferDatasets 的 3m/8m，以及 MAOfflineDatasets_new 的 10 个地图文件。
- 3m.th 大小 17,028,642 字节；3m 离线 200_60labeled.th 大小 17,020,173 字节，适合第 4 阶段首先做小数据读取。存在文件不代表可互相替换。
- 全部 18 个文件的 ZIP 目录可读，使用 pickletools 分析 data.pkl 指令中的字符串，无反序列化、无张量加载。3m.th 出现 obs/state/action/reward/avail_action/filled/terminated/agent_ids/labels/global_labels/between_labels/in_labels 等字符串；这些是候选字段证据，不是已验证 schema。
- reward.th 的元数据带 `cuda:0` 标记，后续 CPU 检查必须显式 map_location。张量形状、配对顺序、标签方向、奖励层级和数值内容都未验证。
- 配置引用作者 `/home`、`/share`、`/data` 绝对路径，不能直接运行；真实本地文件需显式映射。默认离线地图可能不是 3m，不能只替换文件而保留其他维度。
- SADMA CityFlow 使用 `cityflowenv.env.CityflowEnv`，不仅需要 cityflow 引擎。补货依赖外部 ReplenishmentEnv，电网依赖 pandapower 及气候/网络/agent 数据；交付 powergrid 源码目录未包含默认 data 子目录。
- 未在 SC2PATH、常见 Windows SC2 目录和工作区仿真安装位置确认 SC2 安装。本轮未穷举整机所有磁盘，因此结论为“当前会话未发现可用安装”，不是断言整机没有。

## 依赖与保存加载

- PyMARL requirements 锁定 torch 0.4.1、NumPy 1.15.2、PyYAML 3.13、Sacred 0.7.2。ICQ requirements 同属旧锁；外层 pymarl.yml 则记录 Python 3.10.18、torch 2.1.0、NumPy 1.25.0、PyYAML 6.0.2、Sacred 0.8.7、transformers 4.53.0。这些是来源环境描述，不是已验证整合依赖。
- Yi SADMA requirement 指定 gym 0.26.2、NumPy 1.22.4、pyzmq 25.1.2、pandapower 等；wrapper 却仍使用四返回值旧 Gym step，需按实际环境验证。该 requirement 未列完整 torch/外部仿真器依赖。
- PyMARL QLearner 保存 agent.th、mixer.th（按算法）、opt.th，并有 load_models；不应因此声称保存了所有 RNG、回放和计数器状态。
- RGHMF 基础 QLearner 同上；新增角色网络的完整恢复需独立核实。
- HMF 各算法有 save/init_from_save，main 写入 incremental 和 pursuer_model.pt；schema 与 PyMARL 不同。
- Wang SADMA Q/PPO 有 agent.pth、mixer.pth 或 critic.pth、opt.pth 的方法；Yi 版 BasePolicy.save_model 为 pass，未见对应完整保存加载路径。不能合并时丢失 Wang 的有效能力。
- ICQ-MA QLearner 有 save_models，但没有对应 load_models；run.py:345 附近的模型保存分支只创建目录并打印，未调用 learner.save_models。奖励预测另有 th.save；保存预测不等于保存策略或奖励模型状态。
- PyMARL、RGHMF、Kou 运行脚本存在 os._exit；库化必须去掉强制退出并按资源生命周期清理。原版与部分 SADMA 的 collections.Mapping、旧式 yaml.load 也需兼容处理。

## 外围平台

Backend 使用 FastAPI、SQLAlchemy/MySQL、Docker/Kubernetes 接口及 code-server、TensorBoard 功能。main 导入有日志、目录和其他检查副作用，本阶段未启动。前端 package.json 表明 Vue3、ElementPlus、TypeScript、Vite 和 pnpm。

Operator 两版 go.mod 声明 Go 1.19、K8s 0.25.x、controller-runtime 0.13；保留独立构建部署。没有运行集群、数据库或服务可用性测试。

许可元数据：原版 PyMARL 有 Apache-2.0 文件；Yi 两份开源/公开 SADMA 有 MIT 文件；其他交付不能仅凭目录名推断许可。该记录仅作来源追踪。

## 迁移清单解释

MIGRATION_MATRIX.csv 每行包含来源路径、能力、分类、目标模块、计划阶段、依赖、迁移状态、实现/本地/真实验证标记、静态证据与阻塞原因。

ALGORITHM_IMPLEMENTATION 表示发现实现，不表示新包已经实现；CONFIG_VARIANT 不重复计数算法；UNWIRED_EXPERIMENT 表示缺少确认入口或注册。DUPLICATE_IMPLEMENTATION 只依据明确范围的比较，不自动证明整个流水线一致。

所有迁移项当前 NOT_STARTED，三个验证列均 NO。43 条 local_import_findings 全部位于嵌入 MAPT 子树；它们是静态局部模块定位告警，可能涉及外部命名空间或 sys.path，不能直接当成 43 个已确认运行错误。
