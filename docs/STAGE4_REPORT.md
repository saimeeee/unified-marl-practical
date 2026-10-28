# 第 4 阶段：寇谦离线、偏好学习与奖励模型

日期：2026-10-20；发行版本 0.4.0。本阶段完成可验证的本地接入及真实交付数据短更新，保留未迁移项并停止。没有进入 SADMA、第 5 阶段或长时间收敛实验。

## 结论与支持范围

第一条普通离线链路是 `kou/bc`，第一条偏好链路是 `kou/cpl`：真实 3m 轨迹读取 → 原字段/标签显式适配 → 原 Controller/Learner → 有效更新 → 推理及完整状态保存 → 加载/继续训练。训练预算由更新次数推进，不依赖环境采样。

共 17 个策略调用配置完成真实数据短更新、推理加载和继续训练；这是配置/变体数量，不是 17 种独立算法。6 种奖励模型完成真实数据更新、预测和保存加载，另验证 MultiPrefTransformerDivide 多标签模式的初始轮次。112 项完整测试通过（包含此前 59 项）；17 组策略与 7 组奖励模型来源对照通过：更新后网络/预测误差为 0，损失最大绝对误差约 5.96e-8（两种旧奖励模型日志归约的浮点差异）。12 份真实轨迹文件完成小批读取，参数更新使用交付的 3m 数据，不能据此声称其他地图训练成功。

真实 SMAC/SC2 评估被环境依赖阻塞；没有验证策略胜率、论文效果、收敛性、GPU 或多种 Python/PyTorch 版本。来源数值一致仅表示迁移没有改变已测行为，不证明来源算法或代码正确。

## 源码判断与配置清单

参考根目录为工作区 `marl平台代码/寇谦/论文代码及技术文档/论文代码及技术文档`。检查了 ICQ-MA/src 的实际 Controller/Learner 注册与 run.py 分发、嵌入 MAPT 的奖励引用闭包、old_src 的 main/run/globalRM/pref/数据入口及数据制作函数。old_src 的奖励/偏好分支仍有调用关系，不能按名称认定废弃。ICQ-MA 的 QLearner 是含行为约束的实现，不是原 PyMARL QMIX，不能同名覆盖。

| 来源/调用名 | 实际接入内容与验证范围 |
|---|---|
| `kou/bc` | 交付 bc_smac 配置，监督行为克隆；普通离线首链路 |
| `kou/cpl` | CPL 配对偏好损失，沿用标签方向及策略概率；偏好首链路 |
| `kou/icq_ma` | offpg_smac 的 OffPG/ICQ 风格 critic 再 actor 更新，保留 QMixer/OffPGCritic |
| `kou/iql`、`kou/cql` | 各自来源 IQL 与 CQL 的已接通默认训练分支；未用原 PyMARL 同名项替代 |
| `kou/dppo` | 来源 DPPO 配对训练 |
| `kou/madpo`、`kou/madpo_aux`、`kou/bcmapt` | 来源个体奖励/偏好变体；显式要求 agent_reward，保留各自目标 |
| `kou_old/qmix`、`kou_old/qplex` | 历史 QLearner/QPLEX 在记录轨迹上的原 TD 更新 |
| `kou_old/cw_qmix`、`kou_old/ow_qmix` | CW/OW 的 central controller、mixer、权重和目标网络 |
| `kou_old/qmix_w_globalRM`、`kou_old/qplex_w_globalRM` | 先训练 old_global，再把团队预测奖励写入 reward_hat 进行 TD 更新 |
| `kou_old/cw_qmix_w_globalRM`、`kou_old/ow_qmix_w_globalRM` | 仅基础 train() TD 方法通过；该来源方法仍用原团队 reward，完整奖励交互/无监督编排未迁移 |

原配置名也可调用，如 `kou/bc_smac`；不存在 `kou/icq/bc_smac` 这种额外路径。不同来源单独命名。上表 old_src 的记录数据更新不意味着原实验原本就是离线算法，也不意味着完整在线偏好实验已经迁移。

## 数据、配对、奖励与时间位置

`datasets/episodes.py` 使用 tensor-only、CPU、mmap 读取 .th；不启用任意 pickle 回退。action/actions、avail_action/avail_actions/available_actions、rewards/reward 映射后会检查冲突、维度、dtype、合法动作、有效槽和标签数量。批数据复制后再交给可能原地修改 mask 的 Learner，避免改动数据源。数据路径由调用参数提供，不执行作者配置中的历史绝对路径。

| 字段 | 实际布局/规则 |
|---|---|
| obs / state | [E,T,N,O] / [E,T,S]，按原记录保留状态槽 |
| actions / avail_actions | [E,T,N,1] / [E,T,N,A]，离散整数及可用动作 |
| reward / agent_reward | 团队 [E,T,1]；个体 [E,T,N,1]，不自动求和或复制 |
| filled / terminated | [E,T,1]；filled 连续前缀，真实终止后槽无效；TD 使用 [:,:-1] 转移掩码 |
| PreferenceBatch | trajectory0、trajectory1 两个 DataBatch；各自 mask/lengths、labels、pair_indices，以及独立 global/between/in 标签 |
| 标签方向 | 二元 labels 为 1 偏好第二段、0 偏好第一段、0.5 为平局；团队回报造标签的纯函数保留来源严格 `right > left`，相等时为 0 |
| 配对规则 | 默认前半 E/2 对后半 E/2；也可显式 pair_indices；有标签时检查偶数及标签数，不因普通数据无偏好标签而强制偶数 |

偏好 mask 包含真实终止转移，排除终止后的 bootstrap/padding；截断/未终止数据保留原 CPL 使用 filled 槽的约定，不能笼统承诺所有最终槽都会被丢弃。独立 TD transition_mask 截去末格。数据语义测试覆盖可变长度、配对、padding、终止位置、标签方向和排序，不把轨迹内名次改成二元标签。

真实 `MADiversePreferDatasets/3m/3m.th` 有 400 段轨迹、200 对：obs [400,61,3,30]、state [400,61,48]、actions [400,61,3,1]、avail_actions [400,61,3,9]、团队 reward [400,61,1]。labels/global_labels 均 [200,1]，between_labels [200,3]，in_labels [2,200,3] 是名次排列；这些字段并不等价。agent_ids 原 dtype 保留。

同目录 `reward.th` 的个体预测奖励为 [400,60,3,1]，通过 dataset_options.agent_reward_path 显式加载；不把它声称为真实环境个体奖励真值，不补造第 61 格。本次需要个体奖励的短更新显式使用长度 12。另有标签侧文件，仅显式 labels_path 时读取。18 份 .th 归档均能进行 tensor-only 读取，其中 12 份轨迹文件完成实际小批读取；其余是标签/奖励侧文件。

## 训练与公共接口

统一 `train(..., dataset=...)` 分发到离线 Trainer。配置优先级为公共默认 < 来源默认 < 来源算法 YAML < Kou 运行默认 < 用户 config < overrides < 显式调用参数。来源网络/损失参数保留，运行默认规定数据与更新预算。training_updates 为更新总预算；schedule_updates 是独立调度 horizon，继续训练不因改预算偷偷改调度。来源 `t_env/episode_num` 接收有文档约定的更新索引；TrainResult 中真实 env_steps=0、episodes=0，未伪造在线采样量。

ICQ 来源 off_batch_size 表示两段轨迹的总数，偏好对数为其一半；old_src 使用 batch_size。KouLearner 适配原 train 的参数顺序和返回方式，捕获实际反向传播标量及原日志；IQL 的多个网络、OffPG 的 critic/actor 顺序、CW/OW 的 central 网络与优化器均保留。

输出包含 config.json、metrics.jsonl、result.json、policy.pt、training.pt；推理状态与完整续训状态分离，保存目标网络、优化器/调度器、计数器、来源额外状态和随机状态。数据本体不复制进 checkpoint；继续训练需数据仍可访问。循环策略通过 policy.reset() 清空回合隐藏状态。BC/CPL/ICQ/IQL/CQL/CW 的合成连续训练与恢复已做参数逐项对照；独立安装对 17 个配置均完成第 2 次更新保存、加载推理及恢复到第 3 次更新。

eval_interval_updates=0 默认关闭环境评估。显式指定 env/env_args 后才能按更新频率评估；周期逻辑已在 toy 验证，真实 SMAC 评估仍阻塞。数据训练本身不创建环境。train CLI 支持 --dataset；奖励训练 CLI 为 train-reward。Python 和 CLI 示例见 README。

## 奖励模型与保存加载

| 名称 | 来源/损失与预测语义 |
|---|---|
| MR | 来源 MAPT MR；配对偏好目标；source_offline 按来源在智能体维取均值得到团队 [B,T,1] |
| NMR | 来源 LSTM 奖励；source_offline 团队 [B,T,1]；保留下述来源问题 |
| PrefTransformer | 来源 GPT2 类奖励；source_offline 团队 [B,T,1]；没有下载预训练权重 |
| MultiPrefTransformerDivide | 来源多智能体 Transformer；source_offline 个体 [B,T,N,1] |
| old_global | old_src RewardModel 的全局奖励分支、原 ensemble/交叉熵 |
| old_reward_global | old_src RewardModelReward 的全局奖励分支、原 KL 目标 |
| MultiPrefTransformerDivide + mode=diverse | 原 global/between/in 三种损失、KL 信任筛选和阈值课程；仅初始真实轮次有效更新通过 |

`train_reward_model(...)` 返回结构化结果及 reward.pt、reward-training.pt；`load_reward_model(...)` 恢复预测或完整原优化器状态。默认 predict(mode="source_offline") 保留原离线后处理；mode="native" 显式获取底层个体输出。reward_model_checkpoint 接入旧全局 TD Learner 前必须满足团队语义与环境维度；不自动把个体预测变成团队奖励。

来源限制保持可见：

- NMR 在第二段损失复用了第一段预测；相同 mask 可令梯度抵消。初次短窗口真实探测记录了无参数变化，完整 61 槽真实轨迹才验证到非零更新。另有合成测试断言抵消仍存在，未为了“通过”改写公式。
- MR/NMR 的报告总损失含正则项，但反向传播使用 trans_loss/rf_loss；沿用来源，来源损失和更新后参数对照通过。
- PrefTransformer 的 GPT2 blocks 放在普通 Python list，未注册进原优化器，也不随 eval 自动关闭 dropout。保留训练行为，额外把遗漏层权重和 train/eval 状态保存到 checkpoint；相同随机状态下预测加载完全一致。不能承诺不设随机状态的确定性推理。
- diverse 模式保留原跨轨迹索引、阈值和 mini-batch 内重排。合成测试中初始轮次通过，第二轮筛选得到 0 对时明确报错；真实数据只测初始短更新，未声称课程后续阶段有效或训练稳定。
- old_global/old_reward_global 仅全局分支迁移，个体/局部奖励模型与参考策略生成标签未完成。旧新式 reward_model_new 无真实参考策略，明确阻塞。

## 数据辅助工具与外部边界

get_preferences.py 的纯特征计算已迁移到 datasets/prompts.py，使用显式模板目录，3m 真实数据和源特征输出逐项一致；其他地图特征尚未验证。make_diverse_preference.py 仅纯 clip_to_segment_length 迁移到 datasets/segments.py，固定随机状态下与来源完全一致。datasets/labels.py 保留原团队回报严格比较规则，拒绝个体奖励隐式聚合。

未执行 use_llm.py、远程打标签、API 重试或原目录标签重写。外部 LLM 属服务边界，不是算法迁移成功的前提。没有使用用户密钥或发送数据。附带文档中的命令均作为参考，未自动执行。

## 未迁移及未通过项

| 项目 | 具体原因 / 本次处理 |
|---|---|
| ICQ-MA qmix/qmix_smac/vdn_smac（诊断别名 bcq_qmix/bcq_vdn） | QLearner 期待 Q/imitator/aux 三输出，实际注册 BasicMAC 返回单张量；交付未提供匹配模仿 Agent/Controller，未替换成 PyMARL QMIX |
| ACPL | 所配 CPL Learner 缺 NLL_lamda/use_nll 等参数，Q 输出不符合概率契约，原分发也没有 ACPL 分支；不擅自补数学配置 |
| hacpl_smac/maicpl_smac/maidpo_smac | 引用的 Learner 不存在或未注册，明确失败 |
| old qmix_differ | 需要真实 indi_terminated，当前数据只有团队 terminated；未复制团队终止 |
| old qmix_new/pref_qmix/pref_qmix_new | 需要真实参考策略 checkpoint，以及个体终止、agent_state 或局部奖励/标签协议，当前资产不全 |
| old reward_model_new 及历史局部奖励 | 构造时强制加载参考策略；缺匹配资产和 agent-state 标注协议 |
| CW/OW globalRM 完整运行流程 | 基础 TD 更新通过，在线奖励收集、无监督、重置和 reward 训练调度未接入 |
| e2e 联合训练、moderate_lambda 行为预训练、PER、use_reward_hat 额外聚合路径 | 依赖不同的数据/训练协议；显式拒绝启用，避免默默退化成已测分支 |
| GRNN、RNN_NS、MLP/NS、maddpg Controller/critic 等备用组件 | 没有通过已交付离散离线配置验证；GRNN 还需要来源 SMAC unit 元信息和通信设置 |
| 旧 online/pref/采集 runner 与手工 .npy 数据管线 | 生命周期和额外字段不同；未用启动旧 main 充当迁移 |
| MAPT 嵌入的其他策略、MuJoCo/swimmer、未引用工具 | 不在实际奖励依赖闭包内，未作为本阶段支持项 |
| 后续信任课程、其他地图参数更新、策略效果、GPU | 尚未验证，不因初始短更新改为通过 |
| SMAC/SC2 实际评估 | 缺可用后端及游戏/地图，实际调用失败已记录；未回退成 toy |

## 已执行命令与证据

以下均以项目根目录为工作目录，安装验证特例另列；执行时将 TEMP/TMP 指向项目 .work/tmp，设置 PYTHONIOENCODING=utf-8、PYTHONDONTWRITEBYTECODE=1，源对照用 -B 禁写 bytecode。仅使用 .venv 和 .venv-check，没有修改全局 Python。解释器 Python 3.12.14、PyTorch 2.6.0+cpu、NumPy 1.26.4；奖励 extra 固定 ml-collections 1.1.0、transformers 4.53.0。完整依赖见 dependencies.json。

| 实际命令 | 结果/证据 |
|---|---|
| `.venv/Scripts/python.exe -B tools/audit_stage4.py` | 源码注册/入口、配置和全部 18 个 tensor 归档只读盘点，audit.json |
| `.venv/Scripts/python.exe -B tools/probe_stage4_core.py` | 原分支真实数据探测及具体失败，core_probes.json；不把失败项计为完成 |
| `.venv/Scripts/python.exe -B tools/probe_stage4_rewards.py` | 6 模型及 diverse 初始更新、2 个团队奖励 TD 路径；reward_probes.json，初次失败留 reward_probes_initial.json |
| `.venv/Scripts/python.exe -B tools/check_stage4_datasets.py` | 12/12 真实轨迹文件小批读取通过，dataset_batches.json |
| `.venv/Scripts/python.exe -B -m pytest tests -q --junitxml=docs/evidence/stage4/tests.xml` | exit 0，112 passed；14 条已有/来源兼容警告，pytest.log、24 个 parity JSON |
| `.venv/Scripts/python.exe -B -m build --no-isolation` | exit 0，生成 0.4.0 wheel 和 sdist，build.log |
| `.venv-check/Scripts/python.exe -B -m pip install 'dist/unified_marl-0.4.0-py3-none-any.whl[reward]' --cache-dir .work/pip-cache` | 项目独立安装环境升级，install.log |
| 工作区根目录运行 `unified-marl/.venv-check/Scripts/python.exe -B unified-marl/tools/validate_stage4_installed.py` | 已安装 site-packages 包：17 配置真实数据更新及恢复、6 奖励模型加 1 多标签模式、QMIX toy 回归及两种 CLI；installed_validation.json/log；真实 SMAC 失败单列 |
| 两个环境各执行 `python.exe -B -m pip check` | 依赖一致性，dependencies.json；未改动全局环境 |
| `.venv/Scripts/python.exe -B tools/record_stage4.py` | 保存阶段 3 清单快照，更新 361 行矩阵、summary.json，新增合成/真实数据/效果三列 |
| `.venv/Scripts/python.exe -B tools/verify_stage4.py` | 文档/证据/来源哈希/安装内容最终一致性，final_verification.json |

运行中间失败未删除；最终独立安装结果以 installed_validation.json 为准。测试里的源模块导入只用于对照，不是运行包的依赖。包运行不再从交付目录 import 旧入口。

源码/配置/资源的 1228 条初始哈希复核不变；运行源码、wheel 与独立安装文件逐字节复核。迁移清单为 361 行，新增验证列区分合成测试、真实数据、策略效果；前三阶段未记录的新列保留 NOT_RECORDED，不追溯伪造证据。详情见 evidence/stage4/summary.json 与 final_verification.json。

下一步仅等待用户指定第 5 阶段。缺失资产和来源实现缺陷继续保留，不随阶段结束自动视为已解决。
