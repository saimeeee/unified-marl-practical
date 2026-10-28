# 第 5 阶段：SADMA 本地同步训练

日期：2026-10-21（UTC+08:00）；版本 0.5.0。当前阶段已完成可验证的本地同步接入与验收，明确保留未通过项并停止，没有进入第 6 阶段分布式接入。

## 交付结果

采用王思哲 SADMA 的实际 Policy、Controller、Agent、Mixer、Critic、目标函数与 Buffer 代码，接入统一 Python API/CLI、EpisodeBatch 和 checkpoint。保留易才润开源/闭源配置，记录开源/ZMQ 重复关系。完整结果为 2 类策略（Q/PPO）、显式 VDN Mixer 变体、7 个调用配置和 6 组额外有效分支，不是 13 种独立算法。

完整 149 项测试通过（此前 112 项全部保留）；13 组来源对照在已测输入下网络输出、损失与更新后网络最大误差均为 0。每个迁移配置均有有效参数更新、有限指标、动作合法性、隐藏状态重置、保存加载和评估证据。源码目录外安装验证包括 7 个配置各 2 次更新、推理/评估和恢复至第 3 次更新，以及 QMIX/PPO 两种 CLI。

同一 Python 进程顺序启动两次短任务并正常关闭环境；训练异常和环境元信息构造异常也会关闭资源并恢复线程设置。独立安装进一步顺序执行全部 7 个配置，检查采集/评估环境关闭、没有新增遗留子进程。未以旧 entry.py 或注册名称代替算法迁移。

本阶段真实环境、GPU、收敛性与策略效果均未验证。toy 和模拟后端结果只证明接口、数据与短更新；真实 SMAC、补货和 CityFlow 后端不可用，连续 PowerGrid 在 schema 阶段明确阻塞。没有长时间实验、远程消息或全局 Python 修改。

## 版本选择与调用名

完整依据、逐版本配置表和兼容修复见 [SADMA_VERSION_SELECTION.md](SADMA_VERSION_SELECTION.md)，机器比较见 evidence/stage5/audit.json。

| 调用配置 | 来源与保留行为 |
|---|---|
| sadma/qmix | 王思哲；q_mixer、原 Q/TD-lambda、目标更新与 epsilon 调度 |
| sadma/ppo | 王思哲；熵系数 0、原 PPO 目标、中心 critic、ValueNorm、mini_epochs=4 |
| sadma/vdn | 显式王版 Mixer 变体，仅配置 Mixer 改为 vdn_mixer |
| sadma_yi/qmix | 易才润开源；名字为 qmix，实际 vdn_mixer，原批量/更新间隔保留 |
| sadma_yi/ppo | 易才润开源；原 agent_id 输入、熵系数 0.01、batch=32 |
| sadma_yi_closed/qmix | 易才润闭源；q_mixer、batch=8、hidden=64 |
| sadma_yi_closed/ppo | 易才润闭源；mini_epochs=8、熵 0.01、batch=8 |

四份交付中的 MADDPG Policy 文件均为空且未进 policy_REGISTRY；独立算法缺失，不能用 maddpg_net.py 中的网络凑出算法。易开源与 ZMQ 对比的 48 份 Python/YAML/shell 逐字节相同；闭源 Python AST 与开源一致，配置与脚本有差异。主算法采用王版是因为数学实现相同且提供额外保存加载，运行器部分补货拼写有修正；并未按目录名或 README 判断。

## 统一的数据与执行流程

`algorithms/sadma/adapter.py` 桥接 Controller 与 Policy；`trainers/sadma.py` 管理同步执行；`native/` 保留来源类与 Buffer。实际执行不依赖原目录导入，也不触发外部环境、TensorBoard、ZMQ 或旧入口参数解析。

| 内容 | 约定及已验证行为 |
|---|---|
| 字段 | state/reward/avail_actions ↔ states/rewards/available_actions；actions、log_probs 末尾单维显式转换；批数据 clone 后交原 Learner |
| 时间 | obs/state/avail/action 位于 t；action_t 的 reward/terminated/truncated 写 t；同回合最终观测位于 T；旧到达式 ts-1 不重复应用 |
| 隐藏状态 | 每回合从零开始，hidden[t] 存 t 次前向的输出，t+1 消费；推理 policy.reset() 明确重置 |
| PPO 旧策略信息 | 保存 action_log_probs [B,T+1,N,1]、behavior_values [B,T+1,N,1]、policy_version [B,T+1,1]；与原网络重算输出误差为 0 |
| Q 生命周期 | 真实来源 ReplayBuffer 环形插入、无放回回合抽样、filled 长度裁切；跨更新保留记录数据与目标网络 |
| PPO 生命周期 | 同策略不变时收集新回合，满足 train_batch_size/update_rate 后一次消费全部；mini_epochs 内部复用，随后清空；不同旧策略版本明确拒绝 |
| 奖励 | 团队标量与实际个体奖励分开；PPO 对团队奖励重复到智能体目标是来源公式，不是把团队数据伪造为个体真值 |
| 计数与调度 | env_steps 为实际 step；不含 reset；t_max/max_updates 控制运行预算，max_steps 保留源学习率 horizon；回合结束后判断预算，可能完整收完末回合 |
| 保存 | policy.pt 用于推理，training.pt 额外保存目标网络/critic/mixer、优化器、ValueNorm、递归 lr、策略版本、更新计数、Q 回放与 RNG |

PPO 仍按来源在更新前重算 old_values；因为同步期间 critic 未改变，其值与采样保留值一致。来源默认 PPO 也使用历史 ReplayBuffer，当前编排为满足同策略生命周期而改为新数据一次消费，原目标公式和 mini_epochs 没有改成另一种 PPO。

回合边界与来源 EpisodeBuffer 的奖励/终止到达式写入逐字段一致；个体奖励原 Buffer 缺少 ts-1 偏移，本次按实际转移 t 修正。天然终止与 timeout 分开，不把截断当作真实终止，不使用 reset 后的下一回合观测作当前 bootstrap。源 GAE 对终止 bootstrap 的已知行为单独保留，不因数据位置正确而声称目标公式已修正。

完整状态恢复在 7 个配置的“已完成更新、无待消费回合”的 toy 边界与连续训练逐项一致。预算结束时未消费的 PPO 回合仅记录数量并丢弃，外部仿真器内部状态不保存；不能承诺任意中途恢复逐步一致。旧 agent.pth/mixer.pth 文件夹导入尚未验证，不冒充旧文件自动兼容。

## 环境适配与连续动作

SMAC 共享延迟工厂；smac 训练别名映射到 sc2，环境及地图依然需要外部安装。补货共享适配器保留来源 34 档动作、padding、1e4 缩放和真实个体奖励，通过显式 horizon=None 保留 SADMA 从后端配置读取 horizon 的行为。HMF 默认 horizon=21 不变。源配置中未消费的 key/map_name/reward 保存为 metadata，不传给不接受它们的工厂，不将 reward2 字符串假称为生效奖励模式。该映射问题在初次独立检查中被发现，失败日志保留，修复后重新验证。

CityFlow 使用新的延迟桥接工厂，把调用参数传给外部 cityflowenv.env.CityflowEnv，并在后端有 close 时转发。真实后端没有 close 时的释放方式尚未验证；原包装 close 是 pass，不能以原包装空函数当作关闭成功。模拟后端下完成 Q 更新及 close 契约检查，补货模拟后端完成 PPO 个体奖励更新。

易才润 PowerGrid 是连续动作，源 action_shape=(4,)，PPO 含 Normal 分支。当前调用会明确失败；没有把 Box 强转为离散索引，也未将 Gaussian 分支在代码中存在视为完整支持。INTEGRATION_SPEC 已记录所需 float actions/last_actions、action_spec/bounds、按动作维 log-prob 求和、去 OneHot、浮点推理、clip/scale 语义和 checkpoint 扩展。检查的交付缺默认 powergrid/data，当前验收环境也缺 pandapower；没有尝试用假资产评估。

## 兼容修复与保留的来源问题

可运行性修复：包内导入；MLP None hidden-state 处理；PiMLP 的 mask/logits 同形；Q-lambda gather 补索引维；state/states 输入一致；use_value_norm=false 时使用原始 old_values；在 backward 前检查有限性并捕获实际损失；异常关闭与 checkpoint 补齐。源码对照中仅对原始模块在内存中应用同样的形状/拼写兼容修复，不编辑原文件，不替换损失或随机用另一算法。

保留并记录的来源行为：

- GAE 使用 filled 类 mask，终止位置未显式乘 (1-terminated)，可能含最终 critic bootstrap 值；专门测试保留该行为。
- 熵项为 entropy_loss/entropy_loss.item()；单一合法动作导致零熵时可能非有限并明确失败，没有擅加 epsilon 修改目标。
- 学习率递归基于上一次已衰减 lr，优势标准化包含 padding，部分 MLP 声明的 LayerNorm 未用于前向。
- 配置拼写 agenet_type 不会覆盖实际 agent_type。补货原 profit 查询、额外渲染统计和原异步 Runner 没有接入。
- 本阶段 SADMA 一个采集器一个环境；env_batch_size/batch_size_run 必须 1，async_train=true 和 distributed execution 被拒绝。原 PyMARL parallel Runner 不受影响。

这些既是长期训练的限制，也是后续算法修订需要另行决策的事项。来源数值一致不等于来源代码正确或论文效果复现。

## 验证层级与剩余项

| 层级/能力 | 实际结果 |
|---|---|
| 代码实现与统一调用 | 7 个配置、13 组配置/分支短更新；7 配置保存加载及恢复 |
| 来源输出/损失/更新对照 | 13 份 PASS，已测 CPU/固定输入下误差为 0；参考的兼容修复逐项记录 |
| 时间、padding 与生命周期 | arrival Buffer 等价、旧概率/价值/hidden、终止/timeout、Q ring replay、PPO 拒绝旧版本均通过 |
| 资源释放 | 两次同进程任务通过；训练/元信息异常关闭通过；独立安装序列通过 |
| 前期代表算法 | PyMARL QMIX、RODE-HMF、HMF MFQ 的 toy 更新，以及 BC/CPL 的真实 3m 数据更新通过 |
| 真实 SMAC/补货/CityFlow | 实际入口尝试被依赖/资产阻塞，未训练评估成功 |
| 连续 PowerGrid | 明确阻塞，扩展方案已记录；schema/完整接口与资产验证未实现 |
| MADDPG | 原 Policy 为空，未迁移，不补造损失 |
| 分布式、跨机、向量化 SADMA 采集 | 本阶段未执行；ZMQ 第 6 阶段仍待用户授权 |
| GPU、长训/收敛、胜率或效果 | 未验证 |

## 实际运行命令与产物

在项目根目录使用 .venv/.venv-check 的 Python 3.12.14、torch 2.6.0+cpu、NumPy 1.26.4，运行时 TEMP/TMP 指向 .work/tmp，PYTHONIOENCODING=utf-8、PYTHONDONTWRITEBYTECODE=1。仅安装本地 wheel；本阶段没有新增全局依赖或执行下载/部署脚本。

| 已执行命令 | 结果与证据 |
|---|---|
| `.venv/Scripts/python.exe -B tools/audit_stage5.py --migrate` | exit 0；4 版本源码/配置对比，13 份原文件命名空间导入；audit.json、migration_sources.json；该迁移模式一次性，不能重复覆盖 |
| `.venv/Scripts/python.exe -B tools/prepare_stage5_runtime.py` | exit 0；一次性包内兼容处理和来源配置保存，运行适配记录后续补齐在 runtime_adaptations.json |
| `.venv/Scripts/python.exe -B tools/probe_stage5.py` | exit 0；13 配置/分支各 2 次更新及评估通过，probes.json/log；初始 MLP None 错误保留 probes_initial.json |
| `.venv/Scripts/python.exe -B -m pytest tests -q --junitxml=docs/evidence/stage5/tests.xml` | exit 0；149 passed，14 条已有/来源警告；pytest.log、13 个 parity JSON |
| `.venv/Scripts/python.exe -B -m build --no-isolation` | exit 0；0.5.0 wheel/sdist，build.log |
| `.venv-check/Scripts/python.exe -B -m pip install --no-deps --no-index --force-reinstall dist/unified_marl-0.5.0-py3-none-any.whl` | exit 0；仅项目验收环境，本地 wheel 安装，install.log |
| 在工作区根运行 `unified-marl/.venv-check/Scripts/python.exe -B unified-marl/tools/validate_stage5_installed.py` | 源码目录外 site-packages：7 配置、两 CLI、5 个代表回归、真实环境阻塞及生命周期，installed_validation.json/log；初次适配失败保留 installed_validation_initial.log |
| 两个环境各运行 `python.exe -B -m pip check` | 依赖一致性记录在 dependencies.json |
| `.venv/Scripts/python.exe -B tools/record_stage5.py` | 保留第 4 阶段矩阵快照，更新 369 行来源/集成清单与 summary.json |
| `.venv/Scripts/python.exe -B tools/verify_stage5.py` | 最终来源/文档/证据/发行文件一致性，final_verification.json |

原始 1228 条源码/资源哈希复核不变；第 5 阶段额外比较的 192 条 SADMA Python/YAML/shell 来源哈希也不变（与原清单有重叠，不相加冒充唯一文件数）。运行源码、wheel 与独立安装中的 Python/YAML 文件逐字节核对。来源目录未留下本次 Python 3.12 bytecode。所有具体阻塞保留在矩阵，不把缺失资产或设计文档当作迁移完成。

已更新 WORK_ORDER、规格、计划、状态与迁移清单。本阶段结束后停止，等待用户指定第 6 阶段。
