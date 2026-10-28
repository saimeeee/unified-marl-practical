# SADMA 版本选择与行为差异

日期：2026-10-21；适用版本：unified-marl 0.5.0，第 5 阶段。

## 比较方法与结论

比较工作区王思哲 code/sadma，以及易才润论文 SADMA 的开源/闭源目录和 ZMQ 交付目录。原始交付仅作为参考，不执行 entry.py 或附件脚本。依据 Python AST、配置值、实际引用及来源模块运行对照选择，未按“开源/闭源”名称判断质量或许可。

主网络、Controller、Policy、Buffer 和目标计算采用王思哲版。该版与易才润版 Qpolicy/PPOPolicy 的 __init__/train/_update_targets AST 一致，但额外提供保存加载；补货 Runner 部分字段拼写和统计也有修正。采用统一运行器后仍保留王版实际算法代码，增加完整保存状态和生命周期管理，而不执行原入口。

易才润开源与 ZMQ 交付中比对的 48 份 Python/YAML/shell 文件逐字节相同，作为重复来源记录。闭源与开源 Python AST 一致，但 PPO/QMIX/CityFlow 配置和启动脚本有差异；不能视为同一次实验配置。王版与开源版部分原始字节差异只是换行/格式，不能把差异文件数量当作算法数量。

实际注册 Policy 仅 ppo/qmix；四份 MADDPG policy 文件均为空，maddpg_net.py 的网络和注册字典不足以构成训练算法。该项保留阻塞，不编造 actor/critic 损失。

## 必须保留的配置差异

| 能力 | 王思哲 | 易才润开源/ZMQ | 易才润闭源 |
|---|---|---|---|
| 隐藏维度 | 128 | 64 | 64 |
| QMIX 配置 Mixer | q_mixer | vdn_mixer | q_mixer |
| Q train_batch_size / buffer | 8 / 5000 | 64 / 2000 | 8 / 5000 |
| Q update_rate / target_update_interval | 1 / 200 | 8 / 100 | 1 / 200 |
| PPO train_batch_size / buffer | 8 / 8 | 32 / 32 | 8 / 8 |
| PPO mini_epochs / entropy | 4 / 0 | 4 / 0.01 | 8 / 0.01 |
| PPO agent_inputs | obs、last_action | obs、agent_id、last_action | obs、agent_id、last_action |
| CityFlow thread_num | 2 | 1 | 10 |
| PowerGrid 注册与配置 | 无 | 有，连续动作 | 有，连续动作 |
| 原完整 Policy 保存加载方法 | 有；仍漏目标滞后/ValueNorm 等状态 | 无 | 无 |

配置中的 `agenet_type` 拼写原样保留为来源字段；实际 Controller 消费 `agent_type`（default 中为 rnn），不会把拼写误认成有效覆盖。需要切换时显式设置 agent_type=mlp。运行默认只规定本地同步、设备和共享接口，不把这些数学配置统一成同一套。

统一命名：sadma/qmix、sadma/ppo 为王版配置；sadma/vdn 为显式王版 VDN Mixer 变体；sadma_yi/qmix、sadma_yi/ppo 为易开源；sadma_yi_closed/qmix、sadma_yi_closed/ppo 为易闭源。易开源的 qmix 名称沿用来源，但实际 VDN 明确记录。7 个调用配置不是 7 种独立算法。

## 运行器、Buffer 与保留能力

原 sync_run 对 Q 和 PPO 都从 ReplayBuffer 抽样，这可能把历史策略轨迹重复用于 PPO。本次保留 PPO 原 clipped surrogate、旧 log-prob、GAE/ValueNorm、mini_epochs、vtrace 和 dual_clip 分支，数据编排改为：在策略不变期间采集新回合，更新后全部消费并清空；拒绝旧 policy_version。mini_epochs 只在一次更新内部复用同一批。Q 则使用原 ReplayBuffer 的环形插入、无放回 episode 抽样和 filled 长度裁切。

原 EpisodeBuffer 在下一次观测到达 ts 时将 rewards/terminated 写到 ts-1。统一 Runner 已直接在动作 a_t 执行后写 reward_t/terminated_t，因此转换字段时不再减一；obs/state/available/actions/输出 hidden/log-prob 对应当前 t。等价到达数据已与来源 EpisodeBuffer 逐字段对照。原 individual_rewards 没有随团队 reward 一起移到 ts-1，是来源时间错位；统一 Runner 将实际个体奖励同样写在转移 t，没有复制团队值。

原 SMAC/补货 Runner 对 timeout 的结束/重置与 terminated 混用，补货还有 available_actions/avail_actions、individual_rewardss 等不一致。统一 Runner 分离 terminated/truncated、总在回合结束关闭当前轨迹、下回合 reset，并保存同一回合的最终观测用于 bootstrap。真实时间步仅统计 step，不把 reset 算作环境交互。

SMAC 使用共享延迟工厂（来源包装仅额外定义 scheme）；补货使用现有离散适配器，保留 34 档动作、观测补齐、团队奖励与独立个体奖励、1e4 缩放等来源约定。SADMA 显式设置 horizon=None 从后端配置取 horizon，time_limit 沿用来源配置；此前 HMF 的默认 horizon=21 不变。源包装忽略的 key/map_name/reward 保存在 sadma_env_metadata，不伪装为已生效参数。王版补货利润的额外查询/渲染统计没有接通，不能声称完整 Runner 等价。CityFlow 增加按需 backend 桥接与 close 转发，只完成模拟后端契约检查；真实引擎与交通资产未验证。PowerGrid 连续动作明确阻塞，扩展方案写入 INTEGRATION_SPEC。

## 有记录的兼容修复与来源限制

没有改变已接通分支的网络结构或损失公式。下列可运行性修复单独记录，来源文件本身不修改：

- 包内导入；无旧入口解析、日志全局初始化、进程启动方式修改或自动外部后端导入。
- Q-lambda 的 gather 索引补回 actions 最后一维，与源目标函数的形状契约一致。
- PPO use_value_norm=false 时显式使用原始 old_values，避免未赋值变量；central critic 的 state/states 输入拼写统一。
- PiMLP mask 改为与三维 logits 同形，Controller 跳过 MLP 返回的 None hidden state，数学计算不变。
- 捕获实际 backward 标量并检查有限性；完整 checkpoint 补齐目标网络、优化器、ValueNorm、原递归学习率状态、策略版本及 Q 回放。

13 组源码对照包含 7 个配置及 MLP、独立/中心 critic、显式 state、vtrace/dual_clip、Q-lambda 等分支；原始模块在内存中仅应用上述必要形状/拼写兼容修复，再比较同输入输出、损失与更新后网络。未修复即可运行的默认分支也使用相同对照路径；不能把所有分支说成原脚本原封不动通过。

保留的来源行为包括：GAE 使用 filled 类 mask 而没有显式在终止位置乘 (1-terminated)，因此终止转移仍可能含最终 critic 值；熵项使用 entropy_loss/entropy_loss.item()，仅一个可用动作导致零熵时可能报非有限损失；学习率按已衰减 lr 递归衰减；优势标准化包含填充位置；部分已声明的 MLP LayerNorm 未用于前向。这些没有借迁移擅自改成另一种 PPO。来源一致不证明理论正确、收敛或真实效果。

证据：evidence/stage5/audit.json（完整 diff、配置与来源哈希）、migration_sources.json、runtime_adaptations.json、13 份 parity JSON、time_semantics.json、tests.xml 和 installed_validation.json。
