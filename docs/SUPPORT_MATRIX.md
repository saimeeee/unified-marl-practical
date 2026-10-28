# 实用首版支持增补（0.9.0.dev1）

2026-10-28：库存InventoryEnv、MPE simple_spread、原生CityFlow分别与pymarl/qmix、pymarl/vdn、sadma/ppo、example/huber_iql完成短训练/加载/评估/恢复，12/12通过。前两者在Windows隔离Python环境，交通在本机Docker Linux原生引擎。支持数据为演示合成库存、生成路网、MPE仿真；不宣称收敛或业务收益。新增插件/环境契约、配置实验和矩阵，详见 PRACTICAL_V1_REPORT.md。

其余内置算法保持原支持范围；本轮未验证SMAC、原ReplenishmentEnv、足球、MuJoCo、机械手、GPU与跨机。以下旧矩阵保留其原版本和限制，旧未通过项不因首版完成而消失。

# 支持与未支持能力

版本 0.7.0；所有“通过”均限定于报告记录的输入、配置、设备与环境。配置名称不等于独立算法数。

| 能力 | 已接入并验收的范围 | 执行方式 | 验证层次及边界 |
|---|---|---|---|
| PyMARL | QMIX、VDN、IQL、COMA、QTRAN；前三者 beta 配置 | local episode / spawn parallel | CPU toy 有效更新、原输出/损失对照、保存恢复及评估；真实 SMAC 未通过 |
| RGHMF | wang/rode、rode_hmf；来源八个基线配置复用核心实现 | local | CPU toy、角色/均场时间位置、聚类前恢复；动态扩角色完整训练未通过 |
| HMF | dqn、vdn、qmix、mfq、mfvfd、hierarchical、hierarchical_vdn、hierarchical_vdn_IPC、hierarchical_vdn_without_high_level、maddpg、hierarchical_vdn_copy_fixed、hierarchical_vdn_meanq、hierarchical_vdn_k_means_fixed、hierarchical_vdac_fixed | local CPU 固定分组 | 14 个默认更新/变体通过对照、更新、保存恢复；不是全部高层/动态调度。动作需始终合法，原选择器没有通用掩码处理 |
| 普通离线 | kou/bc、icq_ma、iql、cql | local，按 training_updates 推进 | 合成数据、真实 3m 短更新；策略效果未验 |
| 偏好策略 | kou/cpl、dppo、madpo、madpo_aux、bcmapt | local，显式偏好对 | 后三项须真实独立 agent_reward；标签、mask、损失对照和来源数据短更新有证据；未验长期筛选/效果 |
| 历史离线 | kou_old/qmix、qplex、cw_qmix、ow_qmix；qmix_w_globalRM、qplex_w_globalRM | local | globalRM 使用显式团队 reward_hat/奖励 checkpoint。CW/OW 的同名 globalRM 两配置仅基础 TD 方法已验，完整奖励交互仍未接通 |
| 离线 Agent 变体 | kou/bc + rnn_ns；kou_old/qmix + mlp / mlp_ns | local | 原损失/参数对照、真实 3m 更新、保存恢复；不是新增算法或任意组合支持 |
| 奖励模型 | MR、NMR、PrefTransformer、MultiPrefTransformerDivide、old_global、old_reward_global | local | 真实 3m 更新/预测/加载；固定批次完整状态下一次更新对照；diverse 只验证初始课程，后续空信任池明确失败 |
| SADMA | sadma/qmix、ppo、vdn；sadma_yi/qmix、ppo；sadma_yi_closed/qmix、ppo | local 同步 | 7 个配置，保留 Q 回放/PPO 新轨迹、旧概率、原 ValueNorm/更新；易开源 qmix 实际 VDN |
| ZMQ Q/VDN | 上述 SADMA Q/VDN 配置 | sync / async | Windows 本机多进程、参数摘要/版本、样本校验、保存恢复、故障/超时/端口释放 |
| ZMQ PPO | 上述 SADMA PPO 配置 | sync；async 须 use_vtrace=true | 普通 PPO 不允许异步回放；不是其他家族的自动分布式支持 |
| 周期评估 | ZMQ 不可变 policy 快照、独立单个 spawn evaluator | 本机，按更新触发 | 结果/版本、忙时跳过、故障超时/清理通过；原向量永久 worker 与远程评估不在已验范围 |
| 环境 | toy；SMAC、ReplenishmentEnv、CityFlow 延迟适配器 | 按上述算法限制 | toy 实跑；三种真实后端/资产缺失，只有模拟契约及明确失败记录 |

不支持/未通过的组合：PyMARL/RODE/HMF/Kou 的 ZMQ；连续动作 PowerGrid；空 SADMA MADDPG Policy；HMF 四个失败实验与 VDAC 完整 actor/高层流程；RODE 扩角色后的优化器/恢复；缺失 Learner/参考策略/个体字段的 Kou 分支；PER/e2e/moderate CQL；原持续共享缓冲评估 ABI、动态推理合批、向量采样、多 Learner；真实仿真、跨机器、GPU、Linux/macOS、其他 Python/PyTorch 版本与收敛效果。

恢复训练范围：PyMARL、聚类前 RODE、上述固定 HMF、已验离线策略和 SADMA 本地，在 toy/固定数据的完成更新边界对照；ZMQ 保留 Learner/回放/计数/RNG，但重新启动的 worker RNG、异步调度及外部仿真器内部状态不保证逐步相同。奖励模型的高层 diverse 课程/采样池没有完整恢复声明。

逐项细节见 MIGRATION_MATRIX.csv；源码文件到新文件的相对路径和 SHA-256 见 MIGRATION_TRACEABILITY.csv。未通过项完整保留在 KNOWN_ISSUES.md 与 evidence/stage7/migration-audit.json。

## 0.8.0 新增实现（待用户测试）

以下仅记录新增实现，不并入上方0.7.0通过项：warehouse/inventory参数化补货及业务输出；traffic/cityflow_engine原生引擎封装和相位输出；powergrid/bounded_ppo连续动作策略、训练/评估/保存恢复与来源环境适配；统一应用CLI、参数模板及双平台工程包。本轮没有运行训练或仿真，因此不声明有效更新、策略效果、外部引擎或Linux运行通过。原动态HMF/RODE缺项仍保留。详见 ENGINEERING_REPORT.md。

## 0.8.1 报告修复支持层次

仓储MFVFD在Linux合成小样例的运行通过来自用户首轮报告；seed误判的修复和评估计数仅本机无环境单元测试通过。API/CLI恢复、旧0.8.0 checkpoint兼容由二测确认。交通/电网依赖和资产缺项保留；没有新增真实环境、真实数据或策略效果验收。

## 0.8.2修复范围

仓储/交通/电网二轮结果来自用户报告，未在此Windows电脑复现。F1数组公式、F2实际生产reset/step函数计数、N1备份及清单更新/重复执行/损坏拒绝有单元证据。无Linux新补丁训练、潮流或效果证据。旧电网回合语义兼容边界见ENGINEERING_HOTFIX_082。

## 0.8.2第三轮报告接纳

用户报告仓储、CityFlow单路口PPO、case33bw配公开CityLearn指定建筑/三类储能动作的最小链路通过。F1/F2、电网完整恢复和N1已齐依赖重复加装通过；细项见VALIDATION_REPORT_082。保持USER_REPORTED_SCOPED，与本机单元/Windows安装检查分开。干净Linux依赖、reset跨进程复现、交通安全过渡及其他动作建筑未验。
