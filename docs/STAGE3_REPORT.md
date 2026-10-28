# 第 3 阶段迁移与验证报告

日期：2026-10-20。第 2 阶段结果已先记录并向用户输出，按补充授权自动接续本阶段。当前在第 3 阶段结束后停止，第 4 阶段未执行。

本阶段完成可独立验证的接入工作，保留真实环境和来源实验代码阻塞项。不能将本报告理解为王思哲交付的所有实验和长期训练都已通过。

## 实际交付

- 安装包 `dist/unified_marl-0.3.0-py3-none-any.whl` 和源码包；源文件仍集中在 `src/unified_marl`，没有执行旧 main 来代替新包。
- `wang/rode_hmf` 优先接通，随后 `wang/rode`；角色 Agent、角色选择器、动作编码器、角色 Mixer、源 Learner 和统一采样/保存恢复已接入。
- `wang/qmix`、`vdn`、`iql`、`coma`、`qtran` 及 `qmix_beta`、`vdn_beta`、`iql_beta` 交付配置。对应基础网络/Learner 与 PyMARL 字节相同，共用已验证核心；配置和命名空间分别保留。
- HMF 的 13 个公开调用如下表。均经过新包 API 的有效更新、保存、恢复和评估，并有原始源码损失/参数对照。
- 补货环境适配器和包内 YAML，任务/数据路径通过配置传给外部后端；真实后端不随包提供。
- 更新规格、计划、进度及 350 行迁移矩阵；保留阶段 1/2 矩阵快照。矩阵行数包含模块、配置、数据和服务，不能解释成算法数。

| 统一名称（均加 `wang_hmf/` 前缀） | 来源文件 `hmf_replenishment/algorithms/` | 本地接入范围 |
|---|---|---|
| dqn | dqn.py | 固定组转移协议，原生 DQN 更新 |
| vdn | vdn.py | 原生 VDN |
| qmix | qmix.py | HMF 来源实现，与 PyMARL QMIX 分开 |
| mfq | mfq.py | 原生均场 Q 与相邻时刻均场信息 |
| mfvfd | mfvfd.py | 原生局部/均场损失 |
| hierarchical | hierarchical.py | 原生默认分层更新 |
| hierarchical_vdn | hierarchical_vdn.py | 固定分组及可配置 mu/high 权重 |
| hierarchical_vdn_IPC | hierarchical_vdn_02_formulate_meanQ_IPC.py | 主入口实际选择的 IPC 实现，包含来源广播行为 |
| hierarchical_vdn_without_high_level | hierarchical_vdn_without_high_level.py | 无高层变体 |
| maddpg | maddpg.py | 离散动作 MADDPG 原生更新 |
| hierarchical_vdn_copy_fixed | hierarchical_vdn copy.py | 默认固定分组更新；动态方法未接入 |
| hierarchical_vdn_meanq | hierarchical_vdn_02_formulate_meanQ.py | 缺少主入口分支的独立均场 Q 变体，默认 update 已验证 |
| hierarchical_vdn_k_means_fixed | hierarchical_vdn_k_means.py | 默认固定分组更新；动态 K-means 调度未接入 |

后 3 项经过实际更新和数值对照后才开放，未按文件名 copy/k_means 判断重复或完成度。`copy` 与主版本的 __init__/update 等 10 个方法 AST 相同，但 init_from_env 和 update_dynamic_group 不同，因此保留单独来源和调用名称。详细比较见 `evidence/stage3/source_comparisons.json`。

## 批数据、采样和恢复

角色字段完整形状及时间规则：

| 字段 | dtype / 形状 | 写入与使用 |
|---|---|---|
| roles | int64 / `[B,T+1,N,1]` | t 状态选择的角色写在 t；0 和 role_interval 倍数重选，中间保持 |
| role_avail_actions | int32 / `[B,T+1,N,A]` | t 选中角色的动作空间，和环境 avail_actions 分别保存 |
| agent_Q | float32 / `[B,T+1,N,A]` | 采样时的个体 Q，供来源 RODE-HMF 构造均场输入 |
| mean_actions（MFQ） | float32 / `[B,T+1,N,A]` | 原生策略 t 输出的跨智能体均值，复制到各智能体；训练消费 t 与 t+1 |

- 最终 bootstrap 状态和动作也保存；filled 包含末状态槽，损失转移只取 `[:,:-1]`，不把末状态当转移。terminated 与 truncated 分开。
- HMF 默认将 terminated 或 truncated 都传为原生 done，保留来源在时间上限处停止 bootstrap 的目标语义。`hmf_bootstrap_time_limits=true` 才显式采用截断继续 bootstrap；PyMARL/RODE 保留其原 timeout bootstrap 规则。两种 HMF 配置均有专门检查。
- RODE 部分批次选择修复原代码对 `len(slice)` 和子集维度的假设，网络/损失不变。推理 `reset()` 清零循环状态；保留完整历史，角色周期从 t=0 开始。
- HMF 固定组沿原智能体顺序划分，当前要求 N 能被 num_groups 整除。原始 update/target-update 从包内实现调用，不经过旧脚本或源码搜索路径。
- HMF 默认更新规则取自固定入口：steps_per_update=84、num_batch=2，探索率 `max(0.05, 1.0-0.0001*已完成回合数)`；有独立配置和测试。短验证显式缩短更新间隔/批次数、关闭奖励标准化，相关覆盖保存在每次 config.json。
- 团队奖励复制给每个智能体，与来源补货 worker 一致。默认奖励标准化保留；零方差会明确报错，不能悄悄添加常数改动损失。
- 原 buffer 通过相邻存储位置寻找 next_action，旧 worker 自动 reset 会跨回合。新适配保存同回合明确的 next_obs/next_action；这是有记录的转移边界修复。MFQ 在回合开始重置上一动作均值。
- `policy.pt` 保存完整 Controller，包括角色/编码器/角色嵌入；`training.pt` 额外保存在线/目标混合器、均场网络/Attention、所有已使用优化器、回放、原生计数器、选择器调度和全局 RNG。
- 角色两个算法和 HMF 13 个调用在确定性 toy 的回合边界恢复后，与连续训练的 Learner 状态逐项完全一致。本结论仅覆盖聚类前和确定性 toy，不包含外部仿真内部状态。

## 验证结果

| 检查 | 实际结果 | 证据 |
|---|---|---|
| 完整回归 | 59 passed，0 failed；包含阶段 2 的 21 项 | stage3/tests.xml、pytest.log |
| 新调用链 | 23 个算法/配置均有效更新，参数变化量 >0，损失有限，保存和评估成功 | public_api_probes.json |
| 角色来源对照 | 两算法：网络输出、角色选择、损失和更新后参数误差为 0 | parity-rode*.json |
| HMF 来源对照 | 13 实现：固定转移输入，原始损失与更新后网络误差为 0 | parity-hmf-*.json |
| 角色周期/掩码/子集 | 通过；保留来源探索和空交集行为 | role-masks.json、tests.xml |
| MFQ 均场/终止位置 | 通过；双回合 8 条转移，bootstrap 槽未重复采入 | mean-field-boundaries.json |
| 聚类组件 | SC2 兼容十动作嵌入的空间/角色 latent 对照误差 0 | clustering-parity.json |
| 更新调度 | 12 步在第 8 步完成 2 次更新，探索率衰减符合原规则 | hmf-schedule.json |
| 干净安装 | wheel 安装到 .venv-check，在项目源码目录外调用 24 条链路（23 新项 + 原 QMIX），每项 2 次有效更新及评估；CLI 通过 | installed_validation.json、installed.log |
| 依赖 | 两个隔离环境 pip check 均通过 | check.log、installed_packages.json |
| 原始文件 | 1228 条源码/资源 SHA-256 复核一致 | summary.json；阶段 1 manifest |
| 真实 SMAC | BLOCKED：缺少 smac/SC2 | installed_validation.json 的完整 traceback |
| 真实补货 | BLOCKED：缺少 ReplenishmentEnv、任务和数据；仅模拟后端契约测试通过 | installed_validation.json、replenishment-contract.json |

来源对照不执行旧入口，只临时导入原类。原 HMF 固定写 CUDA，参考测试明确改用 CPU 属性；utils.misc 的未使用 cv2.log 导入在参考测试中隔离，新包移除该未用导入。未伪造任何网络或损失。原文件无修改；对照生成的 Python 3.12 缓存已移除，后续测试禁写来源 bytecode。

开发环境：Python 3.12.14 / torch 2.6.0+cpu / NumPy 1.26.4 / PyYAML 6.0.2 / sklearn 1.5.2 / scipy 1.14.1 / gym 0.26.2 / cloudpickle 3.1.1。独立安装验证的 sklearn 1.9.1 / scipy 1.17.1 / cloudpickle 3.1.2；全部版本见 installed_packages.json。GPU、其他 Python 版本和长期收敛实验未运行。

## 未完成项与来源问题

| 来源/能力 | 状态与具体原因 |
|---|---|
| hierarchical_qmix.py | 构造失败：QMixNet.__init__ 不接受 source Agent 传入的 input_dim 等参数；需先核实正确 Mixer 版本/接口，未擅自替换数学实现 |
| hierarchical_vdac.py | 一次更新失败：unsigned char 掩码不被当前 PyTorch masked_fill 接受；此外存在 update_one_group / 多种 update_high_level，未确认完整执行调度，未注册 |
| hierarchical_vdn_IPC.py | 与公开 IPC 名称对应文件不同；默认 update 出现 BatchNorm running_mean 6 vs 5 维不匹配；动态分组入口也未接通 |
| hierarchical_vdn_roma.py | 构造失败：NN_HIDDEN_SIZE 在赋值前使用；角色和动态分组流程尚未建立 |
| mfac.py | init_from_env 缺少 num_evaders；update 使用 next_state/rewards/log_probs/values/entropy/done 的同策略协议，不能套用六字段离策略转移桥 |
| main_replenishment_IPC.py | 动态分组的 agents_group/next_agents_group 采集、中心选择、分组更新未迁移；固定 IPC 算法通过不代表动态主程序通过 |
| copy/k_means 动态方法 | 只验证默认固定组 update，不能称已迁移动态聚类版本 |
| RGHMF q 角色/选择器 | 源模块保留但未算迁移完成；QSelector 固定输出 3 角色，默认配置为 2，缺少活跃交付配置及扩展后参数/调度验证 |
| RODE 聚类后的训练 | 仅组件聚类对照通过；来源对 SC2 动作 0/1 和前 6 动作有特殊规则，小动作空间可能没有两个可用聚类；不擅自改成通用聚类 |
| 动态角色扩展和恢复 | 新增角色参数未加入源优化器；RODE-HMF 的 n_groups 固定，扩角色后可能维度不兼容；扩展后的完整训练恢复明确拒绝 |
| 真实补货 / SMAC / CUDA | 外部条件不具备或尚未验证，保留未通过验收项 |

8 个最初没有直接初始化分支的文件都实际做了默认路径探测，记录在 remaining_variant_probes.json；其中 3 个随后通过完整新包接入和来源对照，5 个仍阻塞。缺少 main 分支本身没有被当作废弃依据。

额外保留并记录的来源行为：

1. RODE-HMF 的 mf_critic 和 attention 不在原优化器参数列表中；下一分组 reshape 使用当前分组；目标均场值由在线 mf_critic 计算。未擅自改正。
2. 固定 IPC 的值损失存在 `[batch*groups,1]` 对 `[batch*groups,actions]` 广播。PyTorch 警告被保留，原/新输出一致不代表此损失符合论文。
3. RODE 的探索动作只受环境 mask 限制，可离开角色 mask。空角色交集时 argmax 可能返回合法动作 0；仅环境不合法时再取随机合法动作，保留原行为。
4. RGHMF 基础 epsilon/multinomial 选择器调用扩展调度器时缺两个参数，原类构造实际报错；基础网络/Learner 与 PyMARL 字节相同，统一包采用已验证核心选择器。原版 RODE YAML 没有 n_roles，默认显式补为 2；原 episode+batch_size_run=8 冲突按单环境 1 修正，COMA 保留同策略并行批次。
5. HMF 原动作选择器无可用动作掩码机制；适配器检查并拒绝不合法输出，未修改其选择分布。当前验证使用所有动作合法的 toy 配置。

## 已执行命令

以下项目内命令 cwd 为 `G:\AI\Codex\多智能体代码整合\unified-marl`；路径均在项目隔离环境中。PowerShell 临时把 TEMP/TMP 指向 `.work/tmp`，PYTHONIOENCODING=utf-8、PYTHONDONTWRITEBYTECODE=1。

```powershell
# 来源比较、真实调用及失败探测，均 exit 0；探测脚本把失败写入 JSON，不将其记为通过。
& .venv/Scripts/python.exe -B tools/audit_stage3.py
& .venv/Scripts/python.exe tools/probe_stage3.py
& .venv/Scripts/python.exe tools/probe_wang_variants.py

# 最终完整验证：exit 0，59 passed。
& .venv/Scripts/python.exe -B -m pytest tests -q --junitxml=docs/evidence/stage3/tests.xml

# 构建、安装：exit 0。
& .venv/Scripts/python.exe -m build --no-isolation
& .venv-check/Scripts/python.exe -m pip install 'dist/unified_marl-0.3.0-py3-none-any.whl[role,hmf]' --cache-dir .work/pip-cache
# 收尾加入 HMF timeout 语义检查后重建同版本 wheel，并从本地离线重装最终产物：exit 0。
& .venv-check/Scripts/python.exe -m pip install --no-index --no-deps --force-reinstall dist/unified_marl-0.3.0-py3-none-any.whl

# 两项 exit 0：No broken requirements found。
& .venv/Scripts/python.exe -m pip check
& .venv-check/Scripts/python.exe -m pip check

# 更新矩阵、复核来源哈希：exit 0，59 tests，15 parity reports，1228 hashes，350 rows。
& .venv/Scripts/python.exe -B tools/record_stage3.py

# 验证文档/清单/来源哈希，以及源码、wheel、实际安装文件逐字节一致：exit 0。
& .venv/Scripts/python.exe -B tools/verify_stage3.py
```

源码目录之外的验证实际 cwd 为 `G:\AI\Codex\多智能体代码整合`：

```powershell
& ./unified-marl/.venv-check/Scripts/python.exe -B ./unified-marl/tools/validate_stage3_installed.py
# exit 0；24 条链路各 2 次有效更新，CLI exit 0；真实环境两项 BLOCKED。
```

日志中早期一次掩码断言失败，是测试错误地假定来源会对空角色交集重新贪心选择。检查原函数后将断言改为来源实际的动作 0 行为，未改算法；最终完整测试通过。独立安装验证未导入参考源码，包位置为 `.venv-check/Lib/site-packages/unified_marl`。

## 下一步

等待用户第 4 阶段指令：寇谦离线/偏好、old_src 与数据能力。以上来源问题、真实补货/SMAC 和动态角色/分组验证继续作为明确技术欠项，不自动归为完成，也不自动更改算法逻辑。
