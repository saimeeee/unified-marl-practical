# 仓储库存补货说明（0.8.0）

## 用途

每个智能体对应一个 warehouse_id / sku_id。输出在当前周期应该补多少货，目标是销售收入减采购、持有和缺货成本。可选 MFVFD、HMF 分层固定组、RODE-HMF 初始角色路径，以及基础对照；新应用封装没有修复原动态分组/扩角色缺项。

内置 `backend: inventory` 是本版新增、公式明确的库存环境，复用既有学习器。它不是缺失的原 ReplenishmentEnv，也不声称复刻原论文环境。原后端仍可通过 `backend: replenishment` 与原 make_env 参数接入。

## 输入文件

`warehouse_agents.csv` 的行顺序固定模型智能体顺序：

| 字段 | 单位和含义 |
|---|---|
| agent_id / warehouse_id / sku_id | 唯一智能体标识、仓库、商品；仓库商品对不得重复 |
| initial_stock / capacity | 初始库存、该仓库该SKU的独立库存容量，商品单位 |
| mean_demand | 业务方提供的历史平均周期需求，大于0；不能使用当前尚未发生的需求 |
| lead_time | 整数周期；0表示下单当期到货，1表示下一期到货 |
| unit_price / unit_cost | 每商品单位的售价/采购价 |
| holding_cost / shortage_cost | 每周期每单位库存持有/未满足需求惩罚 |

`warehouse_demand.csv` 为长表 period,agent_id,demand；period从0到horizon-1，每个智能体每个周期恰有一行，缺失或重复拒绝。训练时该列是环境逐期揭示的需求，观测不暴露未来需求。模板数字仅示例，可替换真实历史序列；重复同一需求序列训练可能过拟合，由使用方自行安排训练和评估时段。

配置路径相对YAML解析。horizon的单位与lead_time、mean_demand及需求表完全一致。backlog=false表示未满足需求丢失；true表示延至下期，并对当期全部未满足量收缺货惩罚。

## 动作与成本

动作索引选择补货倍率，默认沿用来源34档倍率，乘固定mean_demand得到请求数量。执行数量=min(请求量, capacity-现货-在途总量)，下限0。容量为每仓库SKU独立预留，不是整个仓库共享容量；订单可为小数单位，不做隐藏整数舍入。每期顺序：下单并收采购成本→接收当期到货→满足需求和历史欠货→计算收入、期末持有及缺货成本。未到货订单也在下单期付款。最后一个周期没有残值、订单退回或额外终端惩罚。

团队奖励=各智能体利润之和/reward_scale，个体利润独立保存。容量限制作为显式业务执行规则处理，各档位仍是合法动作，不改变HMF来源选择器。

## 选择算法

默认mfvfd。`algorithm` 还可填 hierarchical、hierarchical_vdn、hierarchical_vdn_IPC、hierarchical_vdn_without_high_level、hierarchical_vdn_copy_fixed、hierarchical_vdn_meanq、hierarchical_vdn_k_means_fixed、hierarchical_vdac_fixed、rode_hmf 等已经接入的固定路径。完整列表按包中registry和支持清单，不能通过填未迁移名称启用算法。

固定HMF要求智能体数能被training.num_groups整除；默认2组。若只有一个智能体可改1组并选支持的基础实现。RODE-HMF沿用至少2智能体、batch_size>=2等来源约束。HMF可调 batch_size/buffer_size/hidden_dim/n_heads/embed_dim，embed_dim须能被n_heads整除；示例设置 hmf_steps_per_update=1、hmf_norm_rewards=false是明确应用配置，不改底层默认值。

## 调用与业务结果

```python
import json
from unified_marl.applications import warehouse
result=warehouse.train("my-warehouse/warehouse.yaml",output_dir="results/warehouse-001")
policy=warehouse.load_policy(result.checkpoint)
policy.reset()
with open("my-warehouse/warehouse_state.json",encoding="utf-8") as f:
    state=json.load(f)
decision=policy.act(state)
print(decision["decisions"])  # agent_id,warehouse_id,sku_id,action_index,requested_order,order_quantity
```

命名状态：period和agents列表。每项包含agent_id、stock、backlog、last_demand、pipeline；pipeline每项arrives_in为距当前周期的到货周期、quantity为数量。必须完整提供所有智能体，数量单位与CSV一致。

`encode(state)` 也可得到模型观测：库存/capacity、欠货/capacity、mean_demand/capacity、上期需求/capacity、period/horizon、按到货周期排列的在途量/capacity；在途桶数=max_lead_time+1。直接传观测数组时必须严格使用这一布局。没有商业状态时不要把决策数量当成已实际下单，业务系统仍应接收并执行返回结果。

本版未执行上述命令、库存仿真或训练实验，由用户自行测试。

## 0.8.1 恢复修复

用户在Linux报告0.8.0合成链路可运行，但应用API/CLI恢复因自动seed字段误判失败。0.8.1修复该比较；使用`overrides={"training.max_updates":2}`或CLI `--set training.max_updates=2`指定目标总更新数。评估新增episode_lengths/env_steps。完整训练验证由用户按[二测提示词](LINUX_ROUND2_PROMPTS.md)执行。
