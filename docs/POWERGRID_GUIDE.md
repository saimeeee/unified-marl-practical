# PowerGrid 连续控制应用

PowerGrid 通过 SADMA 的 `GridEnv` 与建筑能量模型运行 case33bw 配电网。环境保留外部气候、建筑和仿真资产要求；连续策略由本包新增的 `bounded_ppo` 提供。当前整合没有运行训练、评估或电网仿真实验；请在准备好来源数据后按本指南自行测试。

## 场景文件

在配置中提供气候/建筑数据目录和建筑状态动作 JSON 的路径。目录至少需要：

```text
Climate_Zone_1/
  weather_data.csv
  solar_generation_1kW.csv
  building_attributes.json
  Building_<Building_Type>.csv
buildings_state_action_space.json
```

`building_attributes.json` 中每个被选中的 ID 都要有 `Building_Type`，且目录内要有相应的 `Building_<Building_Type>.csv`。状态动作 JSON 和属性文件必须使用相同的建筑 ID。仿真器会在初始化时检查这些文件；缺失文件、ID 或类型 CSV 会得到指出具体路径的异常。仓库不携带这些数据资产。

模板位于 `src/unified_marl/applications/templates/powergrid.yaml`：

```yaml
schema_version: 1
application: powergrid
algorithm: bounded_ppo
seed: 1
device: cpu
execution: local
env_args:
  data_path: ./assets/Climate_Zone_1
  buildings_states_actions_file: ./assets/buildings_state_action_space.json
  climate_zone: Climate_Zone_1
  hourly_timesteps: 4
  houses_per_node: 6
training:
  updates: 100
  horizon: 256
  mini_epochs: 4
  hidden_dim: 128
  lr: 0.0003
  gamma: 0.99
  gae_lambda: 0.95
  eps_clip: 0.2
  entropy_coef: 0.0
  critic_coef: 0.5
output: {}
```

相对路径按配置文件所在目录解析。`building_ids` 可缩小抽样池；`building_sampling_weights` 可提供 `{building_id: positive_weight}`，未列出的 ID 权重为 1。省略权重时，未指定 ID 且池内至少有 9 个 ID 的场景保留来源按位置 1、4、5、6、7、8 加权 10 倍的抽样分布；显式 ID 默认均匀抽样，少于 9 个 ID 时也采用均匀抽样。这是对来源固定索引权重的参数化修复。

`episode_limit` 默认沿用来源的 92 步。`horizon` 是每次 PPO 更新最多采集的环境 transition 数；若它在一个 episode 中到期，该片段按时间截断处理。`episodes_per_update`（默认 8）限制每次更新最多采集的 episode 片段数。`updates` 是恢复训练后的目标总更新数；新训练必须大于 0。`t_max` / `max_steps`（若提供）限制总环境步数。

## 调用

```python
from unified_marl.applications import load_config, powergrid

config = load_config("path/to/powergrid.yaml")
result = powergrid.train(config, output_dir="results/powergrid-run")
print(result.checkpoint)          # policy.pt，可用于推理/评估
print(result.training_checkpoint) # training.pt，可恢复训练
print(result.metrics)             # 最近一次 PPO 更新的指标字典

policy = powergrid.load_policy(result.checkpoint, config=config)
policy.reset()
decision = policy.act(observation, deterministic=True)
```

顶层 `powergrid` 门面会返回应用封装策略结果，含每个 agent 的控制向量。底层 `unified_marl.applications.powergrid_engine.load_policy()` 返回 `GridPolicy`；它的 `act(obs, deterministic=True)` 接收形状 `[n_agents, obs_dim]` 的观测，并返回形状 `[n_agents, action_dim]` 的物理连续动作 `numpy.float32` 数组。策略保留循环状态，开始新 episode 前调用 `reset()`。

调用 `powergrid.evaluate(checkpoint, episodes=3)` 可在保存的场景上评估；评估会重新创建真实仿真环境，仍然需要全部外部资产与 PowerGrid 依赖。要恢复训练，提供同场景、同 seed 配置，并把新输出目录传给 `powergrid.train(..., resume=result.training_checkpoint)`；包装层要求输出目录为空。`updates` 表示恢复后的目标总更新数，不能小于 checkpoint 已完成的更新数。`policy.pt` 仅包含推理所需策略与场景规格，`training.pt` 另含 critic、优化器、ValueNorm、更新计数、种子、Python/NumPy/Torch RNG 和环境 episode/action-space RNG 状态。恢复入口只接受完整 `training.pt`；`policy.pt` 不能作为 resume 输入。

## 连续动作定义

策略把高斯 latent action 经 `tanh` 和仿射变换映射到环境的 Box 边界，并在 PPO 概率中包括该变换的 Jacobian。环境边界当前为每个启用控制的 `[-1, 1]`。动作维度由状态动作 JSON 中实际启用的控制生成，按 `GridEnv` 消费动作的固定顺序过滤：

1. `cooling_storage`：冷储能充/放电比例；实现中正值充电、负值放电，并受设备、需求和 SOC 约束。
2. `dhw_storage`：生活热水储能充/放电比例；正值充电、负值放电，受可用功率和 SOC 约束。
3. `pv_curtail`：光伏出力比例；`-1` 表示完全削减，`+1` 表示不削减。
4. `pv_phi`：相位角；`-1` 映射为 0，`+1` 映射为 π/2。
5. `electrical_storage`：电池充/放电比例；正值充电、负值放电，受功率、需求和 SOC 约束。

因此，“4 维动作”仅表示该场景启用了五类控制中的四类，具体坐标必须读取 checkpoint 中的 `env_info.action_spec.names`；不应把四个坐标固定解释成同一组设备。为共享策略保证坐标含义一致，引擎要求所有生成建筑的动作名称顺序相同。`energy_models.py` 原先按 JSON 字典插入顺序建 action space、但 `Building.step()` 按上述顺序消费；此处已统一为 step 顺序，并保存每维名称。

## 来源适配与边界

环境来源为 `marl平台代码/易才润/论文/SADMA/sadma源代码开源版本/envs/powergrid/smart_grid.py` 和 `energy_models.py`。源文件保持不变，适配副本放在 `src/unified_marl/envs/powergrid_native/`：换成本包 `MultiAgentEnv` 导入，数据路径显式参数化，增加资产预检查，按真实动作空间汇报维度和名称，并把潮流计算失败时的 `quit()` 改成带原始异常链的 `RuntimeError`。它仍会重新生成 case33bw 与建筑负载；来源的 `net_path` 和 `agent_path` 未被使用，传入非空值会被拒绝。当前只支持来源固定的 case33bw 配电网和 `percent_rl=1`，不支持任意拓扑、自定义网络 JSON、部分 RL agent、多环境批处理或异步执行。

连续 PPO 是新增的 bounded Gaussian adapter，不声称迁移或复现来源连续策略的算法等价性。它保留来源共享循环 Gaussian actor、集中式 critic、PPO clipping、entropy、GAE 和可选 ValueNorm 风格；不同处为 tanh-affine 有界动作、匹配的 log-prob 修正、动作维度概率求和、区分终止与截断的 bootstrap/GAE mask，以及 update-boundary 完整训练 checkpoint。Entropy 项在未变换的 latent Gaussian 空间计算。episode_limit、horizon 与总步数的时间截断会 bootstrap，但停止跨截断边界递推 GAE；真实终止不 bootstrap。流程潮流失败会中止并报告，不会伪装成终止样本。

## 安装与检查范围

除 `unified-marl` 的基础依赖外，PowerGrid 环境需要 `pandapower`、`gym==0.26.2`、`pandas`、`scikit-learn` 与 `scipy`；版本范围见仓库根目录 `requirements-powergrid.txt` 和 `pyproject.toml` 的 `powergrid` extra。pandapower / SciPy 范围按来源 API 约束收窄；本轮没有安装和验证依赖组合。建议在独立环境安装。引擎模块导入和 `policy.pt` 推理策略构建不导入 pandapower；训练、评估和原生环境初始化会懒加载环境依赖。

本轮只进行代码语法/接口静态检查，没有运行训练、PowerGrid 仿真、评估、收敛或真实场景实验。外部资产、运行依赖完整性、潮流可行性和策略效果都需要接收方在提供数据的环境自行验证。

## 0.8.2修复

净用电量观测先转换数值数组再逐元素计算，不再列表重复；各序列必须等长且有限。初始化预热不计入策略控制步数，episode_limit=4对应4次外部step。info包含control_steps/initialization_steps和时间截断标志，评估新增env_steps。回合语义保存为control_transitions_v2，与旧电网checkpoint的恢复/环境评估不混用；第三轮启用该观测须新建模型。完整补丁也更新安装入口，加装功能不再恢复旧核心。见ENGINEERING_HOTFIX_082.md。
