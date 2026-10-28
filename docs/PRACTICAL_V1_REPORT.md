# 0.9.0.dev1 本地实用首版验收

日期：2026-10-28（Asia/Tokyo）。本轮目标是基于现有 unified-marl 完成算法和环境解耦的实用首版；不扩大到全部研究分支、连续动作统一、K8s 或发布远端。

## 已完成的使用路径

1. 环境通过 `EnvironmentSpec/register_environment` 注册，声明能力、默认配置和可选依赖；算法通过 `AlgorithmSpec/register_algorithm` 独立注册 Controller、Learner、默认参数、数据要求和采样方式。
2. 通用在线训练选择算法与环境时按能力校验，取消原来各训练分支的环境名称白名单。新增算法不需要修改环境或核心分发函数。
3. 新插件目前支持固定智能体数、离散动作、团队标量奖励、全局状态与动作掩码；支持回放或新鲜 rollout 两种训练生命周期。额外字段显式声明，禁止覆盖 `filled/actions_onehot` 等核心字段，未声明的采集字段会报错。
4. 插件列表写入配置和 checkpoint；新进程评估、加载、恢复和 spawn 环境 worker 会重新导入。新增的自定义训练状态保存/恢复钩子支持额外优化器与模型状态。
5. `run` 执行单个 YAML；`run-suite` 按算法、环境、训练 seed 组成矩阵；`check` 预检；`catalog` 展示首版算法和环境能力。实验 YAML 相对路径以 YAML 所在目录解析，未知参数和隐式替换环境被拒绝。
6. 每个实验独立保存配置、训练日志、模型、训练状态与独立评估结果；矩阵输出 JSON/CSV，仅在同一算法/环境条目下统计训练 seeds，不合并不同环境的奖励。
7. 接入交付 MAPT 的 MPE `simple_spread` 粒子物理；Huber-IQL 是新编写的教学插件，用来证明单个算法模块可跨环境使用，不计作三位作者原创。

入口：[使用指南](PRACTICAL_V1_GUIDE.zh-CN.md)、[新增算法](ADDING_ALGORITHM.zh-CN.md)、`examples/experiments/plugin_mpe.yaml`、`ppo_mpe.yaml`、`practical_suite.yaml`。

## 实际仿真链路矩阵

| 环境 | QMIX | VDN | SADMA-PPO | Huber-IQL 示例 | 实际后端及数据 |
|---|---|---|---|---|---|
| inventory | PASS | PASS | PASS | PASS | 包内库存动力学，既有合成双仓/SKU需求场景；不是外部 ReplenishmentEnv |
| mpe | PASS | PASS | PASS | PASS | 从交付迁移的 NumPy 粒子动力学，simple_spread，3智能体、离散5动作 |
| traffic_engine | PASS | PASS | PASS | PASS | 原生 CityFlow 0.1，引擎 .so，既有生成2×2路网/车流；保留既有可选服务约束 |

12/12 组合实际完成：2次有限梯度更新、非零参数变化、保存加载权重一致、合法动作、全新 Python 进程独立评估、恢复后累计3次更新、原 checkpoint 和输入资产摘要保持不变。

Windows 库存/MPE：Python 3.12.14、torch 2.6.0+cpu、NumPy 1.26.4。证据：

- `results/practical-v1-env/20261028T063806Z-36d57543/summary.json`
- 同目录 `evidence-clarifications.json` 说明旧开发环境的 distribution 元数据与当前源码加载路径，以及 MPE 的正确来源。

Linux 交通：本机 Docker Desktop，Python 3.12.15、torch 2.6.0+cpu、CityFlow 0.1。使用已有 `unified-marl-demo:0.8.2` 镜像（ID `sha256:ee98d85031b5a0ec5c40593405ab7543e7c5f0db0fae28f09176992efbe65d4b`），禁网、禁止拉取、限2CPU，挂载本轮 `/work/src`；镜像名不代表加载了旧代码。

- `results/practical-v1-env/20261028T064617Z-810d47ed/summary.json`
- 同目录 `container-execution.json` 记录完整命令、镜像与退出码。

这两次矩阵在本轮开发中执行，源码哈希逐次记录；当时模块版本尚未从0.8.2改为0.9.0.dev1。之后的修改包括预检边界、恢复钩子检查、异常清理和版本/文档更新，最终包另行安装验证。不得把这些短跑描述成长期收敛、论文复现或真实业务效果。

## 测试与最终包

- 回归：`tools/run_practical_regression.py tests --ignore=tests/test_zmq.py -k 'not zmq' -q --tb=short --disable-warnings`，退出0，**319 passed、1 deselected**。本轮没有重跑ZMQ/跨机。结果：`results/practical-v1-tests/20261028T064948Z-3c753400/junit.xml`。
- 最后插件/实验专项复验：**54 passed**，退出0，结果：`results/practical-v1-tests/20261028T065516Z-2bc72cdc/junit.xml`。这些测试与上述回归重叠，不相加为总测试数。
- MPE 40项行为测试包含与交付20步轨迹、观测、奖励逐值一致，种子隔离、碰撞、时间截断和生命周期；插件20项包含跨环境真实更新、独立进程与并行恢复、自定义字段及checkpoint。均包含在对应测试记录内。
- 首次交通单元回归受沙箱系统临时目录权限影响；测试工具改用项目内临时目录后18项通过。该问题未被当作算法通过或真实引擎证据。
- wheel：`dist/practical-v1/unified_marl-0.9.0.dev1-py3-none-any.whl`。387份运行源码/资源与当前源码逐字节一致；最终SHA-256见 `results/practical-v1-package/package.json` 和交付校验文件。
- 包安装在独立 `.work/practical-v1-installed` 中，依赖只读复用现有 `.venv`；确认实际从新环境的 site-packages 导入。这是包隔离安装，不是从零供应依赖的干净安装。
- 安装版完成 MPE 插件训练/恢复/评估、库存训练/评估、PPO YAML示例，以及 CLI运行的 QMIX/VDN/Huber-IQL × MPE/inventory ×2训练seeds，**12/12成功**。证据：`results/practical-v1-package/installed-smoke.json`、`cli-suite/suite_result.json`、`ppo-example/experiment_result.json`。

参考源保护：`source_manifest.json` 中1218项、`supplemental_manifest.json` 中10项全部哈希一致，无缺失或改动。记录：`results/practical-v1-package/source-reference-audit.json`。原交付和原发布附件未覆盖；本轮修改前的主要源码及文档备份在 `.work/practical-v1-backup`。

## 保留的边界与后续

- 自带MPE仅simple_spread；足球、MuJoCo、机械手等尚未进入首版统一矩阵。SMAC和原ReplenishmentEnv保留适配器，本轮未安装或实跑。
- 电网连续动作继续保留既有应用入口；通用离散插件不能用于该入口，不进行隐式离散化。
- 作者角色/均场、离线/偏好及奖励模型保留既有命名空间与限制；本轮回归不等于它们已在三个环境全部接通。其余环境与特色分支按后续优先级逐项迁移。
- 训练恢复保存学习器、优化器、回放与全局RNG等；外部模拟器内部物理状态不在统一恢复范围，不能承诺任意中途逐位复现。
- 单独评估seed不等于独立业务数据。示例CSV/路网用于链路检查，真实研究需准备独立测试场景和足够训练预算。
- 尚无本轮GPU、长期效果、跨机、生产部署验收；本地验收时尚未发布；后续GitHub状态见 docs/PUBLICATION_STATUS_090.md。
