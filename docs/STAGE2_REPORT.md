# 第 2 阶段验证报告

第 2 阶段已实现并通过本地验收；真实 SMAC 被缺少 SMAC/SC2 环境阻塞，未宣称通过。用户已授权结果记录后自动继续第 3 阶段。

## 实现内容

- 可安装 src/unified_marl 包和 pyproject.toml，Python API train/evaluate/load_policy，以及 unified-marl / python -m unified_marl CLI。
- 原版 PyMARL 的 EpisodeBatch、ReplayBuffer、BasicMAC、RNNAgent、Q/COMA/QTRAN Learner、QMIX/VDN/QTRAN Mixer 和 COMA Critic。
- 原算法 YAML 与 beta 配置随 wheel 打包。配置优先级、组合校验、显式输出路径和可选环境延迟导入。
- EpisodeRunner 与 spawn 安全 ParallelRunner。保留 bootstrap 末状态/动作；记录真实终止和截断；关闭、超时与 worker 异常有明确处理。
- policy.pt 与 training.pt 区分。后者保存目标网络、优化器、回放、计数器和全局 RNG，支持回合边界继续训练。
- COMA 使用当前策略新鲜 rollout，不复用离策略样本。循环策略推理支持 reset。

## 实际结果

- 21 项测试通过。五算法分别完成有效参数更新、有限损失、保存加载、评估、动作掩码和隐藏状态重置检查。
- 固定输入种子 2026：QMIX、VDN、IQL、COMA、QTRAN 的网络输出、损失和一次更新后参数最大绝对误差均为 0。
- QMIX 在 toy 中分两段恢复到 24 环境步，与连续训练的 agent、target agent、mixer、target mixer 参数逐项完全相同，计数均为 6 回合 / 5 更新。
- 第二个干净环境 .venv-check 安装 wheel 后，在工作区父目录（非项目/源码目录）调用；8 个算法/配置各完成 2 次有效更新并评估。
- pip check 通过；wheel 和源码发行包构建成功。
- 全部 1228 条盘点来源哈希仍一致。

## 真实执行命令

以下以项目为 cwd；外部调用行单独注明。

```powershell
.venv\Scripts\python.exe -m pip install --no-build-isolation --no-deps -e .
.venv\Scripts\python.exe -m pytest -q --junitxml=docs/evidence/stage2/tests.xml
.venv\Scripts\python.exe -m build --no-isolation
.venv-check\Scripts\python.exe -m pip install --no-index --no-deps --force-reinstall dist/unified_marl-0.2.0-py3-none-any.whl
.venv-check\Scripts\python.exe -m pip check
# cwd 为 G:\AI\Codex\多智能体代码整合：
unified-marl\.venv-check\Scripts\python.exe unified-marl\tools\validate_installed.py unified-marl\docs\evidence\stage2
```

测试环境 Windows / Python 3.12.14 / torch 2.6.0+cpu / NumPy 1.26.4 / PyYAML 6.0.2。项目 .venv 和 .venv-check 独立于应用/全局 Python；所有临时目录与 pip cache 放在项目 .work 内。

## 限制与后续

真实 SMAC 调用明确抛出缺少可选依赖的错误，不回退到 toy。GPU、Python 3.10、旧 SC2 beta 真实场景和长时间收敛均未验证。完整训练状态不捕获外部仿真器内部状态，仅对确定性 toy 的恢复声明精确一致。

基于固定源实现进行对照，不擅自更改 QTRAN 的来源隐藏状态排列或损失。buffer extend、清除旧 padding、空动作掩码校验和 padding 归一化稳定性是明确兼容修复。

证据见 docs/evidence/stage2：tests.xml、parity-*.json、resume.json、installed_validation.json、summary.json、build.log、requirements-validated.txt、migration_sources.json。训练和推理产物的实际路径记录在 installed_validation.json。

第 1 阶段工具 verify_stage1.py 的“尚无 src”等断言已成为历史快照，不再用它判断第 2 阶段。build_migration_inventory.py 会重建初始清单，后续不得用它覆盖现有迁移状态。
