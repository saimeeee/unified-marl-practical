# 交付验收报告

日期：2026-10-21；版本：0.7.0。结论：已验收支持子集可以安装调用，阶段 7 交付验收结束；全部原算法/环境的目标仍有明确未通过项，不标为全部完成。

## 实际通过

| 验收项 | 实际结果 | 证据 |
|---|---|---|
| 完整回归 | 220 passed，14 条来源保留的广播/弃用等警告 | evidence/stage7/tests.xml、pytest.log |
| 新建基础环境、源码目录之外 wheel 安装 | 12 条训练链路，有限损失、正参数变化、保存加载/评估、CLI、中文及空格路径 | release-base.json |
| 可选依赖隔离 | sklearn/gym/cloudpickle/ml_collections/transformers/scipy/zmq/smac 真实缺失时基础训练通过；角色/HMF/奖励/ZMQ 各自明确报错 | release-base.json |
| 新环境追加可选功能的发行验收 | 28 条链路；2 角色、2 HMF、7 SADMA 配置、3 离线 Agent 变体、4 普通/偏好/历史策略、6 奖励模型、4 ZMQ 组合 | release-full.json |
| 真实数据 | 3m.th 实际读取，7 个策略/Agent 配置各 2 次更新并恢复至 3，6 奖励模型更新/预测/加载；数据 SHA-256 在证据中 | release-full.json |
| 来源对照 | 既有全套损失/输出对照回归；新增 VDAC 固定 critic 与 3 个 Agent 变体的损失/参数误差为 0 | vdac-parity.json、agent-*/parity-policy-*.json、extra-agents-tests.xml |
| 恢复训练 | PyMARL/聚类前角色/固定 HMF/离线/SADMA 边界恢复回归通过；6 奖励模型固定偏好批次的下一次更新精确相同 | tests.xml、api-tests.xml、release-full.json |
| ZMQ | Q sync/async、PPO sync/显式 V-trace async，本机多进程、参数同步、端口重绑；原独立三角色及故障/超时回归通过 | release-full.json、tests.xml；历史 stage6 记录保留 |
| 周期独立评估 | 版本与快照记录、忙时跳过、故障/超时及自有进程清理 | periodic-evaluation.json、tests.xml |
| 迁移追踪 | 保留全部 379 历史行，新增 4 行；逐文件 353 条来源/目标及哈希 | MIGRATION_MATRIX.csv、MIGRATION_TRACEABILITY.csv、包内 PROVENANCE.json |

基础环境从 Python 的 venv 创建，include-system-site-packages=false；仅从先前下载的原 wheel 缓存安装依赖，没有复制现有 site-packages。基础验收后才追加 role/hmf/reward/distributed。独立 .venv-sdist 再验证源码包构建和基础安装，证据以 sdist-validation.json 为准。

独立验证脚本复制到项目之外 `.release-check` 执行；所有 unified_marl 模块来自隔离环境 site-packages，显式禁止读取原交付的 .py/.pyc/.yaml/.yml。真实数据作为显式外部 .th 输入；没有把数据或原源码路径注入发行包导入路径。

## 实际验证栈

Windows 11 10.0.26200（AMD64）；Python 3.12.14；PyTorch 2.6.0+cpu，CUDA 不可用；NumPy 1.26.4；PyYAML 6.0.2。可选部分：scikit-learn 1.5.2、gym 0.26.2、cloudpickle 3.1.1、scipy 1.14.1、ml-collections 1.1.0、transformers 4.53.0、pyzmq 25.1.2/libzmq 4.3.4。全部传递依赖版本在 release-base/full.json，构建依赖在 sdist-validation.json。元数据安装约束和 wheel 的 any 标签不等于其他版本/平台验收承诺。

没有真实 SMAC/PySC2/StarCraft II、ReplenishmentEnv、CityFlow 的可用运行版本。三次实际工厂调用均缺后端，错误已记录。未通过：真实仿真训练/评估、跨机器、GPU、Linux/macOS、其他版本和策略效果；它们没有被 toy 或真实离线数据更新替代。

## 实际命令

以下命令均已执行；路径相对项目，独立调用的工作目录为其兄弟 `.release-check`。

```powershell
.venv/Scripts/python.exe -B tools/probe_remaining_stage7.py
.venv/Scripts/python.exe -B tools/prepare_stage7.py
.venv/Scripts/python.exe -B -m pytest tests -q --junitxml=docs/evidence/stage7/tests.xml
.venv/Scripts/python.exe -B -m build --no-isolation --outdir dist
.venv-release/Scripts/python.exe -B -m pip install --no-index --no-deps dist/unified_marl-0.7.0-py3-none-any.whl
# 以下在 .release-check 内，用复制出的独立验证脚本执行：
../unified-marl/.venv-release/Scripts/python.exe -B validate_release.py --mode base --output base-acceptance-03
../unified-marl/.venv-release/Scripts/python.exe -B validate_release.py --mode full --output full-acceptance-02 --real-data "../marl平台代码/寇谦/论文代码及技术文档/论文代码及技术文档/MADiversePreferDatasets/3m/3m.th"
```

退出码均为 0。早期失败已保留原因并修复，不能将其当通过证据；最终以 tests.xml 和 release-*.json 的 PASS 为准。本机 TCP 由任务授权提升运行，默认工具沙箱拒绝 ZMQ context；未修改防火墙、全局 Python 或原始交付。

## 安装包与未通过项

发行文件为 dist/unified_marl-0.7.0-py3-none-any.whl 与 dist/unified_marl-0.7.0.tar.gz，SHA-256 见 dist/SHA256SUMS.txt。源码包含文档、最小示例、测试记录及清单摘要。详细安装/示例/重建证据随工作区 docs/evidence/stage7 交付；sdist-validation.json 与 final_verification.json 记录最终包的实际重建和哈希，不嵌入源码包，避免其自身的哈希循环依赖。

已验/未验支持范围见 SUPPORT_MATRIX.md；每项尚未通过的具体原因与后续工作见 KNOWN_ISSUES.md。许可复核见 LICENSES.md。阶段 7 完成后停止，不自动执行可选 Web/K8s 阶段。
