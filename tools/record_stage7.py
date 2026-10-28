"""Write human delivery reports strictly from recorded release acceptance."""
from collections import Counter
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/"docs/evidence/stage7"
base=json.loads((EVIDENCE/"release-base.json").read_text(encoding="utf-8"))
full=json.loads((EVIDENCE/"release-full.json").read_text(encoding="utf-8"))
audit=json.loads((EVIDENCE/"migration-audit.json").read_text(encoding="utf-8"))
suite=ET.parse(EVIDENCE/"tests.xml").getroot().find('testsuite')
assert int(suite.attrib['tests'])==220 and int(suite.attrib['failures'])==int(suite.attrib['errors'])==0
assert base['status']==full['status']=='PASS' and len(base['runs'])==12 and len(full['runs'])==28
summary=dict(version='0.7.0',tests=220,warnings=14,base_runs=12,full_runs=28,
    matrix_rows=audit['matrix_rows'],review=audit['review_decisions'],trace_records=audit['trace_records'],
    source_code_dependency='No original code/config reads permitted in standalone validation; explicit real .th input allowed',
    real_data='3m short updates only', real_environments_verified=False,cross_machine_verified=False,
    gpu_verified=False,policy_effect_verified=False,stage7_status='DELIVERY_ACCEPTED_WITH_RETAINED_UNMET_ITEMS')
(EVIDENCE/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')

issues='''# 已知问题、未通过项与后续工作

版本 0.7.0，2026-10-21。所有历史 379 行保留，新增 4 项共 383 行；本阶段不删除未完成项，也不将其改成范围外来宣称完成。301 行有本地证据且限定通过范围，78 行在阶段 1—7 内尚未验收，另有原已约定的 4 项 Web/K8s 可选阶段 8 未执行。行数包含配置/组件/外部能力，不是算法数，部分本地通过行仍有真实环境或分支欠项。

## 共同限制与优先顺序

1. 获取 SMAC/StarCraft II/地图、ReplenishmentEnv/数据和 CityFlow/路网资产，先做真实短训练/评估；本机缺失记录仍为 NO。真实 3m 数据更新、toy 和模拟后端不能替代该项。
2. 向来源方确认缺失 Learner、参考策略、agent_state/indi_terminated/个体标签协议与交付权利。SADMA MADDPG 四份 Policy 均为空，不补造算法。易开源/ZMQ 的 MIT 已保留，其他部分授权仍待确认。
3. HMF 四个失败实验已复跑，具体堆栈在 remaining-probes.json。MFAC 除必填参数外还混用了不存在的 critic/optimizer 和 detacj 拼写；不是补一个参数即可完成。hierarchical_qmix 构造器和 Mixer 的输入契约不同，IPC 维度冲突，ROMA 常量未赋值，不能猜测网络/损失修复。新增 VDAC 固定 critic 变体没有实现缺失的 actor/高层策略。
4. RODE 动态扩角色会遗漏新参数的优化器管理；HMF 动态 IPC/Kmeans 和高层调度、Kou PER/e2e/moderate-CQL、完整交互/无监督奖励流程仍需单独实现与验证，保留原范围。三种额外离线 Agent 变体已经补验，GRNN 等所需特征/通信协议仍缺。
5. 连续 PowerGrid 需要浮点动作/边界/高斯概率和独立数据生命周期；来源缺默认数据、当前缺 pandapower。不能通过强行离散化或复用整型动作批次验收。
6. 分布式只有 Windows 本机回环证据；无远程机器/GPU。协调器串行 Learner/推理 RPC，不宣称计算完全重叠、动态推理合批、向量 worker、多 Learner 或加速效果。周期快照评估已接入，旧永久共享缓冲 worker 和向量/远程评估仍未复刻。
7. 来源保留的数学问题：RODE-HMF 均场参数/目标问题、IPC 广播、PPO 终止 bootstrap/熵项自归一化/LR、NMR 首段预测复用、PrefTransformer 未注册层与 dropout、diverse 后续空信任池。来源损失一致不代表理论正确、长期稳定或策略收敛。其他 Python/PyTorch/OS 的兼容范围未验。

## 本阶段发现并修复的集成问题

- 评估切换环境曾泄漏原 SMAC 参数，已按目标环境默认值重建后再覆盖，并增加回归。
- 旧字节掩码只在 uint8 时转 bool；首轮对所有整数掩码转换导致 hierarchical 来源对照失败，已纠正，最终 14 项 HMF 对照通过。初始失败 XML 保留，最终全套 220 项通过。
- 首次发行短任务沿用了易版 update_rate=8，预算不足以继续更新；验收脚本显式设 update_rate=1，保留实际解析配置，没有改动包内默认更新语义。
- Windows 验证脚本读取子进程 JSON 时显式采用 UTF-8，覆盖中文/空格路径。
- SMAC 可选依赖改用原交付 requirements 的 oxwhirl/smac Git 地址，避免仅凭同名包名选依赖；没有安装或验证该外部环境，Git HEAD 不是已验版本。

## 逐项未通过记录

以下逐行保留未通过原因；部分已验收行的额外欠项在下一表及迁移矩阵中。完整源路径见 MIGRATION_MATRIX.csv，机器审计见 evidence/stage7/migration-audit.json。

| ID | 能力 | 状态 | 具体原因 |
|---|---|---|---|
'''
for row in audit['unaccepted']:
    issues+='| '+ ' | '.join(str(row[k]).replace('|','/').replace('\n',' ') for k in ['id','capability','migration_status','blockers'])+' |\n'
issues+='\n## 部分分支通过，仍有欠项\n\n| ID | 能力 | 状态 | 保留原因 |\n|---|---|---|---|\n'
for row in audit['partial']:
    issues+='| '+' | '.join(str(row[k]).replace('|','/').replace('\n',' ') for k in ['id','capability','migration_status','blockers'])+' |\n'
(ROOT/'docs/KNOWN_ISSUES.md').write_text(issues,encoding='utf-8')

validation=f'''# 交付验收报告

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
| 迁移追踪 | 保留全部 379 历史行，新增 4 行；逐文件 {audit['trace_records']} 条来源/目标及哈希 | MIGRATION_MATRIX.csv、MIGRATION_TRACEABILITY.csv、包内 PROVENANCE.json |

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
'''
(ROOT/'docs/VALIDATION_REPORT.md').write_text(validation,encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False))
