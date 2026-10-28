# Linux 第二轮最小测试提示词（0.8.1）

第一轮报告来自 Grok Bot 盒子，不是用户服务器；下列结果只归属于第二轮实际执行机器。补丁在 Windows 上仅做无仿真单元检查、构建和独立安装检查，没有在本机训练或测试 Linux。

将 `unified-marl-0.8.1-engineering-hotfix.zip` 上传到服务器，解压到 `~/marl-hotfix-081`。让该目录直接包含 `install_patch.py` 和 wheel。原 `~/marl` 0.8.0 基础包保持不动。

## 提示词 1：确认机器并应用补丁

```text
协助我完成 unified-marl 0.8.1 Linux 第二轮测试。CPU、单进程、seed=1，只验证功能，不做收敛实验。你不能操作本机时，每次给一小段 Bash 命令，等待我贴回输出，不声称已经执行。

第一轮报告是在 Grok Bot 盒子，不是我的服务器。先输出当前 hostname、系统、uname -m、Python 和包路径，明确本轮实际执行位置。测试限 x86_64、CPython3.12；不改系统 Python，不使用 sudo pip，不修改源码或算法。

原安装：~/marl，补丁解压根目录：~/marl-hotfix-081，测试目录：~/marl-test。
先确认 ~/marl/.venv/bin/python 存在。若本轮是新机器，先安装原0.8.0 Linux包的 warehouse 功能；完整记录新机器环境，不能沿用上一轮通过状态。

核对补丁 ZIP 的 SHA256 与随附 .sha256 文件相符，然后用原项目虚拟环境执行：
~/marl/.venv/bin/python ~/marl-hotfix-081/install_patch.py --bundle ~/marl

输出包版本、实际导入路径、pip check结果，要求 unified_marl.__version__ == '0.8.1'。
补丁只更新本地 wheel，无需联网下载依赖；不要把当前目录或源码插入 PYTHONPATH。

注意：原0.8.0安装脚本会重新安装0.8.0！如之后为了添加功能运行原脚本，完成后必须再应用本补丁。不能修改原manifest绕过校验。

命令和日志放 ~/marl-test/logs/round2-install.log。环境不符合或安装失败就记录原始错误，不伪造成功。完成后停止。
```

## 提示词 2：仓储一键回归

```text
执行第二轮仓储测试。先再次确认版本0.8.1，在源码目录之外运行补丁自带脚本，使用唯一的新工作目录，不覆盖第一轮结果：

mkdir -p ~/marl-test/logs
set -o pipefail
RUN="$HOME/marl-test/round2-warehouse-$(date +%Y%m%d-%H%M%S)"
timeout --kill-after=15s 300s "$HOME/marl/.venv/bin/python" -X utf8 -u "$HOME/marl-hotfix-081/round2_warehouse.py" --work-dir "$RUN" 2>&1 | tee "$HOME/marl-test/logs/round2-warehouse.log"
执行后立即记录真实退出码。不要只记录tee的退出码。

这个脚本使用2智能体、4周期合成库存示例，分别执行一次训练、Python API恢复至第2次更新、从同一首轮checkpoint经CLI恢复至第2次更新、保存加载、确定性推理、4步评估及拒绝非法恢复。脚本里出现测试输出不代表已通过；必须读取退出码和round2-results.json。

验收重点：
1. first.updates=1；resume_api和resume_cli的updates=2、env_steps增加、parameter_delta_l2有限且大于0。
2. 不再出现Cannot change environment arguments during resume。
3. 每个新结果目录都有application.json和模型。
4. evaluate返回episode_lengths=[4]、env_steps=4；不将bootstrap状态算环境步。
5. policy-only checkpoint、改backlog、改seed均被拒绝。
6. 补货量和动作索引合法，重载推理一致，损失有限。

把报告和实际日志路径告诉我。如果失败，只诊断并保留堆栈，不修改算法或安装包源码。完成后停止。
```

## 提示词 3：兼容第一轮旧模型（有旧文件再做）

```text
检查上一轮0.8.0的仓储运行目录是否仍存在，需包含training.pt、application.json和application-result.json，且application.json指向的原CSV仍可访问。

存在时，使用新的唯一work-dir，再运行补丁中的round2_warehouse.py，并追加：
--previous-run "上一轮真实结果目录"

设置300秒超时并保存日志。读取round2-results.json的previous_080_run，确认旧模型通过公开warehouse.train恢复，updates增加至少1，env_steps增加，参数确实更新，并生成新的应用元数据。

禁止用底层核心API绕过应用层，也不修改旧checkpoint。旧文件或原数据路径已丢失时标为“未测试：旧资产不可访问”，不要把新模型恢复成功当成旧模型兼容通过。完成后停止。
```

## 提示词 4：交通和电网前置条件与条件性短测

```text
继续交通与电网。它们上一轮因缺模拟器/数据而阻塞，不属于已经复现的训练缺陷。本轮先检查前置条件，禁止重复把导入或配置检查当训练成功。

交通：检查当前venv的cityflow.Engine，以及实际config、roadnet.json、flow.json和路口相位映射。原生CityFlow不是直接pip install cityflow即可获得；本次不做长时间源码编译。条件仍缺就列清单，记录阻塞。
若已齐全：复制测试配置，按真实路网填全部phase_ids；有效全红相位不可猜测。没有过渡相位的纯软件最小测试可显式yellow_time=0，并记录直接切换。PPO、CPU、1线程、seed1，episode_limit=8，max_updates=1，t_max=64，train_batch_size=2，train_buffer_size=4，update_rate=1。做一次有效更新、保存加载、1回合评估和合法相位推理，300秒超时，损失有限且参数变化。

电网：检查weather_data.csv、solar_generation_1kW.csv、building_attributes.json、每个Building_Type对应CSV、建筑状态动作JSON；建筑ID和列须一致。文件不存在时列出准确缺项，不造物理数据。
若资产已齐全：先用原Linux安装器--Feature powergrid安装可选依赖到项目venv，随后重新执行0.8.1 install_patch.py（原安装器会降级核心），确认版本0.8.1和pip check后才测试。
来源case33bw，CPU，seed1，单环境，updates=1、horizon=4、episodes_per_update=1、mini_epochs=1、episode_limit=4。做环境reset、一次有效更新、保存加载、实际观测推理、1短回合评估；动作有限、形状正确且在Box范围内。300秒超时；保留潮流异常，不用假终止绕过。

每个方向分别报告：依赖检查/配置检查/真实环境启动/更新/推理/评估，状态使用通过、失败、阻塞、未测试。完成后停止。
```

## 提示词 5：汇总

```text
将第二轮实际结果汇总到~/marl-test/TEST_REPORT_ROUND2.md和results-round2.json。
注明实际测试机器、系统、Python、torch、unified-marl版本、包路径、完整命令、退出码、日志和产物路径。
与第一轮逐项比较：应用API恢复、CLI恢复、旧0.8.0模型兼容、评估步数、非法恢复拒绝、交通/电网前置条件及实际运行。
没有执行的项目保留未测试；缺资产保留阻塞。第一轮是用户提供的报告证据，不写成你在本轮复现。
只评价软件链路，不宣称收敛、收益、性能或真实业务效果。不要上传模型、数据或服务器凭据。整理完成后停止。
```
