# 第 6 阶段：SADMA ZMQ 执行

日期：2026-10-21（UTC+08:00）；版本 0.6.0。本阶段完成 Windows 本机回环多进程执行与验收，明确保留跨机和其他未通过项并停止；没有进入第 7 阶段。

## 交付与实际结果

将训练/协调、采样、推理、通信和边界验证接入 execution 层，复用阶段 5 的真实 SADMA Policy/Controller、EpisodeBatch、原 episodic ReplayBuffer 和损失函数。没有通过运行旧 runner 或只发送测试消息代替训练集成。每个角色通过明确的协议、版本和数据契约协作，采样进程只运行环境，推理进程加载参数副本，训练端拥有优化器。

- 完整 196 项测试通过，包含前五阶段 149 项与新增 47 项；14 条已有/来源警告不表示仿真器通过。
- 独立 .venv-check 安装 wheel，在源码目录外完成 7 个配置的 ZMQ sync、王版 Q 的 async、显式 V-trace PPO 的 async，共 9 条短训练；各实际完成 2 次更新、非零参数变化、推理/评估及端口重绑。另验证完整状态继续到第 3 次更新。
- 两个推理进程的同版本参数摘要一致；训练更新后所有副本确认新版本；实际采样计数、完整回合与推理调用数一致。两个 managed CLI 入口和独立 train/infer/sample CLI 角色均在本机通过。
- worker 抛异常、worker 停滞、连接超时、推理端端口已占用、context 权限错误均有明确失败路径；正常退出和故障后没有存活的自有 worker，端口可重新绑定。预期故障的 execution.status 为 FAILED，测试 PASS 表示按要求失败和清理，不能解读为该训练成功。
- 推理 snapshot 与阶段 5 来源 Controller 的输出对照误差为 0；行为版本 0 的旧概率在 Learner 版本 2 的 V-trace 更新中保留，与原 PPOPolicy 损失和更新后 actor 对照误差为 0。
- 代表回归 6 项通过：本地 PyMARL QMIX、SADMA PPO、RODE-HMF、HMF MFQ 的 toy 更新，以及 BC/CPL 的真实 3m 数据更新。真实数据更新不等于真实环境策略效果。

当前执行栈为 Windows / Python 3.12.14 / torch 2.6.0+cpu / NumPy 1.26.4 / pyzmq 25.1.2 / libzmq 4.3.4。所有网络测试仅使用 127.0.0.1；没有可用远程机器信息，没有跨机器验证、真实仿真器运行、GPU、吞吐/收敛或策略效果验证。

## 原实现判断与替换边界

只读比较四份 SADMA 的 36 个通信/worker/runner/queue 文件，保留源码哈希与实际类/入口，见 evidence/stage6/audit.json。原角色划分为 TrainWorker、InferWorker、Local/RemoteSampleWorker、InferClient/Server、EvaluateWorker。TrainWorker 使用原 ReplayBuffer 与 Policy；InferWorker 在 actor_version 增加时复制共享 MAC；网络回传主要为动作和 total_steps，旧概率由服务端回合缓存保留，网络帧没有显式 schema/actor 版本。

原 InferServer 的两个线程分别对同一 REP socket 收/发，Queue.get/put 多处无期限，client poll 为 500000 ms，stop 主要靠标志或 kill，socket/context 与阻塞线程没有统一退出协议。新实现保留算法与必要数据语义，替换通信 ABI 和资源管理，不能把新协议声称为旧 InferClient 字节兼容实现。

新层职责：

| 文件 | 实际职责 |
|---|---|
| execution/trainer.py | ROUTER、会话/信用、回合屏障、回放/更新、参数同步、spawn 与清理 |
| execution/inference.py | 独立 REP 进程，加载完整 actor/必要 critic 副本，按版本/摘要确认，调用来源 Controller |
| execution/sampler.py | 独立环境进程，RPC 取动作/行为信息，回传规范化回合，finally 关闭后端与连接 |
| execution/protocol.py | JSON/原始数值帧、有限尺寸、dtype/shape/相关性检查、有限超时和 HWM |
| execution/batches.py | 逐步观察/动作/概率/hidden/奖励/结束位置与完整回合的摘要一致性 |
| execution/settings.py | managed/trainer、mode、地址/端口、worker/设备/seed、容量与期限 |

所有 socket/context 在其进程内创建和关闭；Windows 使用 spawn 上下文，不修改全局 multiprocessing 启动方式，不依赖 fork、shell、pgrep、POSIX 信号脚本或源码中 PodIp/LeaderAddress 环境变量。

## 每种算法的执行支持与覆盖

“允许”指接口实现，“通过”指本次有实际运行证据；两者不混用。

| 调用/变体 | local | ZMQ sync | ZMQ async |
|---|---|---|---|
| sadma/qmix | 阶段 5 通过 | 本次通过 | 本次通过，含两个推理进程 |
| sadma/vdn | 阶段 5 通过 | 本次通过 | 允许，未单列该配置运行 |
| sadma_yi/qmix（原默认 VDN） | 阶段 5 通过 | 本次通过 | 本次通过 |
| sadma_yi_closed/qmix | 阶段 5 通过 | 本次通过 | 允许，未单列该配置运行 |
| sadma/ppo 普通配置 | 阶段 5 通过 | 本次通过 | 明确拒绝，必须显式启用 V-trace |
| sadma_yi/ppo、sadma_yi_closed/ppo 普通配置 | 阶段 5 通过 | 本次通过 | 普通配置明确拒绝 |
| sadma/ppo + use_vtrace=true | 已有本地分支检查 | 允许，未单列此 ZMQ 组合 | 本次通过，并对照旧行为概率/原损失 |
| 易才润 PPO 配置 + use_vtrace=true | 接口保留 | 允许，未逐一运行 | 允许，未逐一运行 |
| 其他 PyMARL、RODE/HMF、寇谦离线/偏好 | 前阶段支持范围保持 | 未实现，明确拒绝 | 未实现，明确拒绝 |
| SADMA MADDPG / 连续 PowerGrid | 原阻塞保留 | 未实现 | 未实现 |

Q/VDN 用来源 episodic ReplayBuffer、Q/TD-lambda、目标网络和原 train 更新。sync PPO 用当前 actor 的新数据，一次 mini_epochs 后消费清空；async PPO 只有 use_vtrace=true 才使用来源回放、旧动作 log-prob 与原 V-trace/clip/ValueNorm 目标。不擅自把普通 PPO 的旧轨迹回放称为有效异步算法。源 PPO 的 GAE 终止 bootstrap、熵自归一化、递归 LR 和 padding 标准化等数学行为继续保留。

当前协调端串行处理 Learner 更新与推理 RPC，采样环境独立运行。async 指去掉跨 worker 回合屏障并保留行为滞后，未实现/验证训练和多请求推理的全面重叠。多推理副本已通过参数同步与路由，但原 infer_batch_size>1 动态聚批、多环境 vector worker、并发吞吐加速没有迁移验收；增加 worker 数不意味着已证明加速。

## 消息、时序、背压和计数

完整契约见 [ZMQ_PROTOCOL.md](ZMQ_PROTOCOL.md)。protocol=1 采用 JSON 与原始 ndarray 帧，无 pickle/send_pyobj/任意网络 torch.load。限定数值类型与有限值、总字节数/帧数/元数据大小；job/schema/request/session/t 对齐。回合张量必须与协商 scheme 的字段、dtype 和尺寸完全一致。

动作、onehot、可用动作、输入/输出 hidden、PPO log-prob/raw critic 值、actor 版本按每个 t 记录；下一次 INFER 同时确认前一真实转移的 reward/terminated/truncated。完整回合再与逐步摘要逐字段核对，错误动作、概率、版本、奖励偏移、终止位置、padding 或字段类型不能进入 Learner。真实个体奖励保持独立，不复制团队值。

每个采样 worker 至多一条在途回合和一个 RPC；消息和回放容量、HWM、worker 数、启动/请求/会话/停止期限均有界。不使用可永久等待的共享队列。一个 worker 挂住时，其他 worker 的活动不能掩盖该会话超时。参数按完整 snapshot 发送；版本与 SHA-256 摘要共同确认，所有推理端同步完成后才继续调度。

env_steps 只统计后续消息已确认执行的真实转移，不含 reset、未执行的动作和单纯推理；episodes 为已校验完整回合，accepted/dropped_stale、completed_steps/partial_confirmed_steps 单列。正常预算结束不再发新许可，并排空在途回合，因此步数可能超出一个在途批次。异常断联前尚未上报的物理环境步无法保证完整计数，不将推理请求数冒充环境步。

max_policy_lag 默认 null，保持来源历史回放；显式设置后才丢弃到达时过旧回合或跳过含过旧数据的回放更新，事件会记录。PPO 原版本每个 mini_epoch 增加一次，Q 每次 train 增加一次，与外层 updates 区分。

## 启动与当前环境边界

安装可选依赖后，在项目根目录示例：

```powershell
python -m pip install ".[distributed]"
unified-marl train --algorithm sadma/qmix --env toy --execution zmq `
  --config examples/zmq_managed.yaml --output-dir ./results/zmq-qmix
```

普通 PPO 同步使用同一 managed 配置与 `--algorithm sadma/ppo`；异步必须显式加 `--set zmq.mode=async --set use_vtrace=true`。Python 的 Windows 示例为 examples/stage6_zmq.py，必须保留 main 保护。role/device/seed/endpoints/deadline 等见配置文件和协议表，自动分配端口可从 execution-started.json 查看。

独立角色按顺序在本机三个终端执行（示例固定端口必须空闲）：

```powershell
unified-marl zmq-infer --endpoint tcp://127.0.0.1:18401 --idle-timeout 60
```

```powershell
unified-marl train --algorithm sadma/qmix --env toy --execution zmq `
  --config examples/zmq_roles.yaml --output-dir ./results/zmq-roles
```

```powershell
unified-marl zmq-sample --trainer-endpoint tcp://127.0.0.1:18400 `
  --worker-id 0 --job-id local-demo --timeout-ms 30000
```

此独立角色链路已在同机回环运行；没有可用远程机器，不能将改地址或多进程作为跨机器通过。非回环配置必须显式 allow_remote=true，并由用户提供实际可达地址与各机环境/资产；协议没有认证/加密，只用于可信部署网络。Linux、GPU、真实 SMAC/补货/CityFlow、电网和网络中断后的自动重连均未验证。

默认 Codex 工具沙箱创建 ZMQ context 报 Permission denied，首次阻塞证据保留；任务授权下通过工具权限提升执行本机回环测试。没有修改防火墙、全局 Python、原始交付、外部集群或远程机器。沙箱阻塞与算法失败分开记录，不声称默认沙箱可运行 TCP。

## 故障、停止和保存加载

正常结束先停止新许可并排空、接收 BYE；停止推理后关闭各 socket/context，再有界 join。异常或通信超时时给在途调用 STOP，采样端 finally 关闭环境；超期仍不响应的自有进程才 terminate/kill 并记 forced。强制终止不能证明外部后端 close 已执行；外部独立角色不能由 Trainer 跨机强杀，停止确认和超时另记录。

本机正常验收任务没有 forced 进程，所有进程 exitcode=0 且端口能重绑。故障测试包括明确的子进程异常、停滞、空连接 250 ms 超时、已占用推理端口和 context 创建异常；失败任务状态为 FAILED，所有自有进程退出，日志保留具体原因。一次 Windows Pipe EOF 的初始清理问题已修复，初始失败日志没有删除。

policy.pt 可通过原 load_policy/evaluate 读取；training.pt 保存原完整 Learner、目标网络/优化器/ValueNorm、版本、回放及已确认计数。恢复会重建 worker 并加载最新快照，本次验证恢复到第 3 次更新；worker RNG/异步调度、外部环境内部状态不保存，不承诺分布式逐步精确复现。

仍未迁移原持续 EvaluateWorker、旧 IPC SharedBuffer 路径、多 Learner/DDP、动态推理批处理及多环境 worker；训练后保存策略的本地 evaluate() 保持通过。这些均单列清单，不把新角色实现算作所有旧 worker 能力完整复刻。

## 实际命令与证据

所有运行使用项目 .venv 或独立 .venv-check；TEMP/TMP 在项目 .work/tmp，PYTHONIOENCODING=utf-8、PYTHONDONTWRITEBYTECODE=1，参考源码导入使用 -B。涉及 TCP 的命令在工具授权提升后执行，仅连接回环地址。

| 实际命令 | 结果与证据 |
|---|---|
| `.venv/Scripts/python.exe -B -m pip install pyzmq==25.1.2 --cache-dir .work/pip-cache` | 只安装项目依赖；开发/验收环境 pip check 另存 dependencies.json |
| `.venv/Scripts/python.exe -B tools/audit_stage6.py` | 36 份原文件与实际角色/队列/格式，audit.json |
| `.venv/Scripts/python.exe -B tools/probe_stage6.py` | 4 条开发短训练路径通过；初次沙箱阻塞/Windows 管道 EOF 失败另存，最终验收以全回归和独立安装为准 |
| `.venv/Scripts/python.exe -B -m pytest tests -q --junitxml=docs/evidence/stage6/tests.xml` | exit 0；196 passed，14 warnings；包括旧 149 项、47 新项、3 组推理/异步来源对照与故障/端口检查 |
| `.venv/Scripts/python.exe -B -m build --no-isolation` | exit 0；0.6.0 wheel/sdist，build.log |
| `.venv-check/Scripts/python.exe -B -m pip install 'dist/unified_marl-0.6.0-py3-none-any.whl[distributed]' --cache-dir .work/pip-cache` | 独立环境安装本地包及 pyzmq；最终本地 wheel 以 --no-index --no-deps --force-reinstall 复装，install.log/install-final.log |
| `.venv/Scripts/python.exe -B -m pip install --no-index --no-deps --no-build-isolation -e . --cache-dir .work/pip-cache` | 同步项目开发环境的可编辑安装元数据为 0.6.0；无全局环境改动，editable-install.log |
| 工作区根运行 `unified-marl/.venv-check/Scripts/python.exe -B unified-marl/tools/validate_stage6_installed.py` | 源码目录外 9 条 ZMQ 路径、恢复、两 CLI、两预期故障、6 个本地回归；installed_validation.json/log |
| `.venv/Scripts/python.exe -B tools/record_stage6.py` | 保存阶段 5 矩阵快照，更新 379 行及 summary.json；新增跨机验证/执行模式列，历史项不追溯伪造 |
| `.venv/Scripts/python.exe -B tools/verify_stage6.py` | 来源哈希、文档、测试、安装/发行文件一致性，final_verification.json |

机器证据还包括 inference-parity-qmix/ppo.json、async-vtrace-source-parity.json、7 份 managed 场景、multi-inference-resume.json、separate-roles.json、fault-raise/stall.json、connection-timeout.json 和 startup-failure.json。错误场景日志保留 FAILED 原因，不能将测试 PASS 解释为故障任务成功。

原 1228 条来源/资源哈希与 36 条本次通信源码记录均复核不变（两者重叠，不相加计唯一数量）。运行代码/配置、wheel 与独立安装文件逐字节一致，参考目录无本次 Python 3.12 bytecode。更新了 WORK_ORDER、规格、计划、状态与矩阵；本阶段完成后停止，等待用户指定第 7 阶段。
