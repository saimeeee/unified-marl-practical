# SADMA ZMQ 协议与执行约定

版本：unified-marl 0.6.0；wire protocol=1；日期：2026-10-21。

## 职责与拓扑

execution/trainer.py 是单一 Learner/协调端，拥有优化器、目标网络、回放及计数；sampler.py 是只运行环境的独立采样进程；inference.py 是拥有显式参数副本的独立推理进程；protocol.py 负责序列化、超时、socket 配置；batches.py 负责训练边界检查。算法本体复用阶段 5 已验证的 SADMA 类。

Trainer 使用单线程 ROUTER；采样端使用 DEALER，一次仅一个请求在途；推理端使用单线程 REP，Trainer 通过独立 REQ 连接发送参数与推理消息。socket/context 均在所属进程内创建和关闭，不跨线程发送/接收同一 socket。Windows 使用 get_context("spawn")，不改变全局 start method，不执行旧 shell/kill 脚本。

managed 模式由 Python API 启动和管理自有推理/采样子进程；trainer 模式连接用户另行启动的推理和采样角色。两种拓扑均已在同一台 Windows 主机的回环地址运行；独立角色启动不等于跨机器验证。未提供可用远程机器，没有执行跨机实验。

## 配置

使用 `train(..., execution="local")` 进入阶段 5 本地同步路径；`execution="zmq"` 使用本节配置。`config` YAML 或 `overrides` 的 zmq.* 提供配置；最终值写入 config.json 和 execution-started.json。

| 字段 | 默认/含义 |
|---|---|
| role | managed；或 trainer（只启动协调/Learner，外部角色自行启动） |
| mode | sync；或 async，必须与算法支持方式匹配 |
| bind_endpoint | tcp://127.0.0.1:0；0 表示自动绑定空闲端口 |
| trainer_endpoint | 默认使用实际绑定地址；显式填写可供采样端访问的地址 |
| inference_endpoints | managed 默认自动分配；trainer 模式须提供固定 endpoint 列表 |
| num_sampling_workers / num_inference_workers | 2 / 1；本阶段一个训练端，不实现多个 Learner/DDP |
| train device / inference_device / sampler_device | 训练设备由 train(device=...)；推理默认 cpu；采样传输固定 cpu |
| seed / seed_stride | train seed；stride=1000，采样 seed=base+(id+1)*stride，推理 seed=base+(100+id)*stride |
| timeout_ms | 10000，请求/回复和采样会话进度的有限超时 |
| startup_timeout_s / shutdown_timeout_s | 45 / 5，初始化与排空/停止期限 |
| hwm | 2，ZMQ 发送/接收高水位；每个连接每次一个 RPC |
| max_message_bytes | 64 MiB，编码和解码总字节上限；dtype/shape/帧长单独校验 |
| max_policy_lag | null，不自动丢弃历史行为；显式非负值会丢弃到达时过旧回合，并跳过含过旧样本的回放更新 |
| job_id | 默认随机；独立角色必须使用同一显式 job_id |
| allow_remote | false，只允许回环配置；设 true 才允许非回环 IPv4/hostname endpoint |

worker 数、队列、高水位、消息大小和所有等待期限有明确上界；不支持 IPC/Unix domain socket、fork 或无限 timeout。fault_worker/fault_after_infers/fault_kind 仅用于可重复的诊断故障注入，默认关闭。无认证/加密，协议用于可信部署网络；本阶段所有实际测试均为回环地址。

## 消息与帧

每条消息为 multipart：第 0 帧是 UTF-8 JSON，包含 protocol=1 与 message；后续帧是原始数值数组。数组描述符包含唯一 frame index、dtype 和 shape。仅接受 float32/float64/int32/int64/uint8/bool；拒绝 object、任意 Python 对象、NaN/Inf、缺帧、重复帧、错误字节数、超限 shape、额外字段及不兼容协议。不使用 pickle、send_pyobj 或 torch.load 接收网络数据。

采样消息含 job_id、schema_id、worker_id、递增 request_id；响应回显 request_id。schema_id 对算法架构字段与环境维度求摘要。训练端拒绝未知/重复 worker、回退/跳号请求、错误会话及不一致的环境元数据。独立进程必须运行相同协议版本与兼容模型。

| 消息 | 必须信息与用途 |
|---|---|
| HELLO → WELCOME | 注册 worker；返回规范化配置、env_info、schema_id |
| ENV_READY | 采样后端实际 env_info、派生 seed；与 Trainer 核对 |
| READY → COLLECT | 请求回合许可，分配唯一 session；每个 worker 至多一个在途回合 |
| INFER → ACTIONS | session、严格递增 t、bootstrap 标志、当前观测/状态/可用动作/前一动作/输入 hidden，以及上一条真实转移 |
| EPISODE → EPISODE_ACK | 完整裁切后的 EpisodeBatch、实际转移长度；与逐步推理和转移摘要核对后才能进入训练 |
| BYE → BYE_ACK | 后端关闭状态与错误；正常回收采样会话 |
| INIT/SYNC → SYNCED | 模型参数、actor 版本、SHA-256 摘要；推理端加载并精确确认 |
| STOP → STOPPED | 停止推理，返回推理次数和有界同步历史 |
| ERROR / STOP | 明确错误或正常停止指令，采样端释放环境和连接 |

新协议替代原 SADMA 的 native NumPy structured record ABI，不与原 InferClient 字节格式兼容。source audit 保留原格式、调用和版本差异；不能将替换后的网络协议称为原脚本无改动通过。

## 时间位置与数据形状

固定 agent 数 N、离散动作数 A、特征维 O/S、隐藏维 H。采样按完整回合传输 [1,T+1,...]，T 个真实转移与一个同回合 bootstrap 槽；传输不带 padding，进入固定回放时补齐并保留 filled。不同长度的 PPO 回合在合批时 padding。

- obs [1,T+1,N,O]、state [1,T+1,S]、avail_actions [1,T+1,N,A]。
- actions int64 [1,T+1,N,1]、actions_onehot float32 [1,T+1,N,A]，必须彼此和动作掩码一致。
- reward float32 [1,T+1,1]，terminated/truncated uint8 同形；转移 t 的奖励和结束标记写在 t，bootstrap 槽奖励/结束标记为零。
- hidden_states float32 [1,T+1,N,H] 是动作前向输出，下步输入必须与上次输出一致；回合开始为零。
- policy_version int64 [1,T+1,1]，记录每次真实动作前向所用 actor 版本，不用训练更新数代替。
- PPO action_log_probs、behavior_values 为 float32 [1,T+1,N,1]；按原分布保存动作概率和 raw critic 值，回传后不重算为新策略概率。
- 显式个体奖励为 float32 [1,T+1,N,1]，只接受实际环境返回值，保持与团队奖励独立。

下一次 INFER(t+1) 同时确认 reward_t/terminated_t/truncated_t，终止标记必须与 bootstrap 一致；完整回合逐字段摘要再与这些确认信息比较。因此没有旧到达式 ts-1 的二次偏移。观测、动作、行为概率、hidden、策略版本和奖励时间位置任何不一致均拒绝进入训练。

## 同步与异步语义

sync 使用回合屏障，每个参与 worker 一轮一个完整回合；只有该轮全部完成后才能更新模型，所有推理副本确认新 snapshot 后才允许下一轮。Q 使用原 episodic ReplayBuffer；普通 PPO 只使用当前策略新数据、mini_epochs 后消费清空，保持阶段 5 的同策略语义。

async 不设跨 worker 回合屏障，Q 继续按原回放、update_rate 与目标网络逻辑更新，行为版本可在回合内变化。PPO async 必须显式 use_vtrace=true，使用来源的 ReplayBuffer、旧 action_log_probs 和原 vtrace/clipping/ValueNorm 损失；不能把普通 PPO 自动当作异步算法。源 Policy 在更新前重算当前 critic 作为 old_values 的行为保留，传回的 behavior_values 用于可追踪性，不替换原目标。

当前协调端串行处理 Learner 更新和推理 RPC，环境采样在独立进程运行；async 表示没有跨 worker 回合屏障并保留滞后行为回放，不承诺训练与推理完全重叠。多推理进程已经完成参数同步和请求路由验证，但未实现多请求并行派发、原 infer_batch_size>1 的动态聚批或吞吐加速验收。每个采样 worker 一个环境，多环境 vector worker 尚未接入。

PPO actor_version 每个原 mini_epoch 增加一次，Q 每次原 train 增加一次。参数同步包含 actor 与必要 critic，不共享仍在写入的网络；所有推理副本对版本和参数摘要确认。摘要与近期同步记录有界保存，完整事件追加到 execution-events.jsonl。

## 计数、故障、停止和恢复

env_steps 统计下一条消息已确认执行的真实转移，不含 reset/未执行动作/单纯推理；episodes 是已校验完整回合，episodes_accepted/dropped_stale 分开。completed_steps 与 partial_confirmed_steps 分开，异常断联时尚未上报的物理环境步无法保证统计完整。正常预算结束停止发放新许可，排空在途回合，故实际计数可能超过目标最多一批在途回合；排空期间不再更新模型。

每个采样会话有独立进度超时，其他 worker 持续发消息不能掩盖一个挂住的 worker。启动失败、worker 异常/退出、数据错误、通信超时会中止任务并写 FAILED execution.json。先给在途请求 STOP、关闭环境/端口，再有界 join；仍不响应的自有进程 terminate/kill 并明确记录 forced，不能声称这类强制退出一定调用了外部后端 close。外部独立角色不能由 Trainer 跨机终止，其关闭确认与超时另记录。

训练 checkpoint 保留原完整 Learner、目标/优化器/ValueNorm、策略版本、回放和已确认计数；恢复启动新的采样/推理进程并同步最新参数。worker RNG 和异步调度会重启，外部环境内部状态不保存，因此不承诺分布式逐步精确复现。

支持范围及实际测试见 STAGE6_REPORT.md。PyMARL/RODE/HMF/寇谦离线算法没有被自动接到 ZMQ；连续动作电网、原持续 EvaluateWorker、多 Learner、跨机效果和 GPU 仍未通过本阶段验收。
