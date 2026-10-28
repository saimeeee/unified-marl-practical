# 已知问题、未通过项与后续工作

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
| M0027 | REGISTRY/q | SOURCE_ONLY_UNVALIDATED | Alternative q role/selector has no delivered active configuration; selector hardcodes 3 roles while default is 2; post-clustering resizing/optimizer schedule unresolved |
| M0029 | REGISTRY/q | SOURCE_ONLY_UNVALIDATED | Alternative q role/selector has no delivered active configuration; selector hardcodes 3 roles while default is 2; post-clustering resizing/optimizer schedule unresolved |
| M0032 | env_REGISTRY/smac | SHARED_ADAPTER_REAL_ENV_BLOCKED | Optional shared SMAC factory implemented; no running SMAC/SC2/map validation; source scheme translated to canonical fields |
| M0035 | env_runner_REGISTRY/smac | SHARED_ADAPTER_REAL_ENV_BLOCKED | Optional shared SMAC factory implemented; no running SMAC/SC2/map validation; source scheme translated to canonical fields |
| M0040 | agent_REGISTRY/ddpg_mlp | BLOCKED_EMPTY_SOURCE_POLICY | All four delivered maddpg_policy.py files are empty and policy_REGISTRY has no MADDPG entry; actor/critic networks alone are not a training algorithm |
| M0041 | agent_REGISTRY/ddpg_rnn | BLOCKED_EMPTY_SOURCE_POLICY | All four delivered maddpg_policy.py files are empty and policy_REGISTRY has no MADDPG entry; actor/critic networks alone are not a training algorithm |
| M0042 | critic_REGISTRY/mlp | BLOCKED_EMPTY_SOURCE_POLICY | All four delivered maddpg_policy.py files are empty and policy_REGISTRY has no MADDPG entry; actor/critic networks alone are not a training algorithm |
| M0043 | critic_REGISTRY/rnn | BLOCKED_EMPTY_SOURCE_POLICY | All four delivered maddpg_policy.py files are empty and policy_REGISTRY has no MADDPG entry; actor/critic networks alone are not a training algorithm |
| M0065 | env_REGISTRY/smac | SHARED_ADAPTER_REAL_ENV_BLOCKED | Optional shared SMAC factory implemented; no running SMAC/SC2/map validation; source scheme translated to canonical fields |
| M0068 | env_REGISTRY/powergrid | BLOCKED_CONTINUOUS_ACTION_SCHEMA_AND_ASSETS | Continuous [B,T,N,D] float actions, Gaussian log-prob semantics and bounds must extend canonical scheme/Policy/Runner; no discretization. pandapower/backend data absent in inspected setup |
| M0069 | env_runner_REGISTRY/smac | SHARED_ADAPTER_REAL_ENV_BLOCKED | Optional shared SMAC factory implemented; no running SMAC/SC2/map validation; source scheme translated to canonical fields |
| M0072 | env_runner_REGISTRY/powergrid | BLOCKED_CONTINUOUS_ACTION_SCHEMA_AND_ASSETS | Continuous [B,T,N,D] float actions, Gaussian log-prob semantics and bounds must extend canonical scheme/Policy/Runner; no discretization. pandapower/backend data absent in inspected setup |
| M0075 | agent_REGISTRY/ddpg_mlp | BLOCKED_EMPTY_SOURCE_POLICY | All four delivered maddpg_policy.py files are empty and policy_REGISTRY has no MADDPG entry; actor/critic networks alone are not a training algorithm |
| M0076 | agent_REGISTRY/ddpg_rnn | BLOCKED_EMPTY_SOURCE_POLICY | All four delivered maddpg_policy.py files are empty and policy_REGISTRY has no MADDPG entry; actor/critic networks alone are not a training algorithm |
| M0077 | critic_REGISTRY/mlp | BLOCKED_EMPTY_SOURCE_POLICY | All four delivered maddpg_policy.py files are empty and policy_REGISTRY has no MADDPG entry; actor/critic networks alone are not a training algorithm |
| M0078 | critic_REGISTRY/rnn | BLOCKED_EMPTY_SOURCE_POLICY | All four delivered maddpg_policy.py files are empty and policy_REGISTRY has no MADDPG entry; actor/critic networks alone are not a training algorithm |
| M0100 | env_REGISTRY/smac | SHARED_ADAPTER_REAL_ENV_BLOCKED | Optional shared SMAC factory implemented; no running SMAC/SC2/map validation; source scheme translated to canonical fields |
| M0103 | env_REGISTRY/powergrid | BLOCKED_CONTINUOUS_ACTION_SCHEMA_AND_ASSETS | Continuous [B,T,N,D] float actions, Gaussian log-prob semantics and bounds must extend canonical scheme/Policy/Runner; no discretization. pandapower/backend data absent in inspected setup |
| M0104 | env_runner_REGISTRY/smac | SHARED_ADAPTER_REAL_ENV_BLOCKED | Optional shared SMAC factory implemented; no running SMAC/SC2/map validation; source scheme translated to canonical fields |
| M0107 | env_runner_REGISTRY/powergrid | BLOCKED_CONTINUOUS_ACTION_SCHEMA_AND_ASSETS | Continuous [B,T,N,D] float actions, Gaussian log-prob semantics and bounds must extend canonical scheme/Policy/Runner; no discretization. pandapower/backend data absent in inspected setup |
| M0110 | agent_REGISTRY/ddpg_mlp | BLOCKED_EMPTY_SOURCE_POLICY | All four delivered maddpg_policy.py files are empty and policy_REGISTRY has no MADDPG entry; actor/critic networks alone are not a training algorithm |
| M0111 | agent_REGISTRY/ddpg_rnn | BLOCKED_EMPTY_SOURCE_POLICY | All four delivered maddpg_policy.py files are empty and policy_REGISTRY has no MADDPG entry; actor/critic networks alone are not a training algorithm |
| M0112 | critic_REGISTRY/mlp | BLOCKED_EMPTY_SOURCE_POLICY | All four delivered maddpg_policy.py files are empty and policy_REGISTRY has no MADDPG entry; actor/critic networks alone are not a training algorithm |
| M0113 | critic_REGISTRY/rnn | BLOCKED_EMPTY_SOURCE_POLICY | All four delivered maddpg_policy.py files are empty and policy_REGISTRY has no MADDPG entry; actor/critic networks alone are not a training algorithm |
| M0135 | env_REGISTRY/smac | SHARED_ADAPTER_REAL_ENV_BLOCKED | Optional shared SMAC factory implemented; no running SMAC/SC2/map validation; source scheme translated to canonical fields |
| M0138 | env_REGISTRY/powergrid | BLOCKED_CONTINUOUS_ACTION_SCHEMA_AND_ASSETS | Continuous [B,T,N,D] float actions, Gaussian log-prob semantics and bounds must extend canonical scheme/Policy/Runner; no discretization. pandapower/backend data absent in inspected setup |
| M0139 | env_runner_REGISTRY/smac | SHARED_ADAPTER_REAL_ENV_BLOCKED | Optional shared SMAC factory implemented; no running SMAC/SC2/map validation; source scheme translated to canonical fields |
| M0142 | env_runner_REGISTRY/powergrid | BLOCKED_CONTINUOUS_ACTION_SCHEMA_AND_ASSETS | Continuous [B,T,N,D] float actions, Gaussian log-prob semantics and bounds must extend canonical scheme/Policy/Runner; no discretization. pandapower/backend data absent in inspected setup |
| M0145 | agent_REGISTRY/ddpg_mlp | BLOCKED_EMPTY_SOURCE_POLICY | All four delivered maddpg_policy.py files are empty and policy_REGISTRY has no MADDPG entry; actor/critic networks alone are not a training algorithm |
| M0146 | agent_REGISTRY/ddpg_rnn | BLOCKED_EMPTY_SOURCE_POLICY | All four delivered maddpg_policy.py files are empty and policy_REGISTRY has no MADDPG entry; actor/critic networks alone are not a training algorithm |
| M0147 | critic_REGISTRY/mlp | BLOCKED_EMPTY_SOURCE_POLICY | All four delivered maddpg_policy.py files are empty and policy_REGISTRY has no MADDPG entry; actor/critic networks alone are not a training algorithm |
| M0148 | critic_REGISTRY/rnn | BLOCKED_EMPTY_SOURCE_POLICY | All four delivered maddpg_policy.py files are empty and policy_REGISTRY has no MADDPG entry; actor/critic networks alone are not a training algorithm |
| M0174 | REGISTRY/maddpg_mac | SOURCE_ONLY_COMPONENT_UNVALIDATED | No verified delivered discrete offline bundle uses this component; GRNN additionally needs original SMAC feature metadata/communication settings |
| M0177 | REGISTRY/q_learner_new | BLOCKED_SOURCE_ASSETS_OR_PROTOCOL | Missing actual reference policy, individual termination and/or agent-state reward protocol; source presence alone not counted integrated |
| M0178 | REGISTRY/qdiffer_learner | BLOCKED_SOURCE_ASSETS_OR_PROTOCOL | Missing actual reference policy, individual termination and/or agent-state reward protocol; source presence alone not counted integrated |
| M0179 | REGISTRY/pref_qmix_learner | BLOCKED_SOURCE_ASSETS_OR_PROTOCOL | Missing actual reference policy, individual termination and/or agent-state reward protocol; source presence alone not counted integrated |
| M0185 | REGISTRY/pref_qmix_learner_new | BLOCKED_SOURCE_ASSETS_OR_PROTOCOL | Missing actual reference policy, individual termination and/or agent-state reward protocol; source presence alone not counted integrated |
| M0190 | REGISTRY/coma_critic | SOURCE_ONLY_COMPONENT_UNVALIDATED | No verified delivered discrete offline bundle uses this component; GRNN additionally needs original SMAC feature metadata/communication settings |
| M0191 | REGISTRY/maddpg_critic_ns | SOURCE_ONLY_COMPONENT_UNVALIDATED | No verified delivered discrete offline bundle uses this component; GRNN additionally needs original SMAC feature metadata/communication settings |
| M0192 | REGISTRY/episode | NOT_MIGRATED_SOURCE_ONLINE_EXECUTION | Stage 4 uses an independent offline update trainer; historical online preference/collection runner lifecycle and its extra fields are not migrated |
| M0193 | REGISTRY/parallel | NOT_MIGRATED_SOURCE_ONLINE_EXECUTION | Stage 4 uses an independent offline update trainer; historical online preference/collection runner lifecycle and its extra fields are not migrated |
| M0194 | REGISTRY/pref_episode | NOT_MIGRATED_SOURCE_ONLINE_EXECUTION | Stage 4 uses an independent offline update trainer; historical online preference/collection runner lifecycle and its extra fields are not migrated |
| M0195 | REGISTRY/pref_episode_new | NOT_MIGRATED_SOURCE_ONLINE_EXECUTION | Stage 4 uses an independent offline update trainer; historical online preference/collection runner lifecycle and its extra fields are not migrated |
| M0196 | REGISTRY/makedata_runner | NOT_MIGRATED_SOURCE_ONLINE_EXECUTION | Stage 4 uses an independent offline update trainer; historical online preference/collection runner lifecycle and its extra fields are not migrated |
| M0199 | env_REGISTRY/manyagent_swimmer | OUTSIDE_REWARD_IMPORT_CLOSURE | Embedded MuJoCo/swimmer environment is not a reward-model dependency and is not integrated by this stage |
| M0200 | env_REGISTRY/manyagent_swimmer | OUTSIDE_REWARD_IMPORT_CLOSURE | Embedded MuJoCo/swimmer environment is not a reward-model dependency and is not integrated by this stage |
| M0205 | REGISTRY/q_learner | BLOCKED_INCOMPATIBLE_SOURCE_CONTROLLER | BCQ-like triple-output contract mismatches all supplied active basic-controller configurations |
| M0216 | REGISTRY/grnn | SOURCE_ONLY_COMPONENT_UNVALIDATED | No verified delivered discrete offline bundle uses this component; GRNN additionally needs original SMAC feature metadata/communication settings |
| M0219 | REGISTRY/episode | NOT_MIGRATED_SOURCE_ONLINE_EXECUTION | Stage 4 uses an independent offline update trainer; historical online preference/collection runner lifecycle and its extra fields are not migrated |
| M0220 | REGISTRY/parallel | NOT_MIGRATED_SOURCE_ONLINE_EXECUTION | Stage 4 uses an independent offline update trainer; historical online preference/collection runner lifecycle and its extra fields are not migrated |
| M0251 | pref_qmix | BLOCKED_SOURCE_ASSETS_OR_PROTOCOL | Requires actual reference policy checkpoint and per-agent termination/labels |
| M0252 | pref_qmix_new | BLOCKED_SOURCE_ASSETS_OR_PROTOCOL | Requires actual reference policy checkpoint, agent_state and compatible new local reward model |
| M0254 | qmix_differ | BLOCKED_SOURCE_ASSETS_OR_PROTOCOL | Requires original per-agent indi_terminated; dataset only supplies team terminated |
| M0255 | qmix_make_datasets | BLOCKED_SOURCE_ASSETS_OR_PROTOCOL | Online source data-collection runner not migrated; real SMAC and author source .npy assets unavailable |
| M0256 | qmix_new | BLOCKED_SOURCE_ASSETS_OR_PROTOCOL | Requires the actual reference policy checkpoint, per-agent termination and reward-model integration |
| M0260 | acpl_smac | BLOCKED_INCOMPLETE_SOURCE_CONFIG | ACPL config lacks NLL_lamda/use_nll etc and selects Q outputs while bound CPL expects policy probabilities; original dispatcher has no ACPL branch |
| M0266 | hacpl_smac | BLOCKED_MISSING_SOURCE_LEARNER | Configuration references absent/unregistered hacpl_learner |
| M0270 | maicpl_smac | BLOCKED_MISSING_SOURCE_LEARNER | Configuration references absent/unregistered maicpl_learner |
| M0271 | maidpo_smac | BLOCKED_MISSING_SOURCE_LEARNER | Configuration references absent/unregistered maidpo_learner |
| M0273 | qmix | BLOCKED_INCOMPATIBLE_SOURCE_CONTROLLER | Delivered QLearner expects Q/imitator/auxiliary triple; registered BasicMAC returns one tensor. No matching imitation agent/controller supplied; not replaced by PyMARL QMIX |
| M0274 | qmix_smac | BLOCKED_INCOMPATIBLE_SOURCE_CONTROLLER | Delivered QLearner expects Q/imitator/auxiliary triple; registered BasicMAC returns one tensor. No matching imitation agent/controller supplied; not replaced by PyMARL QMIX |
| M0275 | vdn_smac | BLOCKED_INCOMPATIBLE_SOURCE_CONTROLLER | Delivered QLearner expects Q/imitator/auxiliary triple; registered BasicMAC returns one tensor. No matching imitation agent/controller supplied; not replaced by PyMARL QMIX |
| M0278 | Hierarchical_Qmix | BLOCKED_SOURCE_IMPLEMENTATION | QMixNet.__init__() got an unexpected keyword argument 'input_dim'; no confirmed full training schedule; source namespaced only, not counted migrated |
| M0284 | Hierarchical_VDN | BLOCKED_SOURCE_IMPLEMENTATION | running_mean should contain 6 elements not 5; no confirmed full training schedule; source namespaced only, not counted migrated |
| M0286 | Hierarchical_VDN_ROMA | BLOCKED_SOURCE_IMPLEMENTATION | cannot access local variable 'NN_HIDDEN_SIZE' where it is not associated with a value; no confirmed full training schedule; source namespaced only, not counted migrated |
| M0289 | MFAC | BLOCKED_SOURCE_IMPLEMENTATION | MFAC.init_from_env() missing 1 required positional argument: 'num_evaders'; no confirmed full training schedule; source namespaced only, not counted migrated; deeper source inspection: update needs next_state/rews/log_probs/values/entropy/done, calls missing Agent.critic/eval_optimizer and misspelled adv.detacj(); supplying num_evaders alone cannot restore the actor-critic protocol |
| M0294 | MADDPG source without policy registration | BLOCKED_EMPTY_SOURCE_POLICY | All four delivered maddpg_policy.py files are empty and policy_REGISTRY has no MADDPG entry; actor/critic networks alone are not a training algorithm |
| M0295 | MADDPG source without policy registration | BLOCKED_EMPTY_SOURCE_POLICY | All four delivered maddpg_policy.py files are empty and policy_REGISTRY has no MADDPG entry; actor/critic networks alone are not a training algorithm |
| M0296 | MADDPG source without policy registration | BLOCKED_EMPTY_SOURCE_POLICY | All four delivered maddpg_policy.py files are empty and policy_REGISTRY has no MADDPG entry; actor/critic networks alone are not a training algorithm |
| M0297 | MADDPG source without policy registration | BLOCKED_EMPTY_SOURCE_POLICY | All four delivered maddpg_policy.py files are empty and policy_REGISTRY has no MADDPG entry; actor/critic networks alone are not a training algorithm |
| M0301 | active preference training entry | BLOCKED_REFERENCE_POLICY_OR_AGENT_FIELDS | Active old_src workflow preserved in audit; actual reference policy checkpoint plus agent_state/indi_terminated and original local-label protocol are missing |
| M0302 | alternate preference implementation (not current main import) | BLOCKED_REFERENCE_POLICY_OR_AGENT_FIELDS | Active old_src workflow preserved in audit; actual reference policy checkpoint plus agent_state/indi_terminated and original local-label protocol are missing |
| M0306 | reward model variant | BLOCKED_REFERENCE_POLICY_OR_AGENT_FIELDS | Active old_src workflow preserved in audit; actual reference policy checkpoint plus agent_state/indi_terminated and original local-label protocol are missing |
| M0307 | preference construction | BLOCKED_REFERENCE_POLICY_OR_AGENT_FIELDS | Active old_src workflow preserved in audit; actual reference policy checkpoint plus agent_state/indi_terminated and original local-label protocol are missing |
| M0318 | use_llm | EXTERNAL_SERVICE_NOT_EXECUTED | LLM client is an external-service boundary; no new-label generation/API calls requested or executed; existing labels are read explicitly |
| M0337 | TCJob operator (Go 1.19) | NOT_STARTED | Outside core Python package; no service/cluster/database baseline verified |
| M0338 | TCJob operator + scheduling extensions | NOT_STARTED | Outside core Python package; no service/cluster/database baseline verified |
| M0339 | FastAPI task/code/tensorboard backend | NOT_STARTED | Outside core Python package; no service/cluster/database baseline verified |
| M0340 | Vue3/ElementPlus/Vite frontend | NOT_STARTED | Outside core Python package; no service/cluster/database baseline verified |
| M0349 | dynamic IPC/K-means grouping collection and high-level scheduling | NOT_MIGRATED | Source dynamic-group sample includes agents_group/next_agents_group; fixed-group registration does not implement this collection/update schedule |
| M0350 | post-clustering role expansion training | SOURCE_LIMITATION | New role parameters missing from source optimizer; RODE-HMF n_groups fixed; standalone clustering parity does not validate post-expansion updates |
| M0367 | Continuous action schema extension design | BLOCKED_CONTINUOUS_ACTION_IMPLEMENTATION | Design and explicit rejection only; no Gaussian float-action Runner/Policy/checkpoint integration or real grid assets validation |

## 部分分支通过，仍有欠项

| ID | 能力 | 状态 | 保留原因 |
|---|---|---|---|
| M0021 | REGISTRY/rode_learner | PARTIAL_PRE_CLUSTERING_VALIDATED | CPU deterministic toy only; real SMAC/ReplenishmentEnv/data and CUDA training not validated; dynamic role expansion training/resume not validated; source optimizer omissions and grouping indices preserved (STAGE3_REPORT.md) |
| M0022 | REGISTRY/rode_hmf_learner | PARTIAL_PRE_CLUSTERING_VALIDATED | CPU deterministic toy only; real SMAC/ReplenishmentEnv/data and CUDA training not validated; dynamic role expansion training/resume not validated; source optimizer omissions and grouping indices preserved (STAGE3_REPORT.md) |
| M0062 | worker_REGISTRY/evaluate_worker | PARTIAL_PERIODIC_EVALUATION_REPLACEMENT_VERIFIED | Bounded spawn policy-snapshot evaluator integrated and tested locally, with update-based trigger, one pending task, timeout/error propagation and cleanup. Original perpetual EvaluateWorker/shared-buffer trigger, vector evaluation and remote evaluation not reproduced; real simulator unavailable. |
| M0097 | worker_REGISTRY/evaluate_worker | PARTIAL_PERIODIC_EVALUATION_REPLACEMENT_VERIFIED | Bounded spawn policy-snapshot evaluator integrated and tested locally, with update-based trigger, one pending task, timeout/error propagation and cleanup. Original perpetual EvaluateWorker/shared-buffer trigger, vector evaluation and remote evaluation not reproduced; real simulator unavailable. |
| M0132 | worker_REGISTRY/evaluate_worker | PARTIAL_PERIODIC_EVALUATION_REPLACEMENT_VERIFIED | Bounded spawn policy-snapshot evaluator integrated and tested locally, with update-based trigger, one pending task, timeout/error propagation and cleanup. Original perpetual EvaluateWorker/shared-buffer trigger, vector evaluation and remote evaluation not reproduced; real simulator unavailable. |
| M0167 | worker_REGISTRY/evaluate_worker | PARTIAL_PERIODIC_EVALUATION_REPLACEMENT_VERIFIED | Bounded spawn policy-snapshot evaluator integrated and tested locally, with update-based trigger, one pending task, timeout/error propagation and cleanup. Original perpetual EvaluateWorker/shared-buffer trigger, vector evaluation and remote evaluation not reproduced; real simulator unavailable. |
| M0184 | REGISTRY/max_q_learner_w_globalRM | PARTIAL_BASE_TD_METHOD_VERIFIED | CPU short updates only; real SMAC evaluation blocked; no policy effectiveness/convergence claim; source online reward/unsupervised methods remain unconnected |
| M0235 | rode | PARTIAL_PRE_CLUSTERING_VALIDATED | CPU deterministic toy only; real SMAC/ReplenishmentEnv/data and CUDA training not validated; dynamic role expansion training/resume not validated; source optimizer omissions and grouping indices preserved (STAGE3_REPORT.md) |
| M0236 | rode_hmf | PARTIAL_PRE_CLUSTERING_VALIDATED | CPU deterministic toy only; real SMAC/ReplenishmentEnv/data and CUDA training not validated; dynamic role expansion training/resume not validated; source optimizer omissions and grouping indices preserved (STAGE3_REPORT.md) |
| M0248 | cw_qmix_w_globalRM | PARTIAL_BASE_TD_METHOD_VERIFIED | CPU short updates only; real SMAC evaluation blocked; no policy effectiveness/convergence claim; source train() uses original team reward; full online unsupervised/relabel/train_reward schedule is not migrated |
| M0250 | ow_qmix_w_globalRM | PARTIAL_BASE_TD_METHOD_VERIFIED | CPU short updates only; real SMAC evaluation blocked; no policy effectiveness/convergence claim; source train() uses original team reward; full online unsupervised/relabel/train_reward schedule is not migrated |
| M0279 | Hierarchical_VDN | PARTIAL_FIXED_CRITIC_UPDATE_VERIFIED | Only fixed-group update() and existing local/mf critic targets integrated as wang_hmf/hierarchical_vdac_fixed; source Agent has no mf_policy/target_mf_policy, while update_high_level/prep_* reference them. Full VDAC/actor and alternate schedule remain unimplemented; no real replenishment backend/assets. |
| M0280 | Hierarchical_VDN | PARTIAL_FIXED_GROUP_VALIDATED | CPU deterministic toy only; real SMAC/ReplenishmentEnv/data and CUDA training not validated; default fixed-group update only; update_dynamic_group and clustering schedule not migrated |
| M0285 | Hierarchical_VDN_K_Means | PARTIAL_FIXED_GROUP_VALIDATED | CPU deterministic toy only; real SMAC/ReplenishmentEnv/data and CUDA training not validated; default fixed-group update only; update_dynamic_group and clustering schedule not migrated |
| M0300 | run/globalRM/pref/dataset dispatch | PARTIAL_OFFLINE_COMPONENTS_ONLY | Offline model/pairing components are integrated; original interactive feedback, unsupervised reset, online collection and hand-filtered .npy pipeline are not executed |
| M0303 | global reward model training | PARTIAL_OFFLINE_COMPONENTS_ONLY | Offline model/pairing components are integrated; original interactive feedback, unsupervised reset, online collection and hand-filtered .npy pipeline are not executed |
| M0304 | offline dataset generation | PARTIAL_OFFLINE_COMPONENTS_ONLY | Offline model/pairing components are integrated; original interactive feedback, unsupervised reset, online collection and hand-filtered .npy pipeline are not executed |
| M0305 | reward model | PARTIAL_GLOBAL_MODELS_VERIFIED | CPU short updates only; real SMAC evaluation blocked; no policy effectiveness/convergence claim; local/individual reward training and policy-oracle label generation require missing masks/reference assets |
| M0308 | offline/preference dispatch and evaluation | PARTIAL_OFFLINE_DISPATCH_REPLACED | CPU short updates only; real SMAC evaluation blocked; no policy effectiveness/convergence claim; e2e, predictor calibration, behavior pretraining/moderate CQL and PER are explicitly rejected rather than silently ignored |
| M0309 | MR/NMR/PrefTransformer/MultiPrefTransformerDivide | PARTIAL_REWARD_MODELS_AND_DIVERSE_CURRICULUM | CPU short updates only; real SMAC evaluation blocked; no policy effectiveness/convergence claim; source NMR reuses left prediction; PrefTransformer unregistered blocks/optimizer omission preserved; calibration train_predicter/e2e not integrated; diverse filter can select zero pairs |
| M0310 | embedded MAPT library incl unrelated environment assets | PARTIAL_REWARD_IMPORT_CLOSURE_ONLY | Only actual reward-model Python dependency closure is bundled; embedded simulators, other MAPT policy algorithms and unused JAX helpers are not integrated |
| M0311 | shared/episode/replay buffers | PARTIAL_LOCAL_BUFFER_AND_SCHEMA_VERIFIED | CPU toy short updates only; no real simulator, GPU or policy effectiveness validation; source PPO GAE/entropy/LR quirks retained; native ReplayBuffer and arrival-time EpisodeBuffer equivalence verified; IPC SharedBuffer and distributed transition framing not executed |
| M0312 | transition/episode/buffer schemes | PARTIAL_LOCAL_BUFFER_AND_SCHEMA_VERIFIED | CPU toy short updates only; no real simulator, GPU or policy effectiveness validation; source PPO GAE/entropy/LR quirks retained; native ReplayBuffer and arrival-time EpisodeBuffer equivalence verified; IPC SharedBuffer and distributed transition framing not executed |
| M0313 | ZMQ workers and parameter exchange | PARTIAL_ZMQ_ROLES_LOOPBACK_VERIFIED | Windows loopback TCP only; cross-machine, real simulators, GPU and policy effects NOT verified; new protocol=1 replaces original binary ABI; persistent evaluator and original shared-memory queue path remain unintegrated |
| M0317 | make_diverse_preference | PARTIAL_PURE_SEGMENT_HELPER | Only pure clip_to_segment_length is migrated and source-compared; LLM label generation/retries and automatic file rewriting are not executed |
| M0359 | old_global | PARTIAL_GLOBAL_ONLY_VERIFIED | CPU short updates only; real SMAC evaluation blocked; no policy effectiveness/convergence claim; individual/local reward methods not migrated |
| M0360 | old_reward_global | PARTIAL_GLOBAL_ONLY_VERIFIED | CPU short updates only; real SMAC evaluation blocked; no policy effectiveness/convergence claim; individual/local reward methods not migrated |
| M0361 | MultiPrefTransformerDivide/diverse | PARTIAL_INITIAL_EPOCH_VERIFIED | Original global/between/in losses and KL curriculum integrated; first real epoch update passed; next synthetic epoch selects zero pairs and explicitly blocks |

## 0.8.0 工程版补充

新增应用代码按用户要求未跑训练与场景实验：交通仍需用户安装CityFlow并提供路网/车流，电网需安装pandapower等并提供建筑/天气/设备资产；仓储新增库存环境与原ReplenishmentEnv语义不同，公式见WAREHOUSE_GUIDE。连续动作由新的bounded_ppo实现，没有把原SADMA连续分支标成已验收。旧来源阻塞与动态调度缺项继续保留，新增接口不保证所有原算法组合可用。

## 0.8.1 Linux 报告后续

应用恢复时自动seed误判已修复，仅12项无仿真回归通过；完整训练恢复等待用户二测。0.8.0仓储合成链路通过来自用户报告，不是本机复现。交通缺CityFlow/路网、电网缺气候建筑资产继续阻塞。小型补丁不捆绑这些资产。原工程安装器加装功能会降至0.8.0，之后必须再应用0.8.1补丁。

## 0.8.2待用户三测

F1/F2/N1代码修复与无仿真单元回归通过；真实Linux结果未取得。0.8.1旧电网checkpoint恢复/环境评估因回合语义改变明确拒绝，启用净用电量状态也要求新模型。二轮替代数据的其他关闭状态、建筑排除和动作范围限制继续保留。需要完整执行0.8.2补丁安装器才更新旧安装入口，仅装wheel不能解决N1。

## 第三轮后剩余项

报告范围内F1/F2和恢复链路通过。N1重复已齐依赖通过，另有合成探针验证缺失本地wheel分支；真实Linux首次补齐PowerGrid依赖仍未验。reset跨进程差异尚未定位，提供check_powergrid_seed.py进行完整初始化对照；不改训练行为伪装修复。尚未取得用户场景准备脚本，发行不声称自带可重建其完整公开数据配置。
