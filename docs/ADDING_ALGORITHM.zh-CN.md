# 新算法接入：一个插件，多个兼容环境

新算法模块注册自己的 controller、learner、默认参数和数据要求。环境由实验 YAML 单独选择，算法模块无需导入 MPE、库存或其他仿真器，也不应在默认参数中绑定环境。

首版插件协议面向离散动作、固定智能体数、团队奖励的在线训练。已有离线训练、连续电网和 ZMQ 执行仍使用各自的接口；仅注册一个名字不会自动获得这些训练能力。

## 1. 先运行可工作的教学插件

```powershell
python -m unified_marl catalog --plugin unified_marl.contrib.huber_iql
python -m unified_marl check examples/experiments/plugin_mpe.yaml --instantiate
python -m unified_marl run examples/experiments/plugin_mpe.yaml
python -m unified_marl run-suite examples/experiments/practical_suite.yaml
```

源码是 `src/unified_marl/contrib/huber_iql.py`。它复用 `BasicMAC` 与 `QLearner`，只将 IQL 的 TD 损失改为带 mask 的 Smooth-L1。这是演示扩展点的教学变体，不是作者原创方法，也不是性能改进结论。

## 2. 独立模块注册

例如创建一个可通过 Python 导入的 `my_lab.huber_iql` 模块。项目根目录直接放包时需有 `my_lab/__init__.py`；跨目录或机器使用时，建议将自己的包安装到同一 Python 环境。不要把组件注册为 `__main__:Class`，新进程恢复 checkpoint 时无法稳定导入。

下面的示例展示独立实现和注册所需的全部结构：

```python
# my_lab/huber_iql.py
import math
import torch
from unified_marl.config import resource
from unified_marl.learners.q_learner import QLearner
from unified_marl.registry import AlgorithmSpec, register_algorithm


class HuberIQLLearner(QLearner):
    def __init__(self, mac, scheme, logger, args):
        self.huber_delta = float(args.huber_delta)
        if not math.isfinite(self.huber_delta) or self.huber_delta <= 0:
            raise ValueError("huber_delta must be finite and positive")
        super().__init__(mac, scheme, logger, args)

    def td_loss(self, td_error, mask):
        loss = torch.nn.functional.smooth_l1_loss(
            td_error, torch.zeros_like(td_error),
            reduction="none", beta=self.huber_delta,
        )
        return (loss * mask).sum() / mask.sum()


register_algorithm(AlgorithmSpec(
    name="my_lab/huber_iql",
    controller="unified_marl.controllers.basic_controller:BasicMAC",
    learner="my_lab.huber_iql:HuberIQLLearner",
    defaults=dict(resource("algs/iql.yaml"),
                  name="huber_iql", learner="huber_iql", huber_delta=1.0),
    training_mode="off_policy",
    required_capabilities=("discrete", "fixed_agents", "team_reward",
                           "global_state", "action_mask"),
    locked_fields={"mixer": None, "mac": "basic_mac", "agent": "rnn",
                   "agent_output_type": "q", "action_selector": "epsilon_greedy"},
    description="Teaching IQL loss variant",
))
```

将实验的 `plugins` 改为 `[my_lab.huber_iql]`，`algorithm` 改为 `my_lab/huber_iql`，即可分别使用 `env: mpe` 或 `env: inventory`。其余训练入口保持一致。

`plugins` 会执行所选模块的 Python 代码，包括注册语句。单纯读取 `load_experiment()` 不导入插件；`check_experiment()`、`run_experiment()` 和底层配置加载会导入明确指定的插件。

## 3. AlgorithmSpec 字段

| 字段 | 要求 |
|---|---|
| `name` | 命名空间/算法名，例如 `my_lab/new_algorithm`；不得占用内置 `pymarl`、`wang`、`sadma`、`kou` 等命名空间 |
| `controller` | 可导入的 `module:Class` |
| `learner` | 可导入的 `module:Class` |
| `defaults` | 算法默认配置字典；合并在全局默认配置之后、环境配置之前 |
| `training_mode` | `off_policy` 使用回放；`on_policy` 直接使用新采集的 rollout |
| `required_capabilities` | 算法要求的环境能力；首版必须包含 `discrete` |
| `locked_fields` | 固定不允许被 YAML 改动的配置值，用于防止组件组合改变算法语义 |
| `scheme_factory` | 可选 `module:function`；输入 `(env_info, config)`，返回额外 batch 字段声明 |
| `description` | catalog 展示说明 |

默认参数不能包含 `env`、`env_args`、`env_config`、`algorithm`、`plugins`。把算法特有的超参数放在 `defaults` 中，实验 YAML 的 `training` 才能识别它；未知训练参数会报错。默认配置本身也应有效，不能依赖用户补齐后才通过最基本的校验。

完整配置顺序为：全局默认值 → 算法默认值 → 环境配置 → 实验 training/显式环境参数 → overrides。算法注册的锁定值和能力检查在合并后仍然有效。

`on_policy` 要求 `batch_size == batch_size_run`。`episode` runner 一次只收集一个环境，因此使用它时两者都应为 `1`；要多回合新鲜 rollout，显式选 `parallel` 并设置相同的 batch 大小。算法仍需正确实现自己的优势计算、行为 log-prob 和优化轮数，注册字段不会自动生成这些逻辑。

## 4. Controller 与 Learner 合约

最小改动算法可以复用 `BasicMAC` 或继承现有 learner。全新实现需遵守以下接口：

| 对象 | 被训练入口调用的接口 |
|---|---|
| Controller | 构造 `Controller(scheme, groups, args)`；`init_hidden(batch_size)`；`select_actions(batch, t_ep, t_env, bs, test_mode)`；`parameters()`；`to(device)` |
| Learner | 构造 `Learner(mac, scheme, logger, args)`；`update(batch, UpdateContext)` 返回有限数值指标字典；`to(device)` |
| UpdateContext | `env_steps`、`episodes`、`update_step`，分别表示采样步、回合数、已完成更新数 |

普通 Controller 返回形状为 `[active_batch, n_agents]` 的整数动作张量，必须遵守动作掩码。`bs` 可能是当前尚未结束环境的索引子集；不应假设并行采集时所有环境总是同时结束。

现有 PyMARL learner 通过 `LearnerAPI.update()` 调用 `train(batch, t_env, episode_num)`。使用该基类可以沿用空 batch 与非有限指标检查；自己实现 `update()` 时需要承担这些校验和相应梯度逻辑。

现有模型保存和推理路径仍要求 Controller 暴露 `agent`，其上具有 PyTorch `state_dict()`、`eval()` 等接口。多网络 Controller 应额外实现自己的 `state_dict()/load_state_dict()`，确保保存完整状态。普通 `load_policy().act()` 为标准当前观测/上一步动作/RNN 协议；需要完整历史或额外外部输入的 controller 应单独验证推理接口，不能只测试训练。

## 5. Batch 与额外字段

`EpisodeBatch` 的核心布局为 `[B, T, N, ...]`，其中 `T` 包含最后一步 bootstrap 观测。标准字段如下：

| 字段 | 典型形状 |
|---|---|
| `state` | `[B, T, state_dim]` |
| `obs` | `[B, T, N, obs_dim]` |
| `actions` | `[B, T, N, 1]`，整数动作 |
| `actions_onehot` | `[B, T, N, n_actions]`，由预处理生成 |
| `avail_actions` | `[B, T, N, n_actions]` |
| `reward` | `[B, T, 1]`，团队奖励 |
| `terminated` / `truncated` / `filled` | `[B, T, 1]` |

真实终止与时间截断分别记录。算法应使用 `filled` 排除填充，并按自己的目标定义正确处理终止与 bootstrap，不应简单把所有 `done` 当作禁止 bootstrap。

需要行为 log-prob、角色或其他采集期信息时，用 `scheme_factory` 增加字段，不能覆盖标准字段：

```python
import torch

def extra_scheme(env_info, config):
    return {
        "behavior_logp": {"vshape": (1,), "group": "agents", "dtype": torch.float32}
    }
```

在注册时设置 `scheme_factory="my_lab.new_algorithm:extra_scheme"`。Controller 可在选动作后通过 `self.rollout_data = {"behavior_logp": tensor}` 提供当步采集数据；张量须对齐本次活动 batch 索引，并且字段须已在 scheme 声明。新增字段本身不会自动获得环境信息或被 learner 使用，生成与消费两端都要实现。

## 6. Checkpoint 与恢复

复用 `QLearner`/`BasicMAC` 时可以使用现有保存逻辑。全新 learner 若包含默认保存器未识别的网络、优化器或计数器，应同时实现：

```python
def checkpoint_state(self):
    # 返回 tensor / 基本类型组成的字典，包含全部训练状态。
    return {"controller": self.mac.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step}

def load_checkpoint_state(self, state):
    self.mac.load_state_dict(state["controller"])
    self.optimizer.load_state_dict(state["optimizer"])
    self.step = state["step"]
```

如果有 target network、归一化统计或额外调度器，也必须保存并恢复它们。checkpoint 使用 `weights_only=True` 读取，所以不要保存任意 Python 实例。

插件模块列表随 config 保存，`evaluate`、`load_policy` 和训练恢复会自动导入。迁移到另一台机器仍需安装同版本插件；checkpoint 不包含插件源码。训练恢复不允许换算法、环境或插件列表。

## 7. 环境与算法独立扩展

新环境可以在同一插件模块中注册，但不要将选择逻辑写进 learner：

```python
from unified_marl.envs import EnvironmentSpec, register_environment

register_environment(EnvironmentSpec(
    name="lab_env",
    factory=MyEnvironment,
    capabilities=("discrete", "fixed_agents", "team_reward", "global_state", "action_mask"),
    defaults={"env_args": {"episode_limit": 20}},
    description="My cooperative discrete environment",
))
```

`factory` 是可调用对象。环境应提供 `reset()`、`step(actions) -> (team_reward, done, info)`、`get_obs()`、`get_state()`、`get_avail_actions()`、`get_env_info()` 和 `close()` 等 `MultiAgentEnv` 接口。metadata 的 `n_agents/n_actions/obs_shape/state_shape/episode_limit` 必须为正整数。时间截断需在 `info` 中设置 `episode_limit` 或 `TimeLimit.truncated`，否则 runner 无法与真实终止区分。

能力声明是合同，环境必须实际提供相应行为；它不能把连续动作环境自动变成离散动作环境。`check --instantiate` 校验构造与 metadata；动作、奖励、重置、终止及数值稳定性仍需测试。

## 8. 最小验证顺序

1. 配置加载与能力不匹配应明确失败，未知超参数不能悄悄忽略。
2. 在小 batch 上验证损失、梯度、动作掩码和终止处理；若迁移已有算法，做相同输入的数值对照。
3. 在 MPE 与库存分别短跑，确认至少发生一次有限参数更新；不同环境分别看结果。
4. 保存后在新进程评估/推理，恢复训练一次，确认插件可重新导入且训练状态完整。
5. 再增加预算与多个训练 seed，使用独立测试 seeds 和适当的独立数据，进行收敛与性能评估。

仓库里的短跑示例完成的是接口和训练链路检查，不代表充分训练、论文结果复现或生产可用性。
