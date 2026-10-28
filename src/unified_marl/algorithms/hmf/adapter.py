"""EpisodeBatch bridge to the delivered fixed-group transition algorithms.

Each source update and target-update function is called directly from migrated
package modules. No original entry script or external source path is used.
"""
from contextlib import redirect_stdout
import importlib
import inspect
import io
from types import SimpleNamespace
import numpy as np
import torch
from unified_marl.learners.base import LearnerAPI

IMPLEMENTATIONS = {
    "dqn": ("dqn", "DQN"), "vdn": ("vdn", "VDN"), "qmix": ("qmix", "QMIX"), "mfq": ("mfq", "MFQ"),
    "mfvfd": ("mfvfd", "MFVFD"), "hierarchical": ("hierarchical", "Hierarchical"),
    "hierarchical_vdn": ("hierarchical_vdn", "Hierarchical_VDN"),
    "hierarchical_vdn_IPC": ("hierarchical_vdn_02_formulate_meanQ_IPC", "Hierarchical_VDN_02_Formulate_MeanQ_IPC"),
    "hierarchical_vdn_without_high_level": ("hierarchical_vdn_without_high_level", "Hierarchical_VDN_Without_High_Level"),
    "maddpg": ("maddpg", "MADDPG"),
    "hierarchical_vdn_copy_fixed": ("hierarchical_vdn copy", "Hierarchical_VDN"),
    "hierarchical_vdn_meanq": ("hierarchical_vdn_02_formulate_meanQ", "Hierarchical_VDN_02_Formulate_MeanQ"),
    "hierarchical_vdn_k_means_fixed": ("hierarchical_vdn_k_means", "Hierarchical_VDN_K_Means"),
    "hierarchical_vdac_fixed": ("hierarchical_vdac", "Hierarchical_VDN"),
}


def construct_native(args):
    if args.hmf_algorithm not in IMPLEMENTATIONS:
        raise ValueError("HMF implementation has no confirmed fixed-group entry")
    try:
        from gym.spaces import Box, Discrete
    except ImportError as exc:
        raise ImportError("HMF requires optional gym/cloudpickle dependencies: install unified-marl[hmf]") from exc
    module, name = IMPLEMENTATIONS[args.hmf_algorithm]
    cls = getattr(importlib.import_module(f"unified_marl.algorithms.hmf.algorithms.{module}"), name)
    proxy = SimpleNamespace(n_agents=args.n_agents,
        observation_space=[Box(-np.inf, np.inf, (args.obs_shape,), dtype=np.float32) for _ in range(args.n_agents)],
        action_space=[Discrete(args.n_actions) for _ in range(args.n_agents)])
    supplied = dict(n_heads=args.n_heads, embed_dim=args.embed_dim, bsz=args.batch_size, num_groups=args.num_groups,
                    num_agents=args.n_agents, add_agentID=False, explore=args.explore, grad_norm_clip=args.grad_norm_clip,
                    gamma=args.gamma, tau=args.tau, lr=args.lr, hidden_dim=args.hidden_dim)
    kwargs = {key: supplied[key] for key in inspect.signature(cls.init_from_env).parameters if key in supplied}
    with redirect_stdout(io.StringIO()):
        native = cls.init_from_env(proxy, **kwargs)
    native.device = torch.device(args.device)
    return native


class HMFController:
    def __init__(self, scheme, groups, args):
        if args.device != "cpu":
            raise ValueError("HMF integration currently validates CPU only")
        if args.n_agents % args.num_groups:
            raise ValueError("Fixed HMF grouping requires n_agents divisible by num_groups")
        self.args = args
        self.native = construct_native(args)
        native_agent = self.native.agent
        networks = dict(native_agent.named_children()) if isinstance(native_agent, torch.nn.Module) else {
            key: value for key, value in vars(native_agent).items() if isinstance(value, torch.nn.Module)}
        self.agent = torch.nn.ModuleDict(networks)
        if not networks:
            raise ValueError("Native HMF agent exposes no neural modules")
        self.action_selector = SimpleNamespace(epsilon=float(args.explore))
        self.init_hidden(1)

    def to(self, device):
        self.agent.to(device)
        self.native.device = torch.device(device)
        return self

    def parameters(self):
        return self.agent.parameters()

    def init_hidden(self, batch_size):
        self.mean_actions = torch.zeros(batch_size, self.args.n_agents, self.args.n_actions, device=self.args.device)

    def prepare_rollout(self, completed_episodes):
        epsilon = max(self.args.hmf_final_noise_scale, self.args.explore - self.args.hmf_noise_decay * completed_episodes)
        self.native.scale_noise(epsilon)
        self.action_selector.epsilon = float(epsilon)

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        self.agent.eval()
        obs = ep_batch["obs"][:, t_ep]
        if self.args.hmf_algorithm == "mfq":
            obs = torch.cat([obs, self.mean_actions], dim=-1)
        values = self.native.step(obs.reshape(-1, obs.shape[-1]), explore=not test_mode)
        values = values.reshape(ep_batch.batch_size, self.args.n_agents, self.args.n_actions)
        actions = values.argmax(-1)
        available = ep_batch["avail_actions"][:, t_ep]
        if not available[bs].gather(-1, actions[bs].unsqueeze(-1)).all():
            raise ValueError("Native HMF action selector cannot handle this environment's unavailable actions; no silent policy change is applied")
        self.mean_actions = values.mean(dim=1, keepdim=True).repeat(1, self.args.n_agents, 1).detach()
        return actions[bs]

    def state_dict(self):
        return {"networks": self.agent.state_dict(), "niter": self.native.niter,
                "epsilon": self.action_selector.epsilon}

    def load_state_dict(self, state):
        self.agent.load_state_dict(state["networks"])
        self.native.niter = state["niter"]
        self.action_selector.epsilon = state["epsilon"]


class ScalarCapture:
    def __init__(self):
        self.values = {}

    def add_scalar(self, name, value, step):
        self.values[name] = float(value.detach()) if isinstance(value, torch.Tensor) else float(value)


class HMFLearner(LearnerAPI):
    def __init__(self, mac, scheme, logger, args):
        self.mac, self.args, self.logger = mac, args, logger
        self.records = []

    def observe(self, batch):
        # Explicit per-episode t+1 actions prevent treating the next episode's
        # first action as a bootstrap action at the previous episode boundary.
        valid = batch["filled"][:, :-1, 0].bool()
        if valid.shape[1] > 1:
            finished = batch["terminated"][:, :-2, 0].bool() | batch["truncated"][:, :-2, 0].bool()
            valid[:, 1:] &= ~finished
        keys = ["obs", "actions_onehot", "reward", "terminated", "truncated"]
        if self.args.hmf_algorithm == "mfq":
            keys.append("mean_actions")
        data = {k: batch[k].detach().cpu() for k in keys}
        for b, t in valid.nonzero().tolist():
            self.records.append({"obs": data["obs"][b, t], "actions": data["actions_onehot"][b, t],
                "reward": data["reward"][b, t], "next_obs": data["obs"][b, t+1],
                "next_actions": data["actions_onehot"][b, t+1],
                "done": data["terminated"][b, t] if self.args.hmf_bootstrap_time_limits else
                        data["terminated"][b, t] | data["truncated"][b, t]})
            if self.args.hmf_algorithm == "mfq":
                self.records[-1].update(mean=data["mean_actions"][b, t], next_mean=data["mean_actions"][b, t+1])
        self.records = self.records[-self.args.buffer_size:]

    def update(self, batch, context):
        if batch is not None:
            self.observe(batch)
        if len(self.records) < self.args.batch_size:
            return {"loss": 0.0, "warmup": 1.0}
        if context.env_steps % self.args.hmf_steps_per_update >= self.args.batch_size_run:
            return {"loss": 0.0, "warmup": 1.0}
        indices = np.random.choice(len(self.records), self.args.batch_size, replace=False)
        chosen = [self.records[int(i)] for i in indices]
        values = {k: torch.stack([r[k] for r in chosen]) for k in chosen[0]}
        if self.args.hmf_norm_rewards:
            all_rewards = torch.stack([r["reward"] for r in self.records])
            scale = all_rewards.std(unbiased=False)
            if scale <= 0:
                raise ValueError("Source HMF reward normalization has zero variance; explicitly set hmf_norm_rewards=false for constant-reward data")
            values["reward"] = (values["reward"] - all_rewards.mean()) / scale
        split = lambda x: [x[:, i].contiguous() for i in range(self.args.n_agents)]
        rewards = [values["reward"][:, 0] for _ in range(self.args.n_agents)]
        dones = [values["done"][:, 0].float() for _ in range(self.args.n_agents)]
        if self.args.hmf_algorithm == "mfq":
            mean, next_mean = values["mean"], values["next_mean"]
            sample = (split(values["obs"]), split(values["actions"]), split(mean), rewards,
                      split(values["next_obs"]), split(values["next_actions"]), split(next_mean), dones)
        else:
            sample = (split(values["obs"]), split(values["actions"]), rewards,
                      split(values["next_obs"]), split(values["next_actions"]), dones)
        capture = ScalarCapture()
        self.mac.agent.train()
        kwargs = {"use_next_acts": True, "logger": capture}
        update_params = inspect.signature(self.mac.native.update).parameters
        for name, value in [("mu_weight", self.args.hmf_mu_weight), ("high_weight", self.args.hmf_high_weight)]:
            if name in update_params:
                kwargs[name] = value
        self.mac.native.update(sample, **kwargs)
        self.mac.native.update_all_targets()
        self.mac.agent.eval()
        if not capture.values:
            raise RuntimeError("Native HMF update emitted no loss metrics")
        # Source total already includes component losses; do not double-count
        # those in the generic summary metric.
        summary = capture.values.get("total_vf_loss", sum(v for k, v in capture.values.items() if "loss" in k))
        metrics = {"loss": summary, **capture.values}
        if not all(np.isfinite(v) for v in metrics.values()):
            raise FloatingPointError("Non-finite native HMF losses")
        return metrics

    def extra_state(self):
        optimizers = {name: value.state_dict() for name, value in vars(self.mac.native.agent).items() if isinstance(value, torch.optim.Optimizer)}
        return {"records": self.records, "optimizers": optimizers}

    def load_extra_state(self, state):
        self.records = state["records"]
        for key, value in state["optimizers"].items():
            getattr(self.mac.native.agent, key).load_state_dict(value)
