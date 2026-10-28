"""Explicit source learner/controller bridge. No legacy main or data paths."""
import copy
import importlib
import inspect
import math
import torch
from collections import deque
from unified_marl.learners.base import LearnerAPI
from unified_marl.checkpoints import controller_state, restore_controller
from unified_marl.datasets import PreferenceBatch

ICQ_CONFIGS = {"bc": "bc_smac", "cpl": "cpl_smac", "acpl": "acpl_smac", "icq_ma": "offpg_smac",
    "iql": "iql_smac", "cql": "cql_smac", "dppo": "dppo_smac", "madpo": "madpo_smac",
    "madpo_aux": "madpo_aux_smac", "bcmapt": "bcmapt_smac", "bcq_qmix": "qmix_smac", "bcq_vdn": "vdn_smac"}


def resolve(name):
    if name.startswith("kou/"):
        short = name.split("/",1)[1]
        short = ICQ_CONFIGS.get(short, short.removesuffix(".yaml"))
        return "icq", short
    if name.startswith("kou_old/"):
        return "old", name.split("/",1)[1]
    raise ValueError("Not a Kou source algorithm")


class KouMAC:
    def __init__(self, scheme, groups, args):
        registry = importlib.import_module(f"unified_marl.algorithms.kou.{args.kou_family}.controllers").REGISTRY
        self.native = registry[args.mac](scheme, groups, args)
        self.args = args

    def __getattr__(self, key):
        return getattr(object.__getattribute__(self, "native"), key)

    def to(self, device):
        self.native.agent.to(device)
        self.args.device = str(device)
        return self

    def state_dict(self):
        return {"agent": self.agent.state_dict(), "epsilon": float(self.action_selector.epsilon)}

    def load_state_dict(self, state):
        self.agent.load_state_dict(state["agent"])
        self.action_selector.epsilon = state["epsilon"]


class KouLearner(LearnerAPI):
    def __init__(self, mac, scheme, logger, args):
        registry = importlib.import_module(f"unified_marl.algorithms.kou.{args.kou_family}.learners").REGISTRY
        if args.learner not in registry: raise ValueError(f"Delivered {args.learner} is not registered/implemented")
        self.mac, self.args, self.logger = mac, args, logger
        cls = registry[args.learner]
        if args.learner.startswith("pref_qmix") and not args.policy_dir:
            raise ValueError("Historical preference learner requires its actual policy_dir checkpoint; no reference policy is substituted")
        parameters = inspect.signature(cls).parameters
        if "pref_mac" in parameters:
            if not args.policy_dir: raise ValueError("This historical variant requires its actual reference policy_dir; no policy is substituted")
            self.native = cls(mac, scheme, logger, args, copy.deepcopy(mac))
        else:
            self.native = cls(mac, scheme, logger, args)

    def __getattr__(self, key):
        return getattr(object.__getattribute__(self, "native"), key)

    def to(self, device):
        for value in vars(self.native).values():
            if isinstance(value, torch.nn.Module): value.to(device)
            elif isinstance(value, KouMAC): value.to(device)
            elif hasattr(value, "agent") and isinstance(value.agent, torch.nn.Module): value.agent.to(device)
        return self

    def update(self, batch, context):
        # Update count is the independent clock. No synthetic environment steps.
        step = context.update_step + 1
        self.native.log_stats_t = step - self.args.learner_log_interval
        running = {}
        self.native._integration_losses = {}
        method = self.native.train
        parameters = inspect.signature(method).parameters
        kwargs = {}
        for key in parameters:
            if key in {"t_env", "episode_num"}: kwargs[key] = step
            elif key in {"running_log", "log"}: kwargs[key] = running
            elif key == "reward_model":
                if not hasattr(self, "reward_model"):
                    raise ValueError("Historical learner requires a trained compatible reward model and its extra agent-state fields")
                kwargs[key] = self.reward_model
        if isinstance(batch, PreferenceBatch):
            if "batch1" not in parameters: raise ValueError("This learner expects ordinary episodes, not preference pairs")
            if self.args.learner in {"bcmapt_learner", "madpo_learner", "madpo_aux_learner"}:
                for side in [batch.trajectory0, batch.trajectory1]:
                    if "agent_reward" not in side: raise ValueError("Source algorithm requires explicit individual agent_reward data")
                    if side["agent_reward"].shape[1] != side.max_seq_length:
                        raise ValueError("Individual rewards do not cover the requested preference time window; set max_length explicitly, no reward is padded/copied")
            if "labels" in parameters: kwargs["labels"] = batch.labels
            method(batch.trajectory0, batch.trajectory1, **kwargs)
        else:
            if self.args.learner in {"q_learner_w_globalRM","dmaq_qatten_learner_w_globalRM"} and "reward_hat" not in batch:
                raise ValueError("This source learner requires explicit team reward_hat or reward_model_checkpoint")
            if self.args.learner == "offpg_learner":
                self.native.train_critic(batch, best_batch=None, running_log=running, t_env=step)
            method(batch, **kwargs)
        values = {**self.logger.latest, **running}
        metrics = {k: float(v.detach().cpu()) if isinstance(v, torch.Tensor) else float(v)
                   for k,v in values.items() if isinstance(v, (int,float,torch.Tensor))}
        metrics.update(self.native._integration_losses)
        if "loss" not in metrics:
            key = next((k for k in ["policy_loss", "coma_loss", "q_total_loss", "q_loss", "mixer_loss", "loss_td"] if k in metrics), None)
            if key is None: raise RuntimeError("Source update emitted no recognized loss")
            metrics["loss"] = metrics[key]
        if not all(math.isfinite(v) for v in metrics.values()): raise FloatingPointError("Non-finite source learner metrics")
        return metrics

    def extra_state(self):
        result = {"networks": {}, "controllers": {}, "optimizers": {}, "schedulers": {}, "counters": {}, "histories": {}}
        for name, value in vars(self.native).items():
            if isinstance(value, torch.nn.Module): result["networks"][name] = value.state_dict()
            elif isinstance(value, torch.optim.Optimizer): result["optimizers"][name] = value.state_dict()
            elif isinstance(value, torch.optim.lr_scheduler.LRScheduler): result["schedulers"][name] = value.state_dict()
            elif isinstance(value, KouMAC) or (name.endswith("mac") and hasattr(value,"agent")):
                result["controllers"][name] = controller_state(value)
            elif isinstance(value, (int,float,bool)): result["counters"][name] = value
            elif isinstance(value,deque): result["histories"][name] = {"values":list(value),"maxlen":value.maxlen}
        return result

    def load_extra_state(self, state):
        for category in ["networks", "optimizers", "schedulers"]:
            for name, value in state[category].items(): getattr(self.native,name).load_state_dict(value)
        for name, value in state["controllers"].items(): restore_controller(getattr(self.native,name),value)
        for name, value in state["counters"].items(): setattr(self.native,name,value)
        for name,value in state.get("histories",{}).items(): setattr(self.native,name,deque(value["values"],maxlen=value["maxlen"]))
