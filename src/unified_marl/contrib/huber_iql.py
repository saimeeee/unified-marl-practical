"""Teaching plugin: independent Q-learning with masked Smooth-L1 TD loss.

This is an explicit example variant, not an author method or a claim of better
performance. There are no environment names or simulator imports in this file.
"""
import math
import torch
from unified_marl.learners.q_learner import QLearner
from unified_marl.registry import AlgorithmSpec, register_algorithm
from unified_marl.config import resource


class HuberIQLLearner(QLearner):
    def __init__(self, mac, scheme, logger, args):
        self.huber_delta = float(args.huber_delta)
        if not math.isfinite(self.huber_delta) or self.huber_delta <= 0:
            raise ValueError("huber_delta must be finite and positive")
        super().__init__(mac, scheme, logger, args)

    def td_loss(self, td_error, mask):
        values = torch.nn.functional.smooth_l1_loss(
            td_error, torch.zeros_like(td_error), reduction="none", beta=self.huber_delta)
        return (values * mask).sum() / mask.sum()


register_algorithm(AlgorithmSpec(
    name="example/huber_iql",
    controller="unified_marl.controllers.basic_controller:BasicMAC",
    learner="unified_marl.contrib.huber_iql:HuberIQLLearner",
    defaults=dict(resource("algs/iql.yaml"), name="huber_iql", huber_delta=1.0,
                  learner="huber_iql", buffer_size=256),
    locked_fields={"mixer": None, "mac": "basic_mac", "agent": "rnn",
                   "agent_output_type": "q", "action_selector": "epsilon_greedy"},
    description="Example opt-in IQL Smooth-L1 loss variant; not an original author contribution",
))
