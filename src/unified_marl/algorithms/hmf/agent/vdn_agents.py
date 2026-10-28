from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from unified_marl.algorithms.hmf.utils.networks import LocalCritic, MultiHeadAttention, RNN, SUMNet, MLPNetwork
from unified_marl.algorithms.hmf.utils.misc import hard_update, gumbel_softmax, onehot_from_logits
from unified_marl.algorithms.hmf.utils.noise import OUNoise
import numpy as np
import torch
class Agent(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, num_out_local_pol, num_in_local_critic, explore,
                  hidden_dim=64, lr=0.01, discrete_action=True):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.local_critic = MLPNetwork(num_in_local_critic, num_out_local_pol,
                                        hidden_dim=hidden_dim,
                                        constrain_out=True,
                                        discrete_action=discrete_action)
        self.target_local_critic = MLPNetwork(num_in_local_critic, num_out_local_pol,
                                               hidden_dim=hidden_dim,
                                               constrain_out=True,
                                               discrete_action=discrete_action)
        self.sum_Q = SUMNet()
        self.target_sum_Q = SUMNet()

        hard_update(self.target_local_critic, self.local_critic)
        hard_update(self.target_sum_Q, self.sum_Q)

        self.eval_param = list(self.local_critic.parameters()) + list(self.sum_Q.parameters())
        self.eval_optimizer = Adam(self.eval_param, lr=lr)
        self.act_dim = num_out_local_pol
        if not discrete_action:
            self.exploration = OUNoise(num_out_local_pol)
        else:
            self.exploration = explore  # epsilon for eps-greedy
        self.discrete_action = discrete_action

    def reset_noise(self):
        if not self.discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def step(self, input, explore):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        q_value = self.local_critic(input)
        if explore:
            # action = gumbel_softmax(q_value, hard=True)
            action = onehot_from_logits(q_value, self.exploration)
        else:
            action = onehot_from_logits(q_value)
        return action

    def get_params(self):
        return {
                'local_critic': self.local_critic.state_dict(),
                'target_local_critic': self.target_local_critic.state_dict(),
                'eval_optimizer': self.eval_optimizer.state_dict(),
                }

    def load_params(self, params):
        self.local_critic.load_state_dict(params['local_critic'])
        self.target_local_critic.load_state_dict(params['target_local_critic'])
        self.eval_optimizer.load_state_dict(params['eval_optimizer'])
