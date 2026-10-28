from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from unified_marl.algorithms.hmf.utils.networks import LocalCritic, MultiHeadAttention_hie02, RNN, SUMNet, MFNetwork, MLPNetwork2,MultiHeadAttention
from unified_marl.algorithms.hmf.utils.misc import hard_update, gumbel_softmax, onehot_from_logits
from unified_marl.algorithms.hmf.utils.noise import OUNoise
import numpy as np
import torch
class Agent(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, num_out_local_pol, num_in_local_critic, num_in_mf_critic, num_in_mf_policy,
                 n_heads, embed_dim, attention_input_dim, explore, hidden_dim=64, lr=0.01, discrete_action=True):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """

        self.local_critic = MLPNetwork2(num_in_local_critic, num_out_local_pol,
                                        hidden_dim=hidden_dim,
                                        constrain_out=False)
        self.target_local_critic = MLPNetwork2(num_in_local_critic, num_out_local_pol,
                                               hidden_dim=hidden_dim,
                                               constrain_out=False)
        self.mf_critic = MLPNetwork2(num_in_mf_critic, 1,
                                   hidden_dim=hidden_dim,
                                   constrain_out=False)
        self.mf_policy = MLPNetwork2(num_in_mf_policy, num_out_local_pol, hidden_dim=hidden_dim, discrete_action=False,
                                     constrain_out=True)
        self.target_mf_policy = MLPNetwork2(num_in_mf_policy, num_out_local_pol, hidden_dim=hidden_dim, discrete_action=False,
                                            constrain_out=True)

        self.target_mf_critic = MLPNetwork2(num_in_mf_critic, 1,
                                          hidden_dim=hidden_dim,
                                          constrain_out=False)


        self.attention = MultiHeadAttention_hie02(n_heads=n_heads, input_dim=attention_input_dim,
                                            embed_dim=embed_dim)

        hard_update(self.target_local_critic, self.local_critic)
        hard_update(self.target_mf_critic, self.mf_critic)
        hard_update(self.target_mf_policy, self.mf_policy)

        self.low_param = list(self.local_critic.parameters()) + list(self.mf_critic.parameters()) + \
                         list(self.attention.parameters())
        self.high_param = list(self.mf_critic.parameters())


        self.low_optimizer = Adam(self.low_param, lr=lr)
        self.high_optimizer = Adam(self.high_param, lr=lr)
        self.mf_policy_optimizer = Adam(self.mf_policy.parameters(), lr=lr)
        self.mf_policy_exploration = OUNoise(num_out_local_pol)
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

    def step(self, input, explore, return_q_value=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        q_value = self.local_critic(input)
        if self.discrete_action:
            if explore:
                # action = gumbel_softmax(q_value, hard=True)
                action = onehot_from_logits(q_value, self.exploration)
            else:
                action = onehot_from_logits(q_value)
        else:  # continuous action
            # if explore:
            #     action += Variable(Tensor(self.exploration.noise()),
            #                        requires_grad=False)
            # action = action.clamp(-1, 1)
            action = None
            pass
        if return_q_value == True:
            return action, q_value
        return action
    def get_mf_act(self, input, explore):
        mf_act = self.mf_policy(input)
        if explore:
            noise = torch.from_numpy(self.mf_policy_exploration.noise()).float()
            mf_act = mf_act + noise
        mf_act = (mf_act + 1) * 0.5
        return mf_act

    def get_params(self):
        return {
                'local_critic': self.local_critic.state_dict(),
                'mf_critic': self.mf_critic.state_dict(),
                'target_local_critic': self.target_local_critic.state_dict(),
                'target_mf_critic': self.target_mf_critic.state_dict(),
                'attention': self.attention.state_dict(),
                'mf_policy': self.mf_policy.state_dict(),
                'target_mf_policy': self.target_mf_policy.state_dict(),

                'low_optimizer': self.low_optimizer.state_dict(),
                'high_optimizer': self.high_optimizer.state_dict(),
                'mf_policy_optimizer': self.mf_policy_optimizer.state_dict()
                }

    def load_params(self, params):
        self.local_critic.load_state_dict(params['local_critic'])
        self.mf_critic.load_state_dict(params['mf_critic'])

        self.target_local_critic.load_state_dict(params['target_local_critic'])
        self.target_mf_critic.load_state_dict(params['target_mf_critic'])
        self.attention.load_state_dict(params['attention'])
        self.mf_policy.load_state_dict(params['mf_policy'])
        self.target_mf_policy.load_state_dict(params['target_mf_policy'])

        self.low_optimizer.load_state_dict(params['low_optimizer'])
        self.high_optimizer.load_state_dict(params['high_optimizer'])
        self.mf_policy_optimizer.load_state_dict(params['mf_policy_optimizer'])