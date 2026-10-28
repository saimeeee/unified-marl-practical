from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from unified_marl.algorithms.hmf.utils.networks import LocalCritic, MultiHeadAttention_hie, RNN, SUMNet, MFNetwork, MLPNetwork,MultiHeadAttention
from unified_marl.algorithms.hmf.utils.misc import hard_update, gumbel_softmax, onehot_from_logits
from unified_marl.algorithms.hmf.utils.noise import OUNoise
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
class Agent(torch.nn.Module):
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
        print('hiddem_dim', hidden_dim)
        super(Agent, self).__init__()
        self.local_critic = MLPNetwork(num_in_local_critic, num_out_local_pol,
                                        hidden_dim=hidden_dim,
                                        constrain_out=False)
        self.target_local_critic = MLPNetwork(num_in_local_critic, num_out_local_pol,
                                               hidden_dim=hidden_dim,
                                               constrain_out=False)
        self.local_policy = MLPNetwork(num_in_local_critic, num_out_local_pol,
                                        hidden_dim=hidden_dim,
                                        constrain_out=False)
        self.target_local_policy = MLPNetwork(num_in_local_critic, num_out_local_pol,
                                        hidden_dim=hidden_dim,
                                        constrain_out=False)
        self.mf_critic = MLPNetwork(num_in_mf_critic, 1,
                                   hidden_dim=hidden_dim,
                                   constrain_out=False)

        self.target_mf_critic = MLPNetwork(num_in_mf_critic, 1,
                                          hidden_dim=hidden_dim,
                                          constrain_out=False)

        self.mean_policy = MLPNetwork(num_in_mf_critic-num_out_local_pol, 1,
                                   hidden_dim=hidden_dim,
                                   constrain_out=False)

        self.attention = MultiHeadAttention_hie(n_heads=n_heads, input_dim=attention_input_dim,
                                            embed_dim=embed_dim)

        hard_update(self.target_local_critic, self.local_critic)
        hard_update(self.target_mf_critic, self.mf_critic)
        hard_update(self.target_local_policy, self.local_policy)

        self.param = list(self.local_critic.parameters()) + list(self.mf_critic.parameters()) + \
                     list(self.attention.parameters()) + list(self.local_policy.parameters()) + list(self.mean_policy.parameters())


        self.optimizer = Adam(self.param, lr=lr)
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
            
    def cal_dist(self, info):
        pol_out = self.local_policy(info)
        probs = F.softmax(pol_out)
        m = Categorical(probs)
        act = m.sample()
        return m.log_prob(act)
    
    def cal_mean_dist(self, info):
        pol_out = self.mean_policy(info)
        probs = F.softmax(pol_out)
        m = Categorical(probs)
        act = m.sample()
        return m.log_prob(act)

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
        return action


    def get_params(self):
        return {
                'local_critic': self.local_critic.state_dict(),
                'mf_critic': self.mf_critic.state_dict(),
                'target_local_critic': self.target_local_critic.state_dict(),
                'target_mf_critic': self.target_mf_critic.state_dict(),
                'attention': self.attention.state_dict(),
                'optimizer': self.optimizer.state_dict(),

                }

    def load_params(self, params):
        self.local_critic.load_state_dict(params['local_critic'])
        self.mf_critic.load_state_dict(params['mf_critic'])

        self.target_local_critic.load_state_dict(params['target_local_critic'])
        self.target_mf_critic.load_state_dict(params['target_mf_critic'])
        self.attention.load_state_dict(params['attention'])
        self.optimizer.load_state_dict(params['optimizer'])
