import torch
import torch.nn.functional as F
from gym.spaces import Box, Discrete
from unified_marl.algorithms.hmf.utils.misc import soft_update, average_gradients, onehot_from_logits, gumbel_softmax
from unified_marl.algorithms.hmf.agent.hierarchical_qmix_agent import Agent


MSELoss = torch.nn.MSELoss()

class Hierarchical_Qmix(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """
    def __init__(self, agent_init_params, num_agents, n_heads, bsz, num_groups, add_agentID, explore,
                 act_dim, obs_dim, grad_norm_clip, embed_dim,gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64,
                 discrete_action=False):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
                num_in_critic (int): Input dimensions to critic
            alg_types (list of str): Learning algorithm for each agent (DDPG
                                       or MADDPG)
            gamma (float): Discount factor
            tau (float): Target update rate
            lr (float): Learning rate for policy and critic
            hidden_dim (int): Number of hidden dimensions for networks
            discrete_action (bool): Whether or not to use discrete action space
        """
        self.nagents = num_agents
        self.explore = explore
        self.agent = Agent(lr=lr, discrete_action=discrete_action,
                               hidden_dim=hidden_dim, explore=self.explore,
                               **agent_init_params)

        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.discrete_action = discrete_action
        self.add_agentID = add_agentID
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.niter = 0

        self.bsz = bsz
        self.num_groups = num_groups
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.size_of_group = int(self.nagents / self.num_groups)
        self.batch_agent_id = torch.reshape(torch.tensor([[i for i in range(self.nagents)] for _ in range(self.bsz)]),
                                            (self.bsz, self.nagents, -1))
        self.agent_id_mask = torch.zeros(self.bsz, self.nagents, self.nagents).scatter_(2, self.batch_agent_id, 1).type(
            torch.uint8)
        self.n_heads = n_heads
        self.grad_norm_clip = grad_norm_clip
        self.embed_dim = embed_dim

    def scale_noise(self, scale):
        """
        Scale noise for each agent
        Inputs:
            scale (float): scale of noise
        """

        self.agent.scale_noise(scale)

    def reset_noise(self):
        self.agent.reset_noise()

    def step(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        return self.agent.step(observations, explore=explore)
    def update(self, sample, use_next_acts, logger=None):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged
            all : 表示每个agent的均值Q的联合信息输入
            trgt: 表示目标网络的输入  eval
            local: 表示个体网络的输入  mf: 表示均值场网络的输入
            ind_act: 代表从局部神经网络得到的最优行为 option：代表从MF神经网络得到的最优行为
                   act: 动作  obs:观测   vf_in -- >info
        """
        local_obs, local_acs, rewards, next_local_obs, trgt_local_acs, dones = sample
        self.bsz = local_obs[0].shape[0]
        local_obs = torch.reshape(torch.stack(local_obs, dim=1), (self.bsz * self.nagents, -1))
        local_acs = torch.reshape(torch.stack(local_acs, dim=1), (self.bsz * self.nagents, -1))
        next_local_obs = torch.reshape(torch.stack(next_local_obs, dim=1), (self.bsz * self.nagents, -1))
        trgt_local_acs = torch.reshape(torch.stack(trgt_local_acs, dim=1), (self.bsz * self.nagents, -1))

        low_rews = torch.reshape(rewards[0], (self.bsz, 1, -1)).repeat(1, self.num_groups, 1)
        low_dones = torch.reshape(dones[0], (self.bsz, 1, -1)).repeat(1, self.num_groups, 1)

        '''high update'''
        high_rews = torch.reshape(rewards[0], (self.bsz, 1))
        high_dones = torch.reshape(dones[0], (self.bsz, 1))

        agents_id = torch.tensor([[[i] for i in range(self.nagents)] for _ in range(self.bsz)])
        agents_id_onehot = torch.zeros(self.bsz, self.nagents, self.nagents).scatter_(2, agents_id, 1)
        agents_id_onehot = torch.reshape(agents_id_onehot, (self.bsz * self.nagents, -1))
        cat_agent_id = lambda x: torch.cat((x, agents_id_onehot), dim=-1) if self.add_agentID else x
        groups_id = torch.tensor([[[i] for i in range(self.num_groups)] for _ in range(self.bsz)])
        groups_id_onehot = torch.zeros(self.bsz, self.num_groups, self.num_groups).scatter_(2, groups_id, 1). \
            view(self.bsz, self.num_groups, self.num_groups, 1).repeat(1, 1, 1, self.nagents // self.num_groups)
        groups_id_onehot = torch.reshape(groups_id_onehot, (self.bsz, self.num_groups, self.nagents)).type(
            torch.uint8) * -1 + 1
        """
        更新local_critic 和 mf_critic 和 attention
        """
        self.agent.optimizer.zero_grad()
        trgt_local_Q = self.agent.target_local_critic(cat_agent_id(next_local_obs))

        trgt_local_act = onehot_from_logits(trgt_local_Q)
        trgt_local_max_Q = torch.gather(trgt_local_Q, dim=-1, index=torch.argmax(trgt_local_act, dim=-1, keepdim=True))
        trgt_local_obs = next_local_obs


        trgt_local_info = torch.cat((trgt_local_obs, trgt_local_act), dim=-1)
        trgt_local_info = torch.reshape(trgt_local_info, (self.bsz, self.num_groups, self.nagents//self.num_groups, -1))
        trgt_mf_info = torch.mean(trgt_local_info, dim=2)
        trgt_local_info = torch.reshape(trgt_local_info, (self.bsz, self.nagents, -1))
        trgt_mf_info_use_attn, trgt_mf_weight = self.agent.attention(h=trgt_mf_info, he=trgt_local_info, num_groups=self.num_groups,
                                                 mask=groups_id_onehot, return_attn=True, calculate_V=True)

        trgt_mf_info_use_attn = torch.reshape(trgt_mf_info_use_attn, (self.bsz * self.num_groups, -1))

        trgt_mf_Q = self.agent.target_mf_critic(cat_agent_id(trgt_mf_info_use_attn))
        trgt_mf_Q = torch.reshape(trgt_mf_Q, (self.bsz, self.num_groups, 1))

        trgt_mf_weight = torch.reshape(trgt_mf_weight, (self.n_heads, self.bsz, self.num_groups, 1, self.nagents))
        trgt_local_max_Q = torch.reshape(trgt_local_max_Q, (self.bsz, 1, self.nagents, 1)).repeat(1, self.num_groups, 1, 1)
        trgt_local_max_Q_with_wieght = torch.reshape(torch.sum(torch.matmul(trgt_mf_weight, trgt_local_max_Q), dim=0),
                                                     (self.bsz, self.num_groups, 1))

        trgt_total_Q = trgt_local_max_Q_with_wieght + trgt_mf_Q * 2
        trgt_value = (low_rews + self.gamma * trgt_total_Q * (1 - low_dones))


        eval_local_Q = torch.gather(self.agent.local_critic(cat_agent_id(local_obs)), dim=-1,
                                    index=torch.argmax(local_acs, dim=-1, keepdim=True))

        eval_local_info = torch.cat((local_obs, local_acs), dim=-1)

        eval_local_info = torch.reshape(eval_local_info, (self.bsz, self.num_groups, self.nagents//self.num_groups, -1))
        eval_mf_info = torch.mean(eval_local_info, dim=2)
        eval_local_info = torch.reshape(eval_local_info, (self.bsz, self.nagents, -1))
        eval_mf_info_use_attn, eval_mf_weight = self.agent.attention(h=eval_mf_info, he=eval_local_info, num_groups=self.num_groups,
                                              mask=groups_id_onehot, return_attn=True, calculate_V=True)

        eval_mf_info_use_attn = torch.reshape(eval_mf_info_use_attn, (self.bsz * self.num_groups, -1))

        eval_mf_Q = self.agent.mf_critic(cat_agent_id(eval_mf_info_use_attn))
        eval_mf_Q = torch.reshape(eval_mf_Q, (self.bsz, self.num_groups, 1))

        eval_mf_weight = torch.reshape(eval_mf_weight, (self.n_heads, self.bsz, self.num_groups, 1, self.nagents))
        eval_local_Q = torch.reshape(eval_local_Q, (self.bsz, 1, self.nagents, 1)).repeat(1, self.num_groups, 1, 1)
        eval_local_Q_with_weight = torch.reshape(torch.sum(torch.matmul(eval_mf_weight, eval_local_Q), dim=0), (self.bsz, self.num_groups, -1))
        eval_total_Q = eval_local_Q_with_weight + eval_mf_Q * 2

        '''high update'''
        trgt_mf_total_Q = self.agent.target_qmix_net(trgt_mf_Q, trgt_mf_info_use_attn)
        trgt_high_value = (high_rews + self.gamma * trgt_mf_total_Q * (1 - high_dones))
        eval_high_value = self.agent.qmix_net(eval_mf_Q, eval_mf_info_use_attn)
        high_mf_critic_loss = MSELoss(eval_high_value, trgt_high_value.detach())

        local_vf_loss = MSELoss(eval_local_Q, trgt_local_max_Q)
        mf_vf_loss = MSELoss(eval_mf_Q, trgt_mf_Q)
        total_vf_loss = MSELoss(eval_total_Q, trgt_value.detach())
        total_vf_loss += high_mf_critic_loss * 0.1
        total_vf_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.param, self.grad_norm_clip)
        self.agent.optimizer.step()


        if logger is not None:
            logger.add_scalar('total_vf_loss', total_vf_loss, self.niter)
            logger.add_scalar('local_vf_loss', local_vf_loss, self.niter)
            logger.add_scalar('mf_vf_loss', mf_vf_loss, self.niter)
            logger.add_scalar('high_mf_critic_loss', high_mf_critic_loss, self.niter)

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        soft_update(self.agent.target_local_critic, self.agent.local_critic, self.tau)
        soft_update(self.agent.target_mf_critic, self.agent.mf_critic, self.tau)
        soft_update(self.agent.target_qmix_net, self.agent.qmix_net, self.tau)
        self.niter += 1

    def prep_training(self, device='gpu'):
        self.agent.local_critic.train()
        self.agent.mf_critic.train()
        self.agent.target_local_critic.train()
        self.agent.target_mf_critic.train()
        self.agent.attention.train()
        self.agent.qmix_net.train()
        self.agent.target_qmix_net.train()

        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.critic_dev == device:
            self.agent.local_critic = fn(self.agent.local_critic)
            self.agent.mf_critic = fn(self.agent.mf_critic)
            self.agent.attention = fn(self.agent.attention)
            self.agent.qmix_net = fn(self.agent.qmix_net)
            self.critic_dev = device

        if not self.trgt_critic_dev == device:
            self.agent.target_local_critic = fn(self.agent.target_local_critic)
            self.agent.target_mf_critic = fn(self.agent.target_mf_critic)
            self.agent.target_qmix_net = fn(self.agent.target_qmix_net)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):

        self.agent.local_critic.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.critic_dev == device:
            self.agent.local_critic = fn(self.agent.local_critic)
            self.critic_dev = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': self.agent.get_params()}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, n_heads, embed_dim, bsz, num_groups, add_agentID, explore,
                      grad_norm_clip, num_evaders=0, gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64):
        """
        Instantiate instance of this class from multi-agent environment
        """


        obsp = env.observation_space[0]
        acsp = env.action_space[0]

        num_agents = 0
        if isinstance(acsp, Box):
            discrete_action = False
            get_shape = lambda x: x.shape[0]
        else:  # Discrete
            discrete_action = True
            get_shape = lambda x: x.n

        act_dim = get_shape(acsp)
        obs_dim = obsp.shape[0]
        attention_input_dim = act_dim + obs_dim

        num_out_local_pol = act_dim
        num_in_local_critic = obs_dim


        num_agents = env.n_agents
        num_agents -= num_evaders
        num_in_mf_critic = embed_dim
        num_in_mf_policy = embed_dim
        if add_agentID:
            num_in_local_critic += num_agents
            num_in_mf_critic += num_agents
            num_in_mf_policy += num_agents
        agent_init_params = ({'num_out_local_pol': num_out_local_pol,
                              'num_in_local_critic': num_in_local_critic,
                              'num_in_mf_critic': num_in_mf_critic,
                              'n_heads': n_heads,
                              'embed_dim': embed_dim,
                              'attention_input_dim': attention_input_dim,
                              'num_groups': num_groups
                              })

        init_dict = {'gamma': gamma, 'tau': tau, 'lr': lr,
                     'hidden_dim': hidden_dim,
                     'agent_init_params': agent_init_params,
                     'discrete_action': discrete_action,
                     'num_agents': num_agents,
                     'n_heads': n_heads,
                     'bsz': bsz,
                     'num_groups': num_groups,
                     'act_dim': act_dim,
                     'obs_dim': obs_dim,
                     'add_agentID': add_agentID,
                     'explore': explore,
                     'grad_norm_clip': grad_norm_clip,
                     'embed_dim': embed_dim
                     }
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        instance.agent.load_params(save_dict['agent_params'])
        return instance