import torch
import torch.nn.functional as F
from gym.spaces import Box
from unified_marl.algorithms.hmf.utils.misc import soft_update, average_gradients, onehot_from_logits, gumbel_softmax
from unified_marl.algorithms.hmf.agent.maddpg_agents import DDPGAgent

MSELoss = torch.nn.MSELoss()

class MADDPG(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """
    def __init__(self, agent_init_params, alg_types,num_agents,
                 gamma=0.95, tau=0.01, lr=0.01, hidden_dim=256,
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
        self.alg_types = alg_types
        self.agent = DDPGAgent(lr=lr, discrete_action=discrete_action,
                                 hidden_dim=hidden_dim,
                                 **agent_init_params)
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.discrete_action = discrete_action
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.niter = 0
        self.device = torch.device("cpu")

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
        # observations_list= []
        # for i in range(self.nagents):
        #     observations_list.append(observations[:, i, :])
        # actions_list = [a.step(obs, explore=explore) for a, obs in zip(self.agents,
        #                                                          observations_list)]
        # actions = torch.stack(actions_list, dim=1)
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
        """
        local_obs, local_acs, rews, next_local_obs, _, dones = sample
        self.bsz = local_obs[0].shape[0]
        local_obs = torch.reshape(torch.stack(local_obs, dim=1), (self.bsz * self.nagents, -1)).to(self.device)
        local_acs = torch.reshape(torch.stack(local_acs, dim=1), (self.bsz * self.nagents, -1)).to(self.device)
        next_local_obs = torch.reshape(torch.stack(next_local_obs, dim=1), (self.bsz * self.nagents, -1)).to(self.device)
        rews = torch.reshape(rews[0], (self.bsz, -1)).to(self.device)
        dones = torch.reshape(dones[0], (self.bsz, -1)).to(self.device)
        self.agent.critic_optimizer.zero_grad()

        if self.discrete_action: # one-hot encode action
            all_trgt_acs = onehot_from_logits(self.agent.target_policy(next_local_obs))
        else:
            all_trgt_acs = self.agent.target_policy(next_local_obs)
        trgt_vf_in = torch.cat((next_local_obs, all_trgt_acs), dim=-1).reshape(self.bsz, -1)
        target_value = (rews + self.gamma * self.agent.target_critic(trgt_vf_in) * (1 - dones))
        vf_in = torch.cat((local_obs, local_acs), dim=-1).reshape(self.bsz, -1)

        actual_value = self.agent.critic(vf_in)
        vf_loss = MSELoss(actual_value, target_value.detach())
        vf_loss.backward()
        torch.nn.utils.clip_grad_norm(self.agent.critic.parameters(), 0.5)
        self.agent.critic_optimizer.step()

        self.agent.policy_optimizer.zero_grad()
        if self.discrete_action:
            # Forward pass as if onehot (hard=True) but backprop through a differentiable
            # Gumbel-Softmax sample. The MADDPG paper uses the Gumbel-Softmax trick to backprop
            # through discrete categorical samples, but I'm not sure if that is
            # correct since it removes the assumption of a deterministic policy for
            # DDPG. Regardless, discrete policies don't seem to learn properly without it.
            curr_pol_out = self.agent.policy(local_obs)
            curr_pol_vf_in = gumbel_softmax(curr_pol_out, hard=True)
        else:
            curr_pol_out = self.agent.policy(local_obs)
            curr_pol_vf_in = curr_pol_out

        all_pol_acs = curr_pol_vf_in
        vf_in = torch.cat((local_obs, all_pol_acs), dim=-1).reshape(self.bsz, -1)

        pol_loss = -self.agent.critic(vf_in).mean()
        pol_loss += (curr_pol_out**2).mean() * 1e-3
        pol_loss.backward()

        torch.nn.utils.clip_grad_norm(self.agent.policy.parameters(), 0.5)
        self.agent.policy_optimizer.step()
        if logger is not None:
            logger.add_scalar('evader%i/vf_loss', vf_loss, self.niter)
            logger.add_scalar('evader%i/pol_loss', pol_loss, self.niter)


    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        soft_update(self.agent.target_critic, self.agent.critic, self.tau)
        soft_update(self.agent.target_policy, self.agent.policy, self.tau)
        self.niter += 1

    def prep_training(self, device='gpu'):

        self.agent.policy.train()
        self.agent.critic.train()
        self.agent.target_policy.train()
        self.agent.target_critic.train()

        self.agent.policy.to(self.device)
        self.agent.critic.to(self.device)
        self.agent.target_policy.to(self.device)
        self.agent.target_critic.to(self.device)
        # if device == 'gpu':
        #     fn = lambda x: x.cuda()
        # else:
        #     fn = lambda x: x.cpu()
        # if not self.pol_dev == device:
        #     self.agent.policy = fn(self.agent.policy)
        #     self.pol_dev = device
        # if not self.critic_dev == device:
        #     self.agent.critic = fn(self.agent.critic)
        #     self.critic_dev = device
        # if not self.trgt_pol_dev == device:
        #     self.agent.target_policy = fn(self.agent.target_policy)
        #     self.trgt_pol_dev = device
        # if not self.trgt_critic_dev == device:
        #     self.agent.target_critic = fn(self.agent.target_critic)
        #     self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        self.agent.policy.eval()
        self.agent.policy.cpu()
        # if device == 'gpu':
        #     fn = lambda x: x.cuda()
        # else:
        #     fn = lambda x: x.cpu()
        # # only need main policy for rollouts
        # if not self.pol_dev == device:
        #     self.agent.policy = fn(self.agent.policy)
        #     self.pol_dev = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        # self.prep_training(device='cpu')  # move parameters to CPU before saving
        device = torch.device("cpu")
        self.agent.policy.eval()
        self.agent.target_policy.eval()
        self.agent.critic.eval()
        self.agent.target_critic.eval()
        
        self.agent.policy.to(device)
        self.agent.target_policy.to(device)
        self.agent.critic.to(device)
        self.agent.target_critic.to(device)
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': self.agent.get_params()}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, num_agents, agent_alg="MADDPG", adversary_alg="MADDPG",
                      gamma=0.95, tau=0.01, lr=0.01, hidden_dim=256):
        """
        Instantiate instance of this class from multi-agent environment
        """
        obsp = env.observation_space[0]
        acsp = env.action_space[0]
        alg_types = []
        num_in_pol = obsp.shape[0]
        if isinstance(acsp, Box):
            discrete_action = False
            get_shape = lambda x: x.shape[0]
        else:  # Discrete
            discrete_action = True
            get_shape = lambda x: x.n
        num_out_pol = get_shape(acsp)

        num_in_critic = 0
        for _ in range(num_agents):
            num_in_critic += obsp.shape[0]
            num_in_critic += get_shape(acsp)

        agent_init_params = ({'num_in_pol': num_in_pol,
                                  'num_out_pol': num_out_pol,
                                  'num_in_critic': num_in_critic})
        alg_types.append(agent_alg)
        init_dict = {'gamma': gamma, 'tau': tau, 'lr': lr,
                     'hidden_dim': hidden_dim,
                     'alg_types': alg_types,
                     'agent_init_params': agent_init_params,
                     'discrete_action': discrete_action,
                     'num_agents':num_agents}
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