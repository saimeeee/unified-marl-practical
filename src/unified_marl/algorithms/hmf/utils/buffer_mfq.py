import numpy as np
from torch import Tensor
from torch.autograd import Variable

class ReplayBuffer(object):
    """
    Replay Buffer for multi-agent RL with parallel rollouts
    """
    def __init__(self, max_steps, num_agents, num_groups, obs_dims, ac_dims):
        """
        Inputs:
            max_steps (int): Maximum number of timepoints to store in buffer
            num_agents (int): Number of agents in environment
            obs_dims (list of ints): number of obervation dimensions for each
                                     agent
            ac_dims (list of ints): number of action dimensions for each agent
        """
        self.max_steps = max_steps
        self.num_agents = num_agents
        self.obs_buffs = []
        self.ac_buffs = []
        self.rew_buffs = []
        self.next_obs_buffs = []
        self.done_buffs = []
        self.mean_act = []
        self.agents_group = np.zeros((max_steps, num_groups, num_agents))
        odim = obs_dims[0]
        adim = ac_dims[0]
        for _ in range(num_agents):
            self.obs_buffs.append(np.zeros((max_steps, odim)))
            self.ac_buffs.append(np.zeros((max_steps + 1, adim)))
            self.rew_buffs.append(np.zeros(max_steps))
            self.next_obs_buffs.append(np.zeros((max_steps, odim)))
            self.done_buffs.append(np.zeros(max_steps))
            self.mean_act.append(np.zeros((max_steps + 1, adim)))

        self.filled_i = 0  # index of first empty location in buffer (last index when full)
        self.curr_i = 0  # current index to write to (ovewrite oldest data)

    def __len__(self):
        return self.filled_i

    def push(self, observations, actions, mean_act, rewards, next_observations, dones, agents_group=None):
        nentries = observations.shape[0]  # handle multiple parallel environments
        start = 0
        if self.curr_i + nentries > self.max_steps:
            # rollover = self.max_steps - self.curr_i  # num of indices to roll over
            # for agent_i in range(self.num_agents):
            #     self.obs_buffs[agent_i] = np.roll(self.obs_buffs[agent_i],
            #                                       rollover, axis=0)
            #     self.ac_buffs[agent_i] = np.roll(self.ac_buffs[agent_i],
            #                                      rollover, axis=0)
            #     self.rew_buffs[agent_i] = np.roll(self.rew_buffs[agent_i],
            #                                       rollover)
            #     self.next_obs_buffs[agent_i] = np.roll(
            #         self.next_obs_buffs[agent_i], rollover, axis=0)
            #     self.done_buffs[agent_i] = np.roll(self.done_buffs[agent_i],
            #                                        rollover)
            tail = self.max_steps - self.curr_i
            for agent_i in range(self.num_agents):
                self.obs_buffs[agent_i][self.curr_i:self.max_steps] = np.vstack(
                    observations[:tail, agent_i])
                # actions are already batched by agent, so they are indexed differently
                self.ac_buffs[agent_i][self.curr_i:self.max_steps] = actions[:tail, agent_i]
                self.rew_buffs[agent_i][self.curr_i:self.max_steps] = rewards[:tail, agent_i]
                self.next_obs_buffs[agent_i][self.curr_i:self.max_steps] = np.vstack(
                    next_observations[:tail, agent_i])
                self.done_buffs[agent_i][self.curr_i:self.max_steps] = dones[:tail, agent_i]
                self.mean_act[agent_i][self.curr_i:self.max_steps] = mean_act[:tail, agent_i]
            if agents_group is not None:
                self.agents_group[self.curr_i: self.max_steps] = agents_group[:tail]

            self.curr_i = 0
            self.filled_i = self.max_steps
            start = tail
            nentries = nentries - tail
        for agent_i in range(self.num_agents):
            self.obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = np.vstack(
                observations[start:, agent_i])
            # actions are already batched by agent, so they are indexed differently
            self.ac_buffs[agent_i][self.curr_i:self.curr_i + nentries] = actions[start:, agent_i]
            self.mean_act[agent_i][self.curr_i:self.curr_i + nentries] = mean_act[start:, agent_i]
            self.rew_buffs[agent_i][self.curr_i:self.curr_i + nentries] = rewards[start:, agent_i]
            self.next_obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = np.vstack(
                next_observations[start:, agent_i])
            self.done_buffs[agent_i][self.curr_i:self.curr_i + nentries] = dones[start:, agent_i]
        if agents_group is not None:
            self.agents_group[self.curr_i:self.curr_i + nentries] = agents_group[start:]

        self.curr_i += nentries
        if self.filled_i < self.max_steps:
            self.filled_i += nentries
        if self.curr_i == self.max_steps:
            self.curr_i = 0

    def sample(self, N, to_gpu=False, norm_rews=True):
        N = min(N, self.filled_i)
        inds = np.random.choice(np.arange(self.filled_i), size=N,
                                replace=False)
        last_inds = inds - 1
        last_inds = np.where(last_inds == -1, self.filled_i, last_inds)
        if to_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)
        if norm_rews:
            ret_rews = [cast((self.rew_buffs[i][inds] -
                              self.rew_buffs[i][:self.filled_i].mean()) /
                             self.rew_buffs[i][:self.filled_i].std())
                        for i in range(self.num_agents)]
        else:
            ret_rews = [cast(self.rew_buffs[i][inds]) for i in range(self.num_agents)]
        return ([cast(self.obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.ac_buffs[i][inds]) for i in range(self.num_agents)],
                ret_rews,
                [cast(self.next_obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.ac_buffs[i][last_inds]) for i in range(self.num_agents)],
                [cast(self.done_buffs[i][inds]) for i in range(self.num_agents)],
                self.agents_group[inds])
    def sample_with_next_acts(self, N, to_gpu=False, norm_rews=True, dy_group=False):
        N = min(N, self.filled_i-1)
        inds = np.random.choice(np.arange(self.filled_i-1), size=N,
                                replace=False)
        next_inds = inds + 1
        # last_inds = np.where(last_inds == self.filled_i, 0, last_inds)
        if to_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)
        if norm_rews:
            ret_rews = [cast((self.rew_buffs[i][inds] -
                              self.rew_buffs[i][:self.filled_i].mean()) /
                             self.rew_buffs[i][:self.filled_i].std())
                        for i in range(self.num_agents)]
        else:
            ret_rews = [cast(self.rew_buffs[i][inds]) for i in range(self.num_agents)]
        if dy_group:
            return ([cast(self.obs_buffs[i][inds]) for i in range(self.num_agents)],
                    [cast(self.ac_buffs[i][inds]) for i in range(self.num_agents)],
                    ret_rews,
                    [cast(self.next_obs_buffs[i][inds]) for i in range(self.num_agents)],
                    [cast(self.ac_buffs[i][next_inds]) for i in range(self.num_agents)],
                    [cast(self.done_buffs[i][inds]) for i in range(self.num_agents)],
                    self.agents_group[inds],
                    self.agents_group[next_inds])
        else:
            return ([cast(self.obs_buffs[i][inds]) for i in range(self.num_agents)],
                    [cast(self.ac_buffs[i][inds]) for i in range(self.num_agents)],
                    [cast(self.mean_act[i][inds]) for i in range(self.num_agents)],
                    ret_rews,
                    [cast(self.next_obs_buffs[i][inds]) for i in range(self.num_agents)],
                    [cast(self.ac_buffs[i][next_inds]) for i in range(self.num_agents)],
                    [cast(self.mean_act[i][next_inds]) for i in range(self.num_agents)],
                    [cast(self.done_buffs[i][inds]) for i in range(self.num_agents)],)

    def sample_with_mean_acts(self, N, to_gpu=False, norm_rews=True, dy_group=False):
        # next_obs mean_act rews, dones  
        # obs post_mean_act acs
        N = min(N, self.filled_i-1)
        inds = np.random.choice(np.arange(self.filled_i-1), size=N,
                                replace=False)
        next_inds = inds + 1
        # last_inds = np.where(last_inds == self.filled_i, 0, last_inds)
        if to_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)
        if norm_rews:
            ret_rews = [cast((self.rew_buffs[i][inds] -
                              self.rew_buffs[i][:self.filled_i].mean()) /
                             self.rew_buffs[i][:self.filled_i].std())
                        for i in range(self.num_agents)]
        else:
            ret_rews = [cast(self.rew_buffs[i][inds]) for i in range(self.num_agents)]
        if dy_group:
            return ([cast(self.obs_buffs[i][inds]) for i in range(self.num_agents)],
                    [cast(self.ac_buffs[i][inds]) for i in range(self.num_agents)],
                    ret_rews,
                    [cast(self.next_obs_buffs[i][inds]) for i in range(self.num_agents)],
                    [cast(self.ac_buffs[i][next_inds]) for i in range(self.num_agents)],
                    [cast(self.done_buffs[i][inds]) for i in range(self.num_agents)],
                    self.agents_group[inds],
                    self.agents_group[next_inds])
        else:
            return ([cast(self.obs_buffs[i][inds]) for i in range(self.num_agents)],
                    [cast(self.ac_buffs[i][inds]) for i in range(self.num_agents)],
                    ret_rews,
                    [cast(self.next_obs_buffs[i][inds]) for i in range(self.num_agents)],
                    [cast(self.ac_buffs[i][next_inds]) for i in range(self.num_agents)],
                    [cast(self.done_buffs[i][inds]) for i in range(self.num_agents)],)
    def get_average_rewards(self, N):
        if self.filled_i == self.max_steps:
            inds = np.arange(self.curr_i - N, self.curr_i)  # allow for negative indexing
        else:
            inds = np.arange(max(0, self.curr_i - N), self.curr_i)
        return [self.rew_buffs[i][inds].mean() for i in range(self.num_agents)]
