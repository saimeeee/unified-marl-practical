"""Small deterministic test environment; never a substitute for SMAC evidence."""
import numpy as np
from .multiagentenv import MultiAgentEnv


class CoordinationEnv(MultiAgentEnv):
    def __init__(self, n_agents=2, episode_limit=4, seed=1, truncate=False, state_last_action=False, mask_actions=True):
        if state_last_action:
            raise ValueError("Toy environment does not implement state_last_action=True")
        self.n_agents, self.episode_limit = int(n_agents), int(episode_limit)
        self.seed_value, self.truncate = int(seed), bool(truncate)
        self.mask_actions = bool(mask_actions)
        self.n_actions, self.t, self.closed = 3, 0, False
        self.actions_seen = []

    def _check(self):
        if self.closed:
            raise RuntimeError("Environment is closed")

    def _targets(self):
        return (np.arange(self.n_agents) + self.t + self.seed_value) % 2

    def reset(self):
        self._check()
        self.t, self.actions_seen = 0, []
        return self.get_obs(), self.get_state()

    def step(self, actions):
        self._check()
        if self.t >= self.episode_limit:
            raise RuntimeError("Call reset before stepping a finished episode")
        actions = np.asarray(actions, dtype=np.int64).reshape(self.n_agents)
        available = np.asarray(self.get_avail_actions())
        if np.any(actions < 0) or np.any(actions >= self.n_actions) or not np.all(available[np.arange(self.n_agents), actions]):
            raise ValueError("Unavailable action selected")
        reward = float(np.mean(actions == self._targets()))
        self.actions_seen.append(actions.tolist())
        self.t += 1
        done = self.t >= self.episode_limit
        return reward, done, {"episode_limit": bool(done and self.truncate)}

    def get_obs(self):
        return [np.asarray([self.t / self.episode_limit, *self._targets(), i / max(1, self.n_agents - 1)], dtype=np.float32)
                for i in range(self.n_agents)]

    def get_obs_agent(self, agent_id):
        return self.get_obs()[agent_id]

    def get_state(self):
        return np.asarray([self.t / self.episode_limit, *self._targets()], dtype=np.float32)

    def get_avail_actions(self):
        # Actions 0 and 1 remain legal; action 2 is sometimes masked.
        return [[1, 1, int(not self.mask_actions or (self.t + i) % 2 == 0)] for i in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        return self.get_avail_actions()[agent_id]

    def get_obs_size(self):
        return self.n_agents + 2

    def get_state_size(self):
        return self.n_agents + 1

    def get_total_actions(self):
        return self.n_actions

    def get_stats(self):
        return {}

    def seed(self):
        return self.seed_value

    def close(self):
        self.closed = True

    def save_replay(self):
        raise NotImplementedError("Toy environment has no replay format")
