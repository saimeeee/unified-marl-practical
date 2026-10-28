"""Source reward scaling and padding preserved; simulator is an external dependency."""
import numpy as np
from .multiagentenv import MultiAgentEnv


class ReplenishmentAdapter(MultiAgentEnv):
    def __init__(self, task_type="Standard", mode="train", n_agents=100, time_limit=1460,
                 vis_path=None, seed=1, update_config=None, backend_args=None, reward_scale=10000.0, horizon=21):
        try:
            from ReplenishmentEnv import make_env
        except ImportError as exc:
            raise ImportError("ReplenishmentEnv simulator and its task/data assets are required; no simulator is bundled") from exc
        from unified_marl.config import merge
        space = [0.00, 0.16, 0.33, 0.40, 0.45, 0.50, 0.55, 0.60, 0.66, 0.83, 1.00, 1.16, 1.33, 1.50, 1.66, 1.83,
                 2.00, 2.16, 2.33, 2.50, 2.66, 2.83, 3.00, 3.16, 3.33, 3.50, 3.66, 3.83, 4.00, 5.00, 6.00, 7.00, 9.00, 12.00]
        defaults={"action": {"mode": "demand_mean_discrete", "space": space}}
        if horizon is not None: defaults["env"]={"horizon":int(horizon)}
        updates = merge(defaults, update_config)
        self._env = make_env(task_type, wrapper_names=["ObservationWrapper4OldCode", "FlattenWrapper", "OracleWrapper"],
                             mode=mode, vis_path=vis_path, update_config=updates, **(backend_args or {}))
        try:
            self.episode_limit = min(int(time_limit), int(self._env.config["env"]["horizon"]))
            self.n_agents = int(self._env.get_agent_count())
            self.seed_value, self.reward_scale = seed, float(reward_scale)
            if self.reward_scale <= 0:
                raise ValueError("reward_scale must be positive")
            self.n_actions = max(s.n for s in self._env.action_space)
            self.obs_size = max(s.shape[0] for s in self._env.observation_space)
            self.closed = False
            self.reset()
        except BaseException:
            self._env.close()
            raise

    def _set_obs(self, obs):
        self.obs = np.asarray([np.pad(np.asarray(o, dtype=np.float32), (0, self.obs_size - len(o))) for o in obs])
        if not np.isfinite(self.obs).all():
            raise ValueError("Non-finite replenishment observations")

    def reset(self):
        value = self._env.reset()
        obs = value[0] if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], dict) else value
        self._set_obs(obs)
        self.t = 0
        return self.get_obs(), self.get_state()

    def step(self, actions):
        value = self._env.step([int(a) for a in actions])
        if len(value) == 5:
            obs, rewards, terminated, truncated, info = value
            done = bool(np.all(terminated) or np.all(truncated))
            info = dict(info, episode_limit=bool(np.all(truncated) and not np.all(terminated)))
        else:
            obs, rewards, done, info = value
            done = bool(np.all(done))
            info = dict(info)
        self.t += 1
        if self.t >= self.episode_limit and not done:
            done = True
            info["episode_limit"] = True
        self._set_obs(obs)
        individual = np.asarray(rewards, dtype=np.float32).reshape(-1) / self.reward_scale
        info["individual_rewards"] = individual
        return float(individual.sum()), done, info

    def get_obs(self):
        return self.obs

    def get_obs_agent(self, agent_id):
        return self.obs[agent_id]

    def get_state(self):
        return self.obs.reshape(-1).copy()

    def get_obs_size(self):
        return self.obs_size

    def get_state_size(self):
        return self.n_agents * self.obs_size

    def get_total_actions(self):
        return self.n_actions

    def get_avail_agent_actions(self, agent_id):
        count = self._env.action_space[agent_id].n
        return [1] * count + [0] * (self.n_actions - count)

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def get_stats(self):
        return {}

    def close(self):
        if not self.closed:
            self.closed = True
            self._env.close()

    def save_replay(self):
        raise NotImplementedError("Use the backend's explicit render/export API")
