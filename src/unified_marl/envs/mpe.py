"""NumPy-only PyMARL adapter for the delivered MPE simple_spread task.

The particle physics and scenario come from Kou's embedded MAPT code, not the
toy coordination environment. Discrete indices follow that delivery's default
one-hot action path: 0 = stay, 1 = +x, 2 = -x, 3 = +y, 4 = -y.

The delivery sums individual scenario rewards for the cooperative team. Its
collision reward loop includes self-collisions (a constant -1 per agent); this
adapter deliberately retains that detail. There are no walls or position clips.
"""
from numbers import Integral
from types import SimpleNamespace

import numpy as np

from .multiagentenv import MultiAgentEnv
from .mpe_native.simple_spread import Scenario


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


class MPEEnv(MultiAgentEnv):
    """Cooperative landmark coverage with fixed agents and five discrete actions.

    Only ``simple_spread`` is supported. Agents and landmarks are randomly
    placed at each reset using an environment-local RandomState. The observation
    is the source scenario observation plus a one-hot agent ID, as in MAPT's
    wrapper. The central state concatenates these observations, retaining every
    agent's velocity and all entity positions. Time limits are truncations.
    """

    def __init__(self, scenario="simple_spread", n_agents=3, n_landmarks=None,
                 episode_limit=25, seed=1):
        if scenario != "simple_spread":
            raise ValueError("MPE currently supports only cooperative scenario 'simple_spread'")
        self.scenario_name = scenario
        self.n_agents = _positive_integer(n_agents, "n_agents")
        self.n_landmarks = _positive_integer(
            self.n_agents if n_landmarks is None else n_landmarks, "n_landmarks")
        self.episode_limit = _positive_integer(episode_limit, "episode_limit")
        self.n_actions = 5
        self.closed = False
        self.t = 0
        self._has_reset = False
        self._scenario = Scenario()
        self.seed_value = self._validate_seed(seed)
        self._rng = np.random.RandomState(self.seed_value)
        self.world = self._scenario.make_world(SimpleNamespace(
            num_agents=self.n_agents, num_landmarks=self.n_landmarks,
            episode_length=self.episode_limit), rng=self._rng)
        self._agent_ids = np.eye(self.n_agents, dtype=np.float32)
        self._stats = {}

    @staticmethod
    def _validate_seed(value):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
                or not 0 <= value < 2**32):
            raise ValueError("seed must be an integer in [0, 2**32)")
        return int(value)

    def _check_open(self):
        if self.closed:
            raise RuntimeError("MPE environment is closed")

    def _check_ready(self):
        self._check_open()
        if not self._has_reset:
            raise RuntimeError("Call reset before using MPE observations or stepping")

    def _check_agent(self, agent_id):
        if (isinstance(agent_id, (bool, np.bool_)) or not isinstance(agent_id, Integral)
                or not 0 <= agent_id < self.n_agents):
            raise ValueError("agent_id is outside the fixed agent range")
        return int(agent_id)

    def seed(self, seed=None):
        """Return the current seed, or reseed subsequent resets explicitly."""
        self._check_open()
        if seed is not None:
            self.seed_value = self._validate_seed(seed)
            self._rng = np.random.RandomState(self.seed_value)
            self.world.rng = self._rng
            # Match the constructor's source make_world initialisation draw.
            self._scenario.reset_world(self.world)
            self._has_reset = False
        return self.seed_value

    def reset(self):
        self._check_open()
        self.t = 0
        self.world.world_step = 0
        self._scenario.reset_world(self.world)
        for agent in self.world.agents:
            agent.action.u = np.zeros(self.world.dim_p)
            agent.action.c = np.zeros(self.world.dim_c)
        self._has_reset = True
        self._stats = self._metrics()
        return self.get_obs(), self.get_state()

    def step(self, actions):
        self._check_ready()
        if self.t >= self.episode_limit:
            raise RuntimeError("MPE episode finished; call reset before step")
        action_array = np.asarray(actions)
        if (action_array.shape != (self.n_agents,)
                or not np.issubdtype(action_array.dtype, np.integer)
                or np.any(action_array < 0) or np.any(action_array >= self.n_actions)):
            raise ValueError("Expected one valid integer action index in [0, 4] per agent")
        # Equivalent to source _set_action(one_hot(index)), not its unused
        # discrete_action_input branch (which reverses the direction indices).
        directions = ((0., 0.), (1., 0.), (-1., 0.), (0., 1.), (0., -1.))
        for agent, action in zip(self.world.agents, action_array):
            sensitivity = 5.0 if agent.accel is None else agent.accel
            agent.action.u = np.asarray(directions[int(action)]) * sensitivity
            agent.action.c = np.zeros(self.world.dim_c)
        self.world.step()
        self.t += 1
        individual_rewards = [float(self._scenario.reward(agent, self.world)) for agent in self.world.agents]
        reward = float(sum(individual_rewards))
        done = self.t >= self.episode_limit
        self._stats = self._metrics()
        return reward, done, dict(self._stats, individual_rewards=individual_rewards, episode_limit=done,
                                 truncated=done, terminated=False)

    def _metrics(self):
        distances = np.asarray([
            min(np.linalg.norm(agent.state.p_pos - landmark.state.p_pos)
                for agent in self.world.agents)
            for landmark in self.world.landmarks])
        # Report actual pair collisions, independently of the source reward's
        # self-collision baseline and its double counting of agent pairs.
        collisions = sum(self._scenario.is_collision(a, b)
                         for i, a in enumerate(self.world.agents)
                         for b in self.world.agents[i + 1:])
        return {"landmark_distance": float(distances.sum()),
                "occupied_landmarks": int(np.count_nonzero(distances < 0.1)),
                "collisions": int(collisions)}

    def get_obs(self):
        self._check_ready()
        return np.asarray([np.concatenate((self._scenario.observation(agent, self.world), self._agent_ids[i]))
                           for i, agent in enumerate(self.world.agents)], dtype=np.float32)

    def get_obs_agent(self, agent_id):
        return self.get_obs()[self._check_agent(agent_id)]

    def get_state(self):
        return self.get_obs().reshape(-1)

    def get_obs_size(self):
        # Own velocity/position + landmark positions + other positions/comms + ID.
        return 4 + 2 * self.n_landmarks + 4 * (self.n_agents - 1) + self.n_agents

    def get_state_size(self):
        return self.n_agents * self.get_obs_size()

    def get_total_actions(self):
        return self.n_actions

    def get_avail_actions(self):
        self._check_open()
        return np.ones((self.n_agents, self.n_actions), dtype=np.int32)

    def get_avail_agent_actions(self, agent_id):
        return self.get_avail_actions()[self._check_agent(agent_id)]

    def get_stats(self):
        self._check_ready()
        return dict(self._stats)

    def close(self):
        self.closed = True

    def render(self):
        raise NotImplementedError("This NumPy MPE adapter does not include rendering")

    def save_replay(self):
        raise NotImplementedError("This NumPy MPE adapter does not include replay files")
