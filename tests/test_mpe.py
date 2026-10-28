"""MPE adapter physics, source compatibility and episode-contract checks."""
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from unified_marl.envs.mpe import MPEEnv


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT.parent / "marl平台代码/寇谦/论文代码及技术文档/论文代码及技术文档/"
          "ICQ-MA/src/components/MAPT/mat/envs/mpe")


def set_positions(env, agents, landmarks):
    for agent, position in zip(env.world.agents, agents):
        agent.state.p_pos = np.asarray(position, dtype=float)
        agent.state.p_vel = np.zeros(2)
    for landmark, position in zip(env.world.landmarks, landmarks):
        landmark.state.p_pos = np.asarray(position, dtype=float)


@pytest.mark.parametrize("n_agents,n_landmarks", [(1, 1), (3, 3), (4, 2)])
def test_shapes_observations_and_central_state(n_agents, n_landmarks):
    env = MPEEnv(n_agents=n_agents, n_landmarks=n_landmarks)
    obs, state = env.reset()
    assert obs.shape == (n_agents, env.get_obs_size())
    assert obs.dtype == state.dtype == np.float32
    assert state.shape == (env.get_state_size(),)
    np.testing.assert_array_equal(state, obs.reshape(-1))
    np.testing.assert_array_equal(obs[:, -n_agents:], np.eye(n_agents))
    np.testing.assert_array_equal(obs[0, :2], env.world.agents[0].state.p_vel)
    np.testing.assert_allclose(obs[0, 4:6], env.world.landmarks[0].state.p_pos - env.world.agents[0].state.p_pos)
    np.testing.assert_array_equal(env.get_obs_agent(0), obs[0])
    np.testing.assert_array_equal(env.get_avail_actions(), np.ones((n_agents, 5)))
    assert env.get_env_info() == {"state_shape": state.size, "obs_shape": obs.shape[1],
                                  "n_agents": n_agents, "n_actions": 5, "episode_limit": 25}
    # Public observation arrays must not provide writable views of simulator state.
    obs.fill(100)
    assert not np.any(env.get_obs() == 100)


def test_same_seed_multiple_episodes_and_explicit_reseed():
    first, second = MPEEnv(seed=73), MPEEnv(seed=73)
    other = MPEEnv(seed=74)
    initial = first.reset()[1].copy()
    np.testing.assert_array_equal(initial, second.reset()[1])
    assert not np.array_equal(initial, other.reset()[1])
    for _ in range(3):
        for action in ([1, 2, 4], [0, 3, 1], [2, 4, 0]):
            assert first.step(action) == second.step(action)
            np.testing.assert_array_equal(first.get_state(), second.get_state())
        np.testing.assert_array_equal(first.reset()[1], second.reset()[1])
    assert not np.array_equal(initial, first.get_state())
    assert first.seed() == 73
    first.seed(73)
    np.testing.assert_array_equal(first.reset()[1], initial)


def test_environment_rng_does_not_read_or_modify_global_numpy_rng():
    state = np.random.get_state()
    try:
        np.random.seed(1401)
        expected = np.random.random(10)
        np.random.seed(1401)
        first = MPEEnv(seed=29)
        first.reset()
        first.step([1, 2, 3])
        first.reset()
        np.testing.assert_array_equal(np.random.random(10), expected)
        second = MPEEnv(seed=29)
        np.random.random(103)
        second.reset()
        second.step([1, 2, 3])
        np.testing.assert_array_equal(second.reset()[1], first.get_state())
    finally:
        np.random.set_state(state)


@pytest.mark.parametrize("action,direction", [(0, (0, 0)), (1, (1, 0)), (2, (-1, 0)),
                                               (3, (0, 1)), (4, (0, -1))])
def test_discrete_motion_and_source_damping(action, direction):
    env = MPEEnv(n_agents=1)
    env.reset()
    set_positions(env, [[0, 0]], [[0.7, 0.7]])
    env.step([action])
    np.testing.assert_allclose(env.world.agents[0].state.p_vel, np.asarray(direction) * 0.5)
    np.testing.assert_allclose(env.world.agents[0].state.p_pos, np.asarray(direction) * 0.05)
    env.step([0])
    np.testing.assert_allclose(env.world.agents[0].state.p_vel, np.asarray(direction) * 0.375)
    np.testing.assert_allclose(env.world.agents[0].state.p_pos, np.asarray(direction) * 0.0875)


def test_reward_retains_source_team_sum_and_self_collision_baseline():
    env = MPEEnv(n_agents=2)
    env.reset()
    set_positions(env, [[-1, 0], [1, 0]], [[-1, 0], [1, 0]])
    reward, done, info = env.step([0, 0])
    assert reward == -2.0  # Each source scenario reward includes its own collision.
    assert info["individual_rewards"] == [-1.0, -1.0]
    assert reward == sum(info["individual_rewards"])
    assert not done and info["occupied_landmarks"] == 2 and info["collisions"] == 0
    set_positions(env, [[-1, 0], [1, 0]], [[-1, 1], [1, 1]])
    reward, _, info = env.step([0, 0])
    assert info["landmark_distance"] == 2
    assert reward == -6.0  # Both agents receive -sum(min distances)-1, then sum.
    assert reward == sum(info["individual_rewards"])


def test_contacts_push_particles_apart_and_have_source_reward_penalty():
    env = MPEEnv(n_agents=2)
    env.reset()
    set_positions(env, [[-0.1, 0], [0.1, 0]], [[-1, 1], [1, 1]])
    env.world.contact_force = 0
    reward, _, info = env.step([0, 0])
    assert info["collisions"] == 1
    assert reward == pytest.approx(-2 * info["landmark_distance"] - 4)
    env.world.contact_force = 100
    env.step([0, 0])
    assert env.world.agents[0].state.p_pos[0] < -0.1
    assert env.world.agents[1].state.p_pos[0] > 0.1
    np.testing.assert_allclose(env.world.agents[0].state.p_vel, -env.world.agents[1].state.p_vel)


def test_exact_coincidence_has_finite_deterministic_contact():
    first, second = MPEEnv(n_agents=2), MPEEnv(n_agents=2)
    for env in (first, second):
        env.reset()
        set_positions(env, [[0, 0], [0, 0]], [[-1, 1], [1, 1]])
        reward, _, _ = env.step([0, 0])
        assert np.isfinite(reward) and np.isfinite(env.get_state()).all()
        assert np.linalg.norm(env.world.agents[0].state.p_pos - env.world.agents[1].state.p_pos) > 0
    np.testing.assert_array_equal(first.get_state(), second.get_state())


def test_world_has_no_implicit_arena_wall_or_position_clip():
    env = MPEEnv(n_agents=1, episode_limit=30)
    env.reset()
    set_positions(env, [[0.99, 0]], [[0, 0]])
    _, done, info = env.step([1])
    assert env.world.agents[0].state.p_pos[0] > 1
    assert not done and not info["terminated"]


def test_time_limit_reset_and_closed_lifecycle():
    env = MPEEnv(episode_limit=2)
    with pytest.raises(RuntimeError, match="reset"):
        env.step([0, 0, 0])
    env.reset()
    _, done, info = env.step([0, 0, 0])
    assert not done and not info["episode_limit"] and not info["truncated"]
    _, done, info = env.step([0, 0, 0])
    assert done and info["episode_limit"] and info["truncated"] and not info["terminated"]
    assert np.isfinite(env.get_state()).all()  # Final bootstrap observation remains valid.
    with pytest.raises(RuntimeError, match="finished"):
        env.step([0, 0, 0])
    env.reset()
    assert env.t == env.world.world_step == 0
    assert not env.step([0, 0, 0])[1]
    env.close()
    env.close()
    for operation in (env.reset, env.get_obs, env.get_avail_actions, env.get_stats):
        with pytest.raises(RuntimeError, match="closed"):
            operation()


@pytest.mark.parametrize("actions", [[-1, 0, 0], [5, 0, 0], [0.5, 1, 2], [0., 1., 2.],
                                    [True, False, True], [0, 1], [[0], [1], [2]],
                                    ["0", "1", "2"], [np.nan, 1, 2]])
def test_invalid_actions_are_rejected_before_mutating_world(actions):
    env = MPEEnv()
    env.reset()
    state = env.get_state().copy()
    with pytest.raises(ValueError, match="integer action"):
        env.step(actions)
    assert env.t == env.world.world_step == 0
    np.testing.assert_array_equal(env.get_state(), state)


@pytest.mark.parametrize("kwargs", [{"scenario": "simple_tag"}, {"n_agents": 0},
                                    {"n_agents": True}, {"n_agents": 2.5},
                                    {"n_landmarks": 0}, {"episode_limit": 0},
                                    {"episode_limit": 3.5}, {"seed": -1},
                                    {"seed": 2**32}, {"seed": True}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        MPEEnv(**kwargs)


@pytest.mark.parametrize("agent_id", [-1, 3, 1.5, True])
def test_invalid_agent_indices(agent_id):
    env = MPEEnv()
    env.reset()
    with pytest.raises(ValueError, match="agent_id"):
        env.get_obs_agent(agent_id)
    with pytest.raises(ValueError, match="agent_id"):
        env.get_avail_agent_actions(agent_id)


def test_environment_import_does_not_load_gym_rendering_or_mapts_training():
    subprocess.run([sys.executable, "-c", "import sys; from unified_marl.envs.mpe import MPEEnv; "
                    "e=MPEEnv(); e.reset(); e.step([0,1,2]); "
                    "assert not any(n in sys.modules for n in ('gym','pyglet','mat','torch'))"],
                   check=True, cwd=ROOT)


def _load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not SOURCE.is_dir(), reason="Original delivery available only in the development workspace")
def test_delivered_source_trajectory_observation_and_reward_parity(monkeypatch):
    # Load only the three source files, never source Gym/rendering/training code.
    for name in ("mat", "mat.envs", "mat.envs.mpe"):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    for name, filename in (("mat.envs.mpe.core", "core.py"), ("mat.envs.mpe.scenario", "scenario.py")):
        monkeypatch.setitem(sys.modules, name, _load_source(name, SOURCE / filename))
    source_scenario = _load_source("_mpe_source_spread", SOURCE / "scenarios/simple_spread.py").Scenario()
    global_rng_state = np.random.get_state()
    try:
        np.random.seed(2026)
        source_world = source_scenario.make_world(SimpleNamespace(num_agents=3, num_landmarks=3, episode_length=25))
        source_scenario.reset_world(source_world)
    finally:
        np.random.set_state(global_rng_state)
    adapted = MPEEnv(seed=2026)
    adapted.reset()
    random_actions = np.random.RandomState(12).randint(0, 5, size=(20, 3))
    for action in random_actions:
        for entity, original in zip(adapted.world.entities, source_world.entities):
            np.testing.assert_array_equal(entity.state.p_pos, original.state.p_pos)
            np.testing.assert_array_equal(entity.state.p_vel, original.state.p_vel)
        expected_obs = np.asarray([np.concatenate((source_scenario.observation(a, source_world), np.eye(3)[i]))
                                   for i, a in enumerate(source_world.agents)], dtype=np.float32)
        np.testing.assert_array_equal(adapted.get_obs(), expected_obs)
        for agent, index in zip(source_world.agents, action):
            one_hot = np.eye(5)[index]
            # Source environment.py, default discrete_action_input=False.
            agent.action.u = 5.0 * np.asarray([one_hot[1] - one_hot[2], one_hot[3] - one_hot[4]])
            agent.action.c = np.zeros(source_world.dim_c)
        source_world.step()
        expected_reward = sum(source_scenario.reward(a, source_world) for a in source_world.agents)
        reward, _, _ = adapted.step(action)
        assert reward == expected_reward
