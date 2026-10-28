"""
Modified from OpenAI Baselines code to work with multi-agent envs
"""
import numpy as np
from multiprocessing import Process, Pipe
from unified_marl.algorithms.hmf.utils.vec_env import VecEnv, CloudpickleWrapper
from functools import partial
from unified_marl.algorithms.hmf.utils.ReplenishmentEnv import ReplenishmentEnv

def env_fn(**kwargs):
    return ReplenishmentEnv(**kwargs)

def env_worker(remote, env_fn):
    # Make environment
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            reward, terminated, env_info = env.step(actions)
            rewards = np.array(reward).repeat(env.n_agents)
            profit = np.array(env.get_profit()).sum()
            env_info["profit"] = profit
            if terminated:
                env.reset()
            terminated = np.array(terminated).repeat(env.n_agents)
            state = env.get_state()
            obs = env.get_obs()
            remote.send((state, obs, rewards, terminated, env_info))
        elif cmd == "reset":
            env.reset()
            state = env.get_state()
            obs = env.get_obs()
            remote.send((state, obs))
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        elif cmd == "switch_mode":
            mode = data
            env.switch_mode(mode)
        elif cmd == "get_profit":
            remote.send(env.get_profit())
        elif cmd == "get_reward_dict":
            remote.send(env._env.reward_monitor)
        elif cmd == "visualize_render":
            env.visualize_render(data)
        elif cmd == "get_storage_capacity":
            remote.send(env._env.storage_capacity)
        elif cmd == "set_storage_capacity":
            env.set_storage_capacity(data)
        elif cmd == 'get_spaces':
            remote.send((env._env.observation_space, env._env.action_space))
        else:
            raise NotImplementedError
        
class SubprocVecEnv(VecEnv):
    def __init__(self, env_args):
        self.waiting = False
        self.closed = False
        nenvs = env_args["n_rollout_threads"]
        self.n_agents = env_args["n_agents"]
        env_args = [env_args.copy() for _ in range(nenvs)]
        for i in range(len(env_args)):
            env_args[i]["seed"] += i
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=env_worker, args=(work_remotes, CloudpickleWrapper(partial(env_fn, **env_arg)),))
            for (work_remotes, env_arg) in zip(self.work_remotes, env_args)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()

        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, nenvs, observation_space, action_space)
    
    def step_async(self, actions):
        actions = np.argmax(actions, axis=-1)
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True
        
    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        states, obs, rews, dones, infos = zip(*results)
        return states, np.stack(obs), np.stack(rews), np.stack(dones), infos
    
    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        results = [remote.recv() for remote in self.remotes]
        states, obs = zip(*results)
        return states, np.stack(obs)
    
    def get_profit(self, env_idx):
        self.remotes[env_idx].send(("get_profit", None))
        profit = self.remotes[env_idx].recv()
        return np.array(profit).mean()
    
    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

class DummyVecEnv(VecEnv):
    def __init__(self, env_args):
        nenvs = env_args["n_rollout_threads"]
        env_args = [env_args.copy() for _ in range(nenvs)]
        for i in range(len(env_args)):
            env_args[i]["seed"] += i
        self.envs = [env_fn(**env_arg) for env_arg in env_args]
        env = self.envs[0]
        VecEnv.__init__(self, nenvs, env._env.observation_space, env._env.action_space)
        self.ts = np.zeros(len(self.envs), dtype='int')
        self.actions = None
        
    def step_async(self, actions):
        self.actions = actions    

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        rews, dones, infos = map(np.array, zip(*results))
        self.ts += 1
        for (i, done) in enumerate(dones):
            if all(done):
                self.envs[i].reset()
                self.ts[i] = 0
        obs = [env.get_obs() for env in self.envs]
        states = [env.get_state() for env in self.envs]
        avail_actions = [env.get_avail_actions() for env in self.envs]
        self.actions = None
        return states, np.stack(avail_actions), np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        results = [remote.recv() for remote in self.remotes]
        states, avail_actions, obs = zip(*results)
        return states, np.stack(avail_actions), np.stack(obs)

    def close(self):
        return
