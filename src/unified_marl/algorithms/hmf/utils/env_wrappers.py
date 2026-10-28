"""
Modified from OpenAI Baselines code to work with multi-agent envs
"""
import numpy as np
from multiprocessing import Process, Pipe
from vec_env import VecEnv, CloudpickleWrapper
from unified_marl.algorithms.hmf.utils.create_map import create_map
import random


def worker(remote, parent_remote, env_fn_wrapper, agent_pos_list, landmark_pos_list, evader_pos_list, num_map):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            n = None
            if all(done):
                n = random.randint(0, num_map - 1)
                agent_pos, landmark_pos, evader_pos = None, None, None
                if agent_pos_list is not None: agent_pos = agent_pos_list[n]
                if landmark_pos_list is not None : landmark_pos = landmark_pos_list[n]
                if evader_pos_list is not None: evader_pos = evader_pos_list[n]
                ob = env.reset(agent_pos=agent_pos, landmark_pos=landmark_pos, evader_pos=evader_pos)
            remote.send((ob, reward, done, info, n))
        elif cmd == 'reset':
            ob = env.reset(**data)
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        elif cmd == 'get_agent_types':
            if all([hasattr(a, 'adversary') for a in env.agents]):
                remote.send(['adversary' if a.adversary else 'agent' for a in
                             env.agents])
            else:
                remote.send(['agent' for _ in env.agents])
        else:
            raise NotImplementedError


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None, config=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        agent_pos_list, landmark_pos_list, evader_pos_list = create_map(config)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn),
                                                agent_pos_list, landmark_pos_list, evader_pos_list, config.num_map))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        self.remotes[0].send(('get_agent_types', None))
        self.agent_types = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos, n = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos, n

    def reset(self, agent_pos=None, landmark_pos=None, evader_pos=None):
        # pos = {"agent_pos": agent_pos, "landmark_pos": landmark_pos, "evader_pos": evader_pos}
        for i, remote in enumerate(self.remotes):
            a_pos, l_pos, e_pos = None, None, None
            if agent_pos is not None: a_pos = agent_pos[i]
            if landmark_pos is not None: l_pos = landmark_pos[i]
            if evader_pos is not None: e_pos = evader_pos[i]
            pos = {"agent_pos": a_pos, "landmark_pos": l_pos, "evader_pos": e_pos}
            remote.send(('reset', pos))
        results = [remote.recv() for remote in self.remotes]
        return np.stack(results)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

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
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        if all([hasattr(a, 'adversary') for a in env.agents]):
            self.agent_types = ['adversary' if a.adversary else 'agent' for a in
                                env.agents]
        else:
            self.agent_types = ['agent' for _ in env.agents]
        self.ts = np.zeros(len(self.envs), dtype='int')
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, rews, dones, infos = map(np.array, zip(*results))
        self.ts += 1
        for (i, done) in enumerate(dones):
            if all(done):
                obs[i] = self.envs[i].reset()
                self.ts[i] = 0
        self.actions = None
        return np.array(obs), np.array(rews), np.array(dones), infos

    def reset(self):
        results = [env.reset() for env in self.envs]
        return np.array(results)

    def close(self):
        return

    def render(self):
        self.envs[0].render()
