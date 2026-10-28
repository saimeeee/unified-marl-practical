"""Spawn-safe vector environment runner with bounded waits and explicit cleanup.

The environment-message collection follows PyMARL ParallelRunner. Learning and
batched temporal collection are shared with EpisodeRunner to keep masks aligned.
"""
import multiprocessing as mp
import traceback
from unified_marl.envs import make_env
from .episode_runner import EpisodeRunner


def env_worker(connection, name, kwargs, plugins=()):
    env = None
    try:
        from unified_marl.registry import load_plugins
        load_plugins(plugins)
        env = make_env(name, **kwargs)
        while True:
            command, value = connection.recv()
            if command == "close":
                break
            if command == "info":
                result = env.get_env_info()
            else:
                if command == "reset":
                    env.reset()
                    result = {}
                elif command == "step":
                    reward, done, info = env.step(value)
                    result = {"reward": reward, "done": done, "info": info}
                else:
                    raise ValueError(f"Unknown environment command {command}")
                result.update(obs=env.get_obs(), state=env.get_state(), avail_actions=env.get_avail_actions())
            connection.send((True, result))
    except (EOFError, BrokenPipeError):
        pass
    except BaseException:
        try:
            connection.send((False, traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if env is not None:
            env.close()
        connection.close()


class ParallelRunner(EpisodeRunner):
    def __init__(self, args, logger):
        self.args, self.logger = args, logger
        self.batch_size, self.t_env = args.batch_size_run, 0
        self.connections, self.processes = [], []
        self.closed = False
        ctx = mp.get_context("spawn")
        try:
            for index in range(self.batch_size):
                parent, child = ctx.Pipe()
                kwargs = dict(args.env_args)
                kwargs["seed"] = int(kwargs.get("seed", args.seed)) + index
                process = ctx.Process(target=env_worker, args=(child, args.env, kwargs, getattr(args, "plugins", ())), daemon=True)
                process.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
            for connection in self.connections:
                connection.send(("info", None))
            infos = [self._receive(i) for i in range(self.batch_size)]
            if not all(info == infos[0] for info in infos):
                raise ValueError("Parallel environments have incompatible metadata")
            self.env_info = infos[0]
            self.episode_limit = self.env_info["episode_limit"]
        except BaseException:
            self.close_env()
            raise

    def _receive(self, index):
        connection = self.connections[index]
        if not connection.poll(self.args.worker_timeout):
            raise TimeoutError(f"Environment worker {index} did not respond within {self.args.worker_timeout}s")
        try:
            ok, result = connection.recv()
        except (EOFError, OSError) as exc:
            raise RuntimeError(f"Environment worker {index} exited") from exc
        if not ok:
            raise RuntimeError(f"Environment worker {index} failed:\n{result}")
        return result

    def _request(self, indices, command, values=None):
        for i in indices:
            self.connections[i].send((command, None if values is None else values[i]))
        return [self._receive(i) for i in indices]

    def close_env(self):
        if self.closed:
            return
        self.closed = True
        for connection in self.connections:
            try:
                connection.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        for connection in self.connections:
            connection.close()

    def save_replay(self):
        raise NotImplementedError("Use episode runner for environment replays")
