"""PyMARL episode collection, including terminal bootstrap observations/actions."""
import numpy as np
import torch
from unified_marl.components.episode_buffer import EpisodeBatch
from unified_marl.envs import make_env
from unified_marl.registry import get_algorithm_spec


class EpisodeRunner:
    def __init__(self, args, logger):
        if args.batch_size_run != 1:
            raise ValueError("EpisodeRunner collects one environment")
        self.args, self.logger, self.batch_size, self.t_env = args, logger, 1, 0
        self.env = make_env(args.env, **args.env_args)
        try:
            self.env_info = self.env.get_env_info()
            self.episode_limit = self.env_info["episode_limit"]
        except BaseException:
            self.env.close()
            raise

    def _request(self, indices, command, values=None):
        results = []
        for i in indices:
            if command == "reset":
                self.env.reset()
                result = self._observation()
            elif command == "step":
                reward, done, info = self.env.step(values[i])
                result = dict(self._observation(), reward=reward, done=done, info=info)
            else:
                raise ValueError(command)
            results.append(result)
        return results

    def _observation(self):
        return {"obs": self.env.get_obs(), "state": self.env.get_state(), "avail_actions": self.env.get_avail_actions()}

    def get_env_info(self):
        return dict(self.env_info)

    def setup(self, scheme, groups, preprocess, mac):
        self.scheme, self.groups, self.preprocess, self.mac = scheme, groups, preprocess, mac

    def run(self, test_mode=False):
        batch = EpisodeBatch(self.scheme, self.groups, self.batch_size, self.episode_limit + 1,
                             preprocess=self.preprocess, device=self.args.device)
        # A legal dummy action keeps padded policy normalization well-defined.
        batch["avail_actions"][..., 0] = 1
        initial = self._request(list(range(self.batch_size)), "reset")
        batch.update({k: np.asarray([r[k] for r in initial]) for k in ["state", "obs", "avail_actions"]}, ts=0)
        self.mac.init_hidden(self.batch_size)
        active = list(range(self.batch_size))
        need_action = active[:]
        returns, lengths = [0.0] * self.batch_size, [0] * self.batch_size
        infos = [{} for _ in active]
        t = 0
        while need_action:
            with torch.no_grad():
                chosen = self.mac.select_actions(batch, t_ep=t, t_env=self.t_env, bs=need_action, test_mode=test_mode)
                if isinstance(chosen, tuple):
                    actions, roles, role_mask, agent_q = chosen
                    batch.update({"roles": roles.unsqueeze(-1), "role_avail_actions": role_mask, "agent_Q": agent_q},
                                 bs=need_action, ts=t, mark_filled=False)
                else:
                    actions = chosen
                if getattr(self.mac,"rollout_data",None):
                    unknown = set(self.mac.rollout_data) - set(batch.scheme)
                    if unknown and get_algorithm_spec(getattr(self.args, "algorithm", None)):
                        raise ValueError(f"Plugin rollout_data fields missing from scheme_factory: {sorted(unknown)}")
                    batch.update({key:value for key,value in self.mac.rollout_data.items() if key in batch.scheme},bs=need_action,ts=t,mark_filled=False)
                if "mean_actions" in batch.scheme:
                    batch.update({"mean_actions": self.mac.mean_actions[need_action]},
                                 bs=need_action, ts=t, mark_filled=False)
            batch.update({"actions": actions}, bs=need_action, ts=t, mark_filled=False)
            if not active:
                break
            selected = {i: actions[j].detach().cpu().numpy() for j, i in enumerate(need_action) if i in active}
            old_active = active[:]
            results = self._request(old_active, "step", selected)
            rewards, terminated, truncated = [], [], []
            active = []
            for i, result in zip(old_active, results):
                done = bool(result["done"])
                timeout = bool(result["info"].get("episode_limit", False) or result["info"].get("TimeLimit.truncated", False))
                if t + 1 >= self.episode_limit and not done:
                    done, timeout = True, True
                rewards.append((float(result["reward"]),))
                terminated.append((done and not timeout,))
                truncated.append((done and timeout,))
                returns[i] += float(result["reward"])
                lengths[i] += 1
                infos[i] = result["info"]
                if not done:
                    active.append(i)
            batch.update({"reward": rewards, "terminated": terminated, "truncated": truncated}, bs=old_active, ts=t, mark_filled=False)
            if "individual_rewards" in batch.scheme:
                individual=[]
                for result in results:
                    if "individual_rewards" not in result["info"]: raise ValueError("This learner requires actual individual_rewards; team rewards are not copied")
                    value=np.asarray(result["info"]["individual_rewards"],dtype=np.float32)
                    if value.size!=self.env_info["n_agents"]: raise ValueError("Individual reward count differs from number of agents")
                    individual.append(value.reshape(self.env_info["n_agents"],1))
                batch.update({"individual_rewards":np.asarray(individual)},bs=old_active,ts=t,mark_filled=False)
            t += 1
            batch.update({k: np.asarray([r[k] for r in results]) for k in ["state", "obs", "avail_actions"]}, bs=old_active, ts=t)
            need_action = old_active
        if not test_mode:
            self.t_env += sum(lengths)
        self.last_returns, self.last_lengths, self.last_infos = returns, lengths, infos
        self.logger.log_stat(("test_" if test_mode else "") + "return_mean", np.mean(returns), self.t_env)
        return batch[:, :t + 1]

    def close_env(self):
        self.env.close()

    def save_replay(self):
        self.env.save_replay()
