"""Lazy SADMA CityFlow backend bridge; assets and engine stay external."""
from .multiagentenv import MultiAgentEnv


class CityFlowAdapter(MultiAgentEnv):
    def __init__(self,**kwargs):
        try:
            from cityflowenv.env import CityflowEnv
        except ImportError as exc:
            raise ImportError("SADMA CityFlow needs cityflowenv.env.CityflowEnv, the CityFlow engine, traffic maps and flows; none are bundled") from exc
        self.backend=CityflowEnv(**kwargs)
        self.closed=False
        try:
            self.n_agents=int(self.backend.n_agents)
            self.episode_limit=int(self.backend.episode_limit)
            self.backend.reset()
        except BaseException:
            self.close()
            raise
    def reset(self): return self.backend.reset()
    def step(self,actions): return self.backend.step(actions)
    def get_obs(self): return self.backend.get_obs()
    def get_state(self): return self.backend.get_state()
    def get_obs_size(self): return self.backend.get_obs_size()
    def get_state_size(self): return self.backend.get_state_size()
    def get_total_actions(self): return self.backend.get_total_actions()
    def get_avail_actions(self): return self.backend.get_avail_actions()
    def get_stats(self):
        return {"avg_travel_time":self.backend.get_avg_travel_time()} if hasattr(self.backend,"get_avg_travel_time") else {}
    def close(self):
        if not self.closed:
            self.closed=True
            close=getattr(self.backend,"close",None)
            if close is not None: close()
    def save_replay(self): raise NotImplementedError("Configure the external CityFlow backend replay output explicitly")
