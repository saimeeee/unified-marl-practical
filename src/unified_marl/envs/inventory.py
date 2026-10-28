"""Parameterised inventory simulator; an explicit new backend, not ReplenishmentEnv.

One agent is one (warehouse, SKU). Order at the start of a period; receive
orders due this period, satisfy demand, then charge holding/backlog costs.
No vehicle routing or warehouse-to-warehouse transfer is modelled.
"""
import csv
from pathlib import Path
import numpy as np
from .multiagentenv import MultiAgentEnv

DEFAULT_MULTIPLIERS = [0, .16, .33, .4, .45, .5, .55, .6, .66, .83, 1, 1.16,
    1.33, 1.5, 1.66, 1.83, 2, 2.16, 2.33, 2.5, 2.66, 2.83, 3, 3.16, 3.33,
    3.5, 3.66, 3.83, 4, 5, 6, 7, 9, 12]


def read_problem(agents_file, demand_file, horizon, action_multipliers=None):
    with Path(agents_file).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"agent_id", "warehouse_id", "sku_id", "initial_stock", "capacity",
                "mean_demand", "lead_time", "unit_price", "unit_cost", "holding_cost", "shortage_cost"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("agents_file must contain rows with columns: " + ", ".join(sorted(required)))
    ids = [r["agent_id"] for r in rows]
    pairs = [(r["warehouse_id"], r["sku_id"]) for r in rows]
    if len(set(ids)) != len(ids) or any(not i for i in ids) or len(set(pairs)) != len(pairs):
        raise ValueError("Agent IDs and warehouse/SKU pairs must be nonempty and unique")
    numeric = required - {"agent_id", "warehouse_id", "sku_id"}
    for r in rows:
        for k in numeric:
            r[k] = float(r[k])
            if not np.isfinite(r[k]) or r[k] < 0:
                raise ValueError(f"Invalid nonnegative value {k} for {r['agent_id']}")
        if r["capacity"] <= 0 or r["initial_stock"] > r["capacity"] or r["mean_demand"] <= 0:
            raise ValueError("capacity/mean_demand must be positive and stock must fit capacity")
        if int(r["lead_time"]) != r["lead_time"]:
            raise ValueError("lead_time must be an integer number of periods")
        r["lead_time"] = int(r["lead_time"])
    if type(horizon) is not int or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    demands = np.full((horizon, len(rows)), np.nan)
    with Path(demand_file).open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            idx = ids.index(r["agent_id"]) if r["agent_id"] in ids else -1
            period = int(r["period"])
            demand = float(r["demand"])
            if idx < 0 or period < 0 or period >= horizon or not np.isfinite(demand) or demand < 0:
                raise ValueError("Invalid demand row: periods must be 0..horizon-1 and agent IDs must match")
            if np.isfinite(demands[period, idx]):
                raise ValueError("Duplicate demand for period/agent")
            demands[period, idx] = demand
    if not np.isfinite(demands).all():
        raise ValueError("demand_file must contain each agent for every period in horizon")
    multipliers = np.asarray(DEFAULT_MULTIPLIERS if action_multipliers is None else action_multipliers, dtype=float)
    if multipliers.ndim != 1 or len(multipliers) < 2 or not np.isfinite(multipliers).all() or np.any(multipliers < 0):
        raise ValueError("action_multipliers needs at least two finite nonnegative values")
    if len(set(multipliers.tolist())) != len(multipliers):
        raise ValueError("action_multipliers must be unique")
    return rows, demands, multipliers


class InventoryEnv(MultiAgentEnv):
    def __init__(self, agents_file, demand_file, horizon=30, action_multipliers=None,
                 backlog=False, reward_scale=1.0, seed=1):
        self.rows, self.demands, self.multipliers = read_problem(agents_file, demand_file, horizon, action_multipliers)
        if not isinstance(backlog, bool): raise ValueError("backlog must be true or false")
        self.agent_ids = [r["agent_id"] for r in self.rows]
        self.n_agents, self.n_actions = len(self.rows), len(self.multipliers)
        self.episode_limit = horizon
        self.backlog_mode = backlog
        self.reward_scale = float(reward_scale)
        if not np.isfinite(self.reward_scale) or self.reward_scale <= 0: raise ValueError("reward_scale must be finite and positive")
        self.seed_value, self.closed = seed, False
        self.capacity = np.array([r["capacity"] for r in self.rows])
        self.mean_demand = np.array([r["mean_demand"] for r in self.rows])
        self.reset()

    def reset(self):
        self.t = 0
        self.stock = np.array([r["initial_stock"] for r in self.rows])
        self.backlog = np.zeros(self.n_agents)
        self.last_demand = np.zeros(self.n_agents)
        self.pipeline = [[] for _ in self.rows]
        self.totals = dict(profit=0., revenue=0., procurement_cost=0., holding_cost=0., shortage_cost=0., unserved_units=0., sales=0.)
        return self.get_obs(), self.get_state()

    def get_obs(self):
        # All pipeline buckets exposed: the state does not hide lead-time dynamics.
        width = max(r["lead_time"] for r in self.rows) + 1
        pipeline = np.zeros((self.n_agents, width))
        for i, entries in enumerate(self.pipeline):
            for due, quantity in entries:
                pipeline[i, max(0, min(width-1, due-self.t))] += quantity / self.capacity[i]
        core = np.stack([self.stock/self.capacity, self.backlog/self.capacity,
            self.mean_demand/self.capacity, self.last_demand/self.capacity,
            np.full(self.n_agents, self.t/self.episode_limit)], axis=1)
        return np.concatenate([core, pipeline], axis=1).astype(np.float32)

    def get_obs_agent(self, agent_id): return self.get_obs()[agent_id]
    def get_state(self): return self.get_obs().reshape(-1)
    def get_obs_size(self): return 6 + max(r["lead_time"] for r in self.rows)
    def get_state_size(self): return self.n_agents * self.get_obs_size()
    def get_total_actions(self): return self.n_actions
    def get_avail_actions(self): return np.ones((self.n_agents, self.n_actions), dtype=np.int32)
    def get_avail_agent_actions(self, agent_id): return self.get_avail_actions()[agent_id]
    def get_stats(self): return dict(self.totals)
    def close(self): self.closed = True
    def save_replay(self): raise NotImplementedError("Use application rollout CSV export")

    def step(self, actions):
        if self.t >= self.episode_limit: raise RuntimeError("Episode finished; reset before step")
        a = np.asarray(actions)
        if a.shape != (self.n_agents,) or not np.issubdtype(a.dtype, np.integer) or np.any(a < 0) or np.any(a >= self.n_actions):
            raise ValueError("Expected one valid integer action index per agent")
        individual = []
        decisions = []
        for i, r in enumerate(self.rows):
            requested = float(self.multipliers[a[i]] * r["mean_demand"])
            # Capacity reservation includes outstanding orders; clipping is explicit business behaviour.
            outstanding = sum(q for _, q in self.pipeline[i])
            order = max(0., min(requested, r["capacity"] - self.stock[i] - outstanding))
            self.pipeline[i].append((self.t + r["lead_time"], order))
            arrivals = sum(q for due, q in self.pipeline[i] if due <= self.t)
            self.pipeline[i] = [(due, q) for due, q in self.pipeline[i] if due > self.t]
            self.stock[i] += arrivals
            demand = self.demands[self.t, i]
            need = demand + (self.backlog[i] if self.backlog_mode else 0)
            sales = min(self.stock[i], need)
            self.stock[i] -= sales
            unserved = need-sales
            self.backlog[i] = unserved if self.backlog_mode else 0.
            revenue, procurement = sales*r["unit_price"], order*r["unit_cost"]
            holding, shortage = self.stock[i]*r["holding_cost"], unserved*r["shortage_cost"]
            profit = revenue-procurement-holding-shortage
            individual.append(profit/self.reward_scale)
            for key,value in dict(profit=profit,revenue=revenue,procurement_cost=procurement,
                                  holding_cost=holding,shortage_cost=shortage,unserved_units=unserved,sales=sales).items():
                self.totals[key] += value
            decisions.append(dict(agent_id=r["agent_id"],warehouse_id=r["warehouse_id"],sku_id=r["sku_id"],
                action_index=int(a[i]),requested_order=requested,executed_order=order,arrivals=arrivals,
                demand=float(demand),sales=sales,ending_stock=float(self.stock[i]),backlog=float(self.backlog[i]),profit=profit))
        self.last_demand = self.demands[self.t].copy()
        self.t += 1
        done = self.t == self.episode_limit
        info = dict(self.totals,individual_rewards=np.array(individual),decisions=decisions)
        if done: info["episode_limit"] = True
        return float(sum(individual)), done, info
