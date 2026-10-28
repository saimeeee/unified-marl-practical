from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Mapping
import numpy as np
import torch

ALIASES = {"action": "actions", "avail_action": "avail_actions", "available_actions": "avail_actions", "rewards": "reward"}


class DataBatch(dict):
    """Dictionary compatibility plus EpisodeBatch dimension attributes.

    Samples own their tensor storage: native learners may mutate masks without
    modifying the file-backed dataset or another sample.
    """
    @property
    def batch_size(self): return self["obs"].shape[0]
    @property
    def max_seq_length(self): return self["obs"].shape[1]
    @property
    def device(self): return self["obs"].device
    @property
    def transition_mask(self): return valid_mask(self)[:,:-1]

    def __getitem__(self, key):
        if key == "batch_size": return self.batch_size
        if key == "max_seq_length": return torch.tensor(self.max_seq_length, device=self.device)
        return super().__getitem__(key)

    def to(self, device):
        return DataBatch({k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in self.items()})


def valid_mask(batch):
    # A terminal transition itself is valid; following observations/transitions
    # are excluded. filled retains the delivered occupied-slot convention.
    filled = batch["filled"].float()
    ended = batch["terminated"].bool()
    previous = torch.cat([torch.zeros_like(ended[:, :1]), ended[:, :-1]], 1)
    return filled * (previous.cumsum(1) == 0)


@dataclass
class PreferenceBatch:
    trajectory0: DataBatch
    trajectory1: DataBatch
    labels: torch.Tensor
    mask0: torch.Tensor
    mask1: torch.Tensor
    indices0: torch.Tensor
    indices1: torch.Tensor
    label_key: str = "labels"
    label_direction: str = "1 prefers trajectory1; 0 prefers trajectory0; 0.5 is a tie"
    extra_labels: dict = field(default_factory=dict)

    @property
    def lengths0(self): return self.mask0.sum(1).squeeze(-1).long()
    @property
    def lengths1(self): return self.mask1.sum(1).squeeze(-1).long()

    def to(self, device):
        return PreferenceBatch(self.trajectory0.to(device), self.trajectory1.to(device), self.labels.to(device),
            self.mask0.to(device), self.mask1.to(device), self.indices0.to(device), self.indices1.to(device),
            self.label_key, self.label_direction, {k: v.to(device) for k, v in self.extra_labels.items()})


def read_tensors(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file(): raise FileNotFoundError(path)
    # Tensor-only loading, memory-mapped on CPU. No fallback to executable pickle.
    return torch.load(path, weights_only=True, mmap=True, map_location="cpu")


class EpisodeDataset:
    def __init__(self, source, *, reward_semantics="team", labels_path=None, agent_reward_path=None,
                 pair_indices=None, pairing="halves", label_key="labels", max_length=None):
        raw = read_tensors(source) if isinstance(source, (str, Path)) else source
        if not isinstance(raw, Mapping): raise ValueError("Episode dataset must be a tensor mapping")
        self.source = str(Path(source).resolve()) if isinstance(source, (str, Path)) else "in-memory"
        self.data = {}
        for key, value in raw.items():
            name = ALIASES.get(key, key)
            if name in self.data:
                if not torch.equal(torch.as_tensor(value), self.data[name]):
                    raise ValueError(f"Conflicting alias fields for {name}")
            self.data[name] = torch.as_tensor(value)
        if labels_path is not None:
            labels = read_tensors(labels_path)
            if not isinstance(labels, Mapping): raise ValueError("Label sidecar must name its label fields")
            for key, value in labels.items():
                if "label" not in key: raise ValueError("Label sidecar contains a non-label field")
                self.data[key] = value
        self.reward_semantics, self.label_key, self.max_length = reward_semantics, label_key, max_length
        required = {"obs", "state", "actions", "avail_actions", "filled", "terminated"}
        missing = required - self.data.keys()
        if missing: raise ValueError(f"Missing episode fields: {sorted(missing)}")
        obs = self.data["obs"]
        if obs.ndim != 4: raise ValueError("obs must have shape [episodes,time,agents,features]")
        self.episodes, self.time, self.n_agents, self.obs_shape = obs.shape
        avail = self.data["avail_actions"]
        if avail.ndim != 4 or tuple(avail.shape[:3]) != tuple(obs.shape[:3]):
            raise ValueError("avail_actions must have shape [E,T,N,A]")
        self.n_actions = avail.shape[-1]
        if self.data["actions"].is_floating_point() and not torch.equal(self.data["actions"],self.data["actions"].round()):
            raise ValueError("Discrete actions must be integer-valued; refusing lossy conversion")
        expected = {"actions": (self.episodes,self.time,self.n_agents,1), "filled": (self.episodes,self.time,1),
                    "terminated": (self.episodes,self.time,1)}
        for key, shape in expected.items():
            if tuple(self.data[key].shape) != shape: raise ValueError(f"{key} must have shape {shape}")
        if self.data["state"].ndim != 3 or self.data["state"].shape[:2] != obs.shape[:2]:
            raise ValueError("state must have shape [E,T,S]")
        self.state_shape = self.data["state"].shape[-1]
        for key in ["filled", "terminated", "avail_actions"]:
            value = self.data[key]
            if not torch.all((value == 0) | (value == 1)): raise ValueError(f"{key} must be binary")
        if self.time < 2 or not self.episodes: raise ValueError("Episodes need at least two time slots")
        if torch.any(self.data["filled"][:,1:] > self.data["filled"][:,:-1]):
            raise ValueError("filled must be a contiguous prefix, including any recorded bootstrap state")
        if reward_semantics not in {"team", "individual", "none"}: raise ValueError("Explicit reward_semantics must be team, individual, or none")
        if "reward" in self.data:
            expected_reward = (self.episodes,self.time,1) if reward_semantics == "team" else (self.episodes,self.time,self.n_agents,1)
            if reward_semantics != "none" and tuple(self.data["reward"].shape) != expected_reward:
                raise ValueError(f"reward with {reward_semantics} semantics must have shape {expected_reward}; no implicit sum/copy is performed")
        elif reward_semantics != "none":
            raise ValueError("Dataset is missing the declared reward field")
        if agent_reward_path is not None:
            value = read_tensors(agent_reward_path)
            if isinstance(value, Mapping): value = value["agent_reward"]
            self.data["agent_reward"] = value
        if "agent_reward" in self.data:
            r = self.data["agent_reward"]
            if r.ndim != 4 or r.shape[0] != self.episodes or tuple(r.shape[2:]) != (self.n_agents,1) or r.shape[1] not in {self.time,self.time-1}:
                raise ValueError("agent_reward must have shape [E,T or T-1,N,1]")
        self.pairs = None
        if pair_indices is not None:
            self.pairs = torch.as_tensor(pair_indices, dtype=torch.long)
            if self.pairs.ndim != 2 or self.pairs.shape[1] != 2: raise ValueError("pair_indices must have shape [pairs,2]")
            if self.pairs.min() < 0 or self.pairs.max() >= self.episodes: raise ValueError("Pair index outside dataset")
        elif pairing == "halves" and self.episodes % 2 == 0:
            first = torch.arange(self.episodes // 2)
            self.pairs = torch.stack([first, first + self.episodes // 2], -1)
        elif pairing == "halves" and not any("label" in k for k in self.data):
            self.pairs = None  # Ordinary offline data can have an odd row count.
        elif pairing != "none":
            raise ValueError("Halves pairing needs an even episode count; supply explicit pairs or pairing='none'")
        if self.pairs is not None:
            pairs=len(self.pairs)
            for name in ["labels","global_labels","between_labels"]:
                if name in self.data and self.data[name].shape[0] != pairs:
                    raise ValueError(f"{name} count must match the explicit preference pairs")
            if "in_labels" in self.data:
                ranks=self.data["in_labels"]
                if tuple(ranks.shape)!=(2,pairs,self.n_agents): raise ValueError("in_labels must be [2,pairs,N] ordered agent permutations")
                expected=torch.arange(1,self.n_agents+1,dtype=ranks.dtype).expand_as(ranks)
                if not torch.equal(ranks.sort(-1).values,expected): raise ValueError("Each in_labels row must be a permutation of agent IDs 1..N")

    def __len__(self): return self.episodes

    @property
    def env_info(self):
        return dict(n_agents=self.n_agents, n_actions=self.n_actions, obs_shape=self.obs_shape,
                    state_shape=self.state_shape, episode_limit=self.time-1)

    def batch(self, indices, *, trim=True):
        indices = torch.as_tensor(indices, dtype=torch.long)
        maximum = int(self.data["filled"][indices].sum(1).max()) if trim else self.time
        maximum = max(2, min(maximum, self.max_length or self.time))
        fields = {}
        excluded = {k for k in self.data if "label" in k}
        for key, value in self.data.items():
            if key in excluded or value.ndim == 0 or value.shape[0] != self.episodes: continue
            selected = value[indices].clone()
            if value.ndim >= 3 and value.shape[1] == self.time:
                selected = selected[:, :maximum]
            # T-1 individual rewards never acquire a fabricated bootstrap reward.
            elif key == "agent_reward":
                selected = selected[:, :min(maximum, value.shape[1])]
            fields[key] = selected
        batch = DataBatch(fields)
        batch["actions"] = batch["actions"].long()
        if batch["actions"].min() < 0 or batch["actions"].max() >= self.n_actions: raise ValueError("Action index outside action vocabulary")
        batch["actions_onehot"] = torch.nn.functional.one_hot(batch["actions"].squeeze(-1), self.n_actions).float()
        mask = valid_mask(batch)
        legal = batch["avail_actions"].gather(-1, batch["actions"])
        if torch.any((legal == 0) & mask.unsqueeze(-2).bool()): raise ValueError("Dataset has unavailable chosen actions in valid time slots")
        if not torch.isfinite(batch["obs"]).all() or not torch.isfinite(batch["state"]).all(): raise ValueError("Non-finite input data")
        return batch

    def preference(self, pair_ids, *, trim=True, label_key=None):
        if self.pairs is None: raise ValueError("Preference training requires explicit or halves pairing")
        key = label_key or self.label_key
        if key not in self.data: raise ValueError(f"Missing requested preference label field {key!r}")
        ids = torch.as_tensor(pair_ids, dtype=torch.long)
        pair = self.pairs[ids]
        labels = self.data[key][ids].clone().float()
        if labels.shape != (len(ids),1) or not torch.isin(labels, torch.tensor([0.0,0.5,1.0])).all():
            raise ValueError("Policy preference labels must be scalar 0/0.5/1; rankings and per-agent labels remain separate")
        a, b = self.batch(pair[:,0], trim=trim), self.batch(pair[:,1], trim=trim)
        extras = {}
        for name in ["global_labels", "between_labels"]:
            if name in self.data: extras[name] = self.data[name][ids].clone()
        if "in_labels" in self.data: extras["in_labels"] = self.data["in_labels"][:,ids].clone()
        return PreferenceBatch(a,b,labels,valid_mask(a),valid_mask(b),pair[:,0],pair[:,1],key,extra_labels=extras)


def load_dataset(source, **kwargs):
    return source if isinstance(source, EpisodeDataset) else EpisodeDataset(source, **kwargs)
