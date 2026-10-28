"""Versioned tensor/primitive checkpoints, separate policy and training state."""
from pathlib import Path
import random
import numpy as np
import torch

FORMAT_VERSION = 1
NETWORKS = ("mixer", "target_mixer", "critic", "target_critic", "role_mixer", "target_role_mixer", "mf_critic", "target_mf_critic", "attention")
OPTIMIZERS = ("optimiser", "agent_optimiser", "critic_optimiser", "action_encoder_optimiser")
COUNTERS = ("last_target_update_episode", "last_target_update_step", "critic_training_steps", "log_stats_t", "role_action_spaces_updated")


def controller_state(mac):
    return mac.state_dict() if hasattr(mac, "state_dict") else {"agent": mac.agent.state_dict()}


def restore_controller(mac, state):
    if hasattr(mac, "load_state_dict"):
        mac.load_state_dict(state)
    else:
        mac.agent.load_state_dict(state["agent"])


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)
    return str(path.resolve())


def read_checkpoint(path, device="cpu"):
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if payload.get("format_version") != FORMAT_VERSION or payload.get("kind") not in {"policy", "training"}:
        raise ValueError("Unsupported unified_marl checkpoint format")
    return payload


def learner_state(learner):
    if hasattr(learner, "checkpoint_state"):
        if not hasattr(learner, "load_checkpoint_state"):
            raise ValueError("A custom checkpoint_state requires load_checkpoint_state")
        return {"plugin_training_state": learner.checkpoint_state()}
    state = {"agent": learner.mac.agent.state_dict(), "controller": controller_state(learner.mac)}
    if hasattr(learner, "target_mac"):
        state["target_agent"] = learner.target_mac.agent.state_dict()
        state["target_controller"] = controller_state(learner.target_mac)
    for name in NETWORKS + OPTIMIZERS:
        value = getattr(learner, name, None)
        if value is not None:
            state[name] = value.state_dict()
    state["counters"] = {name: getattr(learner, name) for name in COUNTERS if hasattr(learner, name)}
    state["selector_epsilon"] = float(learner.mac.action_selector.epsilon)
    if hasattr(learner, "extra_state"):
        if not hasattr(learner, "load_extra_state"):
            raise ValueError("A custom extra_state requires load_extra_state")
        state["extra"] = learner.extra_state()
    return state


def restore_learner(learner, state):
    if "plugin_training_state" in state:
        if not hasattr(learner, "load_checkpoint_state"):
            raise ValueError("Learner cannot restore its custom training state")
        learner.load_checkpoint_state(state["plugin_training_state"])
        return
    restore_controller(learner.mac, state.get("controller", {"agent": state["agent"]}))
    if "target_agent" in state:
        restore_controller(learner.target_mac, state.get("target_controller", {"agent": state["target_agent"]}))
    for name in NETWORKS + OPTIMIZERS:
        if name in state:
            getattr(learner, name).load_state_dict(state[name])
    for name, value in state["counters"].items():
        setattr(learner, name, value)
    learner.mac.action_selector.epsilon = state["selector_epsilon"]
    if "extra" in state:
        learner.load_extra_state(state["extra"])


def rng_state():
    n = np.random.get_state()
    return {"python": random.getstate(), "numpy": [n[0], n[1].tolist(), n[2], n[3], n[4]],
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def replay_state(buffer):
    if buffer is None:
        return None
    return {"buffer_index": buffer.buffer_index, "episodes_in_buffer": buffer.episodes_in_buffer,
            "transition_data": buffer.data.transition_data, "episode_data": buffer.data.episode_data}


def restore_replay(buffer, state):
    if state is None:
        return
    for kind in ["transition_data", "episode_data"]:
        for name, tensor in state[kind].items():
            getattr(buffer.data, kind)[name].copy_(tensor)
    buffer.buffer_index, buffer.episodes_in_buffer = state["buffer_index"], state["episodes_in_buffer"]


def save_checkpoints(directory, config, env_info, learner, buffer, progress):
    base = {"format_version": FORMAT_VERSION, "config": config, "env_info": env_info,
            "agent": learner.mac.agent.state_dict(), "controller": controller_state(learner.mac)}
    directory = Path(directory)
    policy = atomic_save(dict(base, kind="policy"), directory / "policy.pt")
    training = atomic_save(dict(base, kind="training", learner=learner_state(learner), replay=replay_state(buffer),
                                progress=progress, rng=rng_state(),
                                resume_scope="episode boundary; learner, replay and global RNG; external simulator internal state is not captured"),
                           directory / "training.pt")
    return policy, training
