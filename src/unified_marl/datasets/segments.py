"""Pure delivered random segment selection; no labels/services are invoked."""
import torch

def clip_to_segment_length(obs, state, action, reward, avail_action, filled, terminated, agent_ids, lengths, segment_length):
    """将每个样本裁剪到固定的片段长度。
    Args:
        obs (torch.Tensor): 形状为 (num_samples, seq_len, num_agents, obs_dim) 的张量。
        state (torch.Tensor): 形状为 (num_samples, seq_len, state_dim) 的张量。
        action (torch.Tensor): 形状为 (num_samples, seq_len, num_agents, 1) 的张量。
        reward (torch.Tensor): 形状为 (num_samples, seq_len, 1) 的张量。
        avail_action (torch.Tensor): 形状为 (num_samples, seq_len, num_agents, num_actions) 的张量。
        filled (torch.Tensor): 形状为 (num_samples, seq_len, 1) 的张量。
        terminated (torch.Tensor): 形状为 (num_samples, seq_len, 1) 的张量。
        agent_ids (torch.Tensor): 形状为 (num_samples, num_agents) 的张量。
        segment_length (int): 固定的片段长度。
    Returns:
        tuple: 裁剪后的 obs, state, action, reward, avail_action, filled, terminated, agent_ids
    """
    num_samples = obs.shape[0]
    clipped_obs, clipped_state, clipped_action = [], [], []
    clipped_reward, clipped_avail_action, clipped_filled = [], [], []
    clipped_terminated, clipped_agent_ids = [], []

    for i in range(num_samples):
        length = lengths[i].item()
        if length < segment_length:
            start_idx = 0
            end_idx = segment_length
        else:
            start_idx = torch.randint(0, length - segment_length + 1, (1,)).item()
            end_idx = start_idx + segment_length

        clipped_obs.append(obs[i, start_idx:end_idx])
        clipped_state.append(state[i, start_idx:end_idx])
        clipped_action.append(action[i, start_idx:end_idx])
        clipped_reward.append(reward[i, start_idx:end_idx])
        clipped_avail_action.append(avail_action[i, start_idx:end_idx])
        clipped_filled.append(filled[i, start_idx:end_idx])
        clipped_terminated.append(terminated[i, start_idx:end_idx])
        clipped_agent_ids.append(agent_ids[i])

    return (torch.stack(clipped_obs), torch.stack(clipped_state), torch.stack(clipped_action),
            torch.stack(clipped_reward), torch.stack(clipped_avail_action),
            torch.stack(clipped_filled), torch.stack(clipped_terminated),
            torch.stack(clipped_agent_ids))
