from unified_marl.learners.base import LearnerAPI
import copy
from unified_marl.algorithms.rghmf.components.episode_buffer import EpisodeBatch
from unified_marl.algorithms.rghmf.modules.mixers.vdn import VDNMixer
from unified_marl.algorithms.rghmf.modules.mixers.qmix import QMixer
from unified_marl.algorithms.rghmf.modules.base.networks import MLPNetwork, MultiHeadAttention_hie
import torch as th
from torch.optim import RMSprop

import numpy as np

n_heads = 3
embed_dim = n_heads * 64
MSELoss = th.nn.MSELoss()

class RODEHMFLearner(LearnerAPI):
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.n_agents = args.n_agents
        self.n_groups = args.n_roles

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.role_mixer = None
        if args.role_mixer is not None:
            if args.role_mixer == "vdn":
                self.role_mixer = VDNMixer()
            elif args.role_mixer == "qmix":
                self.role_mixer = QMixer(args)
            else:
                raise ValueError("Role Mixer {} not recognised.".format(args.role_mixer))
            self.params += list(self.role_mixer.parameters())
            self.target_role_mixer = copy.deepcopy(self.role_mixer)

        self.mf_critic = MLPNetwork(embed_dim, 1, hidden_dim=args.rnn_hidden_dim, constrain_out=False)
        self.target_mf_critic = copy.deepcopy(self.mf_critic)
        self.attention = MultiHeadAttention_hie(n_heads=n_heads, input_dim=args.obs_shape + args.n_actions, embed_dim=embed_dim)
        
        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.role_interval = args.role_interval
        self.device = self.args.device

        self.role_action_spaces_updated = True

        # action encoder
        self.action_encoder_params = list(self.mac.action_encoder_params())
        self.action_encoder_optimiser = RMSprop(params=self.action_encoder_params, lr=args.lr,
                                                alpha=args.optim_alpha, eps=args.optim_eps)
        
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        # role_avail_actions = batch["role_avail_actions"]
        roles_shape_o = batch["roles"][:, :-1].shape
        role_at = int(np.ceil(roles_shape_o[1] / self.role_interval))
        role_t = role_at * self.role_interval

        roles_shape = list(roles_shape_o)
        roles_shape[1] = role_t
        roles = th.zeros(roles_shape).to(self.device)
        roles[:, :roles_shape_o[1]] = batch["roles"][:, :-1]
        roles = roles.view(batch.batch_size, role_at, self.role_interval, self.n_agents, -1)[:, :, 0]
        # hmf
        agents_group = th.nn.functional.one_hot(batch["roles"][:, :-1].squeeze(), num_classes=self.n_groups).float().permute(0, 1, 3, 2)
        next_agents_group = th.nn.functional.one_hot(batch["roles"][:, 1:].squeeze(), num_classes=self.n_groups).float().permute(0, 1, 3, 2)
        local_info = th.cat((batch["obs"][:, :-1], batch["agent_Q"][:, :-1]), dim=-1)
        target_local_info = th.cat((batch["obs"][:, 1:], batch["agent_Q"][:, 1:]), dim=-1)

        bsz, length, n_agents, info_dim = local_info.shape
        agents_group = th.reshape(agents_group, (bsz * length, self.n_groups, n_agents))
        next_agents_group = th.reshape(agents_group, (bsz * length, self.n_groups, n_agents))
        local_info = th.reshape(local_info, (bsz * length, n_agents, -1))
        target_local_info = th.reshape(target_local_info, (bsz * length, n_agents, -1))
        low_rewards = th.reshape(rewards, (bsz * length, 1, -1)).repeat(1, self.n_groups, 1).to(self.device)
        low_dones = th.reshape(terminated, (bsz * length, 1, -1)).repeat(1, self.n_groups, 1).to(self.device)
        high_rewards = th.reshape(rewards, (bsz * length, 1)).to(self.device)
        high_dones = th.reshape(terminated, (bsz * length, 1)).to(self.device)

        # Calculate estimated Q-Values
        mac_out = []
        role_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs, role_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
            if t % self.role_interval == 0 and t < batch.max_seq_length - 1:
                role_out.append(role_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        role_out = th.stack(role_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        chosen_role_qvals = th.gather(role_out, dim=3, index=roles.long()).squeeze(3)

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        target_role_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs, target_role_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)
            if t % self.role_interval == 0 and t < batch.max_seq_length - 1:
                target_role_out.append(target_role_outs)

        target_role_out.append(th.zeros(batch.batch_size, self.n_agents, self.mac.n_roles).to(self.device))
        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time
        target_role_out = th.stack(target_role_out[1:], dim=1)

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999
        # target_mac_out[role_avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            # mac_out_detach[role_avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)

            role_out_detach = role_out.clone().detach()
            role_out_detach = th.cat([role_out_detach[:, 1:], role_out_detach[:, 0:1]], dim=1)
            cur_max_roles = role_out_detach.max(dim=3, keepdim=True)[1]
            target_role_max_qvals = th.gather(target_role_out, 3, cur_max_roles).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]
            target_role_max_qvals = target_role_out.max(dim=3)[0]

        # hmf low
        mf_info = th.divide(th.matmul(agents_group, local_info), th.sum(agents_group, dim=-1, keepdim=True).repeat(1, 1, info_dim) + 1e-9)
        mf_mask = (agents_group * -1 + 1).type(th.bool)
        mf_info_use_attn, mf_weight = self.attention(h=mf_info, he=local_info, num_groups=self.n_groups, mask=mf_mask, return_attn=True, calculate_V=True)
        mf_info_use_attn = th.reshape(mf_info_use_attn, (bsz * length * self.n_groups, -1))
        mf_Q = self.mf_critic(mf_info_use_attn)
        mf_Q = th.reshape(mf_Q, (bsz * length, self.n_groups, 1))
        group_mask = th.where(th.sum(agents_group, dim=2, keepdim=True) > 0, 1.0, 0.0)
        mf_Q = th.mul(mf_Q, group_mask)
        mf_weight = th.reshape(mf_weight, (n_heads, bsz * length, self.n_groups, 1, n_agents))
        local_Q = th.reshape(chosen_action_qvals, (bsz * length, 1, n_agents, 1)).repeat(1, self.n_groups, 1, 1)
        local_Q_with_weight = th.reshape(th.sum(th.matmul(mf_weight, local_Q), dim=0), (bsz * length, self.n_groups, -1))
        total_Q = local_Q_with_weight + mf_Q * 2

        target_mf_info = th.divide(th.matmul(next_agents_group, target_local_info), th.sum(next_agents_group, dim=-1, keepdim=True).repeat(1, 1, info_dim) + 1e-9)
        next_mf_mask = (next_agents_group * -1 + 1).type(th.bool)
        target_mf_info_use_attn, target_mf_weight = self.attention(h=target_mf_info, he=target_local_info, num_groups=self.n_groups, mask=next_mf_mask, return_attn=True, calculate_V=True)
        target_mf_info_use_attn = th.reshape(target_mf_info_use_attn, (bsz * length * self.n_groups, -1))
        target_mf_Q = self.mf_critic(target_mf_info_use_attn)
        target_mf_Q = th.reshape(target_mf_Q, (bsz * length, self.n_groups, 1))
        next_group_mask = th.where(th.sum(next_agents_group, dim=2, keepdim=True) > 0, 1.0, 0.0)
        target_mf_Q = th.mul(target_mf_Q, next_group_mask)
        target_mf_weight = th.reshape(target_mf_weight, (n_heads, bsz * length, self.n_groups, 1, n_agents))
        target_local_max_Q = th.reshape(target_max_qvals, (bsz * length, 1, n_agents, 1)).repeat(1, self.n_groups, 1, 1)
        target_local_max_Q_with_wieght = th.reshape(th.sum(th.matmul(target_mf_weight, target_local_max_Q), dim=0), (bsz * length, self.n_groups, -1))
        target_total_Q = target_local_max_Q_with_wieght + target_mf_Q * 2
        target_value = (low_rewards + self.args.gamma * target_total_Q * (1 - low_dones))
        # hmf high
        target_high_value = (high_rewards + self.args.gamma * th.sum(target_mf_Q, dim=1) * (1 - high_dones))
        high_value = th.sum(mf_Q, dim=1)
        high_mf_critic_loss = MSELoss(high_value, target_high_value.detach())

        # local_vf_loss = MSELoss(local_Q_with_weight, low_rewards + self.args.gamma * target_local_max_Q_with_wieght * (1 - low_dones))
        # mf_vf_loss = MSELoss(mf_Q,  low_rewards + self.args.gamma * target_mf_Q * (1 - low_dones))
        total_loss = MSELoss(total_Q, target_value.detach())
        total_loss += high_mf_critic_loss * 0.05

        # Mix
        # if self.mixer is not None:
        #     chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
        #     target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])
        if self.role_mixer is not None:
            state_shape_o = batch["state"][:, :-1].shape
            state_shape = list(state_shape_o)
            state_shape[1] = role_t
            role_states = th.zeros(state_shape).to(self.device)
            role_states[:, :state_shape_o[1]] = batch["state"][:, :-1].detach().clone()
            role_states = role_states.view(batch.batch_size, role_at,self.role_interval, -1)[:, :, 0]
            chosen_role_qvals = self.role_mixer(chosen_role_qvals, role_states)
            role_states = th.cat([role_states[:, 1:], role_states[:, 0:1]], dim=1)
            target_role_max_qvals = self.target_role_mixer(target_role_max_qvals, role_states)

        # Calculate 1-step Q-Learning targets
        rewards_shape = list(rewards.shape)
        rewards_shape[1] = role_t
        role_rewards = th.zeros(rewards_shape).to(self.device)
        role_rewards[:, :rewards.shape[1]] = rewards.detach().clone()
        role_rewards = role_rewards.view(batch.batch_size, role_at,
                                         self.role_interval).sum(dim=-1, keepdim=True)
        # role_terminated
        terminated_shape_o = terminated.shape
        terminated_shape = list(terminated_shape_o)
        terminated_shape[1] = role_t
        role_terminated = th.zeros(terminated_shape).to(self.device)
        role_terminated[:, :terminated_shape_o[1]] = terminated.detach().clone()
        role_terminated = role_terminated.view(batch.batch_size, role_at, self.role_interval).sum(dim=-1, keepdim=True)
        # role_terminated
        role_targets = role_rewards + self.args.gamma * (1 - role_terminated) * target_role_max_qvals

        # Td-error
        role_td_error = (chosen_role_qvals - role_targets.detach())

        mask = mask.repeat(1, 1, n_agents)
        mask_shape = list(mask.shape)
        mask_shape[1] = role_t
        role_mask = th.zeros(mask_shape).to(self.device)
        role_mask[:, :mask.shape[1]] = mask.detach().clone()
        role_mask = role_mask.view(batch.batch_size, role_at, self.role_interval, -1)[:, :, 0]

        # 0-out the targets that came from padded data
        # masked_td_error = td_error * mask
        masked_role_td_error = role_td_error * role_mask

        # Normal L2 loss, take mean over actual data
        # loss = (masked_td_error ** 2).sum() / mask.sum()
        role_loss = (masked_role_td_error ** 2).sum() / role_mask.sum()
        total_loss += role_loss

        self.check_finite(total_loss)
        # Optimise
        self.optimiser.zero_grad()
        total_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        pred_obs_loss = None
        pred_r_loss = None
        pred_grad_norm = None
        if self.role_action_spaces_updated:
            # train action encoder
            no_pred = []
            r_pred = []
            for t in range(batch.max_seq_length):
                no_preds, r_preds = self.mac.action_repr_forward(batch, t=t)
                no_pred.append(no_preds)
                r_pred.append(r_preds)
            no_pred = th.stack(no_pred, dim=1)[:, :-1]  # Concat over time
            r_pred = th.stack(r_pred, dim=1)[:, :-1]
            no = batch["obs"][:, 1:].detach().clone()
            repeated_rewards = batch["reward"][:, :-1].detach().clone().unsqueeze(2).repeat(1, 1, self.n_agents, 1)

            pred_obs_loss = th.sqrt(((no_pred - no) ** 2).sum(dim=-1)).mean()
            pred_r_loss = ((r_pred - repeated_rewards) ** 2).mean()

            pred_loss = pred_obs_loss + 10 * pred_r_loss
            self.action_encoder_optimiser.zero_grad()
            pred_loss.backward()
            pred_grad_norm = th.nn.utils.clip_grad_norm_(self.action_encoder_params, self.args.grad_norm_clip)
            self.action_encoder_optimiser.step()

            if t_env > self.args.role_action_spaces_update_start:
                self.mac.update_role_action_spaces()
                if 'noar' in self.args.mac:
                    self.target_mac.role_selector.update_roles(self.mac.n_roles)
                self.role_action_spaces_updated = False
                self._update_targets()
                self.last_target_update_episode = episode_num

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", (total_loss - role_loss).item(), t_env)
            self.logger.log_stat("role_loss", role_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.cpu(), t_env)
            if pred_obs_loss is not None:
                self.logger.log_stat("pred_obs_loss", pred_obs_loss.item(), t_env)
                self.logger.log_stat("pred_r_loss", pred_r_loss.item(), t_env)
                self.logger.log_stat("action_encoder_grad_norm", pred_grad_norm.cpu(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("role_q_taken_mean",
                                 (chosen_role_qvals * role_mask).sum().item() / (role_mask.sum().item() * self.args.n_agents), t_env)
            self.log_stats_t = t_env

        return {'loss': float(total_loss.detach()), 'role_loss': float(role_loss.detach()), 'grad_norm': float(grad_norm),
                'encoder_loss': float((pred_obs_loss + 10 * pred_r_loss).detach()) if pred_obs_loss is not None else 0.0}

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        if self.role_mixer is not None:
            self.target_role_mixer.load_state_dict(self.role_mixer.state_dict())
        self.target_mac.role_action_spaces_updated = self.role_action_spaces_updated
        self.target_mf_critic.load_state_dict(self.mf_critic.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
        if self.role_mixer is not None:
            self.role_mixer.cuda()
            self.target_role_mixer.cuda()
        self.mf_critic.cuda()
        self.target_mf_critic.cuda()
        self.attention.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        if self.role_mixer is not None:
            th.save(self.role_mixer.state_dict(), "{}/role_mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
        th.save(self.action_encoder_optimiser.state_dict(), "{}/action_repr_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        if self.role_mixer is not None:
            self.role_mixer.load_state_dict(
                th.load("{}/role_mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
        self.action_encoder_optimiser.load_state_dict(th.load("{}/action_repr_opt.th".format(path),
                                                              map_location=lambda storage, loc: storage))
