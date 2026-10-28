import copy
from torch.distributions import Categorical
from unified_marl.components.episode_buffer import EpisodeBatch
import torch as th
from torch.optim import RMSprop
import torch.nn.functional as F
import numpy as np

class BCMAPTLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger
        self.T = args.T
        self.lamda = args.lamda

        self.params = list(mac.parameters())

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        self.log_stats_t = -self.args.learner_log_interval - 1


    def train(self, batch0, batch1, t_env, running_log):
        # # ----------- batch0 preferred -----------------
        states0 = batch0["state"]
        actions0 = batch0["actions"]
        agent_rewards0 = batch0["agent_reward"]
        terminated0 = batch0["terminated"].float()
        avail_actions0 = batch0["avail_actions"].long()

        mask0 = batch0["filled"].float()
        mask0[:, 1:] = mask0[:, 1:] * (1 - terminated0[:, :-1])
        mask0 = mask0.unsqueeze(-2).repeat(1, 1, self.n_agents, 1)

        # Calculate estimated Q-Values
        mac_out0 = []
        self.mac.init_hidden(batch0['batch_size'])
        for t in range(batch0['max_seq_length']):
            if self.args.env == "sc2":
                agent_outs = self.mac.forward(batch0, t=t)
            else:
                pi_distribution , _ = self.mac.forward(batch0, t=t)
                agent_outs = self.get_logprob(pi_distribution, actions0[:,t])
            mac_out0.append(agent_outs)
        mac_out0 = th.stack(mac_out0, dim=1)

        mac_out0[avail_actions0 == 0] = 0
        mac_out0 = mac_out0/mac_out0.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        mac_out0[avail_actions0 == 0] = 0

        # # ----------- batch1 not preferred -----------------
        states1 = batch1["state"]
        actions1 = batch1["actions"]
        agent_rewards1 = batch1["agent_reward"]
        terminated1 = batch1["terminated"].float()
        avail_actions1 = batch1["avail_actions"].long()

        mask1 = batch1["filled"].float()
        mask1[:, 1:] = mask1[:, 1:] * (1 - terminated1[:, :-1])
        mask1 = mask1.unsqueeze(-2).repeat(1, 1, self.n_agents, 1)

        # Calculate estimated Q-Values
        mac_out1 = []
        self.mac.init_hidden(batch1['batch_size'])
        for t in range(batch1['max_seq_length']):
            if self.args.env == "sc2":
                agent_outs = self.mac.forward(batch1, t=t)
            else:
                pi_distribution , _ = self.mac.forward(batch1, t=t)
                agent_outs = self.get_logprob(pi_distribution, actions1[:,t])
            mac_out1.append(agent_outs)
        mac_out1 = th.stack(mac_out1, dim=1)

        mac_out1[avail_actions1 == 0] = 0
        mac_out1 = mac_out1/mac_out1.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        mac_out1[avail_actions1 == 0] = 0

        # pi_taken0 = mac_out0
        pi_taken0 = th.gather(mac_out0, dim=-1, index=actions0)
        pi_taken0[mask0 == 0] = 1.0
        log_pi_taken0 = th.log(pi_taken0)
        # pi_taken1 = mac_out1
        pi_taken1 = th.gather(mac_out1, dim=-1, index=actions1)
        pi_taken1[mask1 == 0] = 1.0
        log_pi_taken1 = th.log(pi_taken1)

        log_pi_taken0_mask = (log_pi_taken0 * mask0).sum(1) / mask0.sum(1)
        log_pi_taken1_mask = (log_pi_taken1 * mask1).sum(1) / mask1.sum(1)
        # reverse agent-wise weights
        # agent_rewards0 = - agent_rewards0
        # agent-wise weights
        agent_weights0_pos = th.softmax(agent_rewards0.sum(1) / self.T, dim=1)
        agent_weights1_pos = th.softmax(agent_rewards1.sum(1) / self.T, dim=1)
        
        log_pi_taken = th.cat([log_pi_taken0_mask, log_pi_taken1_mask], dim=0)
        
        loss = - (((agent_weights0_pos * log_pi_taken0_mask).sum() + self.lamda * (agent_weights1_pos * log_pi_taken1_mask).sum()) + log_pi_taken.sum())

        self.optimiser.zero_grad()
        # integration: detached loss capture, no change to the optimized expression
        if not th.isfinite(loss).all(): raise FloatingPointError('Non-finite source loss: loss')
        if not hasattr(self, '_integration_losses'): self._integration_losses = {}
        self._integration_losses['loss'] = float(loss.detach())
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            self.log_stats_t = t_env
            running_log.update({
                "loss": loss.item(),
                "agent_grad_norm": grad_norm,
            })

    def get_logprob(self, pi_distribution, actions):
        # Sample action with reparameterization
        logp_pi = pi_distribution.log_prob(actions.reshape(self.args.batch_size*self.n_agents, -1)).sum(axis=-1)
        p_pi = th.exp(logp_pi)
        p_pi = p_pi.reshape(self.args.batch_size,1,self.n_agents,-1)
        return p_pi

    def cuda(self):
        self.mac.cuda()
    
    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
    
    def load_models(self, path):
        self.mac.load_models(path)
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))