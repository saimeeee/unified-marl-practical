import copy
from torch.distributions import Categorical
from unified_marl.components.episode_buffer import EpisodeBatch
from unified_marl.algorithms.kou.icq.modules.critics.offpg import OffPGCritic
import torch as th
from unified_marl.algorithms.kou.icq.utils.offpg_utils import build_target_q
from unified_marl.algorithms.kou.icq.utils.rl_utils import build_td_lambda_targets
from torch.optim import RMSprop
from unified_marl.algorithms.kou.icq.modules.mixers.qmix import QMixer
import torch.nn.functional as F
import numpy as np
# Unused remote logging import removed; metrics use the package logger.
class MADPOauxLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger
        self.cpl_lambda = args.cpl_lambda
        self.cpl_alpha = args.cpl_alpha
        self.cpl_constrain_coe = args.cpl_constrain_coe
        self.cpl_constrain_type = args.cpl_constrain_type

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.agent_params = list(mac.parameters())

        self.agent_optimiser = RMSprop(params=self.agent_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

    def train(self, batch0, batch1, t_env, labels, running_log):
        # # ----------- batch0 preferred -----------------
        bs = batch0['batch_size']
        actions0 = batch0["actions"]
        terminated0 = batch0["terminated"].float()
        avail_actions0 = batch0["avail_actions"].long()
        agent_rewards0 = batch0["agent_reward"]
        mask0 = batch0["filled"].float()
        mask0[:, 1:] = mask0[:, 1:] * (1 - terminated0[:, :-1])


        # Calculate estimated Q-Values
        mac_out0 = []
        self.mac.init_hidden(batch0['batch_size'])
        for t in range(batch0['max_seq_length']):
            agent_outs = self.mac.forward(batch0, t=t)
            mac_out0.append(agent_outs)
        mac_out0 = th.stack(mac_out0, dim=1)
        
        mac_out0[avail_actions0 == 0] = 0
        mac_out0 = mac_out0/mac_out0.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        mac_out0[avail_actions0 == 0] = 0

        pi_taken0 = th.gather(mac_out0, dim=-1, index=actions0)
        pi_taken0[mask0.repeat(1, 1, self.n_agents).unsqueeze(-1) == 0] = 1.0
        log_pi_taken0 = th.log(pi_taken0).squeeze(-1)

        # # ----------- batch1 not preferred -----------------
        actions1 = batch1["actions"]
        terminated1 = batch1["terminated"].float()
        avail_actions1 = batch1["avail_actions"].long()
        agent_rewards1 = batch1["agent_reward"]
        mask1 = batch1["filled"].float()
        mask1[:, 1:] = mask1[:, 1:] * (1 - terminated1[:, :-1])

        # Calculate estimated Q-Values
        mac_out1 = []
        self.mac.init_hidden(batch1['batch_size'])
        for t in range(batch1['max_seq_length']):
            agent_outs = self.mac.forward(batch1, t=t)
            mac_out1.append(agent_outs)
        mac_out1 = th.stack(mac_out1, dim=1)
        
        mac_out1[avail_actions1 == 0] = 0
        mac_out1 = mac_out1/mac_out1.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        mac_out1[avail_actions1 == 0] = 0

        pi_taken1 = th.gather(mac_out1, dim=-1, index=actions1)
        pi_taken1[mask1.repeat(1, 1, self.n_agents).unsqueeze(-1) == 0] = 1.0
        log_pi_taken1 = th.log(pi_taken1).squeeze(-1)

        # # ------------- CPL loss ------------------
        adv0 = (self.cpl_alpha * log_pi_taken0.mean(-1) * mask0.squeeze(-1)).sum(1) / mask0.sum(1).squeeze(-1)
        adv1 = (self.cpl_alpha * log_pi_taken1.mean(-1) * mask1.squeeze(-1)).sum(1) / mask1.sum(1).squeeze(-1)
        logit10 = adv1 - self.cpl_lambda * adv0
        logit01 = adv0 - self.cpl_lambda * adv1
        max21 = th.clamp(-logit10, min=0, max=None)
        max12 = th.clamp(-logit01, min=0, max=None)
        nlp21 = th.log(th.exp(-max21) + th.exp(-logit10 - max21)) + max21
        nlp12 = th.log(th.exp(-max12) + th.exp(-logit01 - max12)) + max12
        loss = labels.squeeze(-1) * nlp21 + (1 - labels.squeeze(-1)) * nlp12
        # add constrain item
        loss = loss.mean()

        # # ------------- AUX loss ------------------
        adv0 = (self.cpl_alpha * log_pi_taken0 * mask0).sum(1) / mask0.sum(1)
        adv1 = (self.cpl_alpha * log_pi_taken1 * mask1).sum(1) / mask1.sum(1)
        agent_return0 = agent_rewards0.sum(1)
        agent_return1 = agent_rewards1.sum(1)
        aux_loss = []
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                adv0_i = adv0[:,i]
                adv0_j = adv0[:,j]
                logit0_ji = adv0_j - self.cpl_lambda * adv0_i
                logit0_ij = adv0_i - self.cpl_lambda * adv0_j
                maxji = th.clamp(-logit0_ji, min=0, max=None)
                maxij = th.clamp(-logit0_ij, min=0, max=None)
                nlpji = th.log(th.exp(-maxji) + th.exp(-logit0_ji - maxji)) + maxji
                nlpij = th.log(th.exp(-maxij) + th.exp(-logit0_ij - maxij)) + maxij
                label = 0.5 * (agent_return0[:,i] == agent_return0[:,j])
                label += 1.0 * (agent_return0[:,i] < agent_return0[:,j])
                loss0 = label.squeeze(-1) * nlpji + (1 - label.squeeze(-1)) * nlpij

                adv1_i = adv1[:,i]
                adv1_j = adv1[:,j]
                logit1_ji = adv1_j - self.cpl_lambda * adv1_i
                logit1_ij = adv1_i - self.cpl_lambda * adv1_j
                maxji = th.clamp(-logit1_ji, min=0, max=None)
                maxij = th.clamp(-logit1_ij, min=0, max=None)
                nlpji = th.log(th.exp(-maxji) + th.exp(-logit1_ji - maxji)) + maxji
                nlpij = th.log(th.exp(-maxij) + th.exp(-logit1_ij - maxij)) + maxij
                label = 0.5 * (agent_return1[:,i] == agent_return1[:,j])
                label += 1.0 * (agent_return1[:,i] < agent_return1[:,j])
                loss1 = label.squeeze(-1) * nlpji + (1 - label.squeeze(-1)) * nlpij

                aux_loss.append(loss0)
                aux_loss.append(loss1)
        
        aux_loss = th.stack(aux_loss, dim=-1).sum(-1).mean()

        loss = loss + self.args.aux_loss_coe * aux_loss

        self.agent_optimiser.zero_grad()
        # integration: detached loss capture, no change to the optimized expression
        if not th.isfinite(loss).all(): raise FloatingPointError('Non-finite source loss: loss')
        if not hasattr(self, '_integration_losses'): self._integration_losses = {}
        self._integration_losses['loss'] = float(loss.detach())
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm, t_env)
            self.log_stats_t = t_env
            running_log.update({
                "loss": loss.item(),
                "agent_grad_norm": grad_norm,
            })

    def cuda(self):
        self.mac.cuda()
    
    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
    
    def load_models(self, path):
        self.mac.load_models(path)
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))