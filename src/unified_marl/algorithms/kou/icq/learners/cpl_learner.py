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
class CPLLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger
        self.cpl_lambda = args.cpl_lambda
        self.cpl_alpha = args.cpl_alpha
        self.NLL_lamda = args.NLL_lamda
        self.use_nll = args.use_nll
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
        mask0 = batch0["filled"].float()
        mask0[:, 1:] = mask0[:, 1:] * (1 - terminated0[:, :-1])

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
        mac_out0 = th.stack(mac_out0, dim=1).squeeze(2)
        
        if self.args.env == "sc2":
            mac_out0[avail_actions0 == 0] = 0
            mac_out0 = mac_out0/mac_out0.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            mac_out0[avail_actions0 == 0] = 0
        else:
            mac_out0[avail_actions0 == 0] = 0

        if self.args.env == "sc2":
            pi_taken0 = th.gather(mac_out0, dim=-1, index=actions0)
        else:
            pi_taken0 = mac_out0
        pi_taken0[mask0.repeat(1, 1, self.n_agents).unsqueeze(-1) == 0] = 1.0
        log_pi_taken0 = th.log(pi_taken0).squeeze(-1).mean(-1)
        record_pi_taken0_log = th.exp((log_pi_taken0 * mask0.squeeze(-1))).sum()/(bs*batch0['max_seq_length']).item()
        record_pi_taken0_real = (pi_taken0.sum(-2) * mask0).sum()/(bs*batch0['max_seq_length']*self.n_agents).item()

        # # ----------- batch1 not preferred -----------------
        actions1 = batch1["actions"]
        terminated1 = batch1["terminated"].float()
        avail_actions1 = batch1["avail_actions"].long()
        mask1 = batch1["filled"].float()
        mask1[:, 1:] = mask1[:, 1:] * (1 - terminated1[:, :-1])

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
        mac_out1 = th.stack(mac_out1, dim=1).squeeze(2)
        
        if self.args.env == "sc2":
            mac_out1[avail_actions1 == 0] = 0
            mac_out1 = mac_out1/mac_out1.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            mac_out1[avail_actions1 == 0] = 0
        else:
            mac_out1[avail_actions1 == 0] = 0

        if self.args.env == "sc2":
            pi_taken1 = th.gather(mac_out1, dim=-1, index=actions1)
        else:
            pi_taken1 = mac_out1
        pi_taken1[mask1.repeat(1, 1, self.n_agents).unsqueeze(-1) == 0] = 1.0
        log_pi_taken1 = th.log(pi_taken1).squeeze(-1).mean(-1)
        record_pi_taken1_log = th.exp((log_pi_taken1 * mask1.squeeze(-1))).sum()/(bs*batch1['max_seq_length']).item()
        record_pi_taken1_real = (pi_taken1.sum(-2) * mask1).sum()/(bs*batch1['max_seq_length']*self.n_agents).item()

        # # ------------- CPL loss ------------------
        adv0 = (self.cpl_alpha * log_pi_taken0 * mask0.squeeze(-1)).sum(1) / mask0.sum(1).squeeze(-1)
        adv1 = (self.cpl_alpha * log_pi_taken1 * mask1.squeeze(-1)).sum(1) / mask1.sum(1).squeeze(-1)
        ################
        logit10 = adv1 - self.cpl_lambda * adv0
        logit01 = adv0 - self.cpl_lambda * adv1
        max21 = th.clamp(-logit10, min=0, max=None)
        max12 = th.clamp(-logit01, min=0, max=None)
        nlp21 = th.log(th.exp(-max21) + th.exp(-logit10 - max21)) + max21
        nlp12 = th.log(th.exp(-max12) + th.exp(-logit01 - max12)) + max12
        loss = labels.squeeze(-1) * nlp21 + (1 - labels.squeeze(-1)) * nlp12
        loss = loss.mean()
        # 1.prefer loss / unprefer loss, lambda=1
        # 2.loss + prefer segment nlp
        # 3.MAMujoco
        # add constrain item
        if self.use_nll:
            positive_logits0,positive_labels0,positive_logits1,positive_labels1 = [], [], [], []
            cpl_loss = loss
            positive_indices = labels.bool().squeeze(-1)  # True for positive samples

            # batch0 和 batch1 中正样本的 NLL
            positive_logits0 = mac_out0[~positive_indices]
            positive_labels0 = actions0[~positive_indices]
            num_classes = self.n_actions
            positive_logits0 = positive_logits0.view(-1, num_classes)
            positive_labels0 = positive_labels0.view(-1)
            nll_loss0 = F.nll_loss(F.log_softmax(positive_logits0, dim=-1), positive_labels0.squeeze(-1), reduction='mean')
    
            positive_logits1 = mac_out1[positive_indices]
            positive_labels1 = actions1[positive_indices]
            positive_logits1 = positive_logits1.view(-1, num_classes)
            positive_labels1 = positive_labels1.view(-1)
            nll_loss1 = F.nll_loss(F.log_softmax(positive_logits1, dim=-1), positive_labels1.squeeze(-1), reduction='mean')
            
            # # ------------- Combined Loss ------------------
            # Combine cpl loss with NLL loss using a lambda hyperparameter
            nll_loss = nll_loss0 + nll_loss1
            combined_loss = cpl_loss + self.NLL_lamda * nll_loss
            loss = combined_loss
        else :
            nll_loss = th.zeros(1)
            cpl_loss = loss
        ########################
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
            self.logger.log_stat("cpl_loss", cpl_loss.item(), t_env)
            self.logger.log_stat("nll_loss", nll_loss.item(), t_env)
            self.logger.log_stat("prefer_p_log", record_pi_taken0_log, t_env)
            self.logger.log_stat("prefer_p_real", record_pi_taken0_real, t_env)
            self.logger.log_stat("unprefer_p_log", record_pi_taken1_log, t_env)
            self.logger.log_stat("unprefer_p_real", record_pi_taken1_real, t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm, t_env)
            self.log_stats_t = t_env
            running_log.update({
                "loss": loss.item(),
                "cpl_loss": cpl_loss.item(),
                "nll_loss": nll_loss.item(),
                "prefer_p_log": record_pi_taken0_log,
                "prefer_p_real": record_pi_taken0_real,
                "unprefer_p_log": record_pi_taken1_log,
                "unprefer_p_real": record_pi_taken1_real,
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