import torch.nn as nn
import torch.nn.functional as F
import torch as th
from torch.distributions import kl_divergence
import torch.distributions as D
import math
import time

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

latent_dim = 3
hidden_dim = 64
NN_HIDDEN_SIZE = 16
dis_loss_weight = 0.001
h_loss_weight = 0.0001
kl_loss_weight = 0.0001
dis_time = 0
roma_raw = False
var_floor = 0.002
soft_constraint_weight = 1.0
dis_sigmoid = False

class LatentCEDisRNNAgent(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim, n_agents, n_groups):
        super(LatentCEDisRNNAgent, self).__init__()
        self.input_shape = input_dim
        self.n_agents = n_agents
        self.n_groups = n_groups
        self.n_actions = out_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.embed_fc_input_size = input_dim
        NN_HIDDEN_SIZE = NN_HIDDEN_SIZE
        activation_func=nn.LeakyReLU()

        self.embed_net = nn.Sequential(nn.Linear(self.embed_fc_input_size, NN_HIDDEN_SIZE),
                                       nn.BatchNorm1d(NN_HIDDEN_SIZE),
                                       activation_func,
                                       nn.Linear(NN_HIDDEN_SIZE, latent_dim * 2))

        self.inference_net = nn.Sequential(nn.Linear(self.hidden_dim + self.input_shape, NN_HIDDEN_SIZE),
                                           nn.BatchNorm1d(NN_HIDDEN_SIZE),
                                           activation_func,
                                           nn.Linear(NN_HIDDEN_SIZE, latent_dim * 2))

        self.latent = th.rand(self.n_agents, latent_dim * 2)  # (n,mu+var)
        self.latent_infer = th.rand(self.n_agents, latent_dim * 2)  # (n,mu+var)

        self.latent_net = nn.Sequential(nn.Linear(latent_dim, NN_HIDDEN_SIZE),
                                        nn.BatchNorm1d(NN_HIDDEN_SIZE),
                                        activation_func)

        self.fc1 = nn.Linear(self.input_shape, self.hidden_dim)
        self.rnn = nn.GRUCell(self.hidden_dim, self.hidden_dim)

        self.fc2_w_nn = nn.Linear(NN_HIDDEN_SIZE, self.hidden_dim * self.n_actions)
        self.fc2_b_nn = nn.Linear(NN_HIDDEN_SIZE, self.n_actions)

        # Dis Net
        self.dis_net = nn.Sequential(nn.Linear(latent_dim * 2, NN_HIDDEN_SIZE ),
                                     nn.BatchNorm1d(NN_HIDDEN_SIZE ),
                                     activation_func,
                                     nn.Linear(NN_HIDDEN_SIZE , 1))

        self.mi= th.rand(self.n_agents*self.n_agents)
        self.dissimilarity = th.rand(self.n_agents*self.n_agents)

        if dis_sigmoid:
            print('>>> sigmoid')
            self.dis_loss_weight_schedule = self.dis_loss_weight_schedule_sigmoid
        else:
            self.dis_loss_weight_schedule = self.dis_loss_weight_schedule_step

    def init_latent(self):
        # self.bs = bs
        self.trajectory = []

        var_mean=self.latent[:self.n_agents, self.latent_dim:].detach().mean()

        #mask = 1 - th.eye(self.n_agents).byte()
        #mi=self.mi.view(self.n_agents,self.n_agents).masked_select(mask)
        #di=self.dissimilarity.view(self.n_agents,self.n_agents).masked_select(mask)
        mi = self.mi
        di = self.dissimilarity
        indicator=[var_mean,mi.max(),mi.min(),mi.mean(),mi.std(),di.max(),di.min(),di.mean(),di.std()]
        return indicator, self.latent[:self.n_agents, :].detach(), self.latent_infer[:self.n_agents, :].detach()

    def forward(self, inputs, rnn_hidden_state, t_glob=0, train_mode=False):
        bs = inputs.shape[0]
        inputs = inputs.reshape(-1, self.input_shape)
        h_in = rnn_hidden_state.reshape(-1, self.hidden_dim)

        embed_fc_input = inputs[:, - self.embed_fc_input_size:]  # own features(unit_type_bits+shield_bits_ally)+id

        self.latent = self.embed_net(embed_fc_input)
        self.latent[:, -self.latent_dim:] = th.clamp(th.exp(self.latent[:, -self.latent_dim:]), min=var_floor)  # var
        #self.latent[:, -self.latent_dim:] = th.full_like(self.latent[:, -self.latent_dim:],1.0)

        latent_embed = self.latent.reshape(bs * self.n_agents, self.latent_dim * 2)

        #latent = latent_embed[:, :self.latent_dim]
        gaussian_embed = D.Normal(latent_embed[:, :self.latent_dim], (latent_embed[:, self.latent_dim:]) ** (1 / 2))
        latent = gaussian_embed.rsample()

        c_dis_loss = 0
        ce_loss = 0
        loss = 0
        
        if train_mode and (not roma_raw):
            #gaussian_embed = D.Normal(latent_embed[:, :self.latent_dim], (latent_embed[:, self.latent_dim:]) ** (1 / 2))
            #latent = gaussian_embed.rsample()

            self.latent_infer = self.inference_net(th.cat([h_in.detach(), inputs], dim=1))
            self.latent_infer[:, -self.latent_dim:] = th.clamp(th.exp(self.latent_infer[:, -self.latent_dim:]),min=var_floor)
            #self.latent_infer[:, -self.latent_dim:] = th.full_like(self.latent_infer[:, -self.latent_dim:],1.0)
            gaussian_infer = D.Normal(self.latent_infer[:, :self.latent_dim], (self.latent_infer[:, self.latent_dim:]) ** (1 / 2))
            latent_infer = gaussian_infer.rsample()

            loss = gaussian_embed.entropy().sum(dim=-1).mean() * h_loss_weight + kl_divergence(gaussian_embed, gaussian_infer).sum(dim=-1).mean() * kl_loss_weight   # CE = H + KL
            loss = th.clamp(loss, max=2e3)
            # loss = loss / (self.bs * self.n_agents)
            ce_loss = th.log(1 + th.exp(loss))

            # Dis Loss
            cur_dis_loss_weight = self.dis_loss_weight_schedule(t_glob)
            if cur_dis_loss_weight > 0:
                dis_loss = 0
                dissimilarity_cat = None
                mi_cat = None
                latent_dis = latent.clone().view(bs, self.n_agents, -1)
                latent_move = latent.clone().view(bs, self.n_agents, -1)
                for agent_i in range(self.n_agents):
                    latent_move = th.cat(
                        [latent_move[:, -1, :].unsqueeze(1), latent_move[:, :-1, :]], dim=1)
                    latent_dis_pair = th.cat([latent_dis[:, :, :self.latent_dim],
                                              latent_move[:, :, :self.latent_dim],
                                            # (latent_dis[:, :, :self.latent_dim]-latent_move[:, :, :self.latent_dim])**2
                                              ], dim=2)
                    mi = th.clamp(gaussian_embed.log_prob(latent_move.view(bs * self.n_agents, -1))+13.9, min=-13.9).sum(dim=1,keepdim=True) / self.latent_dim

                    dissimilarity = th.abs(self.dis_net(latent_dis_pair.view(-1, 2 * self.latent_dim)))

                    if dissimilarity_cat is None:
                        dissimilarity_cat = dissimilarity.view(bs, -1).clone()
                    else:
                        dissimilarity_cat = th.cat([dissimilarity_cat, dissimilarity.view(bs, -1)], dim=1)
                    if mi_cat is None:
                        mi_cat = mi.view(bs, -1).clone()
                    else:
                        mi_cat = th.cat([mi_cat,mi.view(bs,-1)],dim=1)

                    #dis_loss -= th.clamp(mi / 100 + dissimilarity, max=0.18).sum() / self.bs / self.n_agents

                mi_min=mi_cat.min(dim=1,keepdim=True)[0]
                mi_max=mi_cat.max(dim=1,keepdim=True)[0]
                di_min = dissimilarity_cat.min(dim=1, keepdim=True)[0]
                di_max = dissimilarity_cat.max(dim=1, keepdim=True)[0]

                mi_cat=(mi_cat-mi_min)/(mi_max-mi_min+ 1e-12 )
                dissimilarity_cat=(dissimilarity_cat-di_min)/(di_max-di_min+ 1e-12 )

                dis_loss = - th.clamp(mi_cat+dissimilarity_cat, max=1.0).sum()/bs/self.n_agents
                #dis_loss = ((mi_cat + dissimilarity_cat - 1.0 )**2).sum() / self.bs / self.n_agents
                dis_norm = th.norm(dissimilarity_cat, p=1, dim=1).sum() /bs/ self.n_agents

                #c_dis_loss = (dis_loss + dis_norm) / self.n_agents * cur_dis_loss_weight
                c_dis_loss = (dis_norm + soft_constraint_weight * dis_loss) / self.n_agents * cur_dis_loss_weight
                loss = ce_loss +  c_dis_loss

                self.mi = mi_cat[0]
                self.dissimilarity = dissimilarity_cat[0]
            else:
                c_dis_loss = th.zeros_like(loss)
                loss = ce_loss


        # Role -> FC2 Params
        clusters = None
        if not train_mode:
            clusters = self.k_means(latent, self.n_groups)
        latent = self.latent_net(latent)

        fc2_w = self.fc2_w_nn(latent)
        fc2_b = self.fc2_b_nn(latent)
        fc2_w = fc2_w.reshape(-1, self.hidden_dim, self.n_actions)
        fc2_b = fc2_b.reshape((-1, 1, self.n_actions))

        x = F.relu(self.fc1(inputs))  # (bs*n,(obs+act+id)) at time t
        h = self.rnn(x)
        h = h.reshape(-1, 1, self.hidden_dim)
        q = th.bmm(h, fc2_w) + fc2_b

        h = h.reshape(-1, self.hidden_dim)

        return q.view(-1, self.n_actions), h.view(-1, self.hidden_dim), loss, c_dis_loss, ce_loss, clusters

    def dis_loss_weight_schedule_step(self, t_glob):
        if t_glob > dis_time:
            return dis_loss_weight
        else:
            return 0

    def dis_loss_weight_schedule_sigmoid(self, t_glob):
        return dis_loss_weight / (1 + math.exp((1e7 - t_glob) / 2e6))

    def k_means(latent, n_groups, max_iters=100, tol=1e-4):
        bs = latent.shape[0]
        centroids = latent[th.randint(0, bs, (1,), device='cuda')]
        for _ in range(n_groups-1):
            dists = th.cdist(latent, centroids, p=2)  # [n_samples, n_centroids]
            min_dists = dists.min(dim=1).values
            probs = min_dists / min_dists.sum()
            new_idx = th.multinomial(probs, 1)
            centroids = th.cat([centroids, latent[new_idx]])
        
        for iter in range(max_iters):
            dists = th.cdist(latent, centroids)  # [n_samples, k]
            labels = dists.argmin(dim=1)      # [n_samples]
            new_centroids = th.stack([
                latent[labels == i].mean(dim=0) if (labels == i).any() else centroids[i]
                for i in range(n_groups)
            ])
            shift = th.norm(new_centroids - centroids)
            if shift < tol:
                break
            centroids = new_centroids
        return labels