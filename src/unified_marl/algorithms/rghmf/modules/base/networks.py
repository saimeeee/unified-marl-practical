import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=True, discrete_action=True):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetwork, self).__init__()

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            # self.out_fn = F.tanh
            self.out_fn = F.softmax
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        h1 = self.nonlin(self.fc1(self.in_fn(X)))
        h2 = self.nonlin(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))
        return out
    

class MultiHeadAttention_hie(nn.Module):
    # taken from https://github.com/wouterkool/attention-tsp/blob/master/graph_encoder.py
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None,
            norm_in=True
    ):
        super(MultiHeadAttention_hie, self).__init__()

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.val_dim = val_dim
        self.key_dim = key_dim
        self.out_dim = embed_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        # self.encoder = nn.Sequential()
        # if norm_in:
        #     self.encoder.add_module('enc_bn', nn.BatchNorm1d(input_dim, affine=False))
        # self.encoder.add_module('enc_fc1', nn.Linear(input_dim, self.key_dim))
        # self.encoder.add_module('enc_nl', nn.ReLU())

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))
        self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, self.out_dim))
        # self.W_query_2 = nn.Parameter(torch.Tensor(n_heads, key_dim, key_dim))
        # self.W_key_2 = nn.Parameter(torch.Tensor(n_heads, key_dim, key_dim))
        # self.W_val_2 = nn.Parameter(torch.Tensor(n_heads, val_dim, val_dim))
        '''hierarchical_vdn'''
        self.pol_obs_emb = nn.ModuleList()
        self.W_out_3 = nn.ModuleList()
        for _ in range(n_heads):
            self.pol_obs_emb.append(nn.Linear(input_dim-5, val_dim, bias=False))
            self.W_out_3.append(nn.Linear(val_dim * n_heads, self.out_dim, bias=False))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, h, num_groups=1, he=None, mask=None, return_attn=False, calculate_V=False, replace=False,
                agent_id=None):
        """
        :param h: queries (batch_size, n_query, input_dim)   代表了3个agent状态的编码 (3, 1, 128)
        :param he: data (batch_size, graph_size, input_dim)  代表了3个agent对于3个entity的相对距离的编码 (3, 3, 128)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        # he 是agent和entity相对位置的信息，（agent_num, entity_num, dim=128）
        if he is None:
            he = h  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        he_batch_size, he_graph_size, he_input_dim = he.size()
        h_batch_size, h_n_query, h_input_dim = h.size()
        assert h_batch_size == he_batch_size
        assert h_input_dim == he_input_dim
        assert he_input_dim == self.input_dim, "Wrong embedding dimension of input"

        heflat = he.contiguous().view(-1, he_input_dim)
        hflat = h.contiguous().view(-1, h_input_dim)


        # last dimension can be different for keys and values
        he_shape = (self.n_heads, he_batch_size, he_graph_size, -1)
        h_shape = (self.n_heads, h_batch_size, h_n_query, -1)

        # hflat = self.encoder(hflat)
        # heflat = self.encoder(heflat)

        # self.W_query = (n_heads, h_input_dim, key_dim)
        # self.W_key = (n_heads, he_input_dim, key_dim)
        # self.W_val = (n_heads, he_input_dim, val_dim)
        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        # 将agent的状态作为查询键 (1, 3, 1, 128)
        Q = torch.matmul(hflat, self.W_query)
        # '''多加一层神经网络'''
        # Q = torch.matmul(Q, self.W_query_2)

        Q = torch.reshape(Q, (self.n_heads, h_batch_size, h_n_query, 1, -1))
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        # 将agent和地标的相对位置作为映射键 (1, 3, 3, 128)
        K_list = []
        for i in range(h_n_query):
            K = torch.matmul(heflat, self.W_key)
            # '''多加一层神经网络'''
            # K = torch.matmul(K, self.W_key_2)

            K = torch.reshape(K, (self.n_heads, he_batch_size, 1, he_graph_size, -1))
            K_list.append(K)
        # K_init_extend self.n_heads,he_batch_size,h_n_query,he_graph_size-1
        K_init_extend = torch.cat(K_list, dim=2)
        # if replace == True:
        #     K_replace_list = []
        #     for i in range(he_graph_size):
        #         K_replace = torch.matmul(hflat, self.W_key)
        #         K_replace = torch.reshape(K_replace, (self.n_heads, h_batch_size, h_n_query, 1, -1))
        #         K_replace_list.append(K_replace)
        #     K_replace_extend = torch.cat(K_replace_list, dim=3)
        #     """
        #     K_replace_extend = [[[0.2, 0.3],[0.2, 0.3],[0.2, 0.3],[0.2, 0.3]],   -->agent1  agent1'size = 4 * 2
        #                         [[0.7, 0.4],[0.7, 0.4],[0.7, 0.4],[0.7, 0.4]],   -->agent2
        #                         [...],                                           -->agent3
        #                         [...]]                                           -->agent4
        #
        #     K_init_extend = [[[0.2, 0.3],[0.7, 0.4],[0.5, 0.5],[0.6, 0.6]],   -->agent1  agent1'size = 4 * 2
        #                      [[0.2, 0.3],[0.7, 0.4],[0.5, 0.5],[0.6, 0.6]],   -->agent2
        #                      [...],                                           -->agent3
        #                      [...]]                                           -->agent4
        #     """
        #     agent_id_mask = torch.reshape(agent_id, (1, he_batch_size, h_n_query, he_graph_size, 1))
        #     agent_id_mask = agent_id_mask.repeat(self.n_heads, 1, 1, 1, self.key_dim)
        #     other_agent_id_mask = (agent_id_mask - 1) * -1
        #
        #     K_replace_extend[other_agent_id_mask] = 0.
        #     K_init_extend[agent_id_mask] = 0.
        #     K_replace = K_init_extend + K_replace_extend
        # else:
        K_replace = K_init_extend

        # 将agent和地标的相对位置作为映射空间(1, 3, 3, 128)
        V, out = None, None
        if calculate_V == True:
            # V = torch.matmul(heflat, self.W_val).view(he_shape)
            '''多加一层神经网络'''
            V = torch.matmul(heflat, self.W_val)
            # V = torch.matmul(V, self.W_val_2)
            V = torch.reshape(V, he_shape)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        # 计算agent状态和agnet与entity相对距离的相似度
        """
        [[x1]                                   [[x1*x1], [x1*x2], [x1*x3], [x1*x4]
         [x2]   *   [[x1], [x2], [x3], [x4]]  =  [x2*x1], [x2*x2], [x2*x3], [x2*x4]                                                   
         [x3]                                    [x3*x1], [x3*x2], [x3*x3], [x3*x4]
         [x4]]                                   [x4*x1], [x4*x2], [x4*x3], [x4*x4]]

        """
        compatibility = self.norm_factor * torch.matmul(Q, K_replace.transpose(3, 4))
        compatibility = torch.reshape(compatibility, (self.n_heads, h_batch_size, h_n_query, he_graph_size))
        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = torch.reshape(mask, (1, he_batch_size, h_n_query, he_graph_size))
            mask_copy = mask
            mask = mask.expand_as(compatibility)
            # mask = mask.contiguous().view(1, he_batch_size, h_n_query, he_graph_size).expand_as(compatibility)
            compatibility[mask] = -math.inf
        #  attn (1, 3, 1, 3)
        compatibility = torch.reshape(compatibility, (
        self.n_heads, he_batch_size, h_n_query, num_groups, int(he_graph_size / num_groups)))
        attn = F.softmax(compatibility, dim=-1)
        attn = torch.reshape(attn, (self.n_heads, he_batch_size, h_n_query, he_graph_size))

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc
        if calculate_V == True:
            heads = torch.matmul(attn, V)
            # out = heads
            # atten (self.n_heads, he_batch_size, h_n_query, he_graph_size)
            # V (self.n_heads, he_batch_size, he_graph_size, 64)
            # heads (self.n_heads, he_batch_size, h_n_query, 64)
            out = torch.mm(
                heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
                self.W_out.view(-1, self.out_dim)
            ).view(he_batch_size, h_n_query, self.out_dim)

        if return_attn:
            return out, attn
        return out