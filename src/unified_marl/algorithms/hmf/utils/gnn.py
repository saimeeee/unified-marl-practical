import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Linear, Parameter
from torch.autograd import Variable


class GCMC(nn.Module):
    
    def __init__(self, config):
        super(GCMC, self).__init__()
        rating_values = config['rating_values']
        in_dim = config['embed_dim']

        assert in_dim % len(rating_values) == 0
        msg_dim = in_dim // len(rating_values)
        out_dim = config['embed_dim']


        self.feat_tran = fea_trans(config)
         
        self.gcenc = GCMCLayer(rating_values, in_dim, msg_dim, out_dim)
        self.Linear_u = Linear(out_dim, out_dim, bias = True)
        self.Linear_i = Linear(out_dim, out_dim, bias = True)
        self.bidec = BiDecoder(rating_values, in_dim)

    def reset_parameters(self):
        self.gcenc.reset_parameters()
        self.bidec.reset_parameters()
        self.feat_tran.reset_parameters()
        self.Linear_u.reset_parameters()
        self.Linear_i.reset_parameters()

        
    def forward(self, u_feat, i_feat, enc_G, dec_G):
        u_feat, i_feat = self.feat_tran(u_feat, i_feat)

        u_emb, i_emb = self.gcenc(enc_G, u_feat, i_feat)

        u_emb = self.Linear_u(u_emb)
        i_emb = self.Linear_i(i_emb)

        pred_ratings = self.bidec(dec_G, u_emb, i_emb)

        return pred_ratings

class fea_trans(nn.Module):

    def __init__(self, config):
        super(fea_trans, self).__init__()

        u_fea_dim = config['u_fea_dim']
        i_fea_dim = config['i_fea_dim']

        if config['dataset'] == 'ml-1m':
            u_layer_dims = [u_fea_dim, u_fea_dim//10, u_fea_dim//50, config['embed_dim'] * 2, config['embed_dim'] ]
            i_layer_dims = [i_fea_dim, i_fea_dim//10, i_fea_dim//50, config['embed_dim'] * 2, config['embed_dim'] ]

        self.u_dense = dense_layer(u_layer_dims)
        self.i_dense = dense_layer(i_layer_dims)
        

    def reset_parameters(self):
        self.u_dense.reset_parameters()
        self.i_dense.reset_parameters()

    def forward(self, u_feat, i_feat):
        trans_u_feat = self.u_dense(u_feat)
        trans_i_feat = self.i_dense(i_feat)

        return trans_u_feat, trans_i_feat

class dense_layer(nn.Module):

    def __init__(self, layer_dims):
        super(dense_layer,self).__init__()
        
        self.layer_dims = layer_dims
        self.Layers = nn.ModuleList()

        for i in range(len(layer_dims) - 1):
            self.Layers.append(Linear(layer_dims[i], layer_dims[i+1], bias = True ))
    
    def reset_parameters(self):
        for layer in self.Layers:
            layer.reset_parameters()

    def forward(self, in_feat):
        assert in_feat.shape[1] == self.layer_dims[0]

        for layer in self.Layers:
            in_feat = layer(in_feat)
            in_feat = F.relu(in_feat)
        
        return in_feat

class GCMCLayer(nn.Module):
    def __init__(self,
                 rating_vals,
                 in_dim,
                 msg_dim,
                 out_dim,
                 dropout_rate=0.0,
                 agg='stack',  # or 'sum'
                 agg_act= True,
                 share_user_item_param=False):
        super(GCMCLayer, self).__init__()
        self.rating_vals = rating_vals
        self.agg = agg
        self.share_user_item_param = share_user_item_param

        self.agg_act = agg_act
        
            
        self.dropout = nn.Dropout(dropout_rate)

        self.conv_u = torch.nn.ModuleList()
        self.conv_i = torch.nn.ModuleList()
        
        if share_user_item_param:
            for i in range(len(rating_vals)):
                self.conv_u.append(Linear(in_dim, msg_dim, bias = False))
                self.conv_i.append(self.conv_u[i])
        else:
            for i in range(len(rating_vals)):
                self.conv_u.append(Linear(in_dim, msg_dim, bias = False))
                self.conv_i.append(Linear(in_dim, msg_dim, bias = False))

    def reset_parameters(self):          
        if self.share_user_item_param:
            for lin in self.conv_u:
                lin.reset_parameters()
        else:
            for lin in self.conv_u:
                lin.reset_parameters()
            for lin in self.conv_i:
                lin.reset_parameters()

    def forward(self, graph, ufeat=None, ifeat=None):

        num_u = graph.number_of_nodes('user')
        num_i = graph.number_of_nodes('item')
        
        funcs = {}
        for i, rating in enumerate(self.rating_vals):
            rating = str(rating)
            # W_r * x
            x_u = self.conv_u[i](ufeat)
            x_i = self.conv_i[i](ifeat)
    
            # left norm and dropout
            x_u = x_u * self.dropout(graph.nodes['user'].data['sqrt_deg'])
            x_i = x_i * self.dropout(graph.nodes['item'].data['sqrt_deg'])

            graph.nodes['user'].data['h%d' % i] = x_u
            graph.nodes['item'].data['h%d' % i] = x_i

            funcs[rating] = (fn.copy_u('h%d' % i, 'm'), fn.sum('m', 'h'))
            funcs['rev-%s' % rating] = (fn.copy_u('h%d' % i, 'm'), fn.sum('m', 'h'))

        # message passing
        graph.multi_update_all(funcs, self.agg)

        ufeat = graph.nodes['user'].data.pop('h').reshape((num_u, -1))
        ifeat = graph.nodes['item'].data.pop('h').reshape((num_i, -1))

        # right norm
        ufeat = ufeat * graph.nodes['user'].data['sqrt_deg']
        ifeat = ifeat * graph.nodes['item'].data['sqrt_deg']

        # non-linear
       
        ufeat = F.relu(ufeat)
        ifeat = F.relu(ifeat)

        return ufeat, ifeat

class BiDecoder(nn.Module):

    def __init__(self,
                 rating_vals,
                 in_dim,
                 num_basis_functions=2,
                 dropout_rate=0.0):
        super(BiDecoder, self).__init__()

        self.rating_vals = rating_vals
        self._num_basis_functions = num_basis_functions
        self.dropout = nn.Dropout(dropout_rate)


        self.Ps = []
        
        for i in range(len(self.rating_vals)):
            self.Ps.append(Parameter(torch.Tensor(in_dim, in_dim)))
            torch.nn.init.xavier_uniform_(self.Ps[i])

        self.rate_out = Linear(in_dim * len(rating_vals), len(rating_vals), bias = True)
    
    def reset_parameters(self):
        
        self.rate_out.reset_parameters

    def forward(self, graph, ufeat, ifeat):
        """Forward function.
        Parameters
        ----------
        graph : DGLHeteroGraph
            "Flattened" user-movie graph with only one edge type.
        ufeat : torch.Tensor
            User embeddings. Shape: (|V_u|, D)
        ifeat : torch.Tensor
            Movie embeddings. Shape: (|V_m|, D)
        Returns
        -------
        torch.Tensor
            Predicting scores for each user-movie edge.
        """
        graph = graph.local_var()
        ufeat = self.dropout(ufeat)
        ifeat = self.dropout(ifeat)
        graph.nodes['item'].data['h'] = ifeat
        basis_out = []
        for i in range(len(self.rating_vals)):
            graph.nodes['user'].data['h'] = torch.mm(ufeat, self.Ps[i])
            graph.apply_edges(fn.u_dot_v('h', 'h', 'sr'))
            # basis_out.append(graph.edata['sr'].expand_dims(1))
            basis_out.append(torch.unsqueeze(graph.edata['sr'],1))

        out = torch.cat(basis_out, dim=1)
        
        out = F.softmax(out, dim = 1)
        possible_ratings = torch.Tensor(self.rating_vals)

        ratings = torch.sum(out*possible_ratings, dim =1)
        return ratings