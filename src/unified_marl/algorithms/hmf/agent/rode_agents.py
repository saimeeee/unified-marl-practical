import torch.nn as nn
import torch.nn.functional as F
import torch as th

class RODEAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RODEAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        return h
    
class DotRole(nn.Module):
    def __init__(self, args):
        super(DotRole, self).__init__()
        self.args = args
        self.n_actions = args.n_actions

        self.q_fc = nn.Linear(args.rnn_hidden_dim, args.action_latent_dim)
        self.action_space = th.ones(args.n_actions).to(args.device)

    def forward(self, h, action_latent):
        role_key = self.q_fc(h)  # [bs, action_latent] [n_actions, action_latent]
        role_key = role_key.unsqueeze(-1)
        action_latent_reshaped = action_latent.unsqueeze(0).repeat(role_key.shape[0], 1, 1)

        q = th.bmm(action_latent_reshaped, role_key).squeeze()

        return q

    def update_action_space(self, new_action_space):
        self.action_space = th.Tensor(new_action_space).to(self.args.device).float()

class QRole(nn.Module):
    def __init__(self, args):
        super(QRole, self).__init__()
        self.args = args
        self.n_actions = args.n_actions

        self.q_fc = nn.Linear(args.rnn_hidden_dim, self.n_actions)
        self.action_space = th.ones(args.n_actions).to(args.device)

    def forward(self, h, action_latent):
        q = self.q_fc(h)
        return q

    def update_action_space(self, new_action_space):
        self.action_space = th.Tensor(new_action_space).to(self.args.device).float()
