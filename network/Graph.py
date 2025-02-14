
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from network.Basic import get_activation_function


class BatchGRUNet(nn.Module):
    def __init__(self, hidden_size=300):
        super(BatchGRUNet, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=True, batch_first=True)
        self.bias = nn.Parameter(torch.Tensor(self.hidden_size))
        self.bias.data.uniform_(-1.0 / math.sqrt(self.hidden_size), 1.0 / math.sqrt(self.hidden_size))
        
    def forward(self, node, a_scope):
        hidden = node
        message = F.relu(node + self.bias)
        max_atom_len = max([a_size for a_start, a_size in a_scope])
        # padding
        message_list = []
        hidden_list = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                assert 0
            cur_message = message.narrow(0, a_start, a_size)
            cur_hidden = hidden.narrow(0, a_start, a_size)
            hidden_list.append(cur_hidden.max(0)[0].unsqueeze(0).unsqueeze(0))
            
            cur_message = nn.ZeroPad2d((0, 0, 0, max_atom_len - cur_message.shape[0]))(cur_message)
            message_list.append(cur_message.unsqueeze(0))
        
        message_list = torch.cat(message_list, dim=0)
        hidden_list = torch.cat(hidden_list, dim=1)
        hidden_list = hidden_list.repeat(2, 2, 1)
        cur_message, cur_hidden = self.gru(message_list, hidden_list)
        
        # unpadding
        cur_message_unpadding = []
        for i, (a_start, a_size) in enumerate(a_scope):
            cur_message_unpadding.append(cur_message[i, :a_size].view(-1, 2 * self.hidden_size))
        cur_message_unpadding = torch.cat(cur_message_unpadding, dim=0)
        
        message = torch.cat([torch.cat([message.narrow(0, 0, 1), message.narrow(0, 0, 1)], 1),
                             cur_message_unpadding], 0)
        
        return message
    

class MPNNEncoder(nn.Module):
    def __init__(
        self, 
        atom_feature_size, 
        bond_feature_size, 
        hidden_size, 
        bias,
        depth,
        dropout,
        activation,
        device
    ):
        super(MPNNEncoder, self).__init__()
        self.atom_feature_size = atom_feature_size
        self.bond_feature_size = bond_feature_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.depth = depth
        self.dropout = dropout
        self.activation = activation
        self.device = device
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(p=self.dropout).to(self.device)
        
        # Activation function
        self.act = get_activation_function(self.activation).to(self.device)
        
        # Input layer
        self.W_i_atom = nn.Linear(self.atom_feature_size, self.hidden_size, bias=self.bias).to(self.device)
        self.W_i_bond = nn.Linear(self.bond_feature_size, self.hidden_size, bias=self.bias).to(self.device)
        
        w_h_input_size_atom = self.hidden_size + self.bond_feature_size
        self.W_h_atom = nn.Linear(w_h_input_size_atom, self.hidden_size, bias=self.bias)
        
        w_h_input_size_bond = self.hidden_size
        
        for dep in range(self.depth - 1):
            self.modules[f'W_h_{dep}'] = nn.Linear(w_h_input_size_bond, self.hidden_size, bias=self.bias)
            
        self.W_o = nn.Linear((self.hidden_size) * 2, self.hidden_size, bias=self.bias).to(self.device)
        
        self.gru = BatchGRUNet(hidden_size=self.hidden_size).to(self.device)

        self.lr = nn.Linear(self.hidden_size * 3, self.hidden_size, bias=self.bias).to(self.device)

    def forward(self, mol_graph): ...

