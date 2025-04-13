import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class PAGTNLayer(MessagePassing):
    r"""
    Single PAGTN layer from `Path-Augmented Graph Transformer Network
    <https://arxiv.org/abs/1905.12712>`__
    This will be used for incorporating the information of edge features
    into node features for message passing.
    """
    def __init__(self, config):
        super(PAGTNLayer, self).__init__()
        self.node_hid_feats = config.node_hid_feats
        self.edge_feats = config.edge_feats
        
        self.attn_src = nn.Linear(self.node_hid_feats, self.node_hid_feats)
        self.attn_dst = nn.Linear(self.node_hid_feats, self.node_hid_feats)
        self.attn_edg = nn.Linear(self.edge_feats, self.node_hid_feats)
        self.attn_dot = nn.Linear(self.node_hid_feats, 1)
        
        # Message construction linear layers
        self.msg_src = nn.Linear(self.node_hid_feats, self.node_hid_feats)
        self.msg_dst = nn.Linear(self.node_hid_feats, self.node_hid_feats)
        self.msg_edg = nn.Linear(self.edge_feats, self.node_hid_feats)
        
        # Skip connection / Final node transformation
        self.wgt_n = nn.Linear(self.node_hid_feats, self.node_hid_feats)
        
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for layer in [
            self.attn_src, self.attn_dst, self.attn_edg, self.attn_dot, 
            self.msg_dst, self.msg_edg, self.wgt_n
        ]:
            if hasattr(layer, 'weight'):
                nn.init.xavier_normal_(layer.weight, gain=gain)
            if hasattr(layer, 'bias') and layer.bias is not None:
                nn.init.zeros_(layer.bias)        
                
    def forward(self, x, edge_attr, edge_index):
        r"""
        Forward function of PAGTN layer.
        
        Parameters
        ----------
        x : torch.Tensor
            Node features with shape (num_nodes, node_in_feats) or (N, num_heads, node_in_feats).
        edge_attr : torch.Tensor
            Edge features with shape (num_edges, edge_attr).
        edge_index : torch.Tensor
            Edge index with shape (2, num_edges).
        
        Returns
        -------
        torch.Tensor
            Updated node features with shape (N, num_heads, node_out_feats) or (N, node_out_feats).
        """
        num_nodes = x.size(0)
        row, col = edge_index # row = target (i), col = source (j)
        
        # 1. Calculate Attention Scores
        x_attn_src = self.attn_src(x) # (N, H, node_in_feats)
        x_attn_dst = self.attn_dst(x) # (N, H, node_in_feats)
        e_attn = self.attn_edg(edge_attr).unsqueeze(-2) # (E, 1, node_in_feats)
        
        # Combine features on edges (source node + target node + edge)
        attn_input = x_attn_src[col] + x_attn_dst[row] + e_attn # (E, 1H, node_in_feats)
        alpha_raw = self.attn_dot(self.act(attn_input)) # (E, H, 1)
        alpha = softmax(alpha_raw, row, num_nodes=num_nodes) # (E, H, 1)

         # 2. Calculate Messages
        x_msg_dst = self.msg_dst(x) # (N, H, node_in_feats)
        e_msg = self.msg_edg(edge_attr) # (E, H, node_out_feats)
        msg_input = x_msg_dst[col] + e_msg.unsqueeze(-2) # (E, H, node_out_feats)
        
        # 3. Propagate: Aggregate weighted messages
        # Pass necessary info (edge_index, alpha, msg_input) to propagate
        # size=num_nodes ensures output tensor has correct shape even with isolated nodes
        # We pass alpha and msg_input explicitly.
        aggr_out = self.propagate(edge_index, size=num_nodes, alpha=alpha, msg_input=msg_input)

        # 4. Update node features: Add aggregated messages and skip connection
        # The skip connection is added outside propagate, similar to the DGL update step.
        out = aggr_out + self.wgt_n(x) # (N, node_out_feats)
        
        return out

    def message(self, alpha, msg_input):
        r"""
        Constructs messages to be passed along edges.
        This function is called by `propagate`.

        Parameters
        ----------
        alpha : torch.Tensor
            Attention scores for each edge with shape (E, 1).
        msg_input : torch.Tensor
            Input message features for each edge with shape (E, node_out_feats).

        Returns
        -------
        torch.Tensor
            The weighted message for each edge, shape (E, node_out_feats).
        """
        # Weight the message input by the attention score
        return alpha * msg_input
    
    
class PAGTNGNN(nn.Module):
    r"""
    Multilayer PAGTN model for updating node representations.
    PAGTN is introduced in `Path-Augmented Graph Transformer Network
    <https://arxiv.org/abs/1905.12712>`__.
    """
    def __init__(self, config):
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.node_in_feats = config.node_in_feats
        self.node_hid_feats = config.node_hid_feats

        self.atom_inp = nn.Linear( 
            self.node_in_feats, 
            self.node_hid_feats * self.num_heads
        )
                        
        self.model = nn.ModuleList(
            [PAGTNLayer(config)
            for _ in range(self.num_layers)]
        )
        
        self.act = nn.ReLU()

    def forward(self, x, edge_attr, edge_index):
        r"""
        Update node representations.
        
        Parameters
        ----------
        x : torch.Tensor
            Node features with shape (num_nodes, node_in_feats).
        edge_attr : torch.Tensor
            Edge features with shape (num_edges, edge_attr).
        edge_index : torch.Tensor
            Edge index with shape (2, num_edges).
        
        Returns
        -------
        torch.Tensor
            Updated node features with shape (num_nodes, node_out_feats).
        """
        num_nodes = x.size(0)
        
        # 1. Initial Projection and Reshape for Multi-Head
        atom_projected = self.atom_inp(x)
        atom_h = atom_projected.view(num_nodes, self.nheads, self.node_hid_feats)
        atom_input = self.act(atom_h)
        # Store for residual connections
        atom_h = atom_input
        
        # 2. Apply PAGTN Layers Sequentially with Residual Connections
        for i in range(self.num_layers):
            attn_h = self.model[i](atom_h, edge_attr, edge_index) # Output: (N, H, F_hid)
            atom_h = attn_h + atom_input
            atom_h = F.relu(atom_h)
        
        final_node_h  = atom_h.mean(1)
        
        return final_node_h 
    
    
class PAGTNConfig:
    def __init__(self, num_layers, num_heads, node_in_feats, node_hid_feats, edge_feats):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.node_in_feats = node_in_feats
        self.node_hid_feats = node_hid_feats
        self.edge_feats = edge_feats
        
    @staticmethod
    def from_dict(config_dict):
        return PAGTNConfig(
            num_layers=config_dict['num_layers'],
            num_heads=config_dict['num_heads'],
            node_in_feats=config_dict['node_in_feats'],
            node_hid_feats=config_dict['node_hid_feats'],
            edge_feats=config_dict['edge_feats']
        )