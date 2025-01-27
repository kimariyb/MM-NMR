import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing

from utils.Features import GetAtomFeaturesDim, GetBondFeaturesDim

full_atom_feature_dims = GetAtomFeaturesDim()
full_bond_feature_dims = GetBondFeaturesDim()


class AtomEncoder(nn.Module):
    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()
        
        self.atom_embedding_list = nn.ModuleList()
        
        for dim in full_atom_feature_dims:
            emb = nn.Embedding(dim, emb_dim)
            nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)
    
    def forward(self, x):
        atom_embedding = 0
        start_idx = 0
        
        for i, dim in enumerate(full_atom_feature_dims):
            feat = x[:, start_idx:start_idx + dim]
            feat_idx = torch.argmax(feat, dim=1)
            atom_embedding += self.atom_embedding_list[i](feat_idx)
            start_idx += dim
        
        return atom_embedding


class BondEncoder(nn.Module):
    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()
        
        self.bond_embedding_list = nn.ModuleList()
        
        for dim in full_bond_feature_dims:
            emb = nn.Embedding(dim, emb_dim)
            nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)
    
    def forward(self, edge_attr):
        edge_embedding = 0
        start_idx = 0
        
        for i, dim in enumerate(full_bond_feature_dims):
            feat = edge_attr[:, start_idx:start_idx + dim]
            feat_idx = torch.argmax(feat, dim=1)
            edge_embedding += self.bond_embedding_list[i](feat_idx)
            start_idx += dim
        
        return edge_embedding
    

class GINConv(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        super(GINConv, self).__init__(aggr=aggr)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim), 
            nn.BatchNorm1d(2 * emb_dim), 
            nn.ReLU(),
            nn.Linear(2*emb_dim, emb_dim)
        )

        self.eps = nn.Parameter(torch.Tensor([0]))
        
        self.bond_encoder = BondEncoder(emb_dim)
    
    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
        return out
    
    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)
    
    def update(self, aggr_out):
        return aggr_out
    

class GNN(nn.Module):
    def __init__(
        self, 
        num_layers: int = 6, 
        emb_dim: int = 300, 
        JK: str = "last", 
        drop_ratio: float = 0.5, 
    ):
        if num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        
        super(GNN, self).__init__()
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        
        self.atom_encoder = AtomEncoder(emb_dim)
        
        self.gnns = nn.ModuleList()
        for layer in range(num_layers):
            self.gnns.append(GINConv(emb_dim, aggr="add"))
        
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))
            
    def forward(self, x, edge_index, edge_attr):
        x = self.atom_encoder(x)
        
        h_list = [x]
        for layer in range(self.num_layers):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layers - 1:
                h = F.dropout(h, p=self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), p=self.drop_ratio, training=self.training)
            h_list.append(h)
        
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]
        else:
            raise ValueError("Invalid JK type.")
        
        return node_representation
    
