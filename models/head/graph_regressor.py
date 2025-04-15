import torch
import torch.nn as nn
import torch.nn.functional as F

from models.graph.gin import GIN


class GINSpectraRegressor(nn.Module):
    def __init__(self, config):
        super(GINSpectraRegressor, self).__init__()

        self.gnn = GIN(
            num_layers=config.num_layers,
            node_dim=config.node_dim,
            edge_dim=config.edge_dim,
            hidden_dim=config.hidden_dim,
            drop_ratio=config.drop_ratio
        )
        
        self.readout_n = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim*3), nn.PReLU(), nn.Dropout(0.5),
            nn.Linear(config.hidden_dim*3, 1),
        )
        
    def forward(self, data):
        x, edge_attr, edge_index, mask = data.x, data.edge_attr, data.edge_index, data.mask

        node_embed_feats = self.gnn(x, edge_index, edge_attr) # (N, out_dim)

        out = self.readout_n(node_embed_feats[mask])[:, 0]
        
        return out
    
