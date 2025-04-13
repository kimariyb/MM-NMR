import torch
import torch.nn as nn

from torch_geometric.nn import Set2Set

from models.graph.pagtn import PAGTNGNN


class PAGTNSpectraRegressor(nn.Module):
    def __init__(self, config):
        super(PAGTNSpectraRegressor, self).__init__()

        self.gnn = PAGTNGNN(
            node_feats=config.node_feats,
            node_hid_feats=config.node_hid_feats,
            edge_feats=config.edge_feats,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
        )

        self.readout_g = Set2Set(
            in_channels=config.node_feats + config.node_hid_feats,
            processing_steps=3, num_layers=1
        )
        
        self.readout_n = nn.Sequential(
            nn.Linear((config.node_feats + config.node_hid_feats) * 3, config.pred_hid_feats), nn.PReLU(), nn.Dropout(0.2),
            nn.Linear(config.pred_hid_feats, config.pred_hid_feats), nn.PReLU(), nn.Dropout(0.2),
            nn.Linear(config.pred_hid_feats, config.pred_hid_feats), nn.PReLU(), nn.Dropout(0.2),
            nn.Linear(config.pred_hid_feats, 1)
        )
        
    def forward(self, data):
        x, edge_attr, edge_index, batch, mask = data.x, data.edge_attr, data.edge_index, data.batch, data.mask

        node_feats_embedding = self.gnn(x, edge_attr, edge_index)
        node_embed_feats = torch.cat([x, node_feats_embedding], dim=1)

        _, counts = torch.unique(batch, return_counts=True)        
        graph_embed_feats = self.readout_g(node_embed_feats, batch)
        graph_embed_feats = torch.repeat_interleave(graph_embed_feats, counts, dim=0)

        out = self.readout_n(torch.hstack([node_embed_feats, graph_embed_feats])[mask])
        
        return out[:, 0]
    
