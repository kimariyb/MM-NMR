import torch
import torch.nn as nn
import torch.nn.functional as F

from models.geometry.comenet import ComENet


class GeometricSpectraRegressor(nn.Module):
    def __init__(self, config):
        super(GeometricSpectraRegressor, self).__init__()

        self.gnn = ComENet(
            cutoff=config.cutoff,
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            middle_dim=config.middle_dim,
            out_dim=config.out_dim,
            num_radial=config.num_radial,
            num_spherical=config.num_spherical,
            num_output_layers=config.num_output_layers,
            dropout=config.dropout,
        )
        
        self.readout_n = nn.Sequential(
            nn.Linear(config.out_dim, config.out_dim * 3), nn.PReLU(), nn.Dropout(0.2),
            nn.Linear(config.out_dim * 3, config.pred_dim), nn.PReLU(), nn.Dropout(0.2),
            nn.Linear(config.pred_dim, 1)
        )
        
    def forward(self, data):
        z, pos, batch, mask = data.z, data.pos, data.batch, data.mask

        node_embed_feats = self.gnn(z, pos, batch) # (N, out_dim)
       
        out = self.readout_n(node_embed_feats[mask])
        
        return out[:, 0]
    
