import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import Set2Set

from models.geometry.comenet import ComENet
from models.geometry.sphere import SphereNet


class ComENetSpectraRegressor(nn.Module):
    def __init__(self, config):
        super(ComENetSpectraRegressor, self).__init__()

        self.gnn = ComENet(
            cutoff=config.cutoff,
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            middle_dim=config.middle_dim,
            out_dim=config.out_dim,
            num_radial=config.num_radial,
            num_spherical=config.num_spherical,
            num_output_layers=config.num_output_layers,
        )
        
        self.readout_n = nn.Sequential(
            nn.Linear(config.out_dim, config.out_dim*3), nn.PReLU(), nn.Dropout(0.2),
            nn.Linear(config.out_dim, 1),
        )
        
    def forward(self, data):
        data = data.to_batch()
        z, pos, batch, mask = data.z, data.pos, data.batch, data.mask

        node_embed_feats = self.gnn(z, pos, batch) # (N, out_dim)

        out = self.readout_n(node_embed_feats[mask])[:, 0]
        
        return out
    

class SphereNetSpectraRegressor(nn.Module):
    def __init__(self, config):
        super(SphereNetSpectraRegressor, self).__init__()

        self.gnn = SphereNet(
            energy_and_force=False,
            cutoff=config.cutoff,
            num_layers=config.num_layers,
            hidden_channels=config.hidden_channels,
            out_channels=config.out_channels,
            int_emb_size=config.int_emb_size,
            basis_emb_size_dist=config.basis_emb_size_dist,
            basis_emb_size_angle=config.basis_emb_size_angle,
            basis_emb_size_torsion=config.basis_emb_size_torsion,
            out_emb_channels=config.out_emb_channels,
            num_spherical=config.num_spherical,
            num_radial=config.num_radial,
            envelope_exponent=config.envelope_exponent,
            num_before_skip=config.num_before_skip,
            num_after_skip=config.num_after_skip,
            num_output_layers=config.num_output_layers,
            output_init=config.output_init,
            use_node_features=config.use_node_features,
        )
        
        self.readout_n = nn.Sequential(
            nn.Linear(config.out_channels, config.out_channels*3), nn.PReLU(), nn.Dropout(0.2),
            nn.Linear(config.out_channels, 1),
        )
                
    def forward(self, data):
        data = data.to_batch()
        z, pos, batch, mask = data.z, data.pos, data.batch, data.mask

        _, node_embed_feats = self.gnn(z, pos, batch) # (N, out_channels)  
        
        out = self.readout_n(node_embed_feats[mask])[:, 0]
        
        return out