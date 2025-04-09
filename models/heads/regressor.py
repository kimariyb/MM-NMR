import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoders.graph import GraphNet
from models.encoders.geometry import SphereNet
from models.fusion.transformer import BiCrossAttention


class PredictorRegressor(nn.Module):
    def __init__(self, input_dim):
        super(PredictorRegressor, self).__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, 128), nn.PReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.PReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )
        
    def forward(self, x):
        return self.predictor(x)
    

class MultiModalFusionRegressor(nn.Module):
    r"""
    Multi-modal fusion regressor module. 
    """
    def __init__(self, gnn_args, sphere_args, mean, std):
        super().__init__()
        self.gnn_args = gnn_args
        self.sphere_args = sphere_args

        # GNN
        self.gnn = GraphNet(**gnn_args)
        self.norm_2d = nn.LayerNorm(self.gnn_args['out_channels'])
        self.drop_2d = nn.Dropout(0.2)
        # Spherenet
        self.sphere = SphereNet(**sphere_args)
        self.norm_3d = nn.LayerNorm(self.sphere_args['out_channels'])
        self.drop_3d = nn.Dropout(0.2)

        # Cross-attention
        self.cross_attn = BiCrossAttention(self.gnn_args['out_channels'], self.sphere_args['out_channels'], 4)
        self.fusion_dim = self.gnn_args['out_channels'] + self.sphere_args['out_channels']
        # Predict head
        self.predictor = PredictorRegressor(self.fusion_dim)

        mean = torch.scalar_tensor(0) if mean is None else mean
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean).float()
        self.register_buffer('mean', mean)

        std = torch.scalar_tensor(1) if std is None else std
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std).float()
        self.register_buffer('std', std)

    def forward(self, data):
        x, edge_index, edge_attr, pos, z, batch, mask = data.x, data.edge_index, data.edge_attr, data.pos, data.z, data.batch, data.mask
        # get the node representations
        _, node_t = self.gnn(x, edge_index, edge_attr, batch) # (batch_size, out_channels), (num_nodes, out_channels)
        _, node_g = self.sphere(z, pos, batch) # (batch_size, out_channels), (num_nodes, out_channels)
        
        node_t = self.norm_2d(self.drop_2d(node_t))
        node_g = self.norm_3d(self.drop_3d(node_g))

        # fuse the node representations
        node_fused = self.cross_attn(node_t, node_g) # (batch_size, out_channels * 2)

        # predict and normalize
        pred = self.predictor(node_fused.clamp(-1e4, 1e4))
        nmr_pred = pred[mask]
      
        return (nmr_pred * self.std) + self.mean



   