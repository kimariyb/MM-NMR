import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoders.graph import GraphNet
from models.encoders.geometry import ComENet
from models.fusion.cross_attn import GeometricBiCrossAttention


class PredictorRegressor(nn.Module):
    def __init__(self, input_dim):
        super(PredictorRegressor, self).__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2), 
            nn.PReLU(), 
            nn.Dropout(0.2),
            nn.Linear(input_dim * 2, 1)
        )

    def forward(self, x):
        return self.predictor(x)
    

class MultiModalFusionRegressor(nn.Module):
    r"""
    Multi-modal fusion regressor module. 
    """
    def __init__(self, gnn_args, geom_args, num_heads, mean, std):
        super().__init__()
        self.gnn_args = gnn_args
        self.geom_args = geom_args
        self.num_heads = num_heads

        # GNN
        self.graph_net = GraphNet(**gnn_args)
        self.geometry_net = ComENet(**geom_args)

        # Cross-attention        
        self.cross_attn = GeometricBiCrossAttention(
            dim_2d=self.gnn_args['hidden_dim'],
            dim_3d=self.geom_args['hidden_dim'],
            num_heads=num_heads,
            cutoff=self.geom_args['cutoff'],
            dropout=0.2
        )

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
        node_t = self.graph_net(x, edge_index, edge_attr, batch) # (batch_size, out_channels), (num_nodes, out_channels)
        node_g = self.geometry_net(z, pos, batch) # (batch_size, out_channels), (num_nodes, out_channels)

        # fuse the node representations
        node_fused = self.cross_attn(node_t, node_g, pos) # (batch_size, out_channels * 2)

        # predict and normalize
        pred = self.predictor(node_fused)
        nmr_pred = pred[mask]
      
        return (nmr_pred * self.std) + self.mean



   