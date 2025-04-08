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
        
        mean = torch.scalar_tensor(0) if mean is None else mean
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean).float()
        self.register_buffer('mean', mean)

        std = torch.scalar_tensor(1) if std is None else std
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std).float()
        self.register_buffer('std', std)

        self.mae_loss = nn.L1Loss()
        
        # GNN
        self.gnn = GraphNet(**gnn_args)
        
        # Spherenet
        self.sphere = SphereNet(**sphere_args)

        # Projection
        self.pro_t = nn.Sequential(
            nn.Linear(self.gnn_args['out_channels'], 256), 
            nn.ReLU(inplace=True),
            nn.Linear(256, 256)
        )
        
        self.pro_g = nn.Sequential(
            nn.Linear(self.sphere_args['out_channels'], 256), 
            nn.ReLU(inplace=True),
            nn.Linear(256, 256)
        )
        
        # Bi-Cross attention Fusion
        self.fusion_dim = self.gnn_args['out_channels'] + self.sphere_args['out_channels']
        self.cross_attn = BiCrossAttention(
            d_2d=self.gnn_args['out_channels'],
            d_3d=self.sphere_args['out_channels'],
            d_model=self.fusion_dim,
            num_heads=8,
            dropout=0.2
        )

        # Predict head
        self.predictor = PredictorRegressor(self.fusion_dim)

    def forward(self, data):
        x, edge_index, edge_attr, pos, z, batch, mask = data.x, data.edge_index, data.edge_attr, data.pos, data.z, data.batch, data.mask
        # get the node representations
        _, node_t = self.gnn(x, edge_index, edge_attr, batch) # (batch_size, out_channels), (num_nodes, out_channels)
        _, node_g = self.sphere(z, pos, batch) # (batch_size, out_channels), (num_nodes, out_channels)
        
        # fusion node representations
        node_fused = self.cross_attn(node_t, node_g)
        
        # predict and normalize
        pred = self.predictor(node_fused)
        nmr_pred = pred[mask]

        if self.std is not None:
            nmr_pred = nmr_pred * self.std
        if self.mean is not None:
            nmr_pred = nmr_pred + self.mean

        return nmr_pred


   