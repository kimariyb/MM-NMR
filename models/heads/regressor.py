import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GlobalAttention

from models.encoders.graph import GraphNet
from models.encoders.geometry import SphereNet



class PredictorRegressor(nn.Module):
    def __init__(self, input_dim):
        super(PredictorRegressor, self).__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, 128), nn.PReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.PReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1)   
        )
        
    def forward(self, x):
        return self.predictor(x)


class BatchGlobalAttention(nn.Module):
    r"""
    Batch-wise global attention module.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.global_attn = GlobalAttention(gate_nn=nn.Linear(hidden_size, 1))

    def forward(self, x, batch):
        return self.global_attn(x, batch)
    

class WeightFusion(nn.Module):
    r"""
    Weight fusion module. 
    
    Parameters
    ----------
    feat_views : int
        Number of feature views.
    feat_dim : int
        Dimension of feature.
    bias : bool, optional
        If set to :obj:`False`, the layer will not learn an additive bias. (default: :obj:`True`)
    """
    def __init__(self, feat_views, feat_dim, bias: bool = True):
        super(WeightFusion, self).__init__()
        self.feat_views = feat_views
        self.feat_dim = feat_dim
        self.weight = nn.Parameter(torch.empty((1, 1, feat_views)))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(int(feat_dim)))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # 输入形状: (batch_size, feat_views, feat_dim)
        # 权重形状: (1, feat_views, 1) 通过广播机制进行逐元素相乘
        fused = torch.sum(input * self.weight, dim=1)  # 沿特征视图维度求和
        
        if self.bias is not None:
            fused += self.bias
        
        return fused
    

class MultiModalFusionRegressor(nn.Module):
    r"""
    Multi-modal fusion regressor module. 
    """
    def __init__(self, gnn_args, sphere_args, proj_dim, dropout):
        super().__init__()
        self.gnn_args = gnn_args
        self.sphere_args = sphere_args
        self.proj_dim = proj_dim
        self.dropout = dropout
        
        # GNN
        self.gnn = GraphNet(
            self.gnn_args.num_layers,
            self.gnn_args.hidden_dim,
            self.gnn_args.JK,
            self.gnn_args.dropout,
            self.gnn_args.gnn_type,
            self.gnn_args.graph_pooling,
        )
        
        # Spherenet
        self.sphere = SphereNet(
            cutoff=self.sphere_args.cutoff,
            num_layers=self.sphere_args.num_layers,
            hidden_channels=self.sphere_args.hidden_dim,
            out_channels=self.sphere_args.hidden_dim,
            int_emb_size=self.sphere_args.int_emb_size,
            basis_emb_size_dist=self.sphere_args.basis_emb_size_dist,
            basis_emb_size_angle=self.sphere_args.basis_emb_size_angle,
            basis_emb_size_torsion=self.sphere_args.basis_emb_size_torsion,
            out_emb_channels=self.sphere_args.out_emb_channels,
            num_spherical=self.sphere_args.num_spherical,
            num_radial=self.sphere_args.num_radial,
            envelope_exponent=self.sphere_args.envelope_exponent,
            num_before_skip=self.sphere_args.num_before_skip,
            num_after_skip=self.sphere_args.num_after_skip,
            num_output_layers=self.sphere_args.num_output_layers,
        )

        # Projection
        self.pro_t = nn.Sequential(
            nn.Linear(self.gnn_args.hiiden_dim, self.proj_dim), 
            nn.ReLU(inplace=True),
            nn.Linear(self.proj_dim, self.proj_dim)
        )
        
        self.pro_g = nn.Sequential(
            nn.Linear(self.sphere_args.hidden_dim, self.proj_dim), 
            nn.ReLU(inplace=True),
            nn.Linear(self.proj_dim, self.proj_dim)
        )

        # Fusion
        self.fusion_dim = (self.gnn_args.hidden_dim + self.sphere_args.hidden_dim) / 2
        self.fusion = WeightFusion(2, self.fusion_dim)
        
        # Pooling
        self.pooled_t = BatchGlobalAttention(self.gnn_args.hidden_dim)
        self.pooled_g = BatchGlobalAttention(self.sphere_args.hidden_dim)

        # predict head
        self.predictor = PredictorRegressor(self.fusion_dim)


    def forward(self, x, edge_index, edge_attr, pos, z, batch, mask):
        # get the node representations
        node_t = self.gnn(x, edge_index, edge_attr)
        node_g = self.sphere(z, pos, batch)
        
        # fusion node representations
        node_fused = self.node_fusion(torch.cat([node_t, node_g], dim=-1))
        node_fused = F.prelu(node_fused, 0.2)
        node_fused = F.dropout(node_fused, self.dropout, training=self.training)
        
        # predict
        pred = self.predictor(node_fused)

        # Contrastive loss calculation
        # get the graph representations
        graph_t = self.pooled_t(node_t, batch)
        graph_g = self.pooled_g(node_g, batch)
        
        # projection
        proj_t = self.pro_t(graph_t)
        proj_g = self.pro_g(graph_g)
        
        
        return pred, proj_t, proj_g, batch
