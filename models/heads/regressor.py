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
            nn.Linear(64, 1)   
        )
        
    def forward(self, x):
        return self.predictor(x)
    

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
    def __init__(self, gnn_args, sphere_args, model_args, mean, std):
        super().__init__()
        self.gnn_args = gnn_args
        self.sphere_args = sphere_args
        self.model_args = model_args
        
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
        self.gnn = GraphNet(**self.gnn_args)
        
        # Spherenet
        self.sphere = SphereNet(**self.sphere_args)

        # Projection
        self.pro_t = nn.Sequential(
            nn.Linear(self.gnn_args.hiiden_dim, self.model_args.proj_dim), 
            nn.ReLU(inplace=True),
            nn.Linear(self.model_args.proj_dim, self.model_args.proj_dim)
        )
        
        self.pro_g = nn.Sequential(
            nn.Linear(self.sphere_args.hidden_dim, self.model_args.proj_dim), 
            nn.ReLU(inplace=True),
            nn.Linear(self.model_args.proj_dim, self.model_args.proj_dim)
        )
        
        # Bi-Cross attention
        self.cross_attn = BiCrossAttention(
            hidden_dim=self.model_args.hidden_dim,
            num_heads=self.model_args.num_heads,
            head_dim=self.model_args.head_dim,
            dropout=self.model_args.dropout,
        )

        # Fusion
        self.fusion_dim = (self.gnn_args.hidden_dim + self.sphere_args.hidden_dim) / 2
        self.fusion = WeightFusion(2, self.fusion_dim)

        # Predict head
        self.predictor = PredictorRegressor(self.fusion_dim)

        # dropout
        self.dropout = nn.Dropout(self.model_args.dropout)


    def forward(self, data):
        x, edge_index, edge_attr, pos, z, batch, mask = data.x, data.edge_index, data.edge_attr, data.pos, data.z, data.batch, data.mask
        # get the node representations
        graph_t, node_t = self.gnn(x, edge_index, edge_attr, batch)
        graph_g, node_g = self.sphere(z, pos, batch)
        
        # bi-cross attention
        graph_t, graph_g = self.cross_attn(graph_t, graph_g)    
        
        # fusion node representations
        node_fused = self.node_fusion(torch.cat([node_t, node_g], dim=-1))
        node_fused = self.dropout(node_fused)
        
        # predict and normalize
        pred = self.predictor(node_fused)
        nmr_pred = pred[mask][:, 0]
        if self.std is not None:
            nmr_pred = nmr_pred * self.std
        if self.mean is not None:
            nmr_pred = nmr_pred + self.mean

        # Contrastive loss calculation
        proj_t = self.pro_t(graph_t)
        proj_g = self.pro_g(graph_g)
        
        return nmr_pred, proj_t, proj_g, mask

    def calc_label_loss(self, pred, label, mask):
        r"""
        Calculate the label loss.
        """
        return self.mae_loss(pred, label[mask])

    def calc_contrastive_loss(self, x1, x2, T=0.1):
        r"""
        Calculate the contrastive loss.
        """
        batch_size, _ = x1.shape

        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)

        pos_sim = sim_matrix[range(batch_size), range(batch_size)]

        loss = -torch.log(
            pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        ).mean()

        return loss

    def calc_loss(self, pred, label, mask, z1, z2, alpha=0.8):
        loss1 = self.calc_label_loss(pred, label, mask)
        loss2 = self.calc_contrastive_loss(z1, z2)

        loss = loss1 + alpha * loss2

        return loss, loss1, loss2
