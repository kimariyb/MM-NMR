import torch
import torch.nn as nn
import torch.nn.functional as F

from models.geometry.comenet import ComENet
from models.graph.gin import GIN


class GeometricCrossAttention(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        # 初始化查询、键、值的投影层
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # 3D几何编码模块
        self.geo_encoder = nn.Sequential(
            nn.Linear(1, 32),  # 输入为距离标量
            nn.SiLU(),
            nn.Linear(32, n_heads)
        )
        
        # 2D拓扑编码模块
        self.top_encoder = nn.Sequential(
            nn.Linear(1, 32),  # 输入为拓扑距离
            nn.SiLU(),
            nn.Linear(32, n_heads))
        
        # 动态门控融合
        self.gate = nn.Sequential(
            nn.Linear(2*hidden_dim, 4*hidden_dim),
            nn.SiLU(),
            nn.Linear(4*hidden_dim, 2),
            nn.Softmax(dim=-1))
        
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, src_feat, tgt_feat, src_pos, tgt_edge_index):
        N, M = src_feat.size(0), tgt_feat.size(0)
        
        # 1. 投影到多头空间
        q = self.q_proj(src_feat).view(N, self.n_heads, self.head_dim)  # [N, H, d]
        k = self.k_proj(tgt_feat).view(M, self.n_heads, self.head_dim)  # [M, H, d]
        v = self.v_proj(tgt_feat).view(M, self.n_heads, self.head_dim)  # [M, H, d]
        
        # 2. 计算基础注意力
        attn = torch.einsum('nhd,mhd->nhm', q, k) / (self.head_dim**0.5)  # [N, H, M]
        
        # 3. 3D几何编码
        dist_3d = torch.cdist(src_pos, src_pos)  # [N, M]
        geo_bias = self.geo_encoder(dist_3d.unsqueeze(-1)).permute(2,0,1)  # [H, N, M]
        
        # 4. 2D拓扑编码（计算最短路径距离）
        adj = torch.zeros(M, M, device=src_feat.device)
        adj[tgt_edge_index[0], tgt_edge_index[1]] = 1
        
        # 使用 Floyd-Warshall 算法计算最短路径
        path_dist = adj.clone()
        for _ in range(3):  # 考虑最多3跳邻居
            path_dist = torch.min(path_dist, torch.einsum('nk,kj->nj', path_dist, adj))
        topo_dist = path_dist[torch.arange(M)[:,None], torch.arange(M)]  # [M, M]
        
        topo_bias = self.top_encoder(topo_dist.unsqueeze(-1)).permute(2,0,1)  # [H, M, M]
        
        # 5. 动态融合双路偏置
        gate = self.gate(torch.cat([src_feat, tgt_feat], dim=-1))  # [N, 2]
        fused_bias = gate[:,0].unsqueeze(1)*geo_bias + gate[:,1].unsqueeze(1)*topo_bias
        
        # 6. 应用注意力机制
        attn = attn + fused_bias.permute(1,0,2)  # [N, H, M]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # 7. 值聚合与输出
        output = torch.einsum('nhm,mhd->nhd', attn, v).reshape(N, -1)
        output = self.out_proj(output)
        
        return self.norm(src_feat + output)
    

class FusionSpectraRegressor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.comenet = ComENet(
            cutoff=config.cutoff, 
            num_layers=config.num_layers, 
            hidden_dim=config.hidden_dim, 
            middle_dim=config.hidden_dim, 
            out_dim=config.hidden_dim,
            num_radial=config.num_radial,
            num_spherical=config.num_spherical,
            num_output_layers=config.num_output_layers
        )
        
        self.gin = GIN(
            num_layers=config.num_layers, 
            node_dim=config.node_dim, 
            edge_dim=config.edge_dim, 
            hidden_dim=config.hidden_dim
        )
        
        self.cross_attn_2d3d = GeometricCrossAttention(config.hidden_dim, config.n_heads)
        self.cross_attn_3d2d = GeometricCrossAttention(config.hidden_dim, config.n_heads)
        
        self.fusion = nn.Sequential(
            nn.Linear(3*config.hidden_dim, 2*config.hidden_dim),
            nn.ELU(),
            nn.Linear(2*config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim)
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 3), nn.PReLU(), nn.Dropout(0.2),
            nn.Linear(config.hidden_dim * 3, 1)
        )

    def forward(self, data):
        # 特征提取
        z, pos, batch = data.z, data.pos, data.batch
        feat_3d = self.comenet(z, pos, batch)  # (N, F)
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        feat_2d = self.gin(x, edge_index, edge_attr)  # (N, F)
        
        # 双向交叉注意力
        attn_2d = self.cross_attn_2d3d(feat_2d, feat_3d, pos, edge_index)  # (N, F)
        attn_3d = self.cross_attn_3d2d(feat_3d, feat_2d, pos, edge_index)  # (N, F)
        
        mask = data.mask

        # 多尺度融合
        fused = self.fusion(torch.cat([attn_2d, attn_3d, feat_2d + feat_3d], dim=-1)) # (N, F)
        out = self.regressor(fused[mask])[:, 0]

        return out


class FusionConfig:
    def __init__(
        self, 
        node_dim,
        edge_dim,
        hidden_dim,
        cutoff,
        num_layers,
        num_radial,
        num_spherical,
        num_output_layers,
        n_heads,
    ):
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.cutoff = cutoff
        self.num_layers = num_layers
        self.num_radial = num_radial
        self.num_spherical = num_spherical
        self.num_output_layers = num_output_layers
        self.n_heads = n_heads

    @staticmethod
    def from_dict(config_dict):
        return FusionConfig(**config_dict)
     