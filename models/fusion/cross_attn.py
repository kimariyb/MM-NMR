import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class GeometricCrossAttention(nn.Module):
    r"""
    Geometric Cross-Attention Module.
    
    Parameters
    ----------
    dim_2d : int
        The dimension of the 2D feature.
    dim_3d : int
        The dimension of the 3D feature.
    cutoff : float
        The cutoff distance for the attention. Default: 5.0.
    num_heads : int
        The number of heads in multi-head attention.
    dropout : float
        The dropout rate. Default: 0.2.

    Inputs
    -------
    x_2d : torch.Tensor
        The 2D feature with shape of [N, D2].
    x_3d : torch.Tensor
        The 3D feature with shape of [N, D3].
    coords : torch.Tensor
        The coordinates of the 3D points with shape of [N, 3].
    
    Outputs
    -------
    torch.Tensor
        The output feature with shape of [N, D1].
    """
    def __init__(self, dim_2d, dim_3d, cutoff=5.0, num_heads=4, dropout=0.2):
        super().__init__()
       
        self.num_heads = num_heads
        self.cutoff = cutoff
        self.head_dim =  (dim_2d + dim_3d) // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.proj_q_2d = nn.Sequential(
            nn.Linear(dim_2d, dim_3d),
            nn.LayerNorm(dim_3d),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        self.coord_proj = nn.Sequential(
            nn.Linear(3, 128),
            nn.SiLU(),
            nn.Linear(128, num_heads)
        )
        
        self.gate_2d = nn.Sequential(
            nn.Linear(dim_3d, 1),
            nn.Sigmoid()
        )
        self.gate_3d = nn.Sequential(
            nn.Linear(dim_3d, 1),
            nn.Sigmoid()
        )
        
        self.norm = nn.LayerNorm(dim_3d)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1e-2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x_2d, x_3d, coords):
        q = self.proj_q_2d(x_2d) # [N, D3]

        attn_out = self.geometric_attention(q, x_3d, coords)
        
        gate_2d = self.gate_2d(q)
        gate_3d = self.gate_3d(x_3d)
        
        fused = gate_2d * attn_out + gate_3d * x_3d
        fused = self.dropout(self.norm(fused))
        
        return fused
    
    def geometric_attention(self, q, k, coords):
        dist_matrix = torch.cdist(coords, coords)  # [N, N]
        mask = (dist_matrix < self.cutoff) & (dist_matrix > 0)  # 排除自身和远距离

        coord_bias = self.coord_proj(coords)  # [N, H]
        coord_bias = torch.einsum('nh,mh->nmh', coord_bias, coord_bias)  # [N, N, H]

        q = rearrange(q, 'n (h d) -> h n d', h=self.num_heads)  # [H, N, D]
        k = rearrange(k, 'n (h d) -> h n d', h=self.num_heads)
        
        attn = torch.einsum('hnd,hmd->hnm', q, k) * self.scale  # [H, N, N]
        attn = attn + coord_bias.permute(2,0,1)  # 注入几何偏置
        
        attn = attn.masked_fill(~mask, -1e9)
        attn = F.softmax(attn, dim=-1)
        
        v = rearrange(k, 'h n d -> h d n')  # [H, D, N]
        output = torch.einsum('hnm,hdm->hnd', attn, v)  # [H, N, D]
        return rearrange(output, 'h n d -> n (h d)')  # [N, D3]


class GeometricBiCrossAttention(nn.Module):
    r"""
    Geometry Bi-Cross-Attention Module.
    
    Parameters
    ----------
    dim_2d : int
        The dimension of the 2D feature.
    dim_3d : int
        The dimension of the 3D feature.
    num_heads : int
        The number of heads in multi-head attention.
    cutoff : float
        The cutoff distance for the attention. Default: 5.0.
    dropout : float
        The dropout rate. Default: 0.2.

    Inputs
    -------
    x_2d : torch.Tensor
        The 2D feature with shape of [N, D2].
    x_3d : torch.Tensor
        The 3D feature with shape of [N, D3].

    
    Outputs
    -------
    torch.Tensor
        The fused feature with shape of [N, fusion_dim].
    """
    def __init__(self, dim_2d, dim_3d, num_heads=4, cutoff=5.0, dropout=0.2):
        super().__init__()
        self.cross_2d3d = GeometricCrossAttention(dim_2d, dim_3d, cutoff, num_heads, dropout)
        self.cross_3d2d = GeometricCrossAttention(dim_3d, dim_2d, cutoff, num_heads, dropout)
        
        self.fusion = nn.Sequential(
            nn.Linear(dim_2d + dim_3d, 4 * dim_3d),
            nn.SiLU(),
            nn.LayerNorm(4 * dim_3d),
            nn.Linear(4 * dim_3d, dim_3d),
            nn.Dropout(dropout),
        )
        
        self.layer_scale = nn.Parameter(torch.ones(1, dim_3d))
        self.drop_path = nn.Dropout(dropout)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.layer_scale, std=1e-6)
        
    def forward(self, x_2d, x_3d, coords):
        # 双向注意力
        attn_2d = self.cross_2d3d(x_2d, x_3d, coords)
        attn_3d = self.cross_3d2d(x_3d, x_2d, coords)
        
        fused = self.fusion(torch.cat([attn_2d, attn_3d], dim=-1))
        
        fused = x_3d + self.drop_path(self.layer_scale * fused)     
           
        return F.layer_norm(fused, fused.shape[-1:])