import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    r"""
    Cross-Attention Module.
    
    Parameters
    ----------
    feat_1_dim : int
        The dimension of the first feature.
    feat_2_dim : int
        The dimension of the second feature.
    num_heads : int
        The number of heads in multi-head attention.
    dropout : float
        The dropout rate. Default: 0.2.
    
    Inputs
    -------
    feat_1 : torch.Tensor
        The first feature with shape of [N, D1].
    feat_2 : torch.Tensor
        The second feature with shape of [N, D2].
    
    Outputs
    -------
    torch.Tensor
        The output feature with shape of [N, D1].
    """
    def __init__(self, feat_1_dim, feat_2_dim, num_heads=4, dropout=0.2):
        super().__init__()
        assert feat_1_dim % num_heads == 0, "feat_1_dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = feat_1_dim // num_heads
        self.scale = 1 / math.sqrt(self.head_dim)
        
        # 增强投影层：添加 LayerNorm 和更合理的维度映射
        self.norm_q = nn.LayerNorm(feat_1_dim)
        self.q_proj = nn.Linear(feat_1_dim, feat_1_dim)
        
        self.norm_kv = nn.LayerNorm(feat_2_dim)
        self.kv_proj = nn.Linear(feat_2_dim, 2 * feat_1_dim)  # 修正维度映射
        
        # 注意力机制增强
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        # 输出变换层
        self.out_proj = nn.Sequential(
            nn.Linear(feat_1_dim, feat_1_dim),
            nn.LayerNorm(feat_1_dim)
        )
        
        # 参数初始化
        self._init_weights()

    def _init_weights(self):
        # Xavier初始化增强稳定性
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.kv_proj.weight)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.kv_proj.bias)
        
    def forward(self, feat_1, feat_2):
        """优化后的前向传播流程"""
        # 输入标准化
        q = self.norm_q(feat_1)
        kv = self.norm_kv(feat_2)
        
        # 投影变换
        q = self.q_proj(q)  # [N, D1]
        k, v = self.kv_proj(kv).chunk(2, dim=-1)  # [N, D1], [N, D1]
        
        # 多头拆分
        q = q.view(-1, self.num_heads, self.head_dim)  # [N, H, D]
        k = k.view(-1, self.num_heads, self.head_dim)  # [N, H, D] 
        v = v.view(-1, self.num_heads, self.head_dim)  # [N, H, D]
        
        # 带稳定机制的注意力计算
        attn = torch.einsum('nhd,nhd->nh', q, k) * self.scale
        attn = attn - attn.amax(dim=-1, keepdim=True)  # 数值稳定技巧
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn) # [N, H]
        
        # 特征聚合
        out = torch.einsum('nh,nhd->nhd', attn, v)  # [N, H, D]
        out = out.reshape(-1, self.num_heads * self.head_dim)  # [N, D]
        
        # 残差连接
        out = self.proj_dropout(out)

        return self.out_proj(out + feat_1)


class BiCrossAttention(nn.Module):
    r"""
    Bi-Cross-Attention Module.
    
    Parameters
    ----------
    dim_2d : int
        The dimension of the 2D feature.
    dim_3d : int
        The dimension of the 3D feature.
    num_heads : int
        The number of heads in multi-head attention.
    fusion_dim : int
        The dimension of the fused feature. Default: 512.
    
    Inputs
    -------
    feat_2d : torch.Tensor
        The 2D feature with shape of [N, D2].
    feat_3d : torch.Tensor
        The 3D feature with shape of [N, D3].
    
    Outputs
    -------
    torch.Tensor
        The fused feature with shape of [N, fusion_dim].
    """
    def __init__(self, dim_2d, dim_3d, num_heads=4, fusion_dim=512):
        super().__init__()
        # 双向注意力模块增强
        self.attn_2d = CrossAttention(
            feat_1_dim=dim_2d,
            feat_2_dim=dim_3d,
            num_heads=num_heads,
            dropout=0.1
        )
        self.attn_3d = CrossAttention(
            feat_1_dim=dim_3d,
            feat_2_dim=dim_2d,
            num_heads=num_heads,
            dropout=0.1
        )
        
        # 动态融合层
        self.fusion = nn.Sequential(
            nn.Linear(dim_2d + dim_3d, fusion_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_dim),
            nn.Dropout(0.2)
        )
        
    def forward(self, feat_2d, feat_3d):
        # 交叉注意力
        feat_2d_out = self.attn_2d(feat_2d, feat_3d)
        feat_3d_out = self.attn_3d(feat_3d, feat_2d)
        
        # 门控融合机制
        fused = torch.cat([feat_2d_out, feat_3d_out], dim=-1)

        return self.fusion(fused)