import torch
import torch.nn as nn

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange


class BiCrossAttention(nn.Module):
    r"""
    Bi-directional cross-attention layer
    """
    def __init__(self, d_2d, d_3d, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        
        # 2D->3D
        self.Wq_2d = nn.Linear(d_2d, d_model)
        self.Wk_3d = nn.Linear(d_3d, d_model)
        self.Wv_3d = nn.Linear(d_3d, d_model)
        
        # 3D->2D 
        self.Wq_3d = nn.Linear(d_3d, d_model)
        self.Wk_2d = nn.Linear(d_2d, d_model)
        self.Wv_2d = nn.Linear(d_2d, d_model)
        
        # 输出层
        self.out_2d3d = nn.Linear(d_model, d_model)
        self.out_3d2d = nn.Linear(d_model, d_model)

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        return x.view(x.size(0), self.num_heads, self.head_dim)

    def forward(self, h_2d, h_3d):
        # --- 2D->3D 方向 ---
        Q_2d = self.split_heads(self.Wq_2d(h_2d))  # [N, num_heads, head_dim]
        K_3d = self.split_heads(self.Wk_3d(h_3d))
        V_3d = self.split_heads(self.Wv_3d(h_3d))

        attn_scores_2d3d = torch.einsum('nhd,mhd->nhm', Q_2d, K_3d) / (self.head_dim**0.5)
        attn_2d3d = torch.softmax(attn_scores_2d3d, dim=-1)

        out_2d3d = torch.einsum('nhm,mhd->nhd', attn_2d3d, V_3d)
        out_2d3d = out_2d3d.reshape(h_2d.size(0), self.d_model)
        out_2d3d = self.out_2d3d(out_2d3d)

        # --- 3D->2D 方向 ---
        Q_3d = self.split_heads(self.Wq_3d(h_3d))
        K_2d = self.split_heads(self.Wk_2d(h_2d))
        V_2d = self.split_heads(self.Wv_2d(h_2d))

        attn_scores_3d2d = torch.einsum('nhd,mhd->nhm', Q_3d, K_2d) / (self.head_dim**0.5)
        attn_3d2d = torch.softmax(attn_scores_3d2d, dim=-1)

        out_3d2d = torch.einsum('nhm,mhd->nhd', attn_3d2d, V_2d)
        out_3d2d = out_3d2d.reshape(h_3d.size(0), self.d_model)
        out_3d2d = self.out_3d2d(out_3d2d)

        # 层归一化
        fused = self.layer_norm(self.dropout(out_2d3d + out_3d2d))

        return fused


class TokenProjection(nn.Module):
    r"""
    Token projection layer 
    """
    def __init__(self, input_dim, output_dim):
        super(TokenProjection, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        return self.mlp(x)
    

class FeedForward(nn.Module):
    r"""
    Feed-forward layer
    """
    def __init__(self, input_dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
    
class Attention(nn.Module):
    def __init__(self, input_dim, num_heads=8, head_dim=64, dropout=0.):
        super().__init__()
        inner_dim = head_dim *  num_heads
        project_out = not (num_heads == 1 and head_dim == input_dim)

        self.num_heads = num_heads
        self.scale = head_dim ** -0.5

        self.norm = nn.LayerNorm(input_dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(input_dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, input_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        
        return out
            
            
class Transformer(nn.Module):
    r"""
    Transformer model
    """
    def __init__(self, input_dim, num_layers, num_heads, head_dim, ffn_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.layers = nn.ModuleList([])
        
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                Attention(input_dim, num_heads, head_dim, dropout),
                FeedForward(input_dim, ffn_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
           x = attn(x) + x
           x = ff(x) + x

        return self.norm(x)


class VisionTransformer(nn.Module):
    r"""
    Vision transformer model
    """
    def __init__(
        self, 
        seq_len, 
        patch_size, 
        num_classes, 
        hidden_dim, 
        num_layers, 
        num_heads, 
        mlp_dim, 
        channels = 3, 
        head_dim = 64, 
        dropout = 0., 
        emb_dropout = 0.
    ):
        super().__init__()
        assert (seq_len % patch_size) == 0

        num_patches = seq_len // patch_size
        patch_dim = channels * patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, hidden_dim))
        self.cls_token = nn.Parameter(torch.randn(hidden_dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(hidden_dim, num_layers,  num_heads, head_dim, mlp_dim, dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, series):
        x = self.to_patch_embedding(series)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, 'd -> b d', b = b)

        x, ps = pack([cls_tokens, x], 'b * d')

        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        cls_tokens, _ = unpack(x, ps, 'b * d')

        return self.mlp_head(cls_tokens)
    

        
        