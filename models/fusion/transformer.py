import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import global_mean_pool
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

from models.encoders.graph import GraphNet
from models.encoders.geometry import SphereNet


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
    
    
class MultiHeadAttention(nn.Module):
    r"""
    Multi-head attention layer 
    """
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
        
        return self.to_out(out)
            
            
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
                MultiHeadAttention(input_dim, num_heads, head_dim, dropout),
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
    
    
class FusionTransformer(nn.Module):
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

        self.transformer = VisionTransformer(
            seq_len, 
            patch_size, 
            num_classes, 
            hidden_dim, 
            num_layers, 
            num_heads, 
            mlp_dim, 
            channels, 
            head_dim, 
            dropout, 
            emb_dropout
        )
        
        self.graph_net = GraphNet(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.sphere_net = SphereNet(hidden_dim, hidden_dim, hidden_dim, dropout)
        
        self.tau_t = TokenProjection(hidden_dim, hidden_dim)
        self.tau_g = TokenProjection(hidden_dim, hidden_dim)
        self.tau_tg = TokenProjection(hidden_dim, hidden_dim)
        
        