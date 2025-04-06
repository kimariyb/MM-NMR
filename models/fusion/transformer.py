import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import global_mean_pool
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from models.encoders.graph import GraphNet
from models.encoders.geometry import SphereNet


class TokenProjection(nn.Module):
    r"""
    Token projection layer for 2D and 3D inputs.
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


class FusionTransformer(nn.Module):
    r"""
    The Fusion Transformer model. It takes 2D and 3D inputs and outputs their fusion.
    The model is based on the ViT (Vision Transformer) architecture.
    """
    def __init__(
        self, 
        input_dim_2d,
        input_dim_3d,
        hidden_dim, 
        num_layers=1, 
        num_heads=8, 
        dropout=0.
    ):
        super(FusionTransformer, self).__init__()
        self.input_dim_2d = input_dim_2d
        self.input_dim_3d = input_dim_3d
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Token Creation, t: 2D, g: 3D, tg: 2D+3D
        self.tau_t = TokenProjection(input_dim_2d, hidden_dim)
        self.tau_g = TokenProjection(input_dim_3d, hidden_dim)
        self.tau_tg = TokenProjection(input_dim_2d+input_dim_3d, hidden_dim)
        
        # Fusion Transformer (based on ViT)
        self.transformer_layers = Transformer(hidden_dim, num_layers, num_heads, hidden_dim, hidden_dim, dropout)
        
        self.pos_embedding = nn.Parameter(torch.randn(1, 4, hidden_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.to_latent = nn.Identity()
                
        # Initialize parameters
        self.reset_parameters()        
        
    def forward(self, h_t, h_g):
        # Token Creation
        h_t = self.tau_t(h_t) # (batch_size, hidden_dim)
        h_g = self.tau_g(h_g) # (batch_size, hidden_dim)
        h_tg = self.tau_tg(torch.cat([h_t, h_g], dim=-1)) # (batch_size, hidden_dim)
        
        # Fusion Transformer
        x = torch.stack([h_t, h_g, h_tg], dim=1) # (batch_size, 3, hidden_dim)
        
        b, n, _ = x.shape
        
        # add CLS token and position embedding
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)] 
        
        # Transformer Encoder
        x = self.transformer_encoder(x) # (batch_size, 4, hidden_dim)
        x = x.mean(dim=1) # (batch_size, hidden_dim)
        x = self.to_latent(x) # (batch_size, hidden_dim)
        
        return z_t, z_g, z_tg