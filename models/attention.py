import torch
import torch.nn as nn
import torch.nn.functional as F

from flash_attn import flash_attn_varlen_qkvpacked_func


class AttnModule(nn.Module):
    def __init__(self, dim_h, num_heads, dropout, batch_first=True):
        super(AttnModule, self).__init__()
        self.dim_h = dim_h
        self.num_heads = num_heads
        self.head_dim = dim_h // num_heads
        self.wqkv = nn.Linear(dim_h, dim_h * 3, bias=True)
        self.wo = nn.Linear(dim_h, dim_h, bias=True)
        self.dropout_p = dropout

    def forward(self, x, cumsum, max_seqlen):
        qkv = self.wqkv(x)
        qkv = qkv.view(x.shape[0], 3, self.num_heads, self.head_dim)
        out = flash_attn_varlen_qkvpacked_func(
            qkv.bfloat16(),
            cumsum,
            max_seqlen=x.shape[0],  # didn't observe any significant effects on throughput
            dropout_p=self.dropout_p,
            causal=False,
        )
        if torch.isnan(out).sum() > 0:
            raise ValueError("NaN in output")

        out = out.contiguous().view(*x.shape).to(x.dtype)
        out = self.wo(out)
        return out


class SwiGLU(nn.Module):
    def __init__(self, dim_h, expansion):
        super(SwiGLU, self).__init__()
        self.dim_h = dim_h
        self.inter_dim = int(dim_h * expansion)
        self.w1w3 = nn.Linear(dim_h, self.inter_dim * 2)
        self.w2 = nn.Linear(self.inter_dim, dim_h)

    def forward(self, x):
        x1, x3 = self.w1w3(x).view(x.shape[0], 2, self.inter_dim).unbind(1)
        return self.w2(F.gelu(x1) * x3)


class TfModule(nn.Module):
    def __init__(
        self,
        dim_h,          # 隐藏层维度
        num_heads,      # 注意力头的数量
        num_layers,     # 层数
        dropout,        # dropout率
        new_arch=True,  # 是否使用新架构
        expansion=8 / 3,# SwiGLU扩展因子
        batch_first=True, # batch维度是否在第一个
    ):
        super(TfModule, self).__init__()  # 调用父类初始化方法
        # 将输入投影到隐藏层维度
        self.proj_to_tf = torch.nn.LazyLinear(dim_h)
        # 创建注意力层列表
        self.attn_layers = torch.nn.ModuleList(
            [
                AttnModule(dim_h, num_heads, dropout, batch_first)
                for _ in range(num_layers)
            ]
        )

        # 输入层归一化
        self.proj_norm = torch.nn.LayerNorm(128)

        if new_arch:  # 如果使用新架构
            # 注意力层归一化
            self.attn_norm = nn.ModuleList(
                [torch.nn.LayerNorm(dim_h) for _ in range(num_layers)]
            )
            # SwiGLU前馈层
            self.fc_layers = torch.nn.ModuleList(
                [SwiGLU(dim_h, expansion) for _ in range(num_layers)]
            )
            # MLP层归一化
            self.mlp_norm = nn.ModuleList(
                [torch.nn.LayerNorm(dim_h) for _ in range(num_layers)]
            )
            # 设置前向传播方法为新架构版本
            self.tf_fwd = self.fwd_new
        else:  # 如果使用旧架构
            # 输入层归一化
            self.norm_input = nn.ModuleList(
                [torch.nn.LayerNorm(dim_h) for _ in range(num_layers)]
            )
            # 注意力层归一化
            self.attn_norm = nn.ModuleList(
                [torch.nn.LayerNorm(dim_h) for _ in range(num_layers)]
            )
            # 线性前馈层
            self.fc_layers = torch.nn.ModuleList(
                [torch.nn.Linear(dim_h, dim_h) for _ in range(num_layers)]
            )
            # 设置前向传播方法为旧架构版本
            self.tf_fwd = self.fwd_old

    def fwd_old(self, x, cumsum_seq, max_len):
        for i, attn_layer in enumerate(self.attn_layers):
            x = self.norm_input[i](x)
            attn = attn_layer(x, cumsum_seq, max_len)
            x = F.gelu(x + attn)
            fc = self.fc_layers[i](x)
            x = x + fc
            x = self.attn_norm[i](x)
        return x

    def fwd_new(self, x, cumsum_seq, max_len):
        for i, attn_layer in enumerate(self.attn_layers):
            attn = attn_layer(x, cumsum_seq, max_len)
            h = x + self.attn_norm[i](attn)
            fc = self.fc_layers[i](h)
            x = h + self.mlp_norm[i](fc)
        return x

    def forward(self, x, cumsum_seq, max_len):
        x = self.proj_norm(x)

        x = self.proj_to_tf(x)

        return self.tf_fwd(x, cumsum_seq, max_len)
