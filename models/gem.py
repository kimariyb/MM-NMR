import torch
import torch.nn as nn
import dgl

from dgl.nn.pytorch import GINEConv
from dgl.nn import SumPooling, AvgPooling, MaxPooling 

from models.encoder import AtomEmbedding, BondEmbedding, BondFloatRBF, BondAngleFloatRBF


class GraphPool(nn.Module):
    r"""
    This is an implementation of graph pooling
    """
    def __init__(self, pool_type="sum"):
        super(GraphPool, self).__init__()
        self.pool_type = pool_type.lower()

        if self.pool_type == "sum":
            self.pool = SumPooling()
        elif self.pool_type == "avg":
            self.pool_func = AvgPooling()
        elif self.pool_type == "max":
            self.pool_func = MaxPooling()
        else:
            raise ValueError("pool_type must be one of sum, mean, max")

    def forward(self, g: dgl.DGLGraph, feat):
        return self.pool(feat,  g.batch_num_nodes()) 


class GraphNorm(nn.Module):
    def __init__(self):
        super(GraphNorm, self).__init__()
        self.node_counter = GraphPool("sum")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, g, feat):
        # 计算每个图的节点数 
        ones = torch.ones(g.num_nodes(),  1, device=feat.device) 
        node_counts = self.node_counter(g,  ones)  # (batch_size, 1)
        
        # 扩展为每个节点对应的计数 
        batch_counts = torch.repeat_interleave( 
            torch.sqrt(node_counts), 
            repeats=g.batch_num_nodes(), 
            dim=0 
        )
        
        # 执行归一化 
        return feat / batch_counts 


class GeoGNNBlock(nn.Module):
    def __init__(self, embed_dim, dropout):
        super(GeoGNNBlock, self).__init__()

        self.embed_dim = embed_dim

        self.gnn = GINEConv(
            nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim * 2),
                nn.LeakyReLU(0.2),
                nn.Linear(self.embed_dim * 2, self.embed_dim),
            ),
        )

        self.layer_norm = nn.LayerNorm(self.embed_dim)
        self.graph_norm = GraphNorm()

        self.act = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, node_feats, edge_feats):
        out = self.gnn(g, node_feats, edge_feats)
        out = self.layer_norm(out)
        out = self.graph_norm(g, out)
        out = self.act(out)
        out = self.dropout(out)

        return out + node_feats


class GeoGNN(nn.Module):
    r"""
    The GeoGNN model used in GEM
    """
    def __init__(self, num_layers, node_dim, edge_dim, embed_dim, dropout):
        super(GeoGNN, self).__init__()

        self.embed_dim = embed_dim


      