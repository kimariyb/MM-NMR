import dgl

import torch
import torch.nn as nn
import torch.nn.functional as F


class CMPNNLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        # 边特征更新网络（融合原子-键联合特征）
        self.edge_gru = nn.GRUCell((2 * node_dim) + edge_dim, edge_dim)
        
        # 原子特征更新网络（保留原文GRU设计）
        self.atom_gru = nn.GRUCell(hidden_dim, node_dim)
        
        # 消息转换网络（增加边权重机制）
        self.msg_booster = nn.Sequential(
            nn.Linear((2 * node_dim) + edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 残差连接
        self.atom_res = nn.Linear(node_dim, node_dim)
        self.edge_res = nn.Linear(edge_dim, edge_dim)

    def forward(self, g: dgl.DGLGraph, node_feats, edge_feats):
        with g.local_scope():
            g.ndata['h'] = node_feats
            g.edata['e'] = edge_feats
            
            # 边特征更新
            g.apply_edges(self.edge_update_func)
            edge_input = self.msg_booster(g.edata['msg'])
            update_edges = F.leaky_relu(
                self.edge_gru(edge_input, edge_feats) + self.edge_res(edge_feats)
            )

            # 消息传递
            g.edata['e'] = update_edges
            g.update_all(self.message_func, self.reduce_func)
            node_input = self.msg_booster(g.ndata['agg'])
            update_nodes = F.leaky_relu(
                self.atom_gru(node_input, node_feats) + self.atom_res(node_feats)
            )

        # 输出
        return update_nodes, update_edges
            
    def edge_update_func(self, edges):
        # 边特征更新函数，保留原文设计
        return {'msg': torch.cat([edges.src['h'], edges.dst['h'], edges.data['e']], dim=1)}

    def message_func(self, edges):
        # 消息传递函数，结合消息转换网络
        return {'m': torch.cat([edges.src['h'], edges.data['e']], dim=1)}

    def reduce_func(self, nodes):
        # 消息聚合函数，保留原文设计
        return {'agg': torch.cat([
            nodes.mailbox['m'].sum(dim=1),
            nodes.mailbox['m'].max(dim=1)[0]
        ], dim=1)}


class CMPNN(nn.Module):
    def __init__(self, num_layers=3, node_dim=64, edge_dim=32, hidden_dim=128, dropout=0.2):
        super().__init__()

        self.atom_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.bond_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
    
        # 多轮消息传递
        self.layers = nn.ModuleList([
            CMPNNLayer(node_dim, edge_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        
    def forward(self, g: dgl.DGLGraph):
        h = g.ndata['h']
        e = g.edata['e']

        # 特征初始化
        h = F.leaky_relu(self.atom_encoder(h))
        e = F.leaky_relu(self.bond_encoder(e))
    
        # 消息传递
        for layer in self.layers:
            h, e = layer(g, h, e)

        return h, e


