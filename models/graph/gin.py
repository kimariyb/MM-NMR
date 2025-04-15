import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops


class GINConv(MessagePassing):
    def __init__(self, edge_dim, hidden_dim, aggr = "add"):
        super(GINConv, self).__init__()

        self.edge_embed = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        nn.init.kaiming_normal_(self.edge_embed[0].weight, mode='fan_out', nonlinearity='relu')
        
        # 增强型MLP（带门控机制）
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 2*hidden_dim),
            nn.GELU(),
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.Dropout(0.2)
        )
        
        # 自环边参数（带约束的初始化）
        self.self_loop_attr = nn.Parameter(torch.Tensor(1, edge_dim))  # 保持二维
        nn.init.normal_(self.self_loop_attr)  # 使用正态分布初始化

        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):        
        # 添加自环边（优化内存效率）
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_attr = self.self_loop_attr.expand(x.size(0), -1)
        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)

        edge_attr = self.edge_embed(edge_attr)

        return self.propagate(edge_index[0], x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GIN(nn.Module):
    def __init__(self, num_layers, node_dim, edge_dim, hidden_dim, drop_ratio=0):
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.drop_ratio = drop_ratio

        self.node_embed = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        nn.init.kaiming_normal_(self.node_embed[0].weight, mode='fan_out', nonlinearity='relu')

        # List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(num_layers):
            self.gnns.append(GINConv(edge_dim, hidden_dim))
        
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, edge_attr):
        x = self.node_embed(x)

        h_list = [x]
        for layer in range(self.num_layers):
            # 带残差的GNN层
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layers - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            h_list.append(h)
      
        node_representation = h_list[-1]

        return node_representation


class GINConfig:
    def __init__(self, node_dim, edge_dim, hidden_dim, num_layers, drop_ratio):
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio

    @staticmethod
    def from_dict(config_dict):
        return GINConfig(**config_dict)