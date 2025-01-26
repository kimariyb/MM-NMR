import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops


class GINConv(MessagePassing):
    r"""
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Parameters
    ----------
    emb_dim : int
        Dimensionality of the input features.
    out_dim : int
        Dimensionality of the output features.
    aggr : str, optional
        Aggregation scheme to use (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
        (default: :obj:`"add"`)
        
    See https://arxiv.org/abs/1810.00826
    """
    def __init__(self, emb_dim, aggr="add"):
        super(GINConv, self).__init__()
        # multi-layer perceptron to introduce non-linearity
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim)
        )

        self.edge_embedding = nn.Sequential(
            nn.Linear(18, emb_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(emb_dim, emb_dim)
        )

        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # self loops are added into the embedding space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add self-loop edge features
        self_loop_attr = torch.zeros(x.size(0), 18)
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding(edge_attr.to(torch.float32))

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GNN(nn.Module):
    r"""
    A GNN model for predicting fluorescence NMR spectra.
    
    Parameters
    ----------
    num_layer : int
        Number of GNN layers.
    emb_dim : int
        Dimensionality of the input features.
    JK : str, optional
        Jumping knowledge strategy to use (:obj:`"last"`, :obj:`"concat"`, :obj:`"max"`, :obj:`"sum"`).
        (default: :obj:`"last"`)
    drop_ratio : float, optional
        Dropout ratio to use in the GNN layers.
        (default: :obj:`0.`)
        
    See https://arxiv.org/abs/1810.00826
    """

    def __init__(
            self,
            num_layer: int,
            emb_dim: int,
            JK: str = "last",
            drop_ratio: float = 0.,
    ):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.mlp = nn.Sequential(
            nn.Linear(178, emb_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(emb_dim, emb_dim)
        )

        # GNN layers
        self.gnns = nn.ModuleList()
        for layer in range(self.num_layer):
            self.gnns.append(GINConv(emb_dim))

        # batch normalization layers
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, edge_index, edge_attr):
        x = self.mlp(x.to(torch.float32))

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]

        return node_representation
