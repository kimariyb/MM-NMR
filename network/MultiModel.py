import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GlobalAttention, global_mean_pool
from network.Sequence import TransformerSeq2Seq
from network.Graph import MPNNEncoder
from network.Geometry import GeoGNNModel



class GlobalAttentionNet(nn.Module): ...

class WeightFusionNet(nn.Module): ...

class MultiModel(nn.Module): ...