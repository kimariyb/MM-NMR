import torch
import torch.nn as nn

from typing import Optional
from models.schnet import SchNet
from models.gnn import GINEConv
from models.encoders import AtomFeatEncoder, BondFeatEncoder


class ChemicalShiftPredictor(nn.Module):
    r"""用于预测原子化学位移的预测头。
    
    该预测头接收模型的原子特征向量输出，并将其转换为化学位移预测值。
    
    Args:
        hidden_channels (int): SchNet模型的隐藏通道数。
        num_targets (int): 要预测的目标数量。对于化学位移预测，通常为1。
        dropout_rate (float, optional): Dropout比率，用于防止过拟合。默认为0.1。
        use_batch_norm (bool, optional): 是否使用批归一化。默认为True。
    """
    def __init__(
        self,
        hidden_channels: int,
        num_targets: int = 1,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.num_targets = num_targets
        self.dropout_rate = dropout_rate
        
        # 预测头网络
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2), 
            nn.PReLU(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_channels // 2, hidden_channels // 4), 
            nn.PReLU(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_channels // 4, num_targets)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化预测头的权重。"""
        for module in self.prediction_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.prediction_head(x)


class SchNetChemicalShiftHead(nn.Module):
    r"""带有化学位移预测头的 SchNet 模型。
    
    该模型将 SchNet 与化学位移预测头结合，用于预测分子中原子的化学位移。
    
    Args:
        hidden_channels (int, optional): SchNet的隐藏通道数。默认为128。
        num_filters (int, optional): SchNet的滤波器数量。默认为128。
        num_interactions (int, optional): SchNet的交互块数量。默认为6。
        num_gaussians (int, optional): 高斯展开的数量。默认为50。
        cutoff (float, optional): 截断距离。默认为10.0。
        max_num_neighbors (int, optional): 最大邻居数。默认为32。
        dropout_rate (float, optional): Dropout比率。默认为0.1。
        use_batch_norm (bool, optional): 是否使用批归一化。默认为True。
        mean (float, optional): 数据的均值，用于标准化。默认为None。
        std (float, optional): 数据的标准差，用于标准化。默认为None。
        atomref (torch.Tensor, optional): 原子参考值。默认为None。
    """
    def __init__(
        self,
        hidden_channels: int = 128,
        num_filters: int = 128,
        num_interactions: int = 6,
        num_gaussians: int = 50,
        cutoff: float = 10.0,
        max_num_neighbors: int = 32,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        atomref: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        
        # SchNet模型
        self.schnet = SchNet(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            mean=mean,
            std=std,
            atomref=atomref,
        )
        
        # 化学位移预测头
        self.predictor = ChemicalShiftPredictor(
            hidden_channels=hidden_channels,
            num_targets=1,  # 化学位移通常是一个标量值
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
        )
    
    def forward(
        self, z: torch.Tensor, pos: torch.Tensor, 
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            z (torch.Tensor): 原子序数，形状为 [num_atoms]。
            pos (torch.Tensor): 原子坐标，形状为 [num_atoms, 3]。
            batch (torch.Tensor, optional): 批索引，形状为 [num_atoms]。默认为None。
            
        Returns:
            torch.Tensor: 预测的化学位移，形状为 [num_atoms, 1]。
        """
        # 获取原子特征
        atom_features = self.schnet(z, pos, batch)
        
        # 预测化学位移
        chemical_shifts = self.predictor(atom_features)
        
        return chemical_shifts


class GINEChemicalShiftHead(nn.Module):
    r"""带有化学位移预测头的 GINE 模型。

    该模型将 GINE 与化学位移预测头结合，用于预测分子中原子的化学位移。

    Args:
        hidden_channels (int, optional): GINE的隐藏通道数。默认为128。
        num_filters (int, optional): GINE的滤波器数量。默认为128。
        num_interactions (int, optional): GINE的交互块数量。默认为6。
        dropout_rate (float, optional): Dropout比率。默认为0.1。
        use_batch_norm (bool, optional): 是否使用批归一化。默认为True。
        mean (float, optional): 数据的均值，用于标准化。默认为None。
        std (float, optional): 数据的标准差，用于标准化。默认为None。
        atomref (torch.Tensor, optional): 原子参考值。默认为None。
    """
    ...