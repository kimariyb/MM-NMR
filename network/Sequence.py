import torch
import torch.nn as nn
import torch.nn.functional as F


class FingerPrintNet(nn.Module):
    def __init__(
        self,
        embed_dim_in: int = 1489,
        embed_dim_out: int = 512,
        hidden_dim: int = 96,
        dropout: float = 0.5,
    ):
        super(FingerPrintNet).__init__()
        self.embed_dim_in = embed_dim_in
        self.embed_dim_out = embed_dim_out
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        self.fc1 = nn.Linear(embed_dim_in, embed_dim_out)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(embed_dim_out, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x # (batch_size, hidden_dim)
    

class SequenceExcitationBlock(nn.Module):
    def __init__(self, in_channels: int):
        super(SequenceExcitationBlock).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv1 = nn.Conv1d(in_channels, in_channels // 16, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels // 16, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        
        return x # (batch_size, 1, seq_len)
    

class SMICNN(nn.Module):
    def __init__(
        self,
        embed_dim: int = 100,
        hidden_dim: int = 128,
        n_filters: int = 32,
        out_dim: int = 96,
        dropout: float = 0.5,
    ):
        super(SMICNN).__init__()
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        self.embedding = nn.Linear(embed_dim, hidden_dim)
        
        self.conv_2_kernel = nn.Conv1d(in_channels=embed_dim, out_channels=n_filters, kernel_size=2)
        self.fc_2_kernel = nn.Linear(32 * 127, out_dim)
        
        self.conv_4_kernel = nn.Conv1d(in_channels=embed_dim, out_channels=n_filters, kernel_size=4)
        self.fc_4_kernel = nn.Linear(32 * 125, out_dim)
        
        self.conv_8_kernel = nn.Conv1d(in_channels=embed_dim, out_channels=n_filters, kernel_size=8)
        self.fc_8_kernel = nn.Linear(32 * 121, out_dim)
        
        self.SE = SequenceExcitationBlock(in_channels=n_filters)
        
        # Combine all the features
        self.fc_combine = nn.Linear(3 * out_dim, out_dim)
    
    def forward(self, x):
        embed_x = self.embedding(x)
        
        # kernel size 2
        conv_2_kernel = self.conv_2_kernel(embed_x)
        conv_2_kernel = self.relu(conv_2_kernel)
        SE_2_kernel = self.SE(conv_2_kernel)
        conv_2_kernel = conv_2_kernel * SE_2_kernel
        
        # kernel size 4
        conv_4_kernel = self.conv_4_kernel(embed_x)
        conv_4_kernel = self.relu(conv_4_kernel)
        SE_4_kernel = self.SE(conv_4_kernel)
        conv_4_kernel = conv_4_kernel * SE_4_kernel
        
        # kernel size 8
        conv_8_kernel = self.conv_8_kernel(embed_x)
        conv_8_kernel = self.relu(conv_8_kernel)
        SE_8_kernel = self.SE(conv_8_kernel)
        conv_8_kernel = conv_8_kernel * SE_8_kernel
        
        # combine all the features
        conv_2_kernel = conv_2_kernel.view(-1, 32 * 127)
        conv_2_kernel = self.fc_2_kernel(conv_2_kernel)
        
        conv_4_kernel = conv_4_kernel.view(-1, 32 * 125)
        conv_4_kernel = self.fc_4_kernel(conv_4_kernel)
        
        conv_8_kernel = conv_8_kernel.view(-1, 32 * 121)        
        conv_8_kernel = self.fc_8_kernel(conv_8_kernel)
        
        combine_features = torch.cat((conv_2_kernel, conv_4_kernel, conv_8_kernel), dim=1)
        combine_features = self.fc_combine(combine_features)
        
        return combine_features # (batch_size, out_dim)