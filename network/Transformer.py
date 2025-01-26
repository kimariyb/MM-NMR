import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum
from einops import rearrange

from network.Sequence import FingerPrintNet, SMICNN


class AttentionBlock(nn.Module):
    def __init__(
        self, 
        hidden_dim, 
        num_heads=8, 
        dropout=0.1, 
    ):
        super(AttentionBlock).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        assert hidden_dim % num_heads == 0, "Hidden dimension must be divisible by number of heads"
        
        self.f_q = nn.Linear(hidden_dim, hidden_dim)
        self.f_k = nn.Linear(hidden_dim, hidden_dim)
        self.f_v = nn.Linear(hidden_dim, hidden_dim)
        
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim // num_heads]))
        
    def forward(self, q, k, v, mask=None):
        r"""
        Parameters
        ----------
        q : torch.Tensor
            Query tensor of shape (batch_size, seq_len, hidden_dim)
        k : torch.Tensor
            Key tensor of shape (batch_size, seq_len, hidden_dim)
        v : torch.Tensor
            Value tensor of shape (batch_size, seq_len, hidden_dim)
        mask : torch.Tensor, optional
            Mask tensor of shape (batch_size, seq_len, seq_len), where mask[i,j,k] = 1 means that the i-th query attends to the k-th key. Default: None
        
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, seq_len, hidden_dim)
        """        
        Q = self.f_q(q)
        K = self.f_k(k)
        V = self.f_v(v)
        
        Q = rearrange(Q, 'b s (n h) -> b n s h', n=self.num_heads, h=self.hidden_dim // self.num_heads)
        K = rearrange(K, 'b s (n h) -> b n s h', n=self.num_heads, h=self.hidden_dim // self.num_heads)
        V = rearrange(V, 'b s (n h) -> b n s h', n=self.num_heads, h=self.hidden_dim // self.num_heads)
        
        # Calculate attention scores
        scores = einsum('b n s h, b n t h -> b n s t', Q, K) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        output = einsum('b n s t, b n t h -> b n s h', attention, V)
        output = rearrange(output, 'b n s h -> b s (n h)')
        
        output = self.fc(output)
        output = self.dropout(output)
        
        return output
    
    
class CrossAttentionBlock(nn.Module):
    def __init__(self):
        super(CrossAttentionBlock).__init__()
        
        self.finger_encoder = FingerPrintNet()
        self.smiles_encoder = SMICNN()
        self.graph_encoder = ...
        
        self.attention = AttentionBlock(hidden_dim=96, num_heads=8, dropout=0.1)
    
    def forward(self, data):
        finger, vector, graph = data.fingerprint, data.vector, data.graph
        
        # Encode fingerprints
        finger_embedding = self.finger_encoder(finger)
        
        # Encode SMILES
        smiles_embedding = self.smiles_encoder(vector)
        
        # Encode graph
        graph_embedding = self.graph_encoder(graph)
        
        return ...