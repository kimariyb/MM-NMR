import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum
from einops import rearrange

from network.Sequence import FingerPrintNet, SMICNN
from network.Graph import GNN


class AttentionBlock(nn.Module):
    def __init__(
        self, 
        hidden_dim, 
        num_heads=8, 
        dropout=0.1, 
    ):
        super(AttentionBlock, self).__init__()
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
        super(CrossAttentionBlock, self).__init__()
        
        self.finger_encoder = FingerPrintNet()
        self.smiles_encoder = SMICNN()
        self.graph_encoder = GNN()
        
        self.attention = AttentionBlock(hidden_dim=96, num_heads=8, dropout=0.1)
    
    def forward(self, data):
        finger, vector, node_features, edge_index, edge_attr = data.fingerprint, data.vector, data.x, data.edge_index, data.edge_attr
        
        # Encode fingerprints
        finger_embedding = self.finger_encoder(finger)
        print(finger_embedding.shape)
        # Encode SMILES
        smiles_embedding = self.smiles_encoder(vector)
        print(smiles_embedding.shape)
        # Encode graph
        graph_embedding = self.graph_encoder(node_features, edge_index, edge_attr)
        print(graph_embedding.shape)
        # Cross-attention for molecule representation
        feat1 = smiles_embedding + self.attention(finger_embedding, smiles_embedding, smiles_embedding)
        feat2 = finger_embedding + self.attention(smiles_embedding, finger_embedding, finger_embedding)
        
        # Cross-attention for interaction representation
        feat3 = graph_embedding + self.attention(finger_embedding, graph_embedding, graph_embedding)
        feat4 = finger_embedding + self.attention(graph_embedding, finger_embedding, finger_embedding)
        
        return feat1, feat2, feat3, feat4
    

class MultiViewRepresentation(nn.Module):
    def __init__(self, embed_dim):
        super(MultiViewRepresentation, self).__init__()
        
        self.cross_attention = CrossAttentionBlock()
        
        # Define readout function
        self.readout = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2), nn.LeakyReLU(0.1), nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, embed_dim // 4), nn.LeakyReLU(0.1), nn.Dropout(0.1),
            nn.Linear(embed_dim // 4, embed_dim // 8), nn.LeakyReLU(0.1), nn.Dropout(0.1),
            nn.Linear(embed_dim // 8, 1)
        )
    
    def forward(self, data):
        # Apply cross-attention
        feat1, feat2, feat3, feat4 = self.cross_attention(data)
        
        x = torch.cat([feat1, feat2, feat3, feat4], dim=1)
        x = self.readout(x)
        
        return x