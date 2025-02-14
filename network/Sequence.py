import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerSeq2Seq(nn.Module):
    r"""
    A transformer-based sequence-to-sequence (seq2seq) model.
    
    Parameters
    ----------
    input_dim : int
        The input dimension of the sequence.
    hidden_size : int
        The hidden size of the transformer.
    num_heads : int
        The number of heads in the transformer.
    num_layers : int
        The number of layers in the transformer.
    dropout : float, optional
        The dropout rate of the transformer, by default 0.1.
    vocab_size : int, optional
        The size of the vocabulary, by default None.
    device : str, optional
        The device to run the model on, by default None.
    reconstruction : bool, optional
        Whether to include reconstruction loss, by default False.
    """
    def __init__(
        self, 
        input_dim, 
        hidden_size, 
        num_heads, 
        num_layers, 
        dropout=0.1,
        vocab_size=None,
        device=None,
        reconstruction=False,
    ):
        super(TransformerSeq2Seq, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.embedding = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_size // 2, 
            bidirectional=True,
            batch_first=True,
            num_layers=3,
            dropout=dropout,
        )
        self.vocab_size = vocab_size
        self.device = device
        self.reconstruction = reconstruction
        
        transformer = nn.Transformer(
            d_model=hidden_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_size
        )
        self.encoder = transformer.encoder
        self.decoder = transformer.decoder
        self.out = nn.Linear(hidden_size, input_dim)
        
    def forward(self, src):
        loss = None
        embedded = self.embedding(src.to(self.device))[0]
        hidden = self.encoder(embedded)
        
        if self.reconstruction:
            output = self.decoder(embedded, hidden)
            output = self.out(output)
            output = F.log_softmax(output, dim=-1)
            loss = self.reconstruction_loss(output, src.to(self.device), self.vocab_size)
            
        return loss, hidden
    
    def reconstruction_loss(self, output, target, vocab_size=None):
        return F.nll_loss(output.view(-1, vocab_size), target.view(-1), ignore_index=0)
