import torch
import torch.nn as nn

class Activation(nn.Module):
    r"""
    Activation function module.
    """
    def __init__(self, activation):
        super(Activation, self).__init__()
        self.activation = activation
        
        if activation =='relu':
            self.activation_func = nn.ReLU()
        elif activation == 'tanh':
            self.activation_func = nn.Tanh()
        elif activation =='sigmoid':
            self.activation_func = nn.Sigmoid()
        elif activation == 'leaky_relu':
            self.activation_func = nn.LeakyReLU()
        elif activation =='softmax':
            self.activation_func = nn.Softmax(dim=1)
        elif activation == 'gelu':
            self.activation_func = nn.GELU()
        elif activation == 'elu':
            self.activation_func = nn.ELU()
        elif activation == 'prelu':
            self.activation_func = nn.PReLU()
        elif activation == 'silu':
            self.activation_func = nn.SiLU()
        else:
            raise ValueError('Invalid activation function')
        
    def forward(self, x):
        return self.activation_func(x)
    

class RBF(nn.Module):
    r"""
    Radial basis function module.
    """
    def __init__(self, num_features, num_centers, sigma):
        super(RBF, self).__init__()
        self.num_features = num_features
        self.num_centers = num_centers
        self.sigma = sigma
        
        self.centers = nn.Parameter(torch.randn(num_centers, num_features))
        
    def forward(self, x):
        x = x.unsqueeze(1) - self.centers
        x = x / self.sigma
        x = torch.sum(x ** 2, dim=2)
        
        return torch.exp(-x)


class MLP(nn.Module):
    r"""
    Multi-layer perceptron module.
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers, activation='relu', dropout=0.0):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.activation = activation
        self.dropout = dropout
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size))
        for i in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers.append(nn.Linear(hidden_size, output_size))
        
        self.activation_func = Activation(activation)
        self.dropout_func = nn.Dropout(dropout)
        
    def forward(self, x):
        for i in range(self.num_layers - 1):
            x = self.layers[i](x)
            x = self.activation_func(x)
            x = self.dropout_func(x)
        x = self.layers[-1](x)   # output layer
        
        return x