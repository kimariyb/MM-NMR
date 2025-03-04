import torch
import torch.nn as nn
import numpy as np


class RBF(nn.Module):
    r"""
    Radial Basis Function (RBF)
    """
    def __init__(self, centers, gamma):
        super(RBF, self).__init__()
        self.centers = torch.reshape(torch.tensor(centers), [1, -1])
        self.gamma = gamma

    def forward(self, x):
        x = torch.reshape(x, [-1, 1])
        
        return torch.exp(-self.gamma * torch.square(x - self.centers))


class AtomEmbedding(torch.nn.Module):
    r"""
    Atom Encoder
    """
    def __init__(self, embed_dim, device):
        super(AtomEmbedding, self).__init__()
        self.atom_names = atom_names
        self.embed_list = nn.ModuleList()
        
        for name in self.atom_names:
            embed = nn.Embedding(CompoundKit.get_atom_feature_size(name) + 5, embed_dim).to(device)
            self.embed_list.append(embed)

    def forward(self, node_features):
        out_embed = 0
        for i, name in enumerate(self.atom_names):
            out_embed += self.embed_list[i](node_features[i])
            
        return out_embed


class AtomFloatEmbedding(torch.nn.Module):
    r"""
    Atom Float Encoder
    """
    def __init__(self, atom_float_names, embed_dim, rbf_params=None, device=None):
        super(AtomFloatEmbedding, self).__init__()
        self.atom_float_names = atom_float_names

        if rbf_params is None:
            self.rbf_params = {
                'van_der_waals_radis': (torch.arange(1, 3, 0.2), 10.0),  # (centers, gamma)
                'partial_charge': (torch.arange(-1, 4, 0.25), 10.0),  # (centers, gamma)
                'mass': (torch.arange(0, 2, 0.1), 10.0),  # (centers, gamma)
            }
        else:
            self.rbf_params = rbf_params
            self.linear_list = nn.ModuleList()
            self.rbf_list = nn.ModuleList()
            for name in self.atom_float_names:
                centers, gamma = self.rbf_params[name]
                rbf = RBF(centers, gamma).to(device)
                self.rbf_list.append(rbf)
                linear = nn.Linear(len(centers), embed_dim).to(device)
                self.linear_list.append(linear)
                self.rbf_list.append(rbf)
                linear = nn.Linear(len(centers), embed_dim).to(device)
                self.linear_list.append(linear)

    def forward(self, feats):
        """
        Args:
            feats(dict of tensor): node float features.
        """
        out_embed = 0
        for i, name in enumerate(self.atom_float_names):
            x = feats[name]
            rbf_x = self.rbf_list[i](x)
            out_embed += self.linear_list[i](rbf_x)
            
        return out_embed


class BondEmbedding(nn.Module):
    r"""
    Bond Encoder
    """
    def __init__(self, bond_names, embed_dim, device):
        super(BondEmbedding, self).__init__()
        self.bond_names = bond_names

        self.embed_list = nn.ModuleList()
        for name in self.bond_names:
            embed = nn.Embedding(CompoundKit.get_bond_feature_size(name) + 5, embed_dim).to(device)
            self.embed_list.append(embed)

    def forward(self, edge_features):
        """
        Args:
            edge_features(dict of tensor): edge features.
        """
        out_embed = 0
        for i, name in enumerate(self.bond_names):
            out_embed += self.embed_list[i](edge_features[i].long())
        return out_embed


class BondFloatRBF(nn.Module):
    r"""
    Bond Float Encoder using Radial Basis Functions
    """
    def __init__(self, bond_float_names, embed_dim, rbf_params=None, device=None):
        super(BondFloatRBF, self).__init__()
        self.bond_float_names = bond_float_names

        if rbf_params is None:
            self.rbf_params = {
                'bond_length': (torch.arange(0, 2, 0.1).to(device), 10.0),  # (centers, gamma)
            }
        else:
            self.rbf_params = rbf_params

        self.linear_list = nn.ModuleList()
        self.rbf_list = nn.ModuleList()
        for name in self.bond_float_names:
            centers, gamma = self.rbf_params[name]
            rbf = RBF(centers, gamma).to(device)
            self.rbf_list.append(rbf)
            linear = nn.Linear(len(centers), embed_dim).to(device)
            self.linear_list.append(linear)

    def forward(self, bond_float_features):
        out_embed = 0
        for i, name in enumerate(self.bond_float_names):
            x = bond_float_features[i]
            rbf_x = self.rbf_list[i](x)
            out_embed += self.linear_list[i](rbf_x.float())
        return out_embed


class BondAngleFloatRBF(nn.Module):
    r"""
    Bond Angle Float Encoder using Radial Basis Functions
    """
    def __init__(self, bond_angle_float_names, embed_dim, rbf_params=None, device=None):
        super(BondAngleFloatRBF, self).__init__()
        self.bond_angle_float_names = bond_angle_float_names

        if rbf_params is None:
            self.rbf_params = {
                'bond_angle': (torch.arange(0, np.pi, 0.1).to(device), 10.0),  # (centers, gamma)
            }
        else:
            self.rbf_params = rbf_params

        self.linear_list = nn.ModuleList()
        self.rbf_list = nn.ModuleList()
        
        for name in self.bond_angle_float_names:
            centers, gamma = self.rbf_params[name]
            rbf = RBF(centers, gamma).to(device)
            self.rbf_list.append(rbf)
            linear = nn.Linear(len(centers), embed_dim).to(device)
            self.linear_list.append(linear)

    def forward(self, bond_angle_float_features):
        out_embed = 0
        for i, name in enumerate(self.bond_angle_float_names):
            x = bond_angle_float_features[i]
            rbf_x = self.rbf_list[i](x)
            out_embed += self.linear_list[i](rbf_x.float())
            
        return out_embed