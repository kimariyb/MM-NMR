# Based on the code from: https://github.com/klicperajo/dimenet,
# https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/nn/models/dimenet_utils.py
import torch
import torch.nn as nn
import sympy as sym

from torch_scatter import scatter
from torch_geometric.nn import radius_graph
from torch_geometric.nn.inits import glorot_orthogonal

from math import pi as PI
from math import sqrt

from models.encoders.utils import bessel_basis, real_sph_harm, swish, xyz2data


class Envelope(nn.Module):
    def __init__(self, exponent):
        super(Envelope, self).__init__()
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x):
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = x.pow(p - 1)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x
        
        return 1. / x + a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p2


class DitanceEmbedding(nn.Module):
    def __init__(self, num_radial, cutoff=5.0, envelope_exponent=5):
        super(DitanceEmbedding, self).__init__()
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        self.freq = nn.Parameter(torch.Tensor(num_radial))

        self.reset_parameters()

    def reset_parameters(self):
        self.freq.data = torch.arange(1, self.freq.numel() + 1).float().mul_(PI)

    def forward(self, dist):
        dist = dist.unsqueeze(-1) / self.cutoff
        return self.envelope(dist) * (self.freq * dist).sin()


class AngleEmbedding(nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff=5.0, envelope_exponent=5):
        super(AngleEmbedding, self).__init__()
        assert num_radial <= 64
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff
        # self.envelope = Envelope(envelope_exponent)

        bessel_forms = bessel_basis(num_spherical, num_radial)
        sph_harm_forms = real_sph_harm(num_spherical)
        self.sph_funcs = []
        self.bessel_funcs = []

        x, theta = sym.symbols('x theta')
        modules = {'sin': torch.sin, 'cos': torch.cos}
        for i in range(num_spherical):
            if i == 0:
                sph1 = sym.lambdify([theta], sph_harm_forms[i][0], modules)(0)
                self.sph_funcs.append(lambda x: torch.zeros_like(x) + sph1)
            else:
                sph = sym.lambdify([theta], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(sph)
            for j in range(num_radial):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

    def forward(self, dist, angle, idx_kj):
        dist = dist / self.cutoff
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        # rbf = self.envelope(dist).unsqueeze(-1) * rbf

        cbf = torch.stack([f(angle) for f in self.sph_funcs], dim=1)

        n, k = self.num_spherical, self.num_radial
        out = (rbf[idx_kj].view(-1, n, k) * cbf.view(-1, n, 1)).view(-1, n * k)
        
        return out


class TorsionEmbedding(nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff=5.0, envelope_exponent=5):
        super(TorsionEmbedding, self).__init__()
        assert num_radial <= 64
        self.num_spherical = num_spherical #
        self.num_radial = num_radial
        self.cutoff = cutoff
        # self.envelope = Envelope(envelope_exponent)

        bessel_forms = bessel_basis(num_spherical, num_radial)
        sph_harm_forms = real_sph_harm(num_spherical, zero_m_only=False)
        self.sph_funcs = []
        self.bessel_funcs = []

        x = sym.symbols('x')
        theta = sym.symbols('theta')
        phi = sym.symbols('phi')
        modules = {'sin': torch.sin, 'cos': torch.cos}
        for i in range(self.num_spherical):
            if i == 0:
                sph1 = sym.lambdify([theta, phi], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(lambda x, y: torch.zeros_like(x) + torch.zeros_like(y) + sph1(0,0)) #torch.zeros_like(x) + torch.zeros_like(y)
            else:
                for k in range(-i, i + 1):
                    sph = sym.lambdify([theta, phi], sph_harm_forms[i][k+i], modules)
                    self.sph_funcs.append(sph)
            for j in range(self.num_radial):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

    def forward(self, dist, angle, phi, idx_kj):
        dist = dist / self.cutoff
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        cbf = torch.stack([f(angle, phi) for f in self.sph_funcs], dim=1)

        n, k = self.num_spherical, self.num_radial
        out = (rbf[idx_kj].view(-1, 1, n, k) * cbf.view(-1, n, n, 1)).view(-1, n * n * k)
        
        return out


class EmbeddingBlock(nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff, envelope_exponent):
        super(EmbeddingBlock, self).__init__()
        self.dist_emb = DitanceEmbedding(num_radial, cutoff, envelope_exponent)
        self.angle_emb = AngleEmbedding(num_spherical, num_radial, cutoff, envelope_exponent)
        self.torsion_emb = TorsionEmbedding(num_spherical, num_radial, cutoff, envelope_exponent)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.dist_emb.reset_parameters()
        
    def forward(self, dist, angle, torsion, idx_kj):
        dist_emb = self.dist_emb(dist)
        angle_emb = self.angle_emb(dist, angle, idx_kj)
        torsion_emb = self.torsion_emb(dist, angle, torsion, idx_kj)
        
        return dist_emb, angle_emb, torsion_emb
    

class ResidualLayer(nn.Module):
    def __init__(self, hidden_channels, act=swish):
        super(ResidualLayer, self).__init__()
        self.act = act
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)

        self.reset_parameters()
        
    def reset_parameters(self):
        glorot_orthogonal(self.lin1.weight, scale=2.0)
        self.lin1.bias.data.fill_(0)
        glorot_orthogonal(self.lin2.weight, scale=2.0)
        self.lin2.bias.data.fill_(0)

    def forward(self, x):
        return x + self.act(self.lin2(self.act(self.lin1(x))))


class Init(nn.Module):
    def __init__(self, num_radial, hidden_channels, act=swish, use_node_features=True):
        super(Init, self).__init__()
        self.act = act
        self.use_node_features = use_node_features
        
        if self.use_node_features:
            self.emb = nn.Embedding(95, hidden_channels)
        else: # option to use no node features and a learned embedding vector for each node instead
            self.node_embedding = nn.Parameter(torch.empty((hidden_channels,)))
            nn.init.normal_(self.node_embedding)
            
        self.lin_rbf_0 = nn.Linear(num_radial, hidden_channels)
        self.lin = nn.Linear(3 * hidden_channels, hidden_channels)
        self.lin_rbf_1 = nn.Linear(num_radial, hidden_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        if self.use_node_features:
            self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))
        self.lin_rbf_0.reset_parameters()
        self.lin.reset_parameters()
        glorot_orthogonal(self.lin_rbf_1.weight, scale=2.0)

    def forward(self, x, emb, i, j):
        rbf,_,_ = emb
        if self.use_node_features:
            x = self.emb(x)
        else:
            x = self.node_embedding[None, :].expand(x.shape[0], -1)
            
        rbf0 = self.act(self.lin_rbf_0(rbf))
        e1 = self.act(self.lin(torch.cat([x[i], x[j], rbf0], dim=-1)))
        e2 = self.lin_rbf_1(rbf) * e1

        return e1, e2
    
    
class UpdateE(nn.Module):
    def __init__(self, hidden_channels, int_emb_size, basis_emb_size_dist, basis_emb_size_angle, basis_emb_size_torsion, num_spherical, num_radial,
        num_before_skip, num_after_skip, act=swish):
        super(UpdateE, self).__init__()
        self.act = act
        self.lin_rbf1 = nn.Linear(num_radial, basis_emb_size_dist, bias=False)
        self.lin_rbf2 = nn.Linear(basis_emb_size_dist, hidden_channels, bias=False)
        self.lin_sbf1 = nn.Linear(num_spherical * num_radial, basis_emb_size_angle, bias=False)
        self.lin_sbf2 = nn.Linear(basis_emb_size_angle, int_emb_size, bias=False)
        self.lin_t1 = nn.Linear(num_spherical * num_spherical * num_radial, basis_emb_size_torsion, bias=False)
        self.lin_t2 = nn.Linear(basis_emb_size_torsion, int_emb_size, bias=False)
        self.lin_rbf = nn.Linear(num_radial, hidden_channels, bias=False)

        self.lin_kj = nn.Linear(hidden_channels, hidden_channels)
        self.lin_ji = nn.Linear(hidden_channels, hidden_channels)

        self.lin_down = nn.Linear(hidden_channels, int_emb_size, bias=False)
        self.lin_up = nn.Linear(int_emb_size, hidden_channels, bias=False)

        self.layers_before_skip = nn.ModuleList([
            ResidualLayer(hidden_channels, act)
            for _ in range(num_before_skip)
        ])
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = nn.ModuleList([
            ResidualLayer(hidden_channels, act)
            for _ in range(num_after_skip)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_rbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_t1.weight, scale=2.0)
        glorot_orthogonal(self.lin_t2.weight, scale=2.0)

        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)

        glorot_orthogonal(self.lin_down.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)

        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)

    def forward(self, x, emb, idx_kj, idx_ji):
        rbf0, sbf, t = emb
        x1,_ = x

        x_ji = self.act(self.lin_ji(x1))
        x_kj = self.act(self.lin_kj(x1))

        rbf = self.lin_rbf1(rbf0)
        rbf = self.lin_rbf2(rbf)
        x_kj = x_kj * rbf

        x_kj = self.act(self.lin_down(x_kj))

        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        x_kj = x_kj[idx_kj] * sbf

        t = self.lin_t1(t)
        t = self.lin_t2(t)
        x_kj = x_kj * t

        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x1.size(0))
        x_kj = self.act(self.lin_up(x_kj))

        e1 = x_ji + x_kj
        for layer in self.layers_before_skip:
            e1 = layer(e1)
        e1 = self.act(self.lin(e1)) + x1
        for layer in self.layers_after_skip:
            e1 = layer(e1)
        e2 = self.lin_rbf(rbf0) * e1

        return e1, e2
    

class UpdateV(nn.Module):
    def __init__(self, hidden_channels, out_emb_channels, out_channels, num_output_layers, act, output_init):
        super(UpdateV, self).__init__()
        self.act = act
        self.output_init = output_init

        self.lin_up = nn.Linear(hidden_channels, out_emb_channels, bias=True)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_output_layers):
            self.lins.append(nn.Linear(out_emb_channels, out_emb_channels))
        self.lin = nn.Linear(out_emb_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_up.weight, scale=2.0)
        for lin in self.lins:
            glorot_orthogonal(lin.weight, scale=2.0)
            lin.bias.data.fill_(0)
        if self.output_init == 'zeros':
            self.lin.weight.data.fill_(0)
        if self.output_init == 'GlorotOrthogonal':
            glorot_orthogonal(self.lin.weight, scale=2.0)

    def forward(self, e, i):
        _, e2 = e
        v = scatter(e2, i, dim=0)
        v = self.lin_up(v)
        
        for lin in self.lins:
            v = self.act(lin(v))
            
        v = self.lin(v)
        
        return v
    

class UpdateU(nn.Module):
    def __init__(self):
        super(UpdateU, self).__init__()

    def forward(self, u, v, batch):
        u += scatter(v, batch, dim=0)
        return u
    

class SphereNet(nn.Module):
    r"""
    The spherical message passing neural network SphereNet from the 
    `"Spherical Message Passing for 3D Molecular Graphs" 
    <https://openreview.net/forum?id=givsRXsOt9r>`_ paper.
    
    Parameters
    ----------
    cutoff : float, optional
        Cutoff distance for interatomic interactions. Default is 5.0.
    num_layers : int, optional
        Number of message passing layers. Default is 4.
    hidden_channels : int, optional
        Number of hidden channels. Default is 128.
    out_channels : int, optional
        Number of output channels. Default is 1.
    int_emb_size : int, optional
        Size of the intermediate embedding size. Default is 64.
    basis_emb_size_dist : int, optional
        Size of the distance basis embedding. Default is 8.
    basis_emb_size_angle : int, optional
        Size of the angle basis embedding. Default is 8.
    basis_emb_size_torsion : int, optional
        Size of the torsion basis embedding. Default is 8.
    out_emb_channels : int, optional
        Number of output embedding channels. Default is 256.
    num_spherical : int, optional
        Number of spherical harmonics. Default is 7.
    num_radial : int, optional
        Number of radial basis functions. Default is 6.
    envelope_exponent : int, optional
        Exponent of the envelope function. Default is 5.
    num_before_skip : int, optional
        Number of residual layers before skip connection. Default is 1.
    num_after_skip : int, optional
        Number of residual layers after skip connection. Default is 2.
    num_output_layers : int, optional
        Number of output layers. Default is 3.
    act : function, optional
        Activation function. Default is swish.
    output_init : str, optional
        Initialization of the output layers. Default is 'GlorotOrthogonal'.
    use_node_features : bool, optional
        Whether to use node features. Default is True.
    """
    def __init__(
        self, 
        cutoff=5.0, 
        num_layers=4,
        hidden_channels=128, 
        out_channels=1, 
        int_emb_size=64,
        basis_emb_size_dist=8, 
        basis_emb_size_angle=8, 
        basis_emb_size_torsion=8, 
        out_emb_channels=256,
        num_spherical=7, 
        num_radial=6, 
        envelope_exponent=5,
        num_before_skip=1, 
        num_after_skip=2, 
        num_output_layers=3,
        act=swish, 
        output_init='GlorotOrthogonal', 
        use_node_features=True
    ):
        super(SphereNet, self).__init__()

        self.cutoff = cutoff

        self.init_e = Init(
            num_radial, 
            hidden_channels,
            act, 
            use_node_features=use_node_features
        )
        
        self.init_v = UpdateV(
            hidden_channels, 
            out_emb_channels, 
            out_channels, 
            num_output_layers, 
            act, 
            output_init
        )
        
        self.emb = EmbeddingBlock(
            num_spherical, 
            num_radial,
            self.cutoff,
            envelope_exponent
        )

        self.update_vs = nn.ModuleList([
            UpdateV(
                hidden_channels, 
                out_emb_channels, 
                out_channels, 
                num_output_layers, 
                act, 
                output_init
            ) for _ in range(num_layers)])

        self.update_es = nn.ModuleList([
            UpdateE(
                hidden_channels, 
                int_emb_size, 
                basis_emb_size_dist, 
                basis_emb_size_angle, 
                basis_emb_size_torsion, 
                num_spherical, 
                num_radial, 
                num_before_skip, 
                num_after_skip,
                act
            ) for _ in range(num_layers)])

        self.reset_parameters()

    def reset_parameters(self):
        self.init_e.reset_parameters()
        self.init_v.reset_parameters()
        self.emb.reset_parameters()
        for update_e in self.update_es:
            update_e.reset_parameters()
        for update_v in self.update_vs:
            update_v.reset_parameters()

    def forward(self, z, pos, batch):        
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        num_nodes = z.size(0)
        
        # Calculate distance, angle, and torsion features
        dist, angle, torsion, i, j, idx_kj, idx_ji = xyz2data(pos, edge_index, num_nodes, use_torsion=True)

        # Calculate embedding
        emb = self.emb(dist, angle, torsion, idx_kj)

        # Initialize edge, node, graph features
        e = self.init_e(z, emb, i, j)
        v = self.init_v(e, i)

        # Message passing
        for update_e, update_v in zip(self.update_es, self.update_vs):
            e = update_e(e, emb, idx_kj, idx_ji)
            v = update_v(e, i)

        return v
