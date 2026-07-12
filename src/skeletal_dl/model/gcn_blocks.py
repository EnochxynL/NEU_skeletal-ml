"""PyTorch implementations of TF graph convolution layers from
skeleton-action-recognition/models/gcn.py.

GraphConv, GraphConvTD, GraphIsoConvTD, ProjectionGraphConv, ProjectionGraphPool.
All layers work on (N, C, T, V) tensors with (K, V, V) adjacency matrices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def _conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def _bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


# ---------------------------------------------------------------------------
# GraphConv  –  (N, C, V)  +  (K, V, V)  →  (N, C_out, V)
# ---------------------------------------------------------------------------
class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        _conv_init(self.conv)

    def forward(self, x, A):
        x = self.conv(x)                         # (N, C_out, V)
        x = torch.einsum('ncv,nvw->ncw', x, A)   # (N, C_out, V)
        return x, A


# ---------------------------------------------------------------------------
# GraphConvTD  –  (N, C, T, V)  +  (K, V, V)  →  (N, C_out, T, V)
# ---------------------------------------------------------------------------
class GraphConvTD(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels * kernel_size,
                              kernel_size=1)
        _conv_init(self.conv)

    def forward(self, x, A):
        x = self.conv(x)                              # (N, KC, T, V)
        N, KC, T, V = x.size()
        K = self.kernel_size
        C = KC // K
        x = x.view(N, K, C, T, V)
        x = torch.einsum('nkctv,kvw->nctw', x, A)     # (N, C, T, V)
        return x, A


# ---------------------------------------------------------------------------
# GraphIsoConvTD  –  (N, C, T, V)  +  (K-1, V, V)  →  (N, out_ch, T, V)
# ---------------------------------------------------------------------------
class GraphIsoConvTD(nn.Module):
    def __init__(self, in_channels, mlp_filters, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.mlps = nn.ModuleList()
        for _ in range(kernel_size):
            layers = []
            prev = in_channels
            for f in mlp_filters:
                layers.append(nn.Conv2d(prev, f, kernel_size=1))
                layers.append(nn.BatchNorm2d(f))
                layers.append(nn.ReLU())
                prev = f
            self.mlps.append(nn.Sequential(*layers))
        self.epsilon = nn.Parameter(torch.zeros(1))

    def forward(self, x, A):
        # A: (K-1, V, V) – no self-connections
        V = A.size(-1)
        eye = torch.eye(V, device=x.device) * (1.0 + self.epsilon)
        A_full = torch.cat([A, eye.unsqueeze(0)], dim=0)   # (K, V, V)

        x = torch.einsum('nctv,kvw->nkctw', x, A_full)     # (N, K, C, T, V)
        out = 0
        for k in range(self.kernel_size):
            out = out + self.mlps[k](x[:, k])
        return out, A


# ---------------------------------------------------------------------------
# ProjectionGraphConv  –  (N, C, T, V)  +  (K, V, V)  →  (N, C, T, V)
# ---------------------------------------------------------------------------
class ProjectionGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, vertices):
        super().__init__()
        self.vertices = vertices
        self.graph_conv = GraphConv(in_channels, out_channels)
        self.centers = nn.Parameter(torch.zeros(1, in_channels, 1, vertices))
        self.variance = nn.Parameter(torch.zeros(1, in_channels, 1, vertices))
        nn.init.normal_(self.centers, 0, math.sqrt(2. / (in_channels * vertices)))
        nn.init.normal_(self.variance, 0, math.sqrt(2. / (in_channels * vertices)))

    def forward(self, x, A):
        N, C, T, V = x.size()

        z = (x.reshape(N, C, -1, 1) - self.centers) / torch.sigmoid(self.variance)
        q = (-1.0 / 2.0) * torch.clamp(torch.sum(z * z, dim=1), min=1e-12)
        q = F.softmax(q, dim=-1)                     # (N, T*V, vertices)

        z = torch.sum(q.unsqueeze(1) * z, dim=-2)
        z = z / (q.sum(dim=-2, keepdim=True) + 1e-12)
        z = F.normalize(z, p=2, dim=-1)              # (N, C, vertices)

        A_proj = torch.matmul(z.transpose(1, 2), z)  # (N, vertices, vertices)

        z, _ = self.graph_conv(z, A_proj)            # (N, out_ch, vertices)

        x_proj = torch.matmul(q, z.transpose(1, 2))
        x_proj = x_proj.transpose(1, 2).reshape(N, -1, T, V)
        return x + x_proj, A


# ---------------------------------------------------------------------------
# ProjectionGraphPool  –  (N, C, T, V)  +  (K, V, V)  →  (N, C, vertices)
# ---------------------------------------------------------------------------
class ProjectionGraphPool(nn.Module):
    def __init__(self, in_channels, vertices):
        super().__init__()
        self.vertices = vertices
        self.centers = nn.Parameter(torch.zeros(1, in_channels, 1, vertices))
        self.variance = nn.Parameter(torch.zeros(1, in_channels, 1, vertices))
        nn.init.normal_(self.centers, 0, math.sqrt(2. / (in_channels * vertices)))
        nn.init.normal_(self.variance, 0, math.sqrt(2. / (in_channels * vertices)))

    def forward(self, x, A):
        N, C = x.shape[:2]

        x_r = x.reshape(N, C, -1, 1)
        z = (x_r - self.centers) / torch.sigmoid(self.variance)
        q = (-1.0 / 2.0) * torch.clamp(torch.sum(z * z, dim=1), min=1e-12)
        q = F.softmax(q, dim=-1)

        z = torch.sum(q.unsqueeze(1) * z, dim=-2)
        z = z / (q.sum(dim=-2, keepdim=True) + 1e-12)
        z = F.normalize(z, p=2, dim=-1)

        A_new = torch.matmul(z.transpose(1, 2), z)
        return z, A_new
