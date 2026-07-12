"""ST-GCN: Spatial-Temporal Graph Convolutional Network.

Adapted from the official AAAI'18 implementation (mmskeleton).
https://github.com/open-mmlab/mmskeleton

Input:  (N, C, T, V, M)
Output: (N, num_class)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _import_class(name):
    import importlib
    module_path, class_name = name.rsplit('.', 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def _zero(x):
    return 0


def _iden(x):
    return x


class ConvTemporalGraphical(nn.Module):
    """Spatial graph convolution: 2D conv → reshape → einsum with adjacency."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 t_kernel_size=1, t_stride=1, t_padding=0,
                 t_dilation=1, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels * kernel_size,
                              kernel_size=(t_kernel_size, 1),
                              padding=(t_padding, 0),
                              stride=(t_stride, 1),
                              dilation=(t_dilation, 1),
                              bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', x, A)
        return x.contiguous(), A


class STGCNBlock(nn.Module):
    """ST-GCN block: spatial GCN → (BN→ReLU→temporal Conv→BN→Dropout) → +residual → ReLU."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dropout=0, residual=True):
        super().__init__()
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      (kernel_size[0], 1),
                      (stride, 1),
                      ((kernel_size[0] - 1) // 2, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = _zero
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = _iden
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x), A


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2,
                 graph=None, graph_args=dict(), in_channels=3,
                 edge_importance_weighting=True, dropout=0,
                 **kwargs):
        super().__init__()

        if graph is None:
            raise ValueError(
                "graph must be specified, "
                "e.g. 'skeletal_dl.graph.ntu_rgb_d.Graph'")
        Graph = _import_class(graph)
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        # data bn: normalizes over (V * C) per person
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))

        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList([
            STGCNBlock(in_channels, 64, kernel_size, 1,
                       residual=False, dropout=dropout, **kwargs0),
            STGCNBlock(64, 64, kernel_size, 1, dropout=dropout, **kwargs),
            STGCNBlock(64, 64, kernel_size, 1, dropout=dropout, **kwargs),
            STGCNBlock(64, 64, kernel_size, 1, dropout=dropout, **kwargs),
            STGCNBlock(64, 128, kernel_size, 2, dropout=dropout, **kwargs),
            STGCNBlock(128, 128, kernel_size, 1, dropout=dropout, **kwargs),
            STGCNBlock(128, 128, kernel_size, 1, dropout=dropout, **kwargs),
            STGCNBlock(128, 256, kernel_size, 2, dropout=dropout, **kwargs),
            STGCNBlock(256, 256, kernel_size, 1, dropout=dropout, **kwargs),
            STGCNBlock(256, 256, kernel_size, 1, dropout=dropout, **kwargs),
        ])

        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for _ in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x):
        N, C, T, V, M = x.size()

        # data bn over (V*C) per person: (N,C,T,V,M) → (N*M,V*C,T)
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)
        x = self.fcn(x)
        return x.view(x.size(0), -1)
