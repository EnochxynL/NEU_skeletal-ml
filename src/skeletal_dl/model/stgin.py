"""ST-GIN: Spatial-Temporal Graph Isomorphism Network (PyTorch).

Re-implementation of the TensorFlow version from skeleton-action-recognition.
Uses GraphIsoConvTD instead of GraphConvTD for the spatial GCN.
Input:  (N, C, T, V, M)
Output: (N, num_class)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from skeletal_dl.model.gcn_blocks import GraphIsoConvTD, _conv_init, _bn_init


def _import_class(name):
    import importlib
    module_path, class_name = name.rsplit('.', 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


class SpatioTemporalGraphConv(nn.Module):
    """ST-GIN block: spatial GIN → BN → ReLU → temporal Conv → BN → +residual → ReLU."""
    def __init__(self, in_channels, out_channels, kernel_size=(3, 9),
                 stride=1, residual=True, dropout=0):
        super().__init__()
        half = out_channels // 2
        self.sgcn = GraphIsoConvTD(in_channels, [half, half],
                                   kernel_size=kernel_size[0])
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(half),
            nn.ReLU(inplace=True),
            nn.Conv2d(half, out_channels,
                      kernel_size=(kernel_size[1], 1),
                      stride=(stride, 1),
                      padding=((kernel_size[1] - 1) // 2, 0)),
            nn.BatchNorm2d(out_channels),
        )
        for m in self.tcn:
            if isinstance(m, nn.Conv2d):
                _conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                _bn_init(m, 1)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout else None
        self._residual_flag = residual
        self._residual = None

    def _build_residual(self, in_ch, out_ch, stride, device):
        if not self._residual_flag:
            self._residual = lambda x: 0
        elif in_ch == out_ch and stride == 1:
            self._residual = lambda x: x
        else:
            self._residual = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_ch),
            ).to(device)
            _conv_init(self._residual[0])
            _bn_init(self._residual[1], 1)

    def forward(self, x, A):
        if self._residual is None:
            stride = self.tcn[2].stride[0]
            self._build_residual(x.size(1), self.tcn[2].out_channels, stride, x.device)
        res = self._residual(x)
        x, A = self.sgcn(x, A)
        x = self.tcn(x)
        x = x + res
        x = self.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x, A


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2,
                 graph=None, graph_args=dict(), in_channels=3, dropout=0):
        super().__init__()
        if graph is None:
            raise ValueError(
                "graph must be specified, "
                "e.g. 'skeletal_dl.graph.ntu_rgb_d.Graph'")
        Graph = _import_class(graph)
        self.graph = Graph(**graph_args)

        # data bn over (V*C) per person, matching TF original
        self.data_bn = nn.BatchNorm1d(in_channels * num_point)
        _bn_init(self.data_bn, 1)

        self.layers = nn.ModuleList([
            SpatioTemporalGraphConv(in_channels, 64, residual=False, dropout=dropout),
            SpatioTemporalGraphConv(64, 64, dropout=dropout),
            SpatioTemporalGraphConv(64, 64, dropout=dropout),
            SpatioTemporalGraphConv(64, 64, dropout=dropout),
            SpatioTemporalGraphConv(64, 128, stride=2, dropout=dropout),
            SpatioTemporalGraphConv(128, 128, dropout=dropout),
            SpatioTemporalGraphConv(128, 128, dropout=dropout),
            SpatioTemporalGraphConv(128, 256, stride=2, dropout=dropout),
            SpatioTemporalGraphConv(256, 256, dropout=dropout),
            SpatioTemporalGraphConv(256, 256, dropout=dropout),
        ])

        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)
        nn.init.normal_(self.fcn.weight, 0, math.sqrt(2. / num_class))
        nn.init.constant_(self.fcn.bias, 0)

    def forward(self, x):
        N, C, T, V, M = x.size()

        # data bn: (N,C,T,V,M) → (N*M,V*C,T) → BN → (N*M,C,T,V)
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # ST-GIN uses only first 2 adjacency partitions (no self-connections)
        A = torch.from_numpy(self.graph.A[:2].astype('float32')).to(x.device)
        for layer in self.layers:
            x, A = layer(x, A)

        # global pool, average over persons, classify
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)
        x = self.fcn(x)
        return x.view(x.size(0), -1)
