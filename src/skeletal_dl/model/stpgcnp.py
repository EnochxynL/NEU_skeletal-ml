"""ST-PGCN-Pool: Spatial-Temporal Projection GCN with Graph Pooling (PyTorch).

Re-implementation of the TensorFlow version from skeleton-action-recognition.
Uses projection-based graph pooling to downsample vertices.
Input:  (N, C, T, V, M)
Output: (N, num_class)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from skeletal_dl.model.gcn_blocks import (GraphConvTD, GraphConv,
                                          ProjectionGraphPool,
                                          _conv_init, _bn_init)


def _import_class(name):
    import importlib
    module_path, class_name = name.rsplit('.', 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


class SpatioTemporalGraphConv(nn.Module):
    """ST-GCN block: spatial GCN → BN → ReLU → temporal Conv → BN → +residual → ReLU."""
    def __init__(self, in_channels, out_channels, kernel_size=(3, 9),
                 stride=1, residual=True):
        super().__init__()
        self.sgcn = GraphConvTD(in_channels, out_channels,
                                kernel_size=kernel_size[0])
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
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
        self._residual_flag = residual
        self._residual = None

    def _build_residual(self, in_ch, out_ch, stride):
        if not self._residual_flag:
            self._residual = lambda x: 0
        elif in_ch == out_ch and stride == 1:
            self._residual = lambda x: x
        else:
            self._residual = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_ch),
            )
            _conv_init(self._residual[0])
            _bn_init(self._residual[1], 1)

    def forward(self, x, A):
        if self._residual is None:
            stride = self.tcn[2].stride[0]
            self._build_residual(x.size(1), self.tcn[2].out_channels, stride)
        res = self._residual(x)
        x, A = self.sgcn(x, A)
        x = self.tcn(x)
        x = x + res
        return self.relu(x), A


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2,
                 graph=None, graph_args=dict(), in_channels=3):
        super().__init__()
        if graph is None:
            raise ValueError(
                "graph must be specified, "
                "e.g. 'skeletal_dl.graph.ntu_rgb_d.Graph'")
        Graph = _import_class(graph)
        self.graph = Graph(**graph_args)

        self.data_bn = nn.BatchNorm1d(in_channels * num_point)
        _bn_init(self.data_bn, 1)

        self.st_gcn_layers = nn.ModuleList([
            SpatioTemporalGraphConv(in_channels, 64, residual=False),
            SpatioTemporalGraphConv(64, 64),
            SpatioTemporalGraphConv(64, 128, stride=2),
            SpatioTemporalGraphConv(128, 128),
            SpatioTemporalGraphConv(128, 256, stride=2),
            SpatioTemporalGraphConv(256, 256),
            SpatioTemporalGraphConv(256, 256, stride=2),
            SpatioTemporalGraphConv(256, 256),
        ])

        self.project_pool1 = ProjectionGraphPool(256, 512)
        self.graph_conv1 = GraphConv(256, 256)
        self.project_pool2 = ProjectionGraphPool(256, 256)
        self.graph_conv2 = GraphConv(256, 512)

        self.fc = nn.Linear(512, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        A = torch.from_numpy(self.graph.A.astype('float32')).to(x.device)
        for layer in self.st_gcn_layers:
            x, A = layer(x, A)

        # Projection pool stages — T dimension is collapsed during pooling
        x, A = self.project_pool1(x, A)     # (N*M, 256, 512)
        x, A = self.graph_conv1(x, A)       # (N*M, 256, 512)
        x, A = self.project_pool2(x, A)     # (N*M, 256, 256)
        x, A = self.graph_conv2(x, A)       # (N*M, 512, 256)

        # Global average pool over vertices
        x = x.mean(-1)                       # (N*M, 512)
        x = x.view(N, M, -1).mean(dim=1)     # (N, 512)
        return self.fc(x)
