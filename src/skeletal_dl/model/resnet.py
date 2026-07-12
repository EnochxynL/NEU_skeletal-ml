"""ResNet + VirtualRadar model for skeleton-based action recognition.

The VirtualRadar layer converts skeleton graph data (N,C,T,V,M) into
spectrogram images, which are then classified by a ResNet18.

Requires: nnAudio (pip install nnAudio)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from skeletal_dl.model.resnet18 import resnet18
from skeletal_dl.model.virtual_radar import VirtualRadar


class Model(torch.nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2,
                 graph=None, graph_args=dict(), in_channels=3,
                 num_filters=64, image_size=256,
                 wavelength=5e-4, device='cuda:0'):
        super().__init__()
        self.base_model = resnet18(num_classes=num_class, num_filters=num_filters)
        self.virtual_radar = VirtualRadar(wavelength=wavelength, device=device)
        self.image_size = image_size

    def forward(self, x):
        # x: (N, C, T, V, M) -> VirtualRadar -> (N, F, L) -> unsqueeze -> (N, 1, F, L)
        x = self.virtual_radar(x)
        x = x.unsqueeze(dim=1)
        x = F.interpolate(x, self.image_size)
        return self.base_model(x)
