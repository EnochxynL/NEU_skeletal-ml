"""VirtualRadar layer: converts skeleton graph data to radar spectrograms.

Translated from skeleton-action-recognition/layers/virtual_radar.py
Requires: nnAudio  (pip install nnAudio)
"""

import torch
import numpy as np

# Default NTU-RGB+D edges
_EDGES = [(0, 1), (1, 20), (20, 2), (2, 3), (20, 4), (4, 5), (5, 6), (6, 7),
          (7, 21), (7, 22), (20, 8), (8, 9), (9, 10), (10, 11), (11, 23),
          (11, 24), (0, 16), (0, 12), (12, 13), (13, 14), (14, 15), (16, 17),
          (17, 18), (18, 19)]


class VirtualRadar(torch.nn.Module):
    """Converts (N, C, T, V, M) skeleton data to spectrogram images."""

    def __init__(self,
                 edges=_EDGES,
                 wavelength=1e-3,
                 radar_location=None,
                 train_wavelength=False,
                 train_radar_location=False,
                 train_stft_kernel=False,
                 n_fft=256,
                 hop_length=16,
                 device='cuda:0'):
        super().__init__()
        from nnAudio.Spectrogram import STFT

        if radar_location is None:
            radar_location = [0., 0., 0.]
        self.wavelength = torch.nn.Parameter(
            torch.as_tensor(wavelength), requires_grad=train_wavelength)
        self.radar_location = torch.nn.Parameter(
            torch.as_tensor(radar_location), requires_grad=train_radar_location)
        self.src, self.dst = map(list, zip(*edges))
        self.stft = STFT(n_fft=n_fft, freq_bins=n_fft, hop_length=hop_length,
                         output_format='Complex', trainable=train_stft_kernel,
                         verbose=False)
        self.stft.to(device)
        self.n_fft = n_fft

    def forward(self, x):
        # x: (N, C, T, V, M)
        source_joints = x[:, :, :, self.src]
        destination_joints = x[:, :, :, self.dst]

        radar_vec = torch.abs(source_joints -
                              self.radar_location[None, :, None, None, None])
        distances = torch.norm(radar_vec, dim=1)

        A = (self.radar_location[None, :, None, None, None] -
             (source_joints + destination_joints) / 2)
        B = destination_joints - source_joints
        theta = torch.acos(
            torch.sum(A * B, dim=1) /
            (torch.norm(A, dim=1) * torch.norm(B, dim=1) + 1e-6))
        phi = torch.asin(
            (self.radar_location[1] - source_joints[:, 1]) /
            (torch.norm(radar_vec[:, :2], dim=1) + 1e-6))

        c = torch.mean(torch.norm(source_joints - destination_joints, dim=1),
                       dim=2, keepdim=True)
        c = c ** 2
        rcs = (np.pi * c) / (
            (torch.sin(theta) ** 2) * (torch.cos(phi) ** 2) +
            (torch.sin(theta) ** 2) * (torch.sin(phi) ** 2) +
            c * (torch.cos(theta) ** 2)) ** 2

        amp = torch.sqrt(rcs)
        theta_val = 4 * np.pi * distances / self.wavelength

        phase_data = torch.stack(
            (amp * torch.cos(theta_val), amp * torch.sin(theta_val)), dim=4)
        phase_data = torch.sum(phase_data, dim=[2, 3])
        stft_real = self.stft(phase_data[..., 0])
        stft_imag = self.stft(phase_data[..., 1])
        phase_data = torch.stack(
            (stft_real[..., 0] - stft_imag[..., 1],
             stft_real[..., 1] + stft_imag[..., 0]),
            dim=-1)

        phase_data = torch.norm(phase_data, dim=-1)
        phase_data = torch.log(phase_data + 1e-6)
        phase_data = torch.roll(phase_data, self.n_fft // 2, dims=1)
        return phase_data
