"""
File to store all decoder architectures
"""
import torch.nn as nn
import torch

from dl4cv.models.model_utils import ResizeConvLayer


class VanillaDecoder(nn.Module):
    def __init__(self, z_dim, out_channels):
        super(VanillaDecoder, self).__init__()

        # Bottleneck
        self.convT1 = nn.ConvTranspose2d(
            in_channels=z_dim,
            out_channels=64,
            kernel_size=(8, 8)
        )

        # Resize convolutional layers are used to prevent checkerboard patterns
        self.resizeConv2 = ResizeConvLayer(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            padding=1,
            stride=1,
            scale_factor=2
        )

        self.resizeConv3 = ResizeConvLayer(
            in_channels=32,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            scale_factor=2
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.relu(self.convT1(x))
        y = self.relu(self.resizeConv2(y))
        y = self.resizeConv3(y)
        y = torch.sigmoid(y)
        return y
