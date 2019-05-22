"""
File to store all decoder architectures
"""
import torch.nn as nn
import torch

from dl4cv.models.model_utils import ResidualBlock, ResizeConvLayer


class VanillaDecoder(nn.Module):
    def __init__(self, bottleneck_channels, out_channels):
        super(VanillaDecoder, self).__init__()

        # Bottleneck
        self.bottleneck = nn.ConvTranspose2d(
            in_channels=bottleneck_channels,
            out_channels=64,
            kernel_size=(8, 8)
        )

        # Residual blocks
        self.res1 = ResidualBlock(64)
        self.res2 = ResidualBlock(64)

        # Resize convolutional layers are used to prevent checkerboard patterns
        self.resizeConv1 = ResizeConvLayer(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            padding=1,
            stride=1,
            scale_factor=2,
            use_relu=True
        )
        self.resizeConv2 = ResizeConvLayer(
            in_channels=32,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            scale_factor=2,
            use_relu=False
        )

    def forward(self, x):
        y = self.bottleneck(x)
        y = self.res1(y)
        y = self.res2(y)
        y = self.resizeConv1(y)
        y = self.resizeConv2(y)
        y = torch.sigmoid(y)
        return y
