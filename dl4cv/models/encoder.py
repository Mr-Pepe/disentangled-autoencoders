"""
File to store all encoder architectures
"""
import torch.nn as nn

from dl4cv.models.model_utils import Conv2dReflectionPadding, ResidualBlock


class VanillaEncoder(nn.Module):
    def __init__(self, in_channels, bottleneck_channels):
        super(VanillaEncoder, self).__init__()

        # Input normalization
        self.normalization = nn.BatchNorm2d(in_channels)

        # Initial convolutions
        self.conv1 = Conv2dReflectionPadding(
            in_channels=in_channels,
            out_channels=256,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.conv2 = Conv2dReflectionPadding(
            in_channels=256,
            out_channels=256,
            kernel_size=4,
            stride=2,
            padding=1
        )

        # Residual blocks
        self.res1 = ResidualBlock(256)
        self.res2 = ResidualBlock(256)

        # Non-linearities
        self.relu = nn.ReLU()

        # Bottleneck
        self.bottleneck = nn.Conv2d(
            in_channels=256,
            out_channels=bottleneck_channels,
            kernel_size=(8, 8),
            stride=1,
            padding=0
        )

    def forward(self, x):
        y = self.normalization(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.res1(y)
        y = self.res2(y)
        y = self.bottleneck(y)
        return y
