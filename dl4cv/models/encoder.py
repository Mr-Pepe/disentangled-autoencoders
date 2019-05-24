"""
File to store all encoder architectures
"""
import torch.nn as nn

from dl4cv.models.model_utils import Conv2dReflectionPadding, ResidualBlock


class VanillaEncoder(nn.Module):
    def __init__(self, in_channels, z_dim):
        super(VanillaEncoder, self).__init__()

        self.conv1 = Conv2dReflectionPadding(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.conv2 = Conv2dReflectionPadding(
            in_channels=32,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1
        )

        # Non-linearities
        self.relu = nn.ReLU()

        # Bottleneck
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=z_dim,
            kernel_size=(8, 8),
            stride=1,
            padding=0
        )

        self.bn = nn.BatchNorm2d(z_dim)

    def forward(self, x):
        y = self.relu(self.conv1(x))
        y = self.relu(self.conv2(y))
        y = self.conv3(y)
        # y = self.bn(y)
        # output shape: [batch, bottleneck, 1, 1]
        return y
