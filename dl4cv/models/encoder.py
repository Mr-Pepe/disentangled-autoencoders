"""
File to store all encoder architectures
"""
import torch.nn as nn

from dl4cv.models.model_utils import Conv2dReflectionPadding, ResidualBlock


class VanillaEncoder(nn.Module):
    def __init__(self):
        super(VanillaEncoder, self).__init__()

        # Input normalization
        self.normalization = nn.BatchNorm2d(3)

        # Initial convolutions
        self.conv1 = Conv2dReflectionPadding(3, 256, kernel_size=4, stride=2, padding=1)
        self.conv2 = Conv2dReflectionPadding(256, 256, kernel_size=4, stride=2, padding=1)

        # Residual blocks
        self.res1 = ResidualBlock(256)
        self.res2 = ResidualBlock(256)

        # Non-linearities
        self.relu = nn.ReLU()

        # Bottleneck
        self.bottleneck = nn.Conv2d(256, 1, kernel_size=(8,8), stride=1, padding=0)

    def forward(self, x):
        y = self.normalization(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.res1(y)
        y = self.res2(y)
        y = self.bottleneck(y)
        return y
