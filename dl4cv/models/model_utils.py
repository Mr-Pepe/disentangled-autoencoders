import numpy as np
import torch.nn as nn


class Conv2dReflectionPadding(nn.Module):
    """ ConvBlock
    simple 2d convolution with reflection padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super(Conv2dReflectionPadding, self).__init__()
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(nn.Module):
    """ ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv2dReflectionPadding(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv2dReflectionPadding(channels, channels, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = out + residual
        return out
