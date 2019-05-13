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
        self.conv1 = Conv2dReflectionPadding(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.conv2 = Conv2dReflectionPadding(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm2d(channels)
        self.batch_norm2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.batch_norm2(out)
        out = out + residual
        return out


class ResizeConvLayer(nn.Module):
    """ ResizeConvLayer
    Upsampling with Nearest neighbor interpolation and a ConvLayer
    to avoid checkerboard artifacts.
    ref: https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, scale_factor=2, use_relu=False):
        super(ResizeConvLayer, self).__init__()

        self.use_relu = use_relu

        self.reflection_pad = nn.ReflectionPad2d(padding)

        self.nearest_neighbor = nn.Upsample(
            scale_factor=scale_factor,
            mode='nearest'
        )
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_in = x
        out = self.nearest_neighbor(x_in)
        out = self.reflection_pad(out)
        out = self.conv2d(out)
        if self.use_relu:
            out = self.relu(out)
        return out
