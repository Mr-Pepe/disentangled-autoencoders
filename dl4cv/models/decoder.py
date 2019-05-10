"""
File to store all decoder architectures
"""
import torch.nn as nn

from dl4cv.models.model_utils import ResidualBlock, Conv2dReflectionPadding


class VanillaDecoder(nn.Module):
    def __init__(self):
        super(VanillaDecoder, self).__init__()
        # Residual blocks
        self.res1 = ResidualBlock(256)
        self.res2 = ResidualBlock(256)

        # Transposed convolutions for upsampling
        # Convolutional layers are used to prevent checkerboard patterns
        self.t_conv1 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2,
                                            padding=1, output_padding=0)
        self.conv1  = Conv2dReflectionPadding(in_channels=256,out_channels=256, kernel_size=3, stride=1, padding=1)
        self.t_conv2 = nn.ConvTranspose2d(in_channels=256, out_channels=3, kernel_size=4, stride=2,
                                            padding=1, output_padding=0)
        self.conv2 = Conv2dReflectionPadding(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        y = self.res1(x)
        y = self.res2(y)
        y = self.t_conv1(y)
        y = self.conv1(y)
        y = self.t_conv2(y)
        y = self.conv2(y)
        return y
