"""
File to store all decoder architectures
"""
import torch.nn as nn

from dl4cv.models.model_utils import TransposeConvLayer, ResidualBlock


class VanillaDecoder(nn.Module):
    def __init__(self):
        super(VanillaDecoder, self).__init__()
        # Residual blocks
        self.res1 = ResidualBlock(256)
        self.res2 = ResidualBlock(256)

        # Transposed convolutions
        self.t_conv1 = TransposeConvLayer(256, 256, 4, 2)
        self.t_conv2 = TransposeConvLayer(256, 3, 4, 2)

    def forward(self, x):
        y = self.res1(x)
        y = self.res2(y)
        y = self.t_conv1(y)
        y = self.t_conv2(y)
        return y
