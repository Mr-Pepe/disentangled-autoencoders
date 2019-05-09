"""
File to store all decoder architectures
"""
import torch.nn as nn

from dl4cv.models.model_utils import ResidualBlock


class VanillaDecoder(nn.Module):
    def __init__(self):
        super(VanillaDecoder, self).__init__()
        # Residual blocks
        self.res1 = ResidualBlock(256)
        self.res2 = ResidualBlock(256)

        # Transposed convolutions
        self.t_conv1 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2,
                                            padding=1, output_padding=0)
        self.t_conv2 = nn.ConvTranspose2d(in_channels=256, out_channels=3, kernel_size=4, stride=2,
                                            padding=1, output_padding=0)

    def forward(self, x):
        y = self.res1(x)
        y = self.res2(y)
        y = self.t_conv1(y)
        y = self.t_conv2(y)
        return y
