"""
File to store all model architectures
"""
import torch.nn as nn
import torch

from dl4cv.models.encoder import VanillaEncoder
from dl4cv.models.decoder import VanillaDecoder


class VanillaVAE(nn.Module):
    def __init__(self, len_in_sequence, bottleneck_channels, grayscale=False):
        super(VanillaVAE, self).__init__()
        # input frames get concatenated, in_channels depends on the length
        # of the sequences
        # number of output channels depend only on using grayscale or not
        if grayscale:
            in_channels = len_in_sequence
            out_channels = 1
        else:
            in_channels = len_in_sequence * 3
            out_channels = 3

        self.encoder = VanillaEncoder(
            in_channels=in_channels,
            bottleneck_channels=bottleneck_channels
        )
        self.decoder = VanillaDecoder(
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels
        )

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self.cpu(), path)
