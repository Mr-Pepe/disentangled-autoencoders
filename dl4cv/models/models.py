"""
File to store all model architectures
"""
import torch.nn as nn
import torch

from dl4cv.models.encoder import VanillaEncoder
from dl4cv.models.decoder import VanillaDecoder


class VanillaVAE(nn.Module):
    def __init__(self):
        super(VanillaVAE, self).__init__()
        self.encoder = VanillaEncoder()
        self.decoder = VanillaDecoder()

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
