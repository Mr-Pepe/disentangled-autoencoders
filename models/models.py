"""
File to store all model architectures
"""
import torch.nn as nn

from models.encoder import VanillaEncoder
from models.decoder import VanillaDecoder


class VanillaVAE(nn.Module):
    def __init__(self):
        super(VanillaVAE, self).__init__()
        self.encoder = VanillaEncoder()
        self.decoder = VanillaDecoder()

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y