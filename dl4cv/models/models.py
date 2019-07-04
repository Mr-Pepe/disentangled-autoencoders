"""
File to store all model architectures
"""

import abc

import torch
import torch.nn as nn

import dl4cv.utils as utils
from dl4cv.models.decoder import VanillaDecoder
from dl4cv.models.encoder import VanillaEncoder
from dl4cv.models.physics_layer import PhysicsPVA, PhysicsLayer


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    @abc.abstractmethod
    def forward(self, x):
        pass

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self.cpu(), path)


class AutoEncoder(BaseModel):
    def __init__(self, len_in_sequence, z_dim, greyscale=False):
        super(AutoEncoder, self).__init__()
        # input frames get concatenated, in_channels depends on the length
        # of the sequences
        # number of output channels depend only on using grayscale or not
        if greyscale:
            in_channels = len_in_sequence
            out_channels = 1
        else:
            in_channels = len_in_sequence * 3
            out_channels = 3

        self.encoder = VanillaEncoder(
            in_channels=in_channels,
            z_dim=z_dim
        )
        self.decoder = VanillaDecoder(
            z_dim=z_dim,
            out_channels=out_channels
        )

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y, (z,)


class VariationalAutoEncoder(BaseModel):
    """"This VAE generates means and log-variances of
    the latent variables and samples from those distributions"""
    def __init__(self, len_in_sequence, len_out_sequence, z_dim_encoder=6, z_dim_decoder=6, use_physics=False):
        super(VariationalAutoEncoder, self).__init__()
        self.z_dim_encoder = z_dim_encoder
        self.z_dim_decoder = z_dim_decoder
        self.use_physics = use_physics

        self.encoder = VanillaEncoder(
            in_channels=len_in_sequence,
            z_dim=z_dim_encoder*2
        )
        self.decoder = VanillaDecoder(
            z_dim=z_dim_decoder,
            out_channels=len_out_sequence
        )

        self.physics_layer = PhysicsLayer(dt=1. / 30.)

    def forward(self, x, q=None):
        z_encoder, mu, logvar = self.encode(x)

        if torch.any(q != -1):
            if self.use_physics:
                z_decoder = self.physics_layer(z_encoder, q)
            else:
                q = q[:, None, None, None]
                z_decoder = torch.cat((z_encoder, q), dim=1)
        else:
            z_decoder = z_encoder

        y = self.decode(z_decoder)

        return y, (mu, logvar)

    def encode(self, x):
        z_params = self.encoder(x)
        mu = z_params[:, :self.z_dim_encoder]
        logvar = z_params[:, self.z_dim_encoder:]

        z_encoder = utils.reparametrize(mu, logvar)

        return z_encoder, mu, logvar

    def decode(self, z_decoder):
        return self.decoder(z_decoder)

    def physics(self, x):
        return self.physics_layer(x)
