"""
File to store all model architectures
"""

import abc
import torch.nn as nn
import torch

from dl4cv.models.encoder import VanillaEncoder
from dl4cv.models.physics_layer import PhysicsPVA
from dl4cv.models.decoder import VanillaDecoder
import dl4cv.utils as utils


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


class PhysicsAutoEncoder(BaseModel):
    def __init__(self, dt, len_in_sequence, greyscale=False):
        super(PhysicsAutoEncoder, self).__init__()
        if greyscale:
            in_channels = len_in_sequence
            out_channels = 1
        else:
            in_channels = len_in_sequence * 3
            out_channels = 3

        self.physics_layer = PhysicsPVAtoP(dt=dt)

        self.encoder = VanillaEncoder(
            in_channels=in_channels,
            z_dim=self.physics_layer.num_latents_in
        )
        self.decoder = VanillaDecoder(
            z_dim=self.physics_layer.num_latents_out,
            out_channels=out_channels
        )

    def forward(self, x):
        z_t = self.encoder(x)
        z_t_plus_1 = self.physics_layer(z_t)
        y = self.decoder(z_t_plus_1)
        return y, tuple(z_t,)


class VariationalAutoEncoder(BaseModel):
    """"This VAE generates means and log-variances of
    the latent variables and samples from those distributions"""
    def __init__(self, len_in_sequence, z_dim):
        super(VariationalAutoEncoder, self).__init__()
        self.z_dim = z_dim

        self.encoder = VanillaEncoder(
            in_channels=len_in_sequence,
            z_dim=z_dim*2
        )
        self.decoder = VanillaDecoder(
            z_dim=z_dim,
            out_channels=1
        )

    def forward(self, x):
        # Taken from https://github.com/1Konny/Beta-VAE/blob/master/model.py
        z_params = self.encoder(x)
        mu = z_params[:, :self.z_dim]
        logvar = z_params[:, self.z_dim:]

        z = utils.reparametrize(mu, logvar)
        y = self.decoder(z)

        return y, (mu, logvar)


class VariationalPhysicsAutoEncoder(BaseModel):

    def __init__(self, len_in_sequence, len_out_sequence, z0_dim=6, z1_dim=6):
        super(VariationalPhysicsAutoEncoder, self).__init__()
        self.z_dim = z0_dim

        self.encoder = VanillaEncoder(
            in_channels=len_in_sequence,
            z_dim=z0_dim * 2
        )
        self.decoder = VanillaDecoder(
            z_dim=z1_dim,
            out_channels=len_out_sequence
        )

        self.physics_layer = PhysicsPVA(dt=1/30, out_dim=z1_dim)

    def forward(self, x):
        z_t, mu, logvar = self.encode(x)

        z_t_plus_1 = self.physics(z_t)

        y = self.decode(z_t_plus_1)

        return y, (mu, logvar)

    def encode(self, x):
        z_params = self.encoder(x)
        mu = z_params[:, :self.z_dim]
        logvar = z_params[:, self.z_dim:]

        z_t = utils.reparametrize(mu, logvar)

        return z_t, mu, logvar

    def physics(self, z_t):
        return self.physics_layer(z_t)

    def decode(self, z_t_plus_1):
        return self.decoder(z_t_plus_1)
