"""
File to store all model architectures
"""

import abc

import torch
import torch.nn as nn

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


class VariationalAutoEncoder(BaseModel):
    """"This VAE generates means and log-variances of
    the latent variables and samples from those distributions"""
    def __init__(self, len_in_sequence, len_out_sequence, z_dim_encoder=6, z_dim_decoder=6, use_physics=False):
        super(VariationalAutoEncoder, self).__init__()
        self.z_dim_encoder = z_dim_encoder
        self.z_dim_decoder = z_dim_decoder
        self.use_physics = use_physics

        self.encoder = nn.Sequential(
            nn.Conv2d(len_in_sequence, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32,  8,  8
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32,  4,  4
            nn.ReLU(True),
            View((-1, 32 * 4 * 4)),  # B, 512
            nn.Linear(32 * 4 * 4, 256),  # B, 256
            nn.ReLU(True),
            # nn.Linear(256, 256),  # B, 256
            # nn.ReLU(True),
            nn.Linear(256, z_dim_encoder * 2),  # B, z_dim*2
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim_decoder, 256),  # B, 256
            nn.ReLU(True),
            # nn.Linear(256, 256),  # B, 256
            # nn.ReLU(True),
            nn.Linear(256, 32 * 4 * 4),  # B, 512
            nn.ReLU(True),
            View((-1, 32, 4, 4)),  # B,  32,  4,  4
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(32, 32, 3, 1, 1),  # B,  32,  8,  8
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(32, 32, 3, 1, 1),  # B,  32, 16, 16
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(32, len_out_sequence, 3, 1, 1),
        )

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x, q=-1):
        z_encoder, mu, logvar = self.encode(x)

        z_decoder = self.bottleneck(z_encoder, q)

        y = self.decode(z_decoder)

        return y, (mu, logvar)

    def bottleneck(self, z_encoder, q):
        if torch.any(q != -1):
            if self.use_physics:
                z_decoder = self.physics_layer(z_encoder, q)
            else:
                z_decoder = torch.cat((z_encoder, q.view(-1, 1)), dim=1)
        else:
            z_decoder = z_encoder

        return z_decoder

    def encode(self, x):
        z_params = self.encoder(x)
        mu = z_params[:, :self.z_dim_encoder]
        logvar = z_params[:, self.z_dim_encoder:]

        z_encoder = utils.reparametrize(mu, logvar)

        return z_encoder, mu, logvar

    def decode(self, z_decoder):
        return self.decoder(z_decoder)


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


class OnlyDecoder(BaseModel):
    def __init__(self, c):
        super(OnlyDecoder, self).__init__()
        self.config = c

        self.decoder = nn.Sequential(
            nn.Linear(c['z_dim_decoder'], 32 * 4 * 4),  # B, 256
            nn.ReLU(True),
            View((-1, 32, 4, 4)),  # B,  32,  4,  4
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 16, 3, 1, 1),  # B,  32,  8,  8
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 8, 3, 1, 1),  # B,  32, 16, 16
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(8, 4, 3, 1, 1),  # B,  32, 32, 32
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(4, c['len_out_sequence'], 3, 1, 1),
        )

    def forward(self, x):
        return self.decoder(x)
