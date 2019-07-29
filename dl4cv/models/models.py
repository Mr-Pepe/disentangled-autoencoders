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
            nn.Conv2d(len_in_sequence, 32, 4, 2, 1),  # 32x32
            SkipConv(32, 32),               # 16x16
            SkipConv(32, 32),               # 8x8
            SkipConv(32, 32),               # 4x4
            Flatten(),
            nn.Linear(4*4*32, 256),
            nn.ReLU(True),
            nn.Linear(256, z_dim_encoder * 2),  # B, z_dim*2
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim_decoder, 256),  # B, 256
            nn.ReLU(True),
            nn.Linear(256, 512),  # B, 256
            nn.ReLU(True),
            View((-1, 32, 4, 4)),  # B,  32,  4,  4
            SkipUpConv(32, 32),  # 8x8
            SkipUpConv(32, 32),  # 16x16
            SkipUpConv(32, 32),  # 32x32
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, len_out_sequence, 3, 1, 1),
        )

        if self.use_physics:
            self.physics_layer = PhysicsLayer(dt=1. / 10.)

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            if block != 'physics_layer':
                for m in self._modules[block]:
                    kaiming_init(m)

    def forward(self, x, q=-1):
        z_encoder, mu, logvar = self.encode(x)

        z_decoder = self.bottleneck(z_encoder, q)

        y = self.decode(z_decoder)

        return y, (mu, logvar)

    def bottleneck(self, z_encoder, q):
        q = q.to(next(self.parameters()).device)
        if torch.any(q != -1):
            if self.use_physics:
                z_decoder = self.physics_layer(z_encoder, q.view(-1))
            else:
                z_decoder = torch.cat((z_encoder, q.view(-1, 1)), dim=1)
        else:
            if self.use_physics:
                z_decoder = self.physics_layer(z_encoder, torch.ones((z_encoder.shape[0]),
                                                                     device=next(self.parameters()).device))
            else:
                z_decoder = z_encoder

        return z_decoder

    def encode(self, x):
        z_params = self.encoder(x)
        mu = z_params[:, :self.z_dim_encoder]
        logvar = z_params[:, self.z_dim_encoder:]

        if self.use_physics:
            z_encoder = mu
        else:
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


class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


class SkipConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SkipConv, self).__init__()
        self.down_sample = nn.AvgPool2d(2, 2)
        self.conv = nn.Conv2d(in_channels, out_channels, 4, 2, 1)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.relu(self.down_sample(x) + self.conv(x))


class SkipUpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SkipUpConv, self).__init__()
        self.up_sample = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.up_sample(x)
        return self.relu(out + self.conv(out))


class PhysicsLayer(nn.Module):
    def __init__(self, dt=1./10):
        super(PhysicsLayer, self).__init__()
        self.dt = torch.nn.Parameter(torch.tensor([dt]))
        self.dt.requires_grad = False

    def forward(self, z, q):
        """
            x.shape: [batch, 6, 1, 1]
            return shape: [batch, 2, 1, 1]
        """
        q *= self.dt

        out = torch.zeros_like(z[:, :2])

        out[:, 0] = z[:, 0] + z[:, 2] * q + z[:, 4] * 0.5 * q.pow(2.)
        out[:, 1] = z[:, 1] + z[:, 3] * q + z[:, 5] * 0.5 * q.pow(2.)

        return out
