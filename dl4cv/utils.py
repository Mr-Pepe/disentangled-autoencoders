import csv

import numpy as np
import torchvision.transforms as transforms
import torch
from PIL import Image


def kl_divergence(mu, logvar):
    # Taken from https://github.com/1Konny/Beta-VAE/blob/master/solver.py
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


def reparametrize(mu, logvar):
    # Taken from https://github.com/1Konny/Beta-VAE/blob/master/model.py
    std = logvar.div(2).exp()
    eps = torch.autograd.Variable(std.data.new(std.size()).normal_())
    return mu + std*eps


def get_normalization_one_frame(filename: str, format: str):
    """
    Gets the mean and standard deviation based on a single frame in the dataset
    This is sufficient for the pong dataset since all frames only contain one
    object and the mean and std are translational and rotational invariant.
    Args:
        filename:
            path to one image in the dataset -> str
        format:
            format of the dataset. Supports 'L' for Grayscale and
            'RGB' for RGB images -> str
    """
    frame = Image.open(filename).convert(format)
    loader = transforms.ToTensor()
    frame = loader(frame)
    mean = frame.mean(dim=(1, 2))
    std = frame.std(dim=(1, 2))
    return mean, std


def tensor_denormalizer(mean, std):
    """
    Denormalizes image to save or display it
    """
    return transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.], std=1/std),
        transforms.Normalize(mean=-mean, std=[1., 1., 1.])])


def save_csv(data, path):
    with open(path, 'w') as f:
        writer = csv.writer(f, delimiter='|')
        writer.writerow(data)


def read_csv(path):
    return np.genfromtxt(path, dtype=np.float, delimiter='|', skip_header=0)
