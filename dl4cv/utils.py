import csv
import datetime
import time
import argparse

import numpy as np
import torch

from scipy.special import gamma, psi
from sklearn.neighbors import NearestNeighbors


class Config(dict):
    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)
        self.__dict__ = self


def kl_divergence(mu, logvar, target_var=1.):
    # Taken from https://github.com/1Konny/Beta-VAE/blob/master/solver.py
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    target_var = torch.ones_like(mu, dtype=torch.float32) * target_var

    # klds = 0.5 * ((mu.pow(2) / target_var) + (logvar.exp() / target_var) - logvar + target_var.log())
    klds = -0.5*(1 + logvar - (mu.pow(2) / target_var) - (logvar.exp() / target_var))
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


def reparametrize(mu, logvar):
    # Taken from https://github.com/1Konny/Beta-VAE/blob/master/model.py
    std = logvar.div(2).exp()
    eps = torch.autograd.Variable(std.data.new(std.size()).normal_())
    return mu + std*eps


def save_csv(data, path):
    with open(path, 'w') as f:
        writer = csv.writer(f, delimiter='|')
        writer.writerow(data)


def read_csv(path):
    return np.genfromtxt(path, dtype=np.float, delimiter='|', skip_header=0)


def time_left(t_start, n_iters, i_iter):
    iters_left = n_iters - i_iter
    time_per_iter = (time.time() - t_start) / i_iter
    time_left = time_per_iter * iters_left
    time_left = datetime.datetime.fromtimestamp(time_left)
    return time_left.strftime("%H:%M:%S")


def nearest_distances(X, k=1):
    """
    Source: https://gist.github.com/GaelVaroquaux/ead9898bd3c973c40429
    X = array(N,M)
    N = number of points
    M = number of dimensions
    returns the distance to the kth nearest neighbor for every point in X
    """
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(X)
    d, _ = knn.kneighbors(X)  # the first nearest neighbor is itself
    return d[:, -1]  # returns the distance to the kth nearest neighbor


def entropy(X, k=1):
    """
    Source: https://gist.github.com/GaelVaroquaux/ead9898bd3c973c40429
    Returns the entropy of the X.
    Parameters
    ===========
    X : array-like, shape (n_samples, n_features)
        The data the entropy of which is computed
    k : int, optional
        number of nearest neighbors for density estimation
    Notes
    ======
    Kozachenko, L. F. & Leonenko, N. N. 1987 Sample estimate of entropy
    of a random vector. Probl. Inf. Transm. 23, 95-101.
    See also: Evans, D. 2008 A computationally efficient estimator for
    mutual information, Proc. R. Soc. A 464 (2093), 1203-1215.
    and:
    Kraskov A, Stogbauer H, Grassberger P. (2004). Estimating mutual
    information. Phys Rev E 69(6 Pt 2):066138.
    """

    # Distance to kth nearest neighbor
    r = nearest_distances(X, k)  # squared distances
    n, d = X.shape
    volume_unit_ball = (np.pi**(.5*d)) / gamma(.5*d + 1)
    '''
    F. Perez-Cruz, (2008). Estimation of Information Theoretic Measures
    for Continuous Random Variables. Advances in Neural Information
    Processing Systems 21 (NIPS). Vancouver (Canada), December.
    return d*mean(log(r))+log(volume_unit_ball)+log(n-1)-log(k)
    '''
    return (d*np.mean(np.log(r + np.finfo(X.dtype).eps))
            + np.log(volume_unit_ball) + psi(n) - psi(k))


def mutual_information(variables, k=1):
    """
    Source: https://gist.github.com/GaelVaroquaux/ead9898bd3c973c40429
    Returns the mutual information between any number of variables.
    Each variable is a matrix X = array(n_samples, n_features)
    where
      n = number of samples
      dx,dy = number of dimensions
    Optionally, the following keyword argument can be specified:
      k = number of nearest neighbors for density estimation
    Example: mutual_information((X, Y)), mutual_information((X, Y, Z), k=5)
    """
    if len(variables) < 2:
        raise AttributeError(
                "Mutual information must involve at least 2 variables")
    all_vars = np.hstack(variables)
    return (sum([entropy(X, k=k) for X in variables])
            - entropy(all_vars, k=k))

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
