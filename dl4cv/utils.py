import csv
import datetime
import time

import numpy as np
import torch


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


class EarlyStopping(object):
    """
    Source: https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d
    Class to perform early stopping in pytorch
    Usage:
    es = EarlyStopping(patience=5)

        num_epochs = 100
        for epoch in range(num_epochs):
            train_one_epoch(model, data_loader)  # train the model for one epoch, on training set
            metric = eval(model, data_loader_dev)  # evalution on dev set (i.e., holdout from training)
            if es.step(metric):
                break  # early stop criterion is met, we can stop now
    """
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)