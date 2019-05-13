import argparse
import logging
import os
import torch
import pickle
import torch.nn as nn
from torchvision import transforms

from dl4cv.models.models import VanillaVAE
from torch.utils.data import DataLoader, SequentialSampler, SubsetRandomSampler
from torchvision import datasets
from dl4cv.solver import Solver

config = {

    'use_cuda': False,

    # Training continuation
    'continue_training':   False,      # Specify whether to continue training with an existing model and solver
    'model_path': '',
    'solver_path': '',

    # Data
    'data_path':        '../datasets/', # Path to the parent directory of the image folder
    'do_overfitting': True,             # Set overfit or regular training
    'num_train_regular':    100000,     # Number of training samples for regular training
    'num_val_regular':      1000,       # Number of validation samples for regular training
    'num_train_overfit':    100,        # Number of training samples for overfitting test runs

    'num_workers': 1,                   # Number of workers for data loading

    ## Hyperparameters ##
    'max_train_time_s': None,
    'num_epochs': 1000,                  # Number of epochs to train
    'batch_size': 1,
    'learning_rate': 2e-4,
    'betas': (0.9, 0.999),              # Beta coefficients for ADAM

    ## Logging ##
    'log_interval': 1,           # Number of mini-batches after which to print training loss
    'save_interval': 50,         # Number of epochs after which to save model and solver
    'save_path': '../saves'
}


""" Add a seed to have reproducible results """

seed = 123
torch.manual_seed(seed)


""" Configure training with or without cuda """

if config['use_cuda'] and torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed(seed)
    kwargs = {'pin_memory': True}
else:
    device = torch.device("cpu")
    torch.set_default_tensor_type('torch.FloatTensor')
    kwargs = {}


""" Load dataset """

logging.info("Loading dataset..")

dataset = datasets.ImageFolder(config['data_path'], transform=transforms.Compose([transforms.ToTensor(),
                                                                                  transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])]))

if config['batch_size'] > len(dataset):
    raise Exception('Batch size bigger than the dataset.')

if config['do_overfitting']:
    if config['batch_size'] > config['num_train_overfit']:
        raise Exception('Batchsize for overfitting bigger than the number of samples for overfitting.')
    else:
        train_data_sampler = SequentialSampler(range(config['num_train_overfit']))
        val_data_sampler = SequentialSampler(range(config['num_train_overfit']))

else:
    if config['num_train_regular']+config['num_val_regular'] > len(dataset):
        raise Exception('Trying to use more samples for training and validation than are available.')
    else:
        train_data_sampler  = SubsetRandomSampler(range(config['num_train_regular']))
        val_data_sampler    = SubsetRandomSampler(range(config['num_train_regular'], config['num_train_regular']+config['num_val_regular']))


train_data_loader   = torch.utils.data.DataLoader(dataset=dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], sampler=train_data_sampler, **kwargs)
val_data_loader     = torch.utils.data.DataLoader(dataset=dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], sampler=val_data_sampler, **kwargs)


""" Initialize model and solver """

if config['continue_training']:
    model = torch.load(config['model_path'])
    solver = pickle.load(open(config['solver_path'], 'rb'))
    loss_criterion = None
    optimizer = None

else:
    model = VanillaVAE()
    solver = Solver()
    loss_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])


""" Perform training """

if __name__ == "__main__":
    solver.train(model=model,
                 optim=optimizer,
                 loss_criterion=loss_criterion,
                 num_epochs=config['num_epochs'],
                 max_train_time_s=config['max_train_time_s'],
                 train_loader=train_data_loader,
                 val_loader=val_data_loader,
                 log_after_iters=config['log_interval'],
                 save_after_epochs=config['save_interval'],
                 save_path=config['save_path'],
                 device=device)