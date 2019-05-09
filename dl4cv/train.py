import argparse
import logging
import os
import torch
import torch.nn as nn
from torchvision import transforms

from dl4cv.models.models import VanillaVAE
from torch.utils.data import DataLoader
from torchvision import datasets
from dl4cv.solver import Solver


parser = argparse.ArgumentParser(description='Train VAE')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 1)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--use-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()


""" Add a seed to have reproducible results """

torch.manual_seed(args.seed)


""" Configure training with or without cuda """

if args.use_cuda and torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed(args.seed)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    kwargs = {'num_workers': 4, 'pin_memory': True}
else:
    device = torch.device("cpu")
    torch.set_default_tensor_type('torch.FloatTensor')
    kwargs = {}


""" Load dataset """

logging.info("Loading dataset..")

DATASET = "../datasets/"
IMAGE_SIZE = 256

train_dataset = datasets.ImageFolder(DATASET, transform=transforms.ToTensor())
train_loader = DataLoader(
    train_dataset, batch_size=1, shuffle=False, **kwargs)


""" Init Network """

model = VanillaVAE()


""" Prepare Training """

LR = 2e-4
NUM_EPOCHS = 100

# init loss
mse_loss = nn.MSELoss()
# init optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

solver = Solver()


""" Perform training """
if __name__ == "__main__":
    solver.train(model=model,
                 optim=optimizer,
                 loss_criterion=mse_loss,
                 num_epochs=NUM_EPOCHS,
                 max_train_time_s=None,
                 start_epoch=0,
                 lr_decay=1,
                 lr_decay_interval=1,
                 train_loader=train_loader,
                 val_loader=train_loader,
                 log_after_iters=5,
                 save_after_epochs=10,
                 save_path='../saves',
                 device=device)

