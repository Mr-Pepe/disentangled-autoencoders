import argparse
import logging
import os
import torch
import torch.nn as nn
from torchvision import transforms

from dl4cv.models.models import VanillaVAE
from torch.utils.data import DataLoader
from torchvision import datasets


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
    train_dataset, batch_size=args.epochs, shuffle=False, **kwargs)


""" Init Network """

model = VanillaVAE().to(device)


""" Prepare Training """

LR = 2e-4
NUM_EPOCHS = 1

# init loss
mse_loss = nn.MSELoss()
# init optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


def train(epoch):
    model.train()
    train_loss = 0
    logging.info("Start training on {}...".format(device))
    for i, (x, _) in enumerate(train_loader):
        optimizer.zero_grad()
        x = x.to(device)

        generated = model(x)
        loss = mse_loss(generated, x)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


""" Perform training """
if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)

