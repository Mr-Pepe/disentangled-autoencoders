import os
import torch
import torchvision.transforms as transforms

from dl4cv.utils import EvalLatentDataset
from torch.utils.data import DataLoader

LATENTS = ['px', 'py', 'vx', 'vy', 'ax', 'ay']

config = {
    'data_path': '../datasets/ball',
    'model_path': '../saves/train20190519171353/model10',
    'sequence_length': 3,
}


def eval_latent(encoder, data_loader):
    x, meta = next(iter(data_loader))
    z = encoder(x)
    z = torch.flatten(z, start_dim=1)
    for i in range(len(LATENTS)):
        print("%s, true: %.4f \t predicted: %.4f"
              % (LATENTS[i], meta[:, i].item(), z[:, i].item()))


dataset = EvalLatentDataset(
    path=config['data_path'],
    transform=transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ]),
    sequence_length=config['sequence_length']
)

data_loader = DataLoader(
    dataset=dataset,
    batch_size=1
)


model = torch.load(config['model_path'])
encoder = model.encoder

eval_latent(encoder, data_loader)
