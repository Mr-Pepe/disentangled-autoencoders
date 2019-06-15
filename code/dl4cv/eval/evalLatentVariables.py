import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from dl4cv.dataset_stuff.dataset_utils import CustomDataset

LATENTS = ['px', 'py', 'vx', 'vy', 'ax', 'ay']

config = {
    'data_path': '../../../datasets/ball',
    'model_path': '../../saves/train20190519171353/model10',
    'len_inp_sequence': 25,
    'len_out_sequence': 1
}

# make all paths absolute
file_dir = os.path.dirname(os.path.realpath(__file__))

config['data_path'] = os.path.join(file_dir, config['data_path'])
config['model_path'] = os.path.join(file_dir, config['model_path'])


def eval_latent(encoder, data_loader):
    x, y, meta = next(iter(data_loader))
    x = torch.cat([x, y], dim=0)
    z = encoder(x)
    z = torch.flatten(z, start_dim=1)
    for i in range(len(LATENTS)):
        print("%s, true: %.4f \t predicted: %.4f"
              % (LATENTS[i], meta[:, i].item(), z[:, i].item()))


dataset = CustomDataset(
    path=config['data_path'],
    transform=transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ]),
    len_inp_sequence=config['sequence_length'],
    len_out_sequence=config['len_out_sequence'],
    load_ground_truth=True
)

data_loader = DataLoader(
    dataset=dataset,
    batch_size=1
)


model = torch.load(config['model_path'])
encoder = model.encoder

eval_latent(encoder, data_loader)
