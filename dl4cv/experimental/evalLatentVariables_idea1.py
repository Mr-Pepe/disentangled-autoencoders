import torch

from dl4cv.dataset_utils import ForwardAndMirroredDatasetRAM
from torch.utils.data import DataLoader

LATENTS = ['px', 'py', 'vx', 'vy']

config = {
    'data_path': '../../datasets/noAcceleration',
    'model_path': '../../saves/train20190527075708/model20',
}


def eval_latent(encoder, data_loader):
    imgs, _, meta = next(iter(data_loader))
    _, _, z_t, z_t_plus_1 = encoder(imgs[:, :-1])
    z_t = torch.flatten(z_t, start_dim=1)
    for i in range(len(LATENTS)):
        print("%s, true: %.4f \t predicted: %.4f"
              % (LATENTS[i], meta[:, i].item(), z_t[:, i].item()))


dataset = ForwardAndMirroredDatasetRAM(
    config['data_path'],
    num_sequences=1000,
    load_meta=True
)

data_loader = DataLoader(
    dataset=dataset,
    batch_size=1
)


model = torch.load(config['model_path'])

eval_latent(model, data_loader)
