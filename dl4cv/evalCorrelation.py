import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from torchvision.transforms import transforms

from dl4cv.dataset_utils import CustomDataset
from dl4cv.utils import reparametrize, read_csv

config = {
    'data_path': '../datasets/evalDataset',  # Path to directory of the image folder
    'len_inp_sequence': 3,
    'len_out_sequence': 0,

    'model_path': '../saves/train20190530134521/model10',
}

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])


def regression_line(x, y):
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c


def get_z(encoder, x, z_dim):
    z_params = encoder(x)
    mu = z_params[:, :z_dim]
    logvar = z_params[:, z_dim:]

    z_t = reparametrize(mu, logvar)
    z_t = z_t.reshape([z_t.shape[0], -1]).transpose(1, 0)

    return z_t.detach().numpy()


def evaluate(model, variables):
    encoder = model.encoder
    for i_var, var in enumerate(variables):
        path = os.path.join(config['data_path'], var)
        dataset = CustomDataset(
            path=path,
            transform=transform,
            len_inp_sequence=config['len_inp_sequence'],
            len_out_sequence=config['len_out_sequence'],
            load_meta=False,
            load_to_ram=False
        )

        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=len(dataset)
        )

        linspace = np.array(read_csv(os.path.join(path, 'linspace.csv')))

        x, _, _ = next(iter(data_loader))
        z_t = get_z(encoder, x, z_dim=model.z_dim)

        f, axes = plt.subplots(len(variables), 1, figsize=(10, 10))
        for i_z, z in enumerate(z_t):
            m, c = regression_line(linspace, z)
            axes[i_z].plot(linspace, z)
            axes[i_z].plot(linspace, m*linspace + c, label='Regression line')
            axes[i_z].set_ylabel("latent %d" % i_z)

        axes[0].legend(loc='upper left')
        axes[-1].set_xlabel(var)

        plt.show()


variables = ['pos_x', 'pos_y', 'vel_x', 'vel_y', 'acc_x', 'acc_y']

# Load model
model = torch.load(config['model_path'])

evaluate(model, variables)

