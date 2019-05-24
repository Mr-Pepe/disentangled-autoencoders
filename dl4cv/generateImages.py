import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms

from dl4cv.utils import CustomDataset
from torch.utils.data.sampler import SequentialSampler

to_pil = transforms.ToPILImage()

config = {
    'model_path': '../saves/train20190524114640/model140',
}

model = torch.load(config['model_path'])
model.eval()

z_dim = 5

for i_variable in range(z_dim):
    print("Modifying variable " + str(i_variable + 1))
    z = torch.zeros((z_dim,))

    for value_variable in torch.linspace(-0.1, 0.1, 10):
        z[i_variable] = value_variable

        plt.imshow(to_pil(model.decoder(z.view(1, z_dim, 1, 1)).detach()[0]), cmap='gray')
        plt.show()
