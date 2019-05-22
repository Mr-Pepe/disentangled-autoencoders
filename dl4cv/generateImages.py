import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms

from dl4cv.utils import CustomDataset
from torch.utils.data.sampler import SequentialSampler

to_pil = transforms.ToPILImage()

config = {

    'data_path': '../datasets/ball/images',  # Path to directory of the image folder

    'model_path': '../saves/train20190522122921/model170',

    'batch_size': 100,
    'num_show_images': 100,              # Number of images to show
}

model = torch.load(config['model_path'])
model.eval()

z_dim = 6

for i_variable in range(z_dim):
    print("Modifying variable " + str(i_variable + 1))
    z = torch.zeros((z_dim,))

    for value_variable in torch.linspace(-0.4, 0.4, 50):
        z[i_variable] = value_variable

        plt.imshow(to_pil(model.decoder(z.view(1, z_dim, 1, 1)).detach()[0]), cmap='gray')
        plt.show()
