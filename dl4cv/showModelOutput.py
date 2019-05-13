import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from dl4cv.utils import get_normalization_one_frame
from torch.utils.data.sampler import SequentialSampler


config = {

    'data_path': '../datasets', # Path to the parent directory of the image folder
    'dataset_name': 'ball',     # Name of the image folder

    'model_path': '../saves/train20190513154226/model40',

    'batch_size': 100,
    'num_show_images': 20,              # Number of images to show
}


def eval_model(model, images):
    plt.interactive(False)
    for image in images:
        to_pil = transforms.ToPILImage()

        plt.subplot(1, 3, 1)
        plt.imshow(to_pil(image), cmap='gray')

        restored = model(torch.unsqueeze(image, 0))

        plt.subplot(1, 3, 2)
        plt.imshow(to_pil(restored[0]), cmap='gray')  # TODO: Unnormalize data before display?

        plt.subplot(1, 3, 3)
        plt.imshow(to_pil(abs(restored[0]-image)), cmap='gray')
        plt.show(block=True)


mean, std = get_normalization_one_frame(
    os.path.join(config['data_path'], config['dataset_name'], 'frame0'), 'L'
)

data_set = datasets.ImageFolder(
    config['data_path'],
    transform=transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
)

data_loader = torch.utils.data.DataLoader(
    dataset=data_set,
    batch_size=config['batch_size']
)

model = torch.load(config['model_path'])

batch = next(iter(data_loader))
batch = batch[0]
images = batch[np.linspace(0, config['batch_size']-1, config['num_show_images'], dtype=int).tolist()]

eval_model(model, images)
