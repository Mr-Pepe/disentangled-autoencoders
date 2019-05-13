import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image, ImageChops
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SequentialSampler, SubsetRandomSampler
import pickle

config = {

    'data_path': '/home/felipe/Projects/dl4cv/datasets', # Path to the parent directory of the image folder

    'model_path': '/home/felipe/Projects/dl4cv/saves/train20190513154226/model40',

    'batch_size': 100,
    'num_show_images': 20,              # Number of images to show
}

def eval_model(model, images):
    plt.interactive(False)
    for image in images:
        to_pil = transforms.ToPILImage()

        plt.subplot(1,3,1)
        plt.imshow(to_pil(image), cmap='gray')

        restored = model(torch.unsqueeze(image, 0))

        plt.subplot(1,3,2)
        plt.imshow(to_pil(restored[0]), cmap='gray')

        plt.subplot(1, 3, 3)
        plt.imshow(to_pil(abs(restored[0]-image)), cmap='gray')
        plt.show(block=True)

data_set = datasets.ImageFolder(config['data_path'], transform=transforms.Compose([transforms.Grayscale(),
                                                                                   transforms.ToTensor(),
                                                                                   transforms.Normalize(mean=[0.5],std=[0.5])]))

data_loader   = torch.utils.data.DataLoader(dataset=data_set, batch_size=config['batch_size'])

model = torch.load(config['model_path'])

batch = next(iter(data_loader))
batch = batch[0]

eval_model(model, batch[np.linspace(0, config['batch_size']-1, config['num_show_images'], dtype=int).tolist()])
