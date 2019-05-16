import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms

from dl4cv.utils import CustomDataset
from torch.utils.data.sampler import SequentialSampler


config = {

    'data_path': '../datasets/ball', # Path to the parent directory of the image folder
    'dataset_name': 'ball',     # Name of the image folder

    'model_path': '../saves/train20190515144524/model400',

    'batch_size': 100,
    'num_show_images': 2,              # Number of images to show
}


def eval_model(model, samples):
    plt.interactive(False)

    num_images = sequence_length+2

    for sample in samples:

        x, y = sample

        to_pil = transforms.ToPILImage()

        for i in range(sequence_length-1):
            plt.subplot(2, num_images/2, i+1)
            plt.imshow(to_pil(x[i]), cmap='gray')

        ax = plt.subplot(2, num_images/2, sequence_length)
        plt.imshow(to_pil(y), cmap='gray')
        ax.set_title('Ground truth')

        y_pred = model(torch.unsqueeze(x, 0))

        ax = plt.subplot(2, num_images/2, sequence_length+1)
        plt.imshow(to_pil(y_pred[0]), cmap='gray')
        ax.set_title('Prediction')

        ax = plt.subplot(2, num_images/2, sequence_length+2)
        plt.imshow(to_pil(abs(y_pred[0]-y)), cmap='gray')
        ax.set_title('Deviation')
        plt.show(block=True)


sequence_length = 4 # 3 images as input sequence, 1 predicted image

dataset = CustomDataset(
    config['data_path'],
    transform=transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ]),
    sequence_length=sequence_length
)

data_loader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=config['batch_size']
)

model = torch.load(config['model_path'])

batch = next(iter(data_loader))
# Pick samples from the batch equidistantly based on "num_show_images"
indices = np.linspace(0, config['batch_size'] - 1, config['num_show_images'], dtype=int).tolist()
samples = [(batch[0][indices[i]], batch[1][indices[i]]) for i in range(len(indices))]

eval_model(model, samples)