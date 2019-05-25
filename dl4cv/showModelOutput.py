import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms

from dl4cv.dataset_utils import CustomDataset
from torch.utils.data.sampler import SequentialSampler


config = {
    'show_images': True,

    'data_path': '../datasets/ball',  # Path to directory of the image folder

    'model_path': '../saves/train20190522122921/model170',

    'batch_size': 1000,
    'num_show_images': 10,              # Number of images to show
}


def eval_model(model, samples):
    plt.interactive(False)

    num_images = sequence_length+2

    all_z = torch.Tensor()

    for sample in samples:

        x, y, meta = sample

        y_pred, latent_stuff = model(torch.unsqueeze(x, 0))

        if config['show_images']:

            to_pil = transforms.ToPILImage()

            for i in range(sequence_length-1):
                ax = plt.subplot(2, num_images/2, i+1)
                plt.imshow(to_pil(x[i]), cmap='gray')
                ax.set_title('frame t{}'.format(-(sequence_length-2) + i))

            ax = plt.subplot(2, num_images/2, sequence_length)
            plt.imshow(to_pil(y), cmap='gray')
            ax.set_title('Ground truth t+1')

            ax = plt.subplot(2, num_images/2, sequence_length+1)
            plt.imshow(to_pil(y_pred[0]), cmap='gray')
            ax.set_title('Prediction t+1')

            ax = plt.subplot(2, num_images/2, sequence_length+2)
            plt.imshow(to_pil(abs(y_pred[0]-y)), cmap='gray')
            ax.set_title('Deviation')

            plt.show(block=True)


sequence_length = 4  # 3 images as input sequence, 1 predicted image

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

x, y, meta = next(iter(data_loader))
# Pick samples from the batch equidistantly based on "num_show_images"
indices = np.linspace(0, config['batch_size'] - 1, config['num_show_images'], dtype=int).tolist()
samples = [(x[indices[i]], y[indices[i]], meta[indices[i]]) for i in range(len(indices))]

eval_model(model, samples)
