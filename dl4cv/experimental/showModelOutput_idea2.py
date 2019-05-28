import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms

from dl4cv.dataset_utils import ForwardDatasetIdea2RAM
from torch.utils.data.sampler import SequentialSampler


config = {
    'show_images': True,

    'data_path': '../../datasets/noAcceleration',  # Path to directory of the image folder

    'model_path': '../../saves/train20190526233933/model100',

    'batch_size': 1000,
    'num_show_images': 10,              # Number of images to show
}

sequence_length = 3
num_images = 6


def eval_model(model, samples):
    plt.interactive(False)

    for sample in samples:

        x, y3, meta = sample

        _, y3_t, _, _, _, _ = model(torch.unsqueeze(x, 0))

        if config['show_images']:
            to_pil = transforms.ToPILImage()

            for i in range(sequence_length - 1):
                ax = plt.subplot(2, num_images / 2, i + 1)
                plt.imshow(to_pil(x[i]), cmap='gray')
                ax.set_title('frame t{}'.format(-(sequence_length-2) + i))

            ax = plt.subplot(2, num_images / 2, sequence_length)
            plt.imshow(to_pil(y3), cmap='gray')
            ax.set_title('frame t+1')

            ax = plt.subplot(2, num_images / 2, sequence_length + 1)
            plt.imshow(to_pil(y3), cmap='gray')
            ax.set_title('Ground Truth t+1')

            ax = plt.subplot(2, num_images / 2, sequence_length + 2)
            plt.imshow(to_pil(y3_t[0]), cmap='gray')
            ax.set_title('Prediction t+1')

            ax = plt.subplot(2, num_images / 2, sequence_length + 3)
            plt.imshow(to_pil(abs(y3_t[0] - y3)), cmap='gray')
            ax.set_title('Deviation t+1')

            plt.show(block=True)


dataset = ForwardDatasetIdea2RAM(
    config['data_path'],
    num_sequences=config['batch_size'],
    load_meta=False
)

data_loader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=config['batch_size']
)

model = torch.load(config['model_path'])

x, y2, y2_hor_flip, y2_ver_flip, y3, y3_hor_flip, y3_ver_flip, meta = next(iter(data_loader))
# Pick samples from the batch equidistantly based on "num_show_images"
indices = np.linspace(0, config['batch_size'] - 1, config['num_show_images'], dtype=int).tolist()
samples = [(x[indices[i]], y3[indices[i]], meta[indices[i]]) for i in range(len(indices))]

eval_model(model, samples)
