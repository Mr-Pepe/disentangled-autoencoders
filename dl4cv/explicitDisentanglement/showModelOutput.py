import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms

from dl4cv.dataset_utils import ForwardAndMirroredDatasetRAM
from torch.utils.data.sampler import SequentialSampler


config = {
    'show_images': True,

    'data_path': '../../datasets/noAcceleration',  # Path to directory of the image folder

    'model_path': '../../saves/train20190526222036/model20',

    'batch_size': 1000,
    'num_show_images': 10,              # Number of images to show
}

sequence_length = 3
num_images = 9


def eval_model(model, samples):
    plt.interactive(False)

    all_z = torch.Tensor()

    for sample in samples:

        imgs, meta = sample

        y2_pred, y3_pred, v_t, v_t_plus_1 = model(torch.unsqueeze(imgs[:-1], 0))

        if config['show_images']:
            to_pil = transforms.ToPILImage()

            for i in range(sequence_length):
                ax = plt.subplot(3, num_images / 3, i + 1)
                plt.imshow(to_pil(imgs[i]), cmap='gray')
                ax.set_title('frame t{}'.format(-(sequence_length-2) + i))

            ax = plt.subplot(3, num_images / 3, sequence_length + 1)
            plt.imshow(to_pil(imgs[1]), cmap='gray')
            ax.set_title('Ground Truth t')

            ax = plt.subplot(3, num_images / 3, sequence_length + 2)
            plt.imshow(to_pil(y2_pred[0]), cmap='gray')
            ax.set_title('Prediction t')

            ax = plt.subplot(3, num_images / 3, sequence_length + 3)
            plt.imshow(to_pil(abs(y2_pred[0]-imgs[1])), cmap='gray')
            ax.set_title('Deviation t')

            ax = plt.subplot(3, num_images / 3, sequence_length + 4)
            plt.imshow(to_pil(imgs[-1]), cmap='gray')
            ax.set_title('Ground Truth t+1')

            ax = plt.subplot(3, num_images / 3, sequence_length + 5)
            plt.imshow(to_pil(y3_pred[0]), cmap='gray')
            ax.set_title('Prediction t+1')

            ax = plt.subplot(3, num_images / 3, sequence_length + 6)
            plt.imshow(to_pil(abs(y3_pred[0] - imgs[-1])), cmap='gray')
            ax.set_title('Deviation t+1')

            plt.show(block=True)


dataset = ForwardAndMirroredDatasetRAM(
    config['data_path'],
    num_sequences=config['batch_size'],
    load_meta=False
)

data_loader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=config['batch_size']
)

model = torch.load(config['model_path'])

imgs_normal, _, meta = next(iter(data_loader))
# Pick samples from the batch equidistantly based on "num_show_images"
indices = np.linspace(0, config['batch_size'] - 1, config['num_show_images'], dtype=int).tolist()
samples = [(imgs_normal[indices[i]], meta[indices[i]]) for i in range(len(indices))]

eval_model(model, samples)
