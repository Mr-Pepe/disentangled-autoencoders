import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms

from dl4cv.dataset_utils import CustomDataset
from torch.utils.data.sampler import SequentialSampler


config = {
    'show_images': True,

    'data_path': '../datasets/ball',  # Path to directory of the image folder
    'len_inp_sequence': 3,
    'len_out_sequence': 3,

    'model_path': '../saves/train20190529113101/model100',

    'batch_size': 1000,
    'num_show_images': 10,              # Number of images to show
}


def eval_model(model, samples):
    plt.interactive(False)

    num_images = sequence_length + 2 * out_length

    all_z = torch.Tensor()

    for sample in samples:

        x, y, meta = sample

        print(x.shape, y.shape)

        y_pred, latent_stuff = model(torch.unsqueeze(x, 0))

        if config['show_images']:

            to_pil = transforms.ToPILImage()

            for i in range(inp_length):
                ax = plt.subplot(3, num_images/3, i + 1)
                plt.imshow(to_pil(x[i]), cmap='gray')
                ax.set_title('frame t{}'.format(-(sequence_length - 2) + i))

            for i in range(inp_length, 3 * out_length, 3):
                # Ground Truth image
                ax = plt.subplot(3, num_images/3, i + 1)
                plt.imshow(to_pil(y[i - inp_length]), cmap='gray')
                ax.set_title('Ground truth t+{}'.format(i + 1 - inp_length))

                # Predicted image
                ax = plt.subplot(3, num_images / 3, i + 2)
                plt.imshow(to_pil(y_pred[i - inp_length]), cmap='gray')
                ax.set_title('Prediction t+{}'.format(i + 1 - inp_length))

                # Deviation
                ax = plt.subplot(3, num_images / 3, i + 3)
                diff = abs(y_pred[i - inp_length] - y[i - inp_length])
                plt.imshow(to_pil(diff), cmap='gray')
                ax.set_title('Deviation t+{}'.format(i + 1 - inp_length))

            plt.show(block=True)


inp_length = config['len_inp_sequence']
out_length = config['len_out_sequence']
sequence_length = inp_length + out_length

dataset = CustomDataset(
    config['data_path'],
    transform=transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ]),
    sequence_length=sequence_length,
    len_inp_sequence=config['len_inp_sequence'],
    len_out_sequence=config['len_out_sequence']
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
