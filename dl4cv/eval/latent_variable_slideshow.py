import copy
import os

import torch
import torchvision.transforms as transforms
from torchvision.datasets.folder import pil_loader
from torchvision.utils import make_grid

import numpy as np

import matplotlib.pyplot as plt

from dl4cv.dataset_stuff.dataset_utils import CustomDataset


class IndexTracker(object):
    def __init__(self, ax, imgs):
        self.ax = ax

        # Remove axis ticks
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_tick_params(which='both', length=0, labelleft=False)

        self.imgs = imgs
        self.num_imgs = imgs.shape[0]
        self.ind = 0

        self.im = ax.imshow(self.imgs[self.ind])
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.num_imgs
        else:
            self.ind = (self.ind - 1) % self.num_imgs
        self.update()

    def update(self):
        self.im.set_data(self.imgs[self.ind])
        self.ax.set_title('Step {} of {}'.format(self.ind, self.num_imgs), rotation=0, size=14)
        self.im.axes.figure.canvas.draw()


def latent_variable_slideshow(model, dataset):
    encoder = model.encoder
    decoder = model.decoder

    # Set question to be a fixed value in the future
    question = torch.tensor([10], dtype=torch.float32)[:, None, None, None]

    # Init the min and max Values for the latent variables
    min_latents = torch.tensor([1000] * model.z_dim_encoder, dtype=torch.float32)[None, :, None, None]
    max_latents = torch.tensor([-1000] * model.z_dim_encoder, dtype=torch.float32)[None, :, None, None]
    all_latents = torch.zeros_like(min_latents)

    # Compute the min and max Values for the latent variables
    for i_sample, sample in enumerate(dataset):
        x, _, _, _ = sample

        z_params = encoder(torch.unsqueeze(x, 0))

        # get mu and std of the sample
        mu = z_params[:, :model.z_dim_encoder]
        logvar = z_params[:, model.z_dim_encoder:]
        std = logvar.div(2).exp()

        min_latents = torch.min(min_latents, mu - 3 * std)
        max_latents = torch.max(min_latents, mu + 3 * std)
        all_latents = torch.cat((all_latents, mu), dim=0)

    # Detach min, max and all latents so they don't require gradients
    min_latents = min_latents.detach()
    max_latents = max_latents.detach()
    all_latents = all_latents.detach()

    # Create linspaces to interpolate over
    lin_spaces = torch.tensor(np.linspace(start=min_latents, stop=max_latents, num=20))  # shape: [20, 1, 6, 1, 1]

    # Use the mean of all latents over the whole dataset as reference
    mean_latents = all_latents.mean(dim=0).unsqueeze(dim=0)  # shape: [1, 6, 1, 1]

    # Generate output images
    all_results = []
    for i_lin_space, lin_space in enumerate(lin_spaces):
        print("Step {} in lin_space".format(i_lin_space + 1))
        output_lin_step = []
        for i_latent_var in range(model.z_dim_encoder):
            print("Decoding with latent variable {} of {} according to lin_space".format(
                i_latent_var + 1, model.z_dim_encoder))
            latents = copy.deepcopy(mean_latents)
            latents[:, i_latent_var] = lin_space[:, i_latent_var]

            # Concatenate question to latents for decoding
            z = torch.cat((latents, question), dim=1)

            output_lin_step.append(decoder(z)[0])  # shape [i, 1, 32, 32]

        output_lin_step = torch.stack(output_lin_step)  # shape [6, 1, 32, 32]
        output_lin_step = make_grid(output_lin_step, nrow=model.z_dim_encoder, padding=0)  # shape [3, 32, 192]

        # Concatenate the outputs of every lin_space
        all_results.append(output_lin_step)  # shape [i, 3, 32, 192]

    all_results = torch.stack(all_results)  # shape [20, 3, 32, 192]

    # Plot the results
    fig, ax = plt.subplots(1, 1)

    imgs = all_results.detach().transpose(1, 2).transpose(2, 3)  # shape [20, 32, 192, 3]
    tracker = IndexTracker(ax, imgs)

    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()


if __name__ == '__main__':
    model = torch.load('../saves/train20190615142355/model30')

    dataset = CustomDataset(
        path='../../datasets/ball',
        transform=transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor()
        ]),
        len_inp_sequence=25,
        len_out_sequence=1,
        load_ground_truth=False,
        question=True,
        load_to_ram=False
    )

    latent_variable_slideshow(model, dataset)
