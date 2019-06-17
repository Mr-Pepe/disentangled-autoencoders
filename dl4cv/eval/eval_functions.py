import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms

from dl4cv.dataset_stuff.dataset_utils import CustomDataset
from dl4cv.utils import read_csv, reparametrize


def analyze_dataset(dataset):
    meta = np.array([dataset.get_ground_truth(i) for i in range(len(dataset))])

    plt.scatter(meta[:, :, 0].reshape(-1), meta[:, :, 1].reshape(-1), s=0.2)
    plt.title("Position")
    plt.xlabel("Position x")
    plt.ylabel("Position y")
    plt.show()


def show_solver_history(solver):
    print("Stop reason: %s" % solver.stop_reason)
    print("Stop time: %fs" % solver.training_time_s)

    train_loss = np.array(solver.history['train_loss'])
    val_loss = np.array(solver.history['val_loss'])
    kl_divergence = np.array(solver.history['kl_divergence'])
    reconstruction_loss = np.array(solver.history['reconstruction_loss'])

    f, (ax1, ax2, ax3) = plt.subplots(3, 1)

    ax1.plot(train_loss)
    ax1.plot(np.linspace(1, len(train_loss), len(val_loss)), val_loss)
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Train/Val loss")

    ax2.plot(kl_divergence)
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("KL Divergence")

    ax3.plot(reconstruction_loss)
    ax3.set_xlabel("Iterations")
    ax3.set_ylabel("Reconstruction Loss")

    plt.show()


def show_latent_variables(model, dataset):
    mu = 0
    std = 0

    len_dataset = len(dataset)
    log_interval = len_dataset//20

    print('\n', end='')

    for i_sample, sample in enumerate(dataset):

        if (i_sample + 1) % log_interval == 0:
            print("\rGetting latent variables for the dataset: {}/{}".format(i_sample+1, len_dataset), end='')

        x, _, _, _ = sample

        z_t, mu_tmp, logvar = model.encode(torch.unsqueeze(x, 0))

        std_tmp = logvar.div(2).exp()

        if not torch.is_tensor(mu):
            mu = mu_tmp.clone().detach()
            std = std_tmp.clone().detach()

        else:
            mu = torch.cat((mu, mu_tmp.detach()))
            std = torch.cat((std, std_tmp.detach()))

    print('\n', end='')

    ax = plt.subplot(2, 1, 1)
    ax.set_title("Mu")
    for i in range(mu.shape[1]):
        plt.scatter(np.ones((mu.shape[0])) * (i + 1), mu.view(mu.shape[0], mu.shape[1]).detach().numpy()[:, i])

    ax = plt.subplot(2, 1, 2)
    ax.set_title("Std")
    for i in range(std.shape[1]):
        plt.scatter(np.ones((std.shape[0])) * (i + 1), std.view(std.shape[0], std.shape[1]).detach().numpy()[:, i])

    plt.show()


def show_model_output(model, dataset):
    plt.interactive(False)

    num_cols = 3
    num_rows = 1

    plt.rcParams.update({'font.size': 8})

    for i_sample, sample in enumerate(dataset):
        x, y, question, meta = sample
        y_pred, latent_stuff = model(
            torch.unsqueeze(x, 0), torch.unsqueeze(question, 0)
        )

        to_pil = transforms.ToPILImage()

        f, axes = plt.subplots(num_rows, num_cols)
        f.suptitle("\nSample {}, question: {}".format(i_sample, question), fontsize=16)

        # Plot ground truth
        axes[0].imshow(to_pil(y), cmap='gray')

        # Plot prediction
        axes[1].imshow(to_pil(y_pred[0]), cmap='gray')

        # Plot Deviation
        diff = abs(y_pred[0] - y)
        axes[2].imshow(to_pil(diff), cmap='gray')

        # Remove axis ticks
        for ax in axes.reshape(-1):
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_tick_params(which='both', length=0, labelleft=False)

        # Label rows
        labels = {0: 'Ground truth',
                  1: 'Prediction',
                  2: 'Deviation'}

        for i in range(num_cols):
            plt.sca(axes[i])
            axes[i].set_title(labels[i], rotation=0, size=14)

        f.tight_layout()

        plt.show(block=True)


def eval_correlation(model, variables, path, len_inp_sequence, len_out_sequence):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    encoder = model.encoder

    for i_var, var in enumerate(variables):
        eval_path = os.path.join(path, var)
        dataset = CustomDataset(
            path=eval_path,
            transform=transform,
            len_inp_sequence=len_inp_sequence,
            len_out_sequence=len_out_sequence,
            load_ground_truth=False,
            load_to_ram=False,
            only_input=True
        )

        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=len(dataset)
        )

        linspace = np.array(read_csv(os.path.join(eval_path, 'linspace.csv')))

        x, _, _, _ = next(iter(data_loader))
        z_t = _get_z(encoder, x, z_dim=model.z_dim_encoder)

        f, axes = plt.subplots(len(variables), 1, figsize=(10, 10))

        for i_z, z in enumerate(z_t):
            m, c = _regression_line(linspace, z)
            axes[i_z].plot(linspace, z)
            axes[i_z].plot(linspace, m*linspace + c, label='Regression line')
            axes[i_z].set_ylabel("latent %d" % i_z)
            axes[i_z].set_ylim([-3, 3])

        axes[0].legend(loc='upper left')
        axes[-1].set_xlabel(var, fontsize=16)

        f.tight_layout()

        plt.show()


def _regression_line(x, y):
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c


def _get_z(encoder, x, z_dim):
    z_params = encoder(x)
    mu = z_params[:, :z_dim]
    logvar = z_params[:, z_dim:]

    z_t = reparametrize(mu, logvar)
    z_t = z_t.reshape([z_t.shape[0], -1]).transpose(1, 0)

    return z_t.detach().numpy()
