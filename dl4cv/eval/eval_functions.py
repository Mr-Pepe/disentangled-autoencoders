import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms

from dl4cv.dataset_stuff.dataset_utils import CustomDataset
from dl4cv.utils import read_csv, reparametrize


def analyze_dataset(dataset, indices):
    meta = np.array([dataset.get_ground_truth(i) for i in indices])

    plt.scatter(meta[:, :, 0].reshape(-1), meta[:, :, 1].reshape(-1), s=0.2)
    plt.title("Position")
    plt.xlabel("Position x")
    plt.ylabel("Position y")
    plt.show()

    n = meta.shape[0]

    meta = meta[:, 0]

    plt.plot(np.sort(meta[:, 0]))

    plt.title("Initial Positions in x")
    plt.show()

    plt.plot(np.sort(meta[:, 2]))
    plt.title("Initial Velocities in x")
    plt.show()

    plt.plot(np.sort(meta[:, 4]))
    plt.title("Initial Accelerations in x")
    plt.show()

    meta_mean = meta.mean(axis=0)
    meta_std = meta.std(axis=0)

    correlations = np.zeros((meta.shape[1], meta.shape[1]))

    # Calculate correlation from every latent variable to every ground truth variable
    for i_z in range(meta.shape[1]):
        for i_gt in range(meta.shape[1]):
            # Calculate correlation
            # From https://www.dummies.com/education/math/statistics/how-to-calculate-a-correlation/
            correlations[i_z, i_gt] = 1 / (n - 1) * ((meta[:, i_z] - meta_mean[i_z]) * (meta[:, i_gt] - meta_mean[i_gt])).sum() / \
                                      (meta_std[i_z] * meta_std[i_gt])

    # correlations = np.abs(correlations)

    plt.imshow(correlations, cmap='hot', interpolation='nearest')
    plt.xlabel('Ground truth variables')
    plt.ylabel('Latent variables')
    plt.colorbar()
    plt.show()


def show_solver_history(solver):

    print("Stop reason: %s" % solver.stop_reason)
    print("Stop time: %fs" % solver.training_time_s)
    print("Epoch: {}".format(solver.epoch))
    print("Beta: {}".format(solver.beta))

    train_loss = np.array(solver.history['train_loss'])
    val_loss = np.array(solver.history['val_loss'])
    total_kl_divergence = np.array(solver.history['total_kl_divergence'])
    kl_divergence_dim_wise = np.array(solver.history['kl_divergence_dim_wise'])
    reconstruction_loss = np.array(solver.history['reconstruction_loss'])
    beta = np.array(solver.history['beta'])

    plt.plot(train_loss, label='Train loss')
    plt.plot(total_kl_divergence*beta, label='Overall KL divergence scaled with beta')
    plt.plot(reconstruction_loss, label='Reconstruction loss')
    plt.plot(np.linspace(1, len(train_loss), len(val_loss)), val_loss, label='Validation loss')
    plt.xlabel("Iterations")
    plt.ylabel("Train/Val loss")
    plt.legend()

    plt.show()

    for i in range(kl_divergence_dim_wise.shape[1]):
        plt.plot(moving_average(kl_divergence_dim_wise[:, i], 20), label='z{}'.format(i))
    plt.xlabel("Iterations")
    plt.ylabel("KL Divergences")
    plt.legend()

    plt.show()


def show_latent_variables(model, dataset, show=True):
    mu = 0
    std = 0
    z = 0

    len_dataset = len(dataset)
    log_interval = len_dataset//20

    print('\n', end='')

    for i_sample, sample in enumerate(dataset):

        if (i_sample + 1) % log_interval == 0:
            print("\rGetting latent variables for the dataset: {}/{}".format(i_sample+1, len_dataset), end='')

        x, _, _, _ = sample

        z_tmp, mu_tmp, logvar = model.encode(torch.unsqueeze(x, 0))

        std_tmp = logvar.div(2).exp()

        if not torch.is_tensor(mu):
            mu = mu_tmp.clone().detach()
            std = std_tmp.clone().detach()
            z = z_tmp.clone().detach()

        else:
            mu = torch.cat((mu, mu_tmp.detach()))
            std = torch.cat((std, std_tmp.detach()))
            z = torch.cat((z, z_tmp.detach()))

    print('\n', end='')

    if show:
        ax = plt.subplot(2, 1, 1)
        ax.set_title("Mu")
        for i in range(mu.shape[1]):
            plt.scatter(np.ones((mu.shape[0])) * (i + 1), mu.view(mu.shape[0], mu.shape[1]).detach().numpy()[:, i])

        ax = plt.subplot(2, 1, 2)
        ax.set_title("Std")
        for i in range(std.shape[1]):
            plt.scatter(np.ones((std.shape[0])) * (i + 1), std.view(std.shape[0], std.shape[1]).detach().numpy()[:, i])

        plt.show()

    return z


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


def show_correlation(model, dataset, z, gt):

    z = z.view(z.shape[0], -1).numpy()

    gt = np.array(gt)
    gt = gt[:, 0, :]

    z_mean = z.mean(axis=0)
    gt_mean = gt.mean(axis=0)

    z_std = z.std(axis=0)
    gt_std = gt.std(axis=0)

    n = z.shape[0]

    correlations = np.zeros((z.shape[1], gt.shape[1]))

    # Calculate correlation from every latent variable to every ground truth variable
    for i_z in range(z.shape[1]):
        for i_gt in range(gt.shape[1]):

            # Calculate correlation
            # From https://www.dummies.com/education/math/statistics/how-to-calculate-a-correlation/
            correlations[i_z, i_gt] = 1/(n-1) * ((z[:, i_z] - z_mean[i_z]) * (gt[:, i_gt] - gt_mean[i_gt])).sum() / \
                                      (z_std[i_z]*gt_std[i_gt])

    correlations = np.abs(correlations)

    plt.imshow(correlations, cmap='hot', interpolation='nearest')
    plt.xlabel('Ground truth variables')
    plt.ylabel('Latent variables')
    plt.colorbar()
    plt.show()

    return correlations


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


def moving_average(a, n=3) :
    # From https://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
