import copy
import os
import random

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

import torch
import torchvision.transforms as transforms

from sklearn import linear_model
from sklearn.metrics import mutual_info_score

from dl4cv.utils import reparametrize, mutual_information, entropy


def analyze_dataset(trajectories, window_size_x=32, window_size_y=32, mode='lines'):

    mpl.rcParams['axes.titlesize'] = 'large'
    mpl.rcParams['axes.labelsize'] = 'large'

    plt.figure(figsize=(6, 6))
    if mode == 'lines':
        for i in range(trajectories.shape[0]):
            plt.plot(trajectories[i, :, 0].reshape(-1), trajectories[i, :, 1].reshape(-1), 'b', linewidth=0.5)
    elif mode == 'points':
        plt.scatter(trajectories[:, :, 0].reshape(-1), trajectories[:, :, 1].reshape(-1), s=0.2)

    plt.title("Position")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(left=0, right=window_size_x)
    plt.ylim(bottom=0, top=window_size_y)
    plt.show()

    n = trajectories.shape[0]

    trajectories = trajectories[:, 0]

    meta_mean = trajectories.mean(axis=0)
    meta_std = trajectories.std(axis=0)

    correlations = np.zeros((trajectories.shape[1], trajectories.shape[1]))

    # Calculate correlation from every ground truth variable to itself
    for i_z in range(trajectories.shape[1]):
        for i_gt in range(trajectories.shape[1]):
            # Calculate correlation
            # From https://www.dummies.com/education/math/statistics/how-to-calculate-a-correlation/
            correlations[i_z, i_gt] = 1 / (n - 1) * ((trajectories[:, i_z] - meta_mean[i_z]) * (trajectories[:, i_gt] - meta_mean[i_gt])).sum() / \
                                      (meta_std[i_z] * meta_std[i_gt])

    correlations = np.abs(correlations)

    plt.imshow(correlations, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
    plt.title('Variable intercorrelation')
    plt.xticks(np.arange(6), ('px', 'py', 'vx', 'vy', 'ax', 'ay'), fontsize=14)
    plt.yticks(np.arange(6), ('px', 'py', 'vx', 'vy', 'ax', 'ay'), fontsize=14)
    plt.colorbar()
    plt.gca().xaxis.tick_top()
    plt.gca().xaxis.set_label_position('top')
    plt.gcf().tight_layout()
    plt.show()


def show_solver_history(solver):

    avg_w = 20

    print("Stop reason: %s" % solver.stop_reason)
    print("Stop time: %fs" % solver.training_time_s)
    print("Epoch: {}".format(solver.epoch))

    total_kl_divergence = np.array(solver.history['total_kl_divergence'])
    kl_divergence_dim_wise = np.array(solver.history['kl_divergence_dim_wise'])
    reconstruction_loss = np.array(solver.history['reconstruction_loss'])
    posterior_mu = np.array(solver.history['posterior_mu'])
    posterior_var = np.array(solver.history['posterior_var'])

    plt.plot(moving_average(reconstruction_loss[100:], 100), label='Reconstruction loss')
    plt.xlabel("Iterations")
    plt.ylabel("Reconstruction loss")
    plt.show()

    plt.plot(total_kl_divergence, label='Total')
    for i in range(kl_divergence_dim_wise.shape[1]):
        plt.plot(moving_average(kl_divergence_dim_wise[:, i], avg_w), label='z{}'.format(i))
    plt.xlabel("Iterations")
    plt.ylabel("KL Divergences")
    plt.legend()
    plt.show()

    f, axes = plt.subplots(2, 1)
    plt.sca(axes[0])
    for i in range(kl_divergence_dim_wise.shape[1]):
        plt.plot(moving_average(posterior_mu[:, i], avg_w), label='z{}'.format(i))
    plt.xlabel("Iterations")
    plt.ylabel("Posterior means")
    plt.legend()

    plt.sca(axes[1])
    for i in range(kl_divergence_dim_wise.shape[1]):
        plt.plot(moving_average(posterior_var[:, i], avg_w), label='z{}'.format(i))
    plt.xlabel("Iterations")
    plt.ylabel("Posterior variances")
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
        ax.set_title("Mu", fontsize=18)
        for i in range(mu.shape[1]):
            plt.scatter(np.ones((mu.shape[0])) * (i + 1), mu.view(mu.shape[0], mu.shape[1]).detach().numpy()[:, i])

        ax.tick_params(axis='both', which='major', labelsize=14)

        ax = plt.subplot(2, 1, 2)
        ax.set_title("Std", fontsize=18)
        for i in range(std.shape[1]):
            plt.scatter(np.ones((std.shape[0])) * (i + 1), std.view(std.shape[0], std.shape[1]).detach().numpy()[:, i])

        ax.tick_params(axis='both', which='major', labelsize=14)

        plt.subplots_adjust(hspace=0.45)

        plt.show()

    return z, mu


def show_model_output(model, dataset, indices, num_rows):
    plt.interactive(False)

    num_cols = 3

    plt.rcParams.update({'font.size': 8})

    for i_sample, index in enumerate(indices):
        sample = dataset.__getitem__((index, True))
        x, y, question, _, full_sequence = sample
        if question != -1:
            y_pred, latent_stuff = model(
                torch.unsqueeze(x, 0), torch.unsqueeze(question, 0)
            )
        else:
            y_pred, latent_stuff = model(
                torch.unsqueeze(x, 0), torch.Tensor([-1])
            )

        y_pred = torch.sigmoid(y_pred)
        to_pil = transforms.ToPILImage()

        f, axes = plt.subplots(num_rows, num_cols)
        # f.suptitle("\nSample {}, question: {}".format(i_sample, question), fontsize=16)
        f.suptitle("\nQuestion: {}".format(question), fontsize=16)

        if num_rows > 1:
            for i_frame in range(num_rows):
                # Plot ground truth
                axes[i_frame, 0].imshow(to_pil(y[i_frame]), cmap='gray')

                # Plot prediction
                axes[i_frame, 1].imshow(to_pil(y_pred[0, i_frame]), cmap='gray')

                # Plot Deviation
                diff = abs(y_pred[0, i_frame] - y[i_frame])
                axes[i_frame, 2].imshow(to_pil(diff), cmap='gray')
        else:
            for i_frame in range(num_rows):
                # Plot ground truth
                if question != -1:
                    im = torch.zeros_like(full_sequence[0])

                    if question > 0:
                        for tmp_im in full_sequence[:question.int()]:
                            tmp_im[tmp_im > 0.5] = 1
                            tmp_im[tmp_im < 0.5] = 0
                            im += tmp_im

                        im = im.clamp(0, 0.5)

                    im = (im + full_sequence[question.int()]).clamp(0, 1)
                    axes[0].imshow(to_pil(im), cmap='gray')
                else:
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

        for i_col in range(num_cols):
            if num_rows > 1:
                plt.sca(axes[0, i_col])
                axes[0, i_col].set_title(labels[i_col], rotation=0, size=14)
            else:
                plt.sca(axes[i_col])
                axes[i_col].set_title(labels[i_col], rotation=0, size=14)

        f.tight_layout()

        plt.show(block=True)


def show_correlation(model, dataset, solver, z, gt):
    z = z.view(z.shape[0], -1).numpy()

    gt = np.array(gt)
    if 'use_physics' in solver.train_config.keys() and solver.train_config['use_physics'] and not solver.train_config['use_question']:
        t = solver.train_config['len_inp_sequence'] - 1
        gt = gt[:, t, :]  # for physics without question, we need to evaluate frame t
    else:
        gt = gt[:, 0, :]

    z_mean = z.mean(axis=0)
    gt_mean = gt.mean(axis=0)

    z_std = z.std(axis=0)
    gt_std = gt.std(axis=0)

    n = z.shape[0]

    correlations = np.zeros((z.shape[1], gt.shape[1]))

    # Calculate correlation from every latent variable to every ground truth variable
    for i_z in range(z.shape[1]):
        for i_z2 in range(gt.shape[1]):

            # Calculate correlation
            # From https://www.dummies.com/education/math/statistics/how-to-calculate-a-correlation/
            correlations[i_z, i_z2] = 1/(n-1) * ((z[:, i_z] - z_mean[i_z]) * (gt[:, i_z2] - gt_mean[i_z2])).sum() / \
                                      (z_std[i_z]*gt_std[i_z2])

    correlations = np.abs(correlations)

    plt.imshow(correlations, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
    plt.xlabel('Ground truth variables', fontsize=18)
    plt.ylabel('Latent variables', fontsize=18)
    plt.xticks(np.arange(6), ('px', 'py', 'vx', 'vy', 'ax', 'ay'), fontsize=14)
    plt.yticks(fontsize=14)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=14)
    plt.show()

    correlations = np.zeros((z.shape[1], z.shape[1]))

    # Calculate intercorrelation of latent variables
    for i_z in range(z.shape[1]):
        for i_z2 in range(z.shape[1]):
            # Calculate correlation
            # From https://www.dummies.com/education/math/statistics/how-to-calculate-a-correlation/
            correlations[i_z, i_z2] = 1 / (n - 1) * ((z[:, i_z] - z_mean[i_z]) * (z[:, i_z2] - z_mean[i_z2])).sum() / \
                                      (z_std[i_z] * z_std[i_z2])

    correlations = np.abs(correlations)

    plt.imshow(correlations, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
    plt.xlabel('Latent variables', fontsize=18)
    plt.ylabel('Latent variables', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=14)
    plt.show()

    return correlations


def show_latent_walk_gifs(model, mus, num_images_per_variable=60, question=False, len_out_sequence=1, create_flipbook=False):

    # mus contains that were obtained on a dataset. The random walk is performed between the min and max value of mu for
    # each variable

    min_mu = mus.min(dim=0).values.view(-1)
    max_mu = mus.max(dim=0).values.view(-1)
    mean_mu = mus.mean(dim=0).view(-1)

    to_pil = transforms.ToPILImage()

    f, axes = plt.subplots(len_out_sequence, len(mean_mu))

    if len_out_sequence > 1:
        images = []
        pil_images = []
    else:
        images = [[] for i in range(num_images_per_variable)]
        pil_images = [[] for i in range(num_images_per_variable)]

    for i_var in range(len(mean_mu)):
        # Do a walk over latent variable i
        z_encoder = mean_mu

        values = np.linspace(min_mu[i_var], max_mu[i_var], num_images_per_variable, dtype=float).tolist()

        for i_frame, value in enumerate(values):
            z_encoder[i_var] = value

            images_this_frame = []

            if question:
                z_decoder = model.bottleneck(z_encoder.view(1, -1), torch.tensor(10.).view(-1))
            else:
                z_decoder = model.bottleneck(z_encoder, torch.tensor(-1.))

            # output = model.decode(z_encoder.view(1, -1, 1, 1))
            output = model.decode(z_decoder)
            output = torch.sigmoid(output)

            if len_out_sequence > 1:
                for i_row in range(len_out_sequence):
                    images_this_frame.append(axes[i_row, i_var].imshow(to_pil(output[0, i_row]), cmap='gray'))

                images.append(images_this_frame)
            else:
                im = axes[i_var].imshow(to_pil(output[0]), cmap='gray')
                images[i_frame].append(im)
                pil_images[i_frame].append(to_pil(output[0]))

    # Remove axis ticks
    for i_ax, ax in enumerate(axes.reshape(-1)):
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_tick_params(which='both', length=0, labelleft=False)
        ax.set_title('Variable {}'.format(i_ax))

    f.tight_layout()

    ani = animation.ArtistAnimation(f, images, blit=True, repeat=True, interval=1000/60)
    # ani.save('gifs/question_AE.gif', writer='imagemagick', fps=60)
    plt.show()

    if create_flipbook:
        values = np.linspace(min_mu, max_mu, num_images_per_variable)

        for i_frame, pil_img in enumerate(pil_images):
            f, axes = plt.subplots(len(mean_mu), 1, figsize=(5, 10))
            figure_title = "Frame {}\nAnnealed VAE".format(i_frame)
            plt.text(0.5, 1.40, figure_title,
                     horizontalalignment='center',
                     fontsize=12,
                     transform=axes[0].transAxes)

            for i_ax in range(len(pil_img)):
                axes[i_ax].imshow(pil_img[i_ax], cmap='gray')

            # Remove axis ticks
            for i_ax, ax in enumerate(axes.reshape(-1)):
                ax.text(x=70, y=35, s='Value: {:.2f}'.format(values[i_frame, i_ax]), fontdict={'fontsize': 8})
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_tick_params(which='both', length=0, labelleft=False)

            f.tight_layout()

            plt.savefig("Annealed VAE Frame {}.png".format(i_frame))
            plt.show()

    pass


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


def moving_average(a, n=3):
    # From https://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def print_traning_config(solver):
    print('""" Training Config """\n')
    max_len = max([len(key) for key in solver.train_config.keys()])
    for key in solver.train_config.keys():
        print("{key: <{fill}}: {val}".format(key=key, val=solver.train_config[key], fill=max_len))


def show_correlation_after_physics(model, dataset):
    physics_layer = model.physics_layer
    z = 0
    gt = 0

    len_dataset = len(dataset)
    log_interval = len_dataset // 20

    for i_sample, sample in enumerate(dataset):
        if (i_sample + 1) % log_interval == 0:
            print("\rGetting latent variables for the dataset: {}/{}".format(i_sample+1, len_dataset), end='')

        x, _, question, gt_tmp = sample
        z_tmp, _, _ = model.encode(x[None, :, :, :])
        z_tmp = physics_layer(z_tmp, question).view(1, -1)
        gt_tmp = torch.tensor(gt_tmp[int(question), :2]).view(1, -1)

        if not torch.is_tensor(z):
            z = z_tmp.clone().detach()
            gt = gt_tmp
        else:
            z = torch.cat((z, z_tmp.detach()))
            gt = torch.cat((gt, gt_tmp))

    print('\n', end='')

    gt = gt.numpy()
    z = z.numpy()
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
            correlations[i_z, i_gt] = 1 / (n - 1) * ((z[:, i_z] - z_mean[i_z]) * (gt[:, i_gt] - gt_mean[i_gt])).sum() / \
                                      (z_std[i_z] * gt_std[i_gt])

    correlations = np.abs(correlations)

    plt.imshow(correlations, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
    plt.xlabel('Ground truth variables')
    plt.ylabel('Latent variables')
    plt.colorbar()
    plt.gca().xaxis.tick_top()
    plt.gca().xaxis.set_label_position('top')
    plt.show()


def generate_img_figure_for_tensorboardx(target, prediction, question):
    to_pil = transforms.ToPILImage()
    y = target[0].cpu().detach()
    y_pred = prediction[0].cpu().detach()
    q = question[0].cpu().detach()

    f, axes = plt.subplots(1, 3)
    f.suptitle("Question: {}".format(q), fontsize=16)

    # Plot ground truth
    axes[0].imshow(to_pil(y), cmap='gray')

    # Plot prediction
    axes[1].imshow(to_pil(y_pred), cmap='gray')

    # Plot Deviation
    diff = abs(y_pred - y)
    axes[2].imshow(to_pil(diff), cmap='gray')

    # Remove axis ticks
    for ax in axes.reshape(-1):
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_tick_params(which='both', length=0, labelleft=False)

    # Label rows
    labels = {0: 'Ground truth',
              1: 'Prediction',
              2: 'Deviation'}

    for i in range(3):
        plt.sca(axes[i])
        axes[i].set_title(labels[i], rotation=0, size=14)

    f.tight_layout()

    return f


def walk_over_question(model, dataset):
    to_pil = transforms.ToPILImage()
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    x, _, ques, _, full_sequence = dataset.__getitem__((5, 1))
    questions = torch.arange(full_sequence.shape[0])

    sum_pred = torch.zeros_like(torch.unsqueeze(x[0], 0))
    sum_gt = torch.zeros_like(torch.unsqueeze(x[0], 0))

    images = []
    f, axes = plt.subplots(1, 2)
    for q in questions:
        pred, _ = model(torch.unsqueeze(x, 0), torch.tensor([q], dtype=torch.float32))
        pred = torch.sigmoid(pred)
        # pred[pred > 0.5] = 1
        # pred[pred < 0.5] = 0

        gt_im = torch.unsqueeze(full_sequence[q], 0)
        gt_im[gt_im > 0.5] = 1
        gt_im[gt_im < 0.5] = 0

        im = axes[0].imshow(to_pil((sum_pred + pred[0]).clamp(0, 1)), cmap='gray')
        gt = axes[1].imshow(to_pil((sum_gt + gt_im).clamp(0, 1)), cmap='gray')
        # plt.show()

        sum_gt += torch.unsqueeze(full_sequence[q], 0)
        gif_gt = sum_gt.clamp(0, 1)
        sum_gt = sum_gt.clamp(0, 0.5)
        sum_pred += pred[0]
        gif_pred = sum_pred.clamp(0, 1)
        sum_pred = sum_pred.clamp(0, 0.5)

        images.append([im, gt])

    # Remove axis ticks
    for ax in axes.reshape(-1):
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_tick_params(which='both', length=0, labelleft=False)

    f.tight_layout()

    axes[0].imshow(to_pil(gif_pred), cmap='gray')
    axes[0].set_title('Prediction', fontsize=18)
    axes[1].imshow(to_pil(gif_gt), cmap='gray')
    axes[1].set_title('Ground truth', fontsize=18)
    # plt.savefig('../trajectories.pdf', format='pdf', dpi=1000)

    ani = animation.ArtistAnimation(f, images, blit=True, repeat=True, interval=500)

    plt.show()


def eval_disentanglement(model, eval_datasets, device, num_epochs=50):
    # initialize z_diffs and targets
    z_diffs = []
    targets = []

    # iterate over every eval subset
    for i_dataset, eval_dataset in enumerate(eval_datasets):
        current_latent = eval_dataset.path.split('/')[-1]
        print("Loading eval subset for latent {}".format(current_latent))

        c = eval_dataset.config

        # load all samples in the subset
        eval_data_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=2 * c.batch_size)

        for i_batch, sample in enumerate(eval_data_loader):
            x, _, _, _ = sample
            x.to(device)

            z1, _, _ = model.encode(x[:c.batch_size])
            z2, _, _ = model.encode(x[c.batch_size:])

            z_diff = torch.abs(z1 - z2)
            z_diff = torch.mean(z_diff, dim=0)
            z_diffs.append(z_diff.detach().numpy())

            target = np.array([i_dataset], dtype=np.long)
            targets.append(target)

    # Shuffle training data
    c = list(zip(z_diffs, targets))
    random.shuffle(c)
    z_diffs, targets = zip(*c)

    split_idx = int(0.8 * len(z_diffs))

    z_diffs_train = np.array(z_diffs)[:split_idx]
    targets_train = np.array(targets)[:split_idx, 0]
    z_diffs_test = np.array(z_diffs)[split_idx:]
    targets_test = np.array(targets)[split_idx:, 0]

    # Linear classifier
    model = linear_model.LogisticRegression()
    model.fit(z_diffs_train, targets_train)

    train_accuracy = np.mean(model.predict(z_diffs_train) == targets_train)
    test_accuracy = np.mean(model.predict(z_diffs_test) == targets_test)

    print("Train accuracy (metric for disentanglement): {:.4f}".format(train_accuracy))
    print("Test accuracy (metric for disentanglement): {:.4f}".format(test_accuracy))


def MIG(model, dataset, num_samples, discrete=True, bins=10):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=num_samples, shuffle=True)
    x, _, _, ground_truth = next(iter(data_loader))
    z, _, _ = model.encode(x)
    z = z.detach().numpy()
    z_true = ground_truth[:, 0].numpy()

    if discrete:
        z_min = z.min(axis=0)
        z_max = z.max(axis=0)
        z_true_min = z_true.min(axis=0)
        z_true_max = z_true.max(axis=0)
        bins_arr = np.linspace(z_min, z_max, bins)
        true_bins_arr = np.linspace(z_true_min, z_true_max, bins)

    num_codes = z.shape[1]
    num_factors = np.count_nonzero(z_true[0])  # Only use ground truth which are nonzero

    # Compute mutual info
    # https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/evaluation/metrics/utils.py
    m = np.zeros([num_codes, num_factors])
    for i in range(num_codes):
        for j in range(num_factors):
            if discrete:
                z_discrete = np.digitize(z[:, i], bins_arr[:, i])
                z_true_discrete = np.digitize(z_true[:, j], true_bins_arr[:, j])
                m[i, j] = mutual_info_score(z_true_discrete, z_discrete)
            else:
                m[i, j] = mutual_information((z_true[:, j].reshape(-1, 1), z[:, i].reshape(-1, 1)), k=2)

    # Compute entropy
    # https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/evaluation/metrics/utils.py
    h = np.zeros(num_factors)
    for j in range(num_factors):
        if discrete:
            z_true_discrete = np.digitize(z_true[:, j], true_bins_arr[:, j])
            h[j] = mutual_info_score(z_true_discrete, z_true_discrete)
        else:
            h[j] = entropy((z_true[:, j].reshape(-1, 1)), k=2)

    sorted_m = np.sort(m, axis=0)[::-1]

    mig = np.mean(np.divide(sorted_m[0, :] - sorted_m[1, :], h[:]))

    print("MIG score: {} (ranges from 0 to 1 with 1=completely disentangled)".format(mig))
