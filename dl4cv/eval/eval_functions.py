import copy
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.utils import make_grid

from dl4cv.dataset_stuff.dataset_utils import CustomDataset
from dl4cv.utils import read_csv, reparametrize


def analyze_dataset(trajectories, mode='lines'):

    x_max = trajectories[:, :, 0].max()+2
    x_min = trajectories[:, :, 0].min()-2
    y_max = trajectories[:, :, 1].max()+2
    y_min = trajectories[:, :, 1].min()-2

    plt.figure(figsize=(6, 6))
    if mode == 'lines':
        for i in range(trajectories.shape[0]):
            plt.plot(trajectories[i, :, 0].reshape(-1), trajectories[i, :, 1].reshape(-1), 'b', linewidth=0.5)
    elif mode == 'points':
        plt.scatter(trajectories[:, :, 0].reshape(-1), trajectories[:, :, 1].reshape(-1), s=0.2)

    plt.title("Position")
    plt.xlabel("Position x")
    plt.ylabel("Position y")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
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
    plt.xlabel('Ground truth variables')
    plt.ylabel('Ground truth variables')
    plt.colorbar()
    plt.gca().xaxis.tick_top()
    plt.gca().xaxis.set_label_position('top')
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

    plt.plot(beta)
    plt.xlabel("Iterations")
    plt.ylabel("Beta")
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

    return z, mu


def show_model_output(model, dataset, num_rows):
    plt.interactive(False)

    num_cols = 3

    plt.rcParams.update({'font.size': 8})

    for i_sample, sample in enumerate(dataset):
        x, y, question, meta = sample
        if question != -1:
            y_pred, latent_stuff = model(
                torch.unsqueeze(x, 0), torch.unsqueeze(question, 0)
            )
        else:
            y_pred, latent_stuff = model(
                torch.unsqueeze(x, 0), torch.Tensor([-1])
            )

        to_pil = transforms.ToPILImage()

        f, axes = plt.subplots(num_rows, num_cols)
        f.suptitle("\nSample {}, question: {}".format(i_sample, question), fontsize=16)

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
        for i_z2 in range(gt.shape[1]):

            # Calculate correlation
            # From https://www.dummies.com/education/math/statistics/how-to-calculate-a-correlation/
            correlations[i_z, i_z2] = 1/(n-1) * ((z[:, i_z] - z_mean[i_z]) * (gt[:, i_z2] - gt_mean[i_z2])).sum() / \
                                      (z_std[i_z]*gt_std[i_z2])

    correlations = np.abs(correlations)

    plt.imshow(correlations, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
    plt.xlabel('Ground truth variables')
    plt.ylabel('Latent variables')
    plt.colorbar()
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
    plt.xlabel('Latent variables')
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


def show_latent_walk_gifs(model, mus, num_images_per_variable=20, question=False, len_out_sequence=1):

    # mus contains that were obtained on a dataset. The random walk is performed between the min and max value of mu for
    # each variable

    min_mu = mus.min(dim=0).values.view(-1)
    max_mu = mus.max(dim=0).values.view(-1)
    mean_mu = mus.mean(dim=0).view(-1)

    to_pil = transforms.ToPILImage()

    f, axes = plt.subplots(len_out_sequence, len(mean_mu))

    if len_out_sequence > 1:
        images = []
    else:
        images = [[] for i in range(num_images_per_variable)]

    for i_var in range(len(mean_mu)):
        # Do a walk over latent variable i
        z_encoder = mean_mu

        values = np.linspace(min_mu[i_var], max_mu[i_var], num_images_per_variable, dtype=float).tolist()

        for i_frame, value in enumerate(values):
            z_encoder[i_var] = value

            images_this_frame = []

            if question:
                z_decoder = model.bottleneck(z_encoder.view(1, -1, 1, 1), torch.tensor(0.).view(-1))
            else:
                z_decoder = model.bottleneck(z_encoder, torch.tensor(-1.))

            output = model.decode(z_decoder.view(1, -1, 1, 1))

            if len_out_sequence > 1:
                for i_row in range(len_out_sequence):
                    images_this_frame.append(axes[i_row, i_var].imshow(to_pil(output[0, i_row]), cmap='gray'))

                images.append(images_this_frame)
            else:
                im = axes[i_var].imshow(to_pil(output[0]), cmap='gray')
                images[i_frame].append(im)


    # Remove axis ticks
    for ax in axes.reshape(-1):
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_tick_params(which='both', length=0, labelleft=False)

    f.tight_layout()

    ani = animation.ArtistAnimation(f, images, blit=True, repeat=True, interval=100)

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
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    x, _, ques, _ = next(iter(dataloader))
    questions = torch.arange(x.shape[1])

    images = []
    f, axes = plt.subplots(1, 2)
    for q in questions:
        pred, _ = model(x, torch.tensor([q], dtype=torch.float32))
        im = axes[0].imshow(to_pil(pred[0]), cmap='gray')
        gt = axes[1].imshow(to_pil(x[0, q]), cmap='gray')
        images.append([im, gt])

    # Remove axis ticks
    for ax in axes.reshape(-1):
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_tick_params(which='both', length=0, labelleft=False)

    f.tight_layout()

    ani = animation.ArtistAnimation(f, images, blit=True, repeat=True, interval=500)

    plt.show()

