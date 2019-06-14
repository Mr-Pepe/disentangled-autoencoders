import matplotlib.pyplot as plt
import numpy as np
import torch

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

    print('\n', end='')

    for i_sample, sample in enumerate(dataset):

        if (i_sample + 1) % 100 == 0:
            print("\rGetting latent variables for the dataset: {}/{}".format(i_sample+1, len_dataset), end='')

        x, _, _, _ = sample

        z_t, mu_tmp, logvar = model.encode(torch.unsqueeze(x,0))

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
