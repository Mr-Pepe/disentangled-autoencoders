import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import numpy as np

from dataset_stuff.dataset_utils import CustomDataset

to_pil = transforms.ToPILImage()

config = {
    'data_path': '../../datasets/ball',
    'model_path': '../../saves/train20190603130451/model10',
    'len_inp_sequence': 25,
    'len_out_sequence': 1,
    'total_number_of_samples': 10000,
    'batch_size': 32
}

dataset = CustomDataset(
    config['data_path'],
    transform=transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ]),
    len_inp_sequence=config['len_inp_sequence'],
    len_out_sequence=config['len_out_sequence'],
    question=True,
    load_meta=False,
    load_to_ram=False
)

data_loader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=config['batch_size']
)

model = torch.load(config['model_path'])
model.eval()

n_batches_total = int(config['total_number_of_samples'] / config['batch_size'])
n_batches = 0
mu_mean = 0

iter_data_loader = iter(data_loader)
mu = 0
std = 0

while n_batches < n_batches_total:
    n_batches += 1
    print("Batch {} of {}.".format(n_batches, n_batches_total))

    x, _, _, _ = next(iter_data_loader)

    z_t, mu_tmp, logvar = model.encode(x)

    std_tmp = logvar.div(2).exp()

    if not torch.is_tensor(mu):
        mu = mu_tmp.clone().detach()
        std = std_tmp.clone().detach()

    else:
        mu = torch.cat((mu, mu_tmp.detach()))
        std = torch.cat((std, std_tmp.detach()))

# Get the mean mu for every latent variable
mu_mean = mu.mean(dim=0)/n_batches_total
mu_mean = mu_mean.view(mu_mean.numel())

# Get the mean standard deviation for all
std_mean = logvar.div(2).exp().mean(dim=0)/n_batches_total
std_mean = std_mean.view(std_mean.numel())

z_t_dim = 6

# Show distributions of means and standard deviations over the minibatch
ax = plt.subplot(2, 1, 1)
ax.set_title("Mu")
for i in range(mu.shape[1]):
    plt.scatter(np.ones((mu.shape[0]))*(i+1), mu.view(mu.shape[0], mu.shape[1]).detach().numpy()[:, i])

ax = plt.subplot(2, 1, 2)
ax.set_title("Std")
for i in range(std.shape[1]):
    plt.scatter(np.ones((std.shape[0])) * (i + 1), std.view(std.shape[0], std.shape[1]).detach().numpy()[:, i])

plt.show()

# Vary the latent variable to see their effects
for i_variable in range(z_t_dim):

    min_mu = mu[:, i_variable].min().item()
    max_mu = mu[:, i_variable].max().item()
    max_std = std[:, i_variable].max().item()

    for value_variable in torch.linspace(min_mu - max_std, max_mu + max_std, 20):
        z = mu_mean

        z[i_variable] = value_variable

        output = model.decode(model.physics(z.view(1, z_t_dim, 1, 1))).detach()[0]

        for i_img, img in enumerate(output):
            plt.subplot(1, len(output), i_img+1)

            plt.imshow(to_pil(img), cmap='gray')

        # plt.title("Variable " + str(i_variable+1) + " = " + str(value_variable.item()))

        plt.show()
