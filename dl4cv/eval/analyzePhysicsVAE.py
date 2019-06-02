import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import numpy as np

from dl4cv.dataset_stuff.dataset_utils import CustomDataset

to_pil = transforms.ToPILImage()

config = {
    'data_path': '../../datasets/ball',
    'model_path': '../../saves/train20190531073151/model100',
    'len_inp_sequence': 3,
    'len_out_sequence': 3,
    'batch_size': 256
}

dataset = CustomDataset(
    config['data_path'],
    transform=transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ]),
    len_inp_sequence=config['len_inp_sequence'],
    len_out_sequence=config['len_out_sequence']
)

data_loader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=config['batch_size']
)

model = torch.load(config['model_path'])
model.eval()

x, _, _ = next(iter(data_loader))

z_t, mu, logvar = model.encode(x)

z_t_plus_1 = model.physics(z_t)

y = model.decode(z_t_plus_1)


# Get the mean mu for every latent variable
mu_mean = mu.mean(dim=0)
mu_mean = mu_mean.view(mu_mean.numel())

# Get the mean standard deviation for all
std = logvar.div(2).exp()
std_mean = logvar.div(2).exp().mean(dim=0)
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

        plt.imshow(to_pil(model.decode(model.physics(z.view(1, z_t_dim, 1, 1))).detach()[0]), cmap='gray')
        plt.title("Variable " + str(i_variable+1) + " = " + str(value_variable.item()))
        plt.show()


z_t_plus_1_mean = z_t_plus_1.mean(dim=0)
