import datetime
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler
from torchvision import transforms

from dl4cv.utils import time_left
from dl4cv.models.models import View
from dl4cv.dataset_stuff.dataset_utils import CustomDataset


config = {
    'use_cuda': True,
    'load_to_ram': False,
    'do_overfitting': False,
    'num_workers': 4,
    'save_path': '../saves',

    # Data configs
    'data_path': '../../datasets/ball',
    'num_train': 4096,
    'num_val': 512,
    'len_inp_sequence': 1,
    'len_out_sequence': 1,
    'question': False,

    # Hyperparameter
    'lr': 1e-3,
    'num_epochs': 1000,
    'batch_size': 64,

    # Model configs
    'z_dim_decoder': 2,

    # Logging
    'save_after_epochs': 50,
    'tensorboard_log_dir': '../tensorboard_log/',
}

file_dir = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(file_dir, config['data_path'])
save_path = os.path.join(file_dir, config['save_path'])
tensorboard_path = os.path.join(file_dir, config['tensorboard_log_dir'])


""" Configure cuda """

if config['use_cuda'] and torch.cuda.is_available():
    device = torch.device("cuda")
    kwargs = {'pin_memory': True}
    print("GPU available. Training on {}.".format(device))
else:
    device = torch.device("cpu")
    torch.set_default_tensor_type('torch.FloatTensor')
    kwargs = {}
    print("No GPU. Training on {}.".format(device))


""" Load data """

dataset = CustomDataset(
    data_path,
    transform=transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ]),
    len_inp_sequence=config['len_inp_sequence'],
    len_out_sequence=config['len_out_sequence'],
    load_to_ram=config['load_to_ram'],
    question=config['question'],
    load_ground_truth=True,
    load_config=True
)


if config['do_overfitting']:
    train_data_sampler = SequentialSampler(range(1))
    val_data_sampler = SequentialSampler(range(1))
else:
    train_data_sampler = SubsetRandomSampler(range(config['num_train']))
    val_data_sampler = SubsetRandomSampler(range(
        config['num_train'], config['num_train']+config['num_val']
    ))


train_data_loader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=1 if config['do_overfitting'] else config['batch_size'],
    num_workers=config['num_workers'],
    sampler=train_data_sampler,
    drop_last=True,
    **kwargs
)
val_data_loader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=1 if config['do_overfitting'] else config['batch_size'],
    num_workers=config['num_workers'],
    sampler=val_data_sampler,
    drop_last=True,
    **kwargs
)


""" Build model """

decoder = nn.Sequential(
    nn.Linear(config['z_dim_decoder'], 256),    # B, 256
    nn.ReLU(True),
    nn.Linear(256, 32 * 4 * 4),                 # B, 512
    nn.ReLU(True),
    View((-1, 32, 4, 4)),                       # B,  32,  4,  4
    nn.Upsample(scale_factor=2),
    nn.ConvTranspose2d(32, 32, 3, 1, 1),        # B,  32,  8,  8
    nn.ReLU(True),
    nn.Upsample(scale_factor=2),
    nn.ConvTranspose2d(32, 32, 3, 1, 1),        # B,  32, 16, 16
    nn.ReLU(True),
    nn.Upsample(scale_factor=2),
    nn.ConvTranspose2d(32, config['len_out_sequence'], 3, 1, 1),
)

optimizer = torch.optim.Adam(lr=config['lr'], params=decoder.parameters())


""" Setup tensorboard """

tensorboard_writer = SummaryWriter(os.path.join(tensorboard_path, 'train' + datetime.datetime.now().strftime("%Y%m%d%H%M%S")),
                                           flush_secs=30)


""" Start training """

num_epochs = config['num_epochs']
save_after_epochs = config['save_after_epochs']
z_dim_decoder = config['z_dim_decoder']
z_dim = z_dim_decoder - 1 if config['question'] else z_dim_decoder
latent_names = dataset.config.latent_names

save_path = os.path.join(save_path, 'train' + datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
iter_per_epoch = len(train_data_loader)
n_iters = num_epochs*iter_per_epoch
i_iter = 0
i_epoch = 0

print("Iterations per epoch: {}".format(iter_per_epoch))

t_start_training = time.time()

for i_epoch in range(num_epochs):
    print("Starting epoch {} / {}".format(i_epoch + 1, num_epochs))
    t_start_epoch = time.time()
    i_epoch += 1

    avg_train_loss = 0

    # Do training loop
    decoder.train()
    for i_batch, batch in enumerate(train_data_loader):
        i_iter += 1

        _, y, q, z = batch

        # z is the first frame of ground truth
        z = z[:, 0].float()
        latent_dim = z.shape[1]

        # Fill zeros in z with random normals
        z[:, len(latent_names):] = torch.randn((z.shape[0], z.shape[1] - len(latent_names)))

        if z_dim > latent_dim:
            # Concatenate random normals to z
            diff = z_dim - latent_dim
            z = torch.cat((z, torch.randn((z.shape[0], diff))), dim=1)
        elif z_dim < latent_dim:
            # Use only first few z as latent input
            z = z[:, :z_dim]

        z = z.to(device)

        y = y.to(device)
        if q[0] > 0:
            q = q.to(device)
            z = torch.cat((z, q.view(-1, 1)), dim=1)

        # Forward pass
        y_pred = decoder(z)

        # Compute loss
        loss = F.binary_cross_entropy_with_logits(y_pred, y, reduction='sum').div(y.shape[0])

        # Backpropagate and update weights
        decoder.zero_grad()
        loss.backward()
        optimizer.step()

        avg_train_loss += loss.item()

        tensorboard_writer.add_scalar('train_loss', loss.item(), i_iter)

    # Do eval loop
    print("\nValidate model after epoch {} / {}".format(i_epoch + 1, num_epochs))

    # Add average train loss to loss history
    avg_train_loss /= iter_per_epoch

    # Set model to evaluation mode
    decoder.eval()

    num_val_batches = 0
    val_loss = 0

    for i_batch, batch in enumerate(val_data_loader):
        num_val_batches += 1

        _, y, q, z = batch

        # z is the first frame of ground truth
        z = z[:, 0].float()
        latent_dim = z.shape[1]

        # Fill zeros in z with random normals
        z[:, len(latent_names):] = torch.randn((z.shape[0], z.shape[1] - len(latent_names)))

        if z_dim > latent_dim:
            # Concatenate random normals to z
            diff = z_dim - latent_dim
            z = torch.cat((z, torch.randn((z.shape[0], diff))), dim=1)
        elif z_dim < latent_dim:
            # Use only first few z as latent input
            z = z[:, :z_dim]

        z = z.to(device)

        y = y.to(device)
        if q[0] > 0:
            q = q.to(device)
            z = torch.cat((z, q.view(-1, 1)), dim=1)

        # Forward pass
        y_pred = decoder(z)

        # Compute loss
        loss = F.binary_cross_entropy_with_logits(y_pred, y, reduction='sum').div(y.shape[0])

        val_loss += loss.item()

    val_loss /= num_val_batches

    tensorboard_writer.add_scalar('Avg train loss', avg_train_loss, i_iter)
    tensorboard_writer.add_scalar('val loss', val_loss, i_iter)

    print('Avg Train Loss: ' + "{0:.6f}".format(avg_train_loss) +
          '   Val loss: ' + "{0:.6f}".format(val_loss) +
          "   - " + str(int((time.time() - t_start_epoch) * 1000)) + "ms" +
          "   time left: {}\n".format(time_left(t_start_training, n_iters, i_iter)))

    # Save model and solver
    if save_after_epochs is not None and (i_epoch % save_after_epochs == 0):
        os.makedirs(save_path, exist_ok=True)
        decoder.save(save_path + '/model' + str(i_epoch))
        decoder.to(device)

# Save model and solver after training
os.makedirs(save_path, exist_ok=True)
decoder.save(save_path + '/model' + str(i_epoch))

print('FINISH.')
