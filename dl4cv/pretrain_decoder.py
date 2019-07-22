import os

import torch

from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler
from torchvision import transforms

from dl4cv.models.models import OnlyDecoder
from dl4cv.dataset_stuff.dataset_utils import CustomDataset
from dl4cv.solver_decoder import SolverDecoder


config = {
    'use_cuda': True,
    'load_to_ram': False,
    'do_overfitting': True,
    'num_workers': 4,
    'save_path': '../saves',

    # Data configs
    'data_path': '../../datasets/ball',
    'num_train': 4096,
    'num_val': 512,
    'len_inp_sequence': 0,
    'len_out_sequence': 1,
    'question': False,

    # Hyperparameter
    'lr': 5e-4,
    'num_epochs': 1000,
    'batch_size': 64,
    'z_scale_factor': 1/30,
    'loss_weight_ball': 4.,

    # Model configs
    'z_dim_decoder': 2,

    # Logging
    'save_interval': 50,
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

model = OnlyDecoder(config).to(device)

optimizer = torch.optim.Adam(lr=config['lr'], params=model.parameters())
solver = SolverDecoder()

""" Perform training """

if __name__ == "__main__":
    solver.train(model=model,
                 train_config=config,
                 dataset_config=dataset.config,
                 tensorboard_path=config['tensorboard_log_dir'],
                 optim=optimizer,
                 num_epochs=config['num_epochs'],
                 train_loader=train_data_loader,
                 val_loader=val_data_loader,
                 save_after_epochs=config['save_interval'],
                 save_path=config['save_path'],
                 device=device,
                 z_scale_factor=config['z_scale_factor'],
                 overfit=config['do_overfitting'],
                 loss_weight_ball=config['loss_weight_ball'])
