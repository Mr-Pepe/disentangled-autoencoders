import os

import torch
import torch.nn as nn

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, SubsetRandomSampler
from torchvision import transforms

from dl4cv.dataset_stuff.dataset_utils import CustomDataset
from dl4cv.models.models import VariationalAutoEncoder
from dl4cv.solver import Solver

config = {

    'use_cuda': True,

    # Training continuation
    'continue_training':   False,      # Specify whether to continue training with an existing model and solver
    'model_path': '../saves/train20190617165601/model10',
    'solver_path': '../saves/train20190617165601/solver10',

    # Data
    'data_path': '../../datasets/ball/',   # Path to the parent directory of the image folder
    'load_data_to_ram': False,
    'dt': 1/30,                         # Frame rate at which the dataset got generated
    'do_overfitting': False,             # Set overfit or regular training
    'num_train_regular':    8196,       # Number of training samples for regular training
    'num_val_regular':      1024,        # Number of validation samples for regular training
    'num_train_overfit':    256,        # Number of training samples for overfitting test runs
    'len_inp_sequence': 15,              # Length of training sequence
    'len_out_sequence': 1,              # Number of generated images

    'num_workers': 4,                   # Number of workers for data loading

    # Hyper parameters
    'max_train_time_s': None,
    'num_epochs': 100,                  # Number of epochs to train
    'batch_size': 1024,
    'learning_rate': 1e-3,
    'betas': (0.9, 0.999),              # Beta coefficients for ADAM
    'cov_penalty': 1e-1,
    'beta': 0.001,
    'beta_decay': 1.,
    'use_question': True,
    'patience': 128,
    'loss_weighting': True,
    'loss_weight_ball': 2.,

    # Logging
    'log_interval': 1,           # Number of mini-batches after which to print training loss
    'save_interval': 10,         # Number of epochs after which to save model and solver
    'save_path': '../saves',
    'tensorboard_log_dir': '../tensorboard_log/exp_1'
}


""" Make paths absolute """

file_dir = os.path.dirname(os.path.realpath(__file__))

config['model_path'] = os.path.join(file_dir, config['model_path'])
config['solver_path'] = os.path.join(file_dir, config['solver_path'])
config['data_path'] = os.path.join(file_dir, config['data_path'])
config['save_path'] = os.path.join(file_dir, config['save_path'])
config['tensorboard_log_dir'] = os.path.join(file_dir, config['tensorboard_log_dir'])


""" Add a seed to have reproducible results """

seed = 123
torch.manual_seed(seed)


""" Configure training with or without cuda """

if config['use_cuda'] and torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed(seed)
    kwargs = {'pin_memory': True}
    print("GPU available. Training on {}.".format(device))
else:
    device = torch.device("cpu")
    torch.set_default_tensor_type('torch.FloatTensor')
    kwargs = {}
    print("No GPU. Training on {}.".format(device))


""" Load dataset """


print("Loading dataset with input sequence length {} and output sequence length {}...".format(
        config['len_inp_sequence'], config['len_out_sequence']))

dataset = CustomDataset(
    config['data_path'],
    transform=transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ]),
    len_inp_sequence=config['len_inp_sequence'],
    len_out_sequence=config['len_out_sequence'],
    load_to_ram=config['load_data_to_ram'],
    question=config['use_question'],
    load_ground_truth=False,
)


if config['batch_size'] > len(dataset):
    raise Exception('Batch size bigger than the dataset.')

if config['do_overfitting']:
    print("Overfitting on a subset of {} samples".format(config['num_train_overfit']))
    if config['batch_size'] > config['num_train_overfit']:
        raise Exception('Batchsize for overfitting bigger than the number of samples for overfitting.')
    else:
        train_data_sampler = SequentialSampler(range(config['num_train_overfit']))
        val_data_sampler = SequentialSampler(range(config['num_train_overfit']))

else:
    print("Training on {} samples".format(config['num_train_regular']))
    if config['num_train_regular']+config['num_val_regular'] > len(dataset):
        raise Exception(
            'Trying to use more samples for training and validation than len(dataset), {} > {}.'.format(
                config['num_train_regular'] + config['num_val_regular'], len(dataset)
            ))
    else:
        train_data_sampler = SubsetRandomSampler(range(config['num_train_regular']))
        val_data_sampler = SubsetRandomSampler(range(
            config['num_train_regular'],
            config['num_train_regular']+config['num_val_regular']
        ))


train_data_loader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=config['batch_size'],
    num_workers=config['num_workers'],
    sampler=train_data_sampler,
    drop_last=True,
    **kwargs
)
val_data_loader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=config['batch_size'],
    num_workers=config['num_workers'],
    sampler=val_data_sampler,
    drop_last=True,
    **kwargs
)


""" Initialize model and solver """

if config['continue_training']:
    print("Continuing training with model: {} and solver: {}".format(
        config['model_path'], config['solver_path'])
    )

    model = torch.load(config['model_path'])
    model.to(device)
    solver = Solver()
    solver.optim = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    solver.load(config['solver_path'])
    loss_criterion = None
    optimizer = None

else:
    print("Initializing model...")
    model = VariationalAutoEncoder(
        len_in_sequence=config['len_inp_sequence'],
        len_out_sequence=config['len_out_sequence'],
        z_dim_encoder=4,
        z_dim_decoder=5,
        use_physics=False
    )
    solver = Solver()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    # When using loss weighting, the loss is reduced only after multiplication with the weight mask
    reduction = 'none' if config['loss_weight_ball'] else 'mean'
    loss_criterion = nn.MSELoss(reduction=reduction)


""" Initialize tensorboard summary writer """

tensorboard_writer = SummaryWriter(config['tensorboard_log_dir'])

# Add graph to tensorboard
# example_input, _, _, _ = next(iter(train_data_loader))
# tensorboard_writer.add_graph(
#     model=model,
#     input_to_model=example_input
# )


""" Perform training """

if __name__ == "__main__":
    solver.train(model=model,
                 config=config,
                 tensorboard_writer=tensorboard_writer,
                 optim=optimizer,
                 loss_criterion=loss_criterion,
                 num_epochs=config['num_epochs'],
                 max_train_time_s=config['max_train_time_s'],
                 train_loader=train_data_loader,
                 val_loader=val_data_loader,
                 log_after_iters=config['log_interval'],
                 save_after_epochs=config['save_interval'],
                 save_path=config['save_path'],
                 device=device,
                 cov_penalty=config['cov_penalty'],
                 beta=config['beta'],
                 beta_decay=config['beta_decay'],
                 patience=config['patience'],
                 loss_weighting=['loss_weighting'],
                 loss_weight_ball=config['loss_weight_ball'])
