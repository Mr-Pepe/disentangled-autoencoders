import torch
import torch.nn as nn

from dl4cv.dataset_utils import ForwardDatasetIdea2RAM

from dl4cv.models.models import AutoEncoderIdea2
from dl4cv.experimental.solver_idea2 import Solver
from torch.utils.data import DataLoader, SequentialSampler

config = {

    'use_cuda': True,

    # Training continuation
    'continue_training':   False,      # Specify whether to continue training with an existing model and solver
    'model_path': '../../saves/train20190523144416/model20',
    'solver_path': '../../saves/train20190523144416/solver20',

    # Data
    'data_path': '../../datasets/noAcceleration/',   # Path to the parent directory of the image folder
    'len_inp_sequence': 2,
    'dt': 1/30,                         # Frame rate at which the dataset got generated
    'do_overfitting': True,             # Set overfit or regular training
    'num_train_regular':    4096,       # Number of training samples for regular training
    'num_val_regular':      256,        # Number of validation samples for regular training
    'num_train_overfit':    256,        # Number of training samples for overfitting test runs

    'num_workers': 4,                   # Number of workers for data loading

    ## Hyperparameters ##
    'max_train_time_s': None,
    'num_epochs': 100,                  # Number of epochs to train
    'batch_size': 16,
    'learning_rate': 1e-3,
    'betas': (0.9, 0.999),              # Beta coefficients for ADAM
    'cov_penalty': 1e-1,
    'beta': 0,                          # beta-coefficient for disentangling

    ## Logging ##
    'log_interval': 8,           # Number of mini-batches after which to print training loss
    'save_interval': 10,         # Number of epochs after which to save model and solver
    'save_path': '../../saves'
}


""" Add a seed to have reproducible results """

seed = 123
torch.manual_seed(seed)


""" Configure training with or without cuda """

if config['use_cuda'] and torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed(seed)
    kwargs = {'pin_memory': True}
    print("GPU available. Training on {}".format(device))
else:
    device = torch.device("cpu")
    torch.set_default_tensor_type('torch.FloatTensor')
    kwargs = {}
    print("No GPU. Training on {}".format(device))


""" Load dataset """

if config['do_overfitting']:
    print("Overfitting on a subset of {} samples".format(config['num_train_overfit']))
    if config['batch_size'] > config['num_train_overfit']:
        raise Exception('Batchsize for overfitting bigger than the number of samples for overfitting.')

    num_train = config['num_train_overfit']
    num_val = 0

    train_data_range = range(num_train)
    val_data_range = range(num_train)
else:
    print("Training on {} samples".format(config['num_train_regular']))

    num_train = config['num_train_regular']
    num_val = config['num_val_regular']

    train_data_range = range(num_train)
    val_data_range = range(num_train, num_train + num_val)


num_sequences = num_train + num_val

train_data_sampler = SequentialSampler(train_data_range)
val_data_sampler = SequentialSampler(val_data_range)

print("Loading dataset with {} sequences...".format(num_sequences))

dataset = ForwardDatasetIdea2RAM(
    path=config['data_path'],
    num_sequences=num_sequences,
    load_meta=False
)


train_data_loader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=config['batch_size'],
    num_workers=config['num_workers'],
    sampler=train_data_sampler,
    **kwargs
)
val_data_loader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=config['batch_size'],
    num_workers=config['num_workers'],
    sampler=val_data_sampler,
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
    model = AutoEncoderIdea2(
        dt=1/30,
        len_in_sequence=config['len_inp_sequence'],
        greyscale=True
    )
    solver = Solver()
    loss_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])


""" Perform training """

print("Calling Solver, train for {} epochs...".format(config['num_epochs']))

if __name__ == "__main__":
    solver.train(model=model,
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
                 beta=config['beta'])
