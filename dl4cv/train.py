import torch
import pickle
import torch.nn as nn
from torchvision import transforms

from dl4cv.utils import CustomDataset

from dl4cv.models.models import VanillaVAE, PhysicsVAE
from dl4cv.solver import Solver
from torch.utils.data import DataLoader, SequentialSampler, SubsetRandomSampler

config = {

    'use_cuda': True,

    # Training continuation
    'continue_training':   False,      # Specify whether to continue training with an existing model and solver
    'model_path': '../saves/train20190520142457/model170',
    'solver_path': '../saves/train20190520142457/solver170',

    # Data
    'data_path': '../datasets/ball/images/',   # Path to the parent directory of the image folder
    'dt': 1/30,                         # Frame rate at which the dataset got generated
    'do_overfitting': False,             # Set overfit or regular training
    'num_train_regular':    2048,     # Number of training samples for regular training
    'num_val_regular':      64,       # Number of validation samples for regular training
    'num_train_overfit':    128,        # Number of training samples for overfitting test runs
    'len_inp_sequence': 3,              # Length of training sequence
    'len_out_sequence': 1,              # Number of generated images

    'num_workers': 4,                   # Number of workers for data loading

    ## Hyperparameters ##
    'max_train_time_s': None,
    'num_epochs': 400,                  # Number of epochs to train
    'batch_size': 32,
    'learning_rate': 1e-3,
    'betas': (0.9, 0.999),              # Beta coefficients for ADAM
    'cov_penalty': 1e-3,

    ## Logging ##
    'log_interval': 1,           # Number of mini-batches after which to print training loss
    'save_interval': 10,         # Number of epochs after which to save model and solver
    'save_path': '../saves'
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

sequence_length = config['len_inp_sequence'] + config['len_out_sequence']

print("Loading dataset with sequence length {}...".format(sequence_length))

dataset = CustomDataset(
    config['data_path'],
    transform=transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ]),
    sequence_length=sequence_length
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
        raise Exception('Trying to use more samples for training and validation than are available.')
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
    solver = pickle.load(open(config['solver_path'], 'rb'))
    loss_criterion = None
    optimizer = None

else:
    print("Initializing model...")
    model = PhysicsVAE(
        dt=config['dt'],
        len_in_sequence=config['len_inp_sequence'],
        greyscale=True
    )
    solver = Solver()
    loss_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])


""" Perform training """

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
                 cov_penalty=config['cov_penalty'])
