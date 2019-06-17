from dl4cv.dataset_stuff.dataset_utils import CustomDataset
import numpy as np
import torch
import torchvision.transforms as transforms
from dl4cv.solver import Solver
from dl4cv.eval.eval_functions import \
    analyze_dataset, \
    show_solver_history, \
    show_latent_variables, \
    show_model_output, \
    eval_correlation
from dl4cv.eval.latent_variable_slideshow import latent_variable_slideshow

import os

config = {
    'analyze_dataset': False,            # Plot positions of the desired datapoints
    'show_solver_history': False,        # Plot losses of the training
    'show_latent_variables': False,      # Show the latent variables for the desired datapoints
    'show_model_output': False,          # Show the model output for the desired datapoints
    'eval_correlation': False,           # Plot the correlation between the latent variables and ground truth
    'latent_variable_slideshow': True,   # Create a slideshow varying over all latent variables

    'data_path': '../../../datasets/ball',  # Path to directory of the image folder
    'eval_data_path': '../../../datasets/evalDataset',
    'len_inp_sequence': 25,
    'len_out_sequence': 1,
    'num_samples': 5,                # Use the whole dataset if none for latent variables
    'num_show_images': 10,              # Number of outputs to show when show_model_output is True


    'save_path': '../../saves/train20190615142355',  # Path to the directory where the model and solver are saved
    'epoch': None,                                  # Use last model and solver if epoch is none

}

# make all paths absolute
file_dir = os.path.dirname(os.path.realpath(__file__))

config['data_path'] = os.path.join(file_dir, config['data_path'])
config['eval_data_path'] = os.path.join(file_dir, config['eval_data_path'])
config['save_path'] = os.path.join(file_dir, config['save_path'])


def get_model_solver_paths(save_path, epoch):
    print("Getting model and solver paths")
    model_paths = []
    solver_paths = []

    for _, _, fnames in os.walk(save_path):
        model_paths = [fname for fname in fnames if 'model' in fname]
        solver_paths = [fname for fname in fnames if 'solver' in fname]

    if not model_paths or not solver_paths:
        raise Exception('Model or solver not found.')

    if not epoch:
        model_path = os.path.join(save_path, sorted(model_paths, key=lambda s: int(s.split("model")[1]))[-1])
        solver_path = os.path.join(save_path, sorted(solver_paths, key=lambda s: int(s.split("solver")[1]))[-1])
    else:
        model_path = os.path.join(save_path, 'model' + str(epoch))
        solver_path = os.path.join(save_path, 'solver' + str(epoch))

    return model_path, solver_path


print("Loading dataset")

dataset = CustomDataset(
    config['data_path'],
    transform=transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ]),
    len_inp_sequence=config['len_inp_sequence'],
    len_out_sequence=config['len_out_sequence'],
    load_ground_truth=False,
    question=True,
    load_to_ram=False
)


if config['analyze_dataset']:
    print("Analysing dataset")
    analyze_dataset(dataset)

if config['show_solver_history'] or \
   config['show_latent_variables'] or \
   config['show_model_output'] or \
   config['eval_correlation'] or \
   config['latent_variable_slideshow']:

    model_path, solver_path = get_model_solver_paths(config['save_path'], config['epoch'])

    print("Loading model and solver")
    solver = Solver()
    solver.load(solver_path, only_history=True)
    model = torch.load(model_path)


if config['show_solver_history']:
    print("Showing solver history")
    show_solver_history(solver)

if config['show_latent_variables']:
    print("Showing latent variables")
    show_latent_variables(model, dataset)

if config['show_model_output']:
    print("Showing model output")
    # Sample equidistantly from dataset
    if config['num_show_images'] > len(dataset):
        raise Exception('Dataset does not contain {} images to show'.format(config['num_show_images']))

    indices = np.linspace(0, len(dataset) - 1, config['num_show_images'], dtype=int).tolist()

    dataset_list = [dataset[i] for i in indices]

    show_model_output(model, dataset_list)

if config['eval_correlation']:
    print("Evaluating correlation")
    # variables to be evaluated, corresponds to the names of the eval mini-datasets
    variables = ['pos_x', 'pos_y', 'vel_x', 'vel_y', 'acc_x', 'acc_y']
    eval_correlation(
        model=model,
        variables=variables,
        path=config['eval_data_path'],
        len_inp_sequence=config['len_inp_sequence'],
        len_out_sequence=config['len_out_sequence']
    )

if config['latent_variable_slideshow']:
    print("Creating slideshow")
    latent_variable_slideshow(model, dataset)
