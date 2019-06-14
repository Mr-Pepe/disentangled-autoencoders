from dl4cv.dataset_stuff.dataset_utils import CustomDataset
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data.sampler import SequentialSampler
from dl4cv.solver import Solver
from dl4cv.eval.eval_functions import analyze_dataset, show_solver_history, show_latent_variables, show_model_output

import os

config = {
    'analyze_dataset': False,            # Plot positions of the desired datapoints
    'show_solver_history': False,        # Plot losses of the training
    'show_latent_variables': False,      # Show the latent variables for the desired datapoints
    'show_model_output': True,          # Show the model output for the desired datapoints

    'data_path': '../../datasets/ball', # Path to directory of the image folder
    'len_inp_sequence': 25,
    'len_out_sequence': 1,
    'num_samples': None,                # Use the whole dataset if none for latent variables
    'num_show_images': 10,              # Number of outputs to show when show_model_output is True


    'save_path': '../../saves/train20190613143218', # Path to the directory where the model and solver are saved
    'epoch': None,                                  # Use last model and solver if epoch is none

}


def get_model_solver_paths(save_path, epoch):
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

if config['num_samples'] is not None:
    # Sample equidistantly from dataset
    indices = np.linspace(0, len(dataset) - 1, config['num_samples'], dtype=int).tolist()

    dataset = [dataset[i] for i in indices]

if config['analyze_dataset']:
    analyze_dataset(dataset)

if config['show_solver_history'] or \
   config['show_latent_variables'] or \
   config['show_model_output']:

    model_path, solver_path = get_model_solver_paths(config['save_path'], config['epoch'])

    solver = Solver()
    solver.load(solver_path, only_history=True)
    model = torch.load(model_path)


if config['show_solver_history']:
    show_solver_history(solver)

if config['show_latent_variables']:
    show_latent_variables(model, dataset)

if config['show_model_output']:
    # Sample equidistantly from dataset
    if config['num_show_images'] > len(dataset):
        raise Exception('Dataset does not contain {} images to show'.format(config['num_show_images']))

    indices = np.linspace(0, len(dataset) - 1, config['num_show_images'], dtype=int).tolist()

    dataset = [dataset[i] for i in indices]

    show_model_output(model, dataset)



pass
