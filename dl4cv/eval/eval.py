import glob
import os
import pickle

import numpy as np

import torch
import torchvision.transforms as transforms

from dl4cv.dataset_stuff.dataset_utils import CustomDataset
from dl4cv.solver import Solver
from dl4cv.eval.eval_functions import \
    analyze_dataset, \
    show_solver_history, \
    show_latent_variables, \
    show_model_output, \
    show_correlation, \
    latent_variable_slideshow, \
    print_traning_config, \
    show_correlation_after_physics, \
    show_latent_walk_gifs, \
    walk_over_question, \
    eval_disentanglement, \
    MIG


config = {
    'analyze_dataset': False,            # Plot positions of the desired datapoints
    'show_solver_history': False,        # Plot losses of the training
    'show_latent_variables': False,      # Show the latent variables for the desired datapoints
    'show_model_output': False,          # Show the model output for the desired datapoints
    'eval_correlation': False,           # Plot the correlation between the latent variables and ground truth
    'latent_variable_slideshow': False,   # Create a slideshow varying over all latent variables
    'print_training_config': False,       # Print the config that was used for training the model
    'latent_walk_gifs': False,
    'walk_over_question': False,
    'eval_disentanglement': False,       # Evaluate disentanglement according to the metric from the BetaVAE paper.
    'mutual_information_gap': True,     # Evaluate disentanglement according to the MIG score

    'data_path': '../../../datasets/ball',  # Path to directory of the image folder
    'eval_data_path': '../../../datasets/evalDataset',
    'len_inp_sequence': 15,
    'len_out_sequence': 1,
    'num_samples': 500,                # Use the whole dataset if none for latent variables
    'num_show_images': 10,              # Number of outputs to show when show_model_output is True

    'question': True,

    'save_path': '../../saves/von_felipe',  # Path to the directory where the model and solver are saved
    'epoch': None,                                  # Use last model and solver if epoch is none

    'use_cuda': False,
}


""" Configure evaluation with or without cuda """

if config['use_cuda'] and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    torch.set_default_tensor_type('torch.FloatTensor')


# make all paths absolute
file_dir = os.path.dirname(os.path.realpath(__file__))

config['data_path'] = os.path.join(file_dir, config['data_path'])
config['eval_data_path'] = os.path.join(file_dir, config['eval_data_path'])
config['save_path'] = os.path.join(file_dir, config['save_path'])

z = None
mu = None


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
    load_ground_truth=True,
    question=config['question'],
    load_to_ram=False,
    load_config=True
)
dataset_config = pickle.load(open(os.path.join(config['data_path'], 'config.p'), 'rb'))

if config['num_samples'] is not None:
    if config['num_show_images'] > len(dataset):
        raise Exception('Dataset does not contain {} images to show'.format(config['num_show_images']))

    # Sample equidistantly from dataset
    indices = np.linspace(0, len(dataset) - 1, config['num_samples'], dtype=int).tolist()

    dataset_list = [dataset[i] for i in indices]
    ground_truth = [dataset.get_ground_truth(i) for i in indices]
else:
    dataset_list = dataset
    ground_truth = [dataset.get_ground_truth(i) for i in range(len(dataset))]

model_path, solver_path = get_model_solver_paths(config['save_path'], config['epoch'])

print("Loading model and solver")
solver = Solver()
solver.load(solver_path, device=device, only_history=True)
model = torch.load(model_path, map_location=device)


if config['analyze_dataset']:
    print("Analysing dataset")
    if config['num_samples'] is not None:
        indices = np.linspace(0, len(dataset) - 1, config['num_samples'], dtype=int).tolist()
    else:
        indices = range(len(dataset))

    trajectories = np.array([dataset.get_ground_truth(i) for i in indices])

    analyze_dataset(
        trajectories,
        window_size_x=dataset_config['window_size_x'],
        window_size_y=dataset_config['window_size_y'],
        mode='points')


if config['show_solver_history']:
    print("Showing solver history")
    show_solver_history(solver)


if config['print_training_config']:
    print_traning_config(solver)


if config['show_latent_variables']:
    print("Using {} samples to show latent variables".format(config['num_samples']))
    z, mu = show_latent_variables(model, dataset_list)


if config['show_model_output']:
    print("Showing model output")
    indices = np.linspace(0, len(dataset) - 1, config['num_show_images'], dtype=int).tolist()
    show_model_output(model, [dataset[i] for i in indices], dataset.len_out_sequence)


if config['eval_correlation']:
    print("Evaluating correlation")

    if z is None:
        z, mu = show_latent_variables(model, dataset_list, show=False)

    show_correlation(model, dataset_list, z, ground_truth)

    if model.use_physics:
        show_correlation_after_physics(model, dataset_list)
    else:
        print("Model without physics layer")


if config['latent_variable_slideshow']:
    print("Creating slideshow")
    latent_variable_slideshow(model, dataset_list)


if config['latent_walk_gifs']:
    print("Creating GIFs for walks over latent variables")
    if config['num_samples'] is not None:
        # Sample equidistantly from dataset
        indices = np.linspace(0, len(dataset) - 1, config['num_samples'], dtype=int).tolist()

        dataset_list = [dataset[i] for i in indices]
        ground_truth = [dataset.get_ground_truth(i) for i in indices]
    else:
        dataset_list = dataset
        ground_truth = [dataset.get_ground_truth(i) for i in range(len(dataset))]

    if mu is None:
        z, mu = show_latent_variables(model, dataset_list, show=False)

    show_latent_walk_gifs(model, mu, question=config['question'], len_out_sequence=dataset.len_out_sequence)

if config['walk_over_question']:
    print("Walk over questions")
    walk_over_question(model, dataset)

if config['eval_disentanglement']:
    print("Evaluating disentanglement")
    # load eval dataset to list
    eval_datasets = [
        CustomDataset(
            path,
            transform=transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor()
            ]),
            len_inp_sequence=config['len_inp_sequence'],
            len_out_sequence=config['len_out_sequence'],
            load_ground_truth=True,
            question=config['question'],
            load_to_ram=False,
            load_config=False,
        )
        for path in glob.glob(config['eval_data_path'] + '/*') if os.path.isdir(path)]

    train_accuracy = eval_disentanglement(model, eval_datasets, device, num_epochs=100)

    print("Training set accuracy (metric for disentanglement): {:.4f}".format(train_accuracy))


if config['mutual_information_gap']:
    print("Computing mutual information gap")
    mig = MIG(model, dataset, config['num_samples'])
    print("MIG score: {} (ranges from 0 to 1 with 1=completely disentangled)".format(mig))
