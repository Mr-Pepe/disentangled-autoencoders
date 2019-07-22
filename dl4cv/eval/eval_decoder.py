import os
import pickle

import torch
import torchvision.transforms as transforms

from dl4cv.dataset_stuff.dataset_utils import CustomDataset
from dl4cv.eval.eval import get_model_solver_paths
from dl4cv.eval.eval_functions import eval_decoder
from dl4cv.solver_decoder import SolverDecoder


config = {
    'data_path': '../../datasets/ball',  # Path to directory of the image folder

    'len_inp_sequence': 0,
    'len_out_sequence': 1,
    'num_samples': 500,  # Use the whole dataset if none for latent variables
    'num_show_images': 2,  # Number of outputs to show when show_model_output is True

    'question': False,

    'save_path': '../saves/temp',  # Path to the directory where the model and solver are saved
    'epoch': None,                                  # Use last model and solver if epoch is none
}

device = torch.device("cpu")


model_path, solver_path = get_model_solver_paths(config['save_path'], config['epoch'])
model = torch.load(model_path, map_location=device)
solver = SolverDecoder()
solver.load(solver_path, device=device, only_history=True)

dataset = CustomDataset(
    solver.train_config['data_path'],
    transform=transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ]),
    len_inp_sequence=solver.train_config['len_inp_sequence'],
    len_out_sequence=solver.train_config['len_out_sequence'],
    load_ground_truth=True,
    question=solver.train_config['question'],
    load_to_ram=False,
    load_config=True
)
dataset_config = pickle.load(open(os.path.join(config['data_path'], 'config.p'), 'rb'))


c = solver.train_config

eval_decoder(model, dataset, c, config['num_show_images'])