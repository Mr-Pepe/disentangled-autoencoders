from dl4cv.dataset_stuff.dataset_utils import CustomDataset
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data.sampler import SequentialSampler
from dl4cv.eval.analyzeDataset import analyze_dataset

import os

config = {
    'analyze_dataset': False,
    'data_path': '../../datasets/ball',  # Path to directory of the image folder
    'len_inp_sequence': 25,
    'len_out_sequence': 1,

    'show_solver_history': True,
    'save_path': '../../saves/VanillaVAE',
    'epoch': 30,                 # use last model and solver if epoch is zero

    'batch_size': 1000,
    'num_show_images': 5,              # Number of images to show
}


def get_model_solver_paths(save_path, epoch):
    for _, _, fnames in os.walk(save_path):
        model_paths = [fname for fname in fnames if 'model' in fname]
        solver_paths = [fname for fname in fnames if 'solver' in fname]

    if not model_paths or not solver_paths:
        raise Exception('Model or solver not found.')

    if epoch == 0:
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

if config['analyze_dataset']:
    analyze_dataset(dataset)

if config['show_solver_history']:
    model_path, solver_path = get_model_solver_paths(config['save_path'], config['epoch'])



pass
