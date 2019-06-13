from dl4cv.dataset_stuff.dataset_utils import CustomDataset
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data.sampler import SequentialSampler
from dl4cv.eval.analyzeDataset import analyze_dataset


config = {
    'analyze_dataset': True,

    'data_path': '../../datasets/ball',  # Path to directory of the image folder
    'len_inp_sequence': 25,
    'len_out_sequence': 1,

    'save': '../../saves/train20190613143218',
    'epoch': 0,                 # use last model and solver if epoch is zero

    'batch_size': 1000,
    'num_show_images': 5,              # Number of images to show
}

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

pass
