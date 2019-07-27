import os
import random

import numpy as np

import torch
import torchvision.transforms as transforms

from sklearn import linear_model
from torchvision.utils import make_grid

from dl4cv.dataset_stuff.createDataWithoutSaving import DataGenerator
from dl4cv.solver import Solver


config = {
    'save_path': '../saves/pos_vel_after_konny',  # Path to the directory where the model and solver are saved
    'epoch': None,                                  # Use last model and solver if epoch is none

    'use_cuda': False,
}

device = 'cpu'


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


model_path, solver_path = get_model_solver_paths(config['save_path'], config['epoch'])

solver = Solver()
solver.load(solver_path, device=device, only_history=True)
model = torch.load(model_path, map_location=device)


def eval_disentanglement(
        model,
        dataset_config,
        device,
        num_train_batches=1000,
        num_test_batches=500,
        batch_size=16
):
    # Create the data
    data_generator = DataGenerator(
        x_std=dataset_config['x_std'],
        y_std=dataset_config['y_std'],
        vx_std=dataset_config['vx_std'],
        vy_std=dataset_config['vy_std'],
        ax_std=dataset_config['ax_std'],
        ay_std=dataset_config['ay_std'],
        sequence_length=dataset_config['sequence_length'],
        x_mean=dataset_config['x_mean'],
        y_mean=dataset_config['y_mean'],
        vx_mean=0,
        vy_mean=0,
        ax_mean=0,
        ay_mean=0,
        window_size_x=dataset_config['window_size_x'],
        window_size_y=dataset_config['window_size_y'],
        t_frame=dataset_config['t_frame'],
        ball_radius=dataset_config['ball_radius']
    )

    # latent_names = dataset_config['latent_names']
    latent_names = ['px', 'py', 'vx', 'vy', 'ax', 'ay']

    # initialize z_diffs and targets
    z_diffs_train = []
    z_diffs_test = []
    targets_train = []
    targets_test = []

    print("Generating train samples")
    for i_train in range(num_train_batches):
        index = np.random.randint(len(latent_names))

        factors1 = data_generator.sample_factors(batch_size)
        factors2 = data_generator.sample_factors(batch_size)

        factors2[index] = factors1[index]

        x1, gt1 = data_generator.sample_observations_from_factors(factors1)
        x2, gt2 = data_generator.sample_observations_from_factors(factors2)

        x1 = x1.to(device)
        x2 = x2.to(device)

        z1, _, _ = model.encode(x1)
        z2, _, _ = model.encode(x2)

        z_diff = torch.mean(torch.abs(z2 - z1), dim=0)
        z_diffs_train.append(z_diff.detach().numpy())

        target = np.array([index], dtype=np.long)
        targets_train.append(target)

        if (i_train % 100) == 0:
            print("\rGenerated {} train samples.".format(i_train), end='')

    print('\rGenerated {} train_samples in total'.format(num_train_batches))

    print("Generating test samples")
    for i_test in range(num_test_batches):
        index = np.random.randint(len(latent_names))

        factors1 = data_generator.sample_factors(batch_size)
        factors2 = data_generator.sample_factors(batch_size)

        factors2[index] = factors1[index]

        x1, gt1 = data_generator.sample_observations_from_factors(factors1)
        x2, gt2 = data_generator.sample_observations_from_factors(factors2)

        x1 = x1.to(device)
        x2 = x2.to(device)

        z1, _, _ = model.encode(x1)
        z2, _, _ = model.encode(x2)

        z_diff = torch.mean(torch.abs(z2 - z1), dim=0)
        z_diffs_test.append(z_diff.detach().numpy())

        target = np.array([index], dtype=np.long)
        targets_test.append(target)

        if (i_test % 100) == 0:
            print("\rGenerated {} test samples.".format(i_test), end='')

    print('\rGenerated {} test_samples in total'.format(num_test_batches))

    z_diffs_train = np.array(z_diffs_train)
    z_diffs_test = np.array(z_diffs_test)
    targets_train = np.array(targets_train)[:, 0]
    targets_test = np.array(targets_test)[:, 0]

    # Linear classifier as in
    # https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/evaluation/metrics/beta_vae.py
    model = linear_model.LogisticRegression()
    model.fit(z_diffs_train, targets_train)

    train_accuracy = np.mean(model.predict(z_diffs_train) == targets_train)
    test_accuracy = np.mean(model.predict(z_diffs_test) == targets_test)

    print("Train accuracy: {}".format(train_accuracy))
    print("Test accuracy: {}".format(test_accuracy))


eval_disentanglement(model, solver.dataset_config, device)
