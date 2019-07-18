import os
import pickle

import numpy as np
import torch
from PIL import Image, ImageDraw

from dl4cv.eval.eval_functions import analyze_dataset


eval_config = {
    'dataset_config_path': '../../../datasets/ball/config.p',  # Path to train dataset config
    'save_dir_path': '../../../datasets/evalDataset',
    'num_sequences': 1000,

    'eval_before_generating': True,  # Evaluate the dataset before generating it
    'generate': True,               # Generate the dataset
}

assert eval_config['num_sequences'] % 2 == 0  # We need to build pairs, select an even number of sequences

# Reuse old config
config = pickle.load(open(eval_config['dataset_config_path'], 'rb'))
config.save_dir_path = eval_config['save_dir_path']
config.num_sequences = eval_config['num_sequences']
config.eval_before_generating = eval_config['eval_before_generating']
config.generate = eval_config['generate']


# make save_dir_path absolute
file_dir = os.path.dirname(os.path.realpath(__file__))
config['save_dir_path'] = os.path.join(file_dir, config['save_dir_path'])


def equation_of_motion(x_0, y_0, vx_0, vy_0, ax, ay, t):
    x_t = x_0.view(-1, 1) + vx_0.view(-1, 1) * t + 0.5 * ax.view(-1, 1) * t * t
    y_t = y_0.view(-1, 1) + vy_0.view(-1, 1) * t + 0.5 * ay.view(-1, 1) * t * t

    vx_t = vx_0.view(-1, 1) + ax.view(-1, 1) * t
    vy_t = vy_0.view(-1, 1) + ay.view(-1, 1) * t

    return x_t, y_t, vx_t, vy_t


def get_collisions(x, y, x_min, x_max, y_min, y_max):
    collisions = np.zeros((x.shape[0]))

    eps = 1e-8

    collisions[(x_min - x.min(dim=1).values > eps).nonzero()] = 1
    collisions[(x.max(dim=1).values - x_max > eps).nonzero()] = 1
    collisions[(y_min - y.min(dim=1).values > eps).nonzero()] = 1
    collisions[(y.max(dim=1).values - y_max > eps).nonzero()] = 1

    return collisions


def draw_ball(screen, window_size_x, window_size_y, ball_radius, x, y):
    # Reset to a black image
    screen.rectangle(
        xy=[(0, 0), (window_size_x, window_size_y)],
        fill='black',
        outline=None
    )
    # Draw the ball on the clean image
    screen.ellipse(
        xy=[(x - ball_radius, y - ball_radius), (x + ball_radius, y + ball_radius)],
        width=0,
        fill='white'
    )


def get_initial_velocities(x_start, x_end, y_start, y_end, ax, ay, t_end):
    vx_0 = (x_end - x_start - 0.5 * ax * t_end * t_end) / t_end
    vy_0 = (y_end - y_start - 0.5 * ay * t_end * t_end) / t_end

    return vx_0, vy_0


def get_trajectories(x_start, y_start, vx_start, vy_start, ax, ay, t_frame, sequence_length):
    t = torch.arange(sequence_length).float() * t_frame
    t = torch.ones(x_start.shape[0], 1) * t.view(1, -1)

    return equation_of_motion(x_start, y_start, vx_start, vy_start, ax, ay, t)


def generate_data(c):

    # x_mean = config.update

    if config['sample_mode'] == 'x_start, v_start, a_start':
        latent_names = ['px', 'py', 'vx', 'vy', 'ax', 'ay']
        latent_means = [c.x_mean, c.y_mean, 0, 0, 0, 0]
    elif config['sample_mode'] == 'only_position':
        latent_names = ['px', 'py']
        latent_means = [c.x_mean, c.y_mean]
    else:
        raise NotImplementedError("Invalid sample_mode: {}".format(config['sample_mode']))

    for i_latent in range(len(latent_names)):
        print("Generating subset where {} is constant".format(latent_names[i_latent]))

        latent_path = os.path.join(c.save_dir_path, latent_names[i_latent])

        os.makedirs(latent_path, exist_ok=True)

        t_end = (c.sequence_length - 1) * c.t_frame

        # create image to draw on
        img = Image.new(mode="L", size=(c.window_size_x, c.window_size_y))
        # create screen
        screen = ImageDraw.Draw(img)

        # Initialize x,y,vx,vy,ax,ay
        start_vars = [torch.zeros((c.num_sequences,)) for _ in range(6)]

        collisions = np.ones((c.num_sequences))

        # Generate Trajectories for all the sequences
        while collisions.any():
            idx = collisions.nonzero()[0]

            if config['sample_mode'] == 'x_start, v_start, a_start':

                start_vars[0][idx] = torch.rand_like(torch.Tensor(idx))*(c.x_max - c.x_min) + c.x_min
                start_vars[1][idx] = torch.rand_like(torch.Tensor(idx))*(c.y_max - c.y_min) + c.y_min

                start_vars[2][idx] = torch.normal(0, torch.ones([idx.shape[0]]) * c.vx_std)
                start_vars[3][idx] = torch.normal(0, torch.ones([idx.shape[0]]) * c.vy_std)

                start_vars[4][idx] = torch.normal(0, torch.ones([idx.shape[0]]) * c.ax_std)
                start_vars[5][idx] = torch.normal(0, torch.ones([idx.shape[0]]) * c.ay_std)

            elif config['sample_mode'] == 'only_position':

                start_vars[0][idx] = torch.normal(c.x_mean, torch.ones([idx.shape[0]]) * c.x_std)
                start_vars[1][idx] = torch.normal(c.y_mean, torch.ones([idx.shape[0]]) * c.y_std)

            else:
                raise Exception("Invalid sample_mode: {}".format(config['sample_mode']))

            # Build pairs where one variable is always fixed for two consecutive sequences
            for i_pair in range(0, start_vars[i_latent].shape[0], 2):
                start_vars[i_latent][i_pair + 1] = start_vars[i_latent][i_pair]

            x, y, vx, vy = get_trajectories(*start_vars, c.t_frame, c.sequence_length)

            collisions = get_collisions(x, y, c.x_min, c.x_max, c.y_min, c.y_max)

            if c.avoid_collisions:
                print("{} collisions of {} sequences".format(collisions.sum(), c.num_sequences))
            else:
                print("No collision avoidance... but we had {} in {} sequences".format(
                    collisions.sum(), c.num_sequences)
                )
                collisions = np.zeros((c.num_sequences))

        if config['eval_before_generating']:
            trajectories = np.zeros((c.num_sequences, c.sequence_length, 6))

            trajectories[:, :, 0] = x
            trajectories[:, :, 1] = y
            trajectories[:, :, 2] = vx
            trajectories[:, :, 3] = vy
            trajectories[:, :, 4] = np.repeat(start_vars[4].reshape(-1, 1), c.sequence_length, 1)
            trajectories[:, :, 5] = np.repeat(start_vars[5].reshape(-1, 1), c.sequence_length, 1)

            analyze_dataset(
                trajectories,
                window_size_x=config['window_size_x'],
                window_size_y=config['window_size_y'],
                mode='points'
            )

        if config['generate']:
            # Save configuration
            with open(os.path.join(c.save_dir_path, 'config.p'), 'wb') as f:
                pickle.dump(c, f)

            # Generate frames for the sequences
            for i_sequence in range(c.num_sequences):

                save_path_sequence = os.path.join(
                    latent_path,
                    'seq' + str(i_sequence)
                )

                os.makedirs(save_path_sequence, exist_ok=True)

                ground_truth = np.zeros((c.sequence_length, 6))

                for i_frame in range(c.sequence_length):

                    save_path_frame = os.path.join(
                        save_path_sequence,
                        'frame' + str(i_frame) + '.jpeg'
                    )

                    ground_truth[i_frame] = np.array([x[i_sequence, i_frame], y[i_sequence, i_frame],
                                                      vx[i_sequence, i_frame], vy[i_sequence, i_frame],
                                                      start_vars[4][i_sequence], start_vars[5][i_sequence]])

                    draw_ball(screen, c.window_size_x, c.window_size_y, c.ball_radius, x[i_sequence, i_frame], y[i_sequence, i_frame])
                    img.save(save_path_frame)

                # Save the values at the last
                save_path_ground_truth = os.path.join(
                    save_path_sequence,
                    'ground_truth'
                )
                np.save(save_path_ground_truth, ground_truth)

                if (i_sequence+1) % 100 == 0:
                    print("Generated sequence: %d of %d with length %d ..." % (
                        i_sequence+1, c.num_sequences, c.sequence_length))


if __name__ == '__main__':
    generate_data(config)
