import os
import pickle
import shutil
import argparse

import numpy as np
import torch

from PIL import Image, ImageDraw
from dl4cv.eval.eval_functions import analyze_dataset
from dl4cv.utils import str2bool


def generate_data(c):

    x_min = c.ball_radius
    x_max = c.window_size_x - c.ball_radius
    y_min = c.ball_radius
    y_max = c.window_size_y - c.ball_radius

    c.latent_names = ['px', 'py']

    # create image to draw on
    img = Image.new(mode="L", size=(c.window_size_x, c.window_size_y))
    # create screen
    screen = ImageDraw.Draw(img)

    # Initialize x,y,vx,vy,ax,ay
    x_start, y_start, vx_start, vy_start, x_end, y_end, ax, ay = [torch.zeros((c.num_sequences,)) for i in range(8)]

    collisions = np.ones((c.num_sequences))

    i_run = 0

    # Generate Trajectories for all the sequences
    while collisions.any():
        i_run += 1

        idx = collisions.nonzero()[0]

        x_start[idx] = torch.rand_like(torch.Tensor(idx)) * (c.x_max_sampling - c.x_min_sampling) + c.x_min_sampling
        y_start[idx] = torch.rand_like(torch.Tensor(idx)) * (c.y_max_sampling - c.y_min_sampling) + c.y_min_sampling

        if c.vx_limit != 0:
            vx_start[idx] = sample_near_limit(idx, c.vx_limit, c.fraction, i_run*c.seed)
            c.latent_names.append('vx')

        if c.vy_limit != 0:
            vy_start[idx] = sample_near_limit(idx, c.vy_limit, c.fraction, (i_run + 1)*c.seed)
            c.latent_names.append('vy')

        if c.ax_limit != 0:
            ax[idx] = sample_near_limit(idx, c.ax_limit, c.fraction, (i_run + 2)*c.seed)
            c.latent_names.append('ax')

        if c.ay_limit != 0:
            ay[idx] = sample_near_limit(idx, c.ay_limit, c.fraction, (i_run+3)*c.seed)
            c.latent_names.append('ay')

        x, y, vx, vy = get_trajectories(x_start, y_start, vx_start, vy_start, ax, ay, c.t_frame,
                                        c.len_sequence)

        collisions = get_collisions(x, y, x_min, x_max, y_min, y_max)

        if c.avoid_collisions:
            print("{} collisions of {} sequences".format(collisions.sum(), c.num_sequences))
        else:
            print("No collision avoidance... but we had {} in {} sequences".format(
                collisions.sum(), c.num_sequences)
            )
            collisions = np.zeros((c.num_sequences))

    if c.eval_before_saving:
        trajectories = np.zeros((c.num_sequences, c.len_sequence, 6))

        trajectories[:, :, 0] = x
        trajectories[:, :, 1] = y
        trajectories[:, :, 2] = vx
        trajectories[:, :, 3] = vy
        trajectories[:, :, 4] = np.repeat(ax.reshape(-1, 1), c.len_sequence, 1)
        trajectories[:, :, 5] = np.repeat(ay.reshape(-1, 1), c.len_sequence, 1)

        analyze_dataset(
            trajectories,
            window_size_x=c.window_size_x,
            window_size_y=c.window_size_y,
            mode=c.mode
        )

    if c.save:
        # delete old dataset
        if os.path.exists(c.save_dir_path):
            shutil.rmtree(c.save_dir_path)

        # Create folder
        os.makedirs(c.save_dir_path, exist_ok=True)

        # Save configuration
        with open(os.path.join(c.save_dir_path, 'config.p'), 'wb') as f:
            pickle.dump(c, f)

        # Generate frames for the sequences
        for i_sequence in range(c.num_sequences):

            save_path_sequence = os.path.join(
                c.save_dir_path,
                'seq' + str(i_sequence)
            )

            os.makedirs(save_path_sequence, exist_ok=True)

            ground_truth = np.zeros((c.len_sequence, 6))

            for i_frame in range(c.len_sequence):

                save_path_frame = os.path.join(
                    save_path_sequence,
                    'frame' + str(i_frame) + '.jpeg'
                )

                ground_truth[i_frame] = np.array([x[i_sequence, i_frame], y[i_sequence, i_frame],
                                                  vx[i_sequence, i_frame], vy[i_sequence, i_frame],
                                                  ax[i_sequence], ay[i_sequence]])

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
                    i_sequence+1, c.num_sequences, c.len_sequence))


def sample_near_limit(idx, limit, fraction, seed):
    # Use this function to only sample from the lowest highest fraction of the sample range

    values = torch.rand_like(torch.Tensor(idx)) * limit * fraction - limit
    torch.manual_seed(seed)
    indices = torch.rand_like(values) >= 0.5
    values[indices] = torch.rand_like(values[indices]) * limit * fraction + limit - limit * fraction

    return values

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


def get_trajectories(x_start, y_start, vx_start, vy_start, ax, ay, t_frame, len_sequence):
    t = torch.arange(len_sequence).float() * t_frame
    t = torch.ones(x_start.shape[0], 1) * t.view(1, -1)

    return equation_of_motion(x_start, y_start, vx_start, vy_start, ax, ay, t)


def equation_of_motion(x_0, y_0, vx_0, vy_0, ax, ay, t):
    x_t = x_0.view(-1, 1) + vx_0.view(-1, 1) * t + 0.5 * ax.view(-1, 1) * t * t
    y_t = y_0.view(-1, 1) + vy_0.view(-1, 1) * t + 0.5 * ay.view(-1, 1) * t * t

    vx_t = vx_0.view(-1, 1) + ax.view(-1, 1) * t
    vy_t = vy_0.view(-1, 1) + ay.view(-1, 1) * t

    return x_t, y_t, vx_t, vy_t


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # General settings
    parser.add_argument('--save_dir_path', default='../../../datasets/ball', type=str, help='Save path for the dataset')
    parser.add_argument('--seed', default=1, type=int, help='Random seed')
    parser.add_argument('--num_sequences', default=8192+512, type=int, help='Number of sequences to be generated')
    parser.add_argument('--len_sequence', default=15, type=int, help='Length of sequences to be generated')
    parser.add_argument('--window_size_x', default=64, type=int, help='Window size in x direction in pixels')
    parser.add_argument('--window_size_y', default=64, type=int, help='Window size in y direction in pixels')
    parser.add_argument('--ball_radius', default=2, type=int, help='Ball radius in pixels')
    parser.add_argument('--t_frame', default=1/30, type=float, help='Frame time')
    parser.add_argument('--eval_before_saving', default=True, type=str2bool, help='Evaluate dataset before saving it')
    parser.add_argument('--mode', default='points', type=str, help='Define mode for plotting the dataset during evaluation')
    parser.add_argument('--save', default=True, type=str2bool, help='Generate images for the dataset and save them')

    # Trajectory parameters
    parser.add_argument('--avoid_collisions', default=True, type=str2bool, help='Resample trajectories with collisions')
    parser.add_argument('--x_min_sampling', default=64/5, type=float, help='Minimum x value to sample')
    parser.add_argument('--x_max_sampling', default=64 - 64/5, type=float, help='Maximum x value to sample')
    parser.add_argument('--y_min_sampling', default=64/4, type=float, help='Minimum y value to sample')
    parser.add_argument('--y_max_sampling', default=64 - 64/4, type=float, help='Maximum y value to sample')
    parser.add_argument('--vx_limit', default=25, type=int, help='Min/Max velocity in x direction to sample')
    parser.add_argument('--vy_limit', default=35, type=int, help='Min/Max velocity in y direction to sample')
    parser.add_argument('--ax_limit', default=50, type=int, help='Min/Max acceleration in x direction to sample')
    parser.add_argument('--ay_limit', default=30, type=int, help='Min/Max acceleration in y direction to sample')
    parser.add_argument('--fraction', default=0.75, type=float, help='Fraction of the limit from which to sample')

    config = parser.parse_args()

    generate_data(config)
