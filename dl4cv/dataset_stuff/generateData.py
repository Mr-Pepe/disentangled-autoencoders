import os
import pickle
import shutil

from dl4cv.utils import Config
import numpy as np
import torch
from PIL import Image, ImageDraw
from dl4cv.eval.eval_functions import analyze_dataset

config = Config({
    'save_dir_path': '../../../datasets/ball',
    'num_sequences': 4096+512,
    'sequence_length': 5,
    'window_size_x': 32,
    'window_size_y': 32,
    'ball_radius': 2,
    't_frame': 1 / 30,
    'eval_before_generating': True,  # Evaluate the dataset before generating it
    'generate': True                # Generate the dataset
})

# Define trajectory properties. Do this separately as some values need the config to already exist
config.update({
    'avoid_collisions': False,
    'sample_mode': 'x_start, v_start, a_start',  # modes: 'x_start, v_start, a_start',
                                                 #        'x_start, x_end, a_start',
                                                 #        'only_position'

    'x_min': config.ball_radius,
    'x_max': config.window_size_x - config.ball_radius,
    'y_min': config.ball_radius,
    'y_max': config.window_size_y - config.ball_radius,

    'x_mean': config.window_size_x / 2,
    'x_std':  config.window_size_x / 1,

    'y_mean': config.window_size_y / 2,
    'y_std':  config.window_size_y / 1,

    'vx_std': 0,
    'vy_std': 0,

    'ax_std': 0,
    'ay_std': 0
})


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
    t_end = (c.sequence_length - 1) * c.t_frame

    # create image to draw on
    img = Image.new(mode="L", size=(c.window_size_x, c.window_size_y))
    # create screen
    screen = ImageDraw.Draw(img)

    # Initialize x,y,vx,vy,ax,ay
    x_start, y_start, vx_start, vy_start, x_end, y_end, ax, ay = [torch.zeros((c.num_sequences,)) for i in range(8)]

    collisions = np.ones((c.num_sequences))

    # Generate Trajectories for all the sequences
    while collisions.any():
        idx = collisions.nonzero()[0]

        if config['sample_mode'] == 'x_start, x_end, a_start':
            x_start[idx] = torch.normal(c.x_mean, torch.ones([idx.shape[0]]) * c.x_std)
            y_start[idx] = torch.normal(c.y_mean, torch.ones([idx.shape[0]]) * c.y_std)

            x_end[idx] = torch.normal(c.x_mean, torch.ones([idx.shape[0]]) * c.x_std)
            y_end[idx] = torch.normal(c.y_mean, torch.ones([idx.shape[0]]) * c.y_std)

            ax[idx] = torch.normal(0, torch.ones([idx.shape[0]]) * c.ax_std)
            ay[idx] = torch.normal(0, torch.ones([idx.shape[0]]) * c.ay_std)

            vx_start, vy_start = get_initial_velocities(x_start, x_end, y_start, y_end, ax, ay, t_end)

            x, y, vx, vy = get_trajectories(x_start, y_start, vx_start, vy_start, ax, ay, c.t_frame,
                                            c.sequence_length)

        elif config['sample_mode'] == 'x_start, v_start, a_start':

            # x_start[idx] = torch.normal(c.x_mean, torch.ones([idx.shape[0]]) * c.x_std)
            # y_start[idx] = torch.normal(c.y_mean, torch.ones([idx.shape[0]]) * c.y_std)

            x_start[idx] = torch.rand_like(torch.Tensor(idx))*(c.x_max - c.x_min) + c.x_min
            y_start[idx] = torch.rand_like(torch.Tensor(idx))*(c.y_max - c.y_min) + c.y_min

            vx_start[idx] = torch.normal(0, torch.ones([idx.shape[0]]) * c.vx_std)
            vy_start[idx] = torch.normal(0, torch.ones([idx.shape[0]]) * c.vy_std)

            ax[idx] = torch.normal(0, torch.ones([idx.shape[0]]) * c.ax_std)
            ay[idx] = torch.normal(0, torch.ones([idx.shape[0]]) * c.ay_std)

            x, y, vx, vy = get_trajectories(x_start, y_start, vx_start, vy_start, ax, ay, c.t_frame,
                                            c.sequence_length)

        elif config['sample_mode'] == 'only_position':
            x_start[idx] = torch.normal(c.x_mean, torch.ones([idx.shape[0]]) * c.x_std)
            y_start[idx] = torch.normal(c.y_mean, torch.ones([idx.shape[0]]) * c.y_std)

            x, y, vx, vy = get_trajectories(x_start, y_start, vx_start, vy_start, ax, ay, c.t_frame,
                                            c.sequence_length)

        else:
            raise Exception("Invalid sample_mode: {}".format(config['sample_mode']))

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
        trajectories[:, :, 4] = np.repeat(ax.reshape(-1, 1), c.sequence_length, 1)
        trajectories[:, :, 5] = np.repeat(ay.reshape(-1, 1), c.sequence_length, 1)

        analyze_dataset(
            trajectories,
            window_size_x=config['window_size_x'],
            window_size_y=config['window_size_y'],
            mode='points'
        )

    if config['generate']:
        # delete old dataset
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

            ground_truth = np.zeros((c.sequence_length, 6))

            for i_frame in range(c.sequence_length):

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
                    i_sequence+1, c.num_sequences, c.sequence_length))


if __name__ == '__main__':
    generate_data(config)
