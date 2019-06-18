import os

import numpy as np
import torch
from PIL import Image, ImageDraw

config = {
    'save_dir_path': '../../datasets/ball',
    'num_sequences': 16384+1024,
    'sequence_length': 50,
    'window_size_x': 32,
    'window_size_y': 32,
    'ball_radius': 3,
    't_frame': 1 / 30,
    'avoid_collisions': True
}

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

    # print(str(vx_0))

    return vx_0, vy_0


def get_trajectories(x_start, y_start, vx_start, vy_start, ax, ay, t_frame, sequence_length):
    t = torch.arange(sequence_length).float()*t_frame
    t = torch.ones(x_start.shape[0],1)*t.view(1, -1)

    return equation_of_motion(x_start, y_start, vx_start, vy_start, ax, ay, t)


def clamped_random(num_values, mean, std, min_value, max_value):
    return torch.normal(mean=mean, std=torch.ones([num_values]) * std).int()


def generate_data(**kwargs):

    try:
        save_dir_path = kwargs['save_dir_path']
        num_sequences = kwargs['num_sequences']
        sequence_length = kwargs['sequence_length']
        window_size_x = kwargs['window_size_x']
        window_size_y = kwargs['window_size_y']
        ball_radius = kwargs['ball_radius']
        t_frame = kwargs['t_frame']
        avoid_collisions = kwargs['avoid_collisions']

    except KeyError:
        print('Incomplete configuration for data generation.')

    os.makedirs(save_dir_path, exist_ok=True)

    t_end = (sequence_length - 1) * t_frame

    # create image to draw on
    img = Image.new(mode="L", size=(window_size_x, window_size_y))
    # create screen
    screen = ImageDraw.Draw(img)

    x_max = window_size_x - ball_radius
    x_min = ball_radius
    y_max = window_size_y - ball_radius
    y_min = ball_radius

    x_mean = window_size_x / 2
    x_std = window_size_x / 4
    y_mean = window_size_y / 2
    y_std = window_size_y / 4

    ax_std = 100
    ay_std = 100

    # Initialize x,y,vx,vy,ax,ay
    x_start, y_start, x_end, y_end, ax, ay = [torch.zeros((num_sequences,)) for i in range(6)]

    collisions = np.ones((num_sequences))

    # Generate new accelerations where the sequence has collisions
    while collisions.any():
        idx = collisions.nonzero()[0]

        x_start[idx] = torch.normal(x_mean, torch.ones([idx.shape[0]]) * x_std)
        y_start[idx] = torch.normal(y_mean, torch.ones([idx.shape[0]]) * y_std)

        x_end[idx] = torch.normal(x_mean, torch.ones([idx.shape[0]]) * x_std)
        y_end[idx] = torch.normal(y_mean, torch.ones([idx.shape[0]]) * y_std)

        ax[idx] = torch.normal(0, torch.ones([idx.shape[0]]) * ax_std)
        ay[idx] = torch.normal(0, torch.ones([idx.shape[0]]) * ay_std)

        vx_start, vy_start = get_initial_velocities(x_start, x_end, y_start, y_end, ax, ay, t_end)

        x, y, vx, vy = get_trajectories(x_start, y_start, vx_start, vy_start, ax, ay, t_frame,
                                        sequence_length)

        collisions = get_collisions(x, y, x_min, x_max, y_min, y_max)

    # Generate frames for the sequences
    for i_sequence in range(num_sequences):

        save_path_sequence = os.path.join(
            save_dir_path,
            'seq' + str(i_sequence)
        )

        os.makedirs(save_path_sequence, exist_ok=True)

        ground_truth = np.zeros((sequence_length, 6))

        for i_frame in range(sequence_length):

            save_path_frame = os.path.join(
                save_path_sequence,
                'frame' + str(i_frame) + '.jpeg'
            )

            ground_truth[i_frame] = np.array([x[i_sequence, i_frame], y[i_sequence, i_frame],
                                              vx[i_sequence, i_frame], vy[i_sequence, i_frame],
                                              ax[i_sequence], ay[i_sequence]])

            draw_ball(screen, window_size_x, window_size_y, ball_radius, x[i_sequence, i_frame], y[i_sequence, i_frame])
            img.save(save_path_frame)


        # Save the values at the last
        save_path_ground_truth = os.path.join(
            save_path_sequence,
            'ground_truth'
        )
        np.save(save_path_ground_truth, ground_truth)

        if (i_sequence+1) % 100 == 0:
            print("Generated sequence: %d of %d with length %d ..." % (
                i_sequence+1, num_sequences, sequence_length))


if __name__ == '__main__':
    generate_data(**config)
