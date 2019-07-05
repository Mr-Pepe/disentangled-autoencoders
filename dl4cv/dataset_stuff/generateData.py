import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

config = {
    'save_dir_path': '../../../datasets/correlated',
    'num_sequences': 2000,
    'sequence_length': 30,
    'window_size_x': 32,
    'window_size_y': 32,
    'ball_radius': 2,
    't_frame': 1 / 30,
    'avoid_collisions': True,
    'sample_mode': 'x_start, v_start, a_start',  # modes: 'x_start, v_start, a_start', 'x_start, x_end, a_start', 'only_position'
    'eval_before_generating': True,  # evaluate the dataset before generating it
    'generate': False
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
    t = torch.arange(sequence_length).float() * t_frame
    t = torch.ones(x_start.shape[0], 1) * t.view(1, -1)

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
    x_std = window_size_x / 1
    y_mean = window_size_y / 2
    y_std = window_size_y / 1

    vx_std = 100
    vy_std = 100

    ax_std = 100
    ay_std = 100

    # Initialize x,y,vx,vy,ax,ay
    x_start, y_start, vx_start, vy_start, x_end, y_end, ax, ay = [torch.zeros((num_sequences,)) for i in range(8)]

    collisions = np.ones((num_sequences))

    # Generate new accelerations where the sequence has collisions
    while collisions.any():
        idx = collisions.nonzero()[0]

        if config['sample_mode'] == 'x_start, x_end, a_start':
            x_start[idx] = torch.normal(x_mean, torch.ones([idx.shape[0]]) * x_std)
            y_start[idx] = torch.normal(y_mean, torch.ones([idx.shape[0]]) * y_std)

            x_end[idx] = torch.normal(x_mean, torch.ones([idx.shape[0]]) * x_std)
            y_end[idx] = torch.normal(y_mean, torch.ones([idx.shape[0]]) * y_std)

            ax[idx] = torch.normal(0, torch.ones([idx.shape[0]]) * ax_std)
            ay[idx] = torch.normal(0, torch.ones([idx.shape[0]]) * ay_std)

            vx_start, vy_start = get_initial_velocities(x_start, x_end, y_start, y_end, ax, ay, t_end)

            x, y, vx, vy = get_trajectories(x_start, y_start, vx_start, vy_start, ax, ay, t_frame,
                                            sequence_length)

        elif config['sample_mode'] == 'x_start, v_start, a_start':
            x_start[idx] = torch.normal(x_mean, torch.ones([idx.shape[0]]) * x_std)
            y_start[idx] = torch.normal(y_mean, torch.ones([idx.shape[0]]) * y_std)

            vx_start[idx] = torch.normal(0, torch.ones([idx.shape[0]]) * vx_std)
            vy_start[idx] = torch.normal(0, torch.ones([idx.shape[0]]) * vy_std)

            ax[idx] = torch.normal(0, torch.ones([idx.shape[0]]) * ax_std)
            ay[idx] = torch.normal(0, torch.ones([idx.shape[0]]) * ay_std)

            x, y, vx, vy = get_trajectories(x_start, y_start, vx_start, vy_start, ax, ay, t_frame,
                                            sequence_length)

        elif config['sample_mode'] == 'only_position':
            x_start[idx] = torch.normal(x_mean, torch.ones([idx.shape[0]]) * x_std)
            y_start[idx] = torch.normal(y_mean, torch.ones([idx.shape[0]]) * y_std)

            x, y, vx, vy = get_trajectories(x_start, y_start, vx_start, vy_start, ax, ay, t_frame,
                                            sequence_length)

        else:
            raise Exception("Invalid sample_mode: {}".format(config['sample_mode']))

        collisions = get_collisions(x, y, x_min, x_max, y_min, y_max)

        if avoid_collisions:
            print("{} collisions of {} sequences".format(collisions.sum(), num_sequences))
        else:
            print("No collision avoidance... but we had {} in {} sequences".format(
                collisions.sum(), num_sequences)
            )
            collisions = np.zeros((num_sequences))

    if config['eval_before_generating']:
        plt.figure(figsize=(6, 6))
        plt.scatter(x, y, s=0.2)
        for i in range(x.shape[0]):
            plt.plot(x[i, :].tolist(), y[i, :].tolist(), 'b', linewidth=0.5)
        plt.title("Position")
        plt.xlabel("Position x")
        plt.ylabel("Position y")
        plt.xlim(0, window_size_x)
        plt.ylim(0, window_size_y)
        plt.show()

        meta = torch.stack([x, y, vx, vy, ax.view(-1, 1).repeat(1, vx.shape[1]), ay.view(-1, 1).repeat(1, vx.shape[1])], dim=-1)
        n = meta.shape[0]
        meta = meta[:, 0]
        meta_mean = meta.mean(dim=0)
        meta_std = meta.std(dim=0)

        correlations = np.zeros((meta.shape[1], meta.shape[1]))

        # Calculate correlation from every latent variable to every ground truth variable
        for i_z in range(meta.shape[1]):
            for i_gt in range(meta.shape[1]):
                # Calculate correlation
                # From https://www.dummies.com/education/math/statistics/how-to-calculate-a-correlation/
                correlations[i_z, i_gt] = 1 / (n - 1) * (
                            (meta[:, i_z] - meta_mean[i_z]) * (meta[:, i_gt] - meta_mean[i_gt])).sum() / \
                                          (meta_std[i_z] * meta_std[i_gt])

        correlations = np.abs(correlations)

        plt.imshow(correlations, cmap='hot', interpolation='nearest')
        plt.xlabel('Ground truth variables')
        plt.ylabel('Ground truth variables')
        plt.colorbar()
        plt.gca().xaxis.tick_top()
        plt.gca().xaxis.set_label_position('top')
        plt.show()

    if config['generate']:
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
