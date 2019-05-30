import math
import numpy as np
import os
import pygame
import time
import torch
from dl4cv.utils import save_csv

pygame.init()


dataset_path = "../../datasets/evalDataset"

SEQUENCE_LENGTH = 3

T_FRAME = 1/30
WINDOW_SIZE_X = 32
WINDOW_SIZE_Y = 32
BALL_RADIUS = 5
V_MAX = 300

"""Generate the variations in every dimension"""

# Variation in pos_x
pos_x_start = BALL_RADIUS
pos_x_end = WINDOW_SIZE_X - BALL_RADIUS
pos_x_num_samples = WINDOW_SIZE_X - 2 * BALL_RADIUS + 1

# Variation in pos_y
pos_y_start = BALL_RADIUS
pos_y_end = WINDOW_SIZE_Y - BALL_RADIUS
pos_y_num_samples = WINDOW_SIZE_Y - 2 * BALL_RADIUS + 1


std_vel = 150

# Variation in vel_x
vel_x_start = - 3 * std_vel
vel_x_end = 3 * std_vel
vel_x_num_samples = (vel_x_end - vel_x_start) // 3 + 1

# Variation in vel_y
vel_y_start = - 3 * std_vel
vel_y_end = 3 * std_vel
vel_y_num_samples = (vel_y_end - vel_y_start) // 3 + 1


std_acc = 100

# Variation in acc_x
acc_x_start = - 3 * std_acc
acc_x_end = 3 * std_acc
acc_x_num_samples = (acc_x_end - acc_x_start) // 2 + 1

# Variation in acc_y
acc_y_start = - 3 * std_acc
acc_y_end = 3 * std_acc
acc_y_num_samples = (acc_y_end - acc_y_start) // 2 + 1


variables = {
    'pos_x': np.linspace(pos_x_start, pos_x_end, pos_x_num_samples),
    'pos_y': np.linspace(pos_y_start, pos_y_end, pos_y_num_samples),
    'vel_x': np.linspace(vel_x_start, vel_x_end, vel_x_num_samples),
    'vel_y': np.linspace(vel_y_start, vel_y_end, vel_y_num_samples),
    'acc_x': np.linspace(acc_x_start, acc_x_end, acc_x_num_samples),
    'acc_y': np.linspace(acc_y_start, acc_y_end, acc_y_num_samples)
}

std = {
    'pos_x': WINDOW_SIZE_X / 4,
    'pos_y': WINDOW_SIZE_Y / 4,
    'vel_x': std_vel,
    'vel_y': std_vel,
    'acc_x': std_acc,
    'acc_y': std_acc
}

class Ball(pygame.sprite.Sprite):
    def __init__(self, radius=50, color=(255,255,255)):
        super(Ball, self).__init__()
        self.surf = pygame.Surface((radius*2,radius*2))
        pygame.draw.circle(self.surf, color, (radius,radius), radius, 0)


def get_new_state(x, y, vx, vy, ax, ay, x_min, x_max, y_min, y_max, t_frame):
    vx_new = vx + ax * t_frame
    vy_new = vy + ay * t_frame

    x_new = x + vx * t_frame + 0.5 * ax * t_frame * t_frame
    y_new = y + vy * t_frame + 0.5 * ay * t_frame * t_frame

    if x_new < x_min:
        x_new = x_min + (x_min - x_new)
        vx_new = -vx_new
    if x_new > x_max:
        x_new = x_max - (x_new - x_max)
        vx_new = -vx_new

    if y_new < y_min:
        y_new = y_min + (y_min - y_new)
        vy_new = -vy_new
    if y_new > y_max:
        y_new = y_max - (y_new - y_max)
        vy_new = -vy_new

    return x_new, y_new, vx_new, vy_new


screen = pygame.display.set_mode((WINDOW_SIZE_X, WINDOW_SIZE_Y))

ball = Ball(BALL_RADIUS)
x_max = WINDOW_SIZE_X - BALL_RADIUS
x_min = BALL_RADIUS
y_max = WINDOW_SIZE_Y - BALL_RADIUS
y_min = BALL_RADIUS


for key in variables:
    path = os.path.join(dataset_path, key)
    num_sequences = len(variables[key])
    print("Generating %d sequences for %s" % (num_sequences, key))

    # Save varying variable
    save_csv(variables[key], os.path.join(path, 'linspace.csv'))

    inner_vars = {}
    for inner_key in variables:
        if key == inner_key:
            inner_vars[inner_key] = variables[key]
        else:
            if inner_key == 'pos_x':
                inner_vars[inner_key] = [torch.normal(
                    mean=WINDOW_SIZE_X/2,
                    std=torch.ones(1) * std[inner_key]
                ).item()] * num_sequences
            elif inner_key == 'pos_y':
                inner_vars[inner_key] = [torch.normal(
                    mean=WINDOW_SIZE_Y / 2,
                    std=torch.ones(1) * std[inner_key]
                ).item()] * num_sequences
            else:
                inner_vars[inner_key] = [torch.normal(
                    mean=0,
                    std=torch.ones(1) * std[inner_key]
                ).item()] * num_sequences

    for i_seq in range(num_sequences):
        sequence_path = os.path.join(path, "seq%d" % i_seq)
        os.makedirs(sequence_path, exist_ok=True)

        x = min(x_max, max(x_min, inner_vars['pos_x'][i_seq]))
        y = min(y_max, max(y_min, inner_vars['pos_y'][i_seq]))
        vx = inner_vars['vel_x'][i_seq]
        vy = inner_vars['vel_y'][i_seq]
        ax = inner_vars['acc_x'][i_seq]
        ay = inner_vars['acc_y'][i_seq]

        for i_frame in range(SEQUENCE_LENGTH):

            screen.fill((0, 0, 0))
            screen.blit(ball.surf, (x - BALL_RADIUS, y - BALL_RADIUS))
            pygame.display.flip()

            frame_path = os.path.join(
                sequence_path, 'frame%d.jpeg' % (SEQUENCE_LENGTH - i_frame - 1)
            )

            pygame.image.save(screen, frame_path)

            # Limit velocities to V_MAX
            vx = math.copysign(V_MAX, vx) if abs(vx) > V_MAX else vx
            vy = math.copysign(V_MAX, vy) if abs(vy) > V_MAX else vy

            x, y, vx, vy = get_new_state(x, y, vx, vy, ax, ay, x_min, x_max, y_min, y_max, T_FRAME)

            # time.sleep(T_FRAME)
