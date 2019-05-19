import pygame
import numpy as np
import time
import os
import math

from dl4cv.utils import save_csv

pygame.init()

# Dummy video driver to handle machines without video device (e.g. server)
# os.environ["SDL_VIDEODRIVER"] = "dummy"
"""
When using this script from a server, make sure to add the -X flag to your
ssh command. This will open all windows from the remote system on your
local machine using XQuartz.
Otherwise the script will create only black images
"""

USE_NUM_IMAGES = True
NUM_IMAGES = 1000

T_FRAME = 1/30
WINDOW_SIZE_X = 32
WINDOW_SIZE_Y = 32
BALL_RADIUS = 5
V_MAX = 300     # Limit speed to pixels per second (separate for x and y)

if USE_NUM_IMAGES:
    T_MAX = NUM_IMAGES * T_FRAME
else:
    T_MAX = 1000

save_dir_path = "../datasets/ball"

os.makedirs(save_dir_path, exist_ok=True)
os.makedirs(os.path.join(save_dir_path, 'images'), exist_ok=True)
os.makedirs(os.path.join(save_dir_path, 'meta'), exist_ok=True)


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

x = WINDOW_SIZE_X / 2
y = WINDOW_SIZE_Y / 2
vx = 15
vy = 0
# ax = np.random.uniform(-1, 1, (int(T_MAX/T_FRAME), ))*500
ax = np.zeros((int(T_MAX/T_FRAME), ))
ay = np.ones((int(T_MAX/T_FRAME), )) * 9.81

n_frames = 0
t = 0
while (T_MAX - t) > 1e-5:  # This is basically (t < T_MAX) but accounting for floats

    screen.fill((0, 0, 0))
    screen.blit(ball.surf, (x - BALL_RADIUS, y - BALL_RADIUS))
    pygame.display.flip()

    save_path_frames = os.path.join(
        save_dir_path,
        'images',
        'frame' + str(n_frames) + '.jpeg'
    )
    pygame.image.save(screen, save_path_frames)
    save_path_meta = os.path.join(
        save_dir_path,
        'meta',
        'frame' + str(n_frames) + '.csv'
    )
    save_csv([x, y, vx, vy, ax[n_frames], ay[n_frames]], save_path_meta)

    # Limit velocities to V_MAX
    vx = math.copysign(V_MAX, vx) if abs(vx) > V_MAX else vx
    vy = math.copysign(V_MAX, vy) if abs(vy) > V_MAX else vy

    x, y, vx, vy = get_new_state(x, y, vx, vy, ax[n_frames], ay[n_frames], x_min, x_max, y_min, y_max, T_FRAME)

    # time.sleep(T_FRAME)
    t += T_FRAME
    n_frames += 1
