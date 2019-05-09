import pygame
import numpy as np
import time
import os
import math

pygame.init()


T_FRAME = 1/30
WINDOW_SIZE_X = 100
WINDOW_SIZE_Y = 100
V_MAX = 300     # Limit speed to pixels per second (separate for x and y)

T_MAX = 1000

save_dir_path = "./datasets/ball"

if not os.path.isdir(save_dir_path):
    os.makedirs(save_dir_path)


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

radius = 10
ball = Ball(radius)
x_max = WINDOW_SIZE_X - radius
x_min = radius
y_max = WINDOW_SIZE_Y - radius
y_min = radius

x = WINDOW_SIZE_X / 2
y = WINDOW_SIZE_Y / 2
vx = 100
vy = 200
# ax = np.random.uniform(-1, 1, (int(T_MAX/T_FRAME), ))*500
ax = np.zeros((int(T_MAX/T_FRAME), ))
ay = np.zeros((int(T_MAX/T_FRAME), ))

n_frames = 0
t = 0
while (T_MAX - t) > 1e-5: # This is basically (t < T_MAX) but accounting for floats

    screen.fill((0, 0, 0))
    screen.blit(ball.surf, (x - radius, y - radius))
    pygame.display.flip()

    save_path = os.path.join(save_dir_path, 'frame' + str(n_frames) + '.jpeg')
    pygame.image.save(screen, save_path)

    # Limit velocities to V_MAX
    vx = math.copysign(V_MAX, vx) if abs(vx) > V_MAX else vx
    vy = math.copysign(V_MAX, vy) if abs(vy) > V_MAX else vy

    x, y, vx, vy = get_new_state(x, y, vx, vy, ax[n_frames], ay[n_frames], x_min, x_max, y_min, y_max, T_FRAME)

    # time.sleep(T_FRAME)
    t += T_FRAME
    n_frames += 1
