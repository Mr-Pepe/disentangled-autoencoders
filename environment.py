import pygame
import numpy as np
import time

pygame.init()


T_FRAME = 1/30
WINDOW_SIZE_X = 800
WINDOW_SIZE_Y = 800


screen = pygame.display.set_mode((WINDOW_SIZE_X, WINDOW_SIZE_Y))


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


radius = 50
ball = Ball(radius)
x_max = WINDOW_SIZE_X - radius
x_min = radius
y_max = WINDOW_SIZE_Y - radius
y_min = radius

x = WINDOW_SIZE_X / 2
y = WINDOW_SIZE_Y / 2
vx = 0
vy = 0
ax = 50
ay = 25

t = 0
running = True
while running:

    screen.fill((0, 0, 0))
    screen.blit(ball.surf, (x - radius, y - radius))
    pygame.display.flip()


    x, y, vx, vy = get_new_state(x, y, vx, vy, ax, ay, x_min, x_max, y_min, y_max, T_FRAME)

    time.sleep(T_FRAME)
    t += T_FRAME


