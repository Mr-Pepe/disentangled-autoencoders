import os
import math
import numpy as np
import torch

from PIL import Image, ImageDraw


USE_NUM_IMAGES = True
NUM_SEQUENCES = 4096+256
SEQUENCE_LENGTH = 50  # including input and output

T_FRAME = 1/30
WINDOW_SIZE_X = 32
WINDOW_SIZE_Y = 32
BALL_RADIUS = 5
V_MAX = 300     # Limit speed to pixels per second (separate for x and y)

save_dir_path = "../../datasets/ball"

os.makedirs(save_dir_path, exist_ok=True)


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


def draw_ball(screen, x, y):
    screen.rectangle(
        xy=[(0, 0), (WINDOW_SIZE_X, WINDOW_SIZE_Y)],
        fill='black',
        outline=None
    )
    screen.ellipse(
        xy=[(x - BALL_RADIUS, y - BALL_RADIUS), (x + BALL_RADIUS, y + BALL_RADIUS)],
        width=0,
        fill='white'
    )


def main():

    # create image to draw on
    img = Image.new(mode="L", size=(WINDOW_SIZE_X, WINDOW_SIZE_Y))
    # create screen
    screen = ImageDraw.Draw(img)

    x_max = WINDOW_SIZE_X - BALL_RADIUS
    x_min = BALL_RADIUS
    y_max = WINDOW_SIZE_Y - BALL_RADIUS
    y_min = BALL_RADIUS

    x_all = torch.normal(WINDOW_SIZE_X / 2, std=torch.ones([NUM_SEQUENCES]) * WINDOW_SIZE_X / 4).int()
    y_all = torch.normal(WINDOW_SIZE_Y / 2, std=torch.ones([NUM_SEQUENCES]) * WINDOW_SIZE_Y / 4).int()

    vx_all = torch.normal(0, std=torch.ones([NUM_SEQUENCES]) * 15)
    vy_all = torch.normal(0, std=torch.ones([NUM_SEQUENCES]) * 15)

    ax_all = torch.normal(0, std=torch.ones([NUM_SEQUENCES]) * 10)
    ay_all = torch.normal(0, std=torch.ones([NUM_SEQUENCES]) * 10)

    for i_sequence in range(NUM_SEQUENCES):

        x = min(x_max, max(x_min, x_all[i_sequence]))
        y = min(y_max, max(y_min, y_all[i_sequence]))

        vx = vx_all[i_sequence]
        vy = vy_all[i_sequence]

        ax = ax_all[i_sequence]
        ay = ay_all[i_sequence]

        save_path_sequence = os.path.join(
                save_dir_path,
                'seq' + str(i_sequence)
        )

        if i_sequence % 100 == 0:
            print("Generating sequence: %d with length %d ..." % (i_sequence, SEQUENCE_LENGTH))

        os.makedirs(save_path_sequence, exist_ok=True)

        ground_truth = np.zeros((SEQUENCE_LENGTH, 6))

        for i_frame in range(SEQUENCE_LENGTH):

            ground_truth[i_frame] = np.array([x, y, vx, vy, ax, ay])

            save_path_frame = os.path.join(
                save_path_sequence,
                'frame' + str(i_frame) + '.jpeg'
            )

            draw_ball(screen, x, y)
            img.save(save_path_frame)

            # Limit velocities to V_MAX
            vx = math.copysign(V_MAX, vx) if abs(vx) > V_MAX else vx
            vy = math.copysign(V_MAX, vy) if abs(vy) > V_MAX else vy

            x, y, vx, vy = get_new_state(x, y, vx, vy, ax, ay, x_min, x_max, y_min, y_max, T_FRAME)

        # Save the values at the last
        save_path_ground_truth = os.path.join(
            save_path_sequence,
            'ground_truth'
        )
        np.save(save_path_ground_truth, ground_truth)


if __name__ == '__main__':
    main()
