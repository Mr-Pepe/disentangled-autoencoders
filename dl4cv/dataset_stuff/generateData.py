import os
import math
import numpy as np
import torch

from PIL import Image, ImageDraw


USE_NUM_IMAGES = True
NUM_SEQUENCES = 1000  # 4096+256
SEQUENCE_LENGTH = 50  # including input and output

T_FRAME = 1/30
WINDOW_SIZE_X = 32
WINDOW_SIZE_Y = 32
BALL_RADIUS = 3
V_MAX = 300     # Limit speed to pixels per second (separate for x and y)
DT = SEQUENCE_LENGTH * T_FRAME  # TODO: Test this

save_dir_path = "../../datasets/ball"

os.makedirs(save_dir_path, exist_ok=True)


def get_new_state(x, y, vx, vy, ax, ay, x_min, x_max, y_min, y_max, t_frame):
    collision = False
    vx_new = vx + ax * t_frame
    vy_new = vy + ay * t_frame

    x_new = x + vx * t_frame + 0.5 * ax * t_frame * t_frame
    y_new = y + vy * t_frame + 0.5 * ay * t_frame * t_frame

    if x_new < x_min:
        x_new = x_min + (x_min - x_new)
        vx_new = -vx_new
        collision = True
    if x_new > x_max:
        x_new = x_max - (x_new - x_max)
        vx_new = -vx_new
        collision = True

    if y_new < y_min:
        y_new = y_min + (y_min - y_new)
        vy_new = -vy_new
        collision = True
    if y_new > y_max:
        y_new = y_max - (y_new - y_max)
        vy_new = -vy_new
        collision = True

    return x_new, y_new, vx_new, vy_new, collision


def draw_ball(screen, x, y):
    # Reset to a black image
    screen.rectangle(
        xy=[(0, 0), (WINDOW_SIZE_X, WINDOW_SIZE_Y)],
        fill='black',
        outline=None
    )
    # Draw the ball on the clean image
    screen.ellipse(
        xy=[(x - BALL_RADIUS, y - BALL_RADIUS), (x + BALL_RADIUS, y + BALL_RADIUS)],
        width=0,
        fill='white'
    )


# TODO: Test this function
def get_y(x_start, x_end, y_start, y_end, ax, ay):
    vx = ((x_end - x_start) / DT) - (0.5 * ax * DT)
    vy = ((y_end - y_start) / DT) - (0.5 * ay * DT)
    return vx, vy


def main():
    # Count collisions
    num_collisions = 0

    # create image to draw on
    img = Image.new(mode="L", size=(WINDOW_SIZE_X, WINDOW_SIZE_Y))
    # create screen
    screen = ImageDraw.Draw(img)

    x_max = WINDOW_SIZE_X - BALL_RADIUS
    x_min = BALL_RADIUS
    y_max = WINDOW_SIZE_Y - BALL_RADIUS
    y_min = BALL_RADIUS

    dt = SEQUENCE_LENGTH * T_FRAME

    x_start_all = torch.normal(WINDOW_SIZE_X / 2, std=torch.ones([NUM_SEQUENCES]) * WINDOW_SIZE_X / 4).int()
    y_start_all = torch.normal(WINDOW_SIZE_Y / 2, std=torch.ones([NUM_SEQUENCES]) * WINDOW_SIZE_Y / 4).int()

    x_end_all = torch.normal(WINDOW_SIZE_X / 2, std=torch.ones([NUM_SEQUENCES]) * WINDOW_SIZE_X / 4).int()
    y_end_all = torch.normal(WINDOW_SIZE_Y / 2, std=torch.ones([NUM_SEQUENCES]) * WINDOW_SIZE_Y / 4).int()

    ax_all = torch.normal(0, std=torch.ones([NUM_SEQUENCES]) * 10)
    ay_all = torch.normal(0, std=torch.ones([NUM_SEQUENCES]) * 10)

    for i_sequence in range(NUM_SEQUENCES):

        x = min(x_max, max(x_min, x_start_all[i_sequence].item()))
        y = min(y_max, max(y_min, y_start_all[i_sequence].item()))

        x_end = min(x_max, max(x_min, x_end_all[i_sequence].item()))
        y_end = min(y_max, max(y_min, y_end_all[i_sequence].item()))

        ax = ax_all[i_sequence].item()
        ay = ay_all[i_sequence].item()

        vx, vy = get_y(x, x_end, y, y_end, ax, ay)

        save_path_sequence = os.path.join(
                save_dir_path,
                'seq' + str(i_sequence)
        )

        if i_sequence % 100 == 0:
            print("Generating sequence: %d of %d with length %d ..." % (
                i_sequence, NUM_SEQUENCES, SEQUENCE_LENGTH))

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

            x, y, vx, vy, collision = get_new_state(
                x, y, vx, vy, ax, ay, x_min, x_max, y_min, y_max, T_FRAME
            )

            if collision:
                num_collisions += 1

        # Save the values at the last
        save_path_ground_truth = os.path.join(
            save_path_sequence,
            'ground_truth'
        )
        np.save(save_path_ground_truth, ground_truth)

    print("Collisions for %d sequences: %d" % (NUM_SEQUENCES, num_collisions))


if __name__ == '__main__':
    main()
