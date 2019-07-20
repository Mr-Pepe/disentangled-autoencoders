import torch
import numpy as np

from PIL import Image, ImageDraw
from torchvision import transforms

from dl4cv.dataset_stuff.generateData import get_trajectories, draw_ball


class DataGenerator:
    def __init__(
            self,
            x_std,
            y_std,
            vx_std,
            vy_std,
            ax_std,
            ay_std,
            sequence_length,
            x_mean=None,
            y_mean=None,
            vx_mean=0,
            vy_mean=0,
            ax_mean=0,
            ay_mean=0,
            window_size_x=32,
            window_size_y=32,
            t_frame=1/30,
            ball_radius=2
    ):
        self.x_mean = x_mean if x_mean is not None else window_size_x // 2
        self.y_mean = y_mean if y_mean is not None else window_size_y // 2
        self.vx_mean = vx_mean
        self.vy_mean = vy_mean
        self.ax_mean = ax_mean
        self.ay_mean = ay_mean
        self.x_std = x_std
        self.y_std = y_std
        self.vx_std = vx_std
        self.vy_std = vy_std
        self.ax_std = ax_std
        self.ay_std = ay_std
        self.window_size_x = window_size_x
        self.window_size_y = window_size_y
        self.sequence_length = sequence_length
        self.t_frame = t_frame
        self.ball_radius = ball_radius

    def sample_factors(self, batch_size):
        """
        Returns a list of initial values of a sequence
        """
        x_start = torch.normal(self.x_mean, torch.ones(batch_size) * self.x_std)
        y_start = torch.normal(self.y_mean, torch.ones(batch_size) * self.y_std)

        vx_start = torch.normal(self.vx_mean, torch.ones(batch_size) * self.vx_std)
        vy_start = torch.normal(self.vy_mean, torch.ones(batch_size) * self.vy_std)

        ax = torch.normal(self.ax_mean, torch.ones(batch_size) * self.ax_std)
        ay = torch.normal(self.ay_mean, torch.ones(batch_size) * self.ay_std)

        return [x_start, y_start, vx_start, vy_start, ax, ay]

    def sample_observations_from_factors(self, factors):
        """
        Generates a batch of sequences from the given initial values (factors)
        """
        sequences = []
        ground_truths = []
        batch_size = factors[0].shape[0]

        to_tensor = transforms.ToTensor()

        # create screen
        img = Image.new(mode="L", size=(self.window_size_x, self.window_size_y))
        screen = ImageDraw.Draw(img)

        # Create images from factors
        x, y, vx, vy = get_trajectories(*factors, self.t_frame, self.sequence_length)
        ax = factors[-2]
        ay = factors[-1]

        for i_sequence in range(batch_size):
            ground_truth = np.zeros((self.sequence_length, 6))
            frames = []
            for i_frame in range(self.sequence_length):
                draw_ball(screen, self.window_size_x, self.window_size_y, self.ball_radius, x[i_sequence, i_frame],
                          y[i_sequence, i_frame])
                frame = to_tensor(img)
                frames.append(frame)

                ground_truth[i_frame] = np.array([x[i_sequence, i_frame], y[i_sequence, i_frame],
                                                  vx[i_sequence, i_frame], vy[i_sequence, i_frame],
                                                  ax[i_sequence], ay[i_sequence]])

            frames = torch.tensor(np.stack(frames), dtype=torch.float32).transpose(0, 1)
            sequences.append(frames)
            ground_truths.append(ground_truth)

        ground_truths = np.stack(ground_truths)

        return torch.cat(sequences), ground_truths

    def sample(self, batch_size):
        """
        Directly samples sequences and ground truth
        """
        factors = self.sample_factors(batch_size)
        sequences, ground_truths = self.sample_observations_from_factors(factors)
        return sequences, ground_truths


cls = DataGenerator(
    x_std=1,
    y_std=1,
    vx_std=15,
    vy_std=15,
    ax_std=0,
    ay_std=0,
    sequence_length=10,
)

sequences, ground_truths = cls.sample(batch_size=1000)

pass
