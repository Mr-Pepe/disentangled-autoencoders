import csv

import numpy as np
import torchvision.transforms as transforms
import torch
from PIL import Image
import os
from torch.utils.data.dataset import Dataset
from torchvision.datasets.folder import pil_loader, has_file_allowed_extension, IMG_EXTENSIONS
import random


class CustomDataset(Dataset):
    def __init__(self, path, transform, sequence_length, load_meta=False):
        self.path = path
        self.transform = transform
        self.sequences = {}
        self.sequence_paths = []
        self.sequence_length = sequence_length
        self.load_meta = load_meta

        # Find all sequences. Taken form torchvision.dataset.folder.make_dataset()
        for root, dir_names, _ in sorted(os.walk(path)):
            for dir_name in sorted(dir_names, key=lambda s: int(s.split("seq")[1])):
                seq_path = os.path.join(root, dir_name)

                self.sequence_paths.append(seq_path)
                self.sequences[seq_path] = {}

                for _, _, fnames in os.walk(seq_path):

                    self.sequences[seq_path]['meta'] = os.path.join(seq_path, 'meta.csv')
                    self.sequences[seq_path]['images'] = []

                    fnames = [fname for fname in fnames if fname != 'meta.csv']
                    for fname in sorted(fnames, key=lambda s: int(s.split("frame")[1].split(".")[0])):
                        if has_file_allowed_extension(fname, IMG_EXTENSIONS):
                            item_path = os.path.join(seq_path, fname)
                            self.sequences[seq_path]['images'].append(item_path)



    def __getitem__(self, index):
        """
        Gets a sequence of image frames starting from index
        Returns:
            x: torch.tensor, shape [batch, sequence_length*channels, width, height]
            sequence_length - 1 frames
            y: torch.tensor, shape [batch, channels, width, height]
            one image frame, the one to be predicted
        """
        seq_path = self.sequence_paths[index]

        if self.load_meta:
            meta = read_csv(self.sequences[seq_path]['meta'])
        else:
            meta = 0

        images = [pil_loader(image_path) for image_path in self.sequences[seq_path]['images']]

        images = [self.transform(image) for image in images]

        x = torch.cat(images[:-1], 0)
        y = images[-1]

        return x, y, meta

    def __len__(self):
        return len(self.sequence_paths)


class EvalLatentDataset(Dataset):
    def __init__(self, path, transform, sequence_length):
        self.image_path = os.path.join(path, 'images')
        self.meta_path = os.path.join(path, 'meta')
        self.transform = transform
        self.sequence_length = sequence_length
        self.images = []
        self.meta_data = []

        # The SubsetRandomSampler samples random subsets but the subsets themselves are
        # contiguous. Therefore the indices are shuffled to have non-contiguous image series in a minibatch
        self.indices = list(range(self.__len__()))
        random.shuffle(self.indices)

        # Get all image paths
        assert os.path.exists(self.image_path)
        for root, _, fnames in sorted(os.walk(self.image_path)):
            for fname in sorted(fnames, key=lambda s: int(s.split("frame")[1].split(".")[0])):
                if has_file_allowed_extension(fname, IMG_EXTENSIONS):
                    item_path = os.path.join(root, fname)
                    self.images.append(item_path)

        # Get all meta paths
        assert os.path.exists(self.meta_path)
        for root, _, fnames in sorted(os.walk(self.meta_path)):
            for fname in sorted(fnames, key=lambda s: int(s.split("frame")[1].split(".")[0])):
                if fname[-4:] == '.csv':
                    item_path = os.path.join(root, fname)
                    self.meta_data.append(item_path)

    def __len__(self):
        return len(self.images)-self.sequence_length

    def __getitem__(self, index):
        """
        Gets a sequence of image frames and the meta data for the last frame
        Returns:
            images: torch.tensor, shape [batch, sequence_length*channels, width, height]
            sequence_length frames
            meta: torch.tensor, [px, py, vx, vy, ax, ay]
            meta data for the last frame in the sequence
        """
        image_paths = self.images[index:index + self.sequence_length]
        meta_path = self.meta_data[index + self.sequence_length]

        images = [pil_loader(image_path) for image_path in image_paths]
        images = [self.transform(image) for image in images]
        images = torch.cat(images, dim=0)
        meta = read_csv(meta_path)

        return images, meta


def get_normalization_one_frame(filename: str, format: str):
    """
    Gets the mean and standard deviation based on a single frame in the dataset
    This is sufficient for the pong dataset since all frames only contain one
    object and the mean and std are translational and rotational invariant.
    Args:
        filename:
            path to one image in the dataset -> str
        format:
            format of the dataset. Supports 'L' for Grayscale and
            'RGB' for RGB images -> str
    """
    frame = Image.open(filename).convert(format)
    loader = transforms.ToTensor()
    frame = loader(frame)
    mean = frame.mean(dim=(1, 2))
    std = frame.std(dim=(1, 2))
    return mean, std


def tensor_denormalizer(mean, std):
    """
    Denormalizes image to save or display it
    """
    return transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.], std=1/std),
        transforms.Normalize(mean=-mean, std=[1., 1., 1.])])


def save_csv(data, path):
    with open(path, 'w') as f:
        writer = csv.writer(f, delimiter='|')
        writer.writerow(data)


def read_csv(path):
    return np.genfromtxt(path, dtype=np.float, delimiter='|', skip_header=0)
