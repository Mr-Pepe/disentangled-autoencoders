import os
import torch

from torch.utils.data.dataset import Dataset
from torchvision.datasets.folder import \
    pil_loader, has_file_allowed_extension, IMG_EXTENSIONS
from torchvision import transforms

from dl4cv.utils import read_csv


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


class CustomDatasetRAM(Dataset):
    """
    Dataset which behaves exactly like the CustomDataset but loads
    the whole data in memory upon init which gives speedup during training
    """
    def __init__(self, path, transform, sequence_length, load_meta=False):
        self.sequence_length = sequence_length
        self.load_meta = load_meta
        self.sequences = []

        for root, dir_names, _ in sorted(os.walk(path)):
            for dir_name in sorted(dir_names, key=lambda s: int(s.split("seq")[1])):
                sequence = {'images': [], 'meta': []}
                seq_path = os.path.join(root, dir_name)

                if self.load_meta:
                    meta_path = os.path.join(seq_path, 'meta.csv')
                    meta = read_csv(meta_path)
                    sequence['meta'] = meta

                for _, _, fnames in os.walk(seq_path):
                    fnames = [fname for fname in fnames if fname != 'meta.csv']

                    for fname in sorted(fnames, key=lambda s: int(s.split("frame")[1].split(".")[0])):
                        if has_file_allowed_extension(fname, IMG_EXTENSIONS):
                            img_path = os.path.join(seq_path, fname)
                            img = pil_loader(img_path)
                            img = transform(img)
                            sequence['images'].append(img)

                self.sequences.append(sequence)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        seq = self.sequences[index]

        if self.load_meta:
            meta = seq['meta']
        else:
            meta = 0

        x = torch.cat(seq['images'][:-1])
        y = seq['images'][-1]

        return x, y, meta


class ForwardAndMirroredDatasetRAM(Dataset):
    """
    Dataset which outputs a sequence once normal and once horizontally flipped.
    Loads all data to RAM upon init, only loads num_sequences.
    """

    def __init__(self, path, num_sequences, load_meta=False):
        self.transform_normal = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor()
        ])
        self.transform_mirrored = transforms.Compose([
            transforms.Grayscale(),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor()
        ])
        self.load_meta = load_meta
        self.sequences = []
        i_sequences = 0

        for root, dir_names, _ in sorted(os.walk(path)):
            for dir_name in sorted(dir_names, key=lambda s: int(s.split("seq")[1])):
                if i_sequences == num_sequences:
                    break

                sequence = {'images': [], 'meta': []}
                seq_path = os.path.join(root, dir_name)

                if self.load_meta:
                    meta_path = os.path.join(seq_path, 'meta.csv')
                    meta = read_csv(meta_path)
                    sequence['meta'] = meta

                for _, _, fnames in os.walk(seq_path):
                    fnames = [fname for fname in fnames if fname != 'meta.csv']

                    for fname in sorted(fnames, key=lambda s: int(s.split("frame")[1].split(".")[0])):
                        if has_file_allowed_extension(fname, IMG_EXTENSIONS):
                            img_path = os.path.join(seq_path, fname)
                            img = pil_loader(img_path)
                            sequence['images'].append(img)

                self.sequences.append(sequence)
                i_sequences += 1

        assert i_sequences == num_sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        seq = self.sequences[index]

        if self.load_meta:
            meta = seq['meta']
        else:
            meta = 0

        imgs_normal = [self.transform_normal(img) for img in seq['images']]
        imgs_mirrored = [self.transform_mirrored(img) for img in seq['images']]

        imgs_normal = torch.cat(imgs_normal)

        imgs_mirrored = torch.cat(imgs_mirrored)

        return imgs_normal, imgs_mirrored, meta


class ForwardDatasetIdea2RAM(Dataset):
    """
    Dataset which outputs a sequence once normal and once horizontally flipped.
    Loads all data to RAM upon init, only loads num_sequences.
    """

    def __init__(self, path, num_sequences, load_meta=False):
        self.transform_normal = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor()
        ])
        self.transform_horizontal = transforms.Compose([
            transforms.Grayscale(),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor()
        ])
        self.transform_vertical = transforms.Compose([
            transforms.Grayscale(),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor()
        ])
        self.load_meta = load_meta
        self.sequences = []
        i_sequences = 0

        for root, dir_names, _ in sorted(os.walk(path)):
            for dir_name in sorted(dir_names, key=lambda s: int(s.split("seq")[1])):
                if i_sequences == num_sequences:
                    break

                sequence = {'images': [], 'meta': []}
                seq_path = os.path.join(root, dir_name)

                if self.load_meta:
                    meta_path = os.path.join(seq_path, 'meta.csv')
                    meta = read_csv(meta_path)
                    sequence['meta'] = meta

                for _, _, fnames in os.walk(seq_path):
                    fnames = [fname for fname in fnames if fname != 'meta.csv']

                    for fname in sorted(fnames, key=lambda s: int(s.split("frame")[1].split(".")[0])):
                        if has_file_allowed_extension(fname, IMG_EXTENSIONS):
                            img_path = os.path.join(seq_path, fname)
                            img = pil_loader(img_path)
                            sequence['images'].append(img)

                self.sequences.append(sequence)
                i_sequences += 1

        assert i_sequences == num_sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        seq = self.sequences[index]

        if self.load_meta:
            meta = seq['meta']
        else:
            meta = 0

        imgs = [self.transform_normal(img) for img in seq['images']]

        y_normal = self.transform_normal(seq['images'][-1])
        y_horizontal_flip = self.transform_horizontal(seq['images'][-1])
        y_vertical_flip = self.transform_vertical(seq['images'][-1])

        x = torch.cat(imgs[:, :-1])

        return x, y_normal, y_horizontal_flip, y_vertical_flip, meta
