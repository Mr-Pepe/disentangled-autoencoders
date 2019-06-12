import numpy as np
import os
import torch

from torch.utils.data.dataset import Dataset
from dl4cv.utils import read_csv
from torchvision.datasets.folder import \
    pil_loader, has_file_allowed_extension, IMG_EXTENSIONS


class CustomDataset(Dataset):
    def __init__(self, path, transform, len_inp_sequence, len_out_sequence,
                 question=False, load_ground_truth=False, load_to_ram=False, only_input=False):
        self.path = path
        self.transform = transform
        self.sequences = {}
        self.sequence_paths = []
        self.len_inp_sequence = len_inp_sequence
        self.len_out_sequence = len_out_sequence
        self.question = question
        self.load_ground_truth = load_ground_truth
        self.load_to_ram = load_to_ram
        self.only_input = only_input

        num_sequences = 0

        # Find all sequences. Taken form torchvision.dataset.folder.make_dataset()
        for root, dir_names, _ in sorted(os.walk(path)):

            for dir_name in sorted(dir_names, key=lambda s: int(s.split("seq")[1])):

                seq_path = os.path.join(root, dir_name)

                self.sequence_paths.append(seq_path)
                self.sequences[seq_path] = {}

                for _, _, fnames in os.walk(seq_path):

                    self.sequences[seq_path]['ground_truth'] = os.path.join(seq_path, 'ground_truth.npy')
                    self.sequences[seq_path]['images'] = []

                    num_sequences += 1

                    if (num_sequences % 100) == 0:
                        print("\rFound {} sequences.".format(num_sequences), end='')

                    fnames = [fname for fname in fnames if fname != 'ground_truth.npy']

                    for fname in sorted(fnames, key=lambda s: int(s.split("frame")[1].split(".")[0])):

                        if has_file_allowed_extension(fname, IMG_EXTENSIONS):

                            item_path = os.path.join(seq_path, fname)
                            self.sequences[seq_path]['images'].append(item_path)

        print('\n', end='')

        if self.load_to_ram:

            for i, seq_path in enumerate(self.sequence_paths):

                if (i % 100) == 0:
                    print("\rLoading sequences to RAM: {}/{}".format(i, num_sequences), end='')

                self.sequences[seq_path]['images'] = [self.transform(pil_loader(img))
                                                      for img in self.sequences[seq_path]['images']]

            print('\n', end='')

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

        sequence = self.sequences[seq_path]

        if not self.load_to_ram:
            sequence['images'] = [self.transform(pil_loader(img)) for img in self.sequences[seq_path]['images']]

        x = torch.cat(sequence['images'][:self.len_inp_sequence])

        if self.question:
            target_idx = np.random.randint(low=0, high=2 * self.len_inp_sequence - self.len_out_sequence)
            y = torch.cat(sequence['images'][target_idx:target_idx + self.len_out_sequence]) if self.len_out_sequence > 0 else 0
            question = torch.tensor(target_idx, dtype=torch.float32)
        else:
            start_y = self.len_inp_sequence
            end_y = self.len_inp_sequence + self.len_out_sequence
            y = torch.cat(sequence['images'][start_y:end_y]) if self.len_out_sequence > 0 else 0
            question = None

        if self.load_ground_truth:
            ground_truth = np.load(sequence['ground_truth'])
        else:
            ground_truth = 0

        return x, y, question, meta

    def get_ground_truth(self, index):
        return read_csv(self.sequences[self.sequence_paths[index]]['ground_truth'])

    def __len__(self):
        return len(self.sequence_paths)
