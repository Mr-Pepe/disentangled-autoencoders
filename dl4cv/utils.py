import torchvision.transforms as transforms
import torch
from PIL import Image
import os
from torch.utils.data.dataset import Dataset
from torchvision.datasets.folder import pil_loader, has_file_allowed_extension, IMG_EXTENSIONS

class customDataset(Dataset):
    def __init__(self, path, transform, sequence_length):
        self.path = path
        self.transform = transform
        self.images = []
        self.sequence_length = sequence_length

        # Find all images in folder. Taken form torchvision.dataset.folder.make_dataset()
        for root, _, fnames in sorted(os.walk(path)):
            for fname in sorted(fnames, key=lambda s: int(s.split("frame")[1].split(".")[0])):
                if has_file_allowed_extension(fname, IMG_EXTENSIONS):
                    item_path = os.path.join(root, fname)
                    self.images.append(item_path)

        self.indices = range(self.__len__()-self.sequence_length-1)


    def __getitem__(self, index):
        image_paths = self.images[index:index+self.sequence_length]

        images = [pil_loader(image_path) for image_path in image_paths]

        images = [self.transform(image) for image in images]

        x = torch.cat(images[:-1], 0)
        y = images[-1]

        return (x,y)



    def __len__(self):
        return len(self.images)





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
