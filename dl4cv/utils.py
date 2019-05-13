import torchvision.transforms as transforms

from PIL import Image


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
