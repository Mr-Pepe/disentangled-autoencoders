import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms

from dl4cv.dataset_stuff.dataset_utils import CustomDataset

config = {
        'data_path'              : '../../datasets/ball',
        'model_path'             : '../../saves/train20190603130451/model10',
        'len_inp_sequence'       : 10,
        'len_out_sequence'       : 5,
        'total_number_of_samples': 10000,
        'batch_size'             : 32
    }


def analyze_dataset(dataset):
    meta = np.array([dataset.get_ground_truth(i) for i in range(len(dataset))])

    plt.scatter(meta[:, :, 0].reshape(-1), meta[:, :, 1].reshape(-1), s=0.2)
    plt.title("Position")
    plt.xlabel("Position x")
    plt.ylabel("Position y")
    plt.show()


if __name__ == '__main__':


    dataset = CustomDataset(
        config['data_path'],
        transform=transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor()
        ]),
        len_inp_sequence=config['len_inp_sequence'],
        len_out_sequence=config['len_out_sequence'],
    )

    analyze_dataset(dataset)
