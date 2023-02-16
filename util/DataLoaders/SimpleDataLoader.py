import numpy
import torch
from torch.utils.data.dataset import T_co, Dataset
import numpy as np


class SimpleDataSet(Dataset):

    def __getitem__(self, index) -> T_co:
        return torch.tensor(self.data[index:index + self.pooling_factor]).float()

    def __len__(self) -> int:
        return self.data.size // self.pooling_factor

    def __init__(self, data_range=100, data_set_size=10000, distribution="default", pooling_factor=1) -> None:
        self.pooling_factor = pooling_factor

        if distribution == "default":
            mean = data_range / 2
            std = 1
            data = np.random.normal(mean, scale=std, size=data_set_size)
            self.data = data.round().astype(np.int64)

    def export_as_csv(self, export_dir):
        file_name = f"{export_dir}/simple_dataset.csv"
        numpy.savetxt(file_name, self.data, delimiter=",")

