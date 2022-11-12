import torch
from torch.utils.data.dataset import T_co, Dataset
import numpy as np


class SimpleDataLoader(Dataset):

    def __getitem__(self, index) -> T_co:
        return torch.tensor(self.data[index])

    def __len__(self) -> int:
        return self.data.size

    def __init__(self, data_range=100, distribution="default") -> None:
        if distribution == "default":
            mean = data_range / 2
            std = 1
            data = np.random.normal(mean, scale=std, size=data_range)
            self.data = data
