import torch
import numpy as np


class SimpleDataSet_Lite:

    def __getitem__(self, index):
        item = self.data[index:index + self.pooling_factor]
        return item

    def __len__(self) -> int:
        return self.data.size // self.pooling_factor

    def __init__(self, data_range=100, data_set_size=int(1e6), distribution="default", pooling_factor=1) -> None:
        self.pooling_factor = pooling_factor

        if distribution == "default":
            mean = data_range / 2
            std = 1
            data = np.random.normal(mean, scale=std, size=data_set_size)
            self.data = data.astype(np.int64)
